#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""

import argparse
import copy
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from PIL import Image
from pixel import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Modality,
    PangoCairoTextRenderer,
    PIXELForSequenceClassification,
    PIXELTrainer,
    PIXELTrainingArguments,
    PoolingMode,
    PyGameTextRenderer,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    log_sequence_classification_predictions,
    resize_model_embeddings,
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: Optional[int] = field(
        default=196,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained processor name or path if not the same as model_name"}
    )
    rendering_backend: Optional[str] = field(
        default="pangocairo", metadata={
            "help": "Rendering backend to use. Options are 'pygame' or 'pangocairo'. For most applications it is "
                    "recommended to use the default 'pangocairo'."}
    )
    fallback_fonts_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing fallback font files used by the text renderer for PIXEL. "
                          "PyGame does not support fallback fonts so this argument is ignored when using the "
                          "PyGame backend."},
    )
    render_rgb: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to render images in RGB. RGB rendering can be useful when working with emoji "
            "but it makes rendering a bit slower, so it is recommended to turn on RGB rendering only "
            "when there is need for it. PyGame does not support fallback fonts so this argument is ignored "
            "when using the PyGame backend."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: str = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    pooling_mode: str = field(
        default="mean",
        metadata={
            "help": f"Pooling mode to use in classification head (options are {[e.value for e in PoolingMode]}."
        },
    )
    pooler_add_layer_norm: bool = field(
        default=True,
        metadata={
            "help": "Whether to add layer normalization to the classification head pooler. Note that this flag is"
            "ignored and no layer norm is added when using CLS pooling mode."
        },
    )
    dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout probability for attention blocks and classification head"}
    )

    def __post_init__(self):
        self.pooling_mode = PoolingMode.from_string(self.pooling_mode)

        if not self.rendering_backend.lower() in ["pygame", "pangocairo"]:
            raise ValueError("Invalid rendering backend. Supported backends are 'pygame' and 'pangocairo'.")
        else:
            self.rendering_backend = self.rendering_backend.lower()


def get_processor(model_args: argparse.Namespace, modality: Modality):
    if modality == Modality.TEXT:
        processor = AutoTokenizer.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            use_fast=True,
            add_prefix_space=True if model_args.model_name_or_path == "roberta-base" else False,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token if model_args.use_auth_token else None,
        )
    elif modality == Modality.IMAGE:
        renderer_cls = PyGameTextRenderer if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
        processor = renderer_cls.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token if model_args.use_auth_token else None,
            fallback_fonts_dir=model_args.fallback_fonts_dir,
            rgb=model_args.render_rgb,
        )
    else:
        raise ValueError("Modality not supported.")
    return processor


def get_model_and_config(model_args: argparse.Namespace, num_labels: int, task_name: str):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token if model_args.use_auth_token else None,
    }

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        attention_probs_dropout_prob=model_args.dropout_prob,
        hidden_dropout_prob=model_args.dropout_prob,
        **config_kwargs,
    )

    logger.info(f"Using dropout with probability {model_args.dropout_prob}")

    if config.model_type in ["bert", "roberta"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            **config_kwargs,
        )
    elif config.model_type in ["vit_mae", "pixel"]:
        model = PIXELForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            pooling_mode=model_args.pooling_mode,
            add_layer_norm=model_args.pooler_add_layer_norm,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported.")

    return model, config


def get_collator(
    training_args: argparse.Namespace,
    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
    modality: Modality,
    is_regression: bool = False
):
    def image_collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        if "label" in examples[0]:
            if is_regression:
                labels = torch.FloatTensor([example["label"] for example in examples])
            else:
                labels = torch.LongTensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "attention_mask": attention_mask, "labels": labels}
        return {"pixel_values": pixel_values, "attention_mask": attention_mask}

    if modality == Modality.IMAGE:
        collator = image_collate_fn
    elif modality == Modality.TEXT:
        collator = DataCollatorWithPadding(processor, pad_to_multiple_of=8) if training_args.fp16 else None
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return collator


def get_preprocess_fn(
    data_args: argparse.Namespace,
    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
    modality: Modality,
    sentence_keys: Tuple[str, Optional[str]],
):
    sentence1_key, sentence2_key = sentence_keys

    if modality == Modality.IMAGE:
        transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )
        format_fn = glue_strip_spaces

        def image_preprocess_fn(examples):
            if sentence2_key:
                encodings = [
                    processor(text=(format_fn(a), format_fn(b)))
                    for a, b in zip(examples[sentence1_key], examples[sentence2_key])
                ]
            else:
                encodings = [processor(text=format_fn(a)) for a in examples[sentence1_key]]

            examples["pixel_values"] = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
            examples["attention_mask"] = [
                get_attention_mask(e.num_text_patches, seq_length=data_args.max_seq_length) for e in encodings
            ]

            return examples

        preprocess_fn = image_preprocess_fn

    elif modality == Modality.TEXT:

        def text_preprocess_fn(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = processor(*args, padding="max_length", max_length=data_args.max_seq_length, truncation=True)

            if "label" in examples:
                result["label"] = [l for l in examples["label"]]

            return result

        preprocess_fn = text_preprocess_fn
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return preprocess_fn


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PIXELTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and config
    model, config = get_model_and_config(model_args, num_labels, data_args.task_name)

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    modality = Modality.TEXT if config.model_type in ["bert", "roberta"] else Modality.IMAGE
    processor = get_processor(model_args, modality)

    if modality == Modality.IMAGE:
        if processor.max_seq_length != data_args.max_seq_length:
            processor.max_seq_length = data_args.max_seq_length

        resize_model_embeddings(model, processor.max_seq_length)

    preprocess_fn = get_preprocess_fn(data_args, processor, modality, (sentence1_key, sentence2_key))

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        if modality == Modality.IMAGE:
            train_dataset.features["pixel_values"] = datasets.Image()
        train_dataset.set_transform(preprocess_fn)

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        if modality == Modality.IMAGE:
            eval_dataset.features["pixel_values"] = datasets.Image()
        eval_examples = copy.deepcopy(eval_dataset)
        eval_dataset.set_transform(preprocess_fn)

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        if modality == Modality.IMAGE:
            predict_dataset.features["pixel_values"] = datasets.Image()
        predict_dataset.set_transform(preprocess_fn)

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the eval set: {eval_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = PIXELTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        data_collator=get_collator(training_args, processor, modality, is_regression=is_regression),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
        if training_args.early_stopping
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        eval_examples_l = [eval_examples]

        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            mismatched_eval_dataset = raw_datasets["validation_mismatched"]
            mismatched_eval_examples = copy.deepcopy(mismatched_eval_dataset)
            eval_examples_l.append(mismatched_eval_examples)
            if modality == Modality.IMAGE:
                mismatched_eval_dataset.features["pixel_values"] = datasets.Image()
            mismatched_eval_dataset.set_transform(preprocess_fn)
            eval_datasets.append(mismatched_eval_dataset)

        for eval_dataset, eval_examples, task in zip(eval_datasets, eval_examples_l, tasks):
            logger.info(f"Task name is {task}")
            outputs = trainer.predict(test_dataset=eval_dataset, metric_key_prefix=f"eval_{task}")
            metrics = outputs.metrics

            # Log predictions
            if training_args.log_predictions:
                log_sequence_classification_predictions(
                    training_args=training_args,
                    features=eval_dataset,
                    examples=eval_examples,
                    predictions=outputs.predictions,
                    sentence1_key=sentence1_key,
                    sentence2_key=sentence2_key,
                    modality=modality,
                    prefix=task,
                )

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics(f"eval_{task}", metrics)
            trainer.save_metrics(f"eval_{task}", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mismatched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            mismatched_test_dataset = raw_datasets["test_mismatched"]
            if modality == Modality.IMAGE:
                mismatched_test_dataset.features["pixel_values"] = datasets.Image()
            mismatched_test_dataset.set_transform(preprocess_fn)
            predict_datasets.append(mismatched_test_dataset)

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix=f"predict_{task}").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
