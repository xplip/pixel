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
""" Finetuning the library models for sequence classification on XNLI in the translate-train-all setting."""

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
from datasets import concatenate_datasets, load_dataset, load_metric
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
    PreTrainedTokenizerFast,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


logger = logging.getLogger(__name__)

XNLI_LANGUAGES = ["en", "fr", "es", "el", "de", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

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
    fallback_fonts_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing fallback font files used by the text renderer for PIXEL."},
    )
    render_rgb: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to render images in RGB. RGB rendering can be useful when working with emoji "
            "but it makes rendering a bit slower, so it is recommended to turn on RGB rendering only "
            "when there is need for it"
        },
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
        default=0.1, metadata={"help": "Dropout probability for attention blocks and " "classification head"}
    )

    def __post_init__(self):
        self.pooling_mode = PoolingMode.from_string(self.pooling_mode)


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
        processor = PangoCairoTextRenderer.from_pretrained(
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


def get_model_and_config(model_args: argparse.Namespace, num_labels: int, dataset_name: str):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token if model_args.use_auth_token else None,
    }

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=dataset_name,
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
    processor: Union[PangoCairoTextRenderer, PreTrainedTokenizerFast],
    modality: Modality,
):
    def image_collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        if "label" in examples[0]:
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
    processor: Union[PangoCairoTextRenderer, PreTrainedTokenizerFast],
    modality: Modality,
    sentence_keys: Tuple[str, Optional[str]],
):
    sentence1_key, sentence2_key = sentence_keys

    if modality == Modality.IMAGE:

        transforms = get_transforms(
            do_resize=True, size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length)
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
            if "label" in examples:
                examples["label"] = [l if l != -1 else -100 for l in examples["label"]]

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
                result["label"] = [l if l != -1 else -100 for l in examples["label"]]

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

    dataset_name = "xnli"

    if training_args.do_train:
        train_datasets = [
            load_dataset(
                dataset_name,
                lang,
                split="train",
                use_auth_token=model_args.use_auth_token,
            )
            for lang in XNLI_LANGUAGES
        ]

        train_dataset = concatenate_datasets(train_datasets)
        label_list = train_dataset.features["label"].names

    if training_args.do_eval:
        eval_datasets = [
            load_dataset(
                dataset_name,
                lang,
                split="validation",
                use_auth_token=model_args.use_auth_token,
            )
            for lang in XNLI_LANGUAGES
        ]
        eval_dataset = concatenate_datasets(eval_datasets)
        label_list = eval_dataset.features["label"].names

    if training_args.do_predict:
        predict_datasets = [
            load_dataset(
                dataset_name,
                lang,
                split="test",
                use_auth_token=model_args.use_auth_token,
            )
            for lang in XNLI_LANGUAGES
        ]
        predict_dataset = concatenate_datasets(predict_datasets)
        label_list = predict_dataset.features["label"].names

    # Labels
    num_labels = len(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}

    # Load pretrained model and config
    model, config = get_model_and_config(model_args, num_labels, dataset_name)

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    sentence1_key, sentence2_key = ("premise", "hypothesis")

    modality = Modality.TEXT if config.model_type in ["bert", "roberta"] else Modality.IMAGE
    processor = get_processor(model_args, modality)

    if modality == Modality.IMAGE:
        if processor.max_seq_length != data_args.max_seq_length:
            processor.max_seq_length = data_args.max_seq_length

        resize_model_embeddings(model, processor.max_seq_length)

    preprocess_fn = get_preprocess_fn(data_args, processor, modality, (sentence1_key, sentence2_key))

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        if modality == Modality.IMAGE:
            train_dataset.features["pixel_values"] = datasets.Image()
        train_dataset.set_transform(preprocess_fn)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        if modality == Modality.IMAGE:
            eval_dataset.features["pixel_values"] = datasets.Image()
        eval_examples_l = [copy.deepcopy(e) for e in eval_datasets]
        eval_dataset.set_transform(preprocess_fn)
        [e.set_transform(preprocess_fn) for e in eval_datasets]

    if training_args.do_predict or data_args.test_file is not None:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        if modality == Modality.IMAGE:
            predict_dataset.features["pixel_values"] = datasets.Image()
        predict_examples_l = [copy.deepcopy(e) for e in predict_datasets]
        predict_dataset.set_transform(preprocess_fn)
        [e.set_transform(preprocess_fn) for e in predict_datasets]

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the eval set: {eval_dataset[index]}.")

    # Get the metric function
    metric = load_metric("xnli")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Initialize our Trainer
    trainer = PIXELTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        data_collator=get_collator(training_args, processor, modality),
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

        for lang, eval_dataset, eval_examples in zip(XNLI_LANGUAGES, eval_datasets, eval_examples_l):
            logger.info(f"Evaluating {lang}")

            outputs = trainer.predict(test_dataset=eval_dataset, metric_key_prefix=f"eval_{lang}")
            metrics = outputs.metrics

            # Log predictions to understand where model goes wrong
            if training_args.log_predictions:
                log_sequence_classification_predictions(
                    training_args=training_args,
                    features=eval_dataset,
                    examples=eval_examples,
                    predictions=outputs.predictions,
                    sentence1_key=sentence1_key,
                    sentence2_key=sentence2_key,
                    modality=modality,
                    prefix=f"eval_{lang}",
                )

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics(f"eval_{lang}", metrics)
            trainer.save_metrics(f"eval_{lang}", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        for lang, predict_dataset, predict_examples in zip(XNLI_LANGUAGES, predict_datasets, predict_examples_l):
            logger.info(f"Predicting {lang}")

            outputs = trainer.predict(test_dataset=predict_dataset, metric_key_prefix=f"test_{lang}")
            metrics = outputs.metrics

            # Log predictions to understand where model goes wrong
            if training_args.log_predictions:
                log_sequence_classification_predictions(
                    training_args=training_args,
                    features=predict_dataset,
                    examples=predict_examples,
                    predictions=outputs.predictions,
                    sentence1_key=sentence1_key,
                    sentence2_key=sentence2_key,
                    modality=modality,
                    prefix=f"test_{lang}",
                )

            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(eval_dataset))

            trainer.log_metrics(f"test_{lang}", metrics)
            trainer.save_metrics(f"test_{lang}", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.dataset_name is not None:
        kwargs["language"] = XNLI_LANGUAGES
        kwargs["dataset_tags"] = "xnli-translate-train-all"
        kwargs["dataset_args"] = "xnli-translate-train-all"
        kwargs["dataset"] = "xnli-translate-train-all"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
