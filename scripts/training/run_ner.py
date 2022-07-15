# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Fine-tuning the library models for named entity recognition"""
import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import transformers
import wandb

from pixel import (
    AutoConfig,
    AutoModelForTokenClassification,
    Modality,
    NERDataset,
    PangoCairoTextRenderer,
    PIXELTrainer,
    PIXELTrainingArguments,
    Split,
    PyGameTextRenderer,
    get_transforms,
    resize_model_embeddings,
)
from pixel.data.datasets.ner_dataset import get_labels as get_ner_labels
from pixel.data.datasets.ner_dataset import write_predictions_to_file as write_ner_predictions_to_file
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    default_data_collator,
    set_seed, PretrainedConfig,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained processor or model identifier from huggingface.co/models"}
    )
    rendering_backend: Optional[str] = field(
        default="pangocairo", metadata={
            "help": "Rendering backend to use. Options are 'pygame' or 'pangocairo'. For most applications it is "
                    "recommended to use the default 'pangocairo'."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
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
    dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout probability for attention blocks and classification head"}
    )

    def __post_init__(self):
        if not self.rendering_backend.lower() in ["pygame", "pangocairo"]:
            raise ValueError("Invalid rendering backend. Supported backends are 'pygame' and 'pangocairo'.")
        else:
            self.rendering_backend = self.rendering_backend.lower()


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=196,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def log_predictions(args: argparse.Namespace, eval_dataset: NERDataset, outputs: EvalPrediction, prefix: str = "eval"):
    # Initialize wandb if not already done
    if not args.do_train:
        wandb.init(reinit=False)

    # Uses default labels
    labels = get_ner_labels("")

    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}

    data = []

    out_file = os.path.join(args.output_dir, f"{prefix}_predictions.csv")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("word\tlabel\tpred\n")
        for ex_id, example in enumerate(eval_dataset.examples):

            ex_mask = np.not_equal(outputs.label_ids[ex_id], np.full_like(outputs.label_ids[ex_id], -100))
            ex_labels = outputs.label_ids[ex_id][ex_mask].tolist()
            ex_preds = [np.argmax(pred) for pred in outputs.predictions[ex_id][ex_mask]]

            if len(example.words) != len(ex_labels):
                logger.warning(f"Fewer labels than words in example {ex_id}: {' '.join(example.words)}")

            for word in example.words:
                if len(ex_preds) == 0 or len(ex_labels) == 0:
                    logger.warning(f"Fewer labels than words in example {ex_id}: {' '.join(example.words)}")
                    continue

                pred = label_map[ex_preds.pop(0)]
                label = label_map[ex_labels.pop(0)]

                data.append([word, label, pred])
                f.write(f"{word}\t{label}\t{pred}\n")
            f.write("\n")

    logger.info(f"Saved predictions and labels to {out_file}")
    logger.info(f"Logging as table to wandb")

    preds_table = wandb.Table(columns=["word", "label", "pred"], data=data)
    wandb.log({f"{prefix}_outputs": preds_table})


def get_dataset(
    config: PretrainedConfig,
    data_args: argparse.Namespace,
    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
    modality: Modality,
    split: Split,
    labels: List[str],
):
    kwargs = {}
    if modality == Modality.IMAGE:

        transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )
    else:
        transforms = None
        kwargs.update({
            "sep_token_extra": bool(config.model_type in ["roberta"]),
            "cls_token": processor.cls_token,
            "sep_token": processor.sep_token,
            "pad_token": processor.convert_tokens_to_ids([processor.pad_token])[0]
        })

    return NERDataset(
        data_dir=data_args.data_dir,
        processor=processor,
        transforms=transforms,
        modality=modality,
        labels=labels,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=split,
        **kwargs
    )


def get_processor(model_args: argparse.Namespace, modality: Modality):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token if model_args.use_auth_token else None,
    }

    if modality == Modality.TEXT:
        processor = AutoTokenizer.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            use_fast=True,
            add_prefix_space=True if model_args.model_name_or_path == "roberta-base" else False,
            **config_kwargs,
        )
    elif modality == Modality.IMAGE:
        renderer_cls = PyGameTextRenderer if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
        processor = renderer_cls.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            fallback_fonts_dir=model_args.fallback_fonts_dir,
            rgb=model_args.render_rgb,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return processor


def get_model_and_config(model_args: argparse.Namespace, labels: List[str]):
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token if model_args.use_auth_token else None,
    }

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        attention_probs_dropout_prob=model_args.dropout_prob,
        hidden_dropout_prob=model_args.dropout_prob,
        **config_kwargs,
    )
    logger.info(f"Using dropout with probability {model_args.dropout_prob}")

    if config.model_type in ["vit_mae", "pixel", "bert", "roberta"]:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported.")

    return model, config


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
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info("Training/evaluation parameters %s", training_args)

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

    # Set seed
    set_seed(training_args.seed)

    # Prepare NER task
    labels = get_ner_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}

    # Load pretrained model
    model, config = get_model_and_config(model_args, labels)

    # Set modality
    modality = Modality.TEXT if model.config.model_type in ["bert", "roberta"] else Modality.IMAGE

    # Load text renderer when using image modality and tokenizer when using text modality
    processor = get_processor(model_args, modality)

    if modality == Modality.IMAGE:
        if processor.max_seq_length != data_args.max_seq_length:
            processor.max_seq_length = data_args.max_seq_length

        resize_model_embeddings(model, processor.max_seq_length)

    train_dataset = (
        get_dataset(config, data_args, processor, modality, Split.TRAIN, labels) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(config, data_args, processor, modality, Split.DEV, labels) if training_args.do_eval else None
    )
    test_dataset = (
        get_dataset(config, data_args, processor, modality, Split.TEST, labels) if training_args.do_predict else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    # Data collator
    data_collator = default_data_collator

    # Initialize our Trainer
    trainer = PIXELTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=processor,
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

        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        outputs = trainer.predict(test_dataset=eval_dataset, metric_key_prefix="eval")
        metrics = outputs.metrics

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if training_args.log_predictions:
            log_predictions(args=training_args, eval_dataset=eval_dataset, outputs=outputs, prefix="eval")

    # Predict
    if training_args.do_predict:
        outputs = trainer.predict(test_dataset)
        preds_list, _ = align_predictions(outputs.predictions, outputs.label_ids)
        metrics = outputs.metrics
        metrics["test_samples"] = len(test_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        if training_args.log_predictions:
            log_predictions(args=training_args, eval_dataset=test_dataset, outputs=outputs, prefix="test")

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
                    write_ner_predictions_to_file(writer, f, preds_list)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
