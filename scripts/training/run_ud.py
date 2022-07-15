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
""" Finetuning a PIXEL model for dependency parsing."""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import datasets
import numpy as np
import transformers
import wandb
from pixel import (
    AutoConfig,
    AutoModelForBiaffineParsing,
    UD_HEAD_LABELS,
    Modality,
    PangoCairoTextRenderer,
    PIXELTrainerForBiaffineParsing,
    PIXELTrainingArguments,
    Split,
    PyGameTextRenderer,
    UDDataset,
    get_transforms,
    resize_model_embeddings,
)
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    default_data_collator,
    set_seed, PretrainedConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to a Universal Dependencies data folder."}
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


def log_predictions(
    args: argparse.Namespace,
    eval_dataset: UDDataset,
    outputs: EvalPrediction,
    prefix: str = "eval",
):
    # Initialize wandb if not already done
    if not args.do_train:
        wandb.init(reinit=False)

    labels = UD_HEAD_LABELS
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}

    data = []

    arc_preds, rel_preds = [preds.tolist() for preds in outputs.predictions]
    arc_labels, rel_labels = [labels.tolist() for labels in outputs.label_ids]

    out_file = os.path.join(args.output_dir, f"{prefix}_predictions.csv")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("word\tarc_label\trel_label\tarc_pred\trel_pred\n")
        for ex_id, example in enumerate(eval_dataset.examples):
            # Some examples were truncated, so we have to mask truncated parts out
            max_len = len(
                [label for label in eval_dataset.features[ex_id]["arc_labels"] if label != eval_dataset.pad_token]
            )
            if max_len < len(example.words):
                logger.warning(f"Only logging truncated predictions for example {ex_id}: {example.words}")
            for word in example.words[:max_len]:
                arc_pred = arc_preds.pop(0)
                rel_pred = label_map[rel_preds.pop(0)]
                arc_label = arc_labels.pop(0)
                rel_label = label_map[rel_labels.pop(0)]

                data.append([word, arc_label, rel_label, arc_pred, rel_pred])
                f.write(f"{word}\t{arc_label}\t{rel_label}\t{arc_pred}\t{rel_pred}\n")
            f.write("\n")

    logger.info(f"Saved predictions and labels to {out_file}")
    logger.info(f"Logging as table to wandb")

    preds_table = wandb.Table(columns=["word", "arc_label", "rel_label", "arc_pred", "rel_pred"], data=data)
    wandb.log({f"{prefix}_outputs": preds_table})


def get_dataset(
    data_args: argparse.Namespace,
    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
    modality: Modality,
    split: Split,
    config: PretrainedConfig
):

    if modality == Modality.IMAGE:
        transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )
    else:
        transforms = None

    return UDDataset(
        data_dir=data_args.data_dir,
        processor=processor,
        transforms=transforms,
        modality=modality,
        labels=UD_HEAD_LABELS,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=split,
        pad_token=config.pad_token_id
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


def get_model_and_config(model_args: argparse.Namespace):

    labels = UD_HEAD_LABELS
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

    if config.model_type in ["vit_mae", "pixel", "bert"]:
        config.pad_token_id = -100

    if config.model_type in ["vit_mae", "pixel", "bert", "roberta"]:
        model = AutoModelForBiaffineParsing.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported.")

    return model, config


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

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

    # Prepare for UD dependency parsing task
    training_args.label_names = ["arc_labels", "rel_labels"]
    label_map: Dict[int, str] = {i: label for i, label in enumerate(UD_HEAD_LABELS)}

    # Load pretrained model
    model, config = get_model_and_config(model_args)

    # Set modality
    modality = Modality.TEXT if model.config.model_type in ["bert", "roberta"] else Modality.IMAGE

    # Load text renderer when using image modality and tokenizer when using text modality
    processor = get_processor(model_args, modality)

    if modality == Modality.IMAGE:
        if processor.max_seq_length != data_args.max_seq_length:
            processor.max_seq_length = data_args.max_seq_length

        resize_model_embeddings(model, processor.max_seq_length)

    train_dataset = get_dataset(data_args, processor, modality, Split.TRAIN, config) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, processor, modality, Split.DEV, config) if training_args.do_eval else None
    test_dataset = get_dataset(data_args, processor, modality, Split.TEST, config) if training_args.do_predict else None

    def compute_metrics(p: EvalPrediction):

        arc_labels, rel_labels = p.label_ids
        arc_preds, rel_preds = p.predictions

        correct_arcs = np.equal(arc_preds, arc_labels)
        correct_rels = np.equal(rel_preds, rel_labels)
        correct_arcs_and_rels = correct_arcs * correct_rels

        unlabeled_correct = correct_arcs.sum()
        labeled_correct = correct_arcs_and_rels.sum()
        total_words = correct_arcs.size

        unlabeled_attachment_score = unlabeled_correct / total_words
        labeled_attachment_score = labeled_correct / total_words

        return {
            "uas": unlabeled_attachment_score * 100,
            "las": labeled_attachment_score * 100,
        }

    # Initialize our Trainer
    trainer = PIXELTrainerForBiaffineParsing(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
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
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        outputs = trainer.predict(test_dataset=eval_dataset, metric_key_prefix="eval")
        metrics = outputs.metrics

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if training_args.log_predictions:
            log_predictions(
                args=training_args,
                eval_dataset=eval_dataset,
                outputs=outputs,
                prefix="eval"
            )

    # Test
    if training_args.do_predict:
        logger.info("*** Test ***")

        outputs = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test")
        metrics = outputs.metrics

        metrics["test_samples"] = len(test_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        if training_args.log_predictions:
            log_predictions(
                args=training_args,
                eval_dataset=test_dataset,
                outputs=outputs,
                prefix="test"
            )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
