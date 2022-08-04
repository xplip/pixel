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
""" Fine-tuning the library models for question-answering."""
import argparse
import logging
import os
import string
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
import transformers
from datasets import Array3D, Features, Sequence, Value, load_dataset, load_metric
from PIL import Image
from pixel import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    Modality,
    PangoCairoTextRenderer,
    PIXELTrainerForQuestionAnswering,
    PIXELTrainingArguments,
    Split,
    get_attention_mask,
    get_transforms,
    postprocess_qa_predictions,
    resize_model_embeddings,
)
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=2.0.0", "To fix: pip install ./datasets")

logger = logging.getLogger(__name__)


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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
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


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=400,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    question_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum number of patches used up by the question. If set, questions longer than this parameter "
            "will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
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
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=160,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    is_rtl_language: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether all examples are written in a right-to-left (RTL) script. If not set,"
            " the text direction will be inferred from the content for each example."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


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


def get_model_and_config(model_args: argparse.Namespace):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token if model_args.use_auth_token else None,
    }

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        attention_probs_dropout_prob=model_args.dropout_prob,
        hidden_dropout_prob=model_args.dropout_prob,
        **config_kwargs,
    )

    logger.info(f"Using dropout with probability {model_args.dropout_prob}")

    if config.model_type in ["vit_mae", "pixel", "bert", "roberta"]:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported.")

    return model, config


def get_collator(modality: Modality):
    def image_collate_fn(examples):
        if not isinstance(examples[0]["pixel_values"], torch.Tensor):
            for example in examples:
                example["pixel_values"] = torch.Tensor(example["pixel_values"])

        if not isinstance(examples[0]["attention_mask"], torch.Tensor):
            for example in examples:
                example["attention_mask"] = torch.Tensor(example["attention_mask"])

        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        batch = {"pixel_values": pixel_values, "attention_mask": attention_mask}
        if "start_positions" in examples[0]:
            batch.update({"start_positions": torch.LongTensor([example["start_positions"] for example in examples])})
        if "end_positions" in examples[0]:
            batch.update({"end_positions": torch.LongTensor([example["end_positions"] for example in examples])})

        # Uncomment this to visualize inputs
        # debug_log_inputs(batch)

        return batch

    if modality == Modality.IMAGE:
        collator = image_collate_fn
    elif modality == Modality.TEXT:
        collator = None
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return collator


def get_preprocess_fn(
    data_args: argparse.Namespace,
    processor: Union[PangoCairoTextRenderer, PreTrainedTokenizerFast],
    modality: Modality,
    split: Split,
    column_names: List[str],
):
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    if modality == Modality.IMAGE:
        transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )

        def prepare_train_image_features(examples):
            encodings = [
                processor(
                    text=(q.strip(), c.strip()),
                    stride=data_args.doc_stride,
                    return_overflowing_patches=True,
                    return_offset_mapping=True,
                    text_a_max_length=data_args.question_max_length,
                    rtl=data_args.is_rtl_language,
                )
                for q, c in zip(examples[question_column_name], examples[context_column_name])
            ]
            sample_mapping = []
            full_encodings = []
            for sample_id, encoding in enumerate(encodings):
                full_encodings.append(encoding)
                sample_mapping.append(sample_id)
                if encoding.overflowing_patches is not None:
                    for overflow_encoding in encoding.overflowing_patches:
                        full_encodings.append(overflow_encoding)
                        sample_mapping.append(sample_id)

            processed_examples = {}
            processed_examples["pixel_values"] = [
                transforms(Image.fromarray(encoding.pixel_values)) for encoding in full_encodings
            ]
            processed_examples["attention_mask"] = [
                get_attention_mask(encoding.num_text_patches, seq_length=data_args.max_seq_length)
                for encoding in full_encodings
            ]

            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = [encoding.offset_mapping for encoding in full_encodings]

            # Let's label those examples!
            processed_examples["start_positions"] = []
            processed_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                cls_index = processor.max_seq_length

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]
                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    processed_examples["start_positions"].append(cls_index)
                    processed_examples["end_positions"].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    char_idx = 0
                    # We strip whitespaces before rendering which can misalign the context string with the start index
                    # Therefore, we shift the starting index to the left by each leading whitespace in the context
                    while examples[context_column_name][sample_index][char_idx] in string.whitespace:
                        start_char -= 1
                        char_idx += 1

                    text_key = "text" if "text" in answers else "answer_text"
                    end_char = start_char + len(answers[text_key][0])

                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while offsets[token_start_index] != (0, 0):
                        token_start_index += 1
                    token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(processed_examples["attention_mask"][i]) - 1
                    while offsets[token_end_index] == (0, 0) or offsets[token_end_index] == (-1, -1):
                        token_end_index -= 1

                    # Detect if the answer is out of the span (in which case the feature is labeled with the CLS index).
                    if not (
                        data_args.is_rtl_language or processor.is_rtl(examples[context_column_name][sample_index])
                    ):
                        if not (
                            offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char
                        ):
                            processed_examples["start_positions"].append(cls_index)
                            processed_examples["end_positions"].append(cls_index)
                        else:
                            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                            # Note: we could go after the last offset if the answer is the last word (edge case).
                            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                                token_start_index += 1
                            processed_examples["start_positions"].append(token_start_index - 1)
                            while offsets[token_end_index][1] >= end_char:
                                token_end_index -= 1
                            processed_examples["end_positions"].append(token_end_index + 1)
                    else:
                        if not (
                            offsets[token_end_index][1] <= start_char and offsets[token_start_index][0] >= end_char
                        ):
                            processed_examples["start_positions"].append(cls_index)
                            processed_examples["end_positions"].append(cls_index)
                        else:
                            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                            # Note: we could go after the last offset if the answer is the last word (edge case).
                            while token_start_index < len(offsets) and offsets[token_start_index][0] >= end_char:
                                token_start_index += 1
                            processed_examples["start_positions"].append(token_start_index - 1)
                            while offsets[token_end_index][1] <= start_char and offsets[token_end_index] != (0, 0):
                                token_end_index -= 1
                            processed_examples["end_positions"].append(token_end_index + 1)

            return processed_examples

        def prepare_validation_image_features(examples):
            encodings = [
                processor(
                    text=(q.strip(), c.strip()),
                    stride=data_args.doc_stride,
                    return_overflowing_patches=True,
                    return_offset_mapping=True,
                    text_a_max_length=data_args.question_max_length,
                    rtl=data_args.is_rtl_language,
                )
                for q, c in zip(examples[question_column_name], examples[context_column_name])
            ]
            sample_mapping = []
            full_encodings = []
            for sample_id, encoding in enumerate(encodings):
                full_encodings.append(encoding)
                sample_mapping.append(sample_id)
                if encoding.overflowing_patches is not None:
                    for overflow_encoding in encoding.overflowing_patches:
                        full_encodings.append(overflow_encoding)
                        sample_mapping.append(sample_id)

            processed_examples = {}
            processed_examples["pixel_values"] = np.array(
                [transforms(Image.fromarray(encoding.pixel_values)).numpy() for encoding in full_encodings]
            )
            processed_examples["attention_mask"] = np.array(
                [
                    get_attention_mask(encoding.num_text_patches, seq_length=data_args.max_seq_length).numpy()
                    for encoding in full_encodings
                ]
            )

            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            processed_examples["offset_mapping"] = [encoding.offset_mapping for encoding in full_encodings]

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            processed_examples["example_id"] = []

            for i, offsets in enumerate(processed_examples["offset_mapping"]):
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                processed_examples["example_id"].append(examples["id"][sample_index])

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                patch_start_idx = 0
                while offsets[patch_start_idx] != (0, 0):
                    offsets[patch_start_idx] = None
                    patch_start_idx += 1
                offsets[patch_start_idx] = None

                # End token index of the current span in the text.
                patch_end_index = len(processed_examples["attention_mask"][i]) - 1
                while offsets[patch_end_index] == (0, 0):
                    offsets[patch_end_index] = None
                    patch_end_index -= 1

            return processed_examples

        preprocess_fn = prepare_train_image_features if split == Split.TRAIN else prepare_validation_image_features

    elif modality == Modality.TEXT:

        pad_on_right = processor.padding_side == "right"

        def prepare_train_text_features(examples):
            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = processor(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=data_args.max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # Let's label those examples!
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(processor.cls_token_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]
                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1

                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)

            return tokenized_examples

        # Validation preprocessing
        def prepare_validation_text_features(examples):
            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = processor(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=data_args.max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

            return tokenized_examples

        preprocess_fn = prepare_train_text_features if split == Split.TRAIN else prepare_validation_text_features
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return preprocess_fn


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, PIXELTrainingArguments))
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
    logger.info(f"Data arguments {data_args}")

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

    model, config = get_model_and_config(model_args)

    modality = Modality.TEXT if config.model_type in ["bert", "roberta"] else Modality.IMAGE
    processor = get_processor(model_args, modality)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=model_args.use_auth_token if model_args.use_auth_token else None,
            ignore_verifications=True,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=model_args.cache_dir)

    # Preprocessing the datasets.
    # Preprocessing is slightly different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    if modality == Modality.IMAGE:
        if processor.max_seq_length != data_args.max_seq_length:
            processor.max_seq_length = data_args.max_seq_length

        resize_model_embeddings(model, processor.max_seq_length)

    train_preprocess_fn = get_preprocess_fn(
        data_args=data_args,
        processor=processor,
        modality=modality,
        split=Split.TRAIN,
        column_names=column_names,
    )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        if modality == Modality.IMAGE:
            train_dataset.features["pixel_values"] = datasets.Image()
            train_dataset.set_transform(train_preprocess_fn)
            # We render the training data on the fly because QA datasets are often large and would take 100+ GB on disk
            # If you have enough space, feel free to pre-render as we do for the validation set below. This will
            # speed up training

        else:
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    train_preprocess_fn,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )

    eval_preprocess_fn = get_preprocess_fn(
        data_args=data_args,
        processor=processor,
        modality=modality,
        split=Split.DEV,
        column_names=column_names,
    )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        if modality == Modality.IMAGE:
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                features = Features(
                    {
                        "pixel_values": Array3D(
                            dtype="float32", shape=(3, processor.pixels_per_patch, processor.max_pixels_len)
                        ),
                        "attention_mask": Sequence(Value(dtype="int32")),
                        "offset_mapping": Sequence(Sequence(Value(dtype="int32"))),
                        "example_id": Value(dtype="string", id=None),
                    }
                )
                eval_dataset = eval_examples.map(
                    eval_preprocess_fn,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running renderer on validation dataset",
                    features=features,
                )
            eval_dataset = eval_dataset.with_format(
                "numpy", columns=["pixel_values", "attention_mask"], output_all_columns=True
            )
        else:
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_examples.map(
                    eval_preprocess_fn,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        if modality == Modality.IMAGE:
            predict_examples.features["pixel_values"] = datasets.Image()
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                eval_preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            modality=modality,
            rtl=data_args.is_rtl_language,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        def remove_extra_keys(d: dict) -> dict:
            return {
                "text" if k == "answer_text" else k: v
                for k, v in d.items()
                if k in ["text", "answer_start", "answer_text"]
            }

        references = [{"id": ex["id"], "answers": remove_extra_keys(ex[answer_column_name])} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    total_params = sum([p.numel() for p in model.parameters()])
    logger.info(f"Total parameters count: {total_params}")

    trainer = PIXELTrainerForQuestionAnswering(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=processor,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        data_collator=get_collator(modality),
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
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
