#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import datasets
import torch
import transformers
from datasets import interleave_datasets, load_dataset
from pixel import (
    PIXELConfig,
    PIXELEmbeddings,
    PIXELForPreTraining,
    PIXELTrainerForPretraining,
    SpanMaskingGenerator,
    PyGameTextRenderer,
    get_attention_mask,
    get_transforms,
    get_2d_sincos_pos_embed
)
from transformers import HfArgumentParser, TrainingArguments, ViTFeatureExtractor
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

""" Pre-training a PIXEL model as an MAE (masked autoencoder)"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_dataset_names: str = field(metadata={"help": "Name of train dataset in HuggingFace dataset hub"})
    train_splits: str = field(metadata={"help": "Name of the training dataset split."})
    validation_dataset_name: str = field(metadata={"help": "Name of validation dataset in HuggingFace dataset hub"})
    validation_split: str = field(metadata={"help": "Name of the validation dataset split."})
    dataset_caches: Optional[str] = field(default=None, metadata={"help": "Directory where the dataset is cached"})
    train_dataset_configs: str = field(default=None, metadata={"help": "Train dataset config/subset"})
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
    streaming: Optional[bool] = field(default=False, metadata={"help": "Whether to stream the training dataset"})
    do_normalize: Optional[bool] = field(
        default=False, metadata={"help": "Whether to normalize to model's feature extractor's mean and std."}
    )

    def __post_init__(self):
        self.train_dataset_names = self.train_dataset_names.split(",")
        self.train_splits = self.train_splits.split(",")
        if self.train_dataset_configs:
            self.train_dataset_configs = self.train_dataset_configs.split(",")
        else:
            self.train_dataset_configs = [None] * len(self.train_dataset_names)
        if self.dataset_caches:
            self.dataset_caches = self.dataset_caches.split(",")
        else:
            self.dataset_caches = [None] * len(self.train_dataset_names)
        assert (
            len(self.train_dataset_names)
            == len(self.train_splits)
            == len(self.train_dataset_configs)
            == len(self.dataset_caches)
        )

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/feature extractor we are going to pre-train.
    """

    text_renderer_name_or_path: str = field(
        metadata={
            "help": "Path / Huggingface identifier of the text renderer that was used to prerender the "
            "training/validation data."
        }
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: str = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    mask_ratio: float = field(
        default=0.25, metadata={"help": "The ratio of the number of masked tokens in the input sequence."}
    )
    norm_pix_loss: bool = field(
        default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )
    span_masking: bool = field(
        default=False, metadata={"help": "Whether to use span masking instead of random masking."}
    )
    masking_max_span_length: Optional[int] = field(
        default=None, metadata={"help": "Maximum span length that can be masked when using span masking."}
    )
    masking_spacing: Optional[int] = field(
        default=None,
        metadata={
            "help": "Spacing between masked spans. Defaults to the length of the span."
            "Use this argument to set it to a fixed number of patches."
            "Recommended setting: For masking ratio <= 0.4 leave the default"
            "For ratios between 0.4 and 0.7 set it to 1. For higher, set it to 0"
        },
    )
    masking_cumulative_span_weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of cumulative probabilities of sampling a span of length n"
            "when using span masking. Must be a list of size model_args.masking_max_span_length."
        },
    )
    dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout probability for attention blocks"}
    )

    def __post_init__(self):
        if self.masking_cumulative_span_weights is not None:
            self.masking_cumulative_span_weights = [float(w) for w in self.masking_cumulative_span_weights.split(",")]

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(
        default=1.5e-4, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."}
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    inputs = {"pixel_values": pixel_values, "attention_mask": attention_mask}
    if "patch_mask" in examples[0]:
        patch_mask = torch.stack([example["patch_mask"] for example in examples])
        inputs.update({"patch_mask": patch_mask})
    return inputs


def main(config_dict: Dict[str, Any] = None):

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if not config_dict:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, training_args = parser.parse_dict(config_dict)

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
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")

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

    # Initialize our datasets
    train_datasets = [
        load_dataset(
            d_name,
            d_config,
            split=d_split,
            use_auth_token=model_args.use_auth_token,
            streaming=data_args.streaming,
            cache_dir=d_cache,
        )
        for d_name, d_config, d_split, d_cache in zip(
            data_args.train_dataset_names,
            data_args.train_dataset_configs,
            data_args.train_splits,
            data_args.dataset_caches,
        )
    ]
    dataset_sizes = [ds._info.splits.total_num_examples for ds in train_datasets]
    combined_size = sum(dataset_sizes)
    dataset_sampling_probs = [d_size / combined_size for d_size in dataset_sizes]

    train_dataset = interleave_datasets(train_datasets, probabilities=dataset_sampling_probs, seed=training_args.seed)
    logger.info("***** Interleaving training datasets *****")
    for d_name, d_config, d_split, d_sampling_prob, d_cache in zip(
        data_args.train_dataset_names,
        data_args.train_dataset_configs,
        data_args.train_splits,
        dataset_sampling_probs,
        data_args.dataset_caches,
    ):
        logger.info(
            f"\tDataset name = {d_name}, config = {d_config}, split = {d_split}, "
            f"sampling probability = {d_sampling_prob:.3f}, cache = {d_cache}"
        )

    validation_dataset = load_dataset(
        data_args.validation_dataset_name, split=data_args.validation_split, use_auth_token=model_args.use_auth_token
    )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token,
    }
    logger.info(f"Using dropout with probability {model_args.dropout_prob}")

    if model_args.config_name:
        config = PIXELConfig.from_pretrained(
            model_args.config_name,
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
    elif model_args.model_name_or_path:
        config = PIXELConfig.from_pretrained(
            model_args.model_name_or_path,
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
    else:
        config = PIXELConfig(
            attention_probs_dropout_prob=model_args.dropout_prob,
            hidden_dropout_prob=model_args.dropout_prob,
            **config_kwargs,
        )
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # Adapt config
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
            "architectures": [PIXELForPreTraining.__name__]
        }
    )

    # Create model
    if model_args.model_name_or_path:
        model = PIXELForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            **config_kwargs,
        )
    else:
        logger.info("Training new model from scratch")
        model = PIXELForPreTraining(config)

    # Load text renderer
    text_renderer = PyGameTextRenderer.from_pretrained(model_args.text_renderer_name_or_path, **config_kwargs)

    # Load or create feature extractor
    if model_args.feature_extractor_name:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_args.feature_extractor_name, **config_kwargs)
    elif model_args.model_name_or_path:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        feature_extractor = ViTFeatureExtractor()

    # Adjust image size
    image_height = text_renderer.pixels_per_patch
    image_width = text_renderer.pixels_per_patch * text_renderer.max_seq_length
    model.config.image_size = (image_height, image_width)
    model.image_size = (image_height, image_width)
    feature_extractor.size = (image_height, image_width)

    # Reinitialize embeddings
    model.vit.embeddings = PIXELEmbeddings(model.config)
    model.decoder.decoder_pos_embed = torch.nn.Parameter(
        torch.zeros((1, text_renderer.max_seq_length + 1, 512)), requires_grad=False
    )
    decoder_pos_embed = get_2d_sincos_pos_embed(
        model.decoder.decoder_pos_embed.shape[-1], int(text_renderer.max_seq_length ** 0.5), add_cls_token=True
    )
    model.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    logger.info("***** Final model config *****")
    logger.info(config)

    total_params = sum([p.numel() for p in model.parameters()])
    logger.info(f"Total parameters count: {total_params}")
    encoder_params = sum([p.numel() for p in model.vit.parameters()])
    logger.info(f"Encoder parameters count: {encoder_params}")
    encoder_embedding_params = sum([p.numel() for p in model.vit.embeddings.parameters()])
    logger.info(f"Encoder embeddings parameters count: {encoder_embedding_params}")
    decoder_params = sum([p.numel() for p in model.decoder.parameters()])
    logger.info(f"Decoder parameters count: {decoder_params}")

    # Get patch mask generator if span masking
    if model_args.span_masking and model_args.masking_max_span_length and model_args.masking_cumulative_span_weights:
        logger.info(
            f'Applying span masking with "max_span_length = {model_args.masking_max_span_length}" '
            f', "cumulative_span_weights = {model_args.masking_cumulative_span_weights}" '
            f' and "spacing = {model_args.masking_spacing if model_args.masking_spacing else "span"}"'
        )
        patch_mask_generator = SpanMaskingGenerator(
            num_patches=text_renderer.max_seq_length,
            num_masking_patches=math.ceil(model_args.mask_ratio * text_renderer.max_seq_length),
            max_span_length=model_args.masking_max_span_length,
            spacing=model_args.masking_spacing if model_args.masking_spacing else "span",
            cumulative_span_weights=model_args.masking_cumulative_span_weights,
        )

    column_names = ["pixel_values", "num_patches"]
    image_column_name = column_names[0]

    if data_args.do_normalize:
        image_mean = feature_extractor.image_mean
        image_std = feature_extractor.image_std
    else:
        image_mean, image_std = (None, None)
        feature_extractor.do_normalize = data_args.do_normalize

    # Set transformations --- resize by default and optionally apply normalization
    transforms = get_transforms(
        do_resize=True,
        size=(image_height, image_width),
        do_normalize=data_args.do_normalize,
        image_mean=image_mean,
        image_std=image_std,
    )

    logger.info(f"Applied transformations: {transforms}")

    def preprocess_images(examples):
        """Preprocess a batch of images by applying transforms."""

        examples["pixel_values"] = [transforms(image) for image in examples[image_column_name]]
        examples["attention_mask"] = [get_attention_mask(num_patches) for num_patches in examples["num_patches"]]
        if model_args.span_masking:
            examples["patch_mask"] = [
                torch.tensor(patch_mask_generator(num_patches + 1), dtype=torch.float32)
                for num_patches in examples["num_patches"]
            ]

        return examples

    if training_args.do_train:
        if data_args.streaming:
            train_dataset = train_dataset.with_format("torch")
            train_dataset = train_dataset.shuffle(training_args.seed, buffer_size=10000)
        # Filter out examples that are less than one row long in the squared input image
        train_dataset = train_dataset.filter(lambda x: (x["num_patches"] >= 22))
        # Set training transforms
        if data_args.streaming:
            train_dataset = train_dataset.map(preprocess_images, batched=True, batch_size=10000)
        else:
            train_dataset.set_transform(preprocess_images)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            validation_dataset = validation_dataset.shuffle(seed=training_args.seed).select(
                range(data_args.max_eval_samples)
            )
        # Set the validation transforms
        validation_dataset.set_transform(preprocess_images)

    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    # Initialize our trainer
    trainer = PIXELTrainerForPretraining(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        tokenizer=text_renderer,
        data_collator=collate_fn,
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
        # Also save feature extractor together with model and text renderer
        feature_extractor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "tasks": "masked-auto-encoding",
        "dataset": "wikipedia + bookcorpus",
        "tags": ["masked-auto-encoding"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
