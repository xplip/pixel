import glob
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from filelock import FileLock
from PIL import Image
from transformers import PreTrainedTokenizerFast, is_torch_available

from ...utils import Modality, Split, get_attention_mask
from ..rendering import PyGameTextRenderer, PangoCairoTextRenderer

logger = logging.getLogger(__name__)


@dataclass
class UDInputExample:
    """
    A single training/test example for universal dependency parsing.
    """

    words: List[str]
    head_labels: Optional[List[int]]
    rel_labels: Optional[List[str]]


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class UDDataset(Dataset):
        """
        Pytorch Dataset for universal dependency parsing.
        """

        features: List[Dict[str, Union[int, torch.Tensor]]]

        def __init__(
            self,
            data_dir: str,
            processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
            modality: Modality,
            labels: List[str],
            transforms: Optional[Callable] = None,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.TRAIN,
            **kwargs
        ):

            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}".format(mode.value, processor.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.examples = read_examples_from_file(data_dir, mode)
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    self.examples = read_examples_from_file(data_dir, mode)
                    examples_to_features_fn = _get_examples_to_features_fn(modality)
                    # Also store pad_token because we need this later on if we want to log truncated predictions
                    self.features, self.pad_token = examples_to_features_fn(
                        examples=self.examples,
                        label_list=labels,
                        max_seq_length=max_seq_length,
                        processor=processor,
                        transforms=transforms,
                        **kwargs
                    )
                    logger.info(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> Dict[str, Union[int, torch.Tensor]]:
            return self.features[i]


def get_file(data_dir: str, mode: Union[Split, str]) -> Optional[str]:
    if isinstance(mode, Split):
        mode = mode.value
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    fp = os.path.join(data_dir, f"*-ud-{mode}.conllu")
    _fp = glob.glob(fp)
    if len(_fp) == 1:
        return _fp[0]
    elif len(_fp) == 0:
        return None
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[UDInputExample]:
    file_path = get_file(data_dir, mode)
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        words: List[str] = []
        head_labels: List[int] = []
        rel_labels: List[str] = []
        for line in f.readlines():
            tok = line.strip().split("\t")
            if len(tok) < 2 or line[0] == "#":
                if words:
                    examples.append(UDInputExample(words=words, head_labels=head_labels, rel_labels=rel_labels))
                    words = []
                    head_labels = []
                    rel_labels = []
            if tok[0].isdigit():
                word, head, label = tok[1], tok[6], tok[7]
                words.append(word)
                head_labels.append(int(head))
                rel_labels.append(label.split(":")[0])
        if words:
            examples.append(UDInputExample(words=words, head_labels=head_labels, rel_labels=rel_labels))
    return examples


def _get_examples_to_features_fn(modality: Modality):
    if modality == Modality.IMAGE:
        return convert_examples_to_image_features
    if modality == Modality.TEXT:
        return convert_examples_to_text_features
    else:
        raise ValueError("Modality not supported.")


def convert_examples_to_image_features(
    examples: List[UDInputExample],
    label_list: List[str],
    max_seq_length: int,
    processor: Union[PyGameTextRenderer, PangoCairoTextRenderer],
    transforms: Optional[Callable] = None,
    pad_token=-100,
    *kwargs
) -> Tuple[List[Dict[str, Union[int, torch.Tensor]]], int]:
    """Loads a data file into a list of `Dict` containing image features"""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")

        encoding = processor(example.words)
        image = encoding.pixel_values
        num_patches = encoding.num_text_patches
        word_starts = encoding.word_starts

        pixel_values = transforms(Image.fromarray(image))
        attention_mask = get_attention_mask(num_patches, seq_length=max_seq_length)

        pad_item = [pad_token]

        if len(example.head_labels) > max_seq_length:
            logger.warning("Sequence %d of len %d truncated: %s", ex_index, len(example.head_labels), example.words)

        # pad or truncate arc labels
        arc_labels = example.head_labels[: min(max_seq_length, len(example.head_labels))]
        arc_labels = [pad_token if al > max_seq_length else al for al in arc_labels]
        arc_labels = arc_labels + (max_seq_length - len(arc_labels)) * pad_item

        # convert rel labels from map, pad or truncate if necessary
        rel_labels = [label_map[i] for i in example.rel_labels[: min(max_seq_length, len(example.rel_labels))]]
        rel_labels = rel_labels + (max_seq_length - len(rel_labels)) * pad_item

        # determine start indices of words, pad or truncate if necessary
        word_starts = word_starts + (max_seq_length + 1 - len(word_starts)) * pad_item

        # sanity check lengths
        assert len(attention_mask) == max_seq_length
        assert len(arc_labels) == max_seq_length
        assert len(rel_labels) == max_seq_length
        assert len(word_starts) == max_seq_length + 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"sentence: {' '.join(example.words)}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"arc_labels: {arc_labels}")
            logger.info(f"rel_labels: {rel_labels}")
            logger.info(f"word_starts: {word_starts}")

        features.append(
            {
                "pixel_values": pixel_values,
                "attention_mask": attention_mask,
                "word_starts": word_starts,
                "arc_labels": arc_labels,
                "rel_labels": rel_labels,
            }
        )

    return features, pad_token


def convert_examples_to_text_features(
    examples: List[UDInputExample],
    label_list: List[str],
    max_seq_length: int,
    processor: PreTrainedTokenizerFast,
    transforms: Optional[Callable] = None,
    pad_token=-100,
    *kwargs
) -> Tuple[List[Dict[str, Union[int, torch.Tensor]]], int]:
    """Loads a data file into a list of `Dict` containing text features"""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = [processor.tokenize(w) for w in example.words]
        word_lengths = [len(w) for w in tokens]
        tokens_merged = []
        list(map(tokens_merged.extend, tokens))

        if 0 in word_lengths:
            logger.warning("Invalid sequence with word length 0 filtered: %s", example.words)
            # continue

        # Warn if a sequence is too long and gets truncated
        if len(tokens_merged) >= (max_seq_length - 2):
            logger.warning("Sequence %d of len %d truncated: %s", ex_index, len(tokens_merged), tokens_merged)

        encoding = processor(
            example.words,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            is_split_into_words=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        input_ids = encoding["input_ids"]
        token_type_ids = encoding["token_type_ids"]
        attention_mask = encoding["attention_mask"]

        pad_item = [pad_token]

        word_starts = [w for w in np.cumsum([1] + word_lengths).tolist() if w < max_seq_length]
        arc_labels = example.head_labels[: len(word_starts)]
        arc_labels = [pad_token if al > max_seq_length else al for al in arc_labels]
        rel_labels = [label_map[i] for i in example.rel_labels[: len(word_starts)]]

        # pad
        arc_labels = arc_labels + (max_seq_length - len(arc_labels)) * pad_item
        rel_labels = rel_labels + (max_seq_length - len(rel_labels)) * pad_item
        word_starts = word_starts + (max_seq_length + 1 - len(word_starts)) * pad_item

        # sanity check lengths
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(arc_labels) == max_seq_length
        assert len(arc_labels) == max_seq_length
        assert len(word_starts) == max_seq_length + 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"sentence: {' '.join(example.words)}")
            logger.info(f"input_ids: {input_ids}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"arc_labels: {arc_labels}")
            logger.info(f"rel_labels: {rel_labels}")
            logger.info(f"word_starts: {word_starts}")

        features.append(
            {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "word_starts": word_starts,
                "arc_labels": arc_labels,
                "rel_labels": rel_labels,
            }
        )

    return features, pad_token
