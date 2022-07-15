import glob
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import torch
from filelock import FileLock
from PIL import Image
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, is_torch_available

from ...utils import Modality, Split, get_attention_mask
from ..rendering import PyGameTextRenderer, PangoCairoTextRenderer

logger = logging.getLogger(__name__)


@dataclass
class POSInputExample:
    """
    A single training/test example for POS tagging.
    Args:
        words: list. The words of the sequence.
        labels: (Optional) list. The parts-of-speech labels for each word
    """

    words: List[str]
    labels: Optional[List[str]]


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class POSDataset(Dataset):
        """
        Pytorch Dataset for POS tagging.
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
                    self.features = examples_to_features_fn(
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


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[POSInputExample]:
    file_path = get_file(data_dir, mode)
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        words: List[str] = []
        labels: List[str] = []
        for line in f.readlines():
            tok = line.strip().split("\t")
            if len(tok) < 2 or line[0] == "#":
                if words:
                    examples.append(POSInputExample(words=words, labels=labels))
                    words = []
                    labels = []
            if tok[0].isdigit():
                word, label = tok[1], tok[3]
                words.append(word)
                labels.append(label)
        if words:
            examples.append(POSInputExample(words=words, labels=labels))
    return examples


def _get_examples_to_features_fn(modality: Modality):
    if modality == Modality.IMAGE:
        return convert_examples_to_image_features
    if modality == Modality.TEXT:
        return convert_examples_to_text_features
    else:
        raise ValueError("Modality not supported.")


def convert_examples_to_image_features(
    examples: List[POSInputExample],
    label_list: List[str],
    max_seq_length: int,
    processor: Union[PyGameTextRenderer, PangoCairoTextRenderer],
    transforms: Optional[Callable] = None,
    pad_token_label_id=-100,
    **kwargs
) -> List[Dict[str, Union[int, torch.Tensor, List[int]]]]:
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

        label_ids = [pad_token_label_id] * max_seq_length
        for idx, word_start in enumerate(word_starts[:-1]):
            label_ids[word_start] = label_map[example.labels[idx]]

        pixel_values = transforms(Image.fromarray(image))
        attention_mask = get_attention_mask(num_patches, seq_length=max_seq_length)

        # sanity check lengths
        assert len(attention_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"sentence: {' '.join(example.words)}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"label_ids: {label_ids}")

        features.append({"pixel_values": pixel_values, "attention_mask": attention_mask, "label_ids": label_ids})

    return features


def convert_examples_to_text_features(
    examples: List[POSInputExample],
    label_list: List[str],
    max_seq_length: int,
    processor: PreTrainedTokenizer,
    # pass the ones below as kwargs to the dataset __init__
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    **kwargs,
) -> List[Dict[str, Union[int, torch.Tensor, List[int]]]]:
    """Loads a data file into a list of `InputFeatures`
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = processor.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = processor.num_special_tokens_to_add() + (1 if sep_token_extra else 0)
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            token_type_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = processor.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
            token_type_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if "token_type_ids" not in processor.model_input_names:
            token_type_ids = None

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"sentence: {' '.join(example.words)}")
            logger.info(f"tokens: {' '.join(tokens)}")
            logger.info(f"input_ids: {input_ids}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"token_type_ids: {token_type_ids}")
            logger.info(f"label_ids: {label_ids}")

        features.append(
            {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label_ids": label_ids,
            }
        )
    return features
