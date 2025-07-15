import argparse
import os

import numpy as np
import wandb
from torch.utils.data.dataset import Dataset
from transformers.utils import logging

from .misc import patchify, unpatchify
from .training import Modality

logger = logging.get_logger(__name__)


def log_sequence_classification_predictions(
    training_args: argparse.Namespace,
    features: Dataset,
    examples: Dataset,
    predictions: np.ndarray,
    sentence1_key: str,
    sentence2_key: str,
    modality: Modality,
    prefix: str = "eval",
):
    # Initialize wandb if not already done
    if not training_args.do_train:
        wandb.init(reinit=False)

    data = []
    out_file = os.path.join(training_args.output_dir, f"{prefix}_outputs.csv")
    with open(out_file, "w", encoding="utf-8") as f:
        if sentence2_key:
            f.write(f"sentence1\tsentence2\tlabel\tprediction\tlength\n")
        else:
            f.write(f"sentence\tlabel\tprediction\tlength\n")
        for feature, example, prediction in zip(features, examples, predictions):
            sentence1 = example[sentence1_key]
            if sentence2_key:
                sentence2 = example[sentence2_key]
            else:
                sentence2 = ""

            if modality == Modality.IMAGE:
                processed_input = wandb.Image(unpatchify(patchify(feature["pixel_values"])))
            elif modality == Modality.TEXT:
                processed_input = feature["input_ids"]
            else:
                raise ValueError(f"Modality {modality} not supported.")

            label = example["label"]
            prediction = np.argmax(prediction)
            seq_len = len(sentence1) + len(sentence2)

            data.append([sentence1, sentence2, processed_input, label, prediction, seq_len])

            f.write(f"{sentence1}\t{sentence2}\t{label}\t{prediction}\t{seq_len}\n")

    logger.info(f"Saved predictions outputs to {out_file}")
    logger.info(f"Logging as table to wandb")

    preds_table = wandb.Table(
        columns=["sentence1", "sentence2", "processed_input", "label", "prediction", "length"], data=data
    )
    wandb.log({f"{prefix}_outputs": preds_table})
