from dataclasses import field, dataclass
from enum import Enum, auto
from typing import Dict, Optional

import torch
import wandb
from transformers import TrainingArguments

from .misc import format_img, format_mask, mark_answer


class Modality(Enum):
    IMAGE = auto()
    TEXT = auto()


@dataclass
class PIXELTrainingArguments(TrainingArguments):
    """
    Custom training arguments that include parameters for early stopping and prediction logging
    """

    early_stopping: Optional[bool] = field(default=True, metadata={"help": "Whether to train with early stopping."})
    early_stopping_patience: Optional[int] = field(
        default=5,
        metadata={
            "help": "Number of evaluation steps without increase in validation performance "
            "until training is terminated if training with early stopping."
        },
    )
    log_predictions: Optional[bool] = field(
        default=False, metadata={"help": "Whether to log predictions to file and wandb."}
    )


def debug_log_inputs(inputs: Dict[str, torch.Tensor]):
    """
    Logs inputs as square images to wandb
    Only works when training with Modality.IMAGE
    """

    wandb.init(reinit=False)

    images = [wandb.Image(format_img(im)) for im in inputs["pixel_values"]]
    attention_masks = [wandb.Image(format_mask(am)) for am in inputs["attention_mask"]]
    seq_length = len(inputs["attention_mask"][0])
    wandb.log(
        {
            "images": images,
            "attention_masks": attention_masks,
        }
    )

    if "patch_mask" in inputs:
        patch_masks = [wandb.Image(format_mask(pm)) for pm in inputs["patch_mask"]]
        wandb.log({"patch_masks": patch_masks})

    if "start_positions" in inputs and "end_positions" in inputs:
        marked_answers = [
            wandb.Image(format_mask(mark_answer(s, e, seq_length)))
            for s, e in zip(inputs["start_positions"], inputs["end_positions"])
        ]
        wandb.log({"answer_spans": marked_answers})
