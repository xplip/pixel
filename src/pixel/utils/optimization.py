import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_cosine_schedule_to_min_lr_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    max_lr: float,
    min_lr: float = 1e-5,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to a minimum learning rate, after a warmup period during which it increases linearly
    between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        max_lr (`float`):
            The maximum learning rate after warming up, right before decaying
        min_lr (`float`):
            The minimum learning rate at the end of training
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to the min
            value following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return (
            max(
                min_lr,
                min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )
            / max_lr  # Scale down by max_lr because LambdaLR multiplies back by max_lr
        )

    logger.info("***** Creating cosine scheduler to min_lr with warmup *****")
    logger.info(f"\t{num_warmup_steps = }")
    logger.info(f"\t{num_training_steps = }")
    logger.info(f"\t{max_lr = }")
    logger.info(f"\t{min_lr = }")
    logger.info(f"\t{num_cycles = }")
    logger.info(f"\t{last_epoch = }")

    return LambdaLR(optimizer, lr_lambda, last_epoch)
