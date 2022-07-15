import math
import numbers
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms import Compose, InterpolationMode, Lambda, Normalize, Resize, ToTensor
from torchvision.utils import _log_api_usage_once
from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.utils import logging

from .misc import patchify, unpatchify

logger = logging.get_logger(__name__)


class RandomErasing(torch.nn.Module):
    """
    Modified version from https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomErasing.html
    to work with non-square images

    Randomly selects a rectangle region in an torch Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=1, inplace=False):
        super().__init__()
        _log_api_usage_once(self)
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(
        img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        ratio = (ratio[0] * (img_h / img_w), ratio[1] * (img_h / img_w))
        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [
                    self.value,
                ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            return F.erase(img, x, y, h, w, v, self.inplace)
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}"
            f"(p={self.p}, "
            f"scale={self.scale}, "
            f"ratio={self.ratio}, "
            f"value={self.value}, "
            f"inplace={self.inplace})"
        )
        return s


def get_transforms(
    do_resize: bool = True,
    size: Union[int, Tuple[int, int]] = (16, 8464),
    do_squarify: bool = False,
    do_normalize: bool = False,
    do_random_erase: bool = False,
    image_mean: Optional[float] = IMAGENET_STANDARD_MEAN,
    image_std: Optional[float] = IMAGENET_STANDARD_STD,
    random_erase_params: Optional[Tuple[float, int, Tuple[float, float], Tuple[float, float]]] = (
        0.25,
        0,
        (0.01, 0.1),
        (0.3, 3.3),
    ),
    **kwargs,
) -> Compose:
    r"""
    Returns a composition of transformations we want to apply to our images
    We always convert to RGB and tensorize the images

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size: (`int` or `Tuple(int)`):
             Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then the input will be resized to (size, size). Only has an effect if `do_resize` is
            set to `True`.
        do_squarify (`bool`, defaults to `False`):
            Whether to squarify images, e.g. from 16x8464 to 368x368. This is necessary for some models
        do_normalize (`bool`, defaults to `False`):
            Whether to apply normalization with image_mean and image_std
        do_random_erase (`bool`, defaults to `False`):
            Whether to apply random erase data augmentation with random_erase_params
        image_mean (`float`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean value used for normalization. Defaults to ImageNet mean
        image_std (`float`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation value used for normalization. Defaults to ImageNet std
        random_erase_params (`Tuple[float, int, Tuple[float, float], Tuple[float, float]]`, *optional*,
                             defaults to `(0.25, 0, (0.01, 0.1), (0.3, 3.3))`):
            Parameter tuple to be used when applying random erase data augmentation. Is a tuple of
            (random_erase_probability, random_erase_value, random_erase_scale, random_erase_ratio).
            More information here: https://pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html

    Returns:
        A composition of transformations of type [~`Compose`]
    """

    # Convert to RGB
    transforms = [Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img)]

    # Optionally, resize to specified size
    if do_resize and size:
        transforms.append(Resize(size=size, interpolation=InterpolationMode.BICUBIC))

    # Tensorize image
    transforms.append(ToTensor())

    # Optionally, turn into square image by patchifying and unpatchifying
    if do_squarify:
        transforms.extend(
            [
                Lambda(lambda img: patchify(img)),
                Lambda(lambda img: unpatchify(img)),
            ]
        )

    # Optionally, apply random erase data augmentation
    if do_random_erase:
        random_erase_p, random_erase_value, random_erase_scale, random_erase_ratio = random_erase_params
        logger.info(
            f"Applying random erase transformation with p={random_erase_p}, value={random_erase_value} "
            f"scale={random_erase_scale}, ratio={random_erase_ratio}"
        )
        transforms.append(
            RandomErasing(
                p=random_erase_p,
                value=random_erase_value,
                scale=random_erase_scale,
                ratio=random_erase_ratio,
            )
        )

    # Optionally, apply normalization
    if do_normalize:
        logger.info(f"Applying normalization with mean={image_mean}, std={image_std}")
        transforms.append(Normalize(mean=image_mean, std=image_std))

    return Compose(transforms)
