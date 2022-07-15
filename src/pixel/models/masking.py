# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
#
# Further modified by Phillip Rust, for masking spans in rendered text inputs used with PIXEL
# --------------------------------------------------------'
import random
from typing import List, Optional, Union

import numpy as np


class SpanMaskingGenerator:
    """
    Generator class that yields span masks

    Args:
        num_patches (`int`):
            The total number of images patches
        num_masking_patches (`int`, defaults to 1):
            The number of patches to be masked out. Typically determined by the masking ratio
        max_span_length (`int`, defaults to 6):
            The maximum number of consecutive masked patches
        spacing (`Union[int, str]`, default to 0):
            The number of non-masked patches in between consecutive masked spans. Can either be an integer value,
            in which case the spacing is fixed, or can be set to "span" in which case the spacing is dynamic such
            that on both sides of a masked span of length N patches, there will be N non-masked patches. Note that
            larger spacing makes it harder to greedily sample masks satisfying these constraints which can slow down
            masking and also cause the algorithm to terminate with a smaller mask than specified. In case of the
            latter, PIXEL randomly masks additional patches until the specified masking ratio is reached.

            These are the recommended settings:
             - For masking ratio <= 0.4 use "span" mode.
             - For ratios between 0.4 and 0.7 set spacing to 1.
             - For higher, set spacing to 0
    """

    def __init__(
        self,
        num_patches: int,
        num_masking_patches: int = 1,
        max_span_length: int = 6,
        spacing: Union[int, str] = 0,
        cumulative_span_weights: Optional[List[float]] = None,
    ):

        self.num_patches = num_patches
        self.num_masking_patches = num_masking_patches

        self.max_span_length = max_span_length
        self.spacing = spacing
        assert spacing == "span" or isinstance(spacing, int)

        self.span_range = range(1, max_span_length + 1)
        self.cumulative_span_weights = cumulative_span_weights

    def _mask(self, mask, max_mask_patches):
        delta = 0
        # Lower number of attempts will speed up mask generation but might cause a lot fewer patches to be masked
        # than desired, particularly for high masking ratios
        for attempt in range(100):
            # Randomly select span length within specified range
            span = random.choices(self.span_range, cum_weights=self.cumulative_span_weights, k=1)[0]
            if span < self.num_patches:
                # This is only the case in the first iteration
                if self.num_text_patches is not None:
                    # Select a span where there is text
                    # This guarantees that we never generate a mask that only masks out padding
                    left = random.randint(0, max(0, self.num_text_patches - span))
                    self.num_text_patches = None
                else:
                    # Start at random horizontal index
                    left = random.randint(0, self.num_patches - span)

                space = span if self.spacing == "span" else self.spacing
                # Ensure no patches within <space> patches to the left are masked
                if space != 0:
                    num_masked_left = mask[max(0, left - space) : left].sum()
                    if num_masked_left > 0:
                        continue
                    # Ensure no patches within <space> patches to the right are masked
                    num_masked_right = mask[left + span : min(left + span + space, self.num_patches)].sum()
                    if num_masked_right > 0:
                        continue

                # Account for overlap
                num_masked_within = mask[left : left + span].sum()
                if 0 < span - num_masked_within <= max_mask_patches:
                    for j in range(left, left + span):
                        if mask[j] == 0:
                            mask[j] = 1
                            delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_text_patches: int):
        # Start with an empty mask
        mask = np.zeros(shape=self.num_patches, dtype=np.int)

        self.num_text_patches = num_text_patches

        # Greedily try to add mask patches until desired number of masked patches is reached
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_span_length)

            # We attempt to add a span to our mask up to 100 times
            delta = self._mask(mask, max_mask_patches)

            if delta == 0:
                # We terminate when no new span could be added to the mask after 100 attempts
                # This can happen before self.num_masking_patches is reached for high masking ratios with
                # strong constraints
                break
            else:
                mask_count += delta

        return mask
