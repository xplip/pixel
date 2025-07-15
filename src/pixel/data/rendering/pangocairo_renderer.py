import glob
import logging
import math
import os
import string
from typing import Any, Dict, List, Optional, Tuple, Union

import cairo
import gi
import manimpango
import numpy as np
from fontTools import ttLib

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo

from ...utils.defaults import *
from .rendering_utils import Encoding, TextRenderingMixin

logger = logging.getLogger(__name__)

SUPPORTED_INPUT_TYPES = [str, Tuple[str, str], List[str]]


class PangoCairoTextRenderer(TextRenderingMixin):
    """
    Constructs a text renderer using Pango and Cairo as rendering backend.
    This feature extractor inherits from [`TextRenderingMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        font_file (`str`):
            The font file (typically a file with a .ttf or .otf extension) that is loaded to render text
        font_size (`int`, defaults to 8):
            The font size that is used when rendering text
        font_color (`str`, defaults to "black"):
            The font color that is used when rendering text
        background_color (`str`, defaults to "white"):
            The background color of the image
        rgb (`bool`, defaults to False):
            Whether or not to render images in RGB. RGB rendering can be useful when working with emoji but it makes
            rendering a bit slower, so it is recommended to turn on RGB rendering only when there is need for it
        dpi (`int`, defaults to 120):
            The dpi (dots per inch) count that determines the resolution of the rendered images
        pad_size (`int`, defaults to 3):
            The amount of padding that is applied. Note: Currently, dynamic padding is not supported so this argument
            does not do anything
        pixels_per_patch (`int`, defaults to 16):
            The number of pixels, both horizontally and vertically, of each patch in the rendered image
        max_seq_length (`int`, defaults to 529):
            The maximum number of patches which, when multiplied with pixels_per_patch, determines the width of each
            rendered image
        fallback_fonts_dir (`str`, *optional*, defaults to None):
            Path to a directory containing font files (.ttf or .otf) which will be registered as fallback fonts. This
            can be useful when working with datasets with a large Unicode range

    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        font_file: str,
        font_size: int = DEFAULT_FONT_SIZE,
        font_color: str = "black",
        background_color: str = "white",
        rgb: bool = False,
        dpi: int = 120,
        pad_size: int = DEFAULT_PAD_SIZE,
        pixels_per_patch: int = DEFAULT_PPB,
        max_seq_length: int = MAX_SEQ_LENGTH,
        fallback_fonts_dir: Optional[str] = None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.font_file = font_file
        self.font_size = font_size
        self.font_color = font_color
        self.background_color = background_color
        self.rgb = rgb

        self.pixels_per_patch = pixels_per_patch
        self.max_seq_length = max_seq_length
        self.pad_size = pad_size
        self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = (pad_size, pad_size, pad_size, pad_size)

        self.dpi = dpi

        self.font = None
        self.fonts_list = None
        self.fallback_fonts_dir = fallback_fonts_dir
        self.load_font()

        self.PANGO_SCALE = 1024

    @property
    def max_pixels_len(self):
        return self.max_seq_length * self.pixels_per_patch

    def px2patch_ceil(self, px: int):
        return math.ceil(px / self.pixels_per_patch)

    def px2patch_floor(self, px: int):
        return math.floor(px / self.pixels_per_patch)

    def patch2px(self, patch: int):
        return patch * self.pixels_per_patch

    @staticmethod
    def is_rtl(text: str) -> bool:
        """
        Returns whether a piece of text is written in a right-to-left (RTL) script based on a majority vote of the
        first, middle, and last characters in the text after removing whitespace, punctuation, and numbers

        Returns:
            Whether the piece of text is RTL, type `bool`
        """
        text = text.translate(str.maketrans("", "", string.whitespace))
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.translate(str.maketrans("", "", string.digits))

        if len(text) == 0:
            return False

        vote = 0
        for char in [text[0], text[-1], text[len(text) // 2]]:
            if Pango.unichar_direction(char) == Pango.Direction.RTL:
                vote += 1

        is_rtl = vote >= 2
        # if not is_rtl:
        #    print(sys._getframe().f_back.f_code.co_name)
        #    print(f"{text[0] = }, {text[-1] = }, {text[len(text)//2] = }")
        return is_rtl

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns the state dict of the renderer without the loaded font to make it pickleable

        Returns:
            The state dict of type `Dict[str, Any]`
        """

        return {
            "font_file": self.font_file,
            "font_size": self.font_size,
            "font_color": self.font_color,
            "background_color": self.background_color,
            "rgb": self.rgb,
            "dpi": self.dpi,
            "pad_size": self.pad_size,
            "pixels_per_patch": self.pixels_per_patch,
            "max_seq_length": self.max_seq_length,
            "fonts_list": self.fonts_list,
        }

    def __setstate__(self, state_dict: Dict[str, Any]) -> None:
        """
        Sets the state dict of the renderer, e.g. from a pickle

        Args:
            state_dict (`Dict[str, Any]`):
                The state dictionary of a `PangoCairoTextRenderer`, containing all necessary and optional fields to
                initialize a `PangoCairoTextRenderer`
        """

        self.__init__(**state_dict)

    def _get_offset_to_next_patch(self, x: int) -> int:
        """
        Get the horizontal position (offset) where the next patch begins, based on how many pixels a patch contains
        and the maximum width

        Args:
            x (`int`):
                The horizontal position from where the next patch is to be found

        Returns:
            The starting position of the next patch (offset) of  type `int`
        """

        return min(
            math.ceil(x / self.pixels_per_patch) * self.pixels_per_patch,
            self.max_pixels_len - self.pixels_per_patch,
        )

    def _get_offset_to_previous_patch(self, x: int) -> int:
        """
        Get the horizontal position (offset) where the previous patch begins, based on how many pixels a patch contains
        and the maximum width

        Args:
            x (`int`):
                The horizontal position from where the next patch is to be found

        Returns:
            The starting position of the next patch (offset) of  type `int`
        """

        return math.floor(x / self.pixels_per_patch) * self.pixels_per_patch

    def get_empty_surface(self) -> Tuple[cairo.ImageSurface, cairo.Context, List[int]]:
        """
        Create and return a tuple containing
        (1) an empty surface that we will later render the text to,
        (2) a context object used to draw on the surface, and
        (3) an empty list in which we keep track of where to insert black separator patches

        Returns:
            A tuple of type (`~cairo.ImageSurface`, `~cairo.Context`, `List[int]`) containing the blank surface,
            the context object, an the empty list for keeping track of black separator patches, respectively
        """

        cairo_format = cairo.FORMAT_RGB24 if self.rgb else cairo.FORMAT_A8
        surface = cairo.ImageSurface(cairo_format, self.max_pixels_len, self.pixels_per_patch)
        context = cairo.Context(surface)
        if self.rgb:
            context.set_source_rgb(1.0, 1.0, 1.0)
            context.rectangle(0, 0, self.max_pixels_len, self.pixels_per_patch)
            context.fill()
            context.set_source_rgb(0.0, 0.0, 0.0)
        sep_patches = []
        return surface, context, sep_patches

    def get_cluster_idx_and_logical_widths(self, layout_iter: Pango.LayoutIter):
        """
        Returns the logical extents (first pixel in text direction) at the grapheme cluster level for a given index

        Args:
            layout_iter (`Pango.LayoutIter`):
                An object used to iterate over a pango layout (here, cluster-by-cluster).
        """
        logical_extents = layout_iter.get_cluster_extents()[1]
        x_offset = logical_extents.x / self.PANGO_SCALE
        idx = layout_iter.get_index()
        return idx, x_offset

    def get_char_idx_and_logical_widths(self, layout_iter: Pango.LayoutIter):
        """
        Returns the logical extents (first pixel in text direction) at the character level for a given index

        Args:
            layout_iter (`Pango.LayoutIter`):
                An object used to iterate over a pango layout (here, character-by-character).
        """
        logical_extents = layout_iter.get_char_extents()
        x_offset = logical_extents.x / self.PANGO_SCALE
        idx = layout_iter.get_index()
        return idx, x_offset

    def get_text_offset_mapping(
        self, layout: Pango.Layout, offset: int, text_width: int, text_shift: int = 0, rtl: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Returns an offset mapping, i.e. a list that keeps track of where in the rendered image each character of
        the input text is located. It has the form [(start_character_index, end_character_index)] with an entry for
        every image patch.

        Args:
            layout (`Pango.Layout`):
                The layout containing the rendered text.
            offset (`int`):
                The offset in pixels of the first character of the text from the beginning of the first patch.
            text_width (`int`):
                The logical width of the rendered text in pixels.
            text_shift (`int`, *optional*, defaults to 0):
                The number of pixels that a text is shifted to the right on the layout, i.e. the starting position
                as pixel offset of the first image patch corresponding to this text. This value is typically set when
                obtaining the offset_mapping for the second text in a rendered text pair.
            rtl (`bool`, *optional*, defaults to False):
                Indicates whether the text is rendered right-to-left (RTL), in which case the offset mapping needs to
                account for the fact that the actual beginning of the text is on the right.
        """
        # Find starting positions for each character in the text
        layout_iter = layout.get_iter()
        # Get offset for first character
        idx, x_offset = self.get_char_idx_and_logical_widths(layout_iter)
        character_positions = [x_offset + offset]
        # Loop through remaining characters
        while layout_iter.next_char():
            idx, x_offset = self.get_char_idx_and_logical_widths(layout_iter)
            character_positions.append(x_offset + offset)

        # Find starting positions for each cluster in the text. A cluster may consist of multiple characters rendered
        # as one glyph
        layout_iter = layout.get_iter()
        # Get offset for first cluster
        idx, x_offset = self.get_cluster_idx_and_logical_widths(layout_iter)
        cluster_positions = [x_offset + offset]
        # Loop through remaining clusters
        while layout_iter.next_cluster():
            idx, x_offset = self.get_cluster_idx_and_logical_widths(layout_iter)
            cluster_positions.append(x_offset + offset)

        # In case clusters exist, the length of the cluster list will be shorter than the length of the character list.
        # However, the offset mapping maps between clusters in the rendered image and characters in the written text,
        # so we need to assign a starting position to each character in the cluster position list. We do this by
        # assigning the starting position of a cluster to each character in that cluster.
        if len(character_positions) != len(cluster_positions):
            buffer = []
            cluster_idx = 0
            for idx in range(len(character_positions)):
                if cluster_idx == len(cluster_positions) or character_positions[idx] != cluster_positions[cluster_idx]:
                    buffer.append(cluster_positions[cluster_idx - 1])
                else:
                    buffer.append(character_positions[idx])
                    cluster_idx += 1

            buffered_cluster_positions = buffer
        else:
            buffered_cluster_positions = character_positions

        # Retrieve the rendered text from the layout. This is necessary for RTL scripts
        text = layout.get_text()

        # This means we add a full blank patch
        if self._get_offset_to_next_patch(text_width) - text_width < offset - self._get_offset_to_previous_patch(
            offset
        ):
            is_blank_patch_inserted = True
        else:
            is_blank_patch_inserted = False

        buffered_cluster_positions.append(self._get_offset_to_next_patch(text_width + offset))

        offset_mapping = []
        patch_start = 0
        cleared = 0
        for k, v in enumerate(buffered_cluster_positions):
            if v - text_shift >= self.pixels_per_patch * (len(offset_mapping) + 1):
                if v - text_shift == self.pixels_per_patch * (len(offset_mapping) + 1):
                    patch_end = k
                else:
                    patch_end = k - 1
                offset_mapping.append(
                    (
                        (len(text) - patch_start) if rtl else patch_start,
                        (len(text) - patch_end) if rtl else patch_end,
                    )
                )

                patch_start = patch_end
                cleared += 1

        # The `cleared` variable counts how many times we have added a character span to the offset mapping, i.e.,
        # cleared the cluster buffer. If at the end of processing the buffered_cluster_positions we still have clusters
        # in the buffer, we add the remainder to the offset mapping
        if cleared < self.px2patch_ceil(text_width + offset - text_shift):
            if rtl:
                offset_mapping.append((len(text) - patch_start, 0))
            else:
                offset_mapping.append((patch_start, len(buffered_cluster_positions)))

        # We add padding between the end of the rendered sequence and the final black separator patch. If this padding
        # happens to be a full patch, this means that we need to merge the penultimate and last patches in the offset
        # mapping and add a buffer to the offset mapping
        if is_blank_patch_inserted:
            offset_mapping[-2] = (
                offset_mapping[-2][0],
                offset_mapping[-1][1],
            )
            offset_mapping[-1] = (-1, -1)

        # print(f"{len(offset_mapping) = }")

        return offset_mapping

    def pad_or_truncate_offset_mapping(self, offset_mapping: List[Tuple[int, int]]):
        if len(offset_mapping) >= self.max_seq_length:
            offset_mapping = offset_mapping[: self.max_seq_length - 1] + [(0, 0)]
        if len(offset_mapping) < self.max_seq_length:
            offset_mapping += (self.max_seq_length - len(offset_mapping)) * [(0, 0)]
        return offset_mapping

    def _render_single_word(
        self, word: str, offset: int, context: cairo.Context, is_last: bool = False
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Renders a single token to a surface with a horizontal offset, i.e. the rendered
        word begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        word vertically on the surface

        Args:
            word (`str`):
                The word to be rendered
            offset (`int`):
                The horizontal starting position of the rendered token on the surface (in pixels)
            context (`~cairo.Context`):
                The context object used to render text to the surface
            is_last (`bool`, *optional*, defaults to False):
                Boolean variable that indicates whether we are rendering the last token of the sequence, in which
                case additional padding is added to the final offset so that the black separator patch is guaranteed
                to be spaced at least this padded amount from the last token

        Returns:
            A tuple containing the context of type `~cairo.Context` that we used to draw on the surface,
            the layout of type `~Pango.Layout` containing the rendered sentence, and the offset to where the next patch
            begins, type `int`
        """

        layout = PangoCairo.create_layout(context)
        layout.set_font_description(self.font)

        layout.set_text(word, -1)

        if layout.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout.get_unknown_glyphs_count()} unknown glyphs in word: {word}. Consider "
                f"double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width, height = layout.get_pixel_size()

        position = (offset, self.pixels_per_patch / 2.0 - height / 2.0 - 2)
        context.move_to(*position)

        PangoCairo.show_layout(context, layout)

        if is_last:
            offset += 2
        offset = self._get_offset_to_next_patch(offset + width)

        return context, layout, offset

    def _render_single_sentence(
        self, sentence: str, offset: int, context, max_length: Optional[int] = None, rtl: bool = False
    ) -> Tuple[cairo.Context, Tuple[Pango.Layout, Pango.Layout], int]:
        """
        Renders a single sentence to a surface with a horizontal offset, i.e. the rendered
        sentence begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        text vertically on the surface

        Args:
            sentence (`str`):
                The sentence to be rendered
            offset (`int`):
                The horizontal starting position of the rendered sentence on the surface (in pixels)
            context (`~cairo.Context`):
                The context object used to render text to the surface
            max_length (`int`, *optional*, defaults to None):
                Maximum number of patches that the rendered sentence may fill on the surface. If set, anything longer
                than this number of patches will be truncated.

        Returns:
            A tuple containing the context of type `~cairo.Context` that we used to draw on the surface,
            the layout of type `~Pango.Layout` containing the rendered sentence, and the width of the rendered
            sentence in pixels, type `int`
        """
        pango_context = PangoCairo.create_context(context)
        pango_context.set_font_description(self.font)
        layout = Pango.Layout(pango_context)

        if rtl:
            layout.set_auto_dir(False)
            pango_context.set_base_dir(Pango.Direction.RTL)
            layout.set_alignment(Pango.Alignment.RIGHT)
        layout.set_text(sentence, -1)

        if layout.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout.get_unknown_glyphs_count()} unknown glyphs in sentence: {sentence}. Consider"
                f" double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width, height = layout.get_pixel_size()
        full_width = width
        full_layout = layout
        truncated_layout = layout

        if max_length is not None:
            if self.px2patch_ceil(offset + width) > max_length:
                truncated_layout = Pango.Layout(pango_context)

                # print(
                #     f"Truncating {sentence} ({self.px2patch_ceil(offset + width)} patches) to fit {max_length = }."
                # )

                # Run binary search to find truncation point
                lo = 0
                hi = len(sentence)
                while lo <= hi:
                    mid = (lo + hi) // 2
                    new_sentence = sentence[:mid]
                    truncated_layout.set_text(new_sentence, -1)
                    width, height = truncated_layout.get_pixel_size()
                    if self.px2patch_ceil(offset + width) < max_length:
                        lo = mid + 1
                    elif self.px2patch_ceil(offset + width) > max_length:
                        hi = mid - 1
                    else:
                        break
                # print(f"New sentence = {new_sentence}, width = {self.px2patch_ceil(offset + width)} patches")

        position = (offset, self.pixels_per_patch / 2.0 - height / 2.0 - 2)
        context.move_to(*position)

        PangoCairo.show_layout(context, truncated_layout)

        return context, (full_layout, truncated_layout), full_width

    def _render_words_to_surface(self, words: List[str], **kwargs) -> Encoding:
        """
        Renders a list of words to a surface and keeps track of
        (a) how many patches in the rendered surface contain text, i.e. are neither blank nor black separator patches
        and (b) the patch index, starting with 0, where each word begins on the rendered surface

        Args:
            words (`List[str]`):
                The list of words to be rendered

        Returns:
            An Encoding of type `Encoding` containing the rendered words and metadata
        """

        # Keep track of the patch index at which each new token begins
        word_start_indices = [0]

        # Start with blank surface
        surface, context, sep_patches = self.get_empty_surface()

        # Pad left with 2px
        offset = 2
        skip_last = False

        # Render each token to the start of the next patch but at least a whitespace width apart
        for word in words[:-1]:
            context, layout, offset = self._render_single_word(f"{word} ", offset, context)
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

            if offset == self.max_pixels_len - self.pixels_per_patch:
                skip_last = True
                break

        # Last token is rendered without whitespace
        if not skip_last:
            context, layout, offset = self._render_single_word(words[-1], offset, context, is_last=True)
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

        # Draw black rectangle on surface as separator patch
        sep_patches.append(offset)

        num_text_patches = math.ceil(offset / self.pixels_per_patch)

        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface, sep_patches=sep_patches),
            sep_patches=sep_patches,
            num_text_patches=num_text_patches,
            word_starts=word_start_indices,
        )
        return encoding

    def _render_text_pair_to_surface_ltr(
        self,
        text_pair: Tuple[str, str],
        return_overflowing_patches: bool = False,
        return_offset_mapping: bool = False,
        stride: int = 0,
        text_a_max_length: Optional[int] = None,
        **kwargs,
    ) -> Encoding:
        """
        Renders a text pair left-to-right (LTR).

        Args:
            text_pair (`Tuple[str, str]`):
                The text pair to be rendered
            return_overflowing_patches (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing patch sequences.
            return_offset_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each patch.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing patches returned when
                `return_overflowing_patches=True` will contain some patches from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping patches.
            text_a_max_length (`int`, *optional*):
                Maximum length (in image patches) of the first text in the text pair.
            rtl (`bool`, *optional*, defaults to `False`):
                Whether text is written in right-to-left (RTL) script. Note: If set to False, the renderer will detect
                the text direction automatically, so the text can still be rendered RTL, depending on its content

        Returns:
            An Encoding of type `Encoding` containing the rendered text pair and metadata
        """

        text_a, text_b = text_pair

        surface, context, sep_patches = self.get_empty_surface()
        sequence_ids = [None]

        offset = 2

        # Render first sentence and draw on surface
        context, (full_layout, truncated_layout), text_a_width = self._render_single_sentence(
            text_a, offset, context, max_length=text_a_max_length
        )

        if return_offset_mapping:
            # cluster_position_dict = self.get_cluster_position_dict(layout, offset, text_a_width + 2)
            text_a_offset_mapping = self.get_text_offset_mapping(
                layout=truncated_layout, offset=offset, text_width=(text_a_width + 2), text_shift=0
            )
        # Offset is left padding + rendered width of text_a + 2 (padding)
        offset = self._get_offset_to_next_patch(offset + text_a_width + 2)

        # Mark patch starting at offset as black separator patch
        sep_patches.append(offset)

        # # Add a 0 to sequence_ids for each patch in text_a and None for the separator patch
        sequence_ids.extend([0] * self.px2patch_floor(offset) + [None])

        # Reserve space for the black separator patch + 2 (padding)
        offset = offset + self.pixels_per_patch + 2

        # Render second sentence and draw on surface
        context, (full_layout, truncated_layout), text_b_width = self._render_single_sentence(text_b, offset, context)

        if return_offset_mapping:
            # cluster_position_dict = self.get_cluster_position_dict()
            text_b_offset_mapping = self.get_text_offset_mapping(
                layout=full_layout, offset=offset, text_width=(text_b_width + 2), text_shift=offset - 2
            )
            offset_mapping = text_a_offset_mapping + [(0, 0)] + text_b_offset_mapping
            offset_mapping = self.pad_or_truncate_offset_mapping(offset_mapping)
        else:
            offset_mapping = None

        # Offset is left padding + rendered width of text_b + 2 (padding)
        eos_patch_offset = self._get_offset_to_next_patch(offset + text_b_width + 2)

        # Mark patch starting at offset as black separator patch
        sep_patches.append(eos_patch_offset)

        # # Add a 1 to sequence_ids for each patch in text_b and None for the separator patch
        b_patches = self.px2patch_floor(eos_patch_offset - offset)
        sequence_ids.extend([1] * b_patches + [None])

        image = self.get_image_from_surface(surface, sep_patches=sep_patches)

        num_text_patches = self.px2patch_floor(eos_patch_offset)

        encoding = Encoding(
            pixel_values=image,
            sep_patches=sep_patches,
            num_text_patches=num_text_patches,
            offset_mapping=offset_mapping,
            overflowing_patches=None,
        )

        # Calculate how many patches / pixels of the overflow sequence are already filled by
        # text_a, the sep patch, padding, and the stride
        num_patches_filled = self.px2patch_floor(self._get_offset_to_next_patch(2 + text_a_width + 2)) + 1 + stride
        num_pixels_filled = self.patch2px(num_patches_filled)

        if return_overflowing_patches:

            if not return_offset_mapping:
                raise ValueError(
                    "The argument return_overflowing_patches=True requires that return_offset_mapping"
                    " is also set to True"
                )
            offset_mapping = text_b_offset_mapping

            pixel_overflow = (offset + text_b_width) - (self.max_pixels_len - self.pixels_per_patch)
            patch_overflow = self.px2patch_ceil(pixel_overflow)

            if pixel_overflow > 0:
                # Determine how many additional sequences we need to generate
                max_num_additional_sequences = math.ceil(
                    pixel_overflow
                    / (
                        self.max_pixels_len
                        - self.pixels_per_patch
                        - (
                            self._get_offset_to_next_patch(2 + text_a_width + 2)
                            + self.pixels_per_patch
                            + stride * self.pixels_per_patch
                        )
                    )
                )

                overflow_encodings = []
                for i in range(max_num_additional_sequences):

                    # By shifting the continuation in each overflowing sequence to the left by some small amount
                    # it can happen that there is actually less overflow than initially calculated, potentially even
                    # requiring fewer additional sequences.
                    if pixel_overflow <= 0:
                        break

                    # Start a new surface for the overflow sequence
                    o_surface, o_context, o_sep_patches = self.get_empty_surface()

                    text_remainder = text_b[offset_mapping[-patch_overflow - stride][0] :]

                    continuation_starting_point = (
                        self._get_offset_to_next_patch(2 + text_a_width + 2) + self.pixels_per_patch + 2
                    )

                    # Render only the continuation (i.e., the part that is new in this overflow sequence and the stride)
                    # onto the surface for now. The text_a content gets copied over later
                    o_context, (o_full_layout, o_truncated_layout), o_text_width = self._render_single_sentence(
                        text_remainder, continuation_starting_point, o_context
                    )

                    # Remember where to put SEP patch
                    o_eos_offset = self._get_offset_to_next_patch(continuation_starting_point + o_text_width + 2)
                    o_sep_patches.append(o_eos_offset)

                    # Determine the real (i.e., excluding additional overflow) rendered width of the continuation
                    # to find its starting and end points in the offset_mapping
                    rendered_width_real = min(
                        2 + o_text_width,
                        self.max_pixels_len - self.pixels_per_patch - continuation_starting_point,
                    )

                    continuation_start_letter = -patch_overflow - stride
                    continuation_end_letter = -patch_overflow + self.px2patch_floor(rendered_width_real)
                    if continuation_end_letter > -1:
                        continuation_end_letter = None

                    o_offset_mapping = offset_mapping[continuation_start_letter:continuation_end_letter]

                    # Calculate overflow again for (potential) subsequent overflow sequence
                    pixel_overflow = (continuation_starting_point + o_text_width) - (
                        self.max_pixels_len - self.pixels_per_patch
                    )
                    patch_overflow = self.px2patch_ceil(pixel_overflow)

                    num_text_patches = self.px2patch_floor(o_eos_offset)

                    # Take original image or previous overflow sequence image to copy data from
                    previous_image = image
                    image = self.get_image_from_surface(o_surface, sep_patches=[sep_patches[0]] + o_sep_patches)

                    # Copy [text_a, sep patch, padding] content from previous image
                    image[:, : num_pixels_filled - self.patch2px(stride)] = previous_image[
                        :, : num_pixels_filled - self.patch2px(stride)
                    ]

                    o_offset_mapping = text_a_offset_mapping + [(0, 0)] + o_offset_mapping
                    o_offset_mapping = self.pad_or_truncate_offset_mapping(o_offset_mapping)

                    overflow_encodings.append(
                        Encoding(
                            pixel_values=image,
                            sep_patches=sep_patches + o_sep_patches,
                            num_text_patches=num_text_patches,
                            offset_mapping=o_offset_mapping,
                        )
                    )
                encoding.overflowing_patches = overflow_encodings

        return encoding

    def _render_text_pair_to_surface_rtl(
        self,
        text_pair: Tuple[str, str],
        return_overflowing_patches: bool = False,
        return_offset_mapping: bool = False,
        stride: int = 0,
        text_a_max_length: Optional[int] = None,
        **kwargs,
    ) -> Encoding:
        """
        Renders a text pair right-to-left (RTL).

        Args:
            text_pair (`Tuple[str, str]`):
                The text pair to be rendered
            return_overflowing_patches (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing patch sequences.
            return_offset_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each patch.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing patches returned when
                `return_overflowing_patches=True` will contain some patches from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping patches.
            text_a_max_length (`int`, *optional*):
                Maximum length (in image patches) of the first text in the text pair.
            rtl (`bool`, *optional*, defaults to `False`):
                Whether text is written in right-to-left (RTL) script. Note: If set to False, the renderer will detect
                the text direction automatically, so the text can still be rendered RTL, depending on its content

        Returns:
            An Encoding of type `Encoding` containing the rendered text pair and metadata
        """

        text_a, text_b = text_pair

        surface, context, sep_patches = self.get_empty_surface()
        sequence_ids = [None]

        offset = 2

        # Render first sentence and draw on surface
        context, (full_layout, truncated_layout), text_a_width = self._render_single_sentence(
            text_a, offset, context, max_length=text_a_max_length, rtl=True
        )

        if return_offset_mapping:
            # cluster_position_dict = self.get_cluster_position_dict(layout, offset, text_a_width + 2)
            text_a_offset_mapping = self.get_text_offset_mapping(
                layout=truncated_layout, offset=offset, text_width=(text_a_width + 2), text_shift=0, rtl=True
            )
        # Offset is left padding + rendered width of text_a + 2 (padding)
        offset = self._get_offset_to_next_patch(offset + text_a_width + 2)

        # Mark patch starting at offset as black separator patch
        sep_patches.append(offset)

        # # Add a 0 to sequence_ids for each patch in text_a and None for the separator patch
        sequence_ids.extend([0] * self.px2patch_floor(offset) + [None])

        # Reserve space for the black separator patch + 2 (padding)
        offset = offset + self.pixels_per_patch + 2

        # Render second sentence and draw on surface
        context, (full_layout, truncated_layout), text_b_width = self._render_single_sentence(
            text_b,
            offset,
            context,
            max_length=self.px2patch_floor(self.max_pixels_len - self.pixels_per_patch),
            rtl=True,
        )

        if return_offset_mapping:
            # Get offset mapping both for the full and the truncated layout. The truncated one is used in any case and
            # full one is used in case there is overflow
            full_text_b_offset_mapping = self.get_text_offset_mapping(
                layout=full_layout, offset=offset, text_width=(text_b_width + 2), text_shift=offset - 2, rtl=True
            )
            truncated_text_b_offset_mapping = self.get_text_offset_mapping(
                layout=truncated_layout, offset=offset, text_width=(text_b_width + 2), text_shift=offset - 2, rtl=True
            )

            offset_mapping = text_a_offset_mapping + [(0, 0)] + truncated_text_b_offset_mapping
            offset_mapping = self.pad_or_truncate_offset_mapping(offset_mapping)
        else:
            offset_mapping = None

        # Offset is left padding + rendered width of text_b + 2 (padding)
        eos_patch_offset = self._get_offset_to_next_patch(offset + text_b_width + 2)

        # Mark patch starting at offset as black separator patch
        sep_patches.append(eos_patch_offset)

        # # Add a 1 to sequence_ids for each patch in text_b and None for the separator patch
        b_patches = self.px2patch_floor(eos_patch_offset - offset)
        sequence_ids.extend([1] * b_patches + [None])

        image = self.get_image_from_surface(surface, sep_patches=sep_patches)

        num_text_patches = self.px2patch_floor(eos_patch_offset)

        encoding = Encoding(
            pixel_values=image,
            sep_patches=sep_patches,
            num_text_patches=num_text_patches,
            offset_mapping=offset_mapping,
            overflowing_patches=None,
        )

        # Calculate how many patches / pixels of the overflow sequence are already filled by
        # text_a, the sep patch, padding, and the stride
        num_patches_filled = self.px2patch_floor(self._get_offset_to_next_patch(2 + text_a_width + 2)) + 1 + stride
        num_pixels_filled = self.patch2px(num_patches_filled)

        if return_overflowing_patches:

            if not return_offset_mapping:
                raise ValueError(
                    "The argument return_overflowing_patches=True requires that return_offset_mapping"
                    " is also set to True"
                )
            offset_mapping = full_text_b_offset_mapping

            pixel_overflow = (offset + text_b_width) - (self.max_pixels_len - self.pixels_per_patch)
            patch_overflow = self.px2patch_ceil(pixel_overflow)

            if pixel_overflow > 0:
                # Determine how many additional sequences we need to generate
                max_num_additional_sequences = math.ceil(
                    pixel_overflow
                    / (
                        self.max_pixels_len
                        - self.pixels_per_patch
                        - (
                            self._get_offset_to_next_patch(2 + text_a_width + 2)
                            + self.pixels_per_patch
                            + stride * self.pixels_per_patch
                        )
                    )
                )

                overflow_encodings = []
                for i in range(max_num_additional_sequences):

                    # By shifting the continuation in each overflowing sequence to the left by some small amount
                    # it can happen that there is actually less overflow than initially calculated, potentially even
                    # requiring fewer additional sequences.
                    if pixel_overflow <= 0:
                        break

                    # Start a new surface for the overflow sequence
                    o_surface, o_context, o_sep_patches = self.get_empty_surface()

                    text_remainder = text_b[offset_mapping[patch_overflow + stride][1] :]

                    continuation_starting_point = (
                        self._get_offset_to_next_patch(2 + text_a_width + 2) + self.pixels_per_patch + 2
                    )

                    # Render only the continuation (i.e., the part that is new in this overflow sequence and the stride)
                    # onto the surface for now. The text_a content gets copied over later
                    o_context, (o_full_layout, o_truncated_layout), o_text_width = self._render_single_sentence(
                        text_remainder,
                        continuation_starting_point,
                        o_context,
                        max_length=self.px2patch_floor(self.max_pixels_len - self.pixels_per_patch),
                        rtl=True,
                    )

                    # Remember where to put SEP patch
                    o_eos_offset = self._get_offset_to_next_patch(continuation_starting_point + o_text_width + 2)
                    o_sep_patches.append(o_eos_offset)

                    # Determine the real (i.e., excluding additional overflow) rendered width of the continuation
                    # to find its starting and end points in the offset_mapping
                    rendered_width_real = min(
                        2 + o_text_width,
                        self.max_pixels_len - self.pixels_per_patch - continuation_starting_point,
                    )

                    continuation_end_letter = patch_overflow + stride
                    continuation_start_letter = max(
                        0, continuation_end_letter - self.px2patch_floor(rendered_width_real)
                    )
                    o_offset_mapping = offset_mapping[continuation_start_letter : continuation_end_letter + 1]

                    # Re-calculate overflow
                    patch_overflow = continuation_start_letter
                    pixel_overflow = self.patch2px(patch_overflow)

                    num_text_patches = self.px2patch_floor(o_eos_offset)

                    # Take original image or previous overflow sequence image to copy data from
                    previous_image = image
                    image = self.get_image_from_surface(o_surface, sep_patches=[sep_patches[0]] + o_sep_patches)

                    # Copy [text_a, sep patch, padding] content from previous image
                    image[:, : num_pixels_filled - self.patch2px(stride)] = previous_image[
                        :, : num_pixels_filled - self.patch2px(stride)
                    ]

                    o_offset_mapping = text_a_offset_mapping + [(0, 0)] + o_offset_mapping
                    o_offset_mapping = self.pad_or_truncate_offset_mapping(o_offset_mapping)

                    overflow_encodings.append(
                        Encoding(
                            pixel_values=image,
                            sep_patches=sep_patches + o_sep_patches,
                            num_text_patches=num_text_patches,
                            offset_mapping=o_offset_mapping,
                        )
                    )
                encoding.overflowing_patches = overflow_encodings

        return encoding

    def _render_text_pair_to_surface(
        self,
        text_pair: Tuple[str, str],
        return_overflowing_patches: bool = False,
        return_offset_mapping: bool = False,
        stride: int = 0,
        text_a_max_length: Optional[int] = None,
        rtl: bool = False,
        **kwargs,
    ) -> Encoding:
        """
        Renders a pair of sentences or paragraphs to a surface and keeps track of
        how many patches in the rendered surface contain text, i.e. are neither blank nor black separator patches

        Args:
            text_pair (`Tuple[str, str]`):
                The text pair to be rendered
            return_overflowing_patches (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing patch sequences.
            return_offset_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each patch.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing patches returned when
                `return_overflowing_patches=True` will contain some patches from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping patches.
            text_a_max_length (`int`, *optional*):
                Maximum length (in image patches) of the first text in the text pair.
            rtl (`bool`, *optional*, defaults to `False`):
                Whether text is written in right-to-left (RTL) script. Note: If set to False, the renderer will detect
                the text direction automatically, so the text can still be rendered RTL, depending on its content

        Returns:
            An Encoding of type `Encoding` containing the rendered text pair and metadata
        """

        text_a, text_b = text_pair

        # Clean texts
        text_a = text_a.replace("\n", " ")
        text_b = text_b.replace("\n", " ")

        # Check whether text is written in a right-to-left script
        if rtl or (self.is_rtl(text_a) and self.is_rtl(text_b)):
            rendering_fn = self._render_text_pair_to_surface_rtl
        else:
            rendering_fn = self._render_text_pair_to_surface_ltr

        return rendering_fn(
            text_pair=(text_a, text_b),
            return_overflowing_patches=return_overflowing_patches,
            return_offset_mapping=return_offset_mapping,
            stride=stride,
            text_a_max_length=text_a_max_length,
            **kwargs,
        )

    def _render_text_to_surface(
        self,
        text: str,
        **kwargs,
    ) -> Encoding:
        """
        Renders a single piece of text, e.g. a sentence or paragraph, to a surface and keeps track of
        metadata, e.g. how many patches in the rendered surface contain text, i.e. are neither blank nor black separator
        patches
        Args:
            text (`str`):
                The piece of text to be rendered


        Returns:
            An Encoding of type `Encoding` containing the rendered text and metadata
        """

        # Clean text
        text = text.replace("\n", " ")

        surface, context, sep_patches = self.get_empty_surface()

        offset = 2

        # Render text
        context, (_, layout), text_width = self._render_single_sentence(text, offset, context)

        # Offset is left padding + rendered width of first sentence + 2 (padding)
        eos_patch_offset = self._get_offset_to_next_patch(2 + text_width + 2)
        sep_patches.append(eos_patch_offset)

        num_text_patches = self.px2patch_floor(eos_patch_offset)

        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface, sep_patches=sep_patches),
            sep_patches=sep_patches,
            num_text_patches=num_text_patches,
        )

        return encoding

    def get_image_from_surface(self, surface: cairo.ImageSurface, sep_patches: List[int]) -> np.ndarray:
        """
        Transforms a surface containing a rendered image into a numpy image and inserts black separator patches.

        Args:
            surface (`cairo.ImageSurface`):
                The cairo surface containing the rendered text
            sep_patches (`List[int]`):
                A list of offset values at which black separator patches will be inserted
        Returns:
            An image of type `np.ndarray` of size [self.pixels_per_patch, self.max_pixels_len]
        """

        # Get image data from surface
        data = surface.get_data()
        if self.rgb:
            data = np.frombuffer(data, dtype=np.uint8).reshape((self.pixels_per_patch, self.max_pixels_len, 4))
            data = data[:, :, :3]
            # Reverse channels BGR -> RGB
            image = data[:, :, ::-1]
            # Insert black separator patches
            for idx, sep_patch in enumerate(sep_patches):
                image[:, sep_patch : sep_patch + self.pixels_per_patch, :] = 0
        else:
            data = np.frombuffer(data, dtype=np.uint8).reshape((self.pixels_per_patch, self.max_pixels_len))
            image = np.invert(data)
            # Insert black separator patches
            for idx, sep_patch in enumerate(sep_patches):
                image[:, sep_patch : sep_patch + self.pixels_per_patch] = 0

        return image

    def __call__(
        self,
        text: Union[str, Tuple[str, str], List[str]],
        return_overflowing_patches: bool = False,
        return_offset_mapping: bool = False,
        stride: int = 0,
        rtl: bool = False,
        **kwargs,
    ) -> Encoding:
        """
        Render a piece of text to a surface, convert the surface into an image and return the image
        along with metadata (the number of patches containing text and, when rendering a list of words, the patch
        indices at which each word starts)

        Args:
            text (`str` or `Tuple[str, str]` or `List[str]`):
                The text to be rendered
            return_overflowing_patches (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing patch sequences.
            return_offset_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each patch.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_seq_length`, the overflowing patches returned when
                `return_overflowing_patches=True` will contain some patches from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping patches.

        Returns:
            An encoding of type `Encoding` containing the rendered image and metadata
        """

        if isinstance(text, list):
            rendering_fn = self._render_words_to_surface
        elif isinstance(text, tuple):
            rendering_fn = self._render_text_pair_to_surface
        elif isinstance(text, str):
            rendering_fn = self._render_text_to_surface
        else:
            raise TypeError(
                f"{self.__class__.__name__} does not support inputs of type {type(text)}. Supported types are "
                f"{SUPPORTED_INPUT_TYPES}"
            )

        encoding = rendering_fn(
            text,
            return_overflowing_patches=return_overflowing_patches,
            return_offset_mapping=return_offset_mapping,
            stride=stride,
            rtl=rtl,
            **kwargs,
        )

        return encoding

    def load_font(self) -> None:
        """
        Loads the font from specified font file with specified font size and color.
        """

        logger.info(f"Loading font from {self.font_file}")

        manimpango.register_font(self.font_file)
        if self.fallback_fonts_dir is not None:
            for fallback_font in glob.glob(os.path.join(self.fallback_fonts_dir, "*tf")):
                logger.info(f"Loading fallback font {fallback_font}")
                manimpango.register_font(fallback_font)
        self.fonts_list = manimpango.list_fonts()

        font_family_name = ttLib.TTFont(self.font_file)["name"].getDebugName(1)

        scaled_font_size = (self.dpi / 72) * self.font_size
        font_str = f"{font_family_name} {scaled_font_size}px"
        self.font = Pango.font_description_from_string(font_str)
