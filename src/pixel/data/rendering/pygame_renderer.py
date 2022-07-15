import logging
import math
import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

from ...utils.defaults import *
from .rendering_utils import TextRenderingMixin, Encoding

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame.freetype

logger = logging.getLogger(__name__)

SUPPORTED_INPUT_TYPES = [str, Tuple[str, str], List[str]]


class PyGameTextRenderer(TextRenderingMixin):
    """
    Constructs a text renderer.
    This feature extractor inherits from [`TextRenderingMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        font_file (`str):
            The font file (typically a file with a .ttf or .otf extension) that is loaded by pygame to render text
        dpi (`int`, defaults to 120):
            The dpi (dots per inch) count that determines the resolution of the rendered images
        background_color (`str`, defaults to "white"):
            The background color of the image
        font_color (`str`, defaults to "black"):
            The font color that is used when rendering text
        font_size (`int`, defaults to 8):
            The font size that is used when rendering text
        pad_size (`int`, defaults to 3):
            The amount of padding that is applied. Note: Currently, variable padding is only applied
            when rendering lists of individual words, and only the top is padded
        pixels_per_patch (`int`, defaults to 16):
            The number of pixels, both horizontally and vertically, of each patch in the rendered image
        max_seq_length (`int`, defaults to 529):
            The maximum number of patches which, when multiplied with pixels_per_patch, determines the width of each
            rendered image
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        font_file: str,
        dpi: int = 120,
        background_color: str = "white",
        font_color: str = "black",
        font_size: int = DEFAULT_FONT_SIZE,
        pad_size: int = DEFAULT_PAD_SIZE,
        pixels_per_patch: int = DEFAULT_PPB,
        max_seq_length: int = MAX_SEQ_LENGTH,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.font_file = font_file
        self.font_size = font_size
        self.font_color = font_color
        self.background_color = background_color

        self.pixels_per_patch = pixels_per_patch
        self.max_seq_length = max_seq_length
        self.pad_size = pad_size
        self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = (pad_size, pad_size, pad_size, pad_size)

        self.dpi = dpi
        pygame.freetype.init()
        pygame.freetype.set_default_resolution(dpi)

        self.font = None
        self.load_font()

    @property
    def max_pixels_len(self):
        return self.max_seq_length * self.pixels_per_patch

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns the state dict of the renderer without the loaded font to make it picklable

        Returns:
            The state dict of type `Dict[str, Any]`
        """

        return {
            "font_file": self.font_file,
            "dpi": self.dpi,
            "background_color": self.background_color,
            "font_color": self.font_color,
            "font_size": self.font_size,
            "pad_size": self.pad_size,
            "pixels_per_patch": self.pixels_per_patch,
            "max_seq_length": self.max_seq_length,
        }

    def __setstate__(self, state_dict: Dict[str, Any]) -> None:
        """
        Sets the state dict of the renderer, e.g. from a pickle

        Args:
            state_dict (`Dict[str, Any]`):
                The state dictionary of a `PyGameTextRenderer`, containing all necessary and optional fields to initialize
                a `PyGameTextRenderer`
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

    def _get_empty_surface(self) -> pygame.Surface:
        """
        Create and return an empty surface that we will later render the text to

        Returns:
            The blank surface of type (`~pygame.Surface`)
        """

        frame = (self.max_pixels_len, self.pixels_per_patch)
        surface = pygame.Surface(frame)
        surface.fill(pygame.color.THECOLORS[self.background_color])
        return surface

    def _draw_black_patch(self, offset: int, surface: pygame.Surface) -> pygame.Surface:
        """
        Draws a black separator patch on a surface a horizontal offset, i.e. the black patch begins <offset> pixels to
        the right from the beginning of the surface

        Args:
            offset (`int`):
                The horizontal starting position of the black patch on the surface (in pixels)
            surface (`~pygame.Surface`):
                The surface that the black patch is drawn on

        Returns:
            A surface of type `~pygame.Surface` with the black patch drawn on it

        """

        sep_rect = pygame.Rect(offset, 0, self.pixels_per_patch, self.pixels_per_patch)
        pygame.draw.rect(surface, self.font.fgcolor, sep_rect)
        return surface

    def _render_single_word(
        self, word: str, offset: int, surface: pygame.Surface, is_last: bool = False
    ) -> Tuple[pygame.Surface, int]:
        """
        Renders a single word to a surface with a horizontal offset, i.e. the rendered
        word begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        word vertically on the surface

        Args:
            word (`str`):
                The word to be rendered
            offset (`int`):
                The horizontal starting position of the rendered word on the surface (in pixels)
            surface (`~pygame.Surface`):
                The surface that the word is rendered to
            is_last (`bool`, *optional*, defaults to False):
                Boolean variable that indicates whether we are rendering the last word of the sequence, in which
                case additional padding is added to the final offset so that the black separator patch is guaranteed
                to be spaced at least this padded amount from the last word

        Returns:
            A tuple containing the surface of type `~pygame.Surface` with the sentence rendered to it and the offset
            to where the next patch begins, type `int`
        """

        text_surface, rect = self.font.render(word, self.font.fgcolor)
        rect.left = offset

        # Align words vertically based on how far they extend above the middle of the patch
        # This is necessary because words can have the same rendered height but different vertical placements
        # based on the combination of letters. E.g., "queue" mostly extends towards bottom whereas "look"
        # extends towards the top
        delta_top = self.pixels_per_patch / 2 - rect.top
        rect.top = self.pad_top + delta_top

        surface.blit(text_surface, rect)

        if is_last:
            offset += 2
        offset = self._get_offset_to_next_patch(offset + rect.width)

        return surface, offset

    def _render_single_sentence(
        self, sentence: str, offset: int, surface: pygame.Surface
    ) -> Tuple[pygame.Surface, int]:
        """
        Renders a single sentence to a surface with a horizontal offset, i.e. the rendered
        sentence begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        text vertically on the surface

        Args:
            sentence (`str`):
                The sentence to be rendered
            offset (`int`):
                The horizontal starting position of the rendered sentence on the surface (in pixels)
            surface (`~pygame.Surface`):
                The surface that the sentence is rendered to

        Returns:
            A tuple containing the surface of type `~pygame.Surface` with the sentence rendered to it and the width
            of the rendered sentence, type `int`
        """
        text_surface, rect = self.font.render(sentence, self.font.fgcolor)
        rect.midleft = (offset, surface.get_height() / 2)
        surface.blit(text_surface, rect)

        return surface, rect.width

    def _render_words_to_surface(self, words: List[str]) -> Encoding:
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

        # Keep track of the patch index at which each new word begins
        word_start_indices = [0]

        # Start with blank surface
        surface = self._get_empty_surface()
        sep_patches = []

        # Pad left with 2px
        offset = 2
        skip_last = False

        # Render each word to the start of the next patch but at least a whitespace width apart
        for word in words[:-1]:
            surface, offset = self._render_single_word(f"{word} ", offset, surface)
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

            if offset == self.max_pixels_len - self.pixels_per_patch:
                skip_last = True
                break

        # Last word is rendered without whitespace
        if not skip_last:
            surface, offset = self._render_single_word(words[-1], offset, surface, is_last=True)
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

        # Draw black rectangle on surface as separator patch
        surface = self._draw_black_patch(offset, surface)
        sep_patches.append(offset // self.pixels_per_patch)

        num_text_patches = math.ceil(offset / self.pixels_per_patch)

        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface),
            num_text_patches=num_text_patches,
            sep_patches=sep_patches,
            word_starts=word_start_indices
        )
        return encoding

    def _render_text_pair_to_surface(self, text_pair: Tuple[str, str]) -> Encoding:
        """
        Renders a pair of sentences or paragraphs to a surface and keeps track of
        how many patches in the rendered surface contain text, i.e. are neither blank nor black separator patches

        Args:
            text_pair (`Tuple[str, str]`):
                The text pair to be rendered

        Returns:
            An Encoding of type `Encoding` containing the rendered text pair and metadata
        """

        surface = self._get_empty_surface()
        sep_patches = []

        text_a, text_b = text_pair

        offset = 2

        # Render first sentence and draw on surface
        surface, text_a_width = self._render_single_sentence(text_a, offset, surface)

        # Offset is left padding + rendered width of text_a + 2 (padding)
        offset = self._get_offset_to_next_patch(offset + text_a_width + 2)

        # Draw black rectangle on surface as separator patch
        surface = self._draw_black_patch(offset, surface)
        sep_patches.append(offset // self.pixels_per_patch)

        # Render second sentence and draw on surface
        offset = offset + self.pixels_per_patch + 2
        surface, text_b_width = self._render_single_sentence(text_b, offset, surface)

        # Offset is left padding + rendered width of text_b + 2 (padding)
        offset = self._get_offset_to_next_patch(offset + text_b_width + 2)

        # Draw black rectangle on surface as separator patch
        surface = self._draw_black_patch(offset, surface)
        sep_patches.append(offset // self.pixels_per_patch)

        num_text_patches = math.ceil(offset / self.pixels_per_patch)

        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface),
            num_text_patches=num_text_patches,
            sep_patches=sep_patches
        )
        return encoding

    def _render_text_to_surface(self, text: str) -> Encoding:
        """
        Renders a single piece of text, e.g. a sentence or paragraph, to a surface and keeps track of
        how many patches in the rendered surface contain text, i.e. are neither blank nor black separator patches

        Args:
            text (`str`):
                The piece of text to be rendered

        Returns:
            An Encoding of type `Encoding` containing the rendered text and metadata
        """

        surface = self._get_empty_surface()
        sep_patches = []

        offset = 2

        # Render text
        surface, text_width = self._render_single_sentence(text, offset, surface)

        # Offset is left padding + rendered width of first sentence + 2 (padding)
        offset = self._get_offset_to_next_patch(2 + text_width + 2)

        # Draw black rectangle on surface as separator patch
        surface = self._draw_black_patch(offset, surface)
        sep_patches.append(offset // self.pixels_per_patch)

        num_text_patches = math.ceil(offset / self.pixels_per_patch)

        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface),
            num_text_patches=num_text_patches,
            sep_patches=sep_patches
        )
        return encoding

    @staticmethod
    def get_image_from_surface(surface: pygame.Surface) -> np.ndarray:
        """
        Transforms a surface containing a rendered image into a numpy image

        Args:
            surface (`pygame.Surface`):
                The pygame surface containing the rendered text

        Returns:
            An image of type `np.ndarray` of size [self.pixels_per_patch, self.max_pixels_len]
        """

        image = pygame.surfarray.pixels3d(surface)
        image = image.swapaxes(0, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def __call__(
        self,
        text: Union[str, Tuple[str, str], List[str]],
    ) -> Encoding:
        """
        Render a piece of text to a surface, convert the surface into an image and return the image
        along with metadata (the number of patches containing text and, when rendering a list of words, the patch
        indices at which each word starts)

        Args:
            text (`str` or `Tuple[str, str]` or `List[str]`):
                The text to be rendered

        Returns:
            An Encoding of type `Encoding` containing the rendered text input and metadata
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

        # Render input
        encoding = rendering_fn(text)

        return encoding

    def load_font(self) -> None:
        """
        Loads the font from specified font file with specified font size and color.
        """

        logger.info(f"Loading font from {self.font_file}")
        font = pygame.freetype.Font(self.font_file, self.font_size)
        font.style = pygame.freetype.STYLE_NORMAL
        font.fgcolor = pygame.color.THECOLORS[self.font_color]

        self.font = font
