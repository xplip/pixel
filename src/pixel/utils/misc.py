import numpy as np
import torch

from .defaults import DEFAULT_PPB, MAX_SEQ_LENGTH


def patchify(imgs: torch.Tensor, patch_size: int = DEFAULT_PPB):
    """
    imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
    or
    imgs: (3, H, W) x: (L, patch_size**2 *3)
    """
    is_single_image = len(imgs.shape) == 3
    if is_single_image:
        imgs = imgs.unsqueeze(0)

    p = patch_size
    assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

    if is_single_image:
        return x.squeeze(0)
    return x


def unpatchify(x: torch.Tensor, patch_size: int = DEFAULT_PPB):
    """
    x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
    or
    x: (L, patch_size**2 *3) imgs: (3, H, W)
    """
    is_single_image = len(x.shape) == 2
    if is_single_image:
        x = x.unsqueeze(0)

    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

    if is_single_image:
        return imgs.squeeze(0)
    return imgs


def glue_strip_spaces(sent: str):
    """
    Preprocessing function for GLUE inputs
    Naively removes whitespaces before and after certain strings
    """
    sent = sent.replace(" ,", ",")
    sent = sent.replace(" .", ".")
    sent = sent.replace(" !", "!")
    sent = sent.replace(" ?", "?")
    sent = sent.replace(" #", "#")
    sent = sent.replace(" /", "/")
    sent = sent.replace(' "', '"')
    sent = sent.replace('" ', '"')
    sent = sent.replace(" '", "'")
    sent = sent.replace("' ", "'")
    sent = sent.replace(" n't", "n't")
    sent = sent.replace("( ", "(")
    sent = sent.replace(" )", ")")
    sent = sent.replace("[ ", "[")
    sent = sent.replace(" ]", "]")
    return sent


def get_attention_mask(num_text_patches: int, seq_length: int = MAX_SEQ_LENGTH):
    """
    Creates an attention mask of size [1, seq_length]
    The mask is 1 where there is text or a [SEP] black patch and 0 everywhere else
    """
    n = min(num_text_patches + 1, seq_length)  # Add 1 for [SEP] token (black patch)
    zeros = torch.zeros(seq_length)
    ones = torch.ones(n)
    zeros[:n] = ones
    return zeros


def clip(x: torch.Tensor):
    """
    Transforms tensor from [0, 1] range into [0, 255] range and clips it for proper display as image
    Expects input and returns output of shape [channels, height, width]
    """
    x = torch.einsum("chw->hwc", x)
    x = torch.clip(x * 255, 0, 255)
    x = torch.einsum("hwc->chw", x)
    return x


def format_mask(x: torch.Tensor):
    """
    Wraps a mask tensor into square, e.g. from 1x529 into 368x368 and clips it for proper display
    """
    x = x.unsqueeze(-1).repeat(1, 1, 768)
    x = unpatchify(x).squeeze()
    x = np.uint8(255 * torch.einsum("chw->hwc", x).detach().cpu().numpy())
    return x


def format_img(x: torch.Tensor):
    """
    Wraps an image tensor into square, e.g. from 16x8464 to 368x368 and clips it for proper display
    """
    return clip(unpatchify(patchify(x)).squeeze())


def mark_answer(start_pos: int, end_pos: int, seq_length):
    n = (end_pos + 1) - start_pos
    zeros = torch.zeros(seq_length)
    ones = torch.ones(n)
    zeros[start_pos : end_pos + 1] = ones
    return zeros
