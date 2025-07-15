
"""
Functions taken and modified from https://github.com/hila-chefer/Transformer-MM-Explainability

Credit:
@InProceedings{Chefer_2021_ICCV,
   author    = {Chefer, Hila and Gur, Shir and Wolf, Lior},
   title     = {Generic Attention-Model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers},
   booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
   month     = {October},
   year      = {2021},
   pages     = {397-406}
}

"""

from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torch import nn

from .misc import patchify, unpatchify


# rule 5 from paper
def avg_heads(cam: torch.Tensor, grad: torch.Tensor):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss: torch.Tensor, cam_ss: torch.Tensor):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_relevance(model: nn.Module, inputs: Dict[str, torch.Tensor], index: Optional[int] = None):
    output = model(**inputs, register_hook=True)["logits"]

    if index is None:
        index = torch.argmax(output, dim=-1)

    one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
    one_hot[0, index] = 1
    one_hot = one_hot.requires_grad_(True)
    one_hot = torch.sum(one_hot * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    num_tokens = model.vit.encoder.layer[0].attention.attention.get_attention_map().shape[-1]

    R = torch.eye(num_tokens, num_tokens)
    for layer in model.vit.encoder.layer:
        grad = layer.attention.attention.get_attention_gradients()
        cam = layer.attention.attention.get_attention_map()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R, cam)

    return R[0, 1:]


# create heatmap from mask on image
def show_cam_on_image(img: torch.Tensor, mask: torch.Tensor):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = torch.einsum("chw->hwc", unpatchify(patchify(torch.einsum("hwc->chw", img))))
    img = np.float32(img)
    cam = heatmap + img
    cam = cam / np.max(cam)
    return cam


def generate_visualization(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    image_hw: int,
    patch_size: int = 16,
    class_index: Optional[int] = None,
):

    transformer_attribution = generate_relevance(model, inputs, index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, image_hw // patch_size, image_hw // patch_size)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=16, mode="bilinear"
    )
    transformer_attribution = transformer_attribution.reshape(image_hw, image_hw)
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
        transformer_attribution.max() - transformer_attribution.min()
    )

    original_image = inputs["pixel_values"].squeeze()
    image_transformer_attribution = original_image.permute(1, 2, 0)
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
        image_transformer_attribution.max() - image_transformer_attribution.min()
    )
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis
