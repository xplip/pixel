
"""
PyTorch PIXEL models
"""

import collections
import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import wandb
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import ViTForImageClassification, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer

from ...utils import DependencyParsingModelOutput, format_mask
from ..biaffine import Biaffine
from ..pooling import PoolingForSequenceClassificationHead, PoolingMode
from ..vit import ViTModel
from .configuration_pixel import PIXELConfig

logger = logging.getLogger(__name__)


PIXEL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Team-PIXEL/pixel-base",
]


class PIXELForBiaffineParsing(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)

        if not hasattr(self.config, "interpolate_pos_encoding"):
            self.config.interpolate_pos_encoding = False

        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=True)

        self.biaffine_arcs = Biaffine(n_in=config.hidden_size, bias_x=True, bias_y=False)
        self.biaffine_rels = Biaffine(n_in=config.hidden_size, n_out=config.num_labels, bias_x=True, bias_y=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_fn = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        word_starts=None,
        head_mask=None,
        arc_labels=None,
        rel_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        r""" """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # wandb.log({"pixel_values": wandb.Image(pixel_values)})

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding
            if interpolate_pos_encoding is not None
            else self.config.interpolate_pos_encoding,
            return_dict=return_dict,
        )

        outs = self.dropout(outputs[0][:, 1:, :])

        word_outputs_deps = self._merge_subword_tokens(outs, word_starts)

        # adding the CLS representation as the representation for the "root" parse token
        word_outputs_heads = torch.cat([outputs[1].unsqueeze(1), word_outputs_deps], dim=1)

        arc_logits = self.biaffine_arcs(word_outputs_deps, word_outputs_heads)
        arc_logits = arc_logits.squeeze()

        rel_logits = self.biaffine_rels(word_outputs_deps, word_outputs_heads)
        rel_logits = rel_logits.permute(0, 2, 3, 1)

        loss = None
        if arc_labels is not None and rel_labels is not None:
            loss = self._get_loss(arc_logits, rel_logits, arc_labels, rel_labels, self.loss_fn)

        if len(arc_logits.shape) == 2:
            arc_logits = arc_logits.unsqueeze(0)

        if not return_dict:
            output = (
                arc_logits,
                rel_logits,
            ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DependencyParsingModelOutput(
            loss=loss,
            arc_logits=arc_logits,
            rel_logits=rel_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _merge_subword_tokens(self, subword_outputs, word_starts):
        instances = []
        max_seq_length = subword_outputs.shape[1]

        # handling instance by instance
        for i in range(len(subword_outputs)):
            subword_vecs = subword_outputs[i]
            word_vecs = []
            starts = word_starts[i]
            mask = starts.ne(self.config.pad_token_id)
            starts = starts[mask]
            for j in range(len(starts) - 1):
                if starts[j + 1] <= 0:
                    break

                start = starts[j]
                end = starts[j + 1]
                if start == end:
                    vecs_range = subword_vecs[start]
                    word_vecs.append(vecs_range.unsqueeze(0))
                else:
                    vecs_range = subword_vecs[start:end]
                    word_vecs.append(torch.mean(vecs_range, 0).unsqueeze(0))

            instances.append(word_vecs)

        t_insts = []
        zero_tens = torch.zeros(self.config.hidden_size).unsqueeze(0)
        zero_tens = zero_tens.to(self.device)

        for inst in instances:
            if len(inst) < max_seq_length:
                for i in range(max_seq_length - len(inst)):
                    inst.append(zero_tens)
            t_insts.append(torch.cat(inst, dim=0).unsqueeze(0))

        w_tens = torch.cat(t_insts, dim=0)
        return w_tens

    def _get_loss(self, arc_preds, rel_preds, labels_arc, labels_rel, loss_fn):
        if len(arc_preds.shape) == 2:
            arc_preds = arc_preds.unsqueeze(0)

        mask = labels_arc.ne(self.config.pad_token_id)
        arc_scores, arcs = arc_preds[mask], labels_arc[mask]
        loss = loss_fn(arc_scores, arcs)

        rel_scores, rels = rel_preds[mask], labels_rel[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        rel_loss = loss_fn(rel_scores, rels)
        loss += rel_loss

        return loss


class PIXELForTokenClassification(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)

        if not hasattr(self.config, "interpolate_pos_encoding"):
            self.config.interpolate_pos_encoding = False

        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)

        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding
            if interpolate_pos_encoding is not None
            else self.config.interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output[:, 1:, :])
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PIXELForSequenceClassification(ViTForImageClassification):
    def __init__(self, config, pooling_mode: PoolingMode = PoolingMode.MEAN, add_layer_norm: bool = True):
        super().__init__(config)

        if not hasattr(self.config, "interpolate_pos_encoding"):
            self.config.interpolate_pos_encoding = False

        self.num_labels = config.num_labels

        self.add_cls_pooling_layer = pooling_mode == PoolingMode.CLS
        self.vit = ViTModel(config, add_pooling_layer=self.add_cls_pooling_layer)

        # Classifier head
        self.pooler = PoolingForSequenceClassificationHead(
            hidden_size=config.hidden_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            add_layer_norm=add_layer_norm,
            pooling_mode=pooling_mode,
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding
            if interpolate_pos_encoding is not None
            else self.config.interpolate_pos_encoding,
            return_dict=return_dict,
        )

        if self.add_cls_pooling_layer:
            sequence_output = outputs[1]
        else:
            # When not using CLS pooling mode, discard it
            sequence_output = outputs[0][:, 1:, :]

        logits = self.pooler(sequence_output, attention_mask)
        logits = self.classifier(logits)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PIXELForQuestionAnswering(ViTForImageClassification):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if not hasattr(self.config, "interpolate_pos_encoding"):
            self.config.interpolate_pos_encoding = False

        self.num_labels = config.num_labels

        self.vit = ViTModel(config, add_pooling_layer=False)

        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        head_mask=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding
            if interpolate_pos_encoding is not None
            else self.config.interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = torch.cat([outputs[0][:, 1:, :], outputs[0][:, 0, :].unsqueeze(1)], dim=1)
        # sequence_output = self.dropout(outputs[0][:, 1:, :])

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return x, x


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.
    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.
    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PIXELPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    Based on timm implementation, which can be found here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class PIXELEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PIXELPatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches ** 0.5), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, attention_mask):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.
        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # Attention mask indicates patches containing actual text
        # Out of the patches containing actual text we take the one with the highest noise
        # And bump up noise to 100 to guarantee that it gets masked
        # We therefore ensure that at least one masked patch has actual text
        # This is necessary because we only compute loss on patches having text, i.e. loss would otherwise be NaN
        noise_mask = torch.argmax(noise * attention_mask, dim=1)
        noise[torch.arange(noise.size(0)), noise_mask] = 100.0

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        attention_mask_masked = torch.gather(attention_mask, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, attention_mask_masked, mask, ids_restore

    def controlled_masking(self, sequence, attention_mask, patch_mask):

        batch_size, seq_length, dim = sequence.shape

        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        # We keep the interface the same as in the original random_masking function above
        # The only difference is that instead of random noise we use the predefined mask
        # Sometimes the greedy span masking yields fewer masked patches than specified through mask_ratio
        # We additionally mask out the difference between them randomly using noise in [0, 0.01]
        noise = patch_mask + (torch.rand(batch_size, seq_length, device=sequence.device) / 100)  # noise in [0, 0.01)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        attention_mask_masked = torch.gather(attention_mask, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, attention_mask_masked, mask, ids_restore

    def forward(self, pixel_values, attention_mask=None, patch_mask=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        if patch_mask is not None:
            embeddings, attention_mask, mask, ids_restore = self.controlled_masking(
                embeddings, attention_mask, patch_mask
            )
        else:
            embeddings, attention_mask, mask, ids_restore = self.random_masking(embeddings, attention_mask)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        attention_mask = torch.cat((torch.ones((batch_size, 1), device=attention_mask.device), attention_mask), dim=1)

        return embeddings, attention_mask, mask, ids_restore


class PIXELSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in PIXELModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class PIXELSelfOutput(nn.Module):
    """
    The residual connection is defined in PIXELLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class PIXELAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PIXELSelfAttention(config)
        self.output = PIXELSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PIXELIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class PIXELOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class PIXELLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PIXELAttention(config)
        self.intermediate = PIXELIntermediate(config)
        self.output = PIXELOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in PIXEL, layernorm is applied before self-attention
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in PIXEL, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        return layer_output


class PIXELEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PIXELLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PIXELPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PIXELConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.vit.modeling_vit.ViTPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PIXELEncoder):
            module.gradient_checkpointing = value


class PIXELModel(PIXELPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = PIXELEmbeddings(config)
        self.encoder = PIXELEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if attention_mask is None:
            attention_mask = torch.ones((pixel_values.shape[0], self.embeddings.num_patches), device=self.device)

        embedding_output, attention_mask, mask, ids_restore = self.embeddings(pixel_values, attention_mask, patch_mask)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, embedding_output.shape, self.device
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return PIXELModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PIXELDecoder(nn.Module):
    def __init__(self, config, num_patches, dtype):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [PIXELLayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size ** 2 * config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.dtype = dtype
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches ** 0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        hidden_states,
        ids_restore,
        attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        batch_size = hidden_states.shape[0]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        hidden_states = x + self.decoder_pos_embed

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, hidden_states.shape[1] - 1), device=hidden_states.device)

        attention_mask = torch.cat((torch.ones((batch_size, 1), device=attention_mask.device), attention_mask), dim=1)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, hidden_states.shape, attention_mask.device
        )

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    None,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    head_mask=None,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return PIXELDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PIXELForPreTraining(PIXELPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = PIXELModel(config)
        self.decoder = PIXELDecoder(config, num_patches=self.vit.embeddings.num_patches, dtype=self.vit.dtype)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.vit.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
        """
        p = self.vit.embeddings.patch_embeddings.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
        """
        p = self.vit.embeddings.patch_embeddings.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W] pred: [N, L, p*p*3] mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        head_mask=None,
        patch_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            patch_mask=patch_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore, attention_mask)  # [N, L, p*p*3]
        logits = decoder_outputs.logits

        merged_mask = torch.bitwise_and(mask == 1, attention_mask == 1).long()
        loss = self.forward_loss(pixel_values, logits, merged_mask)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PIXELForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            attention_mask=attention_mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class PIXELForPreTrainingOutput(ModelOutput):
    """
    Class for PIXELForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are attended to and which are not.
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mask: torch.LongTensor = None
    attention_mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PIXELModelOutput(ModelOutput):
    """
    Class for PIXELModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
                        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
                    `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PIXELDecoderOutput(ModelOutput):
    """
    Class for PIXELDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
