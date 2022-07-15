
"""
Different pooling strategies for sequence classification models
"""

import json
import math
import os
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class MAB(nn.Module):
    """
    Multi-head Attention Block (Modified from https://github.com/juho-lee/set_transformer/blob/master/modules.py)
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, M):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        K = K * M  # apply mask

        # print("K.shape", K.shape)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        M_ = torch.cat(M.split(dim_split, 2), 0)

        # print("K_.shape", K_.shape)
        # print("K_.transpose(1,2).shape", K_.transpose(1,2).shape)

        # S = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        # print("S", S.shape)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O
        # torch.Size([64, 196, 768]) torch.Size([64, 196]) torch.Size([64, 196, 768])
        # K.shape torch.Size([64, 196, 768])
        # K_.shape torch.Size([256, 196, 192])
        # K_.transpose(1,2).shape torch.Size([256, 192, 196])
        # S torch.Size([256, 1, 196])


class PMA(nn.Module):
    """
    Pooling by Multi-head Attention (Modified from https://github.com/juho-lee/set_transformer/blob/master/modules.py)
    """

    def __init__(self, dim, num_heads, num_seeds=1, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, M):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, M)


class PoolingMode(Enum):
    CLS = "cls"
    MEAN = "mean"
    MAX = "max"
    PMA = "pma"

    @classmethod
    def from_string(cls, mode: str):
        cls.mode = mode
        if mode == "cls":
            return PoolingMode.CLS
        elif mode == "mean":
            return PoolingMode.MEAN
        elif mode == "max":
            return PoolingMode.MAX
        elif mode.startswith("pma"):
            return PoolingMode.PMA
        else:
            raise ValueError(f"Pooling mode {mode} not supported. Please choose from {[e.value for e in cls]}")


class Pooling(nn.Module):
    """
    Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py

    Performs pooling (max or mean) on the token embeddings.

    Pools a variable sized sequence of hidden states into a fixed size output vector. This layer also allows
    to use the CLS token if it is returned by the underlying encoder. You can concatenate multiple poolings.

    :param hidden_size: Dimensionality of the hidden states
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but divide by sqrt(input_length)
    :param pooling_mode_pma_tokens: Perform pooling by multi-head attention
    :param pooling_mode: Can be a string: mean/max/cls/pma<k>. If set, overwrites the other pooling_mode_* settings
    """

    def __init__(
        self,
        hidden_size: int,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
        pooling_mode_pma_tokens: bool = False,
        pooling_mode: Optional[str] = None,
    ):
        super(Pooling, self).__init__()

        self.config_keys = [
            "hidden_states_dim",
            "pooling_mode_cls_token",
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_pma_tokens",
        ]

        # Set pooling mode by string
        if pooling_mode is not None:
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in {"mean", "max", "cls"} or pooling_mode.startswith("pma")
            pooling_mode_cls_token = pooling_mode == "cls"
            pooling_mode_max_tokens = pooling_mode == "max"
            pooling_mode_mean_tokens = pooling_mode == "mean"
            pooling_mode_pma_tokens = pooling_mode.startswith("pma")

        self.hidden_size = hidden_size
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_pma_tokens = pooling_mode_pma_tokens

        pooling_mode_multiplier = sum(
            [
                pooling_mode_cls_token,
                pooling_mode_max_tokens,
                pooling_mode_mean_tokens,
                pooling_mode_mean_sqrt_len_tokens,
                pooling_mode_pma_tokens,
            ]
        )
        self.pooling_output_dimension = pooling_mode_multiplier * hidden_size

        if pooling_mode_pma_tokens:
            num_heads = int(pooling_mode.replace("pma", ""))
            self.pooler = PMA(hidden_size, num_heads, num_seeds=1, ln=False)

    def __repr__(self):
        return f"Pooling({self.config_dict})"

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append("cls")
        if self.pooling_mode_mean_tokens:
            modes.append("mean")
        if self.pooling_mode_max_tokens:
            modes.append("max")
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append("mean_sqrt_len_tokens")
        if self.pooling_mode_pma_tokens:
            modes.append("pma")

        return "+".join(modes)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):

        # Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = hidden_states[:, 0, :]
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(hidden_states, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        if self.pooling_mode_pma_tokens:
            # print(attention_mask)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            # print(hidden_states.shape, attention_mask.shape, input_mask_expanded.shape)
            # torch.Size([64, 196, 768]) torch.Size([64, 196]) torch.Size([64, 196, 768])
            # K.shape torch.Size([64, 196, 768])
            # K_.shape torch.Size([256, 196, 192])
            # K_.transpose(1,2).shape torch.Size([256, 192, 196])
            # S torch.Size([256, 1, 196])

            # hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            hidden_states = self.pooler(hidden_states, input_mask_expanded).squeeze(1)
            # stop
            output_vectors.append(hidden_states)

        return torch.cat(output_vectors, 1)

    @property
    def pooled_embedding_dimension(self):
        return self.pooling_output_dimension

    @property
    def config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self.config_dict, f, indent=4, sort_keys=True, separators=(",", ": "), ensure_ascii=False)

    @classmethod
    def load_from_json(cls, input_path):
        with open(os.path.join(input_path, "config.json")) as f:
            config = json.load(f)

        return cls(**config)


class PoolingForSequenceClassificationHead(nn.Module):
    """
    Layer that takes hidden states from an encoder, e.g. BERT or PIXEL, applies some basic transformations and finally
    pools the hidden states into a fixed-size output vector that serves as input to a sequence classifier.

    :param hidden_size: Hidden size of the contextualized token/patch embeddings
    :param hidden_dropout_prob: Dropout probability
    :param add_layer_norm: Whether or not layer normalization is applied
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_dropout_prob: float = 0.0,
        add_layer_norm: bool = True,
        pooling_mode: PoolingMode = PoolingMode.MEAN,
    ):
        super().__init__()

        self.add_layer_norm = add_layer_norm
        self.pooling_mode = pooling_mode

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.ln = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        if pooling_mode == PoolingMode.MEAN:
            self.pooling = Pooling(hidden_size)
        elif pooling_mode == PoolingMode.MAX:
            self.pooling = Pooling(hidden_size, pooling_mode_mean_tokens=False, pooling_mode_max_tokens=True)
        elif pooling_mode == PoolingMode.CLS:
            pass
        elif pooling_mode == PoolingMode.PMA:
            self.pooling = Pooling(hidden_size, pooling_mode_mean_tokens=False, pooling_mode=pooling_mode.mode)
        else:
            raise ValueError(f"Pooling mode {pooling_mode} not supported.")

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):

        if self.pooling_mode == PoolingMode.CLS:
            return hidden_states
        else:
            if self.pooling_mode != PoolingMode.PMA:
                hidden_states = self.activation(self.linear(hidden_states))
            if self.add_layer_norm:
                hidden_states = self.ln(hidden_states)

            hidden_states = self.dropout(hidden_states)
            return self.pooling(hidden_states=hidden_states, attention_mask=attention_mask)
