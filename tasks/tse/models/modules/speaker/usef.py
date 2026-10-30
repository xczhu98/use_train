"""USEF 注意力模块。

参考：Z. Bang et al., "USEF-TSE: Universal Speaker Embedding Free Target Speaker
Extraction"。

Author: Xincheng Zhu
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


class USEF_attentionblock(nn.Module):
    """USEF 跨谱注意力块（mix 为 query，enroll 为 key/value）。"""

    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,  # 128
        n_freqs,  # 65
        n_head,  # 4
        approx_qk_dim,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        if activation != "prelu":
            raise ValueError(f"activation must be 'prelu', got {activation!r}")
        if n_head <= 0:
            raise ValueError(f"n_head must be > 0, got {n_head}")
        if emb_dim % n_head != 0:
            raise ValueError(
                f"emb_dim={emb_dim} must be divisible by n_head={n_head}"
            )

        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)  # approx_qk_dim 仅为近似值

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DCF((n_head, E, n_freqs), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * E, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DCF((n_head, E, n_freqs), eps=eps),
        )

        self.add_module("attn_conv_V",
                        nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1))
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DCF(
                (n_head, emb_dim // n_head, n_freqs), eps=eps),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                nn.PReLU(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.n_head = n_head

    def forward(self, batch, aux):
        """
        Args:
            batch: mix 特征 [B, C, T, Q]
            aux:   enroll 特征 [B, C, T_e, Q]
        Returns:
            out: [B, C, T, Q]
        """
        if batch.ndim != 4 or aux.ndim != 4:
            raise ValueError("USEF attention expects batch/aux [B, C, T, F]")
        if batch.shape[0] < 1 or batch.shape[0] != aux.shape[0]:
            raise ValueError("USEF attention requires matching non-empty batch axes")
        if batch.shape[-2] < 1 or aux.shape[-2] < 1:
            raise ValueError("USEF attention requires non-empty mix and enrollment time axes")
        if batch.device != aux.device or batch.dtype != aux.dtype:
            raise ValueError("USEF attention batch and aux must use the same device and dtype")
        B, _, old_T, old_Q = batch.shape
        aux_T = aux.shape[-2]
        elements = B * self.n_head * old_T * aux_T
        if elements > 16_000_000:
            raise ValueError(
                f"USEF attention has {elements} elements (>16,000,000); "
                "chunk the mixture or shorten enrollment"
            )

        Q = self["attn_norm_Q"](self["attn_conv_Q"](
            batch))  # [B, n_head, C, T, Q]
        K = self["attn_norm_K"](
            self["attn_conv_K"](aux))  # [B, n_head, C, T_e, Q]
        V = self["attn_norm_V"](self["attn_conv_V"](
            aux))  # [B, n_head, CV, T_e, Q]
        Q = Q.view(-1, *Q.shape[2:])  # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:])  # [B*n_head, C, T_e, Q]
        V = V.view(-1, *V.shape[2:])  # [B*n_head, CV, T_e, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, aux_T])  # [B', C*Q, T_e]

        V = V.transpose(1, 2)  # [B', T_e, CV, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T_e, CV*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T_e]
        attn_mat = F.softmax(attn_mat, dim=2)
        V = torch.matmul(attn_mat, V)  # [B', T, CV*Q]

        V = V.reshape([old_shape[0], old_T, old_shape[-2], old_shape[-1]])
        V = V.transpose(1, 2)  # [B', CV, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view([B, self.n_head * emb_dim, old_T,
                                     old_Q])  # [B, emb_dim, T, Q]
        batch = self["attn_concat_proj"](batch)

        return batch


class LayerNormalization4DCF(nn.Module):
    """对 (C, F) 维做 LayerNorm 的 4D 版本（输入 [B, C, T, F]）。"""

    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        if len(input_dimension) != 2:
            raise ValueError(
                f"input_dimension must contain (channels, frequencies), got {input_dimension!r}"
            )
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError(
                "Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) +
            self.eps)  # [B,1,T,1]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class AllHeadPReLULayerNormalization4DCF(nn.Module):
    """所有 head 共享 PReLU + (E, F) 维 LayerNorm（输入 [B, H*E, T, F]）。"""

    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        if len(input_dimension) != 3:
            raise ValueError(
                "input_dimension must contain (heads, channels, frequencies), "
                f"got {input_dimension!r}"
            )
        H, E, n_freqs = input_dimension
        param_size = [1, H, E, 1, n_freqs]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E
        self.n_freqs = n_freqs

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expect x to have 4 dimensions, but got {x.ndim}")
        B, _, T, _ = x.shape
        x = x.view([B, self.H, self.E, T, self.n_freqs])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2, 4)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) +
            self.eps)  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x
