"""speaker 信息融合层（移植自 wesep ``wesep/modules/fusion/speech.py``）.

保留实际用到的三个模块：
- ``SpeakerFuseLayer``：句级 embedding 融合（concat / additive / multiply / FiLM）；
- ``CrossFuse``：帧级 embedding 的 cross-attention 融合（context cue）；
- ``LinearLayer``：``SpeakerFuseLayer`` 内部投影。

Author: Xincheng Zhu
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from tasks.tse.models.modules.fusion.film import FiLM


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, dummy: Optional[torch.Tensor] = None):
        return self.linear(x)


class SpeakerFuseLayer(nn.Module):
    """句级 speaker embedding 与 mix 特征融合（concat/additive/multiply/FiLM）。"""

    def __init__(self, embed_dim=256, feat_dim=512, fuse_type="concat"):
        super(SpeakerFuseLayer, self).__init__()
        if fuse_type not in ("concat", "additive", "multiply", "FiLM"):
            raise ValueError(f"unsupported fuse_type: {fuse_type!r}")

        self.embed_dim = int(embed_dim)
        self.feat_dim = int(feat_dim)
        self.fuse_type = fuse_type
        if fuse_type == "concat":
            self.fc = LinearLayer(embed_dim + feat_dim, feat_dim)
        elif fuse_type == "additive":
            self.fc = LinearLayer(embed_dim, feat_dim)
        elif fuse_type == "multiply":
            self.fc = LinearLayer(embed_dim, feat_dim)
        elif fuse_type == "FiLM":
            self.fc = FiLM(feat_dim, embed_dim)
        else:
            raise ValueError("Fuse type not defined.")

    def forward(self, x, embed):
        """
        :param x: batch x dimension x length 或 batch x band x dimension x length
        :param embed: batch x dimension x 1
        :return: 与 x 同形状
        """
        if not isinstance(x, torch.Tensor) or not isinstance(embed, torch.Tensor):
            raise TypeError("mix feature and speaker embedding must be tensors")
        if x.ndim not in (3, 4) or embed.ndim not in (3, 4):
            raise ValueError("SpeakerFuseLayer expects 3-D/4-D mix and embedding tensors")
        if x.shape[0] < 1 or x.shape[0] != embed.shape[0]:
            raise ValueError("mix and embedding must have the same non-empty batch axis")
        if x.device != embed.device or x.dtype != embed.dtype:
            raise ValueError("mix and embedding must use the same device and dtype")
        if x.shape[-1] < 1 or embed.shape[-1] < 1:
            raise ValueError("mix and embedding require non-empty time axes")
        if embed.shape[-1] not in (1, x.shape[-1]):
            raise ValueError("embedding time axis must be singleton or match mix time axis")
        if self.fuse_type == "concat":
            # For Conv
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                y = torch.cat([x, embed_t], 1)
                y = torch.transpose(y, 1, 2)
                x = torch.transpose(self.fc(y), 1, 2)
            else:
                # len(x.size()) == 4
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                y = torch.cat([x, embed_t], 2)
                y = torch.transpose(y, 2, 3)
                x = torch.transpose(self.fc(y), 2, 3).contiguous()
        elif self.fuse_type == "additive":
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                embed_t = torch.transpose(embed_t, 1, 2)
                x = x + torch.transpose(self.fc(embed_t), 1, 2)
            else:
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                embed_t = torch.transpose(embed_t, 2, 3)
                x = x + torch.transpose(self.fc(embed_t), 2, 3)
        elif self.fuse_type == "multiply":
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                embed_t = torch.transpose(embed_t, 1, 2)
                x = x * torch.transpose(self.fc(embed_t), 1, 2)
            else:
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                embed_t = torch.transpose(embed_t, 2, 3)
                x = x * torch.transpose(self.fc(embed_t), 2, 3)
        else:
            embed = embed.squeeze(-1)
            x = self.fc(embed, x)
        return x


class CrossFuse(nn.Module):
    """帧级 speaker embedding 的 cross-attention 融合（context cue）。"""

    def __init__(self,
                 embed_dim,
                 atten_dim,
                 mix_dim,
                 num_heads=1,
                 nband=1,
                 *args,
                 **kwargs):
        super().__init__()
        self.Linear = nn.ModuleList(
            [nn.Linear(embed_dim, atten_dim) for _ in range(nband)])
        if mix_dim != atten_dim:
            self.Linear_mix = nn.Linear(mix_dim, atten_dim)
        else:
            self.Linear_mix = nn.Identity()
        self.multihead_attn = nn.MultiheadAttention(atten_dim, num_heads,
                                                    *args, **kwargs)

    def forward(self, mix, emb):
        """
        mix: (B, F, T) 或 (B, band, F, T)
        emb: (B, F_e, T_e)
        return:
            spk_embeddings: (B, F, T) 或 (B, band, F, T)
        """
        if not isinstance(mix, torch.Tensor) or not isinstance(emb, torch.Tensor):
            raise TypeError("CrossFuse mix and emb must be tensors")
        if mix.ndim not in (3, 4) or emb.ndim != 3:
            raise ValueError("CrossFuse expects mix [B,F,T]/[B,band,F,T] and emb [B,F_e,T_e]")
        if mix.shape[0] < 1 or mix.shape[0] != emb.shape[0] or mix.shape[-1] < 1 or emb.shape[-1] < 1:
            raise ValueError("CrossFuse requires matching non-empty batch and time axes")
        if mix.device != emb.device or mix.dtype != emb.dtype:
            raise ValueError("CrossFuse mix and emb must use the same device and dtype")
        nband = mix.shape[1] if mix.ndim == 4 else 1
        if nband != len(self.Linear):
            raise ValueError(f"CrossFuse expected {len(self.Linear)} bands, got {nband}")
        # MultiheadAttention 需显式 materialize [B*band*head, T, T_e] 权重；限制
        # 其元素数，避免异常长 enrollment/mix 触发不可控显存申请。
        attn_elements = (
            mix.shape[0]
            * nband
            * self.multihead_attn.num_heads
            * mix.shape[-1]
            * emb.shape[-1]
        )
        if attn_elements > 16_000_000:
            raise ValueError(
                f"CrossFuse attention matrix has {attn_elements} elements (>16,000,000); "
                "chunk the mixture or shorten enrollment"
            )
        if mix.dim() == 4:
            B, nb, _F, T = mix.shape
            query = self.Linear_mix(mix.transpose(2, 3))  # (B, nb, T, atten_dim)
            emb_t = emb.transpose(1, 2)  # (B, T_e, F_e)
            # per-band Linear[i] 合并为批量 matmul：key[:, i] = emb_t @ W_i^T + b_i。
            w = torch.stack([self.Linear[i].weight for i in range(nb)])  # (nb, atten_dim, F_e)
            bias = torch.stack([self.Linear[i].bias for i in range(nb)])  # (nb, atten_dim)
            key = torch.matmul(emb_t.unsqueeze(1), w.transpose(1, 2))
            key = key + bias.view(1, nb, 1, -1)  # (B, nb, T_e, atten_dim)
            # band 维并入 batch 维，一次 MHA 取代 nb 次串行调用（数值等价）。
            query = query.reshape(B * nb, T, query.shape[-1])
            key = key.reshape(B * nb, key.shape[2], key.shape[-1])
            if not self.multihead_attn.batch_first:
                query = query.transpose(0, 1)
                key = key.transpose(0, 1)
            x, _ = self.multihead_attn(query, key, key)
            if not self.multihead_attn.batch_first:
                x = x.transpose(0, 1)
            spk_embeddings = x.reshape(B, nb, T, -1).transpose(2, 3)
        elif mix.dim() == 3:
            query = self.Linear_mix(mix.transpose(1, 2))
            key = self.Linear[0](emb.transpose(1, 2))
            value = key
            x, _ = self.multihead_attn(query, key, value)
            spk_embeddings = x.transpose(1, 2)
        return spk_embeddings
