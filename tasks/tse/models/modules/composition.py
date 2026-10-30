"""TSE 条件谱拼接与 presence-aware latent 融合，不持有 separator。

融合层的 owner 与注入位置：

- speaker context/spkemb：composer 持有，``fuse_latent`` 在 latent 级融合；
- spatial cyclic DOA：composer 持有 ``cyclic_fusions``，``block_injection`` 逐 block 注入；
- visual 帧级条件：composer 持有 ``visual_fusion``（``multi_fuse=False``）或
  ``visual_stages``（``multi_fuse=True``），条件为 ``[R,D,T]`` 帧级 cue；
- textual 句级条件：融合层由 ``TextualFrontend`` 持有（单 owner，避免同一层被两个父模块
  重复登记），composer 只负责在 ``block_injection`` 中把这些层组装成 block hook，
  条件为 ``[R,D]`` 句级 cue 的广播形式。

所有注入都以 ``torch.where`` 按 presence 门控，缺 cue 行保持未融合 latent。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from tasks.tse.models.modules.fusion.speech import CrossFuse, SpeakerFuseLayer


def gate(value: Tensor, present: Tensor) -> Tensor:
    return torch.where(present.reshape(-1, *([1] * (value.ndim - 1))), value, 0)


def gate_fusion(original: Tensor, fused: Tensor, present: Tensor) -> Tensor:
    return torch.where(present.reshape(-1, *([1] * (original.ndim - 1))), fused, original)


class CueComposer(nn.Module):
    """固定调用图：先计算融合，再对最终结果门控，缺 cue 时保持原 latent。

    Context 每行仅消费有效帧；cyclic DOA 在每个 separator block 前融合。
    visual 帧级条件的融合层由本模块持有，``multi_fuse`` 决定 latent 级单次融合还是
    逐 separator block 融合；textual 句级条件的融合层由 ``TextualFrontend`` 持有，
    本模块只把它组装成同一套 block hook。USEF append 是本地多 cue 扩展，追加两个
    U 通道 latent，并非官方联合图。
    """

    def __init__(
        self, *, feature_dim: int, nband: int, num_blocks: int,
        context: Mapping[str, object] | None = None,
        context_dim: int | None = None,
        spkemb: Mapping[str, object] | None = None,
        embedding_dim: int = 192,
        cyclic_fusion: str | None = None,
        visual: Mapping[str, object] | None = None,
        visual_dim: int | None = None,
    ) -> None:
        super().__init__()
        if type(num_blocks) is not int or num_blocks < 1:
            raise ValueError(f"num_blocks must be a positive exact int, got {num_blocks!r}")
        self.num_blocks = num_blocks
        if visual is not None:
            if not isinstance(visual, Mapping):
                raise TypeError("visual must be a mapping with fusion/multi_fuse")
            if type(visual_dim) is not int or visual_dim < 1:
                raise ValueError("visual_dim must be a positive exact int when visual is set")
            fusion = visual.get("fusion", "FiLM")
            if visual.get("multi_fuse", True):
                self.visual_stages = nn.ModuleList(
                    SpeakerFuseLayer(visual_dim, feature_dim, fusion)
                    for _ in range(num_blocks)
                )
            else:
                self.visual_fusion = SpeakerFuseLayer(visual_dim, feature_dim, fusion)
        if context is not None:
            attention_dim = context.get("attention_dim", feature_dim)
            self.context_injection = context.get("injection", "latent")
            self.context_attention = CrossFuse(
                embed_dim=context_dim, atten_dim=attention_dim, mix_dim=feature_dim,
                num_heads=context.get("num_heads", 1), nband=nband, batch_first=True,
            )
            if self.context_injection == "block":
                self.context_stages = nn.ModuleList(
                    SpeakerFuseLayer(
                        attention_dim, feature_dim, context.get("fusion", "concat")
                    )
                    for _ in range(num_blocks)
                )
            else:
                self.context_fusion = SpeakerFuseLayer(
                    attention_dim, feature_dim, context.get("fusion", "concat"),
                )
        if spkemb is not None:
            self.spkemb_injection = spkemb.get("injection", "latent")
            if self.spkemb_injection == "block":
                self.speaker_stages = nn.ModuleList(
                    SpeakerFuseLayer(
                        embedding_dim, feature_dim, spkemb.get("fusion", "concat")
                    )
                    for _ in range(num_blocks)
                )
            else:
                self.speaker_fusion = SpeakerFuseLayer(
                    embedding_dim, feature_dim, spkemb.get("fusion", "concat"),
                )
        self.cyclic_fusions = nn.ModuleList(
            SpeakerFuseLayer(feature_dim, feature_dim, cyclic_fusion)
            for _ in range(num_blocks if cyclic_fusion is not None else 0)
        )

    @staticmethod
    def spectral(
        mixture: Tensor, *, spatial: Tensor | None = None,
        usef: tuple[Tensor, Tensor] | None = None,
        usef_mode: str | None = None, tfmap: Tensor | None = None,
        speaker_present: Tensor | None = None,
    ) -> Tensor:
        representation = torch.cat((mixture.real, mixture.imag), dim=1)
        if usef is not None:
            attended, encoded_mix = usef
            if usef_mode == "replace":
                representation = torch.cat((encoded_mix, gate(attended, speaker_present)), dim=1)
            elif usef_mode == "append":
                representation = torch.cat(
                    (representation, gate(torch.cat((encoded_mix, attended), dim=1), speaker_present)),
                    dim=1,
                )
            else:
                raise ValueError("USEF requires explicit append/replace composition")
        if spatial is not None:
            representation = torch.cat((representation, spatial), dim=1)
        if tfmap is not None:
            representation = torch.cat((representation, gate(tfmap, speaker_present)), dim=1)
        return representation

    @staticmethod
    def _fuse(layer, latent, condition):
        if layer.fuse_type == "FiLM":
            # FiLM 走 transpose 形式的调用，输出必须 contiguous：separator 的
            # ``forward_block`` 对 latent 做 ``view``，非连续 latent 会直接报错。
            return layer.fc(
                condition.transpose(-1, -2), latent.transpose(-1, -2)
            ).transpose(-1, -2).contiguous()
        return layer(latent, condition)

    @staticmethod
    def _frame_condition(embedding: Tensor) -> Tensor:
        """把帧级 cue ``[R,D,T]`` 展成 ``[R,1,D,T]`` 供融合层广播。"""
        if not isinstance(embedding, Tensor) or embedding.ndim != 3:
            raise ValueError(
                f"visual condition must be a [R,D,T] tensor, got {embedding!r}"
            )
        return embedding[:, None, :, :]

    @staticmethod
    def _sentence_condition(cue: Tensor) -> Tensor:
        """把句级 cue ``[R,D]`` 展成 ``[R,1,D,1]`` 供融合层广播。"""
        if not isinstance(cue, Tensor) or cue.ndim != 2:
            raise ValueError(f"textual condition must be a [R,D] tensor, got {cue!r}")
        return cue[:, None, :, None]

    def fuse_latent(
        self, latent: Tensor, *, present: Tensor | None = None,
        context: Sequence[Tensor] | None = None, embedding: Tensor | None = None,
        visual: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """latent 级融合；``visual`` 为 ``(帧级 cue [R,D,T], present)``。"""
        if context is not None:
            if getattr(self, "context_injection", "latent") != "latent":
                raise ValueError("speaker context is configured for block injection")
            if present is None:
                raise ValueError("context fusion requires the speaker presence mask")
            attended = torch.cat([
                self.context_attention(latent[row:row + 1], value)
                for row, value in enumerate(context)
            ], dim=0)
            latent = gate_fusion(latent, self._fuse(self.context_fusion, latent, attended), present)
        if embedding is not None:
            if getattr(self, "spkemb_injection", "latent") != "latent":
                raise ValueError("speaker embedding is configured for block injection")
            if present is None:
                raise ValueError("embedding fusion requires the speaker presence mask")
            fused = self._fuse(self.speaker_fusion, latent, embedding[:, None, :, None])
            latent = gate_fusion(latent, fused, present)
        if visual is not None:
            layer = getattr(self, "visual_fusion", None)
            if layer is None:
                raise ValueError(
                    "visual cue is configured for per-block fusion; use block_injection"
                )
            condition, visual_present = visual
            fused = self._fuse(layer, latent, self._frame_condition(condition))
            latent = gate_fusion(latent, fused, visual_present)
        return latent

    def _stage_conditioners(self, fusions, present: Tensor):
        return tuple(
            lambda latent, condition, fusion=fusion: gate_fusion(
                latent, self._fuse(fusion, latent, condition), present,
            )
            for fusion in fusions
        )

    def block_conditioners(self, present: Tensor):
        """spatial cyclic DOA 的 per-block conditioners；condition 由调用方提供。"""
        return self._stage_conditioners(self.cyclic_fusions, present)

    def block_injection(
        self, *, speaker: tuple[Sequence[Tensor] | None, Tensor | None, Tensor] | None = None,
        spatial: tuple[Tensor, Tensor] | None = None,
        visual: tuple[Tensor, Tensor] | None = None,
        textual_stages: tuple[Sequence[nn.Module], Tensor, Tensor] | None = None,
    ) -> tuple[tuple | None, Tensor | None]:
        """组装 separator 的 per-block 条件注入。

        Args:
            spatial: ``(cyclic embedding [R,1,D,T], present)``；消费 composer 持有的
                ``cyclic_fusions``。
            visual: ``(帧级 cue [R,D,T], present)``；消费 composer 持有的 ``visual_stages``。
            textual_stages: ``(逐 stage 融合层, 句级 cue [R,D], present)``；层由
                ``TextualFrontend`` 持有，composer 只组装 hook。

        Returns:
            ``(conditioners, conditions)``；无 per-block 注入时返回 ``(None, None)``。
            同一时间只允许一个 cue 源，且必须显式给出条件张量。
        """
        sources: list[tuple[str, tuple, Tensor, Tensor]] = []
        if speaker is not None:
            context, embedding, present = speaker
            if context is not None and getattr(self, "context_injection", "latent") == "block":
                if not hasattr(self, "context_stages"):
                    raise ValueError("speaker context block injection is not configured")
                dummy = context[0].new_zeros((len(context), 1)) if context else None
                if dummy is None:
                    raise ValueError("speaker context block injection requires context rows")
                sources.append(("speaker_context", tuple(self.context_stages), dummy, present))
            if embedding is not None and getattr(self, "spkemb_injection", "latent") == "block":
                if not hasattr(self, "speaker_stages"):
                    raise ValueError("speaker embedding block injection is not configured")
                sources.append(("speaker_embedding", tuple(self.speaker_stages), self._sentence_condition(embedding), present))
        if spatial is not None:
            if not len(self.cyclic_fusions):
                raise ValueError("spatial per-block injection requires cyclic fusions")
            sources.append(("spatial", tuple(self.cyclic_fusions), spatial[0], spatial[1]))
        if visual is not None:
            if not len(getattr(self, "visual_stages", ())):
                raise ValueError("visual per-block injection requires multi_fuse=True")
            sources.append(("visual", tuple(self.visual_stages), self._frame_condition(visual[0]), visual[1]))
        if textual_stages is not None:
            fusions, cue, present = textual_stages
            fusions = tuple(fusions)
            if len(fusions) != self.num_blocks:
                raise ValueError(
                    f"textual_stages must provide {self.num_blocks} fusion layers, "
                    f"got {len(fusions)}"
                )
            sources.append(("textual", fusions, self._sentence_condition(cue), present))
        if not sources:
            return None, None
        if len(sources) > 1:
            raise ValueError(
                "only one per-block cue source can be active; the configured cue "
                "combination is not supported by the modular separator"
            )
        name, fusions, condition, present = sources[0]
        if condition is None:
            raise ValueError("per-block injection requires an explicit condition tensor")
        if name == "speaker_context":
            context = speaker[0] if speaker is not None else None
            if context is None:
                raise ValueError("speaker context block injection lost its context rows")
            return tuple(
                lambda latent, _condition, fusion=fusion: gate_fusion(
                    latent,
                    self._fuse(
                        fusion,
                        latent,
                        torch.cat([
                            self.context_attention(latent[row : row + 1], value)
                            for row, value in enumerate(context)
                        ], dim=0),
                    ),
                    present,
                )
                for fusion in fusions
            ), condition
        return self._stage_conditioners(fusions, present), condition
