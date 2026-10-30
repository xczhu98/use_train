"""SpeakerEncoder：批量 fbank + vendored ECAPA-TDNN 的统一封装.

- 句级输出：``forward(wav)`` → ``(B, embed_dim)``（默认 192）；
- 帧级输出：``forward_frame_level(wav)`` → ``(B, channels, T_e)``（ECAPA
  ``_get_frame_level_feat`` 的 ``out4``，默认 512 维，供 ``context`` cue 使用）；
- ``freeze=True`` 时冻结全部参数，并覆写 ``train()`` 强制 ``spk_model`` 恒为
  eval（防止外层 ``model.train()`` 更新 BN running stats）；
- 未冻结时句级 ``forward`` 的句级 head 含 ``nn.BatchNorm1d``，其输入是池化后的
  2-D ``(B, dim)``（没有额外样本轴），train 模式要求每个通道至少 2 个样本。该
  架构事实登记在 :data:`_SENTENCE_BN_PATHS`，并由
  :func:`train_sentence_batch_requires_two` 暴露给训练入口做配置边界 fail-fast；
  长度分桶后的单行组由 :func:`sentence_batch_statistics_off` 临时改用 BN running
  statistics，使每行输出仍只依赖自身且不触发
  ``Expected more than 1 value per channel when training``。帧级
  ``forward_frame_level`` 的 ``nn.BatchNorm1d`` 作用在 ``(B, C, T)`` 上，帧数 > 1
  时 batch=1 合法，eval 模式同样不受约束；
- 预训练权重路径**必须经 YAML 配置注入**（红线：禁硬编码），加载时仅做
  wespeaker 格式键名适配（去 ``module.`` 前缀、显式允许 ``loss.`` / ``projection.``
  官方附加 namespace），其余未知键或不完整权重均 fail-closed。

Author: Xincheng Zhu
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from core.utils.safe_load import safe_torch_load
from tasks.tse.models.speaker.ecapa import ECAPA_TDNN_GLOB_c512
from tasks.tse.models.speaker.fbank import FbankKaldi
from tasks.tse.models.speaker.resnet import ResNet34

logger = logging.getLogger(__name__)

_SPEAKER_MODELS = {
    "ECAPA_TDNN_GLOB_c512": ECAPA_TDNN_GLOB_c512,
    "ResNet34": ResNet34,
}

_SPEAKER_MODEL_DEFAULTS = {
    "ECAPA_TDNN_GLOB_c512": {
        "embed_dim": 192,
        "feat_dim": 80,
        "pooling_func": "ASTP",
    },
    "ResNet34": {
        "embed_dim": 256,
        "feat_dim": 80,
        "pooling_func": "TSTP",
        "two_emb_layer": False,
    },
}

# wespeaker 训练头不属于 encoder 主体；只允许这些明确的官方 namespace。
_WESPEAKER_ALLOWED_EXTRA_PREFIXES = ("loss.", "projection.")
_SENTENCE_EMBEDDING_CACHE_MAXSIZE = 128

# 句级 head 上输入为 2-D ``(B, dim)`` 的 ``nn.BatchNorm1d`` 所在子模块路径。句级
# head 的池化输出是 ``(B, dim)``，没有额外样本轴，因此 train 模式下该 BN 要求每个
# 通道至少 2 个样本。ECAPA 的 ``self.bn``（``ecapa.py::ECAPA_TDNN.forward``）恒在
# 句级路径上，``emb_bn=True`` 时再追加 ``self.bn2``；ResNet34 只在
# ``two_emb_layer=True`` 时经 ``seg_bn_1`` 引入同一约束（否则该路径为
# ``nn.Identity``）。帧级 ``forward_frame_level`` 的 BatchNorm1d 作用在
# ``(batch, channels, frames)`` 上，帧数 > 1 时 batch=1 合法，故 context / tfmap
# 等帧级 cue 不受此约束，也不在本表的登记路径内。
_SENTENCE_BN_PATHS: dict[str, tuple[str, ...]] = {
    "ECAPA_TDNN_GLOB_c512": ("bn", "bn2"),
    "ResNet34": ("seg_bn_1",),
}

# 句级 BN 需要 spk_args 显式打开（ResNet34 的 ``two_emb_layer``）才存在的模型；其余
# 句级 BN 模型的句级 head 无条件含 ``nn.BatchNorm1d``。
_OPTIONAL_SENTENCE_BN_MODELS = frozenset({"ResNet34"})

_TRAIN_SENTENCE_BN_MODELS = (
    frozenset(_SENTENCE_BN_PATHS) - _OPTIONAL_SENTENCE_BN_MODELS
)


def train_sentence_batch_requires_two(
    model: str, spk_args: Mapping | None = None
) -> bool:
    """该 speaker 模型在 train 模式下是否要求每次前向 >= 2 行 enrollment。

    Args:
        model: :data:`_SPEAKER_MODELS` 中的模型名。
        spk_args: 与 :class:`SpeakerEncoder` 相同的模型构造参数覆盖；ResNet34 的
            句级 head 只在 ``two_emb_layer=True`` 时含 ``BatchNorm1d``。

    Returns:
        句级 head 在 train 模式下需要 batch >= 2 时为 ``True``。
    """
    if model not in _SPEAKER_MODELS:
        raise ValueError(
            f"Unknown speaker model {model!r}, available: {list(_SPEAKER_MODELS)}"
        )
    if model in _TRAIN_SENTENCE_BN_MODELS:
        return True
    merged_spk_args = {**_SPEAKER_MODEL_DEFAULTS[model], **(spk_args or {})}
    return bool(merged_spk_args.get("two_emb_layer", False))


def sentence_batch_norm_modules(encoder) -> tuple[nn.BatchNorm1d, ...]:
    """定位该 encoder 句级 head 上输入为 2-D ``(B, dim)`` 的 ``nn.BatchNorm1d``。

    只按 :data:`_SENTENCE_BN_PATHS` 登记的句级路径查找，不遍历模块树，因此
    fbank / 卷积那类作用在 ``(B, C, T)`` 的 ``nn.BatchNorm1d`` 不会被返回。

    Args:
        encoder: :class:`SpeakerEncoder`（或同构对象：``model_name`` + ``spk_model``）。

    Returns:
        句级 BN 模块元组；encoder 不是句级 BN 模型、或登记路径在当前 spk_args 下
        为 ``nn.Identity``（如 ResNet34 ``two_emb_layer=False``）时为空元组。

    Raises:
        RuntimeError: 登记的句级路径在当前模型结构上不是 ``nn.BatchNorm1d``，或该
            BN 未跟踪 running stats（eval 语义仍用 batch statistics，单行组无法保持
            逐行不变）。
    """
    model = getattr(encoder, "model_name", None)
    paths = _SENTENCE_BN_PATHS.get(model)
    spk_model = getattr(encoder, "spk_model", None)
    if paths is None or spk_model is None:
        return ()
    modules: list[nn.BatchNorm1d] = []
    for path in paths:
        module = spk_model
        for part in path.split("."):
            if not hasattr(module, part):
                raise RuntimeError(
                    f"speaker model {model!r} 的句级 BN 路径 {path!r} 不存在："
                    "模型结构与 _SENTENCE_BN_PATHS 登记的架构事实不一致"
                )
            module = getattr(module, part)
        if isinstance(module, nn.Identity):
            continue
        if not isinstance(module, nn.BatchNorm1d):
            raise RuntimeError(
                f"speaker model {model!r} 的句级路径 {path!r} 实际为 "
                f"{type(module).__name__}；句级 BN 合同要求 nn.BatchNorm1d"
            )
        if not module.track_running_stats:
            raise RuntimeError(
                f"speaker model {model!r} 的句级 BN {path!r} 必须 "
                "track_running_stats=True，否则 eval 语义仍使用 batch statistics"
            )
        modules.append(module)
    return tuple(modules)


def sentence_batch_norm_requires_two_rows(encoder) -> bool:
    """该 encoder 的句级 head 是否含 train 模式下要求 >= 2 行的 ``nn.BatchNorm1d``。"""
    return bool(sentence_batch_norm_modules(encoder))


@contextlib.contextmanager
def sentence_batch_statistics_off(encoder):
    """临时把该 encoder 的句级 BN 切到 running statistics，退出时（含异常）恢复。

    只作用于 :func:`sentence_batch_norm_modules` 定位到的句级 ``nn.BatchNorm1d``
    （输入为 2-D ``(B, dim)``）；fbank / 卷积那类作用在 ``(B, C, T)`` 的 BatchNorm
    与 dropout 等其它 train 行为不受影响。用于 train 模式下句级编码的单行长度分组：
    句级 BN 要求每个通道 >= 2 个样本，单行组改用 running statistics 后每行输出仍只
    依赖自身（保持逐行不变），也不会触发
    ``Expected more than 1 value per channel when training``。

    Yields:
        :func:`sentence_batch_norm_modules` 定位到的句级 BN 元组。
    """
    modules = sentence_batch_norm_modules(encoder)
    previous_modes = [(module, module.training) for module in modules]
    try:
        for module, _ in previous_modes:
            module.train(False)
        yield modules
    finally:
        for module, was_training in previous_modes:
            module.train(was_training)


def _can_cast_bn_counter(actual: torch.Tensor, expected_dtype: torch.dtype) -> bool:
    """浮点 BN counter 是否可无损转换到目标整数 dtype。"""
    if (
        not actual.dtype.is_floating_point
        or expected_dtype.is_floating_point
        or expected_dtype == torch.bool
    ):
        return False
    integer_info = torch.iinfo(expected_dtype)
    # 两个 exclusive 上界均保留：前者显式锁定 ``< 2**bits`` 合同，
    # 后者约束目标有符号 dtype 的实际可表示范围。
    return bool(
        torch.isfinite(actual).all()
        and torch.equal(actual, actual.trunc())
        and (actual >= 0).all()
        and (actual < float(2**integer_info.bits)).all()
        and (actual < float(integer_info.max + 1)).all()
    )


class SpeakerEncoder(nn.Module):
    """fbank + vendored WeSpeaker 说话人编码器。

    Args:
        model: 说话人模型名（``ECAPA_TDNN_GLOB_c512`` / ``ResNet34``）。
        pretrained: wespeaker 格式预训练 ckpt 路径（YAML 注入）；``None`` 表示随机初始化。
        pretrained_state: 已由调用方从 immutable bytes 加载的 checkpoint；与
            ``pretrained`` 路径配套用于身份标签，不再重复按路径读取。
        pretrained_prefix: 从完整模型 checkpoint 显式选择 speaker 子模块的键前缀；
            省略时要求 checkpoint 本身就是 speaker encoder state。
        freeze: 是否冻结。``None`` 时取默认值：加载了预训练权重则冻结，否则不冻结。
        spk_args: 传给模型构造的参数（``embed_dim`` / ``feat_dim`` / ``pooling_func``）。
        fbank_args: 传给 :class:`FbankKaldi` 的参数。
    """

    def __init__(
        self,
        model: str = "ECAPA_TDNN_GLOB_c512",
        pretrained: str | None = None,
        freeze: bool | None = None,
        spk_args: dict | None = None,
        fbank_args: dict | None = None,
        *,
        pretrained_state: Mapping | None = None,
        pretrained_prefix: str | None = None,
    ):
        super().__init__()
        if model not in _SPEAKER_MODELS:
            raise ValueError(
                f"Unknown speaker model {model!r}, available: {list(_SPEAKER_MODELS)}"
            )
        merged_spk_args = {**_SPEAKER_MODEL_DEFAULTS[model], **(spk_args or {})}
        self.embed_dim = merged_spk_args["embed_dim"]
        self.model_name = model

        self.fbank = FbankKaldi(**(fbank_args or {}))
        self.spk_model = _SPEAKER_MODELS[model](**merged_spk_args)
        self.frame_dim = getattr(self.spk_model, "frame_dim", 512)
        # 只缓存冻结 encoder 的 eval/no-grad 句级结果；缓存不注册为 module
        # state，避免污染 checkpoint 或改变参数合同。
        self._sentence_embedding_cache: OrderedDict[
            tuple, torch.Tensor
        ] = OrderedDict()
        self._sentence_embedding_cache_hits = 0
        self._sentence_embedding_cache_misses = 0

        if pretrained_prefix is not None and (
            not isinstance(pretrained_prefix, str)
            or not pretrained_prefix
            or pretrained_prefix != pretrained_prefix.strip()
        ):
            raise ValueError("pretrained_prefix must be a non-empty trimmed string or None")
        if pretrained_state is not None and pretrained is None:
            raise ValueError("pretrained_state requires pretrained path for diagnostics")
        if pretrained is not None:
            self._load_wespeaker_ckpt(
                pretrained,
                checkpoint=pretrained_state,
                prefix=pretrained_prefix,
            )

        # 冻结策略：默认"有预训练则冻结"
        self.freeze = (pretrained is not None) if freeze is None else freeze
        if self.freeze:
            for p in self.spk_model.parameters():
                p.requires_grad = False
            self.spk_model.eval()

    def _load_wespeaker_ckpt(
        self,
        path: str,
        *,
        checkpoint: Mapping | None = None,
        prefix: str | None = None,
    ) -> None:
        """严格加载 wespeaker ckpt，并只过滤显式允许的官方附加 namespace。

        ``module.`` 前缀兼容 DDP 保存格式；``loss.`` / ``projection.`` 是
        wespeaker 训练头，不属于 encoder 主体，允许过滤。模型权重键必须完整命中，
        未知键、缺失键以及模型参数和普通 buffer 的 shape/dtype 不匹配都会
        fail-closed；仅允许有限且逐项为整数的 ``num_batches_tracked`` 浮点
        buffer 安全转换为模型要求的整数 dtype。
        """
        checkpoint_path = Path(path).expanduser()
        ckpt = (
            safe_torch_load(
                checkpoint_path,
                map_location="cpu",
                purpose=f"load official WeSpeaker {self.model_name} checkpoint",
            )
            if checkpoint is None
            else checkpoint
        )
        # 兼容 {"state_dict"/"model": ...} 包装格式
        if isinstance(ckpt, dict):
            for wrapper_key in ("state_dict", "model"):
                if wrapper_key in ckpt and isinstance(ckpt[wrapper_key], dict):
                    ckpt = ckpt[wrapper_key]
                    break
            else:
                models = ckpt.get("models")
                if prefix is not None and isinstance(models, list) and models:
                    candidate = models[0]
                    ckpt = (
                        candidate.state_dict()
                        if hasattr(candidate, "state_dict")
                        else candidate
                    )
        if not isinstance(ckpt, dict):
            raise ValueError(
                f"SpeakerEncoder ckpt {path} 的权重主体必须是 dict，"
                f"实际为 {type(ckpt).__name__}"
            )

        model_state = self.spk_model.state_dict()
        model_keys = set(model_state)
        adapted = {}
        unexpected = []
        duplicate = []
        for raw_key, value in ckpt.items():
            if not isinstance(raw_key, str):
                unexpected.append(repr(raw_key))
                continue
            # 去 DDP "module." 前缀；附加 namespace 也按去前缀后的名字判断。
            key = raw_key[len("module."):] if raw_key.startswith("module.") else raw_key
            if prefix is not None:
                if not key.startswith(prefix):
                    continue
                key = key[len(prefix):]
            if key in model_keys:
                if key in adapted:
                    duplicate.append(key)
                else:
                    adapted[key] = value
            elif key.startswith(_WESPEAKER_ALLOWED_EXTRA_PREFIXES):
                continue
            else:
                unexpected.append(key)

        if not adapted:
            raise ValueError(
                f"SpeakerEncoder ckpt {path} 与模型 {len(model_keys)} 个权重键 "
                f"0 匹配，疑似 ckpt 与模型结构不对应"
            )

        missing = sorted(model_keys - set(adapted))
        if missing or unexpected or duplicate:
            details = []
            if missing:
                details.append(f"缺失键={missing}")
            if unexpected:
                details.append(f"未知键={sorted(set(unexpected))}")
            if duplicate:
                details.append(f"重复键={sorted(set(duplicate))}")
            raise ValueError(
                f"SpeakerEncoder ckpt {path} 严格键校验失败：" + "; ".join(details)
            )

        mismatches = []
        for key, expected in model_state.items():
            actual = adapted[key]
            if not isinstance(actual, torch.Tensor):
                mismatches.append(
                    f"{key}: expected Tensor, got {type(actual).__name__}"
                )
                continue
            if actual.shape != expected.shape:
                mismatches.append(
                    f"{key}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
                )
                continue
            if (actual.is_floating_point() or actual.is_complex()) and not bool(
                torch.isfinite(actual).all().item()
            ):
                mismatches.append(f"{key}: contains NaN/Inf")
                continue
            if actual.dtype != expected.dtype:
                # 部分官方 WeSpeaker 存档把 BatchNorm 计数 buffer 保存为
                # float32；它不是模型参数，仅在逐项有限、为整数、非负且未
                # 超出目标整数 dtype 位宽/可表示范围时允许安全转换。
                # 其它权重的 dtype 不匹配仍必须 fail-closed。
                is_bn_counter = key.endswith("num_batches_tracked")
                can_cast_counter = is_bn_counter and _can_cast_bn_counter(
                    actual, expected.dtype
                )
                if can_cast_counter:
                    adapted[key] = actual.to(dtype=expected.dtype)
                else:
                    mismatches.append(
                        f"{key}: dtype {actual.dtype} != {expected.dtype}"
                    )
        if mismatches:
            raise ValueError(
                f"SpeakerEncoder ckpt {path} 权重 shape/dtype 校验失败："
                + "; ".join(mismatches)
            )

        try:
            self.spk_model.load_state_dict(adapted, strict=True)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"SpeakerEncoder ckpt {path} 严格 load_state_dict 失败: {exc}"
            ) from exc
        logger.info(
            "SpeakerEncoder 加载预训练权重 %s（%d/%d 键匹配）",
            path, len(adapted), len(model_keys),
        )

    def train(self, mode: bool = True) -> "SpeakerEncoder":
        """冻结时强制 ``spk_model`` 恒为 eval，防止 BN running stats 被更新。"""
        super().train(mode)
        if mode:
            self.clear_embedding_cache()
        if self.freeze:
            self.spk_model.eval()
        return self

    def clear_embedding_cache(self) -> None:
        """清理句级 embedding cache，不影响模型参数或 buffer。"""
        self._sentence_embedding_cache.clear()

    def embedding_cache_info(self) -> dict[str, int]:
        """返回句级 embedding cache 的有界统计信息。"""
        return {
            "entries": len(self._sentence_embedding_cache),
            "hits": self._sentence_embedding_cache_hits,
            "misses": self._sentence_embedding_cache_misses,
            "max_entries": _SENTENCE_EMBEDDING_CACHE_MAXSIZE,
        }

    @staticmethod
    def _sentence_embedding_cache_key(wav: torch.Tensor) -> tuple:
        """按输入波形内容生成稳定 cache key，而不是依赖 Tensor 地址。"""
        contiguous = wav.detach().contiguous().cpu()
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(raw).digest()
        return tuple(contiguous.shape), str(contiguous.dtype), digest

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """句级 speaker embedding。

        Args:
            wav: ``(B, T)`` 波形（16kHz，[-1, 1] 量级）。
        Returns:
            ``(B, embed_dim)`` 句级 embedding。
        """
        cache_key = None
        cache_enabled = (
            self.freeze and not self.training and not torch.is_grad_enabled()
        )
        if cache_enabled:
            cache_key = self._sentence_embedding_cache_key(wav)
            cached = self._sentence_embedding_cache.get(cache_key)
            if cached is not None:
                self._sentence_embedding_cache.move_to_end(cache_key)
                self._sentence_embedding_cache_hits += 1
                return cached.to(device=wav.device)
            self._sentence_embedding_cache_misses += 1

        fb = self.fbank(wav)
        out = self.spk_model(fb)
        if isinstance(out, tuple):
            out = out[-1]
        if cache_key is not None:
            self._sentence_embedding_cache[cache_key] = out.detach().cpu()
            self._sentence_embedding_cache.move_to_end(cache_key)
            while (
                len(self._sentence_embedding_cache)
                > _SENTENCE_EMBEDDING_CACHE_MAXSIZE
            ):
                self._sentence_embedding_cache.popitem(last=False)
        return out

    def forward_frame_level(self, wav: torch.Tensor) -> torch.Tensor:
        """帧级 speaker 特征（ECAPA 最后一个 SE-Res2Block 输出）。

        Args:
            wav: ``(B, T)`` 波形。
        Returns:
            ``(B, channels, T_e)`` 帧级特征（ECAPA_TDNN_GLOB_c512 下 channels=512）。
        """
        fb = self.fbank(wav)
        if self.model_name == "ECAPA_TDNN_GLOB_c512":
            out = self.spk_model._get_frame_level_feat(fb)
            if isinstance(out, tuple):
                out = out[-1]
            return out
        out = self.spk_model.get_frame_level_feat(fb)
        if out.dim() != 3:
            raise RuntimeError(
                f"{self.model_name}.get_frame_level_feat must return [B, T, D], "
                f"got {tuple(out.shape)}"
            )
        return out.transpose(1, 2).contiguous()
