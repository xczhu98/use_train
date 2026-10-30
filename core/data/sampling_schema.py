"""Validated schemas and parsers for dynamic balanced sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Any, Mapping


_LANGUAGE_KEYS = frozenset({"zh", "en", "other"})
_SPEECH_KEYS = frozenset(
    {
        "language_probs",
        "dataset_alpha",
        "max_dataset_ratio",
        "language",
        "dataset",
        "speaker",
    }
)
_DATASET_KEYS = frozenset({"alpha", "max_ratio"})
_SPEAKER_KEYS = frozenset({"max_samples_per_epoch"})
_NOISE_KEYS = frozenset({"balance_categories", "balance_mode", "alpha"})

# temperature 模式默认参数（内置，非可配置）：
#   p(category | 首个可用 SR tier) ∝ n_tier(category)^alpha × difficulty(category)
# n_tier 是 exact → all higher → all 首个有候选 tier 内、排除项应用后的 UID 数；
# 同 category 跨 SR 聚合后只计一次。构建期只保存 difficulty，运行期按候选重算
# n_tier^alpha。alpha 默认 0.5 且必须在 (0, 1]；难度为领域先验，见
# DEFAULT_DIFFICULTY 注释与 docs/guides/clean_filter/Noise_Attribute_Balancing.md。
DEFAULT_ALPHA: float = 0.5


def difficulty_key(statistics: str, temporal: str) -> str:
    """DEFAULT_DIFFICULTY 的桶键：``"{statistics}:{temporal}"``。"""
    return f"{statistics}:{temporal}"


# (统计×时间) 组合 -> 难度（1=最易 .. 5=最难，先验）。
DEFAULT_DIFFICULTY: dict[str, float] = {
    "stationary:continuous": 1.5,
    "quasi_stationary:continuous": 2.0,
    "stationary:periodic": 2.0,
    "non_stationary:periodic": 2.0,
    "quasi_stationary:periodic": 2.0,
    "impulsive:periodic": 2.0,
    "non_stationary:continuous": 3.0,
    "quasi_stationary:intermittent": 3.5,
    "non_stationary:intermittent": 3.5,
    "impulsive:continuous": 4.5,
    "impulsive:intermittent": 4.5,
    "quasi_stationary:transient": 4.5,
    "impulsive:transient": 5.0,
}


@dataclass(frozen=True)
class LanguageSamplingConfig:
    """language controller：全局 exact margin 的规范概率。"""

    probs: tuple[tuple[str, Fraction], ...]


@dataclass(frozen=True)
class DatasetSamplingConfig:
    """dataset controller：全局 dataset_id 权重与上限。"""

    alpha: Fraction = Fraction(1, 2)
    max_ratio: Fraction = Fraction(3, 10)


@dataclass(frozen=True)
class SpeakerSamplingConfig:
    """speaker controller：canonical identity 的全局 epoch 绝对上限。"""

    max_samples_per_epoch: int


@dataclass(frozen=True)
class SpeechSamplingConfig:
    """Validated independent speech balancing controls."""

    language: LanguageSamplingConfig | None = None
    dataset: DatasetSamplingConfig | None = None
    speaker: SpeakerSamplingConfig | None = None

    @property
    def has_controllers(self) -> bool:
        return any((self.language, self.dataset, self.speaker))


@dataclass(frozen=True)
class NoiseSamplingConfig:
    """Validated noise balancing controls."""

    balance_categories: bool = False
    balance_mode: str = "category"
    alpha: float = 0.5


@dataclass(frozen=True)
class SamplingConfig:
    """Independent optional speech and noise sampling configuration."""

    speech: SpeechSamplingConfig | None = None
    noise: NoiseSamplingConfig | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "SamplingConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("sampling must be a mapping")
        _reject_unknown_keys(value, {"speech", "noise"}, "sampling")
        return cls(
            speech=_parse_speech(value.get("speech")),
            noise=_parse_noise(value.get("noise")),
        )


def _parse_speech(value: Any) -> SpeechSamplingConfig | None:
    if value is None or value == {}:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("sampling.speech must be a mapping")
    unknown = set(value) - _SPEECH_KEYS
    if unknown:
        raise ValueError(
            f"sampling.speech contains unknown speech sampling field(s): {sorted(unknown)}"
        )

    language = None
    dataset = None
    speaker = None
    if "language" in value:
        if "language_probs" in value:
            raise ValueError(
                "flat language_probs cannot be combined with nested speech.language"
            )
        language = _parse_language(value["language"])
    elif "language_probs" in value:
        language = _parse_language_probs(value["language_probs"])
    if "dataset" in value:
        if "dataset_alpha" in value or "max_dataset_ratio" in value:
            raise ValueError(
                "flat dataset_alpha/max_dataset_ratio cannot be combined with nested speech.dataset"
            )
        dataset = _parse_dataset(value["dataset"])
    elif "dataset_alpha" in value or "max_dataset_ratio" in value:
        dataset = _parse_dataset_flat(
            value.get("dataset_alpha"), value.get("max_dataset_ratio")
        )
    if "speaker" in value:
        speaker = _parse_speaker(value["speaker"])
    return SpeechSamplingConfig(language=language, dataset=dataset, speaker=speaker)


def _parse_language(value: Any) -> LanguageSamplingConfig:
    if not isinstance(value, Mapping):
        raise ValueError("sampling.speech.language must be a mapping")
    _reject_unknown_keys(value, {"probs"}, "sampling.speech.language")
    if "probs" not in value:
        raise ValueError("sampling.speech.language.probs is required")
    probs = value["probs"]
    if not isinstance(probs, Mapping):
        raise ValueError("sampling.speech.language.probs must be a mapping")
    _reject_unknown_keys(probs, _LANGUAGE_KEYS, "sampling.speech.language.probs")
    parsed = {
        language: _finite_number(
            probs.get(language, 0), f"sampling.speech.language.probs.{language}"
        )
        for language in ("zh", "en", "other")
    }
    if any(probability < 0 for probability in parsed.values()):
        raise ValueError("sampling.speech.language.probs values must be non-negative")
    total = math.fsum(parsed.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "sampling.speech.language.probs positive probabilities must sum to 1"
        )
    return LanguageSamplingConfig(probs=_probs_tuple(parsed))


def _parse_dataset(value: Any) -> DatasetSamplingConfig:
    if not isinstance(value, Mapping):
        raise ValueError("sampling.speech.dataset must be a mapping")
    _reject_unknown_keys(value, _DATASET_KEYS, "sampling.speech.dataset")
    alpha = (
        _parse_fraction(value["alpha"], "dataset.alpha")
        if "alpha" in value
        else Fraction(1, 2)
    )
    max_ratio = (
        _parse_fraction(value["max_ratio"], "dataset.max_ratio")
        if "max_ratio" in value
        else Fraction(3, 10)
    )
    _validate_dataset_bounds(alpha, max_ratio, "dataset.alpha", "dataset.max_ratio")
    return DatasetSamplingConfig(alpha=alpha, max_ratio=max_ratio)


def _parse_dataset_flat(alpha_value: Any, ratio_value: Any) -> DatasetSamplingConfig:
    alpha = (
        _parse_fraction(alpha_value, "dataset_alpha")
        if alpha_value is not None
        else Fraction(1, 2)
    )
    max_ratio = (
        _parse_fraction(ratio_value, "max_dataset_ratio")
        if ratio_value is not None
        else Fraction(3, 10)
    )
    _validate_dataset_bounds(alpha, max_ratio, "dataset_alpha", "max_dataset_ratio")
    return DatasetSamplingConfig(alpha=alpha, max_ratio=max_ratio)


def _parse_speaker(value: Any) -> SpeakerSamplingConfig:
    if not isinstance(value, Mapping):
        raise ValueError("sampling.speech.speaker must be a mapping")
    _reject_unknown_keys(value, _SPEAKER_KEYS, "sampling.speech.speaker")
    if "max_samples_per_epoch" not in value:
        raise ValueError("sampling.speech.speaker.max_samples_per_epoch is required")
    cap = value["max_samples_per_epoch"]
    if type(cap) is not int or cap <= 0:
        raise ValueError("max_samples_per_epoch must be a positive int")
    return SpeakerSamplingConfig(max_samples_per_epoch=cap)


def _validate_dataset_bounds(
    alpha: Fraction, max_ratio: Fraction, alpha_field: str, ratio_field: str
) -> None:
    if not 0 <= alpha <= 1:
        raise ValueError(f"{alpha_field} must be in [0, 1]")
    if not 0 < max_ratio <= 1:
        raise ValueError(f"{ratio_field} must be in (0, 1]")


def _parse_fraction(value: Any, field: str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    number = _finite_number(value, field)
    return Fraction(str(number))


def _canonicalize_dataset_config(
    config: DatasetSamplingConfig,
) -> DatasetSamplingConfig:
    alpha = _parse_fraction(config.alpha, "dataset.alpha")
    max_ratio = _parse_fraction(config.max_ratio, "dataset.max_ratio")
    _validate_dataset_bounds(alpha, max_ratio, "dataset.alpha", "dataset.max_ratio")
    return DatasetSamplingConfig(alpha=alpha, max_ratio=max_ratio)


def _validate_speaker_config(config: SpeakerSamplingConfig) -> None:
    cap = config.max_samples_per_epoch
    if type(cap) is not int or cap <= 0:
        raise ValueError("speaker.max_samples_per_epoch must be a positive int")


def _parse_noise(value: Any) -> NoiseSamplingConfig | None:
    if value is None or value == {}:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("sampling.noise must be a mapping")
    _reject_unknown_keys(value, _NOISE_KEYS, "sampling.noise")

    balance_categories = value.get("balance_categories", False)
    if type(balance_categories) is not bool:
        raise ValueError("balance_categories must be bool")
    balance_mode = value.get("balance_mode", "category")
    if balance_mode not in ("category", "temperature"):
        raise ValueError("balance_mode must be 'category' or 'temperature'")
    alpha = value.get("alpha", DEFAULT_ALPHA)
    if not (
        isinstance(alpha, (int, float))
        and not isinstance(alpha, bool)
        and 0 < alpha <= 1
    ):
        raise ValueError("alpha must be a number in (0, 1]")
    return NoiseSamplingConfig(
        balance_categories=balance_categories,
        balance_mode=balance_mode,
        alpha=float(alpha),
    )


def _parse_language_probs(value: Any) -> LanguageSamplingConfig:
    if not isinstance(value, Mapping):
        raise ValueError("language_probs must be a mapping")
    _reject_unknown_keys(value, _LANGUAGE_KEYS, "language_probs")
    probs = {
        language: _finite_number(value.get(language, 0), f"language_probs.{language}")
        for language in ("zh", "en", "other")
    }
    return LanguageSamplingConfig(probs=_probs_tuple(_stable_normalise_probs(probs)))


def _stable_normalise_probs(probs: Mapping[str, float]) -> dict[str, float]:
    if any(probability < 0 for probability in probs.values()):
        raise ValueError("language_probs values must be non-negative")
    scale = max(probs.values(), default=0.0)
    if scale <= 0:
        raise ValueError("language_probs must contain at least one positive value")
    scaled = {key: probability / scale for key, probability in probs.items()}
    total = sum(scaled.values())
    return {key: probability / total for key, probability in scaled.items()}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str] | frozenset[str], field: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{field} contains unknown keys: {sorted(unknown)}")


def _normalise_probs(value: Mapping[str, Real]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("language_probs must be a mapping")
    parsed = {}
    for key, probability in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("language_probs keys must be non-empty strings")
        if key not in _LANGUAGE_KEYS:
            raise ValueError(f"language_probs contains unknown key: {key!r}")
        parsed[key] = _finite_number(probability, f"language_probs.{key}")
    return _stable_normalise_probs(parsed)


def _probs_tuple(
    normalized: Mapping[str, float],
) -> tuple[tuple[str, Fraction], ...]:
    """把归一化 float 概率转成 exact-sum 的规范有序 Fraction 元组。"""
    keys = sorted(normalized)
    fractions = {
        language: Fraction(str(normalized[language])) for language in keys
    }
    total = sum(fractions.values(), Fraction())
    fractions = {
        language: probability / total
        for language, probability in fractions.items()
    }
    return tuple((language, fractions[language]) for language in keys)


def _validated_language_probs(
    probs: tuple[tuple[str, Fraction], ...],
) -> None:
    """校验统一内部 schema 的 ``language.probs``，防手建配置绕过 parse。"""
    if not isinstance(probs, tuple) or not probs:
        raise ValueError(
            "language.probs must be a non-empty tuple of (language, Fraction) pairs"
        )
    seen = set()
    for index, pair in enumerate(probs):
        field = f"language.probs[{index}]"
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{field} must be a (language, Fraction) tuple pair")
        key, probability = pair
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field}[0] must be a non-empty string")
        if key in seen:
            raise ValueError(f"language.probs contains duplicate key: {key!r}")
        seen.add(key)
        if key not in _LANGUAGE_KEYS:
            raise ValueError(f"{field}[0] contains unknown key: {key!r}")
        if type(probability) is not Fraction:
            raise ValueError(f"{field}[1] must be a finite Fraction")
        if probability < 0:
            raise ValueError(f"{field}[1] must be non-negative")
        try:
            finite = math.isfinite(float(probability))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{field}[1] must be a finite Fraction")
    if not any(probability > 0 for _key, probability in probs):
        raise ValueError("language.probs must contain at least one positive value")
    total = sum((probability for _key, probability in probs), Fraction())
    if total != 1:
        raise ValueError("language.probs positive probabilities must sum to 1")


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_DIFFICULTY",
    "DatasetSamplingConfig",
    "LanguageSamplingConfig",
    "NoiseSamplingConfig",
    "SamplingConfig",
    "SpeakerSamplingConfig",
    "SpeechSamplingConfig",
    "difficulty_key",
]
