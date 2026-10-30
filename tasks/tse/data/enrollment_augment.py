"""TSE 注册音的无状态、长度保持增强。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np

_CONFIG_VERSION = 1


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"enrollment_augment.{name} must be a finite number")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"enrollment_augment.{name} must be finite")
    return parsed


_ALLOWED_KEYS = frozenset({
    "enabled", "prob_speed", "speed_rate_range", "prob_pitch", "semitone_range",
    "prob_eq", "prob_eq_post", "eq_db_range", "target_duration_seconds", "target_db", "fit_policy",
    "prob_reverb", "rir_normalize", "rir_backend", "rir_scp", "prob_noise", "noise_scp", "snr_range",
    "prob_lowpass", "lowpass_cutoff_range", "rir_static_only",
})


def _probability(value: object, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"enrollment_augment.{name} must be in [0, 1]")
    return parsed


def _parse_range(value: object, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"enrollment_augment.{name} must be a two-item [low, high] list"
        )
    low = _finite_float(value[0], name=f"{name}[0]")
    high = _finite_float(value[1], name=f"{name}[1]")
    if low > high:
        raise ValueError(f"enrollment_augment.{name} must satisfy low <= high")
    return low, high


@dataclass(frozen=True)
class EnrollmentAugmentor:
    """TSE-local 无状态 enrollment 波形增强器。

    ``rir_provider`` 和 ``noise_provider`` 是 Dataset 注入的 I/O 边界。

    近场契约：注册音频在物理上是近场录制（说话人距麦克风 5-20cm），RIR 扰动
    应保持这一事实。``rir_backend="offline"`` 时由配置选择近场 RIR 池
    （如 ``static/near`` 子集；现网该池 dist 约 0.2-0.8m，池选择即契约实现）；
    ``rir_backend="online"`` 时声源距离从 ``gpurir_enrollment_distance_range_m``
    （默认 [0.05, 0.2]）采样，房间/RT60/麦克风高度等几何参数与 mixture 共用
    ``gpurir_*`` 配置族。
    """

    enabled: bool = False
    prob_speed: float = 0.0
    speed_rate_range: tuple[float, float] = (0.8, 1.2)
    prob_pitch: float = 0.0
    semitone_range: tuple[float, float] = (-1.5, 1.5)
    prob_eq: float = 1.0
    prob_eq_post: float = 0.0
    eq_db_range: tuple[float, float] = (-6.0, 6.0)
    target_duration_seconds: float | None = None
    target_db: float | None = None
    fit_policy: str = "random"
    prob_reverb: float = 0.0
    rir_normalize: bool = True
    rir_backend: str = "offline"
    rir_static_only: bool = False
    prob_noise: float = 0.0
    snr_range: tuple[float, float] = (5.0, 30.0)
    prob_lowpass: float = 0.5
    lowpass_cutoff_range: tuple[float, float] = (1000.0, 8000.0)
    preserve_input_length: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> "EnrollmentAugmentor":
        """严格解析 offline/local ``enrollment_augment`` mapping。"""
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise TypeError(
                "enrollment_augment must be a mapping or null, "
                f"got {type(config).__name__}"
            )
        unknown = sorted(set(config) - _ALLOWED_KEYS)
        if unknown:
            raise ValueError(f"enrollment_augment has unknown keys: {unknown}")
        enabled = config.get("enabled", False)
        if type(enabled) is not bool:
            raise TypeError("enrollment_augment.enabled must be a bool")
        if type(config.get("rir_static_only", False)) is not bool:
            raise TypeError("enrollment_augment.rir_static_only must be a bool")
        fit_policy = config.get("fit_policy", "random")
        if fit_policy not in {"random", "fixed"}:
            raise ValueError("enrollment_augment.fit_policy must be 'random' or 'fixed'")
        rir_normalize = config.get("rir_normalize", True)
        if type(rir_normalize) is not bool:
            raise TypeError("enrollment_augment.rir_normalize must be a bool")
        rir_backend = config.get("rir_backend", "offline")
        if rir_backend not in {"offline", "online"}:
            raise ValueError(
                "enrollment_augment.rir_backend must be 'offline' or 'online'"
            )
        if rir_backend == "online" and config.get("rir_scp"):
            raise ValueError(
                "enrollment_augment.rir_backend='online' does not take rir_scp; "
                "online generation uses the gpurir_* geometry config"
            )
        duration = config.get("target_duration_seconds")
        if duration is not None and _finite_float(duration, name="target_duration_seconds") <= 0:
            raise ValueError("enrollment_augment.target_duration_seconds must be > 0")
        target_db = config.get("target_db")
        return cls(
            enabled=enabled,
            prob_speed=_probability(config.get("prob_speed", 0.0), name="prob_speed"),
            speed_rate_range=_parse_range(config.get("speed_rate_range", [0.8, 1.2]), name="speed_rate_range"),
            prob_pitch=_probability(config.get("prob_pitch", 0.0), name="prob_pitch"),
            semitone_range=_parse_range(config.get("semitone_range", [-1.5, 1.5]), name="semitone_range"),
            prob_eq=_probability(config.get("prob_eq", 1.0), name="prob_eq"),
            prob_eq_post=_probability(config.get("prob_eq_post", 0.0), name="prob_eq_post"),
            eq_db_range=_parse_range(config.get("eq_db_range", [-6.0, 6.0]), name="eq_db_range"),
            target_duration_seconds=(None if duration is None else float(duration)),
            target_db=(None if target_db is None else _finite_float(target_db, name="target_db")),
            fit_policy=fit_policy,
            prob_reverb=_probability(config.get("prob_reverb", 0.0), name="prob_reverb"),
            rir_normalize=rir_normalize,
            rir_backend=rir_backend,
            rir_static_only=config.get("rir_static_only", False),
            prob_noise=_probability(config.get("prob_noise", 0.0), name="prob_noise"),
            snr_range=_parse_range(config.get("snr_range", [5.0, 30.0]), name="snr_range"),
            prob_lowpass=_probability(config.get("prob_lowpass", 0.5), name="prob_lowpass"),
            lowpass_cutoff_range=_parse_range(config.get("lowpass_cutoff_range", [1000.0, 8000.0]), name="lowpass_cutoff_range"),
        )

    def for_dynamic(self, *, duration_seconds, target_db, fit_policy):
        """动态数据始终做长度/RMS；enabled 只控制可选扰动。"""
        from dataclasses import replace

        gates = {} if self.enabled else {
            name: 0.0 for name in self.__dataclass_fields__ if name.startswith("prob_")
        }
        return replace(self, enabled=True, preserve_input_length=False,
                       target_duration_seconds=self.target_duration_seconds or duration_seconds,
                       target_db=target_db if self.target_db is None else self.target_db,
                       fit_policy=fit_policy, **gates)

    def requires_rir_provider(self) -> bool:
        return self.enabled and self.prob_reverb > 0.0

    def requires_noise_provider(self) -> bool:
        return self.enabled and self.prob_noise > 0.0

    @staticmethod
    def _fit_length(enrollment: np.ndarray, n_target: int, rng: np.random.Generator, policy: str) -> np.ndarray:
        n = len(enrollment)
        if n == n_target:
            return enrollment.astype(np.float32, copy=False)
        if n == 0:
            return np.zeros(n_target, dtype=np.float32)
        if n < n_target:
            tiled = np.tile(enrollment, (n_target + n - 1) // n)[:n_target]
            shift = int(rng.integers(0, n_target + 1)) if policy != "fixed" and n_target > 1 else 0
            return np.roll(tiled, shift).astype(np.float32)
        start = 0 if policy == "fixed" else int(rng.integers(0, n - n_target + 1))
        return enrollment[start:start + n_target].astype(np.float32)

    @staticmethod
    def _rms_normalize(waveform: np.ndarray, target_db: float) -> np.ndarray:
        if waveform.size == 0:
            return waveform.astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)) + 1e-10))
        gain = max(10.0 ** (float(target_db) / 20.0), 1e-10) / rms
        peak = float(np.max(np.abs(waveform)))
        if peak * gain > 1.0:
            gain = 1.0 / max(peak, 1e-10)
        return (waveform * gain).astype(np.float32)

    @staticmethod
    def _peak_guard(waveform: np.ndarray, threshold: float = 0.99) -> tuple[np.ndarray, bool]:
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > threshold:
            return (waveform * (threshold / peak)).astype(np.float32), True
        return waveform.astype(np.float32, copy=False), False

    def _gate(self, probability: float, rng: np.random.Generator) -> bool:
        return probability > 0.0 and float(rng.random()) < probability

    def apply(
        self,
        enrollment: np.ndarray,
        sample_rate: int,
        rng: np.random.Generator | None = None,
        *,
        rir_provider=None,
        noise_provider=None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """执行完整链；provider 返回的 RIR/noise 均为单声道二维数组。"""
        if not isinstance(enrollment, np.ndarray):
            raise TypeError("enrollment must be a numpy.ndarray")
        if enrollment.ndim != 1 or enrollment.dtype != np.float32:
            raise ValueError(f"enrollment must be a 1-D float32 ndarray, got shape={enrollment.shape}, dtype={enrollment.dtype}")
        if not np.isfinite(enrollment).all():
            raise ValueError("enrollment must contain only finite values")
        if int(sample_rate) <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
        facts: dict[str, object] = {"version": _CONFIG_VERSION, "enabled": self.enabled, "applied": False, "operation": None}
        if not self.enabled:
            return enrollment.copy(), facts
        if rng is None:
            raise ValueError("enabled enrollment_augment requires a NumPy RNG")
        from core.data.augment import pitch_shift, random_eq, speed_perturb
        from core.data.post_augment import lowpass_filter
        from core.data.simulation.engine import add_reverberation, mix_noise

        waveform = enrollment.copy()
        operations: list[str] = []
        if self._gate(self.prob_speed, rng):
            waveform = speed_perturb(waveform, sample_rate, *self.speed_rate_range, rng=rng)
            operations.append("speed")
        if self._gate(self.prob_pitch, rng):
            waveform = pitch_shift(waveform, sample_rate, *self.semitone_range, rng=rng)
            operations.append("pitch")
        if self._gate(self.prob_eq, rng):
            waveform = random_eq(waveform, sample_rate, *self.eq_db_range, rng=rng)
            operations.append("eq_pre")
        n_target = (max(1, int(round(self.target_duration_seconds * sample_rate))) if self.target_duration_seconds is not None else len(waveform))
        if self.target_duration_seconds is not None:
            waveform = self._fit_length(waveform, n_target, rng, self.fit_policy)
        if self.target_db is not None:
            waveform = self._rms_normalize(waveform, self.target_db)
            operations.append("rms")
        if self._gate(self.prob_reverb, rng):
            if rir_provider is None:
                raise ValueError("enrollment_augment.prob_reverb > 0 requires an RIR provider")
            rir, rir_facts = rir_provider(sample_rate, rng)
            waveform = add_reverberation(waveform.reshape(1, -1), rir, normalize_rir=self.rir_normalize).flatten().astype(np.float32)
            facts["rir"] = rir_facts
            operations.append("rir")
        if self._gate(self.prob_noise, rng):
            if noise_provider is None:
                raise ValueError("enrollment_augment.prob_noise > 0 requires a noise provider")
            snr = float(rng.uniform(*self.snr_range))
            noise, noise_facts = noise_provider(sample_rate, len(waveform), rng)
            mixed, _ = mix_noise(waveform.reshape(1, -1), noise, snr=snr, rng=rng, audit_out=(audit := {}))
            waveform = mixed.flatten().astype(np.float32)
            facts.update({"noise": noise_facts, "snr_db": snr, "noise_audit": audit})
            operations.append("noise")
        eq_post_applied = self._gate(self.prob_eq_post, rng)
        if eq_post_applied:
            waveform = random_eq(waveform, sample_rate, *self.eq_db_range, rng=rng)
            operations.append("eq_post")
            if self._gate(self.prob_lowpass, rng):
                waveform = lowpass_filter(waveform, sample_rate, *self.lowpass_cutoff_range, rng=rng)
                operations.append("lowpass")
        waveform, peak_rescaled = self._peak_guard(waveform)
        if self.target_duration_seconds is not None:
            waveform = self._fit_length(waveform, n_target, rng, self.fit_policy)
        if self.preserve_input_length and len(waveform) != len(enrollment):
            waveform = self._fit_length(waveform, len(enrollment), rng, self.fit_policy)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise RuntimeError("enrollment augmentation produced invalid waveform")
        operation = operations[-1] if operations else None
        if operations == ["eq_pre"]:
            operation = "random_eq"
        facts.update({"applied": bool(operations), "operation": operation, "operations": operations, "peak_guard_rescaled": peak_rescaled, "eq_post_applied": eq_post_applied})
        return waveform, facts


__all__ = ["EnrollmentAugmentor"]
