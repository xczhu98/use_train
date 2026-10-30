"""共享的多源语音 timeline：一次规划、无随机重放、显式活动统计。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


TIMELINE_MODES = ("head", "tail", "middle", "crossing", "containment")


def _sample_timeline_mode(cfg, rng: np.random.Generator) -> str:
    """按配置权重抽取多源相对窗口模式。"""
    weights = getattr(cfg, "timeline_mode_weights", None)
    if weights is None:
        return str(rng.choice(TIMELINE_MODES))

    values = np.asarray(
        [float(weights[mode]) for mode in TIMELINE_MODES], dtype=np.float64
    )
    total = float(values.sum())
    if not np.isfinite(values).all() or np.any(values < 0.0) or total <= 0.0:
        raise ValueError(
            "timeline_mode_weights must contain finite non-negative values"
        )
    if np.allclose(values, values[0]) and values[0] > 0.0:
        # 保持默认均匀配置的历史 RNG 路径，避免无必要地改变旧实验序列。
        return str(rng.choice(TIMELINE_MODES))
    return str(rng.choice(TIMELINE_MODES, p=values / total))


def scene_rng(dataset, index: int, stream: int) -> np.random.Generator:
    """共享动态路径的 seed/split/epoch/index/stream 入口，不依赖 worker。

    在线与离线使用同一熵顺序；验证冻结 epoch，显式更换 seed 仍会改变场景。
    """
    validation = getattr(dataset, "_simulation_split", "train") == "validation"
    return np.random.default_rng(np.random.SeedSequence([
        int(dataset._worker_base_seed[0]) % (2 ** 64),
        0 if validation else int(dataset._worker_epoch[0]) % (2 ** 64),
        int(index), int(stream), int(validation),
    ]))


@dataclass(frozen=True)
class SpeechTurn:
    source_start: int
    scene_start: int
    length: int


@dataclass(frozen=True)
class SpeechTimeline:
    source_length: int
    scene_length: int
    turns: tuple[SpeechTurn, ...]

    def apply(self, waveform: np.ndarray) -> np.ndarray:
        """仅搬运计划中的片段，不重复短语音，不额外抽样。"""
        source = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if len(source) != self.source_length:
            raise ValueError("timeline source length changed after planning")
        out = np.zeros(self.scene_length, dtype=np.float32)
        previous_end = 0
        for turn in self.turns:
            if (turn.source_start < 0 or turn.scene_start < previous_end
                    or turn.length <= 0
                    or turn.source_start + turn.length > len(source)
                    or turn.scene_start + turn.length > len(out)):
                raise ValueError("invalid or overlapping speech timeline turns")
            out[turn.scene_start:turn.scene_start + turn.length] = source[
                turn.source_start:turn.source_start + turn.length
            ]
            previous_end = turn.scene_start + turn.length
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def plan_overlap_window(
    source_length: int,
    scene_length: int,
    *,
    ratio: float,
    mode: str,
    rng: np.random.Generator,
) -> SpeechTimeline:
    """对已有 Layer A 语音只规划一次相对窗口，不再插入随机 gap 或重复源。"""
    if source_length < 0 or scene_length < 1 or not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("invalid overlap window length/ratio")
    if mode not in {"head", "tail", "middle", "crossing", "containment"}:
        raise ValueError(f"unknown timeline mode: {mode}")
    length = min(source_length, round(ratio * scene_length))
    if length == 0:
        return SpeechTimeline(source_length, scene_length, ())
    if mode == "head":
        start = 0
    elif mode == "tail":
        start = scene_length - length
    elif mode == "crossing":
        start = 0 if rng.random() < 0.5 else scene_length - length
    else:
        start = int(rng.integers(0, scene_length - length + 1))
    offset = (int(rng.integers(0, source_length - length + 1))
              if mode in {"crossing", "containment"} else 0)
    return SpeechTimeline(source_length, scene_length, (SpeechTurn(offset, start, length),))


def place_scene_sources(sources, cfg, rng):
    """多源共享关系规划；单源和关闭状态均不消费 RNG。"""
    if not isinstance(sources, (tuple, list)):
        raise TypeError("scene sources must be a tuple or list of one-dimensional waveforms")
    source_arrays = tuple(np.asarray(source) for source in sources)
    invalid_shapes = [array.shape for array in source_arrays if array.ndim != 1]
    if invalid_shapes:
        raise ValueError(
            "scene timeline sources must be one-dimensional [T] waveforms; "
            f"got shapes={invalid_shapes!r}"
        )
    probability = float(getattr(cfg, "prob_timeline", 0.0))
    if len(source_arrays) < 2 or probability == 0.0 or rng.random() >= probability:
        return tuple(sources), ()
    placed = [source_arrays[0]]
    plans = []
    for source in source_arrays[1:]:
        local = np.asarray(source, dtype=np.float32).reshape(-1)
        active = np.flatnonzero(activity_mask(local))
        offset = int(active[0]) if active.size else 0
        local = local[offset:int(active[-1]) + 1] if active.size else local[:0]
        ratio = float(rng.uniform(*cfg.overlap_ratio_range))
        mode = _sample_timeline_mode(cfg, rng)
        plan = plan_overlap_window(
            len(local),
            np.size(source_arrays[0]),
            ratio=ratio,
            mode=mode,
            rng=rng,
        )
        placed.append(plan.apply(local).reshape(np.shape(source_arrays[0])))
        plans.append(
            {
                **plan.to_dict(),
                "mode": mode,
                "requested_ratio": ratio,
                "source_trim_offset": offset,
            }
        )
    return tuple(placed), tuple(plans)


def activity_mask(waveform: np.ndarray) -> np.ndarray:
    """相对峰值 -40 dB 的逐样点活动估计；仅供仿真审计，不声明 VAD 精度。"""
    values = np.abs(np.asarray(waveform, dtype=np.float32).reshape(-1))
    peak = float(values.max()) if values.size else 0.0
    return values > max(peak * 0.01, 1e-8)


def scene_activity_facts(sources: tuple[np.ndarray, ...]) -> dict:
    if not sources or len({np.size(source) for source in sources}) != 1:
        raise ValueError("scene activity requires non-empty aligned sources")
    masks = np.stack([activity_mask(source) for source in sources])
    return {
        "measure": "sample_relative_peak_minus_40db",
        "active_ratios": masks.mean(axis=1).tolist(),
        "overlap_ratio": float((masks.sum(axis=0) >= 2).mean()),
        "active_source_count": int(masks.any(axis=1).sum()),
    }
