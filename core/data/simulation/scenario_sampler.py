"""通用 Layer B 场景分支抽样器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from core.data.simulation_utils import rng_random


class ScenarioSampler:
    """按稳定有序概率表选择场景，并支持不消费 RNG 的强制分支。"""

    @staticmethod
    def select_branch(
        branches: Sequence[tuple[str, float]],
        rng: np.random.Generator,
        force_branch: str | None = None,
        normalize_force: Callable[[str], str] | None = None,
    ) -> tuple[str, float | None]:
        if not branches:
            raise ValueError("scenario branches must not be empty")
        if force_branch is not None:
            normalizer = normalize_force or (lambda value: value)
            return normalizer(force_branch), None
        draw = float(rng_random(rng))
        cumulative = 0.0
        for branch, probability in branches:
            cumulative += float(probability)
            if draw < cumulative:
                return branch, draw
        return branches[-1][0], draw


__all__ = ["ScenarioSampler"]
