"""固定 K SS 的 Layer B 三路场景。"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SSScenario:
    name: str
    noise: bool
    speech: bool

    @property
    def probability_key(self) -> str:
        return self.name


SS_SCENARIOS = (
    SSScenario("prob_target_only", False, True),
    SSScenario("prob_target_noise", True, True),
    SSScenario("prob_noise_only", True, False),
)


def validate_ss_scenarios(cfg) -> None:
    probabilities = []
    for item in SS_SCENARIOS:
        raw = getattr(cfg, item.probability_key)
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"SS {item.probability_key} must be finite within [0, 1], got {raw!r}"
            )
        probabilities.append(value)
    total = sum(probabilities)
    if abs(total - 1.0) > 1e-8:
        raise ValueError("SS prob_target_only + prob_target_noise + prob_noise_only must sum to 1")


def select_ss_scenario(cfg, rng, force_branch: str | None = None) -> tuple[SSScenario, float | None]:
    validate_ss_scenarios(cfg)
    if force_branch is not None:
        normalized = force_branch
        for item in SS_SCENARIOS:
            if item.name == normalized:
                return item, None
        raise ValueError(f"unsupported SS scene: {force_branch!r}")
    draw = float(rng.random())
    cumulative = 0.0
    for item in SS_SCENARIOS:
        cumulative += float(getattr(cfg, item.probability_key))
        if draw < cumulative:
            return item, draw
    return SS_SCENARIOS[-1], draw
