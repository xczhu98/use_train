"""TSE Layer B 七路场景；存在性只由一次场景抽样决定。"""
from __future__ import annotations
from dataclasses import dataclass
import math
from collections.abc import Mapping

@dataclass(frozen=True)
class TSEScenario:
    name: str
    target: bool
    interference: bool
    noise: bool

    @property
    def probability_key(self) -> str:
        return self.name


TSE_SCENARIOS = (
    TSEScenario("prob_target_only", True, False, False),
    TSEScenario("prob_interference_only", False, True, False),
    TSEScenario("prob_noise_only", False, False, True),
    TSEScenario("prob_interference_noise_only", False, True, True),
    TSEScenario("prob_target_noise", True, False, True),
    TSEScenario("prob_target_interference", True, True, False),
    TSEScenario("prob_target_interference_noise", True, True, True),
)
TSE_SCENARIO_KEYS = frozenset((s.probability_key for s in TSE_SCENARIOS))
REMOVED_TSE_KEYS = frozenset({'prob_interference', 'interfere_num_distribution', 'interfere_timeline_prob', 'interfere_overlap_ratio_range', 'interference_count_distribution', 'interference_shortage_policy', 'prob_target_absent', 'prob_speed_perturb_enroll', 'prob_pitch_shift_enroll', 'enroll_reverb_prob', 'enroll_noise_prob', 'enroll_snr_range', 'enroll_eq_prob'})
REMOVED_SHARED_KEYS = frozenset({'source_rir_mode', 'scene_snr_reference', 'timeline_overlap_prob', 'timeline_overlap_ratio_range', 'noise_only_db_range', 'rir_overlap_sec'})


def prepare_tse_simulation_cfg(simulation_cfg):
    """校验 TSE canonical 场景字段；本函数只校验，不投影到内部字段。"""
    if simulation_cfg is None or not isinstance(simulation_cfg, Mapping):
        return simulation_cfg
    cfg = dict(simulation_cfg)
    if "prob_interference_noise" in cfg:
        raise ValueError(
            "prob_interference_noise is removed for TSE; "
            "use prob_interference_noise_only"
        )
    return cfg

def validate_tse_scenarios(cfg):
    probabilities = []
    for scenario in TSE_SCENARIOS:
        raw = getattr(cfg, scenario.probability_key)
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"TSE {scenario.probability_key} must be finite within [0, 1], got {raw!r}"
            )
        probabilities.append(value)
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-08):
        raise ValueError('TSE seven scenario probabilities must sum to 1')
    if any((getattr(cfg, s.probability_key) > 0 and s.interference for s in TSE_SCENARIOS)) and cfg.sir_range is None:
        raise ValueError('interference scenarios require sir_range')
    if cfg.peak_normalization_mode not in {'off', 'joint'}:
        raise ValueError('TSE supports only off/joint peak normalization')
    if cfg.rir_backend == 'online':
        max_interferers = max(k for k, p in cfg.num_interferers.items() if p > 0)
        required = max(
            int(s.target) + (max_interferers if s.interference else 0)
            for s in TSE_SCENARIOS
            if getattr(cfg, s.probability_key) > 0
        )
        if required > cfg.gpurir_max_sources:
            raise ValueError('TSE scene exceeds gpurir_max_sources budget')

def select_tse_scenario(cfg, rng, force_branch=None):
    validate_tse_scenarios(cfg)
    if force_branch is not None:
        for scenario in TSE_SCENARIOS:
            if scenario.name == force_branch:
                return (scenario, None)
        raise ValueError(f'unsupported TSE scene: {force_branch!r}')
    draw = float(rng.random())
    cumulative = 0.0
    for scenario in TSE_SCENARIOS:
        cumulative += getattr(cfg, scenario.probability_key)
        if draw < cumulative:
            return (scenario, draw)
    return (TSE_SCENARIOS[-1], draw)
