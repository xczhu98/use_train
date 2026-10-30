"""Builder-level guards for optional dynamic balanced sampling."""

from __future__ import annotations

from copy import deepcopy

from fractions import Fraction

import pytest

from tasks.aec.data import build_dataset as build_aec_dataset
from tasks.aec.data.dynamic_aec_build import resolve_aec_dataset_kwargs
from core.data.sampling import SamplingConfig
from tasks.se.data import build_dataset as build_se_dataset
from tasks.tse.data import build_dataset as build_tse_dataset


class DatasetSpy:
    """Records constructor kwargs without opening any configured audio sources."""

    calls: list[dict] = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)

    def set_epoch(self, *args, **kwargs):
        return None


def _se_cfg(*, mode: str = "dynamic", sampling=None) -> dict:
    train = (
        {"clean_scp": "clean.scp", "noisy_scp": "noisy.scp"}
        if mode == "offline"
        else {
            "speech_scp": "speech.jsonl" if mode == "dynamic_tse" else "speech.scp",
            "noise_scp": "noise.scp",
            "simulation_cfg": {},
        }
    )
    if mode == "dynamic_tse":
        train["spk2enroll_json"] = "enroll.json"
    if sampling is not None:
        train["sampling"] = sampling
    return {
        "mode": mode,
        "sample_rate": 16000,
        "train": train,
        "validation": {"clean_scp": "clean.scp", "noisy_scp": "noisy.scp"},
    }


def _dynamic_validation_cfg(*, inherit_simulation: bool, validation_sampling="missing") -> dict:
    cfg = _se_cfg(sampling={"speech": {"language_probs": {"zh": 1}}})
    cfg["validation"] = {
        "mode": "dynamic",
        "inherit_simulation": inherit_simulation,
        "speech_scp": "val_speech.scp",
    }
    if validation_sampling != "missing":
        cfg["validation"]["sampling"] = validation_sampling
    return cfg


def _aec_specialized_cfg(*, sampling=None) -> dict:
    train = {"real_farend_echo_pair_json": "echo.json", "simulation_cfg": {}}
    if sampling is not None:
        train["sampling"] = sampling
    return {
        "data": {
            "mode": "dynamic_aec",
            "sample_rate": 16000,
            "train": train,
            "validation": {"real_farend_echo_pair_json": "val_echo.json", "simulation_cfg": {}},
        }
    }


def _patch_dataset(monkeypatch, module_path: str):
    DatasetSpy.calls = []
    monkeypatch.setattr(module_path, DatasetSpy)


def test_se_dynamic_parses_nonempty_sampling_and_passes_sampling_cfg(monkeypatch):
    _patch_dataset(monkeypatch, "tasks.se.data.DynamicMixingDataset")

    build_se_dataset(_se_cfg(sampling={"speech": {"language_probs": {"zh": 1}}}))

    kwargs = DatasetSpy.calls[-1]
    assert isinstance(kwargs["sampling_cfg"], SamplingConfig)
    assert kwargs["sampling_cfg"].speech.language.probs == (
        ("en", Fraction(0)),
        ("other", Fraction(0)),
        ("zh", Fraction(1)),
    )
    assert "sampling" not in kwargs


def test_se_offline_rejects_nonempty_sampling_as_dynamic_only():
    with pytest.raises(ValueError, match="dynamic-only"):
        build_se_dataset(_se_cfg(mode="offline", sampling={"speech": {"language_probs": {"zh": 1}}}))


def test_shared_aec_builder_parses_nonempty_sampling(monkeypatch):
    _patch_dataset(monkeypatch, "tasks.aec.data.DynamicMixingDataset")

    build_aec_dataset(_se_cfg(mode="dynamic", sampling={"speech": {"language_probs": {"zh": 1}}}))

    kwargs = DatasetSpy.calls[-1]
    assert isinstance(kwargs["sampling_cfg"], SamplingConfig)
    assert kwargs["sampling_cfg"].speech.language.probs == (
        ("en", Fraction(0)),
        ("other", Fraction(0)),
        ("zh", Fraction(1)),
    )


@pytest.mark.parametrize("inherit_simulation", [True, False])
def test_shared_aec_validation_strips_inherited_num_per_epoch(monkeypatch, inherit_simulation):
    _patch_dataset(monkeypatch, "tasks.aec.data.DynamicMixingDataset")
    cfg = _dynamic_validation_cfg(inherit_simulation=inherit_simulation)
    cfg["train"]["num_per_epoch"] = 123

    build_aec_dataset(cfg, mode="val")

    assert "num_per_epoch" not in DatasetSpy.calls[-1]


@pytest.mark.parametrize(
    ("task", "controller", "accepted"),
    [
        ("se", "language", True), ("se", "dataset", True), ("se", "speaker", True),
        ("aec", "language", True), ("aec", "dataset", True), ("aec", "speaker", True),
        ("tse", "language", True), ("tse", "dataset", True), ("tse", "speaker", True),
        ("se", "noise", True), ("aec", "noise", True), ("tse", "noise", True),
    ],
)
def test_speech_controller_capability(monkeypatch, task, controller, accepted):
    """Task 6 Step 1：逐任务逐 controller 的 capability 矩阵。"""
    builders = {
        "se": (build_se_dataset, "tasks.se.data.DynamicMixingDataset", "dynamic"),
        "aec": (build_aec_dataset, "tasks.aec.data.DynamicMixingDataset", "dynamic"),
        "tse": (build_tse_dataset, "tasks.tse.data.TSEDynamicMixingDataset", "dynamic_tse"),
    }
    build, spy_path, mode = builders[task]
    _patch_dataset(monkeypatch, spy_path)
    blocks = {
        "language": {"speech": {"language_probs": {"zh": 1}}},
        "dataset": {"speech": {"dataset_alpha": 0.5}},
        "speaker": {"speech": {"speaker": {"max_samples_per_epoch": 10}}},
        "noise": {"noise": {"balance_categories": True}},
    }
    cfg = _se_cfg(mode=mode, sampling=blocks[controller])
    if accepted:
        build(cfg)
        assert isinstance(DatasetSpy.calls[-1]["sampling_cfg"], SamplingConfig)
    else:
        with pytest.raises(ValueError, match="sampling.noise is not supported for TSE"):
            build(cfg)
        assert DatasetSpy.calls == []


def test_specialized_aec_builder_accepts_speaker():
    """AEC specialized 路径开放 canonical speaker controller。"""
    kwargs = resolve_aec_dataset_kwargs(
        _aec_specialized_cfg(
            sampling={"speech": {"speaker": {"max_samples_per_epoch": 3}}}
        ),
        "train",
    )

    assert isinstance(kwargs["sampling_cfg"], SamplingConfig)
    assert kwargs["sampling_cfg"].speech.speaker.max_samples_per_epoch == 3


def test_specialized_aec_validation_strips_explicit_effective_empty_sampling():
    """AEC specialized 路径 validation 显式 effective-empty 剥离且不报错。"""
    cfg = _aec_specialized_cfg(sampling={"speech": {"language_probs": {"zh": 1}}})
    cfg["data"]["validation"]["sampling"] = {"speech": {}}

    kwargs = resolve_aec_dataset_kwargs(cfg, "val")

    assert "sampling" not in kwargs
    assert "sampling_cfg" not in kwargs


def test_tse_accepts_noise_sampling_with_explicit_metadata_controller(monkeypatch):
    _patch_dataset(monkeypatch, "tasks.tse.data.TSEDynamicMixingDataset")
    build_tse_dataset(
        _se_cfg(mode="dynamic_tse", sampling={"noise": {"balance_categories": True}})
    )
    sampling_cfg = DatasetSpy.calls[-1]["sampling_cfg"]
    assert isinstance(sampling_cfg, SamplingConfig)
    assert sampling_cfg.noise.balance_categories is True


@pytest.mark.parametrize(
    "sampling",
    [{}, {"speech": {}}, {"noise": {"balance_categories": False}}],
)
def test_effective_empty_sampling_passes_no_sampling_cfg(monkeypatch, sampling):
    """Task 6 Step 2：effective-empty 归一为无有效 sampling（dynamic 不报错、不传参）。"""
    _patch_dataset(monkeypatch, "tasks.se.data.DynamicMixingDataset")

    build_se_dataset(_se_cfg(sampling=sampling))

    kwargs = DatasetSpy.calls[-1]
    assert "sampling" not in kwargs
    assert "sampling_cfg" not in kwargs


@pytest.mark.parametrize(
    "sampling",
    [{}, {"speech": {}}, {"noise": {"balance_categories": False}}],
)
def test_se_offline_effective_empty_sampling_is_not_rejected(monkeypatch, sampling):
    """effective-empty 不应触发 builder 前置拒绝。"""
    _patch_dataset(monkeypatch, "core.data.dataset_build.OfflineDataset")

    build_se_dataset(_se_cfg(mode="offline", sampling=sampling))

    assert len(DatasetSpy.calls) == 1
    assert DatasetSpy.calls[0]["clean_scp"] == "clean.scp"
    assert DatasetSpy.calls[0]["noisy_scp"] == "noisy.scp"


def test_se_offline_speaker_reports_dynamic_only_after_capability_acceptance():
    """SE speaker 合法但仍只允许 dynamic training。"""
    with pytest.raises(ValueError, match="dynamic-only"):
        build_se_dataset(
            _se_cfg(
                mode="offline",
                sampling={"speech": {"speaker": {"max_samples_per_epoch": 5}}},
            )
        )


@pytest.mark.parametrize("inherit_simulation", [True, False])
@pytest.mark.parametrize(
    "validation_sampling",
    [{"speech": {}}, {"noise": {"balance_categories": False}}],
)
def test_dynamic_validation_strips_explicit_effective_empty_sampling(
    monkeypatch, validation_sampling, inherit_simulation
):
    """Task 6 Step 3：validation 显式 effective-empty 配置剥离且不报错。"""
    _patch_dataset(monkeypatch, "tasks.se.data.DynamicMixingDataset")
    cfg = _dynamic_validation_cfg(
        inherit_simulation=inherit_simulation, validation_sampling=validation_sampling
    )

    build_se_dataset(cfg, mode="val")

    kwargs = DatasetSpy.calls[-1]
    assert "sampling" not in kwargs
    assert "sampling_cfg" not in kwargs


@pytest.mark.parametrize(
    "validation_sampling",
    [
        {"speech": {"language_probs": {"en": 1}}},
        {"speech": {"dataset_alpha": 0.5}},
        {"speech": {"speaker": {"max_samples_per_epoch": 3}}},
        {"noise": {"balance_categories": True}},
    ],
)
def test_dynamic_validation_rejects_any_explicit_effective_controller(
    monkeypatch, validation_sampling
):
    """Task 6 Step 3：validation 原 section 显式有效 controller → capability error。"""
    _patch_dataset(monkeypatch, "tasks.se.data.DynamicMixingDataset")
    cfg = _dynamic_validation_cfg(
        inherit_simulation=True, validation_sampling=validation_sampling
    )

    with pytest.raises(ValueError, match="validation.*sampling.*not supported"):
        build_se_dataset(cfg, mode="val")

    assert DatasetSpy.calls == []


def test_specialized_aec_builder_parses_nonempty_sampling():
    kwargs = resolve_aec_dataset_kwargs(
        _aec_specialized_cfg(
            sampling={
                "speech": {"language_probs": {"zh": 1}},
                "noise": {"balance_categories": True},
            }
        ),
        "train",
    )

    assert isinstance(kwargs["sampling_cfg"], SamplingConfig)
    assert kwargs["sampling_cfg"].speech.language.probs == (
        ("en", Fraction(0)),
        ("other", Fraction(0)),
        ("zh", Fraction(1)),
    )
    assert kwargs["sampling_cfg"].noise.balance_categories is True


def test_specialized_aec_builder_strips_num_per_epoch_from_dataset_kwargs():
    cfg = _aec_specialized_cfg()
    cfg["data"]["train"]["num_per_epoch"] = 123

    kwargs = resolve_aec_dataset_kwargs(cfg, "train")

    assert "num_per_epoch" not in kwargs


def test_specialized_aec_validation_strips_inherited_sampling_after_merge():
    cfg = _aec_specialized_cfg(sampling={"speech": {"language_probs": {"zh": 1}}})
    original = deepcopy(cfg)

    kwargs = resolve_aec_dataset_kwargs(cfg, "val")

    assert "sampling" not in kwargs
    assert "sampling_cfg" not in kwargs
    assert cfg == original


def test_specialized_aec_validation_rejects_explicit_nonempty_sampling():
    cfg = _aec_specialized_cfg(sampling={"speech": {"language_probs": {"zh": 1}}})
    cfg["data"]["validation"]["sampling"] = {"speech": {"language_probs": {"en": 1}}}

    with pytest.raises(ValueError, match="validation.*sampling.*not supported"):
        resolve_aec_dataset_kwargs(cfg, "val")


def test_specialized_aec_noninherited_validation_does_not_mutate_nested_simulation():
    cfg = _aec_specialized_cfg()
    cfg["data"]["validation"] = {
        "inherit_simulation": False,
        "real_farend_echo_pair_json": "val_echo.json",
        "simulation_cfg": {"aec_ref_echo_random_delay_prob": 0.25},
    }
    original = deepcopy(cfg)

    kwargs = resolve_aec_dataset_kwargs(cfg, "val")

    assert "sampling" not in kwargs
    assert kwargs["aec_ref_echo_random_delay_prob"] == 0.25
    assert cfg == original


@pytest.mark.parametrize("inherit_simulation", [True, False])
def test_dynamic_validation_strips_sampling_inherited_from_train(monkeypatch, inherit_simulation):
    _patch_dataset(monkeypatch, "tasks.se.data.DynamicMixingDataset")
    cfg = _dynamic_validation_cfg(inherit_simulation=inherit_simulation)
    original = deepcopy(cfg)

    build_se_dataset(cfg, mode="val")

    kwargs = DatasetSpy.calls[-1]
    assert "sampling" not in kwargs
    assert "sampling_cfg" not in kwargs
    assert cfg == original


@pytest.mark.parametrize("inherit_simulation", [True, False])
def test_dynamic_validation_rejects_explicit_nonempty_sampling(monkeypatch, inherit_simulation):
    _patch_dataset(monkeypatch, "tasks.se.data.DynamicMixingDataset")
    cfg = _dynamic_validation_cfg(
        inherit_simulation=inherit_simulation,
        validation_sampling={"speech": {"language_probs": {"en": 1}}},
    )

    with pytest.raises(ValueError, match="validation.*sampling.*not supported"):
        build_se_dataset(cfg, mode="val")

    assert DatasetSpy.calls == []


def test_offline_validation_rejects_explicit_nonempty_sampling(monkeypatch):
    _patch_dataset(monkeypatch, "core.data.dataset_build.OfflineDataset")
    cfg = _se_cfg(sampling={"speech": {"language_probs": {"zh": 1}}})
    cfg["validation"]["sampling"] = {"speech": {"language_probs": {"en": 1}}}

    with pytest.raises(ValueError, match="validation.*sampling.*not supported"):
        build_se_dataset(cfg, mode="val")

    assert DatasetSpy.calls == []


@pytest.mark.parametrize("validation_sampling", [None, {}])
def test_validation_explicit_empty_sampling_keeps_legacy_behavior(monkeypatch, validation_sampling):
    _patch_dataset(monkeypatch, "tasks.se.data.DynamicMixingDataset")
    cfg = _dynamic_validation_cfg(
        inherit_simulation=True,
        validation_sampling=validation_sampling,
    )

    build_se_dataset(cfg, mode="val")

    kwargs = DatasetSpy.calls[-1]
    assert "sampling" not in kwargs
    assert "sampling_cfg" not in kwargs


@pytest.mark.parametrize("sampling", [None, {}])
def test_empty_or_missing_sampling_keeps_legacy_dynamic_kwargs(monkeypatch, sampling):
    _patch_dataset(monkeypatch, "tasks.se.data.DynamicMixingDataset")

    build_se_dataset(_se_cfg(sampling=sampling))

    kwargs = DatasetSpy.calls[-1]
    assert "sampling" not in kwargs
    assert "sampling_cfg" not in kwargs


def test_invalid_se_sampling_uses_strict_sampling_parser(monkeypatch):
    _patch_dataset(monkeypatch, "tasks.se.data.DynamicMixingDataset")

    with pytest.raises(ValueError, match="sampling must be a mapping"):
        build_se_dataset(_se_cfg(sampling="enabled"))

    assert DatasetSpy.calls == []


pytestmark = [pytest.mark.unit]
