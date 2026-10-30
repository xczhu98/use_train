import pytest
import torch

from core.data.constants import normalize_data_mode
from core.trainer.entry import detect_task
from core.data.source_separation_batch import SSBatch
from core.losses.pit import PITLoss
from core.losses.build import build_loss
from tasks.ss.data.config import SSDatasetConfig


@pytest.mark.unit
def test_ss_data_modes_route_to_ss_task():
    assert normalize_data_mode("dynamic_ss").task == "ss"
    assert normalize_data_mode("offline_ss").storage == "offline"
    assert detect_task(
        {"data": {"mode": "dynamic_ss"}, "model": {"type": "TDANet"}}
    ).task == "ss"


@pytest.mark.unit
def test_ss_dynamic_config_exposes_filter_and_sampling_controls(tmp_path):
    speech = tmp_path / "speech.jsonl"
    speech.write_text(
        '{"uid":"u0","fs":16000,"length":16000,"path":"u.wav",'
        '"language":"en","dataset_id":"set","speaker_id":"spk"}\n'
    )
    cfg = {
        "meta": {"seed": 3},
        "data": {
            "mode": "dynamic_ss",
            "sample_rate": 16000,
            "strict_load": True,
            "train": {
                "speech_scp": str(speech),
                "segment_seconds": 3.0,
                "filter_by_duration": True,
                "filter_min_duration": 1.0,
                "filter_max_duration": 10.0,
                "sampling": {"speech": {"speaker": {"max_samples_per_epoch": 2}}},
                "simulation_cfg": {
                    "prob_target_only": 1.0,
                    "prob_target_noise": 0.0,
                    "prob_noise_only": 0.0,
                },
            },
            "validation": {
                "speech_scp": str(speech),
                "segment_seconds": 3.0,
                "filter_by_duration": False,
                "simulation_cfg": {
                    "prob_target_only": 1.0,
                    "prob_target_noise": 0.0,
                    "prob_noise_only": 0.0,
                },
            },
        },
        "ss": {"num_sources": 1},
    }
    resolved = SSDatasetConfig.from_config(cfg, "train")
    assert resolved.filter_by_duration is True
    assert resolved.dynamic_kwargs["filter_min_duration"] == 1.0
    assert resolved.dynamic_kwargs["sampling_cfg"].speech.speaker.max_samples_per_epoch == 2


@pytest.mark.unit
def test_ss_simulation_contract_rejects_retired_scenario_names():
    from core.data.simulation_config_contract import validate_simulation_config_contract

    for key in ("prob_speech_only", "prob_speech_noise"):
        with pytest.raises(ValueError, match="retired SS scenario key"):
            validate_simulation_config_contract(
                task="ss",
                split="train",
                train_section={"simulation_cfg": {key: 0.0}},
            )


@pytest.mark.unit
def test_ss_batch_validates_source_axis_and_mask():
    batch = SSBatch(
        mixture=torch.zeros(2, 8),
        sources=torch.zeros(2, 2, 8),
        lengths=torch.tensor([8, 6], dtype=torch.int64),
        source_mask=torch.ones(2, 2, dtype=torch.bool),
        sr=torch.tensor([16000, 16000], dtype=torch.int64),
        source_ids=(("a", "b"), ("c", "d")),
    )
    batch.validate()
    batch.source_mask[1, 1] = False
    with pytest.raises(ValueError, match="every source"):
        batch.validate()
    with pytest.raises(ValueError, match="sources must have shape"):
        SSBatch(
            mixture=torch.zeros(2, 8),
            sources=torch.zeros(2, 8),
            lengths=torch.tensor([8, 8]),
            source_mask=torch.ones(2, 1, dtype=torch.bool),
            sr=torch.tensor([16000, 16000]),
        ).validate()


@pytest.mark.unit
def test_pit_loss_is_invariant_to_estimate_source_permutation():
    ref = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
    )
    est = ref.flip(1)
    result = PITLoss(pairwise_loss="PairwiseL1Loss").compute(est, ref)
    assert torch.equal(result.assignment, torch.tensor([[1, 0]]))
    assert result.loss.item() == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "lengths",
    [torch.tensor([0]), torch.tensor([5]), torch.tensor([4.0])],
)
def test_strict_pit_loss_rejects_invalid_lengths(lengths):
    loss = build_loss(
        {
            "type": "PITLoss",
            "strict": True,
            "args": {"pairwise_loss": "PairwiseL1Loss"},
        }
    )
    sources = torch.zeros(1, 2, 4)
    with pytest.raises((AssertionError, TypeError, ValueError), match="length"):
        loss(sources, sources, lengths=lengths)


@pytest.mark.unit
def test_strict_pit_loss_rejects_nonfinite_inputs():
    loss = build_loss(
        {
            "type": "PITLoss",
            "strict": True,
            "args": {"pairwise_loss": "PairwiseL1Loss"},
        }
    )
    estimated = torch.zeros(1, 2, 4)
    estimated[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        loss(estimated, torch.zeros_like(estimated), lengths=torch.tensor([4]))


@pytest.mark.unit
def test_ss_batch_rejects_source_id_shape_mismatch():
    batch = SSBatch(
        mixture=torch.zeros(1, 4),
        sources=torch.zeros(1, 2, 4),
        lengths=torch.tensor([4]),
        source_mask=torch.ones(1, 2, dtype=torch.bool),
        sr=torch.tensor([16000]),
        source_ids=(("only-one",),),
    )
    with pytest.raises(ValueError, match="source_ids rows"):
        batch.validate()


@pytest.mark.unit
def test_ss_config_accepts_shared_validation_metadata_contract():
    config = _offline_ss_config()
    config["data"]["validation"].update(
        {"mode": "offline_ss", "use_dynamic": False, "inherit_simulation": True}
    )
    resolved = SSDatasetConfig.from_config(config, "validation")
    assert resolved.storage == "offline"


def _offline_ss_config():
    return {
        "meta": {"seed": 7},
        "data": {
            "mode": "offline_ss",
            "sample_rate": 16000,
            "strict_load": True,
            "train": {"manifest": "train.jsonl", "max_duration_seconds": 3.0},
            "validation": {"manifest": "val.jsonl"},
        },
        "ss": {"num_sources": 2, "sir_range": [-5.0, 5.0]},
    }


@pytest.mark.unit
def test_ss_dynamic_rejects_segment_seconds_conflicting_with_max_duration_seconds():
    from tasks.ss.data import build_dataset

    def section(segment_seconds, max_duration_seconds):
        return {
            "speech_scp": "missing.speech.jsonl",
            "noise_scp": "missing.noise.jsonl",
            "rir_scp": "missing.rir.jsonl",
            "segment_seconds": segment_seconds,
            "max_duration_seconds": max_duration_seconds,
            "simulation_cfg": {
                "prob_target_only": 1.0,
                "prob_target_noise": 0.0,
                "prob_noise_only": 0.0,
            },
        }

    cfg = {
        "meta": {"seed": 1},
        "data": {
            "mode": "dynamic_ss",
            "sample_rate": 16000,
            "strict_load": True,
            "train": section(3.0, 6.0),
            "validation": section(3.0, 6.0),
        },
        "ss": {"num_sources": 1},
    }
    # 冲突在资源加载前即 fail-fast，这里用不存在的 scp 证明检查顺序。
    with pytest.raises(ValueError, match="segment_seconds conflicts"):
        build_dataset(cfg, mode="train")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutate", "error_type", "match"),
    [
        (lambda cfg: cfg["meta"].__setitem__("seed", 1.5), TypeError, "meta.seed"),
        (lambda cfg: cfg["data"].__setitem__("strict_load", "false"), TypeError, "strict_load"),
        (lambda cfg: cfg["data"]["train"].__setitem__("max_duration_seconds", float("nan")), TypeError, "finite"),
        (lambda cfg: cfg["ss"].__setitem__("sir_range", [0.0, float("inf")]), TypeError, "finite"),
        (lambda cfg: cfg["ss"].__setitem__("unexpected", 1), KeyError, "unknown"),
        (lambda cfg: cfg["data"]["train"].__setitem__("speech_scp", "wrong.jsonl"), KeyError, "unknown"),
    ],
)
def test_ss_dataset_config_rejects_coercion_nonfinite_and_unknown_keys(
    mutate, error_type, match
):
    config = _offline_ss_config()
    mutate(config)
    with pytest.raises(error_type, match=match):
        SSDatasetConfig.from_config(config, "train")
