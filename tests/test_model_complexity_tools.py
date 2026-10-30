"""模型复杂度共享合同与审计工具单元测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from core.utils.model_profile import (
    MOS_OFFLINE_BOUNDARY,
    NOT_MEASURED,
    make_complexity_contract,
    profile_model,
)
from tools.model_tools import audit_metrics


pytestmark = pytest.mark.unit


def _contract(**overrides):
    values = {
        "total_params": 12,
        "trainable_params": 8,
        "params_scope": "task model",
        "params_source": "runtime model.parameters()",
        "macs": 34.0,
        "macs_backend": "thop",
        "macs_input": {
            "sample_rate_hz": 16000,
            "duration_seconds": 1.0,
            "batch_size": 1,
            "channels": 1,
            "samples": 16000,
        },
        "macs_scope": "se wav->model path",
        "macs_coverage": (
            "partial/unverified; module hooks may miss functional, complex, "
            "or unsupported custom operators"
        ),
    }
    values.update(overrides)
    return make_complexity_contract(**values)


def test_complexity_contract_has_required_independent_headers():
    report = _contract()

    assert report["Params"] == {
        "total": 12,
        "trainable": 8,
        "scope": "task model",
        "source": "runtime model.parameters()",
    }
    assert set(report["MACs"]) == {
        "value",
        "backend",
        "input",
        "scope",
        "coverage",
    }
    assert report["FLOPs"]["value"] == NOT_MEASURED
    assert "MACs conversion" in report["FLOPs"]["source"]
    assert report["MOS"] == {
        "value": NOT_MEASURED,
        "boundary": MOS_OFFLINE_BOUNDARY,
    }


def test_missing_macs_never_creates_flops_value():
    report = _contract(macs=None, macs_backend=NOT_MEASURED, macs_coverage=NOT_MEASURED)

    assert report["MACs"]["value"] is None
    assert report["FLOPs"]["value"] == NOT_MEASURED


def test_profile_model_keeps_legacy_keys_and_adds_contract():
    result = profile_model(
        nn.Linear(8, 4),
        sample_rate=8,
        duration=1.0,
        warmup=0,
        repeats=1,
        quiet=True,
        params_scope="generator",
        macs_scope="test forward",
    )

    assert {
        "params",
        "trainable",
        "macs",
        "macs_error",
        "latency_ms",
        "rtf",
        "rtf_error",
    } <= set(result)
    assert result["complexity"]["Params"]["scope"] == "generator"
    assert result["complexity"]["MACs"]["input"]["waveform"] == {
        "shape": [1, 8],
        "sample_rate_hz": 8,
        "duration_seconds": 1.0,
    }
    assert result["complexity"]["MACs"]["scope"] == "test forward"
    assert result["complexity"]["FLOPs"]["value"] == NOT_MEASURED


def test_thop_success_still_marks_functional_coverage_unverified():
    pytest.importorskip("thop")

    class FunctionalOnly(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    result = profile_model(
        FunctionalOnly(),
        sample_rate=8,
        duration=1.0,
        warmup=0,
        repeats=1,
        quiet=True,
    )

    macs = result["complexity"]["MACs"]
    assert macs["backend"] == "thop"
    assert macs["value"] is not None
    assert "partial/unverified" in macs["coverage"]
    assert "functional" in macs["coverage"]


def test_audit_config_loader_merges_base_without_rewriting_yaml(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("model:\n  type: BaseModel\nstft:\n  n_fft: 320\n", encoding="utf-8")
    child.write_text("base_config: base.yaml\nmodel:\n  args:\n    width: 8\n", encoding="utf-8")
    before = child.read_bytes()

    cfg = audit_metrics._load_cfg(str(child))

    assert cfg["model"] == {"type": "BaseModel", "args": {"width": 8}}
    assert cfg["stft"]["n_fft"] == 320
    assert "base_config" not in cfg
    assert child.read_bytes() == before


def test_recursive_config_collection_covers_repo_files_without_duplicates():
    args = SimpleNamespace(config=None, configs_dir="configs", recursive=True)

    configs = audit_metrics._collect_configs_cli(args)

    assert configs
    labels = [label for label, _, _ in configs]
    assert len(set(labels)) == len(labels)
    assert "aec/offline_aec_template" in labels
    assert "tse/dynamic/tse_bsrnn_spk" in labels


def test_recursive_collection_supports_yaml_yml_and_preserves_load_failure(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "one.yaml").write_text("model:\n  type: One\n", encoding="utf-8")
    (nested / "two.yml").write_text("model:\n  type: Two\n", encoding="utf-8")
    (nested / "broken.yaml").write_text("model: [\n", encoding="utf-8")
    args = SimpleNamespace(config=None, configs_dir=str(tmp_path), recursive=True)

    configs = audit_metrics._collect_configs_cli(args)

    assert configs is not None
    assert [label for label, _, _ in configs] == [
        "nested/broken",
        "nested/two",
        "one",
    ]
    broken = configs[0]
    assert broken[1] is None
    assert broken[2]["audit_stage"] == "config_load"
    assert "ParserError" in broken[2]["audit_error"]


def test_cli_reports_one_structured_line_per_candidate(monkeypatch, tmp_path, capsys):
    (tmp_path / "good.yaml").write_text("model:\n  type: Tiny\n", encoding="utf-8")
    (tmp_path / "build_fail.yaml").write_text(
        "model:\n  type: Missing\n", encoding="utf-8"
    )
    (tmp_path / "bad.yml").write_text("model: [\n", encoding="utf-8")

    def fake_audit(label, cfg, *args, contract_out, **kwargs):
        if label == "build_fail":
            raise RuntimeError("missing dependency")
        contract_out.update(_contract())
        contract_out["RTF"] = {"value": None, "error": None, "duration_seconds": 1.0}
        return label, 12, 8, None, None, None, 16000

    monkeypatch.setattr(audit_metrics, "_audit_one", fake_audit)
    monkeypatch.setattr(
        audit_metrics.sys,
        "argv",
        [
            "audit_metrics.py",
            "--configs_dir",
            str(tmp_path),
            "--recursive",
            "--no_profile",
        ],
    )

    audit_metrics.main()

    lines = capsys.readouterr().out.splitlines()
    audit_records = [
        json.loads(line.removeprefix("[AUDIT] "))
        for line in lines
        if line.startswith("[AUDIT] ")
    ]
    summary = json.loads(next(
        line.removeprefix("[SUMMARY] ")
        for line in lines
        if line.startswith("[SUMMARY] ")
    ))
    assert len(audit_records) == 3
    assert {record["label"] for record in audit_records} == {
        "bad",
        "build_fail",
        "good",
    }
    assert {record["status"] for record in audit_records} == {"failed", "ok"}
    failed_by_label = {
        record["label"]: record for record in audit_records
        if record["status"] == "failed"
    }
    assert failed_by_label["bad"]["stage"] == "config_load"
    assert failed_by_label["build_fail"]["stage"] == "model_build_or_params"
    assert "missing dependency" in failed_by_label["build_fail"]["error"]
    assert summary == {
        "discovered": 3,
        "failed": 2,
        "mode": "cli",
        "ok": 1,
        "reported": 3,
    }


def test_task_resolution_supports_explicit_and_inferred_tasks():
    assert audit_metrics._resolve_task({"data": {"mode": "offline_aec"}}) == "aec"
    assert audit_metrics._resolve_task({"model": {"type": "TSE_BSRNN_SPK"}}) == "tse"
    assert audit_metrics._resolve_task({}, task="se") == "se"


def test_build_audit_models_supports_task_and_generator_scope(monkeypatch):
    import core.models.build as build_module
    import tasks.se.models as se_models

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(3))

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(5))

    monkeypatch.setattr(se_models, "register_se_models", lambda only=None: None)
    monkeypatch.setattr(
        build_module,
        "build_model",
        lambda cfg, stft_cfg=None: Generator() if cfg["type"] == "Gen" else Discriminator(),
    )
    cfg = {
        "model": {
            "generator": {"type": "Gen"},
            "discriminator": {"type": "Disc"},
        },
        "stft": {"n_fft": 4},
    }

    task_models, task_forward, task, task_params_scope, task_forward_scope = (
        audit_metrics._build_audit_models(cfg, "se", "task", "cpu")
    )
    gen_models, gen_forward, gen_task, gen_params_scope, gen_forward_scope = (
        audit_metrics._build_audit_models(cfg, "se", "generator", "cpu")
    )

    assert sum(audit_metrics.count_params(model)[0] for model in task_models) == 8
    assert task_forward.__class__ is Generator
    assert task == "se"
    assert task_params_scope == "task"
    assert task_forward_scope == "generator"
    assert sum(audit_metrics.count_params(model)[0] for model in gen_models) == 3
    assert gen_forward.__class__ is Generator
    assert gen_task == "se"
    assert gen_params_scope == "generator"
    assert gen_forward_scope == "generator"

    with pytest.raises(ValueError, match="requires model.generator"):
        audit_metrics._build_audit_models(
            {"model": {"type": "Gen"}}, "se", "generator", "cpu"
        )


class _FakeSTFT(nn.Module):
    def forward(self, wav):
        return wav.unsqueeze(1).unsqueeze(-1)

    def inverse(self, spec, length):
        return spec.squeeze(1).squeeze(-1)[..., :length]


def test_resolve_task_and_build_model_support_all_four_task_names():
    configs = {
        "se": {"data": {"mode": "offline"}},
        "aec": {"data": {"mode": "offline_aec"}},
        "tse": {"data": {"mode": "offline_tse"}},
        "ss": {"data": {"mode": "offline_ss"}},
    }
    for expected, cfg in configs.items():
        assert audit_metrics._resolve_task(cfg) == expected
    assert audit_metrics._resolve_task({"model": {"type": "TDANet"}}) == "ss"
    assert audit_metrics._resolve_task({"model": {"type": "TIGER"}}) == "ss"


def test_ss_waveform_and_external_stft_adapters_use_ss_contract(monkeypatch):
    class WaveformSS(nn.Module):
        input_domain = "waveform"
        sample_rate = 8

        def forward(self, mixture, lengths, *, fs):
            assert mixture.ndim == 2 and lengths.tolist() == [8] and fs == 8
            return {"est_sources": mixture.unsqueeze(1).repeat(1, 2, 1)}

    waveform = audit_metrics._SSProfileAdapter(WaveformSS(), sample_rate=8)
    assert waveform(torch.zeros(1, 8)).shape == (1, 2, 8)

    class FakeSTFT(nn.Module):
        def forward(self, mixture):
            return mixture.unsqueeze(1).unsqueeze(-1).repeat(1, 2, 1, 2)

    class SpectrumSS(nn.Module):
        input_domain = "spectrum"
        sample_rate = 8

        def forward(self, spectra, lengths, *, fs):
            assert spectra.ndim == 4 and lengths.tolist() == [8] and fs == 8
            return {"est_sources_spectra": spectra.unsqueeze(1).repeat(1, 2, 1, 1, 1)}

    monkeypatch.setattr(audit_metrics, "_build_stft", lambda _cfg: FakeSTFT())
    spectrum = audit_metrics._SSProfileAdapter(SpectrumSS(), {"n_fft": 4}, sample_rate=8)
    assert spectrum(torch.zeros(1, 8)).shape == (1, 2, 2, 8, 2)


def test_aec_adapter_uses_named_dict_inputs(monkeypatch):
    class AECModel(nn.Module):
        def forward(self, batch):
            self.seen_keys = set(batch)
            return {"nearend_estimate_spectra": batch["mic_spectra"]}

    monkeypatch.setattr(audit_metrics, "_build_stft", lambda cfg: _FakeSTFT())
    model = AECModel()
    adapter = audit_metrics._AECProfileAdapter(model, {"n_fft": 4})
    output = adapter(torch.zeros(1, 2, 8))
    named = audit_metrics._named_profile_input("aec", 8, 1.0, 3.0)

    assert model.seen_keys == {"mic_spectra", "farend_spectra"}
    assert output.shape == (1, 8)
    assert set(named) == {"mic", "farend"}
    assert named["mic"]["duration_seconds"] == 1.0


def test_tse_adapter_uses_cue_aware_enrollment_inputs(monkeypatch):
    class TSEModel(nn.Module):
        sample_rate = 8
        joint_training = True
        features_config = {
            "listen": {"enabled": False},
            "usef": {"enabled": False},
            "tfmap": {"enabled": True},
            "context": {"enabled": True},
            "spkemb": {"enabled": True},
        }

        def forward(self, batch):
            self.seen = batch
            return {"target_estimate_spectra": batch["mix_spectra"]}

    monkeypatch.setattr(audit_metrics, "_build_stft", lambda cfg: _FakeSTFT())
    model = TSEModel()
    adapter = audit_metrics._TSEProfileAdapter(
        model, {"n_fft": 4}, mix_samples=8, enroll_samples=24
    )
    output = adapter(torch.zeros(1, 2, 24))
    named = audit_metrics._named_profile_input("tse", 8, 1.0, 3.0)

    assert set(model.seen) == {
        "mix_spectra",
        "mixture",
        "enroll_spectra",
        "enroll_spectra_lengths",
        "enroll_wav",
        "enroll_lengths",
        "sample_rate",
    }
    assert model.seen["enroll_spectra_lengths"].tolist() == [model.seen["enroll_spectra"].shape[-2]]
    assert model.seen["enroll_lengths"].tolist() == [24]
    assert model.seen["enroll_wav"].shape == (1, 24)
    assert output.shape == (1, 8)
    assert set(named) == {"mixture", "enrollment"}
    assert named["mixture"]["duration_seconds"] == 1.0
    assert named["enrollment"]["duration_seconds"] == 3.0


def test_tse_preextracted_spkemb_variant_is_params_only(monkeypatch):
    class TSEModel(nn.Module):
        joint_training = False
        features_config = {
            "listen": {"enabled": False},
            "usef": {"enabled": False},
            "tfmap": {"enabled": False},
            "context": {"enabled": False},
            "spkemb": {"enabled": True},
        }

    monkeypatch.setattr(audit_metrics, "_build_stft", lambda cfg: _FakeSTFT())
    with pytest.raises(ValueError, match="Params-only.*real pre-extracted spk_embedding"):
        audit_metrics._TSEProfileAdapter(
            TSEModel(), {"n_fft": 4}, mix_samples=8, enroll_samples=24
        )


def test_no_profile_counts_waveform_native_model_without_stft(monkeypatch):
    model = nn.Linear(4, 2)
    monkeypatch.setattr(
        audit_metrics,
        "_build_audit_models",
        lambda cfg, task, scope, device: ([model], model, "se", "task", "task"),
    )
    contract = {}

    row = audit_metrics._audit_one(
        "waveform-native",
        {"model": {"type": "Tiny"}},
        device="cpu",
        duration=1.0,
        profile=False,
        contract_out=contract,
    )

    assert row[1:3] == (10, 10)
    assert row[3] is None
    assert row[4] is None
    assert contract["Params"]["total"] == 10
    assert contract["Params"]["scope"] == "task"


def test_profile_failure_preserves_params_and_contract(monkeypatch):
    model = nn.Linear(4, 2)
    monkeypatch.setattr(
        audit_metrics,
        "_build_audit_models",
        lambda cfg, task, scope, device: ([model], model, "se", "task", "task"),
    )
    monkeypatch.setattr(
        audit_metrics,
        "_make_profile_adapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unsupported forward")),
    )
    contract = {}

    row = audit_metrics._audit_one(
        "broken-profile",
        {"model": {"type": "Tiny"}, "stft": {"n_fft": 4}},
        device="cpu",
        duration=1.0,
        profile=True,
        contract_out=contract,
    )

    assert row[1:3] == (10, 10)
    assert row[3] is None
    assert "Params-only: RuntimeError: unsupported forward" in row[4]
    assert contract["Params"]["total"] == 10
    assert contract["MACs"]["value"] is None
    assert contract["FLOPs"]["value"] == NOT_MEASURED
    assert "unsupported forward" in contract["RTF"]["error"]
    assert contract["RTF"]["duration_seconds"] is None


def test_long_rtf_failure_does_not_pollute_macs_error(monkeypatch):
    model = nn.Linear(4, 2)
    named_input = audit_metrics._named_profile_input("se", 8, 1.0, 3.0)
    monkeypatch.setattr(
        audit_metrics,
        "_build_audit_models",
        lambda cfg, task, scope, device: ([model], model, "se", "task", "task"),
    )
    monkeypatch.setattr(
        audit_metrics,
        "_make_profile_adapter",
        lambda *args, **kwargs: (nn.Identity(), torch.zeros(1, 8), named_input),
    )
    monkeypatch.setattr(
        "core.utils.model_profile.profile_model",
        lambda *args, **kwargs: {
            "macs": 123.0,
            "macs_error": None,
            "rtf": 0.1,
            "rtf_error": None,
            "complexity": _contract(macs=123.0, macs_input=named_input),
        },
    )
    monkeypatch.setattr(
        audit_metrics,
        "_measure_rtf",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("long input failed")),
    )
    contract = {}

    row = audit_metrics._audit_one(
        "rtf-failure",
        {"model": {"type": "Tiny"}, "stft": {"n_fft": 4}},
        device="cpu",
        duration=1.0,
        profile=True,
        rtf_duration=30.0,
        contract_out=contract,
    )

    assert row[3] == 123.0
    assert row[4] is None
    assert row[5] is None
    assert "long input failed" in contract["RTF"]["error"]
    assert contract["RTF"]["duration_seconds"] is None
    assert contract["MACs"]["value"] == 123.0


def test_macs_failure_does_not_block_long_rtf_and_reports_duration(monkeypatch):
    model = nn.Linear(4, 2)
    named_input = audit_metrics._named_profile_input("se", 8, 1.0, 3.0)
    monkeypatch.setattr(
        audit_metrics,
        "_build_audit_models",
        lambda cfg, task, scope, device: ([model], model, "se", "task", "task"),
    )
    monkeypatch.setattr(
        audit_metrics,
        "_make_profile_adapter",
        lambda *args, **kwargs: (nn.Identity(), torch.zeros(1, 8), named_input),
    )
    monkeypatch.setattr(
        "core.utils.model_profile.profile_model",
        lambda *args, **kwargs: {
            "macs": None,
            "macs_error": "RuntimeError: unsupported MACs operator",
            "rtf": 0.1,
            "rtf_error": None,
            "complexity": _contract(macs=None, macs_input=named_input),
        },
    )
    monkeypatch.setattr(audit_metrics, "_measure_rtf", lambda *args, **kwargs: 0.25)
    contract = {}

    row = audit_metrics._audit_one(
        "macs-failure",
        {"model": {"type": "Tiny"}, "stft": {"n_fft": 4}},
        device="cpu",
        duration=1.0,
        profile=True,
        rtf_duration=30.0,
        contract_out=contract,
    )

    assert row[3] is None
    assert "unsupported MACs operator" in row[4]
    assert row[5] == 0.25
    assert contract["RTF"] == {
        "value": 0.25,
        "error": None,
        "duration_seconds": 30.0,
    }


def test_builtin_yaml_paths_exist():
    missing = [
        entry["yaml"]
        for entry in audit_metrics.AUDIT_ENTRIES
        if entry.get("yaml") and not (audit_metrics.os.path.isfile(
            audit_metrics.os.path.join(audit_metrics.REPO_ROOT, entry["yaml"])
        ))
    ]
    usr_entry = next(
        entry for entry in audit_metrics.AUDIT_ENTRIES
        if entry["label"] == "USR-BSRNN causal URGENT26"
    )

    assert missing == []
    assert usr_entry["yaml"] == (
        "configs/se/urgent2026/usr_bsrnn_causal_urgent26_offline_hybrid.yaml"
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"sample_rate": 0}, "sample_rate must be a positive integer"),
        ({"duration": 0.0}, "duration must be positive"),
        ({"batch_size": 0}, "batch_size must be a positive integer"),
        ({"warmup": -1}, "warmup must be a non-negative integer"),
        ({"repeats": 0}, "repeats must be a positive integer"),
    ],
)
def test_profile_model_rejects_invalid_parameter_boundaries(overrides, match):
    kwargs = {
        "sample_rate": 8,
        "duration": 1.0,
        "batch_size": 1,
        "warmup": 0,
        "repeats": 1,
        "quiet": True,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        profile_model(nn.Identity(), **kwargs)


def test_torchinfo_fallback_uses_real_input_target_and_forward_fn(monkeypatch):
    import sys
    from types import ModuleType

    class NeverCalledDirectly(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, value):
            raise AssertionError("torchinfo must profile the selected forward_fn")

    model = NeverCalledDirectly()
    profile_input = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    seen = {}

    def forward_fn(value, *, gain):
        seen["forward_input"] = value
        return value * gain

    def fake_summary(target, *, input_data, device, verbose):
        seen["target"] = target
        seen["summary_input"] = input_data
        seen["output"] = target(input_data)
        return SimpleNamespace(total_mult_adds=17)

    monkeypatch.setitem(sys.modules, "thop", ModuleType("thop"))
    torchinfo_module = ModuleType("torchinfo")
    torchinfo_module.summary = fake_summary
    monkeypatch.setitem(sys.modules, "torchinfo", torchinfo_module)

    result = profile_model(
        model,
        sample_rate=6,
        duration=1.0,
        profile_input=profile_input,
        forward_fn=forward_fn,
        forward_kwargs={"gain": 2.0},
        warmup=0,
        repeats=1,
        quiet=True,
    )

    assert seen["summary_input"] is profile_input
    assert seen["forward_input"] is profile_input
    assert torch.equal(seen["output"], profile_input * 2.0)
    assert dict(seen["target"].named_parameters())["model.weight"] is model.weight
    assert result["macs"] == 17.0
    assert result["complexity"]["MACs"]["backend"] == "torchinfo"


def test_torchinfo_backend_failure_is_not_reported_as_missing_dependency(monkeypatch):
    """torchinfo 自身失败必须暴露原因，不能与“依赖未安装”混为一谈。"""
    import sys
    from types import ModuleType

    class Tiny(nn.Module):
        def forward(self, value):
            return value * 2.0

    def failing_summary(*args, **kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setitem(sys.modules, "thop", ModuleType("thop"))
    torchinfo_module = ModuleType("torchinfo")
    torchinfo_module.summary = failing_summary
    monkeypatch.setitem(sys.modules, "torchinfo", torchinfo_module)

    result = profile_model(
        Tiny(),
        sample_rate=8,
        duration=1.0,
        profile_input=torch.zeros(1, 8),
        warmup=0,
        repeats=1,
        quiet=True,
    )

    assert result["macs"] is None
    assert "backend exploded" in result["macs_error"]


def test_cuda_indexed_device_is_synchronized_for_rtf(monkeypatch):
    from core.utils import model_profile as model_profile_module

    synchronized = []
    monkeypatch.setattr(
        audit_metrics.torch.cuda,
        "synchronize",
        lambda device=None: synchronized.append(device),
    )

    model_profile_module._synchronize_cuda("cuda:3")
    audit_metrics._measure_rtf(
        nn.Identity(), torch.zeros(1, 8), duration=1.0, repeats=1, device="cuda:3"
    )

    assert synchronized == [torch.device("cuda:3")] * 3


def test_task_resolution_uses_shared_data_mode_normalization():
    assert audit_metrics._resolve_task(
        {"data": {"mode": "  OFFLINE_AEC  "}}
    ) == "aec"
    with pytest.raises(ValueError, match=r"^Unsupported data\.mode 'streaming'"):
        audit_metrics._resolve_task({"data": {"mode": "streaming"}})


def test_cli_config_collection_deduplicates_same_canonical_path(tmp_path):
    config = tmp_path / "model.yaml"
    config.write_text("model:\n  type: Tiny\n", encoding="utf-8")
    args = SimpleNamespace(
        config=[str(config)], configs_dir=str(tmp_path), recursive=False
    )

    configs = audit_metrics._collect_configs_cli(args)

    assert configs is not None
    assert [label for label, _, _ in configs] == ["model"]


def test_audit_uses_runtime_model_sample_rate_when_config_omits_it(monkeypatch):
    class NativeRateModel(nn.Module):
        sample_rate = 48000

    model = NativeRateModel()
    monkeypatch.setattr(
        audit_metrics,
        "_build_audit_models",
        lambda cfg, task, scope, device: ([model], model, "se", "task", "task"),
    )

    row = audit_metrics._audit_one(
        "native-rate",
        {"model": {"type": "Tiny"}},
        device="cpu",
        duration=1.0,
        profile=False,
    )

    assert row[-1] == 48000


def test_equal_macs_and_rtf_duration_uses_requested_rtf_repeats(monkeypatch):
    model = nn.Identity()
    named_input = audit_metrics._named_profile_input("se", 8, 2.0, 3.0)
    captured = {}
    monkeypatch.setattr(
        audit_metrics,
        "_build_audit_models",
        lambda cfg, task, scope, device: ([model], model, "se", "task", "task"),
    )
    monkeypatch.setattr(
        audit_metrics,
        "_make_profile_adapter",
        lambda *args, **kwargs: (model, torch.zeros(1, 16), named_input),
    )

    def fake_profile(*args, **kwargs):
        captured.update(kwargs)
        return {
            "macs": 10.0,
            "macs_error": None,
            "rtf": 0.2,
            "rtf_error": None,
            "complexity": _contract(macs=10.0, macs_input=named_input),
        }

    monkeypatch.setattr("core.utils.model_profile.profile_model", fake_profile)
    contract = {}

    row = audit_metrics._audit_one(
        "equal-duration",
        {"model": {"type": "Tiny"}, "stft": {"n_fft": 4}},
        device="cpu",
        duration=2.0,
        rtf_duration=2.0,
        rtf_repeats=7,
        profile=True,
        contract_out=contract,
    )

    assert captured["repeats"] == 7
    assert row[5] == 0.2
    assert contract["RTF"]["duration_seconds"] == 2.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"duration": 0.0}, "duration must be positive"),
        ({"rtf_duration": 0.0}, "rtf_duration must be positive"),
        ({"enroll_duration": 0.0}, "enroll_duration must be positive"),
        ({"rtf_repeats": 0}, "rtf_repeats must be a positive integer"),
    ],
)
def test_audit_rejects_invalid_parameter_boundaries(monkeypatch, kwargs, match):
    monkeypatch.setattr(
        audit_metrics,
        "_build_audit_models",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("validation must happen before model construction")
        ),
    )
    call_kwargs = {
        "device": "cpu",
        "duration": 1.0,
        "profile": False,
        "rtf_duration": 30.0,
        "rtf_repeats": 30,
        "enroll_duration": 3.0,
    }
    call_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=match):
        audit_metrics._audit_one("invalid", {}, **call_kwargs)
