"""根推理入口任务路由与输入合同回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.unit
def test_resolve_inference_task_uses_data_mode_not_config_path():
    from inference import _resolve_inference_task

    cfg = {
        "data": {"mode": "dynamic_aec"},
        "model": {"type": "DeepVQE"},
    }

    assert _resolve_inference_task(cfg) == "aec"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "model_type", "expected_task"),
    [
        ("offline", "GTCRN", "se"),
        ("offline_tse", "TSE_BSRNN_SPK", "tse"),
        ("offline_ss", "TDANet", "ss"),
        ("dynamic_ssl", "FixedArrayIPDNet", "ssl"),
    ],
)
def test_resolve_inference_task_routes_all_supported_task_families(
    mode, model_type, expected_task
):
    from inference import _resolve_inference_task

    assert _resolve_inference_task(
        {"data": {"mode": mode}, "model": {"type": model_type}}
    ) == expected_task


@pytest.mark.unit
def test_descriptor_factory_owns_se_adapter_and_rejects_non_public_tasks():
    import torch

    from tasks.registry import build_inference_adapter
    from tasks.se.inference import SEInferenceAdapter

    class Model(torch.nn.Module):
        sample_rate = 16000

        def forward(self, spectrum, lengths=None, *, fs=None):
            return spectrum

    adapter = build_inference_adapter(
        "se",
        Model(),
        stft={"n_fft": 32, "hop_length": 16, "win_length": 32},
        sample_rate=16000,
    )
    assert isinstance(adapter, SEInferenceAdapter)
    with pytest.raises(ValueError, match="SS root file inference is unavailable"):
        build_inference_adapter("ss", object())
    with pytest.raises(ValueError, match="SSL root file inference is unavailable"):
        build_inference_adapter("ssl", object())


@pytest.mark.unit
def test_descriptor_factory_constructs_aec_adapter_from_root_stft_cfg():
    import torch

    from tasks.aec.inference import AECInferenceAdapter
    from tasks.registry import build_inference_adapter

    class Model(torch.nn.Module):
        def forward(self, inputs):
            return {"nearend_estimate_spectra": inputs["mic_spectra"]}

    adapter = build_inference_adapter(
        "aec",
        Model(),
        stft_cfg={"n_fft": 32, "hop_length": 16, "win_length": 32},
        model_fixed_sr=16000,
    )

    assert isinstance(adapter, AECInferenceAdapter)
    assert adapter.sample_rate == 16000
    assert adapter.stft.n_fft == 32


@pytest.mark.unit
def test_descriptor_factory_constructs_tse_adapter_from_loaded_extractor():
    from tasks.registry import build_inference_adapter
    from tasks.tse.inference import TSEInferenceAdapter

    class Extractor:
        def declared_cue_roles(self):
            return {}

        def extract_speech_with_cues(self, mixture, sr, cues):
            raise AssertionError("factory smoke must not run inference")

    adapter = build_inference_adapter("tse", Extractor())

    assert isinstance(adapter, TSEInferenceAdapter)


@pytest.mark.unit
def test_tse_root_manifest_uses_public_report_and_sidecar_lifecycle(monkeypatch, tmp_path):
    import json
    from pathlib import Path

    import inference
    from core.inference import InferenceReport, bind_artifacts
    from core.utils.inference_artifacts import fingerprint_file
    import tasks.tse.inference as tse_inference
    import tasks.tse.inference.adapter as tse_adapter_module

    mixture = tmp_path / "mix.wav"
    enrollment = tmp_path / "enroll.wav"
    checkpoint = tmp_path / "model.pt"
    config = tmp_path / "config.yaml"
    for path in (mixture, enrollment, checkpoint, config):
        path.write_bytes(path.name.encode())
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "uid": "utt",
                "mixture": mixture.name,
                "speaker_waveform": enrollment.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Extractor:
        device = "cpu"

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Snapshot:
        @classmethod
        def from_loaded_extractor(cls, extractor, params):
            cls.params = params
            return cls()

        def verify_global_identity(self):
            self.verified = True

        def source_file_identities(self):
            return {
                "checkpoint": fingerprint_file(checkpoint),
                "config": fingerprint_file(config),
            }

        def bind_inputs(self, input_paths, *, cues):
            return {"uid_inputs": sorted(input_paths), "cues": cues}

    class Adapter:
        def prepare_item(self, item):
            return (
                {"speaker_waveform": Path(item.inputs["speaker_waveform"])},
                {"speaker_waveform": {"kind": "file", "path": str(item.inputs["speaker_waveform"])}},
            )

    captured = {}
    monkeypatch.setattr(tse_inference, "TSEExtractor", Extractor)
    monkeypatch.setattr(tse_inference, "TSEInferenceArtifactSnapshot", Snapshot)
    monkeypatch.setattr(
        tse_adapter_module, "build_inference_adapter", lambda extractor: Adapter()
    )
    monkeypatch.setattr(
        inference,
        "_tse_root_artifact_binding",
        lambda *args, **kwargs: bind_artifacts(
            {"checkpoint": checkpoint, "config": config}
        ),
    )
    monkeypatch.setattr(
        inference,
        "run_inference_report",
        lambda spec, adapter: (
            captured.update(spec=spec, adapter=adapter)
            or InferenceReport(
                "completed",
                (),
                tmp_path / "run_manifest.json",
                tmp_path / "inf.scp",
            )
        ),
    )
    monkeypatch.setattr(
        tse_inference,
        "publish_sidecars_from_report",
        lambda report, expected, *, output_dir: captured.update(
            expected=expected, output_dir=output_dir
        ),
    )

    args = SimpleNamespace(
        config=str(config),
        checkpoint=str(checkpoint),
        tse_manifest=str(manifest),
        scp_root_dir=None,
        output_relpath_scp=None,
        device="cpu",
        sr=None,
        tse_apply_vad=False,
        skip_existing=True,
        allow_partial=False,
        chunk_seconds=0.0,
        cpu_long_audio_seconds=0.0,
        input_dir=None,
        input_scp=None,
        input_mic_dir=None,
        input_far_dir=None,
        input_mic_scp=None,
        input_far_scp=None,
        aec_flat_dir=None,
        aec_stream_align=False,
        aec_no_stream_align=False,
        aec_review_multichannel=False,
    )

    inference._run_tse_root_inference(
        args, tmp_path / "outputs", effective_identity={"sha256": "cfg"}
    )

    assert captured["spec"].items[0].uid == "utt"
    assert captured["spec"].reuse.enabled is True
    assert captured["adapter"].__class__ is Adapter
    assert set(captured["expected"]) == {"utt"}


@pytest.mark.unit
def test_resolve_inference_task_rejects_task_model_mismatch():
    from inference import _resolve_inference_task

    cfg = {
        "data": {"mode": "dynamic_tse"},
        "model": {"type": "DeepVQE"},
    }

    with pytest.raises(ValueError, match="routes to TSE"):
        _resolve_inference_task(cfg)


@pytest.mark.unit
def test_tse_root_inference_requires_the_public_manifest(monkeypatch, tmp_path):
    import core.utils.config as config_module
    import inference

    monkeypatch.setattr(
        config_module,
        "load_merged_config",
        lambda _path: {
            "data": {"mode": "dynamic_tse"},
            "model": {"type": "TSE_BSRNN_SPK"},
        },
    )
    args = SimpleNamespace(
        config="tse_infer.yaml",
        cpu_long_audio_seconds=0.0,
    )

    with pytest.raises(ValueError, match="--tse-manifest"):
        inference._main_locked(args, tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "model_type", "message"),
    [
        ("offline_ss", "TDANet", "SS root file inference is unavailable.*SSExtractor"),
        ("dynamic_ssl", "FixedArrayIPDNet", "SSL root file inference is unavailable.*dynamic scene"),
    ],
)
def test_root_inference_rejects_non_se_file_tasks_before_model_build(
    monkeypatch, tmp_path, mode, model_type, message
):
    import inference

    monkeypatch.setattr(
        inference,
        "load_effective_config",
        lambda *_args, **_kwargs: (
            {"data": {"mode": mode}, "model": {"type": model_type}},
            "cfg-id",
        ),
    )
    monkeypatch.setattr(
        inference,
        "_load_model",
        lambda *_args, **_kwargs: pytest.fail("non-SE root routing must fail before model build"),
    )
    args = SimpleNamespace(
        config="task_infer.yaml",
        expected_effective_config_sha256=None,
        cpu_long_audio_seconds=0.0,
    )

    with pytest.raises(ValueError, match=message):
        inference._main_locked(args, tmp_path)


@pytest.mark.unit
def test_collect_files_rejects_directory_and_scp_together():
    from inference import _collect_files

    with pytest.raises(ValueError, match="目录和 SCP"):
        _collect_files(input_dir="audio", input_scp="input.scp")


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, -16000, True, "16000"])
def test_resolve_se_sample_rate_rejects_non_positive_or_non_integer(value):
    from inference import _resolve_se_sample_rate

    with pytest.raises(ValueError, match="positive integer"):
        _resolve_se_sample_rate(value)


@pytest.mark.unit
def test_resolve_se_sample_rate_preserves_native_sr_and_cli_override():
    from inference import _resolve_se_sample_rate

    assert _resolve_se_sample_rate(None) is None
    assert _resolve_se_sample_rate(16000, 48000) == 48000


@pytest.mark.unit
def test_train_handoff_reuses_routed_config_through_real_entry_bootstrap(monkeypatch, tmp_path):
    from torch.utils.data import DataLoader

    import train
    from core.trainer.entry import UnifiedTrainingEntry

    config_path = tmp_path / "config.yaml"
    config_path.touch()
    cfg = {"data": {"mode": "offline"}, "model": {}, "trainer": {}}
    observed = {}

    class StrictTrainDataset(list):
        strict_load = True

    train_dataset = StrictTrainDataset(["train dataset"])

    class Entry(UnifiedTrainingEntry):
        def _validate_config(self, config):
            observed["validated_config"] = config

        def build_loaders(self, config, rank, local_rank, world_size):
            observed["loader_config"] = config
            return DataLoader([0]), DataLoader([]), None, train_dataset

        def build_model_for_training(self, config, rank, *, train_dataset):
            observed["model_config"] = config
            observed["train_dataset"] = train_dataset
            return object()

        def build_model(self, config, rank):
            raise AssertionError("training must use build_model_for_training")

        def get_trainer_cls(self):
            return object

        def run_training(self, config, *args):
            observed["training_config"] = config

    entry = Entry("SE_Test")
    monkeypatch.setattr(train, "load_training_entry", lambda task: (lambda _name: entry))
    monkeypatch.setattr(entry, "_import_trainer_class", lambda _name: object)
    monkeypatch.setattr(train, "load_merged_config", lambda _path: cfg)
    monkeypatch.setattr("sys.argv", ["train.py", "-C", str(config_path)])

    train.main()

    assert all(observed[name] is cfg for name in (
        "validated_config", "loader_config", "model_config", "training_config",
    ))
    assert observed["train_dataset"] == ["train dataset"]
    assert observed["train_dataset"] is train_dataset
    assert observed["train_dataset"].strict_load is True


@pytest.mark.unit
def test_bootstrap_rejects_handoff_from_a_different_config_path(tmp_path):
    import logging

    from core.trainer.entry_components import bootstrap_entry

    cli_config = tmp_path / "cli.yaml"
    routed_config = tmp_path / "routed.yaml"
    cli_config.touch()
    routed_config.touch()

    class Entry:
        logger = logging.getLogger("test_config_handoff_path")

        def parse_args(self):
            return SimpleNamespace(
                config=str(cli_config),
                log_level="INFO",
                log_file=None,
                strict=False,
                max_steps=None,
                exp_path=None,
            )

        def _validate_config(self, config):
            raise AssertionError("path mismatch must fail before validation")

    with pytest.raises(ValueError, match="CLI --config path"):
        bootstrap_entry(Entry(), config={}, config_path=routed_config)


@pytest.mark.unit
def test_inference_model_builder_cannot_mutate_effective_config(monkeypatch, tmp_path):
    import inference
    import core.models.build as build_module

    cfg = {
        "model": {"type": "FastEnhancer", "args": {"variant": "T"}},
        "data": {"mode": "offline"},
    }
    before = __import__("copy").deepcopy(cfg)

    class Model:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def mutating_builder(model_cfg, **_kwargs):
        model_cfg["args"]["builder_mutation"] = True
        return Model()

    monkeypatch.setattr(build_module, "build_model", mutating_builder)
    monkeypatch.setattr(inference, "load_checkpoint", lambda *a, **k: (a[0], {}))

    inference._load_model(cfg, tmp_path / "model.pt", "cpu")

    assert cfg == before


@pytest.mark.unit
def test_inference_model_builder_routes_aec_models_to_aec_registry(monkeypatch, tmp_path):
    import inference
    import tasks.aec.models
    import core.models.build as build_module

    calls = []

    class Model:
        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        tasks.aec.models,
        "register_aec_models",
        lambda **kwargs: calls.append(("aec", kwargs)),
    )
    monkeypatch.setattr(
        build_module,
        "build_model",
        lambda model_cfg, **_kwargs: Model(),
    )
    monkeypatch.setattr(inference, "load_checkpoint", lambda *a, **k: (a[0], {}))

    cfg = {
        "model": {"type": "DeepVQE", "args": {"variant": "small"}},
        "data": {"mode": "dynamic_aec"},
    }
    inference._load_model(cfg, tmp_path / "model.pt", "cpu")

    assert calls == [("aec", {"only": "DeepVQE"})]


@pytest.mark.unit
def test_effective_config_mismatch_fails_before_inference_routing(monkeypatch, tmp_path):
    import core.utils.config as config_module
    import inference

    monkeypatch.setattr(
        config_module,
        "load_merged_config",
        lambda _path: {
            "data": {"mode": "offline"},
            "model": {"type": "GTCRN"},
        },
    )
    monkeypatch.setattr(
        inference,
        "_resolve_inference_task",
        lambda _cfg: pytest.fail("identity mismatch 必须先于 routing/model 构造失败"),
    )
    args = SimpleNamespace(
        config="config.yaml",
        expected_effective_config_sha256="0" * 64,
        cpu_long_audio_seconds=0.0,
    )

    with pytest.raises(RuntimeError, match="effective config identity mismatch"):
        inference._main_locked(args, tmp_path)

@pytest.mark.unit
def test_root_se_main_routes_real_adapter_to_public_runner(monkeypatch, tmp_path):
    """根 CLI 必须把真实 SE adapter 交给 runner，不能先跑旧批量循环再复制。"""
    import torch
    import torchaudio
    import inference
    import tasks.se.models

    source = tmp_path / "noisy.wav"
    torchaudio.save(str(source), torch.zeros(1, 32), 16000)
    captured = {}

    class Model:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def forward(self, wav, lengths):
            captured["model_calls"] = captured.get("model_calls", 0) + 1
            return wav

        __call__ = forward

    class Args:
        config = str(tmp_path / "config.yaml")
        expected_effective_config_sha256 = None
        checkpoint = None
        input_dir = str(tmp_path)
        input_scp = None
        scp_root_dir = None
        input_mic_dir = input_far_dir = None
        input_mic_scp = input_far_scp = None
        aec_flat_dir = None
        output_dir = str(tmp_path / "out")
        output_relpath_scp = None
        device = "cpu"
        cpu_long_audio_seconds = 0.0
        sr = None
        no_progress = True
        chunk_seconds = 0.0
        skip_existing = False
        allow_partial = False
        aec_stream_align = False
        aec_no_stream_align = False
        aec_review_multichannel = False

    cfg = {
        "data": {"mode": "offline"},
        "model": {"type": "WaveformTest"},
        "meta": {"allow_no_stft": True},
    }
    monkeypatch.setattr(inference, "load_effective_config", lambda *_a, **_k: (cfg, "cfg-id"))
    from core.utils.inference_artifacts import fingerprint_json
    mocked_fingerprint = {
        "algorithm": "sha256",
        "files": {"config": None, "checkpoint": None},
        "parameters": {},
    }
    mocked_fingerprint["sha256"] = fingerprint_json(mocked_fingerprint)
    monkeypatch.setattr(inference, "build_artifact_fingerprint", lambda *_a, **_k: mocked_fingerprint)
    monkeypatch.setattr(inference, "inference_source_files", lambda *_a, **_k: [])
    monkeypatch.setattr(inference, "configured_checkpoint_files", lambda *_a, **_k: {"artifact": str(source)})
    monkeypatch.setattr(inference, "verify_configured_checkpoint_fingerprints", lambda *_a, **_k: None)
    monkeypatch.setattr(inference, "_collect_files", lambda *_a, **_k: [("utt", str(source))])
    monkeypatch.setattr(tasks.se.models, "build_model", lambda *_a, **_k: Model())
    assert not hasattr(inference, "run_waveform_inference"), "旧批量循环必须已从主路径移除"
    assert not hasattr(inference, "run_se_inference"), "旧批量循环必须已从主路径移除"

    def run_public(spec, adapter):
        captured["adapter"] = adapter
        item = spec.items[0]
        output = adapter.infer(item)
        output.writer(tmp_path / "staged.wav")
        return SimpleNamespace(results=())

    monkeypatch.setattr(inference, "run_inference_report", run_public)
    inference._main_locked(Args(), tmp_path / "out")

    assert isinstance(captured["adapter"], inference._SEInferenceAdapter)
    assert captured["model_calls"] == 1
    assert (tmp_path / "staged.wav").is_file()


@pytest.mark.unit
def test_root_aec_rejects_multichannel_mic_without_explicit_adapter(monkeypatch, tmp_path):
    import inference
    from tasks.aec.inference.adapter import AECInferenceAdapter

    mic = tmp_path / "mic.wav"
    farend = tmp_path / "far.wav"
    mic.touch()
    farend.touch()

    monkeypatch.setattr(
        inference.torchaudio,
        "load",
        lambda path: (
            (__import__("torch").ones(2, 8), 16000)
            if str(path).endswith("mic.wav")
            else (__import__("torch").ones(1, 8), 16000)
        ),
    )

    class Model:
        def eval(self):
            return self

        def parameters(self):
            return iter(())

    adapter = AECInferenceAdapter(
        Model(),
        stft={"n_fft": 4, "hop_length": 2, "win_length": 4},
        sample_rate=16000,
    )
    with pytest.raises(ValueError, match="native model IO contract"):
        adapter.infer(
            SimpleNamespace(
                uid="utt", inputs={"mic": mic, "farend": farend}
            )
        )


@pytest.mark.unit
def test_root_aec_rejects_spatial_mic_for_mono_model(monkeypatch, tmp_path):
    import torch
    import inference
    import tasks.aec.inference.adapter as aec_adapter_module
    from tasks.aec.inference.adapter import AECInferenceAdapter

    mic = tmp_path / "mic.wav"
    farend = tmp_path / "far.wav"
    mic.touch()
    farend.touch()
    mic_wave = torch.stack((torch.full((8,), 2.0), torch.full((8,), 7.0)))
    far_wave = torch.ones(1, 8)
    monkeypatch.setattr(
        inference.torchaudio,
        "load",
        lambda path: (mic_wave, 16000) if str(path).endswith("mic.wav") else (far_wave, 16000),
    )

    class STFT:
        n_fft = 4

        def to(self, _device):
            return self

        def __call__(self, value):
            assert value.ndim == 2
            return value.unsqueeze(-1).repeat(1, 1, 2)

        def inverse(self, value, _length):
            return value[..., 0]

    class Model:
        def eval(self):
            return self

        def parameters(self):
            return iter(())

        def __call__(self, batch):
            raise AssertionError("mono model must not receive multichannel input")

    monkeypatch.setattr(aec_adapter_module, "build_stft_istft", lambda _cfg: STFT())
    adapter = AECInferenceAdapter(
        Model(),
        stft={"n_fft": 4, "hop_length": 2, "win_length": 4},
        sample_rate=16000,
    )
    with pytest.raises(ValueError, match="native model IO contract"):
        adapter.infer(SimpleNamespace(uid="utt", inputs={"mic": mic, "farend": farend}))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "model_type", "task", "guidance"),
    [
        (
            "dynamic_ss",
            "TDANet",
            "SS",
            "tasks.ss.inference.SSExtractor",
        ),
        (
            "dynamic_ssl",
            "FixedArrayIPDNet",
            "SSL",
            "dynamic scene",
        ),
    ],
)
def test_root_inference_rejects_non_file_tasks_before_model_build(
    monkeypatch, tmp_path, mode, model_type, task, guidance
):
    import core.utils.config as config_module
    import inference

    monkeypatch.setattr(
        config_module,
        "load_merged_config",
        lambda _path: {"data": {"mode": mode}, "model": {"type": model_type}},
    )
    monkeypatch.setattr(
        inference,
        "_load_model",
        lambda *_args, **_kwargs: pytest.fail("SS/SSL must not enter the SE model loader"),
    )
    args = SimpleNamespace(config="task.yaml", cpu_long_audio_seconds=0.0)

    with pytest.raises(ValueError, match=rf"{task}.*{guidance}"):
        inference._main_locked(args, tmp_path)


@pytest.mark.unit
def test_resolve_inference_task_rejects_cross_task_ss_and_ssl_models():
    from inference import _resolve_inference_task

    with pytest.raises(ValueError, match="routes to SS"):
        _resolve_inference_task(
            {"data": {"mode": "dynamic_ss"}, "model": {"type": "GTCRN"}}
        )
    with pytest.raises(ValueError, match="routes to SSL"):
        _resolve_inference_task(
            {"data": {"mode": "dynamic_ssl"}, "model": {"type": "GTCRN"}}
        )


@pytest.mark.unit
def test_apply_meta_tf32_policy_honours_declared_flag():
    """显式声明的 meta.allow_tf32 必须在推理侧生效；未声明时保持全局默认。"""
    import torch

    from inference import _apply_meta_tf32_policy

    cudnn_before = torch.backends.cudnn.allow_tf32
    matmul_before = torch.backends.cuda.matmul.allow_tf32
    try:
        _apply_meta_tf32_policy({"meta": {"allow_tf32": False}})
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is False

        _apply_meta_tf32_policy({"meta": {"allow_tf32": True}})
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is True

        _apply_meta_tf32_policy({"meta": {}})
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_before
        torch.backends.cuda.matmul.allow_tf32 = matmul_before
