"""evaluate 实际 inference 入口接入公共 runner 的回归测试。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate.eval_pipeline_common as common

pytestmark = pytest.mark.unit


def _config(path: Path) -> None:
    path.write_text(
        "stft: {n_fft: 8, hop_length: 4, win_length: 8}\n"
        "data: {sample_rate: 16000}\n",
        encoding="utf-8",
    )


def test_se_entry_dispatches_to_task_adapter_runner(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    input_scp = tmp_path / "input.scp"
    _config(config)
    checkpoint.write_bytes(b"checkpoint")
    input_scp.write_text("utt /tmp/input.wav\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        common,
        "_run_se_adapter_inference",
        lambda **kwargs: calls.append(kwargs) or kwargs["output_path"],
    )

    result = common.run_inference_se(
        config,
        checkpoint,
        input_scp=input_scp,
        output_dir=tmp_path / "out",
        device="cpu",
    )

    assert result == (tmp_path / "out").resolve()
    assert calls and calls[0]["input_scp"] == input_scp.resolve()


def test_aec_entry_dispatches_to_task_adapter_runner(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    mic_scp = tmp_path / "mic.scp"
    far_scp = tmp_path / "far.scp"
    _config(config)
    checkpoint.write_bytes(b"checkpoint")
    mic_scp.write_text("utt /tmp/mic.wav\n", encoding="utf-8")
    far_scp.write_text("utt /tmp/far.wav\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        common,
        "_run_aec_adapter_inference",
        lambda **kwargs: calls.append(kwargs) or kwargs["output_path"],
    )

    result = common.run_inference_aec_paired(
        config,
        checkpoint,
        mic_scp=mic_scp,
        far_scp=far_scp,
        output_dir=tmp_path / "out",
        device="cpu",
        no_stream_align=True,
        skip_existing=True,
    )

    assert result == (tmp_path / "out").resolve()
    assert calls and calls[0]["mic_scp"] == mic_scp.resolve()
    assert calls[0]["far_scp"] == far_scp.resolve()
    assert calls[0]["no_stream_align"] is True
    assert calls[0]["skip_existing"] is True


def test_aec_entry_reads_top_level_frontend_and_forwards_runtime_flags(
    tmp_path, monkeypatch
):
    import evaluate.eval_pipeline_common as common_module
    import inference
    import tasks.aec.inference as aec_inference

    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    mic_scp = tmp_path / "mic.scp"
    far_scp = tmp_path / "far.scp"
    config.write_text("fixture\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    mic_scp.write_text("utt /tmp/mic.wav\n", encoding="utf-8")
    far_scp.write_text("utt /tmp/far.wav\n", encoding="utf-8")
    cfg = {
        "stft": {"n_fft": 8, "hop_length": 4, "win_length": 8},
        "data": {"sample_rate": 16000},
        "aec_frontend": {
            "delay_alignment": {"enabled": True},
            "input_routing": "mic_farend",
        },
    }
    captured = {}

    class Model:
        pass

    class Adapter:
        def __init__(self, model, **kwargs):
            captured["adapter_model"] = model
            captured["adapter_kwargs"] = kwargs

    monkeypatch.setattr(
        common_module,
        "load_pipeline_config",
        lambda *_args, **_kwargs: (cfg, {"sha256": "cfg-id"}),
    )
    monkeypatch.setattr(inference, "_load_model", lambda *_args, **_kwargs: Model())
    monkeypatch.setattr(aec_inference, "AECInferenceAdapter", Adapter)
    monkeypatch.setattr(common_module, "_paired_scp_items", lambda *_args: ["item"])
    monkeypatch.setattr(
        common_module,
        "_adapter_artifact_fingerprint",
        lambda *_args, **kwargs: captured.setdefault("fingerprint", kwargs) or {},
    )
    monkeypatch.setattr(
        common_module,
        "run_adapter_inference",
        lambda **kwargs: captured.setdefault("runner", kwargs),
    )

    result = common_module._run_aec_adapter_inference(
        config_path=config,
        checkpoint_path=checkpoint,
        mic_scp=mic_scp,
        far_scp=far_scp,
        output_path=tmp_path / "out",
        device="cpu",
        effective_identity={"sha256": "cfg-id"},
        allow_partial=True,
        no_stream_align=True,
        skip_existing=True,
    )

    assert result == tmp_path / "out"
    frontend = captured["adapter_kwargs"]["frontend"]
    assert frontend.to_dict()["delay_alignment"]["enabled"] is False
    assert "mic_adapter" not in captured["adapter_kwargs"]
    assert captured["runner"]["skip_existing"] is True
    assert captured["runner"]["allow_partial"] is True
    assert captured["fingerprint"]["parameters"] == {
        "aec_no_stream_align": True,
    }


def test_waveform_native_se_entry_uses_public_runner_and_reuses_outputs(
    tmp_path, monkeypatch
):
    import torch
    import torchaudio
    import tasks.se.models

    config = tmp_path / "waveform.yaml"
    checkpoint = tmp_path / "unused-training-checkpoint.tar"
    artifact = tmp_path / "encoder.pt"
    noisy = tmp_path / "noisy.wav"
    input_scp = tmp_path / "input.scp"
    output_dir = tmp_path / "out"
    config.write_text(
        "meta: {allow_no_stft: true}\n"
        "data: {mode: dynamic}\n"
        "model: {type: WaveformTest, args: {encoder_ckpt_path: %s}}\n" % artifact,
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"unused")
    artifact.write_bytes(b"encoder-v1")
    torchaudio.save(str(noisy), torch.linspace(-0.5, 0.5, 64).unsqueeze(0), 16000)
    input_scp.write_text(f"utt {noisy}\n", encoding="utf-8")

    class WaveformModel:
        calls = 0
        sr_out = 16000
        inference_peak_normalization = False

        def to(self, _device):
            return self

        def eval(self):
            return self

        def forward(self, waveform, lengths=None):
            type(self).calls += 1
            return waveform

        __call__ = forward

    monkeypatch.setattr(tasks.se.models, "build_model", lambda *_a, **_k: WaveformModel())

    result = common.run_inference_se(
        config,
        checkpoint,
        input_scp=input_scp,
        output_dir=output_dir,
        device="cpu",
        skip_existing=False,
        chunk_seconds=0.0,
    )
    assert result == output_dir.resolve()
    assert WaveformModel.calls == 1
    assert (output_dir / "run_manifest.json").is_file()

    common.run_inference_se(
        config,
        checkpoint,
        input_scp=input_scp,
        output_dir=output_dir,
        device="cpu",
        skip_existing=True,
        chunk_seconds=0.0,
    )
    assert WaveformModel.calls == 1
    assert "utt " in (output_dir / "inf.scp").read_text(encoding="utf-8")
