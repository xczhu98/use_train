"""文件推理加载身份、不可变计划和真实 WAV 复用的生命周期回归。"""

import json

import pytest
import torch
import torchaudio

from core.inference import (
    ArtifactBinding, FileInferenceItem, InferenceSpec, PreparedOutput,
    ReusePolicy, run_inference_report,
)
from core.utils.inference_artifacts import (
    MANIFEST_SCHEMA_VERSION,
    load_run_manifest,
)
from tasks.ss.inference import SSInferenceAdapter
from tasks.ss.model_io import resolve_ss_execution_plan

pytestmark = pytest.mark.unit


class _Separator(torch.nn.Module):
    input_domain = "waveform"
    sample_rate = 16000

    def forward(self, mixture, lengths, *, fs=None):
        return {"est_sources": torch.stack((mixture, -mixture), dim=1)}


def test_real_ss_wav_source_selection_invalidates_reuse(tmp_path):
    source = tmp_path / "input.wav"
    waveform = torch.linspace(0.05, 0.5, 256).unsqueeze(0)
    torchaudio.save(str(source), waveform, 16000)
    config = tmp_path / "config.yaml"
    config.write_text("data: {mode: offline_ss, sample_rate: 16000}\nss: {num_sources: 2}\n")
    model = _Separator()
    adapter = SSInferenceAdapter(model, plan=resolve_ss_execution_plan(
        {"data": {"mode": "offline_ss", "sample_rate": 16000}, "ss": {"num_sources": 2}}, model,
    ))
    output = tmp_path / "out" / "u.wav"

    def run(index):
        return run_inference_report(InferenceSpec(
            config=config, output_dir=output.parent,
            items=[FileInferenceItem("u", {"mixture": source}, output, {"source_index": index})],
            artifacts=ArtifactBinding(files={"config": config}),
            reuse=ReusePolicy(enabled=True),
        ), adapter)

    assert run(0).results[0].status == "ok"
    assert run(0).results[0].status == "skipped"
    assert run(1).results[0].status == "ok"
    negative, rate = torchaudio.load(str(output))
    assert rate == 16000
    torch.testing.assert_close(negative, -waveform, atol=7e-5, rtol=0)
    assert run(1).results[0].status == "skipped"
    manifest, _ = load_run_manifest(output.parent / "run_manifest.json")
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["input_inventory"]["items"][0]["attributes"] == {"source_index": 1}


class _Gain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, spectrum, lengths=None):
        if isinstance(spectrum, dict):
            return {"nearend_estimate_spectra": spectrum["mic_spectra"] * self.gain}
        return spectrum * self.gain


@pytest.mark.parametrize("entry", ["root", "eval_se", "eval_aec"])
@pytest.mark.parametrize("allow_partial", [False, True])
def test_real_loaded_checkpoint_replacement_rejected(tmp_path, monkeypatch, entry, allow_partial):
    import core.models.build as model_build
    import inference
    from evaluate.eval_pipeline_common import run_inference_aec_paired, run_inference_se

    config = tmp_path / "config.yaml"
    is_aec = entry == "eval_aec"
    config.write_text(
        f"data: {{mode: {'offline_aec' if is_aec else 'offline'}, sample_rate: 16000}}\n"
        f"model: {{type: {'DeepVQE' if is_aec else 'GTCRN'}}}\n"
        "stft: {n_fft: 32, hop_length: 16, win_length: 32}\n"
    )
    checkpoint = tmp_path / "model.pt"
    replacement = tmp_path / "replacement.pt"
    torch.save({"model": {"gain": torch.tensor(1.0)}}, checkpoint)
    torch.save({"model": {"gain": torch.tensor(-1.0)}}, replacement)
    source = tmp_path / "input.wav"
    torchaudio.save(str(source), torch.full((1, 256), 0.2), 16000)
    scp = tmp_path / "input.scp"
    scp.write_text(f"u {source}\n")
    output = tmp_path / "out"
    loaded = []
    monkeypatch.setattr(model_build, "build_model", lambda *args, **kwargs: _Gain())
    real_loader = inference._load_model

    def load_and_replace(*args, **kwargs):
        model = real_loader(*args, **kwargs)
        assert model.gain.item() == 1.0
        loaded.append(model)
        replacement.replace(checkpoint)
        return model

    monkeypatch.setattr(inference, "_load_model", load_and_replace)
    with pytest.raises(RuntimeError, match="artifact changed|内容发生变化"):
        if entry == "root":
            argv = ["inference.py", "-C", str(config), "--checkpoint", str(checkpoint),
                    "--input_scp", str(scp), "--output_dir", str(output), "--device", "cpu"]
            if allow_partial:
                argv.append("--allow-partial")
            monkeypatch.setattr("sys.argv", argv)
            inference.main()
        elif is_aec:
            run_inference_aec_paired(config, checkpoint, mic_scp=scp, far_scp=scp,
                                     output_dir=output, device="cpu", allow_partial=allow_partial)
        else:
            run_inference_se(config, checkpoint, input_scp=scp, output_dir=output,
                             device="cpu", allow_partial=allow_partial)
    assert len(loaded) == 1
    assert not (output / "run_manifest.json").exists()
    assert not (output / "u.wav").exists()


def test_inference_plan_deep_freezes_json_contract_and_rejects_old_manifest(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("model: test\n")
    source = tmp_path / "input.wav"
    source.write_bytes(b"input")
    attrs = {"nested": {"values": [1, 2]}}
    metadata = {"runtime": {"flags": ["a"]}}
    item = FileInferenceItem("u", {"noisy": source}, tmp_path / "u.wav", attrs)
    spec = InferenceSpec(
        config=config,
        output_dir=tmp_path,
        items=[item],
        artifacts=ArtifactBinding(files={"config": config}),
        run_metadata=metadata,
    )
    attrs["nested"]["values"].append(3)
    metadata["runtime"]["flags"].append("b")
    with pytest.raises(TypeError):
        item.attributes["nested"] = {}
    assert item.attributes["nested"]["values"] == (1, 2)
    assert spec.run_metadata["runtime"]["flags"] == ("a",)

    class Adapter:
        def infer(self, current):
            return PreparedOutput(current.uid, lambda path: path.write_bytes(b"output"))

    report = run_inference_report(spec, Adapter())
    assert report.status == "completed"
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION - 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        run_inference_report(
            InferenceSpec(
                config=config,
                output_dir=tmp_path,
                items=[FileInferenceItem("u", {"noisy": source}, tmp_path / "u.wav")],
                artifacts=ArtifactBinding(files={"config": config}),
                reuse=ReusePolicy(enabled=True, on_rejection="fail"),
            ),
            Adapter(),
        )
