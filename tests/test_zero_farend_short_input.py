"""JAE-M04 回归测试：run_aec_inference_zero_farend 短输入处理。

评估管线内进程路径（evaluate/eval_pipeline_common.py）与 inference.py 同口径：
- 短于 STFT 最小长度（n_fft//2 + 1）的输入在 STFT 前右侧补零，enhanced 输出裁回原始长度；
- 空输入（T=0）明确拒绝——按该函数既有契约计入 per-file warning 跳过，不静默产出空文件。

模型用恒等 dummy（mic 谱直接透传为 near-end 估计），仅验证补齐/裁剪/拒绝逻辑，与真实网络无关；
config / checkpoint / torchaudio.load / soundfile.write 全部 monkeypatch，测试不依赖真实音频与权重。
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

import tasks.aec.models
import core.utils.safe_load
from core.utils.inference_artifacts import (
    MANIFEST_SCHEMA_VERSION,
    load_run_manifest,
)
import evaluate.eval_pipeline_common as epc
import tasks.se.models

# 与审计实验一致的 FFT 口径：n_fft=320, hop=160 → 最小输入长度 n_fft//2 + 1 = 161
N_FFT = 320
HOP = 160
STFT_CFG = {"n_fft": N_FFT, "hop_length": HOP, "win_length": N_FFT}
SR = 16000

# 1、hop（补齐前必失败）、hop+1（最小可用长度）、n_fft-1、非 hop 整数倍
SHORT_LENS = [1, HOP, HOP + 1, N_FFT - 1, 500]

FAKE_CFG = {
    "model": {"type": "DummyAEC", "args": {}},
    "stft": STFT_CFG,
    "data": {"sample_rate": SR},
}


class _IdentityAECModel(torch.nn.Module):
    """恒等 AEC 模型：near-end 估计直接回传 mic 谱。"""

    def __init__(self, on_forward=None):
        super().__init__()
        self.checkpoint_probe = torch.nn.Parameter(torch.zeros(1))
        self.on_forward = on_forward

    def forward(self, model_input):
        if self.on_forward is not None:
            callback, self.on_forward = self.on_forward, None
            callback()
        return {"nearend_estimate_spectra": model_input["mic_spectra"]}


def _run_zero_farend(
    monkeypatch,
    tmp_path,
    wav: torch.Tensor,
    *,
    allow_partial: bool = False,
    load_impl=None,
    config_loader=None,
    on_forward=None,
    expected_effective_config_identity=None,
):
    """以伪造 I/O 跑一次 zero-farend AEC 推理，返回 (out_dir, sf.write 捕获)。"""
    captured: dict = {}
    in_dir = tmp_path / "in"
    in_dir.mkdir(exist_ok=True)
    input_file = in_dir / "u1.wav"
    if not input_file.exists():
        input_file.touch()  # 仅需文件存在以被 rglob 发现，内容由 fake load 提供
    config = tmp_path / "fake.yaml"
    checkpoint = tmp_path / "fake.pt"
    config.write_text("model: DummyAEC\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    real_write = sf.write

    def fake_write(path, data, sr, **kwargs):
        captured["path"] = path
        captured["data"] = np.asarray(data).copy()
        captured["sr"] = sr
        real_write(path, data, sr, **kwargs)

    model = _IdentityAECModel(on_forward=on_forward)
    monkeypatch.setattr(
        epc,
        "load_merged_config",
        config_loader or (lambda *a, **k: FAKE_CFG),
    )
    monkeypatch.setattr(tasks.se.models, "build_model", lambda *a, **k: model)
    monkeypatch.setattr(tasks.aec.models, "register_aec_models", lambda *a, **k: None)
    monkeypatch.setattr(
        core.utils.safe_load,
        "safe_torch_load",
        lambda *a, **k: {"model": model.state_dict()},
    )
    monkeypatch.setattr(
        torchaudio,
        "load",
        load_impl or (lambda _path: (wav.clone(), SR)),
    )
    monkeypatch.setattr(sf, "write", fake_write)

    out_dir = epc.run_aec_inference_zero_farend(
        config_path=config,
        checkpoint_path=checkpoint,
        input_dir=in_dir,
        output_dir=tmp_path / "out",
        device="cpu",
        allow_partial=allow_partial,
        expected_effective_config_identity=expected_effective_config_identity,
    )
    return out_dir, captured


@pytest.mark.parametrize("n_samples", SHORT_LENS)
def test_zero_farend_short_input_padded_and_cropped(monkeypatch, tmp_path, n_samples):
    """短输入补零后推理成功，输出为有限值且长度准确等于原长。"""
    rng = np.random.default_rng(0)
    wav = torch.from_numpy(rng.standard_normal(n_samples).astype(np.float32)).unsqueeze(0)
    out_dir, captured = _run_zero_farend(monkeypatch, tmp_path, wav)
    out = captured["data"]
    assert out.shape[-1] == n_samples  # 输出长度准确等于原长
    assert np.isfinite(out).all()  # 输出全为有限值
    # 恒等模型下 STFT->iSTFT 应近似还原原波形
    assert np.allclose(out, wav.squeeze(0).numpy(), atol=1e-4)
    # inf.scp 与标准 manifest 均记录该 utterance，输出结构身份可复核。
    scp_text = (out_dir / "inf.scp").read_text(encoding="utf-8")
    assert "u1" in scp_text
    manifest, results = load_run_manifest(out_dir / "run_manifest.json")
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert [result.status for result in results] == ["ok"]
    assert results[0].output_identity["path"] == results[0].output_path
    assert results[0].output_identity["algorithm"] == "stat-v1"


def test_zero_farend_success_resumes_without_reopening_audio(monkeypatch, tmp_path):
    wav = torch.ones(1, SHORT_LENS[0])
    out_dir, _ = _run_zero_farend(monkeypatch, tmp_path, wav)

    resumed_dir, captured = _run_zero_farend(
        monkeypatch,
        tmp_path,
        wav,
        load_impl=lambda _path: pytest.fail(
            "可信输出恢复时不得重新打开输入音频"
        ),
    )
    manifest, results = load_run_manifest(resumed_dir / "run_manifest.json")

    assert resumed_dir == out_dir
    assert captured == {}
    assert manifest["skipped_uids"] == ["u1"]
    assert [result.status for result in results] == ["skipped"]


def test_zero_farend_empty_input_rejected(monkeypatch, tmp_path):
    """默认 fail closed；显式 allow_partial 后仍记录失败且不产出音频。"""
    with pytest.raises(RuntimeError, match="failed for 1/1"):
        _run_zero_farend(monkeypatch, tmp_path, torch.zeros(1, 0))

    partial_root = tmp_path / "partial"
    partial_root.mkdir()
    out_dir, captured = _run_zero_farend(
        monkeypatch, partial_root, torch.zeros(1, 0), allow_partial=True
    )
    assert "data" not in captured
    assert "u1" not in (out_dir / "inf.scp").read_text(encoding="utf-8")
    manifest = __import__("json").loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["failed_uids"] == ["u1"]
    assert manifest["results"][0]["status"] == "failed"
    assert "empty input waveform" in manifest["results"][0]["error"]


def test_zero_farend_manifest_binds_parent_effective_config_identity(
    monkeypatch, tmp_path
):
    from evaluate.inference_components import effective_config_identity

    expected = effective_config_identity(FAKE_CFG)
    out_dir, _ = _run_zero_farend(
        monkeypatch,
        tmp_path,
        torch.ones(1, SHORT_LENS[0]),
        expected_effective_config_identity=expected,
    )
    manifest, _ = load_run_manifest(out_dir / "run_manifest.json")

    assert manifest["artifact_fingerprint"]["parameters"][
        "effective_config_identity"
    ] == expected


def test_zero_farend_reloads_effective_config_before_manifest_publish(
    monkeypatch, tmp_path
):
    calls = 0

    def changing_loader(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return FAKE_CFG if calls == 1 else {**FAKE_CFG, "stft": {**STFT_CFG, "n_fft": 512}}

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    manifest_path = out_dir / "run_manifest.json"
    old_manifest = b'{"old": true}\n'
    manifest_path.write_bytes(old_manifest)

    with pytest.raises(RuntimeError, match="effective config identity mismatch"):
        _run_zero_farend(
            monkeypatch,
            tmp_path,
            torch.ones(1, SHORT_LENS[0]),
            config_loader=changing_loader,
        )

    assert calls == 2
    assert manifest_path.read_bytes() == old_manifest


def test_zero_farend_rechecks_input_inventory_before_manifest_publish(
    monkeypatch, tmp_path
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    manifest_path = out_dir / "run_manifest.json"
    old_manifest = b'{"old": true}\n'
    manifest_path.write_bytes(old_manifest)
    input_path = tmp_path / "in" / "u1.wav"

    with pytest.raises(RuntimeError, match="input 发布前发生变化"):
        _run_zero_farend(
            monkeypatch,
            tmp_path,
            torch.ones(1, SHORT_LENS[0]),
            on_forward=lambda: input_path.write_bytes(b"changed-input"),
        )

    assert manifest_path.read_bytes() == old_manifest


def test_zero_farend_effective_config_mismatch_fails_before_model_builder(
    monkeypatch, tmp_path
):
    built = []
    monkeypatch.setattr(epc, "load_merged_config", lambda *_args: FAKE_CFG)
    monkeypatch.setattr(tasks.se.models, "build_model", lambda *_a, **_k: built.append(True))
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    input_dir = tmp_path / "input"
    config.write_text("model: DummyAEC\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    input_dir.mkdir()
    (input_dir / "u.wav").touch()

    with pytest.raises(RuntimeError, match="effective config identity mismatch"):
        epc.run_aec_inference_zero_farend(
            config,
            checkpoint,
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            device="cpu",
            expected_effective_config_identity="0" * 64,
        )

    assert built == []


pytestmark = [pytest.mark.unit]
