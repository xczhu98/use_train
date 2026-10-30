"""Follow-up tests for the 2026-07 review fixes.

Covers previously untested behaviors:
- registry 冲突注册拒绝（core/utils/registry.py）
- inference._collect_files 嵌套 uid 与重复 uid 守卫
- tools/shared/pickle_io.py 受限反序列化
- RWSAMambaUNetGenerator 奇数 T pad/crop
- evaluate_audio_quality 跳过缺失 ref（且不做多余解码）
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch


# ----------------------------------------------------------------------
# registry 冲突注册
# ----------------------------------------------------------------------

def test_registry_lazy_map_conflict_raises():
    from core.utils.registry import Registry

    reg = Registry("TEST_REG")
    reg.set_lazy_map({"Foo": "some.module.a"})
    # 同名同路径：幂等，不报错
    reg.set_lazy_map({"Foo": "some.module.a"})
    # 同名不同路径：必须拒绝（防静默覆盖）
    with pytest.raises(ValueError, match="Foo"):
        reg.set_lazy_map({"Foo": "some.module.b"})


# ----------------------------------------------------------------------
# inference uid 规则与重复守卫
# ----------------------------------------------------------------------

def test_collect_files_nested_uid_and_dup_guard(tmp_path):
    from inference import _collect_files

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "x.wav").write_bytes(b"0")
    (tmp_path / "b" / "x.wav").write_bytes(b"0")
    (tmp_path / "top.wav").write_bytes(b"0")
    files = _collect_files(input_dir=str(tmp_path))
    uids = sorted(u for u, _ in files)
    assert uids == ["a_x", "b_x", "top"]


def test_collect_files_dup_uid_raises(tmp_path):
    from inference import _collect_files

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.wav").write_bytes(b"0")   # -> uid a_b
    (tmp_path / "a_b.wav").write_bytes(b"0")       # -> uid a_b（冲突）
    with pytest.raises(ValueError, match="重复 uid"):
        _collect_files(input_dir=str(tmp_path))


def test_collect_files_scp_is_strict_and_rejects_duplicate_uid(tmp_path):
    from inference import _collect_files

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"0")
    scp = tmp_path / "input.scp"
    scp.write_text(f"dup {audio}\ndup {audio}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复 uid"):
        _collect_files(input_scp=str(scp))

    scp.write_text("broken-only-one-column\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SCP 格式错误"):
        _collect_files(input_scp=str(scp))


def test_collect_files_scp_rejects_unsafe_uid_and_missing_path(tmp_path):
    from inference import _collect_files

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"0")
    scp = tmp_path / "input.scp"
    scp.write_text(f"../escape {audio}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非法 uid"):
        _collect_files(input_scp=str(scp))

    scp.write_text(f"u1 {tmp_path / 'missing.wav'}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="SCP path 不存在"):
        _collect_files(input_scp=str(scp))


def test_collect_files_scp_root_escape_fails_closed(tmp_path):
    from inference import _collect_files

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"0")
    scp = tmp_path / "input.scp"
    scp.write_text(f"u1 {outside}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="超出 --scp_root_dir"):
        _collect_files(input_scp=str(scp), scp_root_dir=str(root))


def test_collect_files_scp_resolves_relative_audio_from_scp_directory(
    tmp_path, monkeypatch
):
    from inference import _collect_files

    scp_dir = tmp_path / "scp"
    audio_dir = scp_dir / "audio"
    audio_dir.mkdir(parents=True)
    audio = audio_dir / "sample.wav"
    audio.write_bytes(b"0")
    scp = scp_dir / "input.scp"
    scp.write_text("utt audio/sample.wav\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    assert _collect_files(input_scp=str(scp)) == [("utt", str(audio.resolve()))]


def test_collect_files_scp_public_writer_quote_roundtrip(tmp_path):
    from inference import _collect_files
    from tools.shared.scp_io import write_scp_entries

    root = tmp_path / "audio root"
    root.mkdir()
    two_col_audio = root / "two col's audio.wav"
    four_col_audio = root / "four col audio.wav"
    two_col_audio.write_bytes(b"0")
    four_col_audio.write_bytes(b"0")

    two_col_scp = tmp_path / "two_col.scp"
    write_scp_entries([("u2", str(two_col_audio))], two_col_scp)
    assert _collect_files(
        input_scp=str(two_col_scp), scp_root_dir=str(root)
    ) == [("u2", str(two_col_audio.resolve()))]

    four_col_scp = tmp_path / "four_col.scp"
    write_scp_entries(
        [("u4", 16000, 1, str(four_col_audio))],
        four_col_scp,
        four_col=True,
    )
    assert _collect_files(
        input_scp=str(four_col_scp), scp_root_dir=str(root)
    ) == [("u4", str(four_col_audio.resolve()))]


def test_collect_files_scp_shlex_contract_rejects_bad_quotes_and_columns(tmp_path):
    from inference import _collect_files

    scp = tmp_path / "input.scp"
    scp.write_text("u1 'unterminated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SCP 格式错误.*引号不匹配"):
        _collect_files(input_scp=str(scp))

    scp.write_text("u1 16000 unexpected.wav\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SCP 格式错误.*仅支持 2/4 列"):
        _collect_files(input_scp=str(scp))


# ----------------------------------------------------------------------
# pickle_io 受限反序列化
# ----------------------------------------------------------------------

@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_io_restricted_roundtrip(tmp_path, monkeypatch, protocol):
    from tools.shared.pickle_io import load_pkl_file

    p = tmp_path / "ok.pkl"
    fixture = {
        "rir": np.zeros(8, dtype=np.float32),
        "scalar": np.float32(1.25),
        "dtype": np.dtype("<f4"),
        "sr": 16000,
    }
    with open(p, "wb") as f:
        pickle.dump(fixture, f, protocol=protocol)
    monkeypatch.delenv("PREPARE_SCP_ALLOW_UNSAFE_PICKLE", raising=False)
    out = load_pkl_file(p, purpose="load test RIR fixture")
    assert out["sr"] == 16000
    assert np.array_equal(out["rir"], fixture["rir"])
    assert out["scalar"] == fixture["scalar"]
    assert out["dtype"] == fixture["dtype"]


class _NumpyCtypesLibraryEvil:
    def __reduce__(self):
        return np.ctypeslib.load_library, ("malicious", ".")


def test_pickle_io_rejects_numpy_ctypeslib_load_library_in_find_class(
    tmp_path, monkeypatch,
):
    from tools.shared.pickle_io import load_pkl_file

    p = tmp_path / "numpy_ctypeslib_evil.pkl"
    with open(p, "wb") as f:
        pickle.dump(_NumpyCtypesLibraryEvil(), f)
    monkeypatch.delenv("PREPARE_SCP_ALLOW_UNSAFE_PICKLE", raising=False)

    with pytest.raises(
        pickle.UnpicklingError,
        match=r"forbidden global numpy\.ctypeslib\.load_library",
    ):
        load_pkl_file(p, purpose="verify numpy global allowlist")


class _Evil:
    pass


def _write_evil_pickle(path):
    with open(path, "wb") as f:
        pickle.dump(_Evil(), f)


def test_pickle_io_rejects_custom_class(tmp_path, monkeypatch):
    from tools.shared.pickle_io import load_pkl_file

    p = tmp_path / "evil.pkl"
    _write_evil_pickle(p)
    monkeypatch.delenv("PREPARE_SCP_ALLOW_UNSAFE_PICKLE", raising=False)
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        load_pkl_file(p, purpose="verify restricted unpickler rejection")


def test_pickle_io_unsafe_requires_nonempty_purpose(tmp_path, monkeypatch):
    from tools.shared.pickle_io import load_pkl_file

    p = tmp_path / "evil.pkl"
    _write_evil_pickle(p)
    monkeypatch.setenv("PREPARE_SCP_ALLOW_UNSAFE_PICKLE", "1")

    with pytest.raises(ValueError, match="non-empty purpose"):
        load_pkl_file(p)
    with pytest.raises(ValueError, match="non-empty purpose"):
        load_pkl_file(p, purpose="   ")


def test_pickle_io_unsafe_with_purpose_loads_trusted_pickle(
    tmp_path, monkeypatch, caplog
):
    from tools.shared.pickle_io import load_pkl_file

    p = tmp_path / "evil.pkl"
    _write_evil_pickle(p)
    monkeypatch.setenv("PREPARE_SCP_ALLOW_UNSAFE_PICKLE", "yes")

    out = load_pkl_file(p, purpose="load trusted review fixture")
    assert isinstance(out, _Evil)
    assert "Purpose: load trusted review fixture" in caplog.text


# ----------------------------------------------------------------------
# RWSAMambaUNet 奇数 T
# ----------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="RWSA mamba 实现依赖 Triton CUDA kernel")
def test_rwsa_mamba_unet_odd_t():
    from tasks.se.models.gan.generators.rwsa_mamba_unet import RWSAMambaUNetGenerator

    torch.manual_seed(0)
    model = RWSAMambaUNetGenerator(num_tfmamba=1).eval().cuda()
    f = model.n_fft // 2 + 1
    x = torch.randn(1, f, 255, 2, device="cuda")  # 奇数 T
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape  # 输出契约 (B, F, T, 2)，T 裁回原长


# ----------------------------------------------------------------------
# 长音频分块推理（_chunked_waveform_forward）
# ----------------------------------------------------------------------

class _IdentityModel(torch.nn.Module):
    def forward(self, wav, lengths=None, *, fs=None):
        return wav


class _NormIdentityModel(torch.nn.Module):
    """带 sr_out / normalize_output 的 identity（模拟 UniPASE 接口）。"""

    sr_out = 16000

    def forward(self, wav, lengths=None, *, fs=None, normalize_output=True):
        if normalize_output:
            peak = wav.abs().amax(dim=-1, keepdim=True)
            return wav / (peak + 1e-8) * peak
        return wav


class _ResamplingStub(torch.nn.Module):
    """仅模拟每块输出长度按采样率 floor 缩放。"""

    sr_out = 16000

    def forward(self, wav, lengths=None, *, fs=None):
        out_len = int(wav.shape[-1] * self.sr_out / fs)
        return wav.new_zeros(wav.shape[0], out_len)


def test_chunked_forward_identity_reconstruction():
    from inference import _chunked_waveform_forward

    torch.manual_seed(0)
    fs = 16000
    for dur in (0.5, 1.0, 2.3, 5.7):  # 覆盖 单块/多块/短尾 情形
        wav = torch.randn(1, int(dur * fs))
        y = _chunked_waveform_forward(_IdentityModel(), wav, fs, chunk_seconds=1.0, multi_sr=True)
        assert y.shape[-1] == wav.shape[-1], (dur, y.shape, wav.shape)
        assert torch.equal(y, wav), f"dur={dur} 拼接不还原"


@pytest.mark.parametrize("seg", [1, 2, 5, 6])
def test_chunked_forward_identity_exact_for_tiny_odd_and_even_segments(seg):
    from inference import _chunked_waveform_forward

    fs = 100
    wav = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    y = _chunked_waveform_forward(
        _IdentityModel(), wav, fs, chunk_seconds=seg / fs, multi_sr=True,
    )

    assert y.shape == wav.shape
    assert torch.equal(y, wav)


def test_chunked_forward_cross_sample_rate_exact_output_length():
    from inference import _chunked_waveform_forward

    fs = 22050
    n = 220623
    wav = torch.zeros(1, n)
    y = _chunked_waveform_forward(
        _ResamplingStub(), wav, fs, chunk_seconds=0.5, multi_sr=True,
    )

    assert y.shape == (1, 160089)
    assert y.shape[-1] == int(n * _ResamplingStub.sr_out / fs)


def test_chunked_forward_rejects_blocks_too_short_for_resampled_target():
    from inference import _chunked_waveform_forward

    fs = 22050
    with pytest.raises(ValueError, match="chunk output too short"):
        _chunked_waveform_forward(
            _ResamplingStub(), torch.zeros(1, 8), fs,
            chunk_seconds=1 / fs, multi_sr=True,
        )


def test_chunked_forward_global_peak_norm():
    from inference import _chunked_waveform_forward

    fs = 16000
    wav = torch.randn(1, 3 * fs) * 0.5
    y = _chunked_waveform_forward(_NormIdentityModel(), wav, fs, chunk_seconds=1.0, multi_sr=True)
    # 全局峰值归一只做一次：输出峰值 == 输入峰值
    assert torch.isclose(y.abs().max(), wav.abs().max(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("chunk_seconds", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_chunked_forward_rejects_nonpositive_or_nonfinite_chunk_seconds(chunk_seconds):
    from inference import _chunked_waveform_forward

    with pytest.raises(ValueError, match="chunk_seconds"):
        _chunked_waveform_forward(
            _IdentityModel(), torch.zeros(1, 16), fs=16000,
            chunk_seconds=chunk_seconds, multi_sr=True,
        )


def test_chunked_forward_rejects_subsample_chunk_seconds():
    from inference import _chunked_waveform_forward

    with pytest.raises(ValueError, match="at least 1 sample"):
        _chunked_waveform_forward(
            _IdentityModel(), torch.zeros(1, 16), fs=16000,
            chunk_seconds=0.5 / 16000, multi_sr=True,
        )


def test_evaluate_audio_quality_skips_missing_ref(tmp_path, monkeypatch):
    import evaluate.common as ec

    # 用桩替代真实音频解码与指标计算，只验证跳过逻辑与 IO 顺序
    loaded = []
    monkeypatch.setattr(ec, "load_audio", lambda p: (loaded.append(p) or (np.zeros(160, dtype=np.float32), 16000)))
    monkeypatch.setattr(ec, "compute_metrics_batch", lambda *a, **k: {})

    inf = tmp_path / "inf.scp"
    ref = tmp_path / "ref.scp"
    inf.write_text("u1 /a/u1.wav\nu2 /a/u2.wav\n")
    ref.write_text("u1 /b/u1.wav\n")  # u2 缺失
    out = ec.evaluate_audio_quality(inf, ref)
    assert out == {}
    # 缺失 ref 的 u2 不应被解码（检查先于 load_audio）
    assert loaded == ["/a/u1.wav", "/b/u1.wav"]

pytestmark = [pytest.mark.unit]
