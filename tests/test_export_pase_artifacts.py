"""export_pase_artifacts 单元测试（纯逻辑，不构造大模型）。

验证训练 checkpoint → 独立 artifact 的前缀剥离、cfg 提取/字段映射、错误处理
（审计 R-04）。不构造 WavLM-Large（315M）等大模型——dewavlm 的 _build_cfg 分支
依赖 WavLM_feat，单独的 round-trip 集成测试需真实训练 checkpoint，此处不覆盖。
"""

from __future__ import annotations

import pytest
import torch

from tools.model_tools import export_pase_artifacts as exporter
from tools.model_tools.export_pase_artifacts import (
    STAGE_SPEC,
    _build_cfg,
    _extract,
    _load_train_ckpt,
    _strip_prefix,
)


def test_strip_prefix():
    sd = {"student.wavlm.a": 1, "teacher.wavlm.b": 2, "student.wavlm.c": 3}
    assert _strip_prefix(sd, "student.wavlm.") == {"a": 1, "c": 3}


def test_strip_prefix_no_match_returns_empty():
    assert _strip_prefix({"foo.x": 1}, "student.wavlm.") == {}


def test_extract_cfm_strips_prefix():
    ckpt = {
        "model": {
            "cfm.transformer.layer": torch.tensor([1.0]),
            "cfm.mel_spec.x": torch.tensor([2.0]),
            "encoder.wavlm.y": torch.tensor([3.0]),
        },
        "config": {"model": {"args": {"cfm_kwargs": {"sigma": 0.0}}}},
    }
    sd, cfg = _extract(ckpt, "cfm")
    assert "transformer.layer" in sd
    assert "mel_spec.x" in sd
    assert "encoder.wavlm.y" not in sd  # encoder 被排除
    assert "y" not in sd
    assert cfg["model"]["args"]["cfm_kwargs"]["sigma"] == 0.0


def test_extract_dewavlm_strips_student_prefix():
    ckpt = {
        "model": {
            "student.wavlm.a": torch.tensor([1.0]),
            "teacher.wavlm.b": torch.tensor([2.0]),  # teacher 必须被排除
        },
        "config": {},
    }
    sd, _ = _extract(ckpt, "dewavlm")
    assert set(sd.keys()) == {"a"}  # teacher.wavlm.* 必须被排除


def test_extract_pase_vocoder_strips_decoder_prefix():
    ckpt = {
        "model_generator": {
            "decoder.decoder.y": torch.tensor([1.0]),
            "decoder.head.z": torch.tensor([2.0]),
            "decoder.proj.w": torch.tensor([3.0]),
            "encoder.wavlm.e": torch.tensor([4.0]),  # encoder 被排除
        },
        "config": {"model": {"generator": {"args": {}}}},
    }
    sd, _ = _extract(ckpt, "pase_vocoder")
    assert "decoder.y" in sd and "head.z" in sd and "proj.w" in sd
    assert "encoder.wavlm.e" not in sd and "e" not in sd


def test_extract_stupase_vocoder_strips_vocoder_prefix():
    ckpt = {
        "model_generator": {
            "vocoder.decoder.y": torch.tensor([1.0]),
            "vocoder.head.z": torch.tensor([2.0]),
            "encoder.x": torch.tensor([3.0]),  # Mel_feat 无参数，但若存在应被排除
        },
        "config": {"model": {"generator": {"args": {}}}},
    }
    sd, _ = _extract(ckpt, "stupase_vocoder")
    assert "decoder.y" in sd and "head.z" in sd
    assert "encoder.x" not in sd and "x" not in sd


def test_build_cfg_cfm():
    config = {"model": {"args": {"cfm_kwargs": {"sigma": 0.0, "audio_drop_prob": 0.5}}}}
    assert _build_cfg("cfm", config) == {"sigma": 0.0, "audio_drop_prob": 0.5}


def test_build_cfg_pase_vocoder_field_mapping():
    config = {"model": {"generator": {"args": {
        "input_channels": 1024, "vocoder_dim": 768, "vocoder_intermediate_dim": 2304,
        "vocoder_num_layers": 12, "n_fft": 1280, "hop_length": 320,
    }}}}
    cfg = _build_cfg("pase_vocoder", config)
    assert cfg == {
        "input_channels": 1024, "dim": 768, "intermediate_dim": 2304,
        "num_layers": 12, "n_fft": 1280, "hop_length": 320,
    }


def test_build_cfg_stupase_vocoder_excludes_mel_kwargs():
    config = {"model": {"generator": {"args": {
        "input_channels": 100, "dim": 768, "intermediate_dim": 2304, "num_layers": 12,
        "n_fft": 1280, "hop_length": 320,
        "mel_kwargs": {"n_mels": 100},  # Mel_feat 的 kwargs，不应进入 VocosVocoder cfg
    }}}}
    cfg = _build_cfg("stupase_vocoder", config)
    assert "mel_kwargs" not in cfg
    assert cfg["input_channels"] == 100 and cfg["dim"] == 768


def test_extract_rejects_wrong_top_key():
    # cfm 期望单模型 "model"，但给了 GAN "model_generator"
    ckpt = {"model_generator": {"cfm.x": torch.tensor([1.0])}, "config": {}}
    with pytest.raises(ValueError, match=r"缺顶层 'model'"):
        _extract(ckpt, "cfm")


def test_extract_rejects_mismatched_prefix():
    ckpt = {"model": {"foo.x": torch.tensor([1.0])}, "config": {}}
    with pytest.raises(ValueError, match=r"前缀不匹配"):
        _extract(ckpt, "cfm")


def test_load_train_ckpt_rejects_non_framework(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"not_a_framework_ckpt": True}, p)
    with pytest.raises(ValueError, match=r"不是本框架"):
        _load_train_ckpt(str(p))


def test_stage_spec_covers_all_artifacts():
    assert set(STAGE_SPEC) == {
        # PASE / StuPASE
        "dewavlm", "dewavlm_r", "pase_vocoder", "stupase_vocoder", "cfm",
        # UniPASE（Stage 1-4）
        "unipase_dewavlm_omni", "unipase_vocoder", "unipase_adapter", "unipase_postnet",
        # GAP-URGENet（Phase 1-2）
        "gap_predictor", "gap_postnet",
    }


# ----------------------------------------------------------------------
# UniPASE stages
# ----------------------------------------------------------------------

def test_extract_unipase_adapter_strips_prefix():
    ckpt = {
        "model_generator": {
            "adapter.embed_p.weight": torch.tensor([1.0]),
            "adapter.head.bias": torch.tensor([2.0]),
            "encoder.wavlm.x": torch.tensor([3.0]),  # 冻结 encoder 被排除
        },
        "config": {"model": {"generator": {"args": {}}}},
    }
    sd, _ = _extract(ckpt, "unipase_adapter")
    assert set(sd.keys()) == {"embed_p.weight", "head.bias"}


def test_extract_unipase_postnet_strips_prefix():
    ckpt = {
        "model_generator": {
            "postnet.conv.weight": torch.tensor([1.0]),
            "adapter.embed_p.weight": torch.tensor([2.0]),  # 冻结链路被排除
            "decoder.head.z": torch.tensor([3.0]),
        },
        "config": {"model": {"generator": {"args": {}}}},
    }
    sd, _ = _extract(ckpt, "unipase_postnet")
    assert set(sd.keys()) == {"conv.weight"}


def test_build_cfg_wavlm_stages_use_layer_norm_mode():
    # WavLM-Large 真实 cfg 为 extractor_mode="layer_norm"（官方 checkpoint cfg 证实），
    # 决定 conv_layers 归一化层嵌套结构（conv_layers.*.2.1.* key）；PASE/StuPASE/UniPASE
    # 三个蒸馏阶段的 artifact 都必须使用同一 cfg。
    for stage in ("dewavlm", "dewavlm_r", "unipase_dewavlm_omni"):
        cfg = _build_cfg(stage, {})
        assert cfg["extractor_mode"] == "layer_norm"
        assert cfg["encoder_layers"] == 24 and cfg["encoder_embed_dim"] == 1024
        assert cfg["max_distance"] == 800


def test_build_cfg_unipase_vocoder_and_adapter():
    config = {"model": {"generator": {"args": {
        "input_channels": 1024, "dim": 1024, "intermediate_dim": 3072, "num_layers": 12,
        "num_res": 4, "num_attn": 1, "n_fft": 1280, "hop_length": 320,
        "output_channels": 1024, "encoder_ckpt_path": "/x/y.pt",
    }}}}
    voc_cfg = _build_cfg("unipase_vocoder", config)
    assert voc_cfg["dim"] == 1024 and voc_cfg["n_fft"] == 1280
    assert "encoder_ckpt_path" not in voc_cfg and "output_channels" not in voc_cfg
    adp_cfg = _build_cfg("unipase_adapter", config)
    assert adp_cfg["output_channels"] == 1024 and "n_fft" not in adp_cfg
    assert "encoder_ckpt_path" not in adp_cfg


def test_build_cfg_unipase_postnet_uses_postnet_kwargs():
    config = {"model": {"generator": {"args": {
        "postnet_kwargs": {"n_layers": 5, "lstm_hidden_units": 100},
        "encoder_ckpt_path": "/x/y.pt",
    }}}}
    assert _build_cfg("unipase_postnet", config) == {"n_layers": 5, "lstm_hidden_units": 100}


# ----------------------------------------------------------------------
# GAP-URGENet stages
# ----------------------------------------------------------------------

def test_extract_gap_predictor_empty_prefix_keeps_all_keys():
    ckpt = {
        "model": {
            "conv.0.weight": torch.tensor([1.0]),
            "blocks.0.lstm.weight": torch.tensor([2.0]),
            "deconv.weight": torch.tensor([3.0]),
        },
        "config": {"model": {"args": {"n_layers": 10}}},
    }
    sd, cfg = _extract(ckpt, "gap_predictor")
    # 空前缀：整体复制，key 不做任何裁剪
    assert set(sd.keys()) == {"conv.0.weight", "blocks.0.lstm.weight", "deconv.weight"}
    assert cfg["model"]["args"]["n_layers"] == 10


def test_extract_gap_predictor_rejects_empty_state_dict():
    ckpt = {"model": {}, "config": {"model": {"args": {}}}}
    with pytest.raises(ValueError, match=r"state_dict 为空"):
        _extract(ckpt, "gap_predictor")


def test_extract_gap_postnet_strips_postnet_prefix():
    ckpt = {
        "model_generator": {
            "postnet.conv.0.weight": torch.tensor([1.0]),
            "postnet.blocks.0.lstm.weight": torch.tensor([2.0]),
            "encoder.wavlm.x": torch.tensor([3.0]),   # 冻结 encoder 被排除
            "adapter.embed_p.weight": torch.tensor([4.0]),  # 冻结 adapter 被排除
        },
        "config": {"model": {"generator": {"args": {}}}},
    }
    sd, _ = _extract(ckpt, "gap_postnet")
    assert set(sd.keys()) == {"conv.0.weight", "blocks.0.lstm.weight"}


def test_build_cfg_gap_predictor_uses_model_args():
    config = {"model": {"args": {
        "n_layers": 10, "lstm_hidden_units": 200, "emb_dim": 96,
    }}}
    assert _build_cfg("gap_predictor", config) == {
        "n_layers": 10, "lstm_hidden_units": 200, "emb_dim": 96,
    }
    # 返回副本，调用方修改 artifact cfg 不应污染训练 config
    cfg = _build_cfg("gap_predictor", config)
    cfg["n_layers"] = 1
    assert config["model"]["args"]["n_layers"] == 10


def test_build_cfg_gap_postnet_uses_postnet_kwargs():
    config = {"model": {"generator": {"args": {
        "postnet_kwargs": {
            "n_imics": 2, "n_layers": 5, "normalization_mode": "paper_variance",
        },
        "encoder_ckpt_path": "/x/y.pt",
    }}}}
    # normalization_mode 是本仓库扩展键；artifact cfg 必须剔除，才能被官方
    # TFGridNet 构造器直接加载。本仓库默认值为 legacy_centered。
    assert _build_cfg("gap_postnet", config) == {
        "n_imics": 2, "n_layers": 5,
    }
    assert config["model"]["generator"]["args"]["postnet_kwargs"]["normalization_mode"] == "paper_variance"


def test_export_verifies_staging_before_replacing_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "artifact.pt"
    output.write_bytes(b"old artifact")
    monkeypatch.setattr(exporter, "_load_train_ckpt", lambda _path: {"config": {}})
    monkeypatch.setattr(
        exporter,
        "_extract",
        lambda _ckpt, _stage: ({"weight": torch.tensor([1.0])}, {}),
    )
    monkeypatch.setattr(exporter, "_build_cfg", lambda _stage, _config: {})

    def fail_verify(_stage, artifact_path):
        assert artifact_path != str(output)
        raise RuntimeError("verification failed")

    monkeypatch.setattr(exporter, "_verify", fail_verify)
    with pytest.raises(RuntimeError, match="verification failed"):
        exporter.export("cfm", "train.pt", str(output), verify=True)

    assert output.read_bytes() == b"old artifact"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["artifact.pt"]


def test_export_rejects_symlink_output_without_touching_referent(tmp_path, monkeypatch):
    referent = tmp_path / "referent.pt"
    referent.write_bytes(b"old artifact")
    output = tmp_path / "artifact.pt"
    output.symlink_to(referent)
    monkeypatch.setattr(exporter, "_load_train_ckpt", lambda _path: {"config": {}})
    monkeypatch.setattr(
        exporter,
        "_extract",
        lambda _ckpt, _stage: ({"weight": torch.tensor([1.0])}, {}),
    )
    monkeypatch.setattr(exporter, "_build_cfg", lambda _stage, _config: {})

    with pytest.raises(RuntimeError, match="symbolic link"):
        exporter.export("cfm", "train.pt", str(output), verify=False)

    assert referent.read_bytes() == b"old artifact"


pytestmark = [pytest.mark.unit]


def test_wavlm_large_cfg_matches_official_dropout():
    """导出的 DeWavLM cfg 必须与官方 artifact 一致：dropout=0。"""
    cfg = exporter._wavlm_large_cfg()
    assert cfg["dropout"] == 0.0
    assert cfg["attention_dropout"] == 0.0
