"""Tests for tools/shared/scp_io.py atomic-write permission handling + audio_io validation reasons."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.shared.scp_io import write_scp_entries


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_atomic_write_preserves_existing_mode(tmp_path):
    """重写已有 SCP 必须继承原文件权限（回归：曾误用 mkstemp 的 0600）。"""
    target = tmp_path / "a.scp"
    target.write_text("old\n")
    os.chmod(target, 0o644)
    write_scp_entries([("u1", "p1")], target)
    assert _mode(target) == 0o644

    os.chmod(target, 0o600)  # 用户自定义更严格权限也应保留
    write_scp_entries([("u2", "p2")], target)
    assert _mode(target) == 0o600


def test_atomic_write_new_file_default_mode(tmp_path):
    """新文件默认 0644（mkstemp 0600 过严）。"""
    target = tmp_path / "b.scp"
    write_scp_entries([("u1", "p1")], target)
    assert _mode(target) == 0o644


def test_load_audio_validation_reason(tmp_path):
    """load_audio 质量检查失败带结构化 reason（AudioValidationError）。"""
    from tools.shared.audio_io import AudioValidationError, load_audio

    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(8000, dtype=np.float32), 16000)
    with pytest.raises(AudioValidationError) as ei:
        load_audio(str(silent))
    assert ei.value.reason == "too_silent"

    loud = tmp_path / "loud.wav"
    # 显式 FLOAT 子类型，避免默认 PCM 写入时把 >1 的幅度截断到 1.0
    sf.write(loud, np.ones(8000, dtype=np.float32) * 2.0, 16000, subtype="FLOAT")
    with pytest.raises(AudioValidationError) as ei:
        load_audio(str(loud))
    assert ei.value.reason == "peak_overflow"

    # 仍是 ValueError 子类，兼容既有 except ValueError
    try:
        load_audio(str(silent))
    except ValueError as e:
        assert getattr(e, "reason", None) == "too_silent"
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

pytestmark = [pytest.mark.unit]
