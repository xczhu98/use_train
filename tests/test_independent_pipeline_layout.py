"""独立音频管道的目录边界与 canonical 入口契约。"""
from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "tools" / "audio_filter_pipeline"
NOISE = ROOT / "tools" / "noise_filter_pipeline"


def test_audio_pipeline_has_single_config_boundary_and_canonical_cli():
    assert (AUDIO / "cli" / "main.py").is_file()
    assert (AUDIO / "cli" / "__main__.py").is_file()
    assert (AUDIO / "config" / "schema.py").is_file()
    assert (AUDIO / "config" / "selection_policy.py").is_file()
    assert not (AUDIO / "config.py").exists()
    assert not (AUDIO / "main.py").exists()
    assert not (AUDIO / "selection_policy.py").exists()


def test_noise_pipeline_separates_cli_and_document_generation():
    assert (NOISE / "cli" / "main.py").is_file()
    assert (NOISE / "cli" / "__main__.py").is_file()
    assert (NOISE / "scripts" / "generate_pipeline_html.py").is_file()
    assert not (NOISE / "main.py").exists()
    assert not (NOISE / "generate_noise_pipeline_html.py").exists()


def test_pipeline_architecture_docs_match_layout():
    for pipeline in (AUDIO, NOISE):
        doc = (pipeline / "ARCHITECTURE.md").read_text(encoding="utf-8")
        assert "canonical" in doc
        assert "兼容转发" not in doc
