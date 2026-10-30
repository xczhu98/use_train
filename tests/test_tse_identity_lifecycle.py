"""TSE 推理、评测与 ONNX 身份衔接的最小回归。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.inference.artifact_binding import verify_file_bindings
from core.utils.inference_artifacts import fingerprint_file
from evaluate.eval_pipeline_tse import _public_inference_artifact_binding
from tasks.tse.inference.artifacts import TSEInferenceArtifactSnapshot


pytestmark = pytest.mark.unit


def _snapshot(tmp_path: Path, *, with_speaker: bool) -> TSEInferenceArtifactSnapshot:
    model = tmp_path / "model.tar"
    config = tmp_path / "config.yaml"
    model.write_bytes(b"model-loaded-by-tse")
    config.write_bytes(b"config-loaded-by-tse")
    speaker = tmp_path / "speaker.pt"
    speaker.write_bytes(b"speaker-loaded-by-tse")
    model_identity = fingerprint_file(model)
    config_identity = {**fingerprint_file(config), "kind": "file"}
    speaker_identity = fingerprint_file(speaker) if with_speaker else None
    return TSEInferenceArtifactSnapshot(
        meta={
            "artifacts": {
                "model": model_identity,
                "config": config_identity,
                "speaker_checkpoint": speaker_identity,
            }
        },
        mode="loaded",
    )


def test_snapshot_exposes_only_loader_used_source_identities(tmp_path: Path):
    snapshot = _snapshot(tmp_path, with_speaker=True)

    assert snapshot.source_file_identities() == {
        "checkpoint": fingerprint_file(tmp_path / "model.tar"),
        "config": fingerprint_file(tmp_path / "config.yaml"),
        "speaker_checkpoint": fingerprint_file(tmp_path / "speaker.pt"),
    }


def test_public_binding_uses_loaded_identities_without_rehashing(tmp_path: Path):
    snapshot = _snapshot(tmp_path, with_speaker=True)
    identities = snapshot.source_file_identities()
    binding = _public_inference_artifact_binding(snapshot, {})

    assert binding["files"] == identities
    (tmp_path / "model.tar").write_bytes(b"model-replaced-after-load")
    with pytest.raises(RuntimeError, match="source artifact changed"):
        verify_file_bindings(binding)


def test_unused_speaker_identity_is_not_published(tmp_path: Path):
    snapshot = _snapshot(tmp_path, with_speaker=False)

    assert snapshot.source_file_identities() == {
        "checkpoint": fingerprint_file(tmp_path / "model.tar"),
        "config": fingerprint_file(tmp_path / "config.yaml"),
    }
