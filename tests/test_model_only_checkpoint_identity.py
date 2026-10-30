"""model-only checkpoint 的来源身份与自动输出隔离回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.utils.inference_artifacts import fingerprint_file
from tools.model_tools import extract_model_checkpoint as extractor

pytestmark = pytest.mark.unit


def _write_training_checkpoint(path: Path, value: float) -> None:
    torch.save({"model": {"weight": torch.tensor([value])}}, path)


def test_default_paths_preserve_only_best_model_tar_compatibility(tmp_path):
    names = (
        "best_model.tar",
        "best_model.pth",
        "model_0060.pth",
        "model_0060.tar",
        "best_model",
    )

    paths = {name: extractor.default_model_only_path(tmp_path / name).name for name in names}

    assert paths == {
        "best_model.tar": "best_model.model_only.pt",
        "best_model.pth": "best_model.pth.model_only.pt",
        "model_0060.pth": "model_0060.pth.model_only.pt",
        "model_0060.tar": "model_0060.tar.model_only.pt",
        "best_model": "best_model.model_only.pt.model_only.pt",
    }
    assert len(set(paths.values())) == len(paths)


def test_rejects_canonical_model_only_checkpoint_as_training_source(tmp_path):
    source = tmp_path / "model_0060.pth.model_only.pt"
    _write_training_checkpoint(source, 1.0)

    with pytest.raises(ValueError, match="不能作为训练来源"):
        extractor.materialize_model_only_checkpoint(source, family="single_model")


def test_automatic_target_refuses_existing_artifact_bound_to_other_source(tmp_path):
    first_source = tmp_path / "first.pth"
    second_source = tmp_path / "second.pth"
    _write_training_checkpoint(first_source, 1.0)
    _write_training_checkpoint(second_source, 2.0)
    first_artifact = extractor.materialize_model_only_checkpoint(
        first_source, family="single_model"
    )
    second_target = extractor.default_model_only_path(second_source)
    second_target.write_bytes(first_artifact.read_bytes())

    with pytest.raises(FileExistsError, match="绑定其他来源"):
        extractor.materialize_model_only_checkpoint(
            second_source, family="single_model"
        )


def test_automatic_target_refreshes_when_same_source_content_changes(tmp_path):
    source = tmp_path / "model_0060.pth"
    _write_training_checkpoint(source, 1.0)
    artifact = extractor.materialize_model_only_checkpoint(source, family="single_model")
    original_fingerprint = fingerprint_file(artifact)

    _write_training_checkpoint(source, 2.0)
    refreshed = extractor.materialize_model_only_checkpoint(
        source, family="single_model"
    )

    assert refreshed == artifact
    assert fingerprint_file(refreshed) != original_fingerprint
    payload = extractor.validate_model_only_checkpoint(
        refreshed, source_path=source, family="single_model"
    )
    torch.testing.assert_close(payload["model"]["weight"], torch.tensor([2.0]))


def test_automatic_target_refreshes_same_source_validation_mismatch(tmp_path):
    source = tmp_path / "best_model.pth"
    _write_training_checkpoint(source, 3.0)
    artifact = extractor.default_model_only_path(source)
    torch.save(
        {
            "schema": extractor.MODEL_ONLY_SCHEMA,
            "family": "gan_generator",
            "source_fingerprint": fingerprint_file(source),
            "model_generator": {"weight": torch.tensor([99.0])},
        },
        artifact,
    )

    refreshed = extractor.materialize_model_only_checkpoint(
        source, family="single_model"
    )

    assert refreshed == artifact
    payload = extractor.validate_model_only_checkpoint(
        refreshed, source_path=source, family="single_model"
    )
    assert payload["family"] == "single_model"
    torch.testing.assert_close(payload["model"]["weight"], torch.tensor([3.0]))
