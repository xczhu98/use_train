"""OnlineSpatialNet generic exporter release gate contract tests."""

import importlib
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "model",
    ["streaming_online_spatialnet", "online_spatialnet", "online_spatial"],
)
def test_generic_export_rejects_online_spatialnet_release_before_model_build(
    monkeypatch, tmp_path, model,
):
    export_module = importlib.import_module("onnx.export")

    monkeypatch.setattr(
        export_module,
        "parse_args",
        lambda: SimpleNamespace(
            config=str(tmp_path / "model.yaml"),
            model=model,
            checkpoint=str(tmp_path / "model.ckpt"),
            save_onnx=str(tmp_path / "release.onnx"),
            output=None,
        ),
    )
    monkeypatch.setattr(
        export_module,
        "load_merged_config",
        lambda _path: pytest.fail(
            "generic exporter built OnlineSpatialNet before dedicated three-way verifier"
        ),
    )

    with pytest.raises(
        ValueError,
        match="generic onnx/export.py --save-onnx.*no dedicated OnlineSpatialNet verifier",
    ):
        export_module.main()

    assert not (tmp_path / "release.onnx").exists()
