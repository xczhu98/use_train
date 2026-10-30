"""TSE sidecar 基于公共 InferenceReport 发布的合同测试。"""

from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from core.inference import InferenceReport
from core.utils.inference_artifacts import InferenceResult, stat_audio_identity
from tasks.tse.inference import artifacts
from tasks.tse.inference.sidecar import publish_sidecars_from_report

pytestmark = pytest.mark.unit


def _expected(input_path: Path) -> dict:
    return artifacts._artifact_meta(
        {"inputs": {"mixture": artifacts.stat_audio_artifact(input_path)}}
    )


def test_sidecar_publication_is_derived_from_report(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "wav"
    output_dir.mkdir()
    source = tmp_path / "mix.wav"
    sf.write(source, np.zeros(16, dtype=np.float32), 16000)
    output = output_dir / "utt.wav"
    sf.write(output, np.ones(16, dtype=np.float32), 16000)
    expected = _expected(source)
    result = InferenceResult(
        uid="utt",
        status="ok",
        output_path=str(output.resolve()),
        output_identity=stat_audio_identity(output),
    )
    report = InferenceReport(
        status="completed",
        results=(result,),
        manifest_path=tmp_path / "run_manifest.json",
        scp_path=tmp_path / "inf.scp",
    )

    calls = []
    monkeypatch.setattr(
        artifacts,
        "write_output_sidecar_locked",
        lambda sidecar, output_path, metadata: calls.append((sidecar, output_path, metadata)),
    )
    published = publish_sidecars_from_report(
        report, {"utt": expected}, output_dir=output_dir
    )

    sidecar = output.with_name("utt.meta.json")
    assert published == {"utt": sidecar}
    assert calls == [(sidecar, output, expected)]


def test_sidecar_does_not_publish_failed_report_item(tmp_path: Path):
    report = InferenceReport(
        status="failed",
        results=(InferenceResult(uid="bad", status="failed", error="boom"),),
        manifest_path=tmp_path / "run_manifest.json",
        scp_path=tmp_path / "inf.scp",
    )
    assert publish_sidecars_from_report(report, {}, output_dir=tmp_path) == {}
