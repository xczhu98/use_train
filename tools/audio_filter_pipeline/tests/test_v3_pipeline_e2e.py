"""Generation-3 七步 artifact 管线端到端覆盖。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
import soundfile as sf

from audio_filter_pipeline.config import PipelineConfig
from audio_filter_pipeline.processors.stages_v3.audio_export import Step6AudioExporter
from audio_filter_pipeline.processors.stages_v3.base import ArtifactStage, StageRunContext
from audio_filter_pipeline.processors.stages_v3.export import Step6MultirateExportStage
from audio_filter_pipeline.processors.stages_v3.runner import V3PipelineRunner
from audio_filter_pipeline.processors.stages_v3.select import Step3MosSelectionStage
from audio_filter_pipeline.processors.stages_v3.speaker_shadow import Step5SpeakerShadowStage
from audio_filter_pipeline.utils.artifacts import ArtifactIndex, PipelineStep, iter_artifact_rows


pytestmark = pytest.mark.unit


def _config(output_root: Path, input_root: Path) -> PipelineConfig:
    return PipelineConfig(
        pipeline_generation=3,
        step="step6",
        output_root=str(output_root),
        datasets={"alpha": {
            "input_root": str(input_root),
            "language": "en",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "speaker": {"source": "none", "required": False},
        }},
        selection_policies={
            "global": {"mos_models": ["scoreq", "urgent_pk", "dnsmos"], "min_mos_models": 3},
            "quality_profiles": {
                name: {"thresholds": {
                    "scoreq": 4.2,
                    "urgent_pk": 3.5,
                    "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
                }}
                for name in ("quality_balanced", "quality_high", "quality_premium")
            },
        },
    )


class _ForwardingStage(ArtifactStage):
    def __init__(self, step: PipelineStep, rows: list[dict[str, Any]], upstream_step: Optional[PipelineStep] = None):
        self.step = step
        if upstream_step is None:
            upstream_step = None if step is PipelineStep.STEP0 else PipelineStep(
                f"step{int(step.value[-1]) - 1}"
            )
        self.upstream_step = upstream_step
        self._rows = rows

    def config_payload(self, _config: PipelineConfig) -> Mapping[str, Any]:
        return {"test_stage": self.step.value}

    def produce_rows(
        self,
        _context: StageRunContext,
        _upstream_index: Optional[ArtifactIndex],
        _upstream_rows: Iterable[Mapping[str, Any]],
    ) -> Iterable[Mapping[str, Any]]:
        yield from self._rows


def test_v3_seven_step_chain_preserves_rejects_but_exports_only_kept_step6_rows(tmp_path):
    source = tmp_path / "input" / "speaker" / "source.wav"
    source.parent.mkdir(parents=True)
    waveform = np.linspace(-0.5, 0.5, 48_000, dtype=np.float32)
    sf.write(source, waveform, 48_000, subtype="FLOAT")

    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    config = _config(tmp_path / "out", source.parent.parent)
    shared = {
        "dataset_id": "alpha",
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "input_root": str(source.parent.parent),
        "original_path": str(source),
        "original_relative_path": "speaker",
        "source_relative_path": "speaker/source.wav",
        "source_size_bytes": source.stat().st_size,
        "sample_rate": 48_000,
        "frames": 48_000,
        "channels": 1,
        "format": "WAV",
        "subtype": "FLOAT",
        "duration_sec": 1.0,
        "source_sha256": source_sha256,
        "speaker_id": "speaker-a",
        "speaker_scope": "alpha",
        "source_start_sample": 0,
        "source_end_sample": 48_000,
        "signal_finite": True,
        "signal_zero_ratio": 0.0,
        "signal_peak": 0.5,
        "signal_clipping_ratio": 0.0,
        "signal_dc_offset": 0.0,
        "signal_rms_dbfs": -8.0,
        "signal_num_samples": 48_000,
        "vad_speech_ratio": 1.0,
        "vad_timeline_sample_rate": 16_000,
        "vad_timeline_num_samples": 3_000,
        "boundary_reason": "test_fixture",
        "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
        "mos_scoring_policy_hash": config.mos_scoring_projection(
            "alpha", "en",
        )["scoring_policy_hash"],
    }
    rows = [
        {
            **shared,
            "source_key": "source-a",
            "segment_id": "segment-keep",
            "start_sample": 0,
            "end_sample": 24_000,
            "bandwidth_hz": 3_900.0,
            "estimated_sr": 8_000,
            "mos_scores": {
                "scoreq": 4.5, "urgent_pk": 3.8,
                "dnsmos": {"ovrl": 3.5, "sig": 3.6, "bak": 4.2, "p808": 4.0},
            },
        },
        {
            **shared,
            "source_key": "source-a",
            "segment_id": "segment-reject",
            "start_sample": 24_000,
            "end_sample": 48_000,
            "bandwidth_hz": 11_900.0,
            "estimated_sr": 24_000,
            "mos_scores": {
                "scoreq": 2.0, "urgent_pk": 2.0,
                "dnsmos": {"ovrl": 2.0, "sig": 2.0, "bak": 2.0, "p808": 2.0},
            },
        },
    ]

    factories = {
        step: (lambda step: lambda _config, _dataset: _ForwardingStage(step, rows))(step)
        for step in (PipelineStep.STEP0, PipelineStep.STEP1, PipelineStep.STEP2, PipelineStep.STEP3)
    }
    # Step4 是独立的 SE 筛选链（forwarding 保持分数行），Step5 直接挂 Step3。
    factories[PipelineStep.STEP4] = lambda _config, _dataset: _ForwardingStage(
        PipelineStep.STEP4, rows, upstream_step=PipelineStep.STEP3,
    )
    factories[PipelineStep.STEP5] = lambda _config, _dataset: Step5SpeakerShadowStage()
    factories[PipelineStep.STEP6] = lambda _config, _dataset: Step6MultirateExportStage()

    result = V3PipelineRunner(config, factories).run("alpha", PipelineStep.STEP6)
    published_rows = list(iter_artifact_rows(result.target))

    assert tuple(result) == (
        PipelineStep.STEP0, PipelineStep.STEP1, PipelineStep.STEP2,
        PipelineStep.STEP3, PipelineStep.STEP5, PipelineStep.STEP6,
    )
    assert "mos_selection" not in published_rows[0]
    assert [row["speaker_shadow"]["status"] for row in published_rows] == ["disabled", "disabled"]
    assert [row["final_selection"] for row in published_rows] == [
        {"version": 3, "export_eligible": True, "reason": "mos_selected"},
        {"version": 3, "export_eligible": False, "reason": "mos_rejected"},
    ]
    assert [row["target_sample_rate"] for row in published_rows] == [16_000, 24_000]
    assert all(row["export_routing"]["audio_write"] is False for row in published_rows)

    exported = Step6AudioExporter(str(tmp_path / "audio")).export_artifact(result.target)

    assert [item.target_sample_rate for item in exported] == [16_000]
    audio, sample_rate = sf.read(exported[0].output_path, dtype="float32", always_2d=True)
    assert sample_rate == exported[0].target_sample_rate
    assert audio.shape == (8_000, 1)
    assert not list((tmp_path / "audio" / "audio_exports").glob(
        "step6/en/alpha/*/24000hz/speaker/source--segment-reject.flac"
    ))
