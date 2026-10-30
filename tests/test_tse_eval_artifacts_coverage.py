"""TSE 评测 artifact、manifest 与 coverage 的纯 unit 契约测试。"""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from core.inference import ArtifactBinding, FileInferenceItem, InferenceSpec, ReusePolicy, run_inference_report
from core.utils.inference_artifacts import (
    fingerprint_file,
    fingerprint_json,
    stat_audio_identity,
)
from tasks.tse.inference import artifacts as tse_inference_artifacts
from evaluate.aggregate_eval_summary import format_coverage, load_coverage
from evaluate.artifact_lifecycle import CURRENT_FILENAME, load_committed_artifact
from evaluate.common import validate_result_manifest
import evaluate.eval_pipeline_tse as eval_pipeline_tse
from tasks.tse.inference import TSEInferenceAdapter, publish_sidecars_from_report
from evaluate.eval_pipeline_tse import (
    _bind_input_artifacts,
    _coverage_status,
    _meta_matches,
    _model_meta,
    _write_inference_manifest,
    _write_tse_summary,
)


class _SnapshotShim:
    def __init__(self, meta):
        self.meta = meta

    def bind_inputs(self, paths):
        artifacts = dict(self.meta["artifacts"])
        artifacts["inputs"] = {
            name: tse_inference_artifacts.stat_audio_artifact(path)
            for name, path in sorted(paths.items())
        }
        return tse_inference_artifacts._artifact_meta(artifacts)


def _run_public_inference(rows, extractor, wav_dir, mapping, artifact_meta, model, config, allow_partial=False):
    snapshot = _SnapshotShim(artifact_meta)
    wav_dir = Path(wav_dir)
    expected = {}
    items = []
    for row in rows:
        uid = eval_pipeline_tse._make_uid(row)
        mixture = Path(mapping[row["mixture_utterance"]])
        enrollment = Path(mapping[row["enrolment_speakers_utterance"]])
        expected[uid] = snapshot.bind_inputs({"mixture": mixture, "speaker_waveform": enrollment})
        items.append(FileInferenceItem(
            uid=uid,
            inputs={"mixture": mixture, "speaker_waveform": enrollment},
            output=wav_dir / f"{uid}.wav",
        ))
    spec = InferenceSpec(
        config=model,
        checkpoint=model,
        output_dir=wav_dir,
        items=items,
        artifacts=ArtifactBinding(
            files={"checkpoint": model, "config": model},
            parameters={"task": "tse", "artifact_meta": artifact_meta},
        ),
        reuse=ReusePolicy(enabled=True),
        allow_partial=allow_partial,
    )
    report = run_inference_report(spec, TSEInferenceAdapter(extractor))
    publish_sidecars_from_report(report, expected, output_dir=wav_dir)
    by_uid = {result.uid: result for result in report.results}
    output = []
    for row in rows:
        result = by_uid[eval_pipeline_tse._make_uid(row)]
        current = dict(row)
        current["tse_path"] = result.output_path
        current["inference_status"] = "ok" if result.status in {"ok", "skipped"} else "failed"
        current["inference_reused"] = result.status == "skipped"
        output.append(current)
    return output


def _row() -> dict:
    return {
        "dataset": "mock",
        "mixture_utterance": "mix",
        "enrolment_speakers_utterance": "enroll",
    }


def _reject_audio_payload_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """禁止 artifact 校验为 WAV/FLAC 额外打开或计算内容 hash。"""
    from core.utils import inference_artifacts

    audio_suffixes = {".wav", ".flac"}
    original_builtin_open = builtins.open
    original_path_open = Path.open
    original_fingerprint_file = inference_artifacts.fingerprint_file

    def _is_audio(path) -> bool:
        return Path(os.fspath(path)).suffix.lower() in audio_suffixes

    def guarded_builtin_open(path, *args, **kwargs):
        if _is_audio(path):
            raise AssertionError(f"音频 payload 不得被额外 open: {path}")
        return original_builtin_open(path, *args, **kwargs)

    def guarded_path_open(path, *args, **kwargs):
        if _is_audio(path):
            raise AssertionError(f"音频 payload 不得被额外 Path.open: {path}")
        return original_path_open(path, *args, **kwargs)

    def guarded_fingerprint_file(path):
        if _is_audio(path):
            raise AssertionError(f"音频 payload 不得计算 SHA-256: {path}")
        return original_fingerprint_file(path)

    def reject_soundfile_info(path, *args, **kwargs):
        raise AssertionError(f"音频 payload 不得额外调用 sf.info: {path}")

    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(inference_artifacts, "fingerprint_file", guarded_fingerprint_file)
    monkeypatch.setattr(sf, "info", reject_soundfile_info)


def test_fingerprint_binds_model_config_parameters_and_audio_stat_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = tmp_path / "model.tar"
    config = tmp_path / "config.yaml"
    mix = tmp_path / "mix.wav"
    enroll = tmp_path / "enroll.wav"
    model.write_bytes(b"model-v1")
    config.write_text("model: v1\n", encoding="utf-8")
    speaker = tmp_path / "speaker.pt"
    speaker.write_bytes(b"speaker-v1")
    mix.write_bytes(b"mix-v1")
    enroll.write_bytes(b"enroll-v1")
    speaker_identity = fingerprint_file(speaker)
    _reject_audio_payload_access(monkeypatch)

    base = _model_meta(
        str(model),
        str(config),
        {"apply_vad": False},
        speaker_checkpoint=speaker,
        speaker_checkpoint_identity=speaker_identity,
    )
    first = _bind_input_artifacts(base, {"mixture": mix, "speaker_waveform": enroll})
    assert first["schema"] == "tse-inference-artifact-v6"
    assert first["artifacts"]["inputs"] == {
        "speaker_waveform": stat_audio_identity(enroll),
        "mixture": stat_audio_identity(mix),
    }
    assert _meta_matches(dict(first), first)

    linked_mix = tmp_path / "linked-mix.wav"
    linked_mix.symlink_to(mix)
    with pytest.raises(ValueError, match="不允许符号链接"):
        _bind_input_artifacts(base, {"mixture": linked_mix, "speaker_waveform": enroll})

    mix_stat = mix.stat()
    os.utime(mix, ns=(mix_stat.st_atime_ns, mix_stat.st_mtime_ns + 1))
    changed_input = _bind_input_artifacts(
        base, {"mixture": mix, "speaker_waveform": enroll}
    )
    assert not _meta_matches(first, changed_input)
    assert not _meta_matches({**first, "schema": "tse-inference-artifact-v3"}, first)
    assert not _meta_matches(
        {"model": str(model), "model_mtime": model.stat().st_mtime}, first
    )

    changed_params = _model_meta(str(model), str(config), {"apply_vad": True})
    assert changed_params["fingerprint"] != base["fingerprint"]


def test_realt_dataset_identity_binds_release_and_all_consumed_annotations(
    tmp_path: Path,
):
    package = tmp_path / "REAL-TSE"
    root = package / "REAL-T-eval2"
    meta_dir = root / "EVAL2"
    timing = meta_dir / "json" / "unseen_EN"
    timing.mkdir(parents=True)
    meta = meta_dir / "unseen_EN_meta.csv"
    meta.write_text(
        "mixture_utterance,enrolment_speakers_utterance\nmix,enroll\n",
        encoding="utf-8",
    )
    mapping = root / "mapping.csv"
    mapping.write_text("utterance,path\nmix,mixtures/mix.wav\n", encoding="utf-8")
    scene = meta_dir / "scene_channel.csv"
    scene.write_text("mixture_utterance,enrolment_speakers_utterance\nmix,enroll\n")
    overlap = timing / "overlap_records.json"
    overlap.write_text("{}\n", encoding="utf-8")
    (package / "README.md").write_text(
        "# REAL-T: benchmark\n\n**Release:** v1.0.0 (2026-08-23).\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        meta=[str(meta)],
        mapping_csv=str(mapping),
        audio_root=str(root),
        timing_gt_json_base_dir=str(meta_dir / "json"),
    )
    rows = [{"dataset": "unseen_EN"}]

    identity = eval_pipeline_tse._build_dataset_identity(args, rows)

    assert identity["parameters"]["benchmark"] == "REAL-T"
    assert identity["parameters"]["release"] == "v1.0.0"
    assert identity["parameters"]["release_date"] == "2026-08-23"
    assert identity["parameters"]["split"] == "EVAL2"
    assert set(identity["files"]) == {
        "meta:unseen_EN",
        "mapping",
        "timing:unseen_EN:overlap_records.json",
        "annotation:EVAL2:scene_channel.csv",
        "release:README.md",
    }

    model = tmp_path / "model.tar"
    torch.save({"model": {}}, model)
    metadata = _model_meta(str(model), dataset_identity=identity)
    assert metadata["artifacts"]["dataset"] == identity
    meta.write_text(meta.read_text(encoding="utf-8") + "mix2,enroll2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dataset.*changed"):
        tse_inference_artifacts.verify_artifact_meta(metadata)


def test_realt_dataset_identity_rejects_mixed_release_roots(tmp_path: Path):
    roots = [tmp_path / "release-a" / "REAL-T-dev", tmp_path / "release-b" / "REAL-T-dev"]
    metas = []
    for root in roots:
        meta = root / "DEV" / "AMI_meta.csv"
        meta.parent.mkdir(parents=True)
        meta.write_text(
            "mixture_utterance,enrolment_speakers_utterance\nmix,enroll\n",
            encoding="utf-8",
        )
        metas.append(str(meta))

    with pytest.raises(ValueError, match="different release roots"):
        eval_pipeline_tse._build_dataset_identity(
            SimpleNamespace(
                meta=metas,
                mapping_csv=None,
                audio_root=None,
                timing_gt_json_base_dir=None,
            ),
            [{"dataset": "AMI"}],
        )


@pytest.mark.parametrize(
    "changed_source",
    [
        "tasks/tse/inference/artifacts.py",
        "tasks/tse/inference/extractor.py",
        "tasks/tse/models/mock_model.py",
        "core/stft.py",
        "core/models/build.py",
        "core/utils/config.py",
        "core/utils/registry.py",
        "core/utils/safe_load.py",
        "core/utils/inference_artifacts.py",
        "tasks/tse/model_io.py",
    ],
)
def test_tse_implementation_fingerprint_binds_explicit_source_boundaries(
    tmp_path: Path, changed_source: str
):
    source_paths = (
        "tasks/tse/inference/artifacts.py",
        "tasks/tse/inference",
        "tasks/tse/models",
        "core/stft.py",
        "core/models/build.py",
        "core/utils/config.py",
        "core/utils/registry.py",
        "core/utils/safe_load.py",
        "core/utils/inference_artifacts.py",
        "tasks/tse/model_io.py",
    )
    source_files = [
        "tasks/tse/inference/artifacts.py",
        "tasks/tse/inference/extractor.py",
        "tasks/tse/models/mock_model.py",
        "core/stft.py",
        "core/models/build.py",
        "core/utils/config.py",
        "core/utils/registry.py",
        "core/utils/safe_load.py",
        "core/utils/inference_artifacts.py",
        "tasks/tse/model_io.py",
    ]
    for relative in source_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"SOURCE = {relative!r}\n", encoding="utf-8")

    model = tmp_path / "model.tar"
    model.write_bytes(b"model")
    first = _model_meta(
        str(model),
        implementation_project_root=tmp_path,
        implementation_source_paths=source_paths,
    )
    changed = tmp_path / changed_source
    changed.write_text(
        changed.read_text(encoding="utf-8") + "VERSION = 2\n", encoding="utf-8"
    )
    second = _model_meta(
        str(model),
        implementation_project_root=tmp_path,
        implementation_source_paths=source_paths,
    )

    assert first["fingerprint"] != second["fingerprint"]
    files = first["artifacts"]["inference_implementation"]["files"]
    assert "implementation:tasks/tse/inference/artifacts.py" in files
    assert "implementation:tasks/tse/inference/extractor.py" in files
    assert "implementation:tasks/tse/models/mock_model.py" in files
    assert "implementation:core/stft.py" in files


def test_speaker_checkpoint_content_changes_fingerprint(tmp_path: Path):
    model = tmp_path / "model.tar"
    speaker = tmp_path / "speaker.pt"
    model.write_bytes(b"model")
    speaker.write_bytes(b"speaker-v1")
    first_identity = fingerprint_file(speaker)

    first = _model_meta(
        str(model),
        speaker_checkpoint=speaker,
        speaker_checkpoint_identity=first_identity,
    )
    speaker.write_bytes(b"speaker-v2")
    second_identity = fingerprint_file(speaker)
    second = _model_meta(
        str(model),
        speaker_checkpoint=speaker,
        speaker_checkpoint_identity=second_identity,
    )

    assert first["fingerprint"] != second["fingerprint"]
    assert (
        first["artifacts"]["speaker_checkpoint"]["sha256"]
        != second["artifacts"]["speaker_checkpoint"]["sha256"]
    )
    assert not _meta_matches(first, second)


def test_model_meta_rejects_speaker_checkpoint_without_loader_identity(tmp_path: Path):
    model = tmp_path / "model.tar"
    speaker = tmp_path / "speaker.pt"
    model.write_bytes(b"model")
    speaker.write_bytes(b"speaker")

    with pytest.raises(ValueError, match="identity captured by its loader"):
        _model_meta(str(model), speaker_checkpoint=speaker)


def test_loaded_speaker_identity_is_not_replaced_by_post_loader_file_change(
    tmp_path: Path
):
    model = tmp_path / "model.tar"
    speaker = tmp_path / "speaker.pt"
    model.write_bytes(b"model")
    speaker.write_bytes(b"speaker-v1")
    loaded_identity = fingerprint_file(speaker)
    speaker.write_bytes(b"speaker-v2-after-loader")

    metadata = _model_meta(
        str(model),
        speaker_checkpoint=speaker,
        speaker_checkpoint_identity=loaded_identity,
    )

    assert metadata["artifacts"]["speaker_checkpoint"] == loaded_identity
    assert metadata["artifacts"]["speaker_checkpoint"] != fingerprint_file(speaker)


def test_loaded_model_identity_is_not_replaced_by_post_loader_file_change(
    tmp_path: Path
):
    model = tmp_path / "model.tar"
    model.write_bytes(b"model-v1")
    loaded_identity = fingerprint_file(model)
    model.write_bytes(b"model-v2-after-loader")

    metadata = _model_meta(
        str(model),
        model_checkpoint=model,
        model_checkpoint_identity=loaded_identity,
    )

    assert metadata["artifacts"]["model"] == loaded_identity
    assert metadata["artifacts"]["model"] != fingerprint_file(model)


def test_current_model_checkpoint_change_invalidates_loaded_sidecar(tmp_path: Path):
    model = tmp_path / "model.tar"
    model.write_bytes(b"model-v1")
    loaded = _model_meta(str(model))
    model.write_bytes(b"model-v2")
    current = _model_meta(str(model))

    assert loaded["fingerprint"] != current["fingerprint"]
    assert not _meta_matches(loaded, current)


def test_model_dir_artifact_binds_avg_model_checkpoint(tmp_path: Path):
    model_dir = tmp_path / "model-dir"
    model_dir.mkdir()
    checkpoint = model_dir / "avg_model.pt"
    checkpoint.write_bytes(b"average-model")

    metadata = _model_meta(str(model_dir))

    assert metadata["artifacts"]["model"] == fingerprint_file(checkpoint)


def test_skip_inspection_rejects_config_change_during_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = tmp_path / "model.tar"
    config = tmp_path / "config.yaml"
    torch.save({"model": {}, "resolved_config": {}}, model)
    config.write_text("model:\n  args: {}\n", encoding="utf-8")
    original_loader = tse_inference_artifacts.load_merged_config

    def mutating_loader(path):
        loaded = original_loader(path)
        config.write_text("model:\n  args:\n    changed: true\n", encoding="utf-8")
        return loaded

    monkeypatch.setattr(tse_inference_artifacts, "load_merged_config", mutating_loader)

    with pytest.raises(RuntimeError, match="模型配置在加载期间发生变化"):
        tse_inference_artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
            model, config, {"device": "cpu", "apply_vad": False}
        )


def test_resolved_config_identity_changes_when_base_content_changes(tmp_path: Path):
    model = tmp_path / "model.tar"
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    torch.save({"model": {}, "resolved_config": {}}, model)
    base.write_text("model:\n  args:\n    feature_dim: 16\n", encoding="utf-8")
    child.write_text("base_config: base.yaml\n", encoding="utf-8")

    first = eval_pipeline_tse._load_model_artifacts_for_skip(model, child)
    first_meta = _model_meta(
        str(model),
        str(first[1]),
        model_checkpoint=first[3],
        model_checkpoint_identity=first[4],
        config_identity=first[2],
        resolved_config_identity=first[5],
    )
    base.write_text("model:\n  args:\n    feature_dim: 32\n", encoding="utf-8")
    second = eval_pipeline_tse._load_model_artifacts_for_skip(model, child)
    second_meta = _model_meta(
        str(model),
        str(second[1]),
        model_checkpoint=second[3],
        model_checkpoint_identity=second[4],
        config_identity=second[2],
        resolved_config_identity=second[5],
    )

    assert first[2] == second[2]
    assert first[5] != second[5]
    assert first_meta["fingerprint"] != second_meta["fingerprint"]
    assert not _meta_matches(first_meta, second_meta)


def test_relative_config_artifact_path_uses_current_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    caller = tmp_path / "caller"
    caller.mkdir()
    config = caller / "same-name.yaml"
    config.write_text("model: caller\n", encoding="utf-8")
    repo_named = eval_pipeline_tse._PROJECT_ROOT / "same-name.yaml"
    monkeypatch.chdir(caller)

    resolved = eval_pipeline_tse._model_config_path_for_artifacts(
        tmp_path / "model.tar", "same-name.yaml"
    )

    assert resolved == config.resolve()
    assert resolved != repo_named


def test_speaker_checkpoint_artifact_is_explicitly_none_when_unused(tmp_path: Path):
    model = tmp_path / "model.tar"
    model.write_bytes(b"model")

    metadata = _model_meta(str(model))

    assert metadata["artifacts"]["speaker_checkpoint"] is None


def test_tse_cli_rejects_partial_checkpoint_loading(monkeypatch, tmp_path: Path):
    argv = [
        "eval_pipeline_tse.py",
        "--meta",
        str(tmp_path / "meta.csv"),
        "--model",
        str(tmp_path / "model.tar"),
        "--allow_partial_load",
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        eval_pipeline_tse.parse_args()


def test_inference_manifest_records_expected_ok_and_failed(tmp_path: Path):
    rows = [
        {**_row(), "inference_status": "ok", "tse_path": "/tmp/a.wav"},
        {
            **_row(),
            "enrolment_speakers_utterance": "other",
            "inference_status": "failed",
            "inference_error": "RuntimeError: boom",
        },
    ]
    path = tmp_path / "manifest.json"
    payload = _write_inference_manifest(rows, path)
    assert payload["expected"] == 2
    assert payload["ok"] == 1
    assert payload["failed"] == 1
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_run_inference_is_fail_closed_and_allow_partial_is_explicit(tmp_path: Path):
    model = tmp_path / "model.tar"
    config = tmp_path / "config.yaml"
    torch.save({"resolved_config": {}}, model)
    config.write_text("x: 1\n", encoding="utf-8")
    sf.write(tmp_path / "mix.wav", np.zeros(16, dtype=np.float32), 16000)
    sf.write(tmp_path / "enroll.wav", np.zeros(16, dtype=np.float32), 16000)

    class BrokenExtractor:
        sample_rate = 16000

        def declared_cue_roles(self):
            return {"speaker_waveform": SimpleNamespace(required=True)}

        def extract_speech_with_cues(self, *_args, **_kwargs):
            raise RuntimeError("boom")

        def prepare_speaker_waveform_cue(self, pcm, sr):
            return pcm[0] if getattr(pcm, "dim", lambda: 1)() == 2 else pcm

    mapping = {
        "mix": str(tmp_path / "mix.wav"),
        "enroll": str(tmp_path / "enroll.wav"),
    }
    base = _model_meta(str(model), None, {"apply_vad": False})
    with pytest.raises(RuntimeError, match="boom"):
        _run_public_inference(
            [_row()], BrokenExtractor(), tmp_path / "strict" / "wav",
            mapping, base, model, None
        )
    rows = _run_public_inference(
        [_row()], BrokenExtractor(), tmp_path / "partial" / "wav",
        mapping, base, model, None, allow_partial=True,
    )
    assert rows[0]["inference_status"] == "failed"
    manifest = json.loads(
        (tmp_path / "partial" / "wav" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["expected_uids"] and len(manifest["failed_uids"]) == 1
    assert not manifest["ok_uids"]


def test_run_inference_reuses_only_matching_output_stat_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = tmp_path / "model.tar"
    config = tmp_path / "config.yaml"
    mix = tmp_path / "mix.wav"
    enroll = tmp_path / "enroll.wav"
    torch.save({"resolved_config": {}}, model)
    config.write_text("x: 1\n", encoding="utf-8")
    sf.write(mix, np.zeros(16, dtype=np.float32), 16000)
    sf.write(enroll, np.zeros(16, dtype=np.float32), 16000)

    class CountingExtractor:
        sample_rate = 16000

        def __init__(self):
            self.calls = 0

        def declared_cue_roles(self):
            return {"speaker_waveform": SimpleNamespace(required=True)}

        def extract_speech_with_cues(self, *_args, **_kwargs):
            self.calls += 1
            return np.full(16, self.calls / 10, dtype=np.float32)

        def prepare_speaker_waveform_cue(self, pcm, sr):
            return pcm[0] if getattr(pcm, "dim", lambda: 1)() == 2 else pcm

    extractor = CountingExtractor()
    mapping = {"mix": str(mix), "enroll": str(enroll)}
    base = _model_meta(str(model), None, {"apply_vad": False})
    wav_dir = tmp_path / "eval" / "wav"

    initial = _run_public_inference([_row()], extractor, wav_dir, mapping, base, model, None)
    output = Path(initial[0]["tse_path"])
    sidecar_path = output.with_name(output.stem + ".meta.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["schema"] == "tse-inference-artifact-v6"
    assert sidecar["output"] == stat_audio_identity(output)

    with monkeypatch.context() as payload_guard:
        _reject_audio_payload_access(payload_guard)
        reused = _run_public_inference([_row()], extractor, wav_dir, mapping, base, model, None)
    assert extractor.calls == 1
    assert reused[0]["inference_reused"] is True

    output_stat = output.stat()
    os.utime(output, ns=(output_stat.st_atime_ns, output_stat.st_mtime_ns + 1))
    with monkeypatch.context() as payload_guard:
        _reject_audio_payload_access(payload_guard)
        assert not _meta_matches(sidecar, sidecar, output)

    rerun = _run_public_inference([_row()], extractor, wav_dir, mapping, base, model, config)
    assert extractor.calls == 2
    assert rerun[0]["inference_reused"] is False


def test_requested_backend_skip_does_not_degrade_status_or_coverage(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(eval_pipeline_tse, "_build_ter_recognizers", lambda _args: None)
    metrics = ["ter"]
    rows = [{**_row(), "ground_truth_transcript": "hello", "tse_path": None}]

    _, summary = eval_pipeline_tse.run_evaluation(
        rows, SimpleNamespace(), metrics, None, {}, eval_pipeline_tse.torch.device("cpu")
    )

    assert metrics == []
    assert summary["status"] == "done"
    assert summary["coverage"] == {}
    assert summary["metrics"]["requested"] == ["ter"]
    assert summary["metrics"]["active"] == []
    assert summary["metrics"]["skipped"] == [
        {"metric": "ter", "reason": "TER backend or language model unavailable"}
    ]


@pytest.mark.parametrize(
    ("meta_row", "max_samples"),
    [("", None), ("mix,enroll\n", 0)],
    ids=["empty-meta", "max-samples-zero"],
)
def test_main_rejects_zero_selected_rows_with_committed_error_lifecycle(
    tmp_path: Path, monkeypatch, meta_row: str, max_samples: int | None
):
    model = tmp_path / "model.tar"
    meta = tmp_path / "meta.csv"
    output_dir = tmp_path / "eval"
    model.write_bytes(b"model")
    meta.write_text(
        "mixture_utterance,enrolment_speakers_utterance\n" + meta_row,
        encoding="utf-8",
    )
    args = SimpleNamespace(
        metrics="sisdr",
        device="cpu",
        output_dir=str(output_dir),
        meta=[str(meta)],
        max_samples=max_samples,
        mapping_csv=None,
        model=str(model),
        config=None,
        apply_vad=False,
        skip_inference=False,
        audio_root=None,
        ref_root=None,
        allow_partial=False,
    )
    monkeypatch.setattr(eval_pipeline_tse, "parse_args", lambda: args)

    assert eval_pipeline_tse.main() != 0
    payload = json.loads(
        (output_dir / "tse_eval_summary.json").read_text(encoding="utf-8")
    )
    envelope = load_committed_artifact(output_dir)
    current = json.loads(
        (output_dir / CURRENT_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["summary"]["status"] == "error"
    assert envelope["status"] == "error"
    assert current["status"] == "committed"
    assert payload["run_id"] == payload["summary"]["run_id"]
    assert payload["run_id"] == envelope["run_id"] == current["run_id"]


def test_main_rejects_negative_max_samples_before_reading_meta(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "eval"
    args = SimpleNamespace(
        metrics="sisdr",
        device="cpu",
        output_dir=str(output_dir),
        meta=[str(tmp_path / "meta.csv")],
        max_samples=-1,
    )
    monkeypatch.setattr(eval_pipeline_tse, "parse_args", lambda: args)
    monkeypatch.setattr(
        eval_pipeline_tse,
        "_read_meta",
        lambda _paths: pytest.fail("negative max_samples must fail before meta read"),
    )

    with pytest.raises(SystemExit, match="--max_samples must be >= 0"):
        eval_pipeline_tse.main()

    assert not (output_dir / "tse_eval_summary.json").exists()
    assert not (output_dir / CURRENT_FILENAME).exists()


def test_main_runs_active_sisdr_and_skips_ter_with_default_success(
    tmp_path: Path, monkeypatch
):
    model = tmp_path / "model.tar"
    mix = tmp_path / "mix.wav"
    enroll = tmp_path / "enroll.wav"
    target = tmp_path / "target.wav"
    output_dir = tmp_path / "eval"
    speaker = tmp_path / "configured-speaker.pt"
    torch.save({"model": {}, "resolved_config": {"fixture": True}}, model)
    speaker.write_bytes(b"speaker-v1")
    (tmp_path / "meta.csv").write_text("fixture\n", encoding="utf-8")
    signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 320, endpoint=False)).astype(
        np.float32
    )
    sf.write(mix, signal, 16000)
    sf.write(enroll, signal[:160], 16000)
    sf.write(target, signal, 16000)

    args = SimpleNamespace(
        metrics="sisdr,ter",
        device="cpu",
        output_dir=str(output_dir),
        meta=[str(tmp_path / "meta.csv")],
        max_samples=None,
        mapping_csv=None,
        model=str(model),
        config=None,
        apply_vad=False,
        skip_inference=False,
        audio_root=None,
        ref_root=None,
        allow_partial=False,
    )

    class FakeExtractor:
        sample_rate = 16000
        speaker_encoder_pretrained = "configured-speaker.pt"
        external_speaker_encoder_pretrained = str(speaker.resolve())
        model_checkpoint_path = str(model.resolve())
        loaded_model_checkpoint_identity = fingerprint_file(model)
        config_path = None
        loaded_config_identity = None
        resolved_config_identity = {
            "algorithm": "sha256-json-v1",
            "sha256": fingerprint_json({"fixture": True}),
        }

        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_external_speaker_encoder(self):
            return fingerprint_file(speaker)

        def loaded_artifact_snapshot(self):
            return {
                "model_checkpoint": self.model_checkpoint_path,
                "model_checkpoint_identity": self.loaded_model_checkpoint_identity,
                "config_path": self.config_path,
                "config_identity": self.loaded_config_identity,
                "resolved_config_identity": self.resolved_config_identity,
                "speaker_encoder_pretrained": self.speaker_encoder_pretrained,
                "speaker_checkpoint": self.external_speaker_encoder_pretrained,
                "speaker_checkpoint_identity": self.prepare_external_speaker_encoder(),
            }

        def extract_speech(self, *_args):
            return signal.copy()

        def declared_cue_roles(self):
            return {"speaker_waveform": SimpleNamespace(required=True)}

        def extract_speech_with_cues(self, *_args, **_kwargs):
            return signal.copy()

        def prepare_speaker_waveform_cue(self, pcm, sr):
            return pcm[0] if getattr(pcm, "dim", lambda: 1)() == 2 else pcm

    row = {
        **_row(),
        "target_utterance": "target",
        "ground_truth_transcript": "hello",
    }
    mapping = {
        "mix": str(mix),
        "enroll": str(enroll),
        "target": str(target),
    }
    monkeypatch.setattr(eval_pipeline_tse, "parse_args", lambda: args)
    monkeypatch.setattr(eval_pipeline_tse, "_read_meta", lambda _paths: [row])
    monkeypatch.setattr(eval_pipeline_tse, "_load_mapping", lambda _path: mapping)
    monkeypatch.setattr("tasks.tse.inference.TSEExtractor", FakeExtractor)
    monkeypatch.setattr(eval_pipeline_tse, "_build_ter_recognizers", lambda _args: None)

    assert eval_pipeline_tse.main() == 0
    payload = json.loads(
        (output_dir / "tse_eval_summary.json").read_text(encoding="utf-8")
    )
    summary = payload["summary"]
    assert args._model_spk_pretrained == "configured-speaker.pt"
    assert summary["status"] == "done"
    assert summary["metrics"] == {
        "requested": ["sisdr", "ter"],
        "active": ["sisdr"],
        "skipped": [
            {"metric": "ter", "reason": "TER backend or language model unavailable"}
        ],
    }
    assert summary["coverage"]["sisdr"] == {
        "eligible": 1,
        "attempted": 1,
        "computed": 1,
        "failed": 0,
        "coverage": 1.0,
    }
    assert np.isfinite(summary["sisdr"])
    assert summary["inference"]["ok"] == 1
    envelope = json.loads(
        (output_dir / "evaluation_artifact.json").read_text(encoding="utf-8")
    )
    assert envelope["task"] == "tse"
    assert envelope["status"] == "done"
    assert envelope["provenance"]["artifact_fingerprint"]["artifacts"][
        "speaker_checkpoint"
    ]["path"] == str(speaker.resolve())
    assert envelope["inference_coverage"] == summary["inference"]
    assert envelope["metric_coverage"] == summary["coverage"]
    manifest = validate_result_manifest(
        output_dir, expected_run_id=payload["run_id"]
    )
    assert set(manifest["files"]) == {"tse_eval_results.csv"}
    assert summary["result_manifest"] == envelope["summary"]["result_manifest"]


def test_summary_current_commit_reverifies_global_artifacts(tmp_path: Path):
    output_dir = tmp_path / "eval"
    model = tmp_path / "model.tar"
    config = {"model": {"args": {}}, "stft": {"n_fft": 512}}
    torch.save({"model": {}, "resolved_config": config}, model)
    snapshot = tse_inference_artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
        model, None, {"device": "cpu", "apply_vad": False}
    )
    model.write_bytes(b"changed-before-current")

    with pytest.raises(RuntimeError, match="model checkpoint changed"):
        _write_tse_summary(
            output_dir,
            {
                "pipeline": "eval_pipeline_tse",
                "model": str(model),
                "summary": {"status": "error", "coverage": {}},
            },
            model=model,
            artifact_fingerprint=snapshot.meta,
        )

    current = json.loads(
        (output_dir / CURRENT_FILENAME).read_text(encoding="utf-8")
    )
    assert current["status"] == "in_progress"


def test_aggregate_coverage_and_legacy_unknown(tmp_path: Path):
    status, coverage = load_coverage(tmp_path)
    assert status == "unknown"
    assert coverage == {}

    payload = {
        "summary": {
            "status": "partial",
            "coverage": {
                "ter": {
                    "eligible": 4,
                    "attempted": 3,
                    "computed": 2,
                    "failed": 1,
                    "coverage": 0.5,
                }
            },
        }
    }
    (tmp_path / "tse_eval_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    status, coverage = load_coverage(tmp_path)
    assert status == "partial"
    assert format_coverage(coverage["ter"]) == (
        "eligible=4 attempted=3 computed=2 failed=1 coverage=50.00%"
    )


def test_coverage_status_done_partial_and_error():
    assert _coverage_status({
        "ter": {"eligible": 2, "computed": 2},
        "stoi": {"eligible": 0, "computed": 0},
    }) == "done"
    assert _coverage_status({
        "ter": {"eligible": 2, "computed": 1},
    }) == "partial"
    assert _coverage_status({
        "ter": {"eligible": 2, "computed": 0},
    }) == "error"


pytestmark = [pytest.mark.unit]
