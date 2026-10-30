"""评估流水线 SCP provenance、manifest 与 coverage 的 unit 契约测试。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import evaluate.common as eval_common
import evaluate.eval_io as eval_io
import evaluate.eval_pipeline_aec as pipeline_aec
import evaluate.eval_pipeline_common as pipeline_common
import evaluate.eval_pipeline_se as pipeline_se
import evaluate.process_runner as process_runner
import tools.ops.reporting.experiment_report as experiment_report
from core.utils.inference_artifacts import (
    InferenceResult,
    atomic_write_json,
    build_content_fingerprint,
    build_input_inventory,
    build_run_manifest,
    stat_audio_identity,
)
from evaluate.eval_io import read_scp
from evaluate.inference_components import effective_config_identity
from evaluate.eval_pipeline_common import (
    copy_scp_with_provenance,
    generate_scp_from_dir,
    prepare_testset_scps,
)
from tools.shared.scp_provenance import scp_provenance_fresh


def _write_wav(path: Path) -> None:
    sf.write(path, np.zeros(8, dtype=np.float32), 16000)


def _write_manifest(
    enhanced_dir: Path,
    *,
    expected: list[str],
    ok: list[str],
    skipped: list[str] | None = None,
    failed: list[str] | None = None,
) -> Path:
    skipped = skipped or []
    failed = failed or []
    config = enhanced_dir / "config.yaml"
    checkpoint = enhanced_dir / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    input_paths = {}
    for uid in expected:
        input_path = enhanced_dir / f"input-{uid}.wav"
        _write_wav(input_path)
        input_paths[uid] = input_path

    results = []
    for uid in expected:
        if uid in failed or uid not in set(ok) | set(skipped):
            results.append(InferenceResult(uid=uid, status="failed", error="test failure"))
            continue
        output_path = (enhanced_dir / f"{uid}.wav").resolve()
        if not output_path.exists():
            _write_wav(output_path)
        results.append(
            InferenceResult(
                uid=uid,
                status="skipped" if uid in skipped else "ok",
                output_path=str(output_path),
                output_identity=stat_audio_identity(output_path),
            )
        )
    manifest = build_run_manifest(
        config=config,
        checkpoint=checkpoint,
        artifact_fingerprint=build_content_fingerprint(
            {"config": config, "checkpoint": checkpoint},
            parameters={
                "effective_config_identity": effective_config_identity(
                    {"model": "test"}
                )
            },
        ),
        input_inventory=build_input_inventory(
            (uid, {"noisy": input_paths[uid]}) for uid in expected
        ),
        results=results,
        allow_partial=bool(failed),
        created_at="2026-08-12T00:00:00",
    )
    path = enhanced_dir / "run_manifest.json"
    atomic_write_json(path, manifest)
    return path


def test_generated_scp_provenance_is_fresh_and_refreshes_after_source_change(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "first.wav")
    scp = tmp_path / "generated.scp"

    generate_scp_from_dir(audio_dir, scp_path=scp)
    sidecar_path = tmp_path / "generated.scp.provenance.json"
    initial_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = {key: initial_sidecar[key] for key in (
        "version", "kind", "source", "pattern", "recursive", "source_snapshot"
    )}
    assert read_scp(scp) == {"first": audio_dir / "first.wav"}
    assert scp_provenance_fresh(scp, expected)

    _write_wav(audio_dir / "second.wav")
    generate_scp_from_dir(audio_dir, scp_path=scp)
    refreshed_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert refreshed_sidecar["source_snapshot"] != initial_sidecar["source_snapshot"]
    assert set(read_scp(scp)) == {"first", "second"}


def test_generated_scp_snapshot_never_hashes_audio_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "first.wav")
    original_sha256 = pipeline_common._sha256_file

    def reject_audio_sha(path):
        if Path(path).suffix.lower() in {".wav", ".flac", ".rir"}:
            pytest.fail("音频目录 snapshot 不得调用 payload SHA")
        return original_sha256(path)

    monkeypatch.setattr(pipeline_common, "_sha256_file", reject_audio_sha)

    scp = generate_scp_from_dir(audio_dir, scp_path=tmp_path / "generated.scp")
    sidecar = json.loads(Path(f"{scp}.provenance.json").read_text(encoding="utf-8"))

    assert sidecar["source_snapshot"]["algorithm"] == "stat-v1"


def test_generated_scp_rejects_audio_symlink(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target.wav"
    _write_wav(target)
    (source / "linked.wav").symlink_to(target)

    with pytest.raises(ValueError, match="不允许符号链接"):
        generate_scp_from_dir(source, scp_path=tmp_path / "linked.scp")


def test_scp_preparation_never_probes_audio_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    generic = tmp_path / "generic"
    generic.mkdir()
    _write_wav(generic / "sample.wav")
    sf.write(generic / "sample.flac", np.zeros(8, dtype=np.float32), 16000)
    (generic / "room.rir").write_bytes(b"rir-payload")

    official_root = tmp_path / "official"
    official_subset = (
        official_root / "blind_test_set_icassp2023" / "doubletalk"
    )
    official_subset.mkdir(parents=True)
    _write_wav(official_subset / "pair_mic.wav")
    _write_wav(official_subset / "pair_lpb.wav")

    sim_root = tmp_path / "sim"
    sim_sample = sim_root / "prob_target_only" / "sample_000"
    sim_sample.mkdir(parents=True)
    _write_wav(sim_sample / "mic.wav")
    (sim_root / "manifest.json").write_text("{}\n", encoding="utf-8")

    def reject_audio_info(path):
        if Path(path).suffix.lower() in {".wav", ".flac", ".rir"}:
            pytest.fail("评估 SCP/provenance 准备不得调用 sf.info")
        raise AssertionError(f"unexpected sf.info call: {path}")

    monkeypatch.setattr(pipeline_common.sf, "info", reject_audio_info)
    monkeypatch.setattr(pipeline_common, "PROJECT_ROOT", tmp_path / "project")
    # 回归防线：即使重新引入对 DEFAULT_TEST_DATA_ROOT 的读取，也不得写入真实仓库。
    monkeypatch.setattr(
        pipeline_common, "DEFAULT_TEST_DATA_ROOT", tmp_path / "prepared"
    )

    generated_scps = [
        pipeline_common.generate_scp_from_dir(
            generic, pattern="*.wav", scp_path=tmp_path / "wav.scp"
        ),
        eval_io.generate_scp_from_dir(
            generic, pattern="*.flac", scp_path=tmp_path / "flac.scp"
        ),
        pipeline_common.generate_scp_from_dir(
            generic, pattern="*.rir", scp_path=tmp_path / "rir.scp"
        ),
    ]
    official = prepare_testset_scps(
        aec_testset_root=official_root,
        aec_subsets=["doubletalk"],
        selected_datasets=["aec_official"],
        output_root=tmp_path / "prepared",
    )["blind_test_set_icassp2023_doubletalk"]
    simulated = pipeline_common.ensure_aec_sim_scps(sim_root)["prob_target_only"]

    generated_scps.extend(
        [official["mic"], official["lpb"], simulated["mic"]]
    )
    for scp in generated_scps:
        lines = Path(scp).read_text(encoding="utf-8").splitlines()
        assert lines
        assert all(len(line.split()) == 2 for line in lines)


def test_generated_scp_reuses_trusted_four_column_metadata(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    scp = tmp_path / "trusted.scp"
    snapshot = {"algorithm": "stat-v1", "digest": "fixture"}

    pipeline_common.write_generated_scp(
        [("sample", 16000, 8, str(audio))],
        scp,
        source=tmp_path,
        source_snapshot=snapshot,
    )
    pipeline_common.write_generated_scp(
        [("sample", str(audio))],
        scp,
        source=tmp_path,
        source_snapshot=snapshot,
    )

    assert scp.read_text(encoding="utf-8").split() == [
        "sample", "16000", "8", str(audio)
    ]


def test_copy_and_read_scp_accept_only_two_or_four_columns(tmp_path: Path):
    wav = tmp_path / "audio.wav"
    _write_wav(wav)
    two_col = tmp_path / "source.scp"
    two_col.write_text(f"two {wav}\n", encoding="utf-8")
    four_col = tmp_path / "four.scp"
    four_col.write_text(f"four 16000 8 {wav}\n", encoding="utf-8")

    copied = copy_scp_with_provenance(two_col, tmp_path / "copied.scp")
    assert read_scp(copied) == {"two": wav}
    assert read_scp(four_col) == {"four": wav}

    invalid = tmp_path / "invalid.scp"
    invalid.write_text(f"bad 16000 {wav}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非法列数 3"):
        read_scp(invalid)


def test_relative_scp_paths_resolve_from_source_and_survive_copy(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    audio = source_dir / "audio.wav"
    _write_wav(audio)
    source = source_dir / "relative.scp"
    source.write_text("two audio.wav\nfour 16000 8 audio.wav\n", encoding="utf-8")

    assert read_scp(source) == {"two": audio, "four": audio}
    copied = copy_scp_with_provenance(source, tmp_path / "nested" / "copied.scp")

    assert read_scp(copied) == {"two": audio, "four": audio}
    copied_lines = copied.read_text(encoding="utf-8").splitlines()
    assert len(copied_lines[0].split()) == 2
    assert len(copied_lines[1].split()) == 4
    sidecar = json.loads(Path(f"{copied}.provenance.json").read_text(encoding="utf-8"))
    assert sidecar["source_fingerprint"]["sha256"]


def test_copy_scp_content_and_provenance_share_one_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.scp"
    old = b"old old.wav\n"
    new = b"new new.wav\n"
    source.write_bytes(old)
    original = pipeline_common._read_stable_file_bytes

    def mutate_after_snapshot(path):
        result = original(path)
        source.write_bytes(new)
        return result

    monkeypatch.setattr(
        pipeline_common, "_read_stable_file_bytes", mutate_after_snapshot
    )
    copied = copy_scp_with_provenance(source, tmp_path / "copied.scp")
    sidecar = json.loads(Path(f"{copied}.provenance.json").read_text(encoding="utf-8"))

    assert read_scp(copied) == {"old": tmp_path / "old.wav"}
    assert sidecar["source_fingerprint"]["sha256"] == __import__("hashlib").sha256(
        old
    ).hexdigest()


def test_ensure_inf_scp_prefers_and_validates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    enhanced_dir = tmp_path / "enhanced"
    enhanced_dir.mkdir()
    _write_manifest(enhanced_dir, expected=["manifest_uid"], ok=["manifest_uid"])
    wav = enhanced_dir / "manifest_uid.wav"
    (enhanced_dir / "inf.scp").write_text(f"manifest_uid {wav}\n", encoding="utf-8")

    monkeypatch.setattr(
        eval_common,
        "generate_scp_from_dir",
        lambda *_args, **_kwargs: pytest.fail("manifest 存在时不得重新生成 inf.scp"),
    )
    assert eval_common.ensure_inf_scp(enhanced_dir) == enhanced_dir / "inf.scp"


def test_load_inference_coverage_rejects_manifest_partition_mismatch(tmp_path: Path):
    enhanced_dir = tmp_path / "enhanced"
    enhanced_dir.mkdir()
    manifest_path = _write_manifest(
        enhanced_dir, expected=["a", "b"], ok=["a"], failed=["b"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["failed_uids"] = []
    atomic_write_json(manifest_path, manifest)
    (enhanced_dir / "inf.scp").write_text(
        f"a {enhanced_dir / 'a.wav'}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="failed_uids 与 results 不一致"):
        eval_common.load_inference_coverage(enhanced_dir)


def test_load_inference_coverage_rejects_inf_scp_manifest_mismatch(tmp_path: Path):
    enhanced_dir = tmp_path / "enhanced"
    enhanced_dir.mkdir()
    wav = enhanced_dir / "a.wav"
    _write_wav(wav)
    _write_manifest(enhanced_dir, expected=["a"], ok=["a"])
    (enhanced_dir / "inf.scp").write_text(f"unexpected {wav}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inf.scp uids do not match"):
        eval_common.load_inference_coverage(enhanced_dir)


def test_load_inference_coverage_rejects_tampered_output(tmp_path: Path):
    enhanced_dir = tmp_path / "enhanced"
    enhanced_dir.mkdir()
    _write_manifest(enhanced_dir, expected=["a"], ok=["a"])
    output = enhanced_dir / "a.wav"
    (enhanced_dir / "inf.scp").write_text(f"a {output}\n", encoding="utf-8")
    output.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="identity mismatch"):
        eval_common.load_inference_coverage(enhanced_dir)


def test_wrapper_artifact_validation_rejects_stale_config_and_output(tmp_path: Path):
    enhanced_dir = tmp_path / "enhanced"
    enhanced_dir.mkdir()
    _write_manifest(enhanced_dir, expected=["a"], ok=["a"])
    config = enhanced_dir / "config.yaml"
    checkpoint = enhanced_dir / "model.tar"
    output = enhanced_dir / "a.wav"
    (enhanced_dir / "inf.scp").write_text(f"a {output}\n", encoding="utf-8")

    pipeline_common.validate_inference_artifacts(
        enhanced_dir, config=config, checkpoint=checkpoint
    )
    config.write_text("model: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config fingerprint is stale"):
        pipeline_common.validate_inference_artifacts(
            enhanced_dir, config=config, checkpoint=checkpoint
        )

    config.write_text("model: test\n", encoding="utf-8")
    output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="output identity mismatch"):
        pipeline_common.validate_inference_artifacts(
            enhanced_dir, config=config, checkpoint=checkpoint
        )


def test_se_evaluate_only_partial_is_blocked_before_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    enhanced_dir = tmp_path / "eval" / "voicebank" / "enhanced"
    enhanced_dir.mkdir(parents=True)
    manifest_path = _write_manifest(
        enhanced_dir, expected=["ok", "failed"], ok=["ok"], failed=["failed"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = Path(manifest["config"])
    checkpoint = Path(manifest["checkpoint"])
    noisy_scp = tmp_path / "noisy.scp"
    noisy_scp.write_text(
        "\n".join(
            f"{uid} {enhanced_dir / f'input-{uid}.wav'}"
            for uid in ("ok", "failed")
        ) + "\n",
        encoding="utf-8",
    )
    (enhanced_dir / "inf.scp").write_text(
        f"ok {enhanced_dir / 'ok.wav'}\n", encoding="utf-8"
    )
    args = SimpleNamespace(
        step="evaluate",
        allow_partial_inference=False,
        device="cpu",
        audio_quality=False,
        downstream=False,
    )
    monkeypatch.setattr(
        pipeline_se,
        "run_standard_evaluation",
        lambda **_kwargs: pytest.fail("partial inference 不得进入 evaluator"),
    )
    summary = {}

    pipeline_se._run_se_dataset(
        "voicebank",
        args,
        config,
        checkpoint,
        tmp_path / "eval",
        summary,
        {"voicebank": {"noisy": noisy_scp}},
        ref_key=None,
        blind=True,
    )

    assert summary["voicebank"]["status"] == "error"
    assert summary["voicebank"]["coverage"] == {
        "expected": 2,
        "produced": 1,
        "failed": 1,
        "evaluated": 0,
    }


def test_generated_scp_stat_identity_ignores_same_size_mtime_payload_change(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav = audio_dir / "same.wav"
    _write_wav(wav)
    scp = tmp_path / "same.scp"
    generate_scp_from_dir(audio_dir, scp_path=scp)
    first = json.loads(Path(f"{scp}.provenance.json").read_text(encoding="utf-8"))
    stat = wav.stat()
    raw = bytearray(wav.read_bytes())
    raw[-1] ^= 1
    wav.write_bytes(raw)
    os.utime(wav, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    generate_scp_from_dir(audio_dir, scp_path=scp)
    second = json.loads(Path(f"{scp}.provenance.json").read_text(encoding="utf-8"))

    assert second["source_snapshot"] == first["source_snapshot"]
    assert second["source_snapshot"]["algorithm"] == "stat-v1"


def test_generate_scp_missing_source_preserves_existing_artifacts(tmp_path: Path):
    scp = tmp_path / "existing.scp"
    sidecar = tmp_path / "existing.scp.provenance.json"
    scp.write_text("old /tmp/old.wav\n", encoding="utf-8")
    sidecar.write_text('{"old": true}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="source directory"):
        generate_scp_from_dir(tmp_path / "missing", scp_path=scp)

    assert scp.read_text(encoding="utf-8") == "old /tmp/old.wav\n"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {"old": True}


def test_generate_scp_empty_source_preserves_existing_artifacts(tmp_path: Path):
    source = tmp_path / "empty"
    source.mkdir()
    scp = tmp_path / "existing.scp"
    sidecar = tmp_path / "existing.scp.provenance.json"
    scp.write_text("old /tmp/old.wav\n", encoding="utf-8")
    sidecar.write_text('{"old": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="zero entries"):
        generate_scp_from_dir(source, scp_path=scp)

    assert scp.read_text(encoding="utf-8") == "old /tmp/old.wav\n"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {"old": True}


def test_prepare_testset_scps_consumes_standardized_aec_scps_without_wav_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "AEC_Challenge_test"
    subset = source_root / "blind_test_set_icassp2023" / "doubletalk"
    subset.mkdir(parents=True)
    raw_mic = tmp_path / "raw_mic.scp"
    raw_lpb = tmp_path / "raw_lpb.scp"
    raw_mic.write_text("u /audio/mic.wav\n", encoding="utf-8")
    raw_lpb.write_text("u /audio/lpb.wav\n", encoding="utf-8")
    copy_scp_with_provenance(raw_mic, subset / "mic.scp")
    copy_scp_with_provenance(raw_lpb, subset / "lpb.scp")

    monkeypatch.setattr(
        pipeline_common,
        "generate_scp_from_dir",
        lambda *_args, **_kwargs: pytest.fail("standardized AEC SCP 不得扫描 WAV"),
    )

    result = prepare_testset_scps(
        aec_testset_root=source_root,
        aec_subsets=["doubletalk"],
        selected_datasets=["aec_official"],
        output_root=tmp_path / "prepared",
    )

    prepared = result["blind_test_set_icassp2023_doubletalk"]
    assert read_scp(prepared["mic"]) == {"u": Path("/audio/mic.wav")}
    assert read_scp(prepared["lpb"]) == {"u": Path("/audio/lpb.wav")}


def test_prepare_testset_scps_rejects_empty_standardized_dns_without_overwrite(
    tmp_path: Path,
):
    source = tmp_path / "DNS_Challenge_test" / "blind_test_set_v3" / "noisy.scp"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    copy_scp_with_provenance(source, source)

    with pytest.raises(ValueError, match="zero entries"):
        prepare_testset_scps(
            dns_challenge_noisy_dir=source,
            selected_datasets=["dns_challenge"],
            output_root=tmp_path / "prepared",
        )

    assert source.read_text(encoding="utf-8") == ""


def test_default_paths_use_ignored_placeholder_without_machine_absolute_path():
    script = (
        "import json; "
        "from evaluate.eval_pipeline_common import DEFAULT_PATHS; "
        "print(json.dumps({k: str(v) for k, v in DEFAULT_PATHS.items()}))"
    )
    env = os.environ.copy()
    env.pop("USE_TRAIN_TEST_DATA_ROOT", None)
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = json.loads(completed.stdout)

    assert all(path.startswith("data/test/") for path in paths.values())
    assert all("/opt/nonexistent" not in path for path in paths.values())


def test_default_paths_follow_explicit_environment_root(tmp_path: Path):
    script = (
        "import json; "
        "from evaluate.eval_pipeline_common import DEFAULT_PATHS, DEFAULT_TEST_DATA_ROOT, default_eval_output_root; "
        "print(json.dumps({"
        "'paths': {k: str(v) for k, v in DEFAULT_PATHS.items()}, "
        "'root': str(DEFAULT_TEST_DATA_ROOT), "
        "'eval_root': str(default_eval_output_root())}))"
    )
    env = os.environ.copy()
    expected_root = tmp_path / "raw_test"
    env["USE_TRAIN_TEST_DATA_ROOT"] = str(expected_root)
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    paths = payload["paths"]

    assert payload["root"] == str(expected_root)
    assert payload["eval_root"] == str(expected_root / "exp/eval")
    assert all(path.startswith(str(expected_root)) for path in paths.values())


def test_prepare_testset_scps_default_output_follows_configured_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    noisy = tmp_path / "noisy.scp"
    clean = tmp_path / "clean.scp"
    noisy.write_text("noisy /tmp/noisy.wav\n", encoding="utf-8")
    clean.write_text("clean /tmp/clean.wav\n", encoding="utf-8")
    expected_root = tmp_path / "external_test"
    monkeypatch.setattr(pipeline_common, "DEFAULT_TEST_DATA_ROOT", expected_root)

    result = prepare_testset_scps(
        voicebank_noisy_scp=noisy,
        voicebank_clean_scp=clean,
        selected_datasets=["voicebank"],
    )

    assert result["voicebank"]["noisy"] == (
        expected_root / "voicebank_demand_16k/noisy.scp"
    )
    assert result["voicebank"]["clean"] == (
        expected_root / "voicebank_demand_16k/clean.scp"
    )


def test_experiment_report_badcase_paths_follow_environment_root(tmp_path: Path):
    script = (
        "import json; "
        "import tools.ops.reporting.experiment_report as report; "
        "adapter = report._evaluation_reservation_adapter("
        "task_type='aec', output_name='demo', config_path=None, "
        "checkpoint_path=None, device='cpu', inference_devices=None); "
        "print(json.dumps({'test': str(report.BADCASE_TEST_DIR), "
        "'result': str(report.BADCASE_RESULT_DIR), "
        "'eval': str(report.default_eval_output_root(report._PROJECT_ROOT)), "
        "'run': str(adapter.output_dir)}))"
    )
    env = os.environ.copy()
    expected_root = tmp_path / "external_test"
    env["USE_TRAIN_TEST_DATA_ROOT"] = str(expected_root)
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = json.loads(completed.stdout)

    assert paths["test"] == str(expected_root / "badcase/samples")
    assert paths["result"] == str(expected_root / "badcase/outputs")
    assert paths["eval"] == str(expected_root / "exp/eval")
    assert paths["run"] == str(expected_root / "exp/eval/aec/demo")


def test_find_exp_assets_rejects_cross_directory_heuristic_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project_root = tmp_path / "project"
    exp_dir = project_root / "exp" / "se" / "demo"
    exp_dir.mkdir(parents=True)
    (exp_dir / "best_model.tar").write_bytes(b"checkpoint")
    config = project_root / "configs" / "se" / "nested" / "demo.yml"
    config.parent.mkdir(parents=True)
    config.write_text("model: {type: Demo}\n", encoding="utf-8")
    monkeypatch.setattr(pipeline_common, "PROJECT_ROOT", project_root)

    with pytest.raises(FileNotFoundError, match="same-run config/checkpoint pair"):
        pipeline_common.find_exp_assets(exp_dir)


def test_prepare_testset_scps_keeps_valid_explicit_voicebank_paths(tmp_path: Path):
    noisy = tmp_path / "noisy.scp"
    clean = tmp_path / "clean.scp"
    noisy.write_text("noisy /tmp/noisy.wav\n", encoding="utf-8")
    clean.write_text("clean /tmp/clean.wav\n", encoding="utf-8")

    result = prepare_testset_scps(
        voicebank_noisy_scp=noisy,
        voicebank_clean_scp=clean,
        output_root=tmp_path / "prepared",
    )

    assert result["voicebank"]["noisy"].read_text(encoding="utf-8") == noisy.read_text(
        encoding="utf-8"
    )
    assert result["voicebank"]["clean"].read_text(encoding="utf-8") == clean.read_text(
        encoding="utf-8"
    )


def test_prepare_testset_scps_copies_paired_urgent_four_column_scps(tmp_path: Path):
    noisy = tmp_path / "urgent_noisy.scp"
    clean = tmp_path / "urgent_clean.scp"
    noisy.write_text(
        "sample_a 16000 8 nested/noisy_a.flac\n"
        "sample_b 16000 16 nested/noisy_b.flac\n",
        encoding="utf-8",
    )
    clean.write_text(
        "sample_a 16000 8 nested/clean_a.flac\n"
        "sample_b 16000 16 nested/clean_b.flac\n",
        encoding="utf-8",
    )

    result = prepare_testset_scps(
        urgent_noisy_scp=noisy,
        urgent_clean_scp=clean,
        urgent_noisy_dir=tmp_path / "missing_noisy_dir",
        urgent_clean_dir=tmp_path / "missing_clean_dir",
        selected_datasets=["urgent"],
        output_root=tmp_path / "prepared",
    )

    copied = result["urgent"]
    assert all(len(line.split()) == 4 for path in copied.values() for line in path.read_text(encoding="utf-8").splitlines())
    assert {Path(line.split()[-1]).name for line in copied["noisy"].read_text(encoding="utf-8").splitlines()} == {"noisy_a.flac", "noisy_b.flac"}
    assert Path(f"{copied['noisy']}.provenance.json").is_file()
    assert Path(f"{copied['clean']}.provenance.json").is_file()


def test_prepare_testset_scps_requires_paired_urgent_scps(tmp_path: Path):
    noisy = tmp_path / "urgent_noisy.scp"
    noisy.write_text("sample 16000 8 noisy.flac\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be provided as a pair"):
        prepare_testset_scps(
            urgent_noisy_scp=noisy,
            selected_datasets=["urgent"],
            output_root=tmp_path / "prepared",
        )


def test_prepare_testset_scps_rejects_urgent_scp_uid_mismatch(tmp_path: Path):
    noisy = tmp_path / "urgent_noisy.scp"
    clean = tmp_path / "urgent_clean.scp"
    noisy.write_text("noisy_only 16000 8 noisy.flac\n", encoding="utf-8")
    clean.write_text("prob_target_only 16000 8 clean.flac\n", encoding="utf-8")

    with pytest.raises(ValueError, match="UID sets do not match"):
        prepare_testset_scps(
            urgent_noisy_scp=noisy,
            urgent_clean_scp=clean,
            selected_datasets=["urgent"],
            output_root=tmp_path / "prepared",
        )


def test_prepare_testset_scps_prefers_urgent_scps_over_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    noisy = tmp_path / "urgent_noisy.scp"
    clean = tmp_path / "urgent_clean.scp"
    noisy.write_text("sample 16000 8 noisy.flac\n", encoding="utf-8")
    clean.write_text("sample 16000 8 clean.flac\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_common,
        "generate_scp_from_dir",
        lambda *_args, **_kwargs: pytest.fail("提供 URGENT SCP 时不得扫描目录"),
    )

    result = prepare_testset_scps(
        urgent_noisy_scp=noisy,
        urgent_clean_scp=clean,
        urgent_noisy_dir=tmp_path / "missing_noisy_dir",
        urgent_clean_dir=tmp_path / "missing_clean_dir",
        selected_datasets=["urgent"],
        output_root=tmp_path / "prepared",
    )

    assert set(result) == {"urgent"}


def test_missing_explicit_dataset_directory_fails_fast(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc_info:
        prepare_testset_scps(
            dns_challenge_noisy_dir=tmp_path / "missing_dns",
            output_root=tmp_path / "prepared",
        )

    message = str(exc_info.value)
    assert "dns_challenge_noisy_dir" in message
    assert "does not exist" in message


def test_prepare_testset_scps_ignores_unselected_missing_sources(tmp_path: Path):
    noisy = tmp_path / "noisy.scp"
    clean = tmp_path / "clean.scp"
    noisy.write_text("noisy /tmp/noisy.wav\n", encoding="utf-8")
    clean.write_text("clean /tmp/clean.wav\n", encoding="utf-8")

    result = prepare_testset_scps(
        voicebank_noisy_scp=noisy,
        voicebank_clean_scp=clean,
        dns_challenge_noisy_dir=tmp_path / "missing_dns",
        selected_datasets=["voicebank"],
        output_root=tmp_path / "prepared",
    )

    assert set(result) == {"voicebank"}
    assert not (tmp_path / "prepared" / "DNS_Challenge_test").exists()


def test_voicebank_missing_explicit_source_fails_before_stale_destination_publish(
    tmp_path: Path,
):
    output_root = tmp_path / "prepared"
    stale_dir = output_root / "voicebank_demand_16k"
    stale_dir.mkdir(parents=True)
    stale_scp = stale_dir / "noisy.scp"
    stale_scp.write_text("stale /tmp/stale.wav\n", encoding="utf-8")
    clean = tmp_path / "clean.scp"
    clean.write_text("clean /tmp/clean.wav\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as exc_info:
        prepare_testset_scps(
            voicebank_noisy_scp=tmp_path / "missing.scp",
            voicebank_clean_scp=clean,
            output_root=output_root,
        )

    message = str(exc_info.value)
    assert "voicebank_noisy_scp" in message
    assert "does not exist" in message
    assert "USE_TRAIN_TEST_DATA_ROOT" in message
    assert stale_scp.read_text(encoding="utf-8") == "stale /tmp/stale.wav\n"


def test_coverage_status_done_partial_and_error():
    assert eval_common.coverage_status(
        {"expected": 2, "produced": 2, "failed": 0, "evaluated": 2},
        allow_partial=False,
    ) == "done"
    assert eval_common.coverage_status(
        {"expected": 2, "produced": 1, "failed": 1, "evaluated": 1},
        allow_partial=True,
    ) == "partial"
    assert eval_common.coverage_status(
        {"expected": 2, "produced": 1, "failed": 1, "evaluated": 1},
        allow_partial=False,
    ) == "error"
    assert eval_common.coverage_status(
        {"expected": 0, "produced": 0, "failed": 0, "evaluated": 0},
        allow_partial=False,
    ) == "error"


def test_finalize_pipeline_summary_aggregates_status_and_coverage():
    summary = {
        "datasets": {
            "done": {
                "status": "done",
                "coverage": {"expected": 2, "produced": 2, "failed": 0, "evaluated": 2},
            },
            "partial": {
                "status": "partial",
                "coverage": {"expected": 2, "produced": 1, "failed": 1, "evaluated": 1},
            },
        }
    }
    eval_common.finalize_pipeline_summary(summary)
    assert summary["status"] == "partial"
    assert summary["coverage"] == {
        "expected": 4, "produced": 3, "failed": 1, "evaluated": 3
    }


def test_inference_wrappers_use_absolute_paths_without_changing_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    working_dir = tmp_path / "caller"
    working_dir.mkdir()
    config = working_dir / "model.yaml"
    checkpoint = working_dir / "model.tar"
    input_scp = working_dir / "input.scp"
    far_scp = working_dir / "far.scp"
    for path in (config, checkpoint, input_scp, far_scp):
        path.write_text("fixture\n", encoding="utf-8")
    output_dir = working_dir / "output"
    commands: list[tuple[list[str], dict]] = []

    monkeypatch.setattr(
        pipeline_common,
        "_validate_fresh_inference_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_common.subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )
    monkeypatch.chdir(working_dir)
    cwd_before = Path.cwd()

    expected_identity = effective_config_identity({"model": "test"})
    se_output = pipeline_common.run_inference_se(
        "model.yaml", "model.tar", input_scp="input.scp", output_dir="output",
        expected_effective_config_identity=expected_identity,
    )
    aec_output = pipeline_common.run_inference_aec_paired(
        "model.yaml",
        "model.tar",
        mic_scp="input.scp",
        far_scp="far.scp",
        output_dir="output",
        expected_effective_config_identity=expected_identity,
    )

    assert Path.cwd() == cwd_before
    assert se_output == output_dir.resolve()
    assert aec_output == output_dir.resolve()
    assert len(commands) == 2
    for command, kwargs in commands:
        assert "cwd" not in kwargs
        assert kwargs["check"] is True
        assert Path(command[1]) == pipeline_common.PROJECT_ROOT / "inference.py"
        assert command[
            command.index("--expected-effective-config-sha256") + 1
        ] == expected_identity["sha256"]
        for option in (
            "-C",
            "--checkpoint",
            "--output_dir",
        ):
            assert Path(command[command.index(option) + 1]).is_absolute()
    se_command = commands[0][0]
    aec_command = commands[1][0]
    assert Path(se_command[se_command.index("--input_scp") + 1]).is_absolute()
    assert Path(aec_command[aec_command.index("--input_mic_scp") + 1]).is_absolute()
    assert Path(aec_command[aec_command.index("--input_far_scp") + 1]).is_absolute()


def test_pipeline_config_rejects_base_change_between_parent_and_child(tmp_path: Path):
    base = tmp_path / "base.yaml"
    config = tmp_path / "config.yaml"
    base.write_text("stft:\n  n_fft: 320\n", encoding="utf-8")
    config.write_text(
        "base_config: base.yaml\nmodel:\n  type: GTCRN\n",
        encoding="utf-8",
    )
    _, parent_identity = pipeline_common.load_pipeline_config(config)

    base.write_text("stft:\n  n_fft: 512\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="effective config identity mismatch"):
        pipeline_common.load_pipeline_config(
            config, expected_sha256=parent_identity["sha256"]
        )


def test_other_repository_subprocesses_use_absolute_paths_without_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = tmp_path / "config.yaml"
    inf_scp = tmp_path / "inf.scp"
    config.write_text("model: test\n", encoding="utf-8")
    inf_scp.write_text("u /tmp/u.wav\n", encoding="utf-8")
    commands: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(
        process_runner.subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )
    monkeypatch.setattr(pipeline_common.subprocess, "run", process_runner.subprocess.run)

    process_runner.run_evaluate_urgent(inf_scp, tmp_path / "urgent")
    pipeline_common.prepare_aec_sim_testset(config, tmp_path / "sim")

    assert len(commands) == 2
    urgent_command, urgent_kwargs = commands[0]
    assert "cwd" not in urgent_kwargs
    assert urgent_kwargs["check"] is True
    assert Path(urgent_command[1]) == (
        process_runner.PROJECT_ROOT / "evaluate/evaluate_urgent.py"
    )
    assert Path(urgent_command[1]).is_absolute()

    generator_command, generator_kwargs = commands[1]
    assert "cwd" not in generator_kwargs
    assert generator_kwargs["check"] is True
    assert generator_command[1:3] == [
        "-m",
        "tools.prepare_data_tools.generators.aec.generate_simulated_testset",
    ]
    assert generator_command[generator_command.index("--num-samples") + 1] == "100"
    assert generator_kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(
        pipeline_common.PROJECT_ROOT
    )


def test_common_does_not_import_pipeline_orchestration():
    source = Path(eval_common.__file__).read_text(encoding="utf-8")
    assert "eval_pipeline_common" not in source


def test_evaluate_only_with_assets_requires_inference_manifest(tmp_path: Path):
    enhanced_dir = tmp_path / "enhanced"
    enhanced_dir.mkdir()
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"model")
    wav = enhanced_dir / "u.wav"
    _write_wav(wav)
    (enhanced_dir / "inf.scp").write_text(f"u {wav}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run_manifest"):
        eval_common.load_inference_coverage(
            enhanced_dir, config=config, checkpoint=checkpoint,
            evaluation_performed=False,
        )


def test_result_manifest_binds_run_id_and_metric_file_hash(tmp_path: Path):
    results = tmp_path / "dataset" / "RESULTS.txt"
    results.parent.mkdir()
    results.write_text("PESQ: 2.0\n", encoding="utf-8")
    summary = {
        "run_id": "run-1",
        "datasets": {
            "dataset": {
                "status": "done",
                "results_file": str(results),
            }
        },
    }

    eval_common.publish_result_manifest(tmp_path, summary)
    manifest = eval_common.validate_result_manifest(
        tmp_path, expected_run_id="run-1"
    )
    assert manifest["files"]["dataset/RESULTS.txt"]["sha256"]

    results.write_text("PESQ: 9.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        eval_common.validate_result_manifest(tmp_path, expected_run_id="run-1")
    with pytest.raises(ValueError, match="run_id mismatch"):
        eval_common.validate_result_manifest(tmp_path, expected_run_id="run-2")


def test_experiment_report_rejects_unbound_or_tampered_results(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    results = dataset / "RESULTS.txt"
    results.write_text("PESQ: 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="result manifest missing"):
        experiment_report.parse_all_results(tmp_path)

    summary = {
        "run_id": "run-1",
        "datasets": {
            "dataset": {"status": "done", "results_file": str(results)}
        },
    }
    eval_common.publish_result_manifest(tmp_path, summary)
    assert experiment_report.parse_all_results(tmp_path)["dataset"]["PESQ"] == 2.0
    results.write_text("PESQ: 8.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        experiment_report.parse_all_results(tmp_path)


def test_scp_read_and_fingerprint_use_one_bytes_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    scp = tmp_path / "input.scp"
    scp.write_text("old old.wav\n", encoding="utf-8")
    snapshots = iter([b"old old.wav\n", b"new new.wav\n"])
    monkeypatch.setattr(
        eval_io.Path,
        "read_bytes",
        lambda _self: next(snapshots),
    )

    parsed = eval_io.read_scp(scp)

    assert parsed == {"old": tmp_path / "old.wav"}


def test_scp_provenance_validation_rejects_tampered_scp(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "first.wav")
    scp = eval_io.generate_scp_from_dir(audio_dir, scp_path=tmp_path / "input.scp")
    eval_io.validate_scp_provenance(scp)
    scp.write_text("other /tmp/other.wav\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        eval_io.validate_scp_provenance(scp)


def test_aec_unknown_dataset_is_machine_readable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.tar"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"model")
    args = SimpleNamespace(
        exp_dir=None,
        config=str(config),
        checkpoint=str(checkpoint),
        datasets="unknown",
        aec_subsets="all",
        step="evaluate",
        aec_testset_root="unused",
        urgent_noisy_dir="unused",
        urgent_clean_dir="unused",
        voicebank_noisy_scp="unused",
        voicebank_clean_scp="unused",
        dns_challenge_noisy_dir="unused",
    )
    monkeypatch.setattr(pipeline_aec, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_aec, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline_aec, "detect_model_type", lambda _config, **_kwargs: "aec")
    monkeypatch.setattr(pipeline_aec, "detect_aec_subtype", lambda _config, **_kwargs: "joint")
    monkeypatch.setattr(pipeline_aec, "prepare_testset_scps", lambda **_kwargs: {})
    monkeypatch.setattr(pipeline_aec, "print_summary_table", lambda _summary: None)

    assert pipeline_aec.main() == 1
    summary = json.loads(
        (tmp_path / "exp/eval/aec" / tmp_path.name / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "error"
    assert summary["datasets"]["unknown"] == {
        "status": "error",
        "reason": "unknown_dataset",
        "dataset": "unknown",
    }


pytestmark = [pytest.mark.unit]
