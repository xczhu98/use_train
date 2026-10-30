"""TSE 推理 artifact 深 Module 的安全与并发合同。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from core.utils import file_transaction
from core.utils.inference_artifacts import fingerprint_file
from tasks.tse.inference import artifacts


def _row(
    *, dataset: str = "ds", mixture: str = "m1", enrollment: str = "e1"
) -> dict[str, str]:
    return {
        "dataset": dataset,
        "mixture_utterance": mixture,
        "enrolment_speakers_utterance": enrollment,
    }


def test_uid_is_canonical_dataset_mixture_enrollment_sha256_128():
    assert artifacts.make_uid(_row()) == "b14766a7171b726e8257f2076ae0c7f1"
    assert artifacts.make_uid(_row(dataset="other")) != artifacts.make_uid(_row())
    assert len(artifacts.make_uid(_row())) == 32


def test_batch_preflight_rejects_duplicate_identity_before_output():
    with pytest.raises(ValueError, match="重复 identity"):
        artifacts.validate_batch([_row(), dict(_row())])


def test_manifest_v4_declares_uid_scheme_and_item_identity(tmp_path: Path):
    row = {**_row(), "inference_status": "ok", "tse_path": "/tmp/result.wav"}
    target = tmp_path / "manifest.json"

    payload = artifacts.write_inference_manifest([row], target)

    assert payload["schema"] == "tse-inference-manifest-v4"
    assert payload["uid_scheme"] == artifacts.UID_SCHEME
    assert payload["expected_uids"] == [artifacts.make_uid(row)]
    assert payload["items"][0]["identity"] == artifacts.row_identity(row)
    assert json.loads(target.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("link_kind", ["leaf", "ancestor"])
def test_audio_identity_rejects_leaf_and_ancestor_symlink(
    tmp_path: Path, link_kind: str
):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    audio = real_dir / "audio.wav"
    audio.write_bytes(b"wav")
    if link_kind == "leaf":
        candidate = tmp_path / "linked.wav"
        candidate.symlink_to(audio)
    else:
        linked_dir = tmp_path / "linked-dir"
        linked_dir.symlink_to(real_dir, target_is_directory=True)
        candidate = linked_dir / "audio.wav"

    with pytest.raises(ValueError, match="symbolic link"):
        artifacts.stat_audio_artifact(candidate)


def test_bound_audio_read_detects_change_during_loader(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"before")
    expected = artifacts.stat_audio_artifact(audio)

    def mutating_loader(_stream):
        audio.write_bytes(b"after-loader")
        return "pcm", 16000

    with pytest.raises(RuntimeError, match="读取期间发生变化"):
        artifacts.load_bound_audio(audio, mutating_loader, expected)


def test_output_lock_is_nonblocking_and_second_writer_writes_nothing(tmp_path: Path):
    output = tmp_path / "eval"
    with artifacts.exclusive_output_run(output):
        with pytest.raises(RuntimeError, match="已有 writer"):
            with artifacts.exclusive_output_run(output):
                raise AssertionError("second writer must never enter")
        assert not (output / "evaluation_current.json").exists()


def test_public_manifest_writer_uses_output_lock(tmp_path: Path):
    target = tmp_path / "manifest.json"
    row = {**_row(), "inference_status": "failed"}

    with artifacts.exclusive_output_run(tmp_path):
        with pytest.raises(RuntimeError, match="已有 writer"):
            artifacts.write_inference_manifest([row], target)

    assert not target.exists()


def test_non_utf8_sidecar_is_not_reusable(tmp_path: Path):
    sidecar = tmp_path / "output.meta.json"
    sidecar.write_bytes(b"\xff\xfe\x00")

    assert artifacts.load_sidecar(sidecar) is None


def test_immutable_checkpoint_loader_consumes_captured_bytes(tmp_path: Path):
    checkpoint = tmp_path / "model.tar"
    detached = tmp_path / "trusted.detached"
    attacker = tmp_path / "attacker.tar"
    checkpoint.write_bytes(b"trusted-checkpoint")
    attacker.write_bytes(b"attacker-checkpoint")

    def replacing_loader(stream):
        os.replace(checkpoint, detached)
        os.replace(attacker, checkpoint)
        loaded = stream.read()
        checkpoint.unlink()
        os.replace(detached, checkpoint)
        return loaded

    loaded, identity = artifacts.load_immutable_checkpoint(
        checkpoint, loader=replacing_loader, label="model checkpoint"
    )

    assert loaded == b"trusted-checkpoint"
    assert identity == fingerprint_file(checkpoint)


def test_audio_and_sidecar_publish_roll_back_as_one_set(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "model.tar"
    config = {"model": {"args": {}}, "stft": {"n_fft": 512}}
    torch.save({"model": {}, "resolved_config": config}, checkpoint)
    mixture = tmp_path / "mixture.wav"
    enrollment = tmp_path / "enrollment.wav"
    mixture.write_bytes(b"mixture")
    enrollment.write_bytes(b"enrollment")
    snapshot = artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
        checkpoint, None, {"device": "cpu", "apply_vad": False}
    )
    expected = snapshot.bind_inputs(
        {"mixture": mixture, "enrollment": enrollment}
    )
    output = tmp_path / "result.wav"
    sidecar = tmp_path / "result.meta.json"
    output.write_bytes(b"old-wav")
    sidecar.write_bytes(b"old-sidecar")

    def fail_sidecar_publish(staged, targets, **kwargs):
        def replace(source, target):
            if Path(target) == sidecar and Path(source).name.endswith(".tmp"):
                raise OSError("simulated sidecar publish failure")
            os.replace(source, target)

        file_transaction.publish_staged_files(
            staged, targets, replace=replace, **kwargs
        )

    monkeypatch.setattr(artifacts, "publish_staged_files", fail_sidecar_publish)

    with pytest.raises(OSError, match="simulated sidecar publish failure"):
        artifacts.publish_audio_output(
            output,
            sidecar,
            np.zeros(32, dtype=np.float32),
            16000,
            expected,
            output_dir=tmp_path,
        )

    assert output.read_bytes() == b"old-wav"
    assert sidecar.read_bytes() == b"old-sidecar"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_audio_set_rolls_back_when_post_publish_verification_fails(
    monkeypatch, tmp_path: Path
):
    checkpoint = tmp_path / "model.tar"
    config = {"model": {"args": {}}, "stft": {"n_fft": 512}}
    torch.save({"model": {}, "resolved_config": config}, checkpoint)
    mixture = tmp_path / "mixture.wav"
    enrollment = tmp_path / "enrollment.wav"
    mixture.write_bytes(b"mixture")
    enrollment.write_bytes(b"enrollment")
    snapshot = artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
        checkpoint, None, {"device": "cpu", "apply_vad": False}
    )
    expected = snapshot.bind_inputs(
        {"mixture": mixture, "enrollment": enrollment}
    )
    output = tmp_path / "result.wav"
    sidecar = tmp_path / "result.meta.json"
    output.write_bytes(b"old-wav")
    sidecar.write_bytes(b"old-sidecar")
    original_audio_identity = artifacts.stat_audio_artifact
    calls = 0

    def fail_published_audio(path):
        nonlocal calls
        calls += 1
        if Path(path) == output and calls >= 2:
            raise RuntimeError("simulated published identity failure")
        return original_audio_identity(path)

    monkeypatch.setattr(artifacts, "stat_audio_artifact", fail_published_audio)

    with pytest.raises(RuntimeError, match="published WAV/sidecar set cannot be verified"):
        artifacts.publish_audio_output(
            output,
            sidecar,
            np.zeros(32, dtype=np.float32),
            16000,
            expected,
            output_dir=tmp_path,
        )

    assert output.read_bytes() == b"old-wav"
    assert sidecar.read_bytes() == b"old-sidecar"


def test_snapshot_audio_publish_delegates_to_public_writer(monkeypatch, tmp_path: Path):
    snapshot = artifacts.TSEInferenceArtifactSnapshot(meta={}, mode="loaded")
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(artifacts, "publish_audio_output", record)
    snapshot.publish_audio_output(
        tmp_path / "result.wav",
        tmp_path / "result.meta.json",
        np.zeros(1),
        16000,
        {"expected": True},
        output_dir=tmp_path,
    )

    assert len(calls) == 1
    assert calls[0][1] == {"output_dir": tmp_path}


def test_manifest_verifier_binds_uid_counts_and_sidecar_global_identity(tmp_path: Path):
    checkpoint = tmp_path / "model.tar"
    config = {"model": {"args": {}}, "stft": {"n_fft": 512}}
    torch.save({"model": {}, "resolved_config": config}, checkpoint)
    mixture = tmp_path / "mixture.wav"
    enrollment = tmp_path / "enrollment.wav"
    mixture.write_bytes(b"mixture")
    enrollment.write_bytes(b"enrollment")
    snapshot = artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
        checkpoint, None, {"device": "cpu", "apply_vad": False}
    )
    expected = snapshot.bind_inputs(
        {"mixture": mixture, "enrollment": enrollment}
    )
    output = tmp_path / "result.wav"
    sidecar = tmp_path / "result.meta.json"
    snapshot.publish_audio_output(
        output,
        sidecar,
        np.zeros(32, dtype=np.float32),
        16000,
        expected,
        output_dir=tmp_path,
    )
    row = {
        **_row(mixture="mixture", enrollment="enrollment"),
        "inference_status": "ok",
        "tse_path": str(output),
        "_inference_artifacts": expected,
    }
    manifest = tmp_path / "manifest.json"
    payload = artifacts.write_inference_manifest(
        [row], manifest, artifact_meta=snapshot.meta
    )
    assert artifacts.verify_inference_manifest(
        manifest, artifact_meta=snapshot.meta
    ) == payload

    tampered = dict(payload)
    tampered["ok_uids"] = []
    manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ok_uids is inconsistent"):
        artifacts.verify_inference_manifest(manifest, artifact_meta=snapshot.meta)

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["artifacts"]["parameters"] = {"device": "tampered"}
    sidecar_payload["fingerprint"] = artifacts.fingerprint_json(
        sidecar_payload["artifacts"]
    )
    sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
    payload["items"][0]["artifacts"]["sidecar"] = fingerprint_file(sidecar)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind global artifact: parameters"):
        artifacts.verify_inference_manifest(manifest, artifact_meta=snapshot.meta)


def test_checkpoint_symlink_is_rejected_for_skip(tmp_path: Path):
    checkpoint = tmp_path / "model.tar"
    config = {
        "model": {"args": {"joint_training": True, "features": {}}},
        "stft": {"n_fft": 512},
    }
    torch.save({"model": {}, "resolved_config": config}, checkpoint)
    linked = tmp_path / "linked-model.tar"
    linked.symlink_to(checkpoint)

    with pytest.raises(ValueError, match="symbolic link"):
        artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
            linked, None, {"device": "cpu", "apply_vad": False}
        )


pytestmark = [pytest.mark.unit, pytest.mark.contract]
