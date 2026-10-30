"""Generation-3 CLI dispatch and asset-preflight tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import wave

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio_filter_pipeline.cli import main
from audio_filter_pipeline.processors.mos_contract import MOS_ASSET_LOAD_BINDING
from audio_filter_pipeline.processors.optional_runtime import (
    LocalAssetError,
    OptionalRuntimeConfigError,
    local_model_tree_sha256,
)
from audio_filter_pipeline.processors.stages_v3 import cli, mos_assets, mos_pool, mos_runtime
from audio_filter_pipeline.processors.stages_v3.mos_asset_preflight import (
    verify_mos_assets,
)
from audio_filter_pipeline.processors.stages_v3.mos_execution import MosExecutionPlan
from audio_filter_pipeline.utils import artifacts
from audio_filter_pipeline.utils.artifacts import PipelineStep
from audio_filter_pipeline.utils.run_lock import RunLock, RunLockError


pytestmark = pytest.mark.unit

_EXPECTED_GENERATION_3_CONSUMED_FIELDS = frozenset({
    "audio_export_max_inflight_bytes",
    "audio_export_max_pending",
    "audio_export_workers",
    "datasets",
    "dnsmos_model_dir",
    "min_source_sample_rate_hz",
    "nisqa_model_path",
    "nisqa_model_sha256",
    "nisqa_model_version",
    "output_root",
    "output_sample_rate_hz",
    "pipeline_generation",
    "scoreq_model_dir",
    "selection_policies",
    "speaker_consistency",
    "step",
    "step0_workers",
    "step2_max_segment_duration_sec",
    "step2_min_segment_duration_sec",
    "step3_gpu_ids",
    "step3_worker_hard_timeout_seconds",
    "step3_worker_idle_timeout_seconds",
    "step3_worker_stderr_tail_bytes",
    "step4_export_root",
    "step4_mode",
    "step5_apply_speaker_filter",
    "step6_export_root",
    "step6_mode",
    "urgent_pk_backbone",
    "urgent_pk_checkpoint",
    "urgent_pk_encoder",
    "urgent_pk_expected_commit",
    "urgent_pk_expected_sha256",
    "urgent_pk_score_range",
    "urgent_pk_source_dir",
    "use_gpu",
    "utmos_model_path",
    "utmos_model_sha256",
    "utmos_model_version",
    "vad",
})


def _config(tmp_path, **overrides):
    config = {
        "pipeline_generation": 3,
        "output_root": str(tmp_path / "out"),
        "datasets": {
            "alpha": {
                "input_root": str(tmp_path / "input"),
                "language": "en",
                "category": "read_speech",
                "quality_profile": "quality_high",
                "speaker": {"source": "none", "required": False},
            },
        },
        "selection_policies": {
            "global": {
                "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
                "min_mos_models": 3,
            },
            "quality_profiles": {
                name: {"thresholds": {
                    "scoreq": 4.0,
                    "urgent_pk": 3.5,
                    "dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6},
                }}
                for name in ("quality_balanced", "quality_high", "quality_premium")
            },
        },
    }
    config.update(overrides)
    return config


def _args(**overrides):
    values = {
        "pipeline_generation": None,
        "step": None,
        "output_root": None,
        "step0_workers": None,
        "use_gpu": None,
        "no_use_gpu": False,
        "step3_gpu_ids": None,
        "config": None,
        "drop_steps": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _scoreq_only(raw_config, tmp_path):
    """把 CLI 测试配置收窄为一个可本地 preflight 的 selected model。"""
    scoreq_dir = tmp_path / "scoreq"
    scoreq_dir.mkdir(exist_ok=True)
    (scoreq_dir / "model.bin").write_bytes(b"scoreq-test-model")
    raw_config["scoreq_model_dir"] = str(scoreq_dir)
    raw_config["selection_policies"]["global"] = {
        "mos_models": ["scoreq"], "min_mos_models": 1,
    }
    return raw_config


def _mos_config(
    tmp_path, *, step, speaker_consistency=None, step3_gpu_ids=(),
):
    """构造一个只选择 SCOREQ、可完成 plan/preflight 的 PipelineConfig。"""
    raw = _scoreq_only(_config(tmp_path, step=step), tmp_path)
    raw["speaker_consistency"] = speaker_consistency
    raw["step6_mode"] = "dry_run"
    raw["step3_gpu_ids"] = list(step3_gpu_ids)
    return cli._build_config(_args(step=step), raw, PipelineStep(step))


def _mos_plan_and_verified(config, *, use_gpu=False):
    plan = MosExecutionPlan.from_config(config, use_gpu=use_gpu)
    return plan, verify_mos_assets(plan)


def _verified_factories(
    args, raw_config, config, target_step, *, use_gpu=False, config_base_dir=None,
):
    plan, verified = _mos_plan_and_verified(config, use_gpu=use_gpu)
    return cli._factories(
        args,
        raw_config,
        config,
        target_step,
        plan=plan,
        verified=verified,
        config_base_dir=config_base_dir,
    )


def test_cli_factories_explicitly_bind_dataset_local_stages(tmp_path):
    config = _mos_config(tmp_path, step="step6")
    factories = _verified_factories(
        _args(step="step6"),
        {},
        config,
        PipelineStep.STEP6,
    )

    assert factories[PipelineStep.STEP3](config, "alpha")._dataset_id == "alpha"
    assert factories[PipelineStep.STEP4](config, "alpha")._dataset_id == "alpha"
    assert factories[PipelineStep.STEP6](config, "alpha")._dataset_id == "alpha"


def test_parser_accepts_v3_step0(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["audio_filter_pipeline.cli", "--step", "step0"])
    assert main.parse_args().step == "step0"


def test_parser_rejects_conflicting_gpu_flags(monkeypatch):
    monkeypatch.setattr(
        main.sys, "argv", ["audio_filter_pipeline.cli", "--use_gpu", "--no_use_gpu"],
    )

    with pytest.raises(SystemExit) as raised:
        main.parse_args()

    assert raised.value.code == 2


@pytest.mark.parametrize("generation", [None, 2, True, 3.0, "3"])
def test_cli_requires_explicit_exact_integer_generation_three(tmp_path, generation):
    raw = _config(tmp_path)
    if generation is None:
        raw.pop("pipeline_generation")
    else:
        raw["pipeline_generation"] = generation

    with pytest.raises(ValueError, match="pipeline_generation"):
        cli._build_config(_args(), raw, PipelineStep.STEP0)


@pytest.mark.parametrize(
    "legacy_field",
    [
        "mos_types",
        "mos_threshold",
        "mos_thresholds",
        "mos_fusion",
        "enable_multi_mos",
        "min_mos_models",
        "enable_signal_integrity_check",
        "signal_integrity_max_zero_ratio",
        "signal_integrity_min_rms_dbfs",
        "signal_integrity_max_dc_offset",
        "signal_integrity_clip_threshold",
        "step2_direct_from_input",
        "step2_mos_early_exit_threshold",
        "step1_vad_workers",
        "step1_vad_max_pending",
        "dataset_name",
    ],
)
def test_cli_rejects_all_unconsumed_legacy_constructor_fields(
    tmp_path, legacy_field,
):
    raw = _config(tmp_path)
    raw[legacy_field] = None

    with pytest.raises(ValueError, match="unconsumed"):
        cli._build_config(_args(), raw, PipelineStep.STEP0)


def test_cli_accepts_each_explicitly_consumed_top_level_field(tmp_path):
    assert cli._GENERATION_3_CONSUMED_FIELDS == (
        _EXPECTED_GENERATION_3_CONSUMED_FIELDS
    )
    raw = _config(tmp_path)
    for field in _EXPECTED_GENERATION_3_CONSUMED_FIELDS:
        candidate = dict(raw)
        if field not in candidate:
            candidate[field] = None
        cli._validate_top_level_config_fields(candidate)


@pytest.mark.parametrize("scenario", ["empty", "config_error", "drop"])
def test_cli_lightweight_paths_do_not_import_torch_in_subprocess(tmp_path, scenario):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    script = f"""
import sys
from types import SimpleNamespace
from audio_filter_pipeline.cli import main
if {scenario!r} in ('empty', 'config_error'):
    contents = {{
        'empty': '{{}}\\n',
        'config_error': 'pipeline_generation: 3\\nmos_types: []\\n',
    }}[{scenario!r}]
    with open({str(config_path)!r}, 'w', encoding='utf-8') as stream:
        stream.write(contents)
    sys.argv = ['audio_filter_pipeline.cli', '--config', {str(config_path)!r}]
    assert main.main() == 1
else:
    from audio_filter_pipeline.processors.stages_v3 import cli
    args = SimpleNamespace(step='step0', output_root=None, output=None,
        step0_workers=None, use_gpu=None, no_use_gpu=False,
        step3_gpu_ids=None, drop_steps='step0')
    raw = {{'pipeline_generation': 3, 'output_root': {str(tmp_path / 'out')!r},
        'datasets': {{'alpha': {{'input_root': {str(tmp_path / 'input')!r},
            'language': 'en', 'category': 'read_speech',
            'quality_profile': 'quality_high',
            'speaker': {{'source': 'none', 'required': False}}}}}},
        'selection_policies': {{'quality_profiles': {{name: {{'thresholds': {{
            'scoreq': 4.0, 'urgent_pk': 3.5,
            'dnsmos': {{'ovrl': 3.2, 'sig': 3.0, 'bak': 3.8, 'p808': 3.6}},
        }}}} for name in ('quality_balanced', 'quality_high', 'quality_premium')}}}}}}
    assert cli.run_v3_cli(args, raw) == 0
assert 'torch' not in sys.modules
"""
    subprocess_cwd = tmp_path / "subprocess-cwd"
    subprocess_cwd.mkdir()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=subprocess_cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_drop_steps_help_is_explicitly_non_cascading(monkeypatch, capsys):
    monkeypatch.setattr(main.sys, "argv", ["audio_filter_pipeline.cli", "--help"])

    with pytest.raises(SystemExit) as raised:
        main.parse_args()

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "不会级联删除" in help_text
    assert "不会删除已导出的 FLAC/manifest" in help_text
    assert "另行运行目标 step 重建" in help_text


def test_help_marks_spk_filter_as_non_executable_asset_template(monkeypatch, capsys):
    monkeypatch.setattr(main.sys, "argv", ["audio_filter_pipeline.cli", "--help"])

    with pytest.raises(SystemExit) as raised:
        main.parse_args()

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "schema-valid fail-closed 模板" in help_text
    assert "替换 speaker assets" in help_text
    assert "不可直接运行" in help_text


def test_cli_build_config_preserves_yaml_step3_gpu_ids(tmp_path):
    config = cli._build_config(
        _args(), _config(tmp_path, step3_gpu_ids=[2, 5]), PipelineStep.STEP3,
    )
    assert config.step3_gpu_ids == (2, 5)


def test_cli_output_root_uses_canonical_cli_then_yaml(tmp_path):
    """输出根只接受 canonical CLI `--output_root` 与 YAML `output_root`。"""
    raw = _config(tmp_path, output_root="yaml-canonical-output")

    assert cli._build_config(
        _args(output_root="cli-canonical-output"), raw, PipelineStep.STEP0,
    ).output_root == "cli-canonical-output"
    assert cli._build_config(
        _args(), raw, PipelineStep.STEP0,
    ).output_root == "yaml-canonical-output"


def test_cli_rejects_legacy_output_root_alias(tmp_path):
    """YAML `output` 别名已移除，必须按未知字段 fail-fast。"""
    raw = _config(tmp_path)
    raw["output"] = "yaml-alias-output"

    with pytest.raises(ValueError, match="unconsumed"):
        cli._build_config(_args(), raw, PipelineStep.STEP0)


def test_parser_rejects_legacy_output_cli_alias(monkeypatch):
    """`--output/-o` CLI 别名已移除，argparse 必须直接拒绝。"""
    for flag in ("--output", "-o"):
        monkeypatch.setattr(
            main.sys,
            "argv",
            ["audio_filter_pipeline.cli", "--step", "step0", flag, "/tmp/out"],
        )
        with pytest.raises(SystemExit) as raised:
            main.parse_args()
        assert raised.value.code == 2


@pytest.mark.parametrize(
    "legacy_field",
    [
        "step6_export_workers",
        "step6_export_max_pending",
        "step6_export_max_inflight_bytes",
        "output_format",
        "output_subtype",
    ],
)
def test_cli_rejects_unconsumed_export_fields_as_unknown(tmp_path, legacy_field):
    raw = _config(tmp_path)
    raw[legacy_field] = 1

    with pytest.raises(ValueError, match="unknown generation-3 configuration fields"):
        cli._build_config(_args(), raw, PipelineStep.STEP4)


def test_cli_step3_gpu_ids_override_yaml_and_reject_malformed_values(tmp_path):
    raw = _config(tmp_path, step3_gpu_ids=[2, 5])
    assert cli._build_config(_args(step3_gpu_ids="3,7"), raw, PipelineStep.STEP3).step3_gpu_ids == (3, 7)
    with pytest.raises(ValueError, match="step3_gpu_ids"):
        cli._build_config(_args(step3_gpu_ids="3,,7"), raw, PipelineStep.STEP3)


@pytest.mark.parametrize("workers", [0, -1, True, 1.0, "2"])
def test_cli_rejects_nonpositive_or_noninteger_step0_workers(tmp_path, workers):
    with pytest.raises(ValueError, match="step0_workers"):
        cli._build_config(_args(step0_workers=workers), _config(tmp_path), PipelineStep.STEP0)


def test_step3_gpu_plan_requires_physical_gpu_ids_before_factories(
    monkeypatch, tmp_path,
):
    config = _mos_config(tmp_path, step="step3")
    monkeypatch.setattr(
        cli, "_factories",
        lambda *_args, **_kwargs: pytest.fail("invalid plan must not reach factories"),
    )

    with pytest.raises(ValueError, match="gpu_ids"):
        MosExecutionPlan.from_config(config, use_gpu=True)


@pytest.mark.parametrize("use_gpu", [False, True], ids=["runtime", "pool"])
def test_step3_runtime_and_pool_receive_complete_verified_plan_without_signature_probe(
    monkeypatch, tmp_path, use_gpu,
):
    captured = {}

    class Runtime:
        def __init__(self, *, verified_plan):
            captured.update(kind="runtime", verified_plan=verified_plan)

        def provenance_payload(self):
            return {"scorer_execution": "test-runtime"}

        def score_task_groups(self, *_args, **_kwargs):
            return ()

    class Pool:
        def __init__(
            self, gpu_ids, *, verified_plan, dataset_id,
            worker_idle_timeout_seconds, worker_hard_timeout_seconds,
            worker_stderr_tail_bytes,
        ):
            captured.update(
                kind="pool",
                gpu_ids=gpu_ids,
                verified_plan=verified_plan,
                dataset_id=dataset_id,
                worker_idle_timeout_seconds=worker_idle_timeout_seconds,
                worker_hard_timeout_seconds=worker_hard_timeout_seconds,
                worker_stderr_tail_bytes=worker_stderr_tail_bytes,
            )

        def provenance_payload(self):
            return {"scorer_execution": "test-pool"}

        def score_task_groups(self, *_args, **_kwargs):
            return ()

    monkeypatch.setattr(cli, "Step3MosRuntime", Runtime)
    monkeypatch.setattr(cli, "Step3MosScorerPool", Pool)
    config = _mos_config(
        tmp_path, step="step3", step3_gpu_ids=(7,) if use_gpu else (),
    )
    plan, verified = _mos_plan_and_verified(config, use_gpu=use_gpu)
    factories = cli._factories(
        _args(use_gpu=not use_gpu), {}, config, PipelineStep.STEP3,
        plan=plan, verified=verified,
    )
    factories[PipelineStep.STEP3](config, "alpha")

    assert captured["kind"] == ("pool" if use_gpu else "runtime")
    assert captured["verified_plan"] is verified
    if use_gpu:
        assert captured["gpu_ids"] == (7,)
        assert captured["worker_idle_timeout_seconds"] == 900.0
        assert captured["worker_hard_timeout_seconds"] is None


def test_cli_rejects_removed_step3_worker_timeout_key(tmp_path):
    """`step3_worker_timeout_seconds` 已移除，配置里出现即按未知字段 fail-fast。"""
    raw = _config(
        tmp_path,
        step3_worker_timeout_seconds=7_200,
        step3_worker_hard_timeout_seconds=21_600,
    )

    with pytest.raises(ValueError, match="unconsumed"):
        cli._build_config(_args(), raw, PipelineStep.STEP3)


def test_cli_consumes_canonical_step3_worker_timeouts(tmp_path):
    raw = _config(
        tmp_path,
        step3_worker_idle_timeout_seconds=7_200,
        step3_worker_hard_timeout_seconds=21_600,
    )

    config = cli._build_config(_args(), raw, PipelineStep.STEP3)

    assert config.step3_worker_idle_timeout_seconds == 7_200.0
    assert config.step3_worker_hard_timeout_seconds == 21_600.0
    assert config.to_dict()["step3_worker_idle_timeout_seconds"] == 7_200.0


def test_wespeaker_payload_requires_canonical_to_payload():
    class LegacyOptions:
        model_path = "/model"

    with pytest.raises(AttributeError, match="to_payload"):
        cli._wespeaker_options_payload(LegacyOptions())


def test_wespeaker_payload_rejects_non_json_canonical_payload():
    class Options:
        def to_payload(self):
            return {"model_path": "/model", "worker_limits": object()}

    with pytest.raises(TypeError, match="JSON-safe"):
        cli._wespeaker_options_payload(Options())


def test_wespeaker_payload_preserves_serialized_nested_limits():
    class Options:
        def to_payload(self):
            return {
                "model_path": "/model",
                "worker_limits": {"timeout_ms": 1_000},
                "asset_limits": {"max_entries": 12},
            }

    assert cli._wespeaker_options_payload(Options()) == {
        "model_path": "/model",
        "worker_limits": {"timeout_ms": 1_000},
        "asset_limits": {"max_entries": 12},
    }


def test_load_config_propagates_yaml_and_top_level_type_errors(tmp_path):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("datasets: [", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        main.load_config(str(malformed))

    for filename, contents in (
        ("empty.yaml", ""),
        ("scalar.yaml", "- not-a-mapping\n"),
    ):
        config_path = tmp_path / filename
        config_path.write_text(contents, encoding="utf-8")
        with pytest.raises(TypeError, match="mapping"):
            main.load_config(str(config_path))


def test_load_config_propagates_missing_file_error(tmp_path):
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError) as raised:
        main.load_config(str(missing))

    assert raised.value.filename == str(missing)


def test_run_lock_writes_owner_rejects_conflicts_and_reuses_persistent_inode(tmp_path):
    output_root = tmp_path / "out"
    lock = RunLock(str(output_root), "alpha")
    lock.acquire()
    lock_path = Path(lock.lock_path)
    identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    previous_token = lock.token
    try:
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["schema"] == "audio_filter_pipeline.run_lock.v1"
        assert owner["token"] == previous_token
        assert owner["output_root"] == str(output_root.resolve())
        assert owner["dataset_id"] == "alpha"
        with pytest.raises(RunLockError, match="already held"):
            RunLock(str(output_root), "alpha").acquire()
    finally:
        lock.release()

    assert lock_path.exists()
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == identity
    lock_path.write_text(json.dumps({"token": "stale-owner"}), encoding="utf-8")

    recovered = RunLock(str(output_root), "alpha")
    recovered.acquire()
    try:
        assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == identity
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["token"] == recovered.token
        assert recovered.token != previous_token
    finally:
        recovered.release()
    assert lock_path.exists()


def test_run_lock_allows_different_datasets_under_same_output_root(tmp_path):
    output_root = tmp_path / "out"
    alpha = RunLock(str(output_root), "alpha").acquire()
    beta = RunLock(str(output_root), "beta").acquire()
    try:
        assert alpha._fd is not None
        assert beta._fd is not None
        assert alpha._anchor_fd is not None
        assert beta._anchor_fd is not None
        with pytest.raises(RunLockError, match="already held"):
            RunLock(str(output_root), "alpha").acquire()
    finally:
        beta.release()
        alpha.release()


def test_run_lock_holds_real_flock_on_dataset_inode(tmp_path):
    import fcntl

    lock = RunLock(str(tmp_path / "out"), "alpha").acquire()
    contender_fd = os.open(lock.lock_path, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(contender_fd)
        lock.release()


def test_run_lock_name_is_independent_of_output_root_pathname(tmp_path):
    first = RunLock(str(tmp_path / "mount-a"), "alpha")
    second = RunLock(str(tmp_path / "mount-b"), "alpha")

    assert Path(first.lock_path).name == Path(second.lock_path).name
    assert first._protocol_anchor_name == second._protocol_anchor_name


def test_run_lock_rejects_preexisting_external_hardlink(tmp_path):
    lock = RunLock(str(tmp_path / "out"), "alpha")
    lock_path = Path(lock.lock_path)
    lock_path.parent.mkdir(parents=True)
    external = tmp_path / "external.lock"
    external.write_bytes(b"foreign inode\n")
    os.link(external, lock_path)

    with pytest.raises(RunLockError, match="link topology"):
        lock.acquire()

    assert external.read_bytes() == b"foreign inode\n"
    assert lock_path.stat().st_nlink == 2


def test_run_lock_rejects_third_hardlink_to_persistent_inode(tmp_path):
    initial = RunLock(str(tmp_path / "out"), "alpha").acquire()
    lock_path = Path(initial.lock_path)
    initial.release()
    third_link = tmp_path / "third.lock"
    os.link(lock_path, third_link)

    with pytest.raises(RunLockError, match="link topology"):
        RunLock(str(tmp_path / "out"), "alpha").acquire()

    assert lock_path.stat().st_nlink == 3


def test_run_lock_rejects_lock_inode_owned_by_other_uid(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.run_lock as run_lock_module

    lock = RunLock(str(tmp_path / "out"), "alpha")
    real_fstat = os.fstat

    def fstat_with_foreign_file_owner(descriptor):
        result = real_fstat(descriptor)
        if not stat.S_ISREG(result.st_mode):
            return result
        return SimpleNamespace(
            st_mode=result.st_mode,
            st_dev=result.st_dev,
            st_ino=result.st_ino,
            st_uid=os.geteuid() + 1,
            st_nlink=result.st_nlink,
        )

    import stat

    monkeypatch.setattr(run_lock_module.os, "fstat", fstat_with_foreign_file_owner)
    with pytest.raises(RunLockError, match="owner UID"):
        lock.acquire()


def test_run_lock_rejects_pathname_replacement_after_file_flock(
    tmp_path, monkeypatch,
):
    import fcntl
    import stat
    import audio_filter_pipeline.utils.run_lock as run_lock_module

    lock = RunLock(str(tmp_path / "out"), "alpha")
    lock_path = Path(lock.lock_path)
    displaced_path = lock_path.with_suffix(".displaced")
    replacement = b"foreign replacement\n"
    real_flock = fcntl.flock
    replaced = False

    def flock_then_replace(descriptor, operation):
        nonlocal replaced
        result = real_flock(descriptor, operation)
        if (
            not replaced
            and operation == fcntl.LOCK_EX | fcntl.LOCK_NB
            and stat.S_ISREG(os.fstat(descriptor).st_mode)
        ):
            replaced = True
            lock_path.rename(displaced_path)
            lock_path.write_bytes(replacement)
        return result

    monkeypatch.setattr(run_lock_module.fcntl, "flock", flock_then_replace)

    with pytest.raises(RunLockError, match="replaced|disappeared"):
        lock.acquire()

    assert replaced
    assert lock_path.read_bytes() == replacement
    assert lock._fd is None
    assert lock._acquired is False


def test_run_lock_rejects_contender_after_held_pathname_is_replaced(tmp_path):
    lock = RunLock(str(tmp_path / "out"), "alpha").acquire()
    lock_path = Path(lock.lock_path)
    displaced_path = lock_path.with_suffix(".held")
    contender = RunLock(str(tmp_path / "out"), "alpha")
    lock_path.rename(displaced_path)
    lock_path.write_bytes(b"foreign replacement\n")

    try:
        with pytest.raises(RunLockError, match="already held"):
            contender.acquire()
        assert contender._fd is None
        assert contender._acquired is False
    finally:
        if contender._acquired:
            contender.release()
        displaced_path.replace(lock_path)
        lock.release()


def test_run_lock_release_fails_closed_on_held_pathname_replacement(tmp_path):
    lock = RunLock(str(tmp_path / "out"), "alpha").acquire()
    lock_path = Path(lock.lock_path)
    displaced_path = lock_path.with_suffix(".held")
    lock_path.rename(displaced_path)
    lock_path.write_bytes(b"foreign replacement\n")

    try:
        with pytest.raises(RunLockError, match="replaced|disappeared"):
            lock.release()
        assert lock._acquired is True
        assert lock._fd is not None
    finally:
        displaced_path.replace(lock_path)
        if lock._acquired:
            lock.release()


def test_run_lock_rejects_oversized_owner_before_truncating_existing_owner(
    tmp_path, monkeypatch,
):
    output_root = tmp_path / "out"
    initial = RunLock(str(output_root), "alpha").acquire()
    lock_path = Path(initial.lock_path)
    initial.release()
    sentinel = b'{"token":"sentinel"}\n'
    lock_path.write_bytes(sentinel)
    monkeypatch.setattr(sys, "argv", ["pipeline", "x" * RunLock._MAX_OWNER_BYTES])

    with pytest.raises(RunLockError, match="too large"):
        RunLock(str(output_root), "alpha").acquire()

    assert lock_path.read_bytes() == sentinel


def test_run_lock_does_not_remove_unknown_token(tmp_path):
    lock = RunLock(str(tmp_path / "out"), "alpha")
    lock.acquire()
    lock_path = Path(lock.lock_path)
    original_owner = lock_path.read_bytes()
    lock_path.write_text(
        json.dumps({"token": "different-owner"}), encoding="utf-8",
    )
    with pytest.raises(RunLockError, match="token changed"):
        lock.release()
    assert lock_path.exists()
    lock_path.write_bytes(original_owner)
    lock.release()


def test_run_lock_rejects_owner_metadata_tamper_with_same_token(tmp_path):
    lock = RunLock(str(tmp_path / "out"), "alpha")
    lock.acquire()
    lock_path = Path(lock.lock_path)
    original_owner = lock_path.read_bytes()
    tampered_owner = json.loads(original_owner)
    tampered_owner["pid"] = tampered_owner["pid"] + 1
    lock_path.write_text(json.dumps(tampered_owner), encoding="utf-8")
    with pytest.raises(RunLockError, match="metadata changed"):
        lock.release()
    lock_path.write_bytes(original_owner)
    lock.release()


def test_run_lock_rejects_symlinked_directory_and_file(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    lock_directory = output_root / RunLock._LOCK_DIRECTORY
    target_directory = tmp_path / "target-locks"
    target_directory.mkdir()
    lock_directory.symlink_to(target_directory, target_is_directory=True)
    with pytest.raises(RunLockError, match="symlink"):
        RunLock(str(output_root), "alpha").acquire()

    lock_directory.unlink()
    lock_directory.mkdir()
    lock = RunLock(str(output_root), "alpha")
    target_file = tmp_path / "target.lock"
    target_file.write_text("not-owner", encoding="utf-8")
    Path(lock.lock_path).symlink_to(target_file)
    with pytest.raises(RunLockError, match="symlink"):
        lock.acquire()


def test_run_lock_release_uses_held_fd_and_preserves_inode(tmp_path):
    lock = RunLock(str(tmp_path / "out"), "alpha")
    lock.acquire()
    original_path = Path(lock.lock_path)
    original_bytes = original_path.read_bytes()
    replacement_path = tmp_path / "not-the-lock.lock"
    lock.lock_path = str(replacement_path)
    lock.release()
    assert original_path.exists()
    assert original_path.read_bytes() == original_bytes
    assert not replacement_path.exists()


def test_run_lock_exit_preserves_business_exception_when_release_fails(tmp_path):
    lock = RunLock(str(tmp_path / "out"), "alpha")
    original_path = Path(lock.lock_path)
    with pytest.raises(ValueError, match="business") as raised:
        with lock:
            Path(lock.lock_path).write_text(
                json.dumps({"token": lock.token}), encoding="utf-8",
            )
            raise ValueError("business failure")

    assert any(
        "RunLock release failed" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    original_path.write_bytes(lock._owner_bytes)
    lock.release()


def test_cli_normal_run_and_drop_steps_release_dataset_lock(monkeypatch, tmp_path):
    config = _config(tmp_path, step="step0")
    calls = []

    class FakeRunner:
        def __init__(self, _config, _factories):
            pass

        def run(self, dataset_id, requested_step):
            calls.append((dataset_id, requested_step))
            return SimpleNamespace(target=None)

    monkeypatch.setattr(cli, "V3PipelineRunner", FakeRunner)
    assert cli.run_v3_cli(_args(step="step0"), config) == 0
    lock_path = RunLock(config["output_root"], "alpha").lock_path
    assert Path(lock_path).exists()
    assert calls == [("alpha", PipelineStep.STEP0)]

    assert cli.run_v3_cli(_args(step="step0", drop_steps="step0"), config) == 0
    assert Path(lock_path).exists()


def test_cli_run_and_drop_fail_closed_on_lock_conflict(monkeypatch, tmp_path):
    config = _config(tmp_path, step="step0")
    held = RunLock(config["output_root"], "alpha")
    held.acquire()
    try:
        with pytest.raises(RunLockError, match="already held"):
            cli.run_v3_cli(_args(step="step0"), config)
        with pytest.raises(RunLockError, match="already held"):
            cli.run_v3_cli(_args(step="step0", drop_steps="step0"), config)
    finally:
        held.release()


def test_cli_cleans_staging_inside_dataset_lock_before_runner(
    monkeypatch, tmp_path,
):
    config = _config(tmp_path, step="step0")
    lock_path = RunLock(config["output_root"], "alpha").lock_path
    calls = []

    def cleanup(output_root, dataset_id):
        calls.append(("cleanup", output_root, dataset_id, Path(lock_path).exists()))
        return 0

    class FakeRunner:
        def __init__(self, _config, _factories):
            pass

        def run(self, dataset_id, requested_step):
            calls.append(("run", dataset_id, requested_step, Path(lock_path).exists()))
            return SimpleNamespace(target=None)

    monkeypatch.setattr(cli, "_cleanup_dataset_temp_artifacts", cleanup)
    monkeypatch.setattr(cli, "V3PipelineRunner", FakeRunner)

    assert cli.run_v3_cli(_args(step="step0"), config) == 0
    assert calls == [
        ("cleanup", config["output_root"], "alpha", True),
        ("run", "alpha", PipelineStep.STEP0, True),
    ]
    assert Path(lock_path).exists()


def test_cli_cleanup_removes_only_current_dataset_staging_and_preserves_published(
    tmp_path,
):
    config = _config(tmp_path, step="step0")
    published = Path(cli.v3_artifact_dir(config["output_root"], PipelineStep.STEP0, "alpha"))
    staging = published.parent / f".{published.name}.stage-old"
    new_staging = published.parent / f".{published.name}.stage-new"
    unrelated = published.parent / ".beta.stage-old"
    published.mkdir(parents=True)
    (published / "_SUCCESS").write_text("published", encoding="utf-8")
    staging.write_text("old staging", encoding="utf-8")
    new_staging.write_text("new staging", encoding="utf-8")
    unrelated.write_text("unrelated", encoding="utf-8")
    os.utime(staging, (1, 1))
    os.utime(unrelated, (1, 1))

    assert cli._cleanup_dataset_temp_artifacts(config["output_root"], "alpha") == 1
    assert published.exists()
    assert (published / "_SUCCESS").exists()
    assert not staging.exists()
    assert new_staging.exists()
    assert unrelated.exists()


def _write_pcm_wav(path, sample_rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x00\x00" * max(1, sample_rate // 20))


def _artifact_rows(raw_config, step=PipelineStep.STEP0):
    rows_path = Path(cli.v3_artifact_dir(
        raw_config["output_root"], step, "alpha",
    )) / "rows.jsonl"
    return [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
    ]


def test_main_dispatches_canonical_explicit_config_directory(monkeypatch, tmp_path):
    config_path = tmp_path / "canonical" / "pipeline.yaml"
    config_path.parent.mkdir()
    config_path.write_text("pipeline_generation: 3\n", encoding="utf-8")
    alias_path = tmp_path / "pipeline-link.yaml"
    alias_path.symlink_to(config_path)
    args = _args(
        pipeline_generation=3,
        step="step0",
        config=str(alias_path),
    )
    config = _config(tmp_path)
    calls = []
    monkeypatch.setattr(main, "parse_args", lambda: args)

    def load_config(path):
        calls.append(("load", path))
        return config

    def run_v3_cli(got_args, got_config, *, config_base_dir=None):
        calls.append(("run", got_args, got_config, config_base_dir))
        return 0

    monkeypatch.setattr(main, "load_config", load_config)
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.cli.run_v3_cli",
        run_v3_cli,
    )

    assert main.main() == 0
    canonical_path = str(config_path.resolve())
    assert args.config == canonical_path
    assert calls == [
        ("load", canonical_path),
        ("run", args, config, str(config_path.parent.resolve())),
    ]


@pytest.mark.parametrize(
    "asset_error",
    [
        LocalAssetError("missing model tree"),
        OptionalRuntimeConfigError("invalid local asset configuration"),
    ],
)
def test_main_reports_expected_local_asset_errors_without_traceback(
    monkeypatch, tmp_path, capsys, asset_error,
):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("pipeline_generation: 3\n", encoding="utf-8")
    monkeypatch.setattr(
        main, "parse_args", lambda: _args(config=str(config_path)),
    )
    monkeypatch.setattr(main, "load_config", lambda _path: _config(tmp_path))
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.cli.run_v3_cli",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(asset_error),
    )

    assert main.main() == 1
    captured = capsys.readouterr()
    assert "配置或本地资产无效" in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_main_does_not_swallow_unexpected_runtime_error(monkeypatch, tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("pipeline_generation: 3\n", encoding="utf-8")
    runtime_error = RuntimeError("unexpected stage failure")
    monkeypatch.setattr(
        main, "parse_args", lambda: _args(config=str(config_path)),
    )
    monkeypatch.setattr(main, "load_config", lambda _path: _config(tmp_path))
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.cli.run_v3_cli",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(runtime_error),
    )

    with pytest.raises(RuntimeError) as raised:
        main.main()
    assert raised.value is runtime_error


def test_main_without_config_rejects_existing_local_legacy_fallback(
    monkeypatch, tmp_path, capsys,
):
    module_path = (
        tmp_path / "checkout" / "tools" / "audio_filter_pipeline" / "cli" / "main.py"
    )
    legacy_config = module_path.parent / "config" / "audio_filter.yaml"
    legacy_config.parent.mkdir(parents=True)
    legacy_config.write_text(
        yaml.safe_dump(_config(tmp_path)), encoding="utf-8",
    )
    monkeypatch.setattr(main, "__file__", str(module_path))
    monkeypatch.setattr(main, "parse_args", lambda: _args(config=None))
    monkeypatch.setattr(
        main,
        "load_config",
        lambda _path: pytest.fail("missing --config must not load a fallback"),
    )
    monkeypatch.setattr(
        "audio_filter_pipeline.processors.stages_v3.cli.run_v3_cli",
        lambda *_args, **_kwargs: pytest.fail(
            "missing --config must not dispatch the v3 CLI"
        ),
    )

    assert main.main() == 1
    assert "请用 --config 显式指定" in capsys.readouterr().out


def test_main_without_config_returns_one_when_isolated_fallback_is_missing(
    monkeypatch, tmp_path, capsys,
):
    module_path = (
        tmp_path / "checkout" / "tools" / "audio_filter_pipeline" / "cli" / "main.py"
    )
    monkeypatch.setattr(main, "__file__", str(module_path))
    monkeypatch.setattr(main, "parse_args", lambda: _args(config=None))
    monkeypatch.setattr(
        main,
        "load_config",
        lambda _path: pytest.fail("missing --config must not load a fallback"),
    )

    assert main.main() == 1
    assert "请用 --config 显式指定" in capsys.readouterr().out


def test_main_missing_explicit_config_returns_one_without_loading(
    monkeypatch, tmp_path, capsys,
):
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(
        main, "parse_args", lambda: _args(config=str(missing)),
    )
    monkeypatch.setattr(
        main,
        "load_config",
        lambda _path: pytest.fail("missing explicit config must not be loaded"),
    )

    assert main.main() == 1
    assert str(missing.resolve()) in capsys.readouterr().out


def test_main_reports_config_removed_after_isfile_check(
    monkeypatch, tmp_path, capsys,
):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("pipeline_generation: 3\n", encoding="utf-8")
    canonical_path = str(config_path.resolve())
    monkeypatch.setattr(
        main, "parse_args", lambda: _args(config=str(config_path)),
    )

    def isfile_then_remove(path):
        assert path == canonical_path
        config_path.unlink()
        return True

    monkeypatch.setattr(main.os.path, "isfile", isfile_then_remove)

    assert main.main() == 1
    captured = capsys.readouterr()
    assert "配置文件不存在" in captured.out
    assert canonical_path in captured.out
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize(
    ("filename", "contents"),
    [
        ("malformed.yaml", b"datasets: ["),
        ("empty.yaml", b""),
        ("sequence.yaml", b"- not-a-mapping\n"),
        ("invalid-utf8.yaml", b"\xff"),
    ],
    ids=["yaml", "empty", "top-level-type", "unicode-decode"],
)
def test_main_reports_invalid_explicit_config_without_traceback(
    monkeypatch, tmp_path, capsys, filename, contents,
):
    config_path = tmp_path / "config" / filename
    config_path.parent.mkdir()
    config_path.write_bytes(contents)
    alias_path = tmp_path / "pipeline-link.yaml"
    alias_path.symlink_to(config_path)
    canonical_path = str(config_path.resolve())
    monkeypatch.setattr(
        main, "parse_args", lambda: _args(config=str(alias_path)),
    )

    assert main.main() == 1
    captured = capsys.readouterr()
    assert "错误: 无法加载配置文件" in captured.out
    assert canonical_path in captured.out
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize(
    "load_error",
    [
        PermissionError("permission denied"),
        IsADirectoryError("is a directory"),
        OSError("I/O error"),
    ],
    ids=["permission", "is-a-directory", "os-error"],
)
def test_main_reports_os_errors_after_explicit_config_isfile_check(
    monkeypatch, tmp_path, capsys, load_error,
):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("pipeline_generation: 3\n", encoding="utf-8")
    canonical_path = str(config_path.resolve())
    monkeypatch.setattr(
        main, "parse_args", lambda: _args(config=str(config_path)),
    )
    monkeypatch.setattr(
        main, "load_config", lambda _path: (_ for _ in ()).throw(load_error),
    )

    assert main.main() == 1
    captured = capsys.readouterr()
    assert "错误: 无法加载配置文件" in captured.out
    assert canonical_path in captured.out
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize(
    "fatal_error",
    [KeyboardInterrupt(), SystemExit(7), MemoryError("out of memory")],
    ids=["keyboard-interrupt", "system-exit", "memory-error"],
)
def test_main_does_not_swallow_process_control_or_memory_errors(
    monkeypatch, tmp_path, fatal_error,
):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("pipeline_generation: 3\n", encoding="utf-8")
    monkeypatch.setattr(
        main, "parse_args", lambda: _args(config=str(config_path)),
    )
    monkeypatch.setattr(
        main, "load_config", lambda _path: (_ for _ in ()).throw(fatal_error),
    )

    with pytest.raises(type(fatal_error)) as raised:
        main.main()
    assert raised.value is fatal_error


def test_cli_min_source_sample_rate_floor_runs_through_real_step0(tmp_path):
    input_root = tmp_path / "input"
    _write_pcm_wav(input_root / "low.wav", 8_000)
    _write_pcm_wav(input_root / "kept.wav", 16_000)
    raw = _config(
        tmp_path,
        step="step0",
        min_source_sample_rate_hz=16_000,
    )

    assert cli.run_v3_cli(_args(step="step0"), raw) == 0
    rows = _artifact_rows(raw)
    assert [row["source_relative_path"] for row in rows] == ["kept.wav"]
    assert rows[0]["sample_rate"] == 16_000


def test_cli_resolves_relative_speaker_mapping_from_config_directory(tmp_path):
    input_root = tmp_path / "input"
    _write_pcm_wav(input_root / "utt.wav", 16_000)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "speakers.jsonl").write_text(
        json.dumps({
            "source_relative_path": "utt.wav",
            "speaker_id": "speaker-a",
        }) + "\n",
        encoding="utf-8",
    )
    raw = _config(tmp_path, step="step0")
    raw["datasets"]["alpha"]["speaker"] = {
        "source": "mapping_file",
        "mapping_path": "speakers.jsonl",
        "required": True,
    }

    assert cli.run_v3_cli(
        _args(step="step0"),
        raw,
        config_base_dir=str(config_dir.resolve()),
    ) == 0
    rows = _artifact_rows(raw)
    assert rows[0]["speaker_id"] == "speaker-a"
    assert "config_base_dir" not in cli._build_config(
        _args(step="step0"), raw, PipelineStep.STEP0,
    ).to_dict()


def test_cli_config_base_dir_must_be_absolute_existing_and_canonical(
    tmp_path,
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bad_values = (
        "config",
        str((tmp_path / "missing").resolve()),
        str(config_dir / ".." / "config"),
    )

    for value in bad_values:
        with pytest.raises(ValueError, match="config_base_dir"):
            cli.run_v3_cli(
                _args(step="step0", drop_steps="step0"),
                _config(tmp_path),
                config_base_dir=value,
            )


def test_drop_steps_does_not_construct_mapping_resolver(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    raw = _config(tmp_path, step="step0")
    raw["datasets"]["alpha"]["speaker"] = {
        "source": "mapping_file",
        "mapping_path": "missing-speakers.jsonl",
        "required": True,
    }
    dropped = []
    monkeypatch.setattr(
        cli,
        "drop_artifact",
        lambda root, step, dataset: dropped.append((root, step, dataset)),
    )

    assert cli.run_v3_cli(
        _args(step="step0", drop_steps="step0"),
        raw,
        config_base_dir=str(config_dir.resolve()),
    ) == 0
    assert dropped == [
        (raw["output_root"], PipelineStep.STEP0, "alpha"),
    ]


def test_cli_orders_validate_plan_verify_factories_before_runner(
    monkeypatch, tmp_path,
):
    raw = _scoreq_only(_config(tmp_path, step="step3"), tmp_path)
    config = cli._build_config(_args(step="step3"), raw, PipelineStep.STEP3)
    calls = []
    plan = object()
    verified = object()
    factories = {step: object() for step in PipelineStep}

    monkeypatch.setattr(config, "validate", lambda **_kwargs: calls.append("validate"))

    class PlanFactory:
        @staticmethod
        def from_config(got_config, *, use_gpu):
            assert got_config is config
            assert use_gpu is False
            calls.append("plan")
            return plan

    def verify(got_plan):
        assert got_plan is plan
        calls.append("verify")
        return verified

    def build_factories(
        _args_value, _raw, got_config, target_step, got_plan, got_verified,
        *, config_base_dir=None,
    ):
        assert got_config is config
        assert target_step is PipelineStep.STEP3
        assert got_plan is plan
        assert got_verified is verified
        assert config_base_dir is None
        calls.append("factories")
        return factories

    class Runner:
        def __init__(self, got_config, got_factories):
            assert got_config is config
            assert got_factories is factories
            calls.append("runner")

        def run(self, dataset_id, requested_step):
            assert (dataset_id, requested_step) == ("alpha", PipelineStep.STEP3)
            return SimpleNamespace(target=None)

    monkeypatch.setattr(cli, "_build_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(cli, "MosExecutionPlan", PlanFactory)
    monkeypatch.setattr(cli, "verify_mos_assets", verify)
    monkeypatch.setattr(cli, "_factories", build_factories)
    monkeypatch.setattr(cli, "V3PipelineRunner", Runner)
    monkeypatch.setattr(cli, "_cleanup_dataset_temp_artifacts", lambda *_args: 0)

    assert cli.run_v3_cli(_args(step="step3"), raw) == 0
    assert calls == ["validate", "plan", "verify", "factories", "runner"]


def test_cli_gpu_fleet_is_lazy_on_cache_hits_and_shared_across_datasets(
    monkeypatch, tmp_path,
):
    raw = _scoreq_only(
        _config(tmp_path, step="step3", step3_gpu_ids=[7]), tmp_path,
    )
    raw["datasets"]["beta"] = dict(raw["datasets"]["alpha"])
    raw["datasets"]["beta"]["input_root"] = str(tmp_path / "input-beta")
    fleets = []
    starts = []
    views = []

    def build_fleet(*args, **kwargs):
        kwargs["popen"] = lambda *_args, **_kwargs: starts.append(True)
        fleet = mos_pool.Step3MosWorkerFleet(*args, **kwargs)
        original = fleet.for_dataset

        def for_dataset(dataset_id):
            views.append(dataset_id)
            return original(dataset_id)

        fleet.for_dataset = for_dataset
        fleets.append(fleet)
        return fleet

    class CacheHitRunner:
        def __init__(self, config, factories):
            self._config = config
            self._factories = factories

        def run(self, dataset_id, requested_step):
            assert requested_step is PipelineStep.STEP3
            # Cache-hit 仍解析 stage/config digest，但不会消费 scorer、启动 child。
            stage = self._factories[PipelineStep.STEP3](self._config, dataset_id)
            stage.config_payload(self._config)
            return SimpleNamespace(target=None)

    monkeypatch.setattr(cli, "Step3MosWorkerFleet", build_fleet)
    monkeypatch.setattr(cli, "V3PipelineRunner", CacheHitRunner)
    monkeypatch.setattr(cli, "_cleanup_dataset_temp_artifacts", lambda *_args: 0)

    assert cli.run_v3_cli(_args(step="step3", use_gpu=True), raw) == 0
    assert len(fleets) == 1
    assert views == ["alpha", "beta"]
    assert starts == []
    assert fleets[0]._closed is True


def test_cli_cpu_step3_does_not_construct_worker_fleet(monkeypatch, tmp_path):
    raw = _scoreq_only(_config(tmp_path, step="step3"), tmp_path)

    monkeypatch.setattr(
        cli, "Step3MosWorkerFleet",
        lambda *_args, **_kwargs: pytest.fail("CPU Step3 must not construct a fleet"),
    )
    monkeypatch.setattr(
        cli, "V3PipelineRunner",
        lambda *_args, **_kwargs: SimpleNamespace(
            run=lambda *_run_args: SimpleNamespace(target=None),
        ),
    )
    monkeypatch.setattr(cli, "_cleanup_dataset_temp_artifacts", lambda *_args: 0)

    assert cli.run_v3_cli(_args(step="step3", use_gpu=False), raw) == 0


def test_cli_step3_failure_stops_later_dataset_and_finally_closes_fleet(
    monkeypatch, tmp_path,
):
    raw = _scoreq_only(
        _config(tmp_path, step="step3", step3_gpu_ids=[7]), tmp_path,
    )
    raw["datasets"]["beta"] = dict(raw["datasets"]["alpha"])
    raw["datasets"]["beta"]["input_root"] = str(tmp_path / "input-beta")
    business_error = ValueError("Step3 dataset failed")
    cleanup_error = RuntimeError("fleet close failed")
    runs = []
    fleets = []

    class FailingFleet:
        def __init__(self, *_args, **_kwargs):
            self.broken = None
            self.close_calls = 0
            fleets.append(self)

        def for_dataset(self, dataset_id):
            fleet = self

            class Scorer:
                @staticmethod
                def provenance_payload():
                    return {"scorer_execution": "test-fleet"}

                def score_task_groups(self, *_args, **_kwargs):
                    fleet.broken = business_error
                    raise business_error

            return Scorer()

        def close(self):
            self.close_calls += 1
            raise cleanup_error

    class FailingRunner:
        def __init__(self, config, factories):
            self._config = config
            self._factories = factories

        def run(self, dataset_id, requested_step):
            runs.append(dataset_id)
            stage = self._factories[PipelineStep.STEP3](self._config, dataset_id)
            list(stage._scorer.score_task_groups(()))
            pytest.fail("Step3 failure must escape the runner")

    monkeypatch.setattr(cli, "Step3MosWorkerFleet", FailingFleet)
    monkeypatch.setattr(cli, "V3PipelineRunner", FailingRunner)
    monkeypatch.setattr(cli, "_cleanup_dataset_temp_artifacts", lambda *_args: 0)

    with pytest.raises(ValueError, match="Step3 dataset failed") as raised:
        cli.run_v3_cli(_args(step="step3", use_gpu=True), raw)

    assert raised.value is business_error
    assert runs == ["alpha"]
    assert fleets[0].broken is business_error
    assert fleets[0].close_calls == 1
    assert any(
        "fleet close failed" in note
        for note in getattr(business_error, "__notes__", ())
    )


def test_cli_propagates_fleet_close_failure_on_normal_path(monkeypatch, tmp_path):
    raw = _scoreq_only(
        _config(tmp_path, step="step3", step3_gpu_ids=[7]), tmp_path,
    )

    class CloseFailingFleet:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            raise RuntimeError("fleet close failed")

    monkeypatch.setattr(cli, "Step3MosWorkerFleet", CloseFailingFleet)
    monkeypatch.setattr(
        cli, "V3PipelineRunner",
        lambda *_args, **_kwargs: SimpleNamespace(
            run=lambda *_run_args: SimpleNamespace(target=None),
        ),
    )
    monkeypatch.setattr(cli, "_cleanup_dataset_temp_artifacts", lambda *_args: 0)

    with pytest.raises(RuntimeError, match="fleet close failed"):
        cli.run_v3_cli(_args(step="step3", use_gpu=True), raw)


def test_mos_preflight_failure_constructs_no_factory_or_runner(
    monkeypatch, tmp_path,
):
    raw = _scoreq_only(_config(tmp_path, step="step3"), tmp_path)
    config = cli._build_config(_args(step="step3"), raw, PipelineStep.STEP3)
    plan = object()
    monkeypatch.setattr(config, "validate", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "_build_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        cli, "MosExecutionPlan",
        SimpleNamespace(from_config=lambda *_args, **_kwargs: plan),
    )
    monkeypatch.setattr(
        cli, "verify_mos_assets",
        lambda got: (_ for _ in ()).throw(ValueError("preflight failed"))
        if got is plan else pytest.fail("wrong plan"),
    )
    monkeypatch.setattr(
        cli, "_factories",
        lambda *_args, **_kwargs: pytest.fail("factory must not be constructed"),
    )
    monkeypatch.setattr(
        cli, "V3PipelineRunner",
        lambda *_args, **_kwargs: pytest.fail("runner must not be constructed"),
    )

    with pytest.raises(ValueError, match="preflight failed"):
        cli.run_v3_cli(_args(step="step3"), raw)


@pytest.mark.parametrize(
    ("missing_plan", "missing_verified"),
    [(True, True), (True, False), (False, True)],
    ids=["both", "plan", "verified"],
)
def test_step3_factories_fail_immediately_without_complete_verified_plan(
    monkeypatch, tmp_path, missing_plan, missing_verified,
):
    config = _mos_config(tmp_path, step="step3")
    plan, verified = _mos_plan_and_verified(config)
    monkeypatch.setattr(
        cli,
        "Step3MosRuntime",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid Step3 capability must fail before runtime construction"
        ),
    )

    with pytest.raises(TypeError, match="Step3|plan|verified"):
        cli._factories(
            _args(step="step3"),
            {},
            config,
            PipelineStep.STEP3,
            None if missing_plan else plan,
            None if missing_verified else verified,
        )


def test_verified_plan_includes_route_only_local_model(tmp_path):
    model_dir = tmp_path / "utmos-model"
    model_dir.mkdir()
    model_path = model_dir / "utmos.pt"
    model_path.write_bytes(b"route-only-utmos")
    scoreq_dir = tmp_path / "scoreq"
    scoreq_dir.mkdir()
    raw = _config(
        tmp_path,
        step="step3",
        scoreq_model_dir=str(scoreq_dir),
        utmos_model_path=str(model_path.resolve()),
        utmos_model_version="UTMOS-test",
        utmos_model_sha256=local_model_tree_sha256(str(model_dir.resolve())),
    )
    for profile in raw["selection_policies"]["quality_profiles"].values():
        profile["thresholds"]["utmos"] = 3.5
    raw["selection_policies"]["global"] = {
        "mos_models": ["scoreq"], "min_mos_models": 1,
    }
    raw["selection_policies"]["sources"] = {"alpha": {
        "speaker/route.wav": {"mos_models": ["utmos"], "min_mos_models": 1},
    }}
    config = cli._build_config(_args(step="step3"), raw, PipelineStep.STEP3)

    _plan, verified = _mos_plan_and_verified(config)
    payload = verified.as_mapping()

    assert config.selected_mos_models() == ("scoreq", "utmos")
    assert payload["selected_models"] == ["scoreq", "utmos"]
    assert payload["options_by_model"]["utmos"]["model_path"] == str(
        model_path.resolve()
    )
    assert payload["options_by_model"]["utmos"]["checkpoint_sha256"] == (
        hashlib.sha256(b"route-only-utmos").hexdigest()
    )
    assert "nisqa" not in payload["options_by_model"]


def _five_model_plan_and_golden(tmp_path, *, use_gpu):
    selected = ("scoreq", "urgent_pk", "dnsmos", "nisqa", "utmos")
    model_bytes = {
        "scoreq": b"scoreq-five-model",
        "dnsmos": b"dnsmos-five-model",
        "nisqa": b"nisqa-five-model",
        "utmos": b"utmos-five-model",
    }
    scoreq_dir = tmp_path / "scoreq"
    dnsmos_dir = tmp_path / "dnsmos"
    nisqa_dir = tmp_path / "nisqa"
    utmos_dir = tmp_path / "utmos"
    for model, directory in (
        ("scoreq", scoreq_dir),
        ("dnsmos", dnsmos_dir),
        ("nisqa", nisqa_dir),
        ("utmos", utmos_dir),
    ):
        directory.mkdir()
        (directory / f"{model}.bin").write_bytes(model_bytes[model])
    nisqa_path = nisqa_dir / "nisqa.bin"
    utmos_path = utmos_dir / "utmos.bin"
    urgent_path = tmp_path / "urgent.ckpt"
    urgent_bytes = b"urgent-five-model"
    urgent_path.write_bytes(urgent_bytes)
    urgent_source = tmp_path / "urgent-source"
    urgent_source.mkdir()
    urgent_commit = "8" * 40
    urgent_sha256 = hashlib.sha256(urgent_bytes).hexdigest()
    tree_digests = {
        model: local_model_tree_sha256(str(directory.resolve()))
        for model, directory in (
            ("scoreq", scoreq_dir),
            ("dnsmos", dnsmos_dir),
            ("nisqa", nisqa_dir),
            ("utmos", utmos_dir),
        )
    }
    raw = _config(
        tmp_path,
        step="step3",
        step3_gpu_ids=[3, 7] if use_gpu else [],
        scoreq_model_dir=str(scoreq_dir.resolve()),
        dnsmos_model_dir=str(dnsmos_dir.resolve()),
        urgent_pk_checkpoint=str(urgent_path.resolve()),
        urgent_pk_source_dir=str(urgent_source.resolve()),
        urgent_pk_encoder="mel",
        urgent_pk_backbone="resnet34",
        urgent_pk_score_range="mos_1_to_5",
        urgent_pk_expected_sha256=urgent_sha256,
        urgent_pk_expected_commit=urgent_commit,
        nisqa_model_path=str(nisqa_path.resolve()),
        nisqa_model_version="NISQA-five-model",
        nisqa_model_sha256=tree_digests["nisqa"],
        utmos_model_path=str(utmos_path.resolve()),
        utmos_model_version="UTMOS-five-model",
        utmos_model_sha256=tree_digests["utmos"],
    )
    raw["selection_policies"]["global"] = {
        "mos_models": list(selected), "min_mos_models": 3,
    }
    for profile in raw["selection_policies"]["quality_profiles"].values():
        profile["thresholds"].update({"nisqa": 3.5, "utmos": 3.5})
    config = cli._build_config(_args(step="step3"), raw, PipelineStep.STEP3)
    plan = MosExecutionPlan.from_config(config, use_gpu=use_gpu)
    urgent_identity = {
        "checkpoint_sha256": urgent_sha256,
        "source_commit": urgent_commit,
        "source_package_sha256": hashlib.sha256(
            b"urgent-five-model-source-package",
        ).hexdigest(),
    }
    verified = verify_mos_assets(
        plan, urgent_verifier=lambda *_args: dict(urgent_identity),
    )
    checkpoint_identities = {
        model: {
            "checkpoint_sha256": hashlib.sha256(model_bytes[model]).hexdigest(),
            "checkpoint_size_bytes": len(model_bytes[model]),
        }
        for model in ("nisqa", "utmos")
    }
    options = {
        "scoreq": {
            "model_dir": str(scoreq_dir.resolve()),
            "verified_model_tree_sha256": tree_digests["scoreq"],
        },
        "urgent_pk": {
            "checkpoint": str(urgent_path.resolve()),
            "source_dir": str(urgent_source.resolve()),
            "encoder": "mel",
            "backbone": "resnet34",
            "score_range": "mos_1_to_5",
            "expected_sha256": urgent_sha256,
            "expected_commit": urgent_commit,
        },
        "dnsmos": {
            "model_dir": str(dnsmos_dir.resolve()),
            "verified_model_tree_sha256": tree_digests["dnsmos"],
        },
        "nisqa": {
            "model_path": str(nisqa_path.resolve()),
            "model_version": "NISQA-five-model",
            "model_sha256": tree_digests["nisqa"],
            **checkpoint_identities["nisqa"],
        },
        "utmos": {
            "model_path": str(utmos_path.resolve()),
            "model_version": "UTMOS-five-model",
            "model_sha256": tree_digests["utmos"],
            **checkpoint_identities["utmos"],
        },
    }
    provenance = {
        "selected_models": list(selected),
        "models": {
            "scoreq": {
                "model_dir": str(scoreq_dir.resolve()),
                "verified_model_tree_sha256": tree_digests["scoreq"],
            },
            "urgent_pk": urgent_identity,
            "dnsmos": {
                "model_dir": str(dnsmos_dir.resolve()),
                "verified_model_tree_sha256": tree_digests["dnsmos"],
            },
            "nisqa": {
                "model_path": str(nisqa_path.resolve()),
                "model_version": "NISQA-five-model",
                "expected_model_tree_sha256": tree_digests["nisqa"],
                "verified_model_tree_sha256": tree_digests["nisqa"],
                **checkpoint_identities["nisqa"],
            },
            "utmos": {
                "model_path": str(utmos_path.resolve()),
                "model_version": "UTMOS-five-model",
                "expected_model_tree_sha256": tree_digests["utmos"],
                "verified_model_tree_sha256": tree_digests["utmos"],
                **checkpoint_identities["utmos"],
            },
        },
    }
    golden = {
        "selected_models": list(selected),
        "options_by_model": options,
        "provenance": provenance,
    }
    return raw, config, plan, verified, golden, urgent_identity


@pytest.mark.parametrize("use_gpu", [False, True], ids=["cpu", "gpu"])
def test_step3_five_model_verified_plan_matches_current_execution_golden(
    monkeypatch, tmp_path, use_gpu,
):
    raw, config, plan, verified, golden, urgent_identity = (
        _five_model_plan_and_golden(tmp_path, use_gpu=use_gpu)
    )
    assert artifacts._canonical_json_bytes(verified.as_mapping()) == (
        artifacts._canonical_json_bytes(golden)
    )

    def direct_provenance(selected_models, options_by_model, **_kwargs):
        return mos_assets.selected_mos_asset_provenance(
            selected_models,
            options_by_model,
            tree_digest=local_model_tree_sha256,
            urgent_verifier=lambda *_args: dict(urgent_identity),
        )

    options = golden["options_by_model"]
    if use_gpu:
        monkeypatch.setattr(mos_pool, "selected_mos_asset_provenance", direct_provenance)
        direct_scorer = mos_pool.Step3MosScorerPool(
            (3, 7),
            scoreq_options=options["scoreq"],
            urgent_options=options["urgent_pk"],
            dnsmos_options=options["dnsmos"],
            nisqa_options=options["nisqa"],
            utmos_options=options["utmos"],
            dataset_id="alpha",
        )
    else:
        monkeypatch.setattr(
            mos_runtime, "selected_mos_asset_provenance", direct_provenance,
        )
        direct_scorer = mos_runtime.Step3MosRuntime(
            options=mos_runtime.MosRuntimeOptions(
                scoreq_options=options["scoreq"],
                urgent_options=options["urgent_pk"],
                dnsmos_options=options["dnsmos"],
                nisqa_options=options["nisqa"],
                utmos_options=options["utmos"],
            ),
        )
    direct_stage = cli.Step3MosSelectionStage(
        scorer=direct_scorer, dataset_id="alpha",
    )
    direct_payload = direct_stage.config_payload(config)

    factories = cli._factories(
        _args(step="step3", use_gpu=not use_gpu),
        raw,
        config,
        PipelineStep.STEP3,
        plan,
        verified,
    )
    production_stage = factories[PipelineStep.STEP3](config, "alpha")
    production_payload = production_stage.config_payload(config)
    execution = production_payload["scorer_execution"]

    assert production_stage._dataset_id == "alpha"
    assert production_payload["policy"] == direct_payload["policy"]
    assert production_stage.config_digest(config) == direct_stage.config_digest(config)
    assert execution["assets"] == golden["provenance"]
    assert execution["asset_load_binding"] == dict(MOS_ASSET_LOAD_BINDING)
    if use_gpu:
        assert execution["scorer_execution"] == "step3_mos_pool.v6"
        assert execution["protocol"] == "audio_filter_pipeline.step3_mos_worker.v6"
        legacy_execution = {
            key: value
            for key, value in execution.items()
            if key not in {"protocol", "asset_load_binding", "assets"}
        }
    else:
        assert execution["scorer_execution"] == "single_process"
        legacy_execution = {
            "scorer_execution": "single_process",
            "cpu_fallback": False,
            "runtime": {"requested_gpu": False},
        }

    class LegacyScorer:
        @staticmethod
        def provenance_payload():
            return legacy_execution

        @staticmethod
        def score_task_groups(*_args, **_kwargs):
            return ()

    legacy_stage = cli.Step3MosSelectionStage(scorer=LegacyScorer())
    assert production_stage.config_digest(config) != legacy_stage.config_digest(config)


@pytest.mark.parametrize("step", ["step3", "step4", "step5", "step6"])
def test_v3_fails_closed_when_urgent_assets_are_missing(tmp_path, step):
    config = _config(tmp_path, step=step)
    scoreq_dir = tmp_path / "scoreq"
    dnsmos_dir = tmp_path / "dnsmos"
    scoreq_dir.mkdir()
    dnsmos_dir.mkdir()
    config["scoreq_model_dir"] = str(scoreq_dir)
    config["dnsmos_model_dir"] = str(dnsmos_dir)

    with pytest.raises((TypeError, ValueError), match="urgent_pk"):
        cli.run_v3_cli(_args(step=step), config)


@pytest.mark.parametrize("model_version", [
    "voxblink2_samresnet34",
    "voxblink2_samresnet100",
])
def test_cli_builds_wespeaker_runtime_with_actual_worker_digest_when_enabled(
    monkeypatch, tmp_path, model_version,
):
    policy = {
        "enabled": True, "runtime": "wespeaker", "model_dir": "/opt/wespeaker",
        "model_version": model_version, "model_sha256": "b" * 64, "runtime_version": "1.0", "target_sample_rate": 16000,
        "enrollment_bundle_path": "/bundles/alpha.jsonl", "enrollment_bundle_sha256": "c" * 64, "enrollment_exclusion_path": "/bundles/alpha-exclusions.jsonl", "enrollment_exclusion_sha256": "d" * 64,
    }
    options = []

    class Runtime:
        provenance = {"capability": "test-wespeaker-runtime"}

        def __init__(self, value):
            options.append(value)

    monkeypatch.setattr(cli, "WeSpeakerEmbeddingRuntime", Runtime)
    config = _mos_config(
        tmp_path, step="step5", speaker_consistency=policy,
    )
    raw = {"speaker_consistency": policy}
    factories = _verified_factories(_args(), raw, config, PipelineStep.STEP5)
    stage = factories[PipelineStep.STEP5](SimpleNamespace(), "alpha")

    assert isinstance(stage, cli.Step5SpeakerShadowStage)
    assert len(options) == 1
    assert options[0].model_version == policy["model_version"]
    assert options[0].model_sha256 == policy["model_sha256"]
    assert options[0].worker_script_sha256 == cli._worker_sha256(options[0].worker_script)
    assert options[0].python_executable == os.path.realpath(sys.executable)


def test_cli_passes_single_wespeaker_gpu_and_worker_limits_to_runtime(
    monkeypatch, tmp_path,
):
    policy = {
        "enabled": True, "runtime": "wespeaker", "model_dir": "/opt/wespeaker",
        "model_version": "model-v1", "model_sha256": "b" * 64,
        "runtime_version": "1.0", "target_sample_rate": 16000,
        "enrollment_bundle_path": "/bundles/alpha.jsonl",
        "enrollment_bundle_sha256": "c" * 64,
        "enrollment_exclusion_path": "/bundles/alpha-exclusions.jsonl",
        "enrollment_exclusion_sha256": "d" * 64,
        "gpu_ids": [7], "worker_timeout_seconds": 12.5,
        "worker_stderr_tail_bytes": 123,
        "worker_limits": {
            "timeout_ms": 12_500,
            "max_stderr_bytes": 123,
            "rlimit_nofile": 128,
            "max_request_bytes": 2_048,
            "max_json_depth": 16,
        },
    }
    captured = []

    class Runtime:
        provenance = {"capability": "test-wespeaker-runtime"}

        def __init__(self, options):
            captured.append(options)

    monkeypatch.setattr(cli, "WeSpeakerEmbeddingRuntime", Runtime)
    config = _mos_config(
        tmp_path, step="step5", speaker_consistency=policy,
    )
    factories = _verified_factories(
        _args(), {"speaker_consistency": policy}, config, PipelineStep.STEP5,
    )
    factories[PipelineStep.STEP5](SimpleNamespace(), "alpha")

    assert len(captured) == 1
    assert captured[0].cuda_visible_devices == "7"
    assert captured[0].worker_limits.timeout_ms == 12_500
    assert captured[0].worker_limits.max_stderr_bytes == 123
    assert captured[0].worker_limits.rlimit_nofile == 128
    assert captured[0].worker_limits.max_request_bytes == 2_048
    assert captured[0].worker_limits.max_json_depth == 16


def test_cli_builds_group_child_pool_for_multiple_speaker_gpu_ids(monkeypatch, tmp_path):
    policy = {
        "enabled": True, "runtime": "wespeaker", "model_dir": "/opt/wespeaker", "model_version": "model-v1",
        "model_sha256": "b" * 64, "runtime_version": "1.0", "target_sample_rate": 16000,
        "enrollment_bundle_path": "/bundles/alpha.jsonl", "enrollment_bundle_sha256": "c" * 64, "enrollment_exclusion_path": "/bundles/alpha-exclusions.jsonl", "enrollment_exclusion_sha256": "d" * 64, "gpu_ids": [2, 5],
    }
    captured = {}

    class Pool:
        provenance = {"capability": "test-wespeaker-pool"}

        def __init__(self, options, gpu_ids, **kwargs):
            captured.update(options=options, gpu_ids=gpu_ids, **kwargs)

    monkeypatch.setattr(cli, "SpeakerGroupExtractorPool", Pool)
    config = _mos_config(
        tmp_path, step="step5", speaker_consistency=policy,
    )
    raw = {"speaker_consistency": policy}
    factories = _verified_factories(_args(), raw, config, PipelineStep.STEP5)
    factories[PipelineStep.STEP5](SimpleNamespace(), "alpha")

    assert captured["gpu_ids"] == [2, 5]
    assert captured["options"]["cuda_visible_devices"] is None
    assert captured["worker_timeout_seconds"] == 900.0
    assert captured["worker_stderr_tail_bytes"] == 4_096
    assert captured["options"]["worker_limits"]["timeout_ms"] == 900_000
    assert captured["options"]["worker_limits"]["max_stderr_bytes"] == 4_096
    assert captured["options"]["asset_limits"]["max_entries"] == 100_000


@pytest.mark.parametrize("concurrent", [False, True], ids=["sequential", "concurrent"])
def test_step5_factory_isolates_pool_and_process_registry_per_dataset(
    tmp_path, concurrent,
):
    policy = {
        "enabled": True, "runtime": "wespeaker", "model_dir": "/opt/wespeaker",
        "model_version": "model-v1", "model_sha256": "b" * 64,
        "runtime_version": "1.0", "target_sample_rate": 16000,
        "enrollment_bundle_path": "/bundles/alpha.jsonl",
        "enrollment_bundle_sha256": "c" * 64,
        "enrollment_exclusion_path": "/bundles/alpha-exclusions.jsonl",
        "enrollment_exclusion_sha256": "d" * 64, "gpu_ids": [2, 5],
    }
    config = _mos_config(
        tmp_path, step="step5", speaker_consistency=policy,
    )
    factories = _verified_factories(
        _args(), {"speaker_consistency": policy}, config, PipelineStep.STEP5,
    )
    factory = factories[PipelineStep.STEP5]
    dataset_ids = ("alpha", "beta")
    if concurrent:
        with ThreadPoolExecutor(max_workers=2) as executor:
            stages = tuple(executor.map(
                lambda dataset_id: factory(SimpleNamespace(), dataset_id),
                dataset_ids,
            ))
    else:
        stages = tuple(
            factory(SimpleNamespace(), dataset_id) for dataset_id in dataset_ids
        )

    extractors = tuple(stage._embedding_extractor for stage in stages)
    assert all(isinstance(pool, cli.SpeakerGroupExtractorPool) for pool in extractors)
    assert extractors[0] is not extractors[1]
    assert extractors[0]._processes is not extractors[1]._processes
    extractors[0]._processes.add(object())
    assert not extractors[1]._processes
    assert extractors[0]._processes_lock is not extractors[1]._processes_lock
    assert extractors[0]._options is not extractors[1]._options
    assert extractors[0].provenance == extractors[1].provenance
    assert stages[0].config_payload(config)["speaker_shadow"]["embedding_runtime"] == (
        stages[1].config_payload(config)["speaker_shadow"]["embedding_runtime"]
    )


def test_step6_dry_run_publishes_selection_without_constructing_audio_exporter(monkeypatch, tmp_path):
    config = _scoreq_only(
        _config(tmp_path, step="step6", step6_mode="dry_run"), tmp_path,
    )
    calls = []

    class FakeRunner:
        def __init__(self, _config, factories):
            assert set(factories) == set(PipelineStep)

        def run(self, dataset_id, requested_step):
            calls.append((dataset_id, requested_step))
            return SimpleNamespace(target="published-step6-index")

    monkeypatch.setattr(cli, "V3PipelineRunner", FakeRunner)
    monkeypatch.setattr(
        cli, "Step6AudioExporter",
        lambda *_args: pytest.fail("dry_run must not construct audio exporter"),
    )
    monkeypatch.setattr(
        cli, "iter_artifact_rows",
        lambda target: ({
            "sample_rate": 16_000, "start_sample": 0, "end_sample": 16_000,
            "final_selection": {
                "version": 3, "export_eligible": True, "reason": "mos_selected",
            },
        },) if target == "published-step6-index" else pytest.fail("unpublished target"),
    )

    assert cli.run_v3_cli(_args(step="step6"), config) == 0
    assert calls == [("alpha", "step6")]


def test_step6_filter_exports_only_after_runner_returns_published_target(monkeypatch, tmp_path):
    model_dir = tmp_path / "wespeaker"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"wespeaker")
    bundle = tmp_path / "enrollment.jsonl"
    bundle.write_bytes(b'{"bundle":"verified"}\n')
    exclusion = tmp_path / "exclusion.jsonl"
    exclusion.write_bytes(b'{"exclusion":"verified"}\n')
    policy = {
        "enabled": True, "runtime": "wespeaker",
        "model_dir": str(model_dir), "model_version": "model-v1",
        "model_sha256": local_model_tree_sha256(str(model_dir)), "runtime_version": "1.0",
        "target_sample_rate": 16000, "enrollment_bundle_path": str(bundle),
        "enrollment_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "enrollment_exclusion_path": str(exclusion),
        "enrollment_exclusion_sha256": hashlib.sha256(exclusion.read_bytes()).hexdigest(),
        "min_cosine_similarity": 0.8,
    }
    config = _scoreq_only(_config(
        tmp_path,
        step="step6",
        speaker_consistency=policy,
        step6_mode="filter",
    ), tmp_path)
    calls = []

    class FakeRunner:
        def __init__(self, _config, factories):
            assert set(factories) == set(PipelineStep)

        def run(self, dataset_id, requested_step):
            calls.append((dataset_id, requested_step))
            return SimpleNamespace(target="published-step6-index")

    class FakeExporter:
        def __init__(
            self, output_root, *, export_root=None, workers, max_pending,
            max_inflight_bytes,
        ):
            assert output_root == config["output_root"]
            assert export_root is None
            assert workers == 32
            assert max_pending == 64
            assert max_inflight_bytes == 4 * 1024 * 1024 * 1024

        def export_artifact(self, index):
            assert index == "published-step6-index"
            calls.append("exported")
            return ()

    monkeypatch.setattr(cli, "V3PipelineRunner", FakeRunner)
    monkeypatch.setattr(cli, "Step6AudioExporter", FakeExporter)

    assert cli.run_v3_cli(_args(step="step6"), config) == 0
    assert calls == [("alpha", "step6"), "exported"]


def test_step6_dry_run_statistics_aggregate_selection_reasons_and_durations():
    rows = [
        {
            "sample_rate": 16_000, "start_sample": 0, "end_sample": 16_000,
            "final_selection": {
                "version": 3, "export_eligible": True, "reason": "speaker_consistent",
            },
        },
        {
            "sample_rate": 8_000, "start_sample": 0, "end_sample": 4_000,
            "final_selection": {
                "version": 3, "export_eligible": False, "reason": "speaker_inconsistent",
            },
        },
    ]

    assert cli._step6_selection_statistics(rows) == {
        "total_segments": 2,
        "eligible_segments": 1,
        "rejected_segments": 1,
        "eligible_duration_sec": 1.0,
        "rejected_duration_sec": 0.5,
        "reasons": {"speaker_consistent": 1, "speaker_inconsistent": 1},
    }


def test_step6_statistics_rejects_stale_final_selection_version():
    with pytest.raises(ValueError, match="exactly 3"):
        cli._step6_selection_statistics([{
            "sample_rate": 16_000,
            "start_sample": 0,
            "end_sample": 16_000,
            "final_selection": {
                "version": 2, "export_eligible": True, "reason": "mos_selected",
            },
        }])


def _decision_row(profile="quality_high", *, decision="keep", passed_models=3, duration_sec=1.0):
    sample_rate = 16_000
    return {
        "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
        "sample_rate": sample_rate,
        "start_sample": 0,
        "end_sample": int(duration_sec * sample_rate),
        "quality_profile": profile,
        "mos_selection": {
            "decision": decision,
            "passed_models": passed_models,
            "required_models": 3,
            "policy_id": "alpha:en",
            "policy_hash": "h" * 64,
        },
    }


def test_step4_dry_run_statistics_aggregate_decisions_and_profiles():
    rows = [
        _decision_row("quality_high", decision="keep", passed_models=3, duration_sec=1.0),
        _decision_row("quality_high", decision="reject", passed_models=1, duration_sec=0.5),
        _decision_row("quality_premium", decision="keep", passed_models=2, duration_sec=2.0),
    ]

    assert cli._step4_dry_run_statistics(rows) == {
        "total_segments": 3,
        "keep_segments": 2,
        "reject_segments": 1,
        "keep_duration_sec": 3.0,
        "reject_duration_sec": 0.5,
        "by_quality_profile": {
            "quality_high": {"total": 2, "keep": 1, "reject": 1},
            "quality_premium": {"total": 1, "keep": 1, "reject": 0},
        },
        "passed_models_histogram": {0: 0, 1: 1, 2: 1, 3: 1},
        "policy_hash": "h" * 64,
        "policy_hashes": ["h" * 64],
    }


def test_step4_dry_run_statistics_uses_selected_n_not_required_k():
    row = _decision_row(decision="keep", passed_models=2)
    row["mos_models"] = ["utmos", "nisqa"]
    row["mos_selection"]["required_models"] = 1

    statistics = cli._step4_dry_run_statistics([row])

    assert statistics["keep_segments"] == 1
    assert statistics["passed_models_histogram"] == {0: 0, 1: 0, 2: 1}


def test_step4_dry_run_statistics_reports_multiple_policy_hashes():
    rows = [
        _decision_row(),
        {**_decision_row(), "mos_selection": {
            "decision": "keep", "passed_models": 3, "required_models": 3,
            "policy_id": "alpha:en", "policy_hash": "0" * 64,
        }},
    ]

    statistics = cli._step4_dry_run_statistics(rows)

    assert statistics["policy_hash"] is None
    assert statistics["policy_hashes"] == ["0" * 64, "h" * 64]


def test_step4_dry_run_consumes_step3_rows_without_publishing(monkeypatch, tmp_path):
    config = _scoreq_only(
        _config(tmp_path, step="step4", step4_mode="dry_run"), tmp_path,
    )
    calls = []
    decision_rows = (_decision_row(),)

    class FakeRunner:
        def __init__(self, _config, factories):
            assert set(factories) == set(PipelineStep)

        def run(self, dataset_id, requested_step):
            calls.append((dataset_id, requested_step))
            return {PipelineStep.STEP3: "step3-index"}

    class FakeStep4:
        def __init__(self, *, dataset_id):
            assert dataset_id == "alpha"

        def produce_rows(self, context, upstream_index, upstream_rows):
            calls.append("produce_rows")
            assert upstream_index == "step3-index"
            return decision_rows

    monkeypatch.setattr(cli, "V3PipelineRunner", FakeRunner)
    monkeypatch.setattr(cli, "Step4MosFilterStage", FakeStep4)
    monkeypatch.setattr(
        cli, "iter_artifact_rows",
        lambda index: ["step3-row"] if index == "step3-index" else pytest.fail("unexpected index"),
    )
    logged = []
    monkeypatch.setattr(cli.logger, "info", lambda *args: logged.append(args))

    assert cli.run_v3_cli(_args(step="step4"), config) == 0
    # 只跑到 step3，绝不发布 step4
    assert calls == [("alpha", "step3"), "produce_rows"]
    assert logged and logged[0][0] == (
        "Step4 dry_run dataset=%s total=%d keep=%d reject=%d "
        "keep_duration_sec=%.3f reject_duration_sec=%.3f "
        "by_quality_profile=%s passed_models_histogram=%s policy_hash=%s"
    )


def _drop_config(tmp_path):
    """满足 step6 validate MOS 资产 preflight 的 drop 命令配置。"""
    checkpoint = tmp_path / "urgent.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    source_dir = tmp_path / "urgent-source"
    source_dir.mkdir()
    scoreq_dir = tmp_path / "scoreq"; scoreq_dir.mkdir()
    dnsmos_dir = tmp_path / "dnsmos"; dnsmos_dir.mkdir()
    config = _config(tmp_path, step="step6")
    config["urgent_pk_checkpoint"] = str(checkpoint)
    config["urgent_pk_source_dir"] = str(source_dir)
    config["urgent_pk_encoder"] = "mel"
    config["urgent_pk_backbone"] = "resnet34"
    config["urgent_pk_score_range"] = "mos_1_to_5"
    config["scoreq_model_dir"] = str(scoreq_dir)
    config["dnsmos_model_dir"] = str(dnsmos_dir)
    return config


def test_speaker_placeholder_preflight_fails_before_factories_and_runner(
    monkeypatch, tmp_path,
):
    config = _drop_config(tmp_path)
    config["speaker_consistency"] = {
        "enabled": True,
        "runtime": "wespeaker",
        "model_dir": "/REPLACE_ME/audio_filter_pipeline/wespeaker_model",
        "model_version": "replace-me",
        "model_sha256": "0" * 64,
        "runtime_version": "replace-me",
        "target_sample_rate": 16_000,
        "enrollment_bundle_path": "/REPLACE_ME/audio_filter_pipeline/enrollment.jsonl",
        "enrollment_bundle_sha256": "0" * 64,
        "enrollment_exclusion_path": "/REPLACE_ME/audio_filter_pipeline/exclusion.jsonl",
        "enrollment_exclusion_sha256": "0" * 64,
        "min_cosine_similarity": 0.8,
    }
    monkeypatch.setattr(
        cli, "_factories",
        lambda *_args, **_kwargs: pytest.fail("factories must not be constructed"),
    )
    monkeypatch.setattr(
        cli, "V3PipelineRunner",
        lambda *_args, **_kwargs: pytest.fail("runner must not be constructed"),
    )

    with pytest.raises(ValueError, match="placeholder|REPLACE_ME"):
        cli.run_v3_cli(_args(step="step6"), config)


def test_drop_steps_deletes_artifacts_and_exits_without_factories(monkeypatch, tmp_path):
    config = _drop_config(tmp_path)
    dropped = []

    monkeypatch.setattr(
        cli, "drop_artifact",
        lambda root, step, dataset: dropped.append((root, step, dataset)),
    )

    assert cli.run_v3_cli(_args(step="step6", drop_steps="step4,step6"), config) == 0
    assert dropped == [
        (config["output_root"], PipelineStep.STEP4, "alpha"),
        (config["output_root"], PipelineStep.STEP6, "alpha"),
    ]


def test_drop_steps_rejects_invalid_values(tmp_path):
    config = _drop_config(tmp_path)

    with pytest.raises(ValueError, match="drop_steps"):
        cli.run_v3_cli(_args(step="step6", drop_steps="step9"), config)
    with pytest.raises(ValueError, match="drop_steps"):
        cli.run_v3_cli(_args(step="step6", drop_steps="step4,step4"), config)


def test_drop_steps_skips_execution_validation_without_mos_assets(
    monkeypatch, tmp_path,
):
    """drop 是独立清理命令：MOS 资产缺失时仍必须能删除已发布 artifact。"""
    config = _config(tmp_path, step="step6")
    dropped = []
    monkeypatch.setattr(
        cli, "drop_artifact",
        lambda root, step, dataset: dropped.append((root, step, dataset)),
    )

    assert cli.run_v3_cli(_args(step="step6", drop_steps="step6"), config) == 0
    assert dropped == [(config["output_root"], PipelineStep.STEP6, "alpha")]


def test_step4_filter_exports_keep_rows_after_runner_publishes(monkeypatch, tmp_path):
    config = _scoreq_only(
        _config(tmp_path, step="step4", step4_mode="filter"), tmp_path,
    )
    calls = []

    class FakeRunner:
        def __init__(self, _config, factories):
            assert set(factories) == set(PipelineStep)

        def run(self, dataset_id, requested_step):
            calls.append((dataset_id, requested_step))
            return SimpleNamespace(target="published-step4-index")

    class FakeExporter:
        def __init__(
            self, output_root, *, export_root=None, workers, max_pending,
            max_inflight_bytes,
        ):
            assert output_root == config["output_root"]
            assert export_root is None
            assert workers == 32
            assert max_pending == 64
            assert max_inflight_bytes == 4 * 1024 * 1024 * 1024

        def export_artifact(self, index):
            calls.append("exported")
            assert index == "published-step4-index"
            return ()

    monkeypatch.setattr(cli, "V3PipelineRunner", FakeRunner)
    monkeypatch.setattr(cli, "Step4AudioExporter", FakeExporter)

    assert cli.run_v3_cli(_args(step="step4"), config) == 0
    assert calls == [("alpha", "step4"), "exported"]
