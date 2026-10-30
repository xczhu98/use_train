"""Generation 3 stage execution shell contract tests."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

import audio_filter_pipeline.processors.stages_v3.base as stage_base
from audio_filter_pipeline.config import PipelineConfig
from audio_filter_pipeline.processors.stages_v3.base import (
    EMPTY_LINEAGE_DIGEST,
    ArtifactExpectation,
    ArtifactStage,
    StageRunContext,
)
from audio_filter_pipeline.processors.stages_v3.export import (
    Step6MultirateExportStage,
)
from audio_filter_pipeline.processors.stages_v3.mos_filter import (
    Step4MosFilterStage,
)
from audio_filter_pipeline.processors.stages_v3.speaker_shadow import (
    Step5SpeakerShadowStage,
)
from audio_filter_pipeline.utils.artifacts import (
    ArtifactWriter,
    PipelineStep,
    iter_artifact_rows,
    open_artifact_for_read,
)
from audio_filter_pipeline.utils.path_utils import v3_artifact_dir
from stage_test_helpers import runner_expectation_for_test

pytestmark = pytest.mark.unit


def _dataset(input_root):
    return {
        "input_root": str(input_root),
        "language": "en",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "speaker": {"source": "none", "required": False},
    }


def _quality_profiles():
    """generation-3 封闭 quality_profiles registry（Task 1 精确结构）。"""
    return {
        "quality_balanced": {"thresholds": {
            "dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6},
            "scoreq": 3.6,
            "urgent_pk": 3.5,
        }},
        "quality_high": {"thresholds": {
            "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
            "scoreq": 4.2,
            "urgent_pk": 3.5,
        }},
        "quality_premium": {"thresholds": {
            "dnsmos": {"ovrl": 3.4, "sig": 3.6, "bak": 4.2, "p808": 4.0},
            "scoreq": 4.5,
            "urgent_pk": 3.5,
        }},
    }


def _config(output_root, *, dataset_ids=("alpha",), threshold=3.0, generation=3):
    return PipelineConfig(
        pipeline_generation=generation,
        step="step5" if generation == 3 else "step3",
        output_root=str(output_root),
        # 输出根与输入根必须是互不重叠的 sibling，覆盖 production 的路径边界契约。
        datasets={
            name: _dataset(output_root.parent / f"{output_root.name}-input" / name)
            for name in dataset_ids
        },
        selection_policies=(
            {
                "global": {
                    "mos_models": ["dnsmos"],
                    "thresholds": {"dnsmos": threshold},
                    "min_mos_models": 1,
                },
                "quality_profiles": _quality_profiles(),
            }
            if generation == 3
            else None
        ),
    )


class SourceStage(ArtifactStage):
    step = PipelineStep.STEP0
    upstream_step = None

    def __init__(self, rows=(), observer=None):
        self.rows = tuple(rows)
        self.observer = observer

    def config_payload(self, config):
        return {"source": {"recursive": True}}

    def produce_rows(self, context, upstream_index, upstream_rows):
        if self.observer is not None:
            self.observer(context.dataset_id)
        assert upstream_index is None
        assert list(upstream_rows) == []
        yield from self.rows


class TransformStage(ArtifactStage):
    step = PipelineStep.STEP1
    upstream_step = PipelineStep.STEP0

    def __init__(self, transform=lambda row: row):
        self.transform = transform

    def config_payload(self, config):
        return {"transform": "identity"}

    def produce_rows(self, context, upstream_index, upstream_rows):
        assert upstream_index is not None
        for row in upstream_rows:
            yield self.transform(row)


class Step3DigestStage(SourceStage):
    step = PipelineStep.STEP3

    def config_payload(self, config):
        return {"datasets": config.to_dict()["datasets"]}


class Step5DigestStage(SourceStage):
    step = PipelineStep.STEP5

    def config_payload(self, config):
        return {"selection_policies": config.to_dict()["selection_policies"]}


def _context(config, dataset_id="alpha", upstream=None):
    return StageRunContext(
        output_root=config.output_root,
        dataset_id=dataset_id,
        config=config,
        upstream=upstream,
    )


def _expectation(index):
    return runner_expectation_for_test(index)


def _target(root, step, dataset_id="alpha"):
    return Path(v3_artifact_dir(str(root), step.value, dataset_id))


def test_config_digest_is_canonical_deterministic_and_stage_owned(tmp_path):
    class OrderedStage(SourceStage):
        def __init__(self, payload):
            self.payload = payload

        def config_payload(self, config):
            return self.payload

    config = _config(tmp_path)
    first = OrderedStage({"z": 2, "a": [1, {"x": "中文"}]})
    second = OrderedStage({"a": [1, {"x": "中文"}], "z": 2})

    assert first.config_digest(config) == (
        "3b65e1465958b3f0a34f3dd772ea8120021be0ee92217225196f74211611b608"
    )
    assert second.config_digest(config) == first.config_digest(config)

    low = _config(tmp_path, threshold=3.0)
    high = _config(tmp_path, threshold=4.0)
    assert Step3DigestStage().config_digest(low) == Step3DigestStage().config_digest(high)
    assert Step5DigestStage().config_digest(low) != Step5DigestStage().config_digest(high)


@pytest.mark.parametrize(
    "payload,error",
    [
        (["not", "mapping"], "mapping"),
        ({1: "coerced"}, "string"),
        ({"nested": {1: "coerced"}}, "string"),
        ({"value": float("nan")}, "finite"),
        ({"value": float("inf")}, "finite"),
        ({"value": object()}, "canonical JSON"),
    ],
)
def test_config_digest_rejects_ambiguous_or_noncanonical_payloads(
    tmp_path, payload, error,
):
    class InvalidStage(SourceStage):
        def config_payload(self, config):
            return payload

    with pytest.raises((TypeError, ValueError), match=error):
        InvalidStage().config_digest(_config(tmp_path))


def test_config_digest_rejects_duplicate_mapping_items(tmp_path):
    class DuplicateMapping(dict):
        def items(self):
            return [("same", 1), ("same", 2)]

    class InvalidStage(SourceStage):
        def config_payload(self, config):
            return DuplicateMapping()

    with pytest.raises(ValueError, match="duplicate"):
        InvalidStage().config_digest(_config(tmp_path))


def test_source_stage_uses_empty_lineage_and_empty_upstream(tmp_path):
    config = _config(tmp_path)
    observed = []
    stage = SourceStage([{"id": 1}, {"id": 2}], observer=observed.append)

    index = stage.run(_context(config))

    assert EMPTY_LINEAGE_DIGEST == hashlib.sha256(b"").hexdigest()
    assert index.lineage_digest == EMPTY_LINEAGE_DIGEST
    assert list(iter_artifact_rows(index)) == [{"id": 1}, {"id": 2}]
    assert observed == ["alpha"]


def test_downstream_reads_bound_upstream_and_chains_artifact_digest(tmp_path):
    config = _config(tmp_path)
    upstream = SourceStage([{"id": 1}, {"id": 2}]).run(_context(config))
    stage = TransformStage(lambda row: {**row, "seen": True})

    index = stage.run(_context(config, upstream=_expectation(upstream)))

    assert index.lineage_digest == upstream.artifact_digest
    assert list(iter_artifact_rows(index)) == [
        {"id": 1, "seen": True},
        {"id": 2, "seen": True},
    ]


def test_stage_upstream_validates_then_opens_one_business_reader(tmp_path, monkeypatch):
    import audio_filter_pipeline.utils.artifacts as artifacts

    config = _config(tmp_path)
    upstream = SourceStage([{"id": 1}, {"id": 2}]).run(_context(config))
    scanner_calls = 0
    real_scanner = artifacts._scan_artifact_rows

    def recording_scanner(stream):
        nonlocal scanner_calls
        scanner_calls += 1
        yield from real_scanner(stream)

    monkeypatch.setattr(artifacts, "_scan_artifact_rows", recording_scanner)
    TransformStage().run(_context(config, upstream=_expectation(upstream)))

    assert scanner_calls == 1


@pytest.mark.parametrize(
    "upstream,error",
    [
        (None, "requires.*upstream"),
        (ArtifactExpectation(PipelineStep.STEP2, "1" * 64, "2" * 64), "step"),
    ],
)
def test_downstream_rejects_missing_or_wrong_upstream_expectation(
    tmp_path, upstream, error,
):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match=error):
        TransformStage().run(_context(config, upstream=upstream))


@pytest.mark.parametrize(
    "stage_factory,upstream_step",
    [
        (Step4MosFilterStage, PipelineStep.STEP3),
        (Step5SpeakerShadowStage, PipelineStep.STEP3),
        (Step6MultirateExportStage, PipelineStep.STEP5),
    ],
)
def test_direct_step4_through_step6_reject_old_upstream_before_business_side_effect(
    tmp_path, monkeypatch, stage_factory, upstream_step,
):
    config = _config(tmp_path)
    writer = ArtifactWriter(
        config.output_root,
        upstream_step,
        "alpha",
        "1" * 64,
        "2" * 64,
    )
    writer.append({"legacy": True})
    legacy_upstream = writer.finalize()
    stage = stage_factory()
    business_calls = []

    def business_side_effect(*_args, **_kwargs):
        business_calls.append(True)
        yield {"unexpected": True}

    monkeypatch.setattr(stage, "produce_rows", business_side_effect)
    context = _context(config, upstream=ArtifactExpectation(
        upstream_step,
        legacy_upstream.config_digest,
        legacy_upstream.lineage_digest,
    ))
    with pytest.raises(RuntimeError, match="runner-minted"):
        stage.run(context)

    assert business_calls == []
    assert not _target(tmp_path, stage.step).exists()


@pytest.mark.parametrize(
    "values,error",
    [
        (("step0", "1" * 64, "2" * 64), "PipelineStep"),
        ((PipelineStep.STEP0, "ABC", "2" * 64), "config_digest"),
        ((PipelineStep.STEP0, "1" * 64, "g" * 64), "lineage_digest"),
    ],
)
def test_artifact_expectation_validates_bound_identity(values, error):
    with pytest.raises((TypeError, ValueError), match=error):
        ArtifactExpectation(*values)


def test_runner_minted_expectation_copy_does_not_inherit_capability(tmp_path):
    config = _config(tmp_path)
    upstream = SourceStage([{"id": 1}]).run(_context(config))
    authorized = _expectation(upstream)
    copied = replace(authorized)

    assert copied == authorized
    with pytest.raises(RuntimeError, match="runner-minted"):
        TransformStage().run(_context(config, upstream=copied))

    assert not _target(tmp_path, PipelineStep.STEP1).exists()


def test_source_rejects_upstream_expectation(tmp_path):
    config = _config(tmp_path)
    expectation = ArtifactExpectation(PipelineStep.STEP0, "1" * 64, "2" * 64)
    with pytest.raises(ValueError, match="source.*upstream"):
        SourceStage().run(_context(config, upstream=expectation))


def test_producer_consumes_upstream_lazily_and_preserves_rows(tmp_path):
    config = _config(tmp_path)
    upstream = SourceStage([{"position": 1}, {"position": 2}]).run(_context(config))
    observed = []

    def transform(row):
        observed.append(row["position"])
        if row["position"] == 2:
            assert observed == [1, 2]
        return row

    index = TransformStage(transform).run(
        _context(config, upstream=_expectation(upstream))
    )

    assert observed == [1, 2]
    assert list(iter_artifact_rows(index)) == [{"position": 1}, {"position": 2}]


def test_shell_closes_partially_consumed_upstream_reader(tmp_path, monkeypatch):
    config = _config(tmp_path)
    upstream = SourceStage([{"id": 1}, {"id": 2}]).run(_context(config))
    closed = []
    retained = []
    real_open_reader = stage_base.open_artifact_reader

    def tracked_open_reader(*args, **kwargs):
        reader = real_open_reader(*args, **kwargs)
        real_close = reader.close

        def tracked_close():
            try:
                return real_close()
            finally:
                closed.append(True)

        reader.close = tracked_close
        retained.append(reader)
        return reader

    class FirstOnlyStage(TransformStage):
        def produce_rows(self, context, upstream_index, upstream_rows):
            yield next(iter(upstream_rows))

    monkeypatch.setattr(stage_base, "open_artifact_reader", tracked_open_reader)
    FirstOnlyStage().run(_context(config, upstream=_expectation(upstream)))

    assert retained
    assert retained[0].closed
    assert closed == [True]


def test_resume_reopens_verified_artifact_without_rerunning_producer(tmp_path):
    config = _config(tmp_path)
    observed = []
    stage = SourceStage([{"id": 1}], observer=observed.append)
    first = stage.run(_context(config))

    resumed = stage.run(_context(config))

    assert resumed.artifact_digest == first.artifact_digest
    assert observed == ["alpha"]
    assert list(iter_artifact_rows(resumed)) == [{"id": 1}]


def test_downstream_cache_hit_validates_upstream_without_business_spool(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.utils.artifacts as artifacts

    config = _config(tmp_path)
    upstream = SourceStage([{"id": 1}, {"id": 2}]).run(_context(config))
    stage = TransformStage(lambda row: {**row, "seen": True})
    first = stage.run(_context(config, upstream=_expectation(upstream)))
    producer_calls = []
    rows_hash_calls = 0
    real_rows_hash = artifacts._rows_hash_and_count_fd

    def recording_rows_hash(descriptor):
        nonlocal rows_hash_calls
        rows_hash_calls += 1
        return real_rows_hash(descriptor)

    def fail_if_business_reader_opens(*_args, **_kwargs):
        raise AssertionError("cache hit must not create a business rows reader")

    def fail_if_producer_runs(*_args, **_kwargs):
        producer_calls.append(True)
        raise AssertionError("cache hit must not run producer")

    monkeypatch.setattr(artifacts, "_rows_hash_and_count_fd", recording_rows_hash)
    monkeypatch.setattr(stage_base, "open_artifact_reader", fail_if_business_reader_opens)
    monkeypatch.setattr(stage, "produce_rows", fail_if_producer_runs)

    resumed = stage.run(_context(config, upstream=_expectation(upstream)))

    assert resumed.serialized() == first.serialized()
    assert rows_hash_calls == 3
    assert producer_calls == []


def test_racing_matching_publication_is_reopened_and_verified(tmp_path, monkeypatch):
    config = _config(tmp_path)
    observed = []
    stage = SourceStage([{"id": 1}], observer=observed.append)
    real_writer = stage_base.ArtifactWriter

    def racing_writer(*args, **kwargs):
        competitor = real_writer(*args, **kwargs)
        competitor.append({"id": 1})
        competitor.finalize()
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(stage_base, "ArtifactWriter", racing_writer)

    index = stage.run(_context(config))

    assert list(iter_artifact_rows(index)) == [{"id": 1}]
    assert observed == []


def test_finalize_metadata_must_match_bound_reopen(tmp_path, monkeypatch):
    config = _config(tmp_path)
    real_finalize = stage_base.ArtifactWriter.finalize

    def mismatched_finalize(self):
        published = real_finalize(self)
        return replace(published, artifact_digest="0" * 64)

    monkeypatch.setattr(stage_base.ArtifactWriter, "finalize", mismatched_finalize)

    with pytest.raises(RuntimeError, match="metadata.*mismatch"):
        SourceStage([{"id": 1}]).run(_context(config))


def test_finalize_race_reopens_matching_competitor(tmp_path, monkeypatch):
    config = _config(tmp_path)
    real_finalize = stage_base.ArtifactWriter.finalize

    def racing_finalize(self):
        competitor = stage_base.ArtifactWriter(
            self._output_root,
            self._step,
            self._dataset_id,
            self._config_digest,
            self._lineage_digest,
        )
        competitor.append({"id": 1})
        real_finalize(competitor)
        return real_finalize(self)

    monkeypatch.setattr(stage_base.ArtifactWriter, "finalize", racing_finalize)

    index = SourceStage([{"id": 1}]).run(_context(config))

    assert list(iter_artifact_rows(index)) == [{"id": 1}]


def test_finalize_race_cleanup_failure_is_not_hidden(tmp_path, monkeypatch):
    config = _config(tmp_path)
    real_finalize = stage_base.ArtifactWriter.finalize

    def failing_finalize(self):
        competitor = stage_base.ArtifactWriter(
            self._output_root,
            self._step,
            self._dataset_id,
            self._config_digest,
            self._lineage_digest,
        )
        competitor.append({"id": 1})
        real_finalize(competitor)
        self._state = "cleanup_failed"
        error = FileExistsError("publication race")
        error.__notes__ = ["artifact cleanup failed: OSError('cleanup boom')"]
        raise error

    monkeypatch.setattr(stage_base.ArtifactWriter, "finalize", failing_finalize)

    with pytest.raises(FileExistsError, match="publication race") as caught:
        SourceStage([{"id": 1}]).run(_context(config))

    assert "cleanup boom" in " ".join(caught.value.__notes__)


def test_writer_metadata_never_gains_iterator_capability(tmp_path):
    writer = ArtifactWriter(
        tmp_path,
        PipelineStep.STEP0,
        "alpha",
        "1" * 64,
        EMPTY_LINEAGE_DIGEST,
    )
    metadata = writer.finalize()

    with pytest.raises(ValueError, match="verified.*opener"):
        list(iter_artifact_rows(metadata))


def test_post_publication_finalize_failure_propagates_then_resumes(
    tmp_path, monkeypatch,
):
    import audio_filter_pipeline.utils.artifacts as artifacts

    config = _config(tmp_path)
    target = _target(tmp_path, PipelineStep.STEP0)
    observed = []
    stage = SourceStage([{"id": 1}], observer=observed.append)
    real_fsync = artifacts.os.fsync

    def failing_fsync(descriptor):
        name = os.path.basename(os.readlink(f"/proc/self/fd/{descriptor}"))
        if name == "step0" and target.exists():
            raise OSError("post-publication fsync boom")
        return real_fsync(descriptor)

    monkeypatch.setattr(artifacts.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="post-publication fsync boom"):
        stage.run(_context(config))

    assert target.is_dir()
    monkeypatch.setattr(artifacts.os, "fsync", real_fsync)
    resumed = stage.run(_context(config))
    assert observed == ["alpha"]
    assert list(iter_artifact_rows(resumed)) == [{"id": 1}]


@pytest.mark.parametrize("mismatch", ["config", "lineage"])
def test_resume_fails_closed_on_stale_existing_artifact(
    tmp_path, monkeypatch, mismatch,
):
    config = _config(tmp_path)
    stage = SourceStage([{"replacement": True}])
    config_digest = "4" * 64 if mismatch == "config" else stage.config_digest(config)
    lineage = EMPTY_LINEAGE_DIGEST if mismatch == "config" else "3" * 64
    writer = ArtifactWriter(
        tmp_path,
        PipelineStep.STEP0,
        "alpha",
        config_digest,
        lineage,
    )
    writer.append({"keep": True})
    writer.finalize()
    target = _target(tmp_path, PipelineStep.STEP0)
    before = (target / "rows.jsonl").read_bytes()

    def fail_if_stale_rows_scanned(_descriptor):
        raise AssertionError("stale cache must reject before rows scan")

    monkeypatch.setattr(
        "audio_filter_pipeline.utils.artifacts._rows_hash_and_count_fd",
        fail_if_stale_rows_scanned,
    )
    with pytest.raises(ValueError, match=f"{mismatch} digest"):
        SourceStage([{"replacement": True}]).run(_context(config))

    assert (target / "rows.jsonl").read_bytes() == before


def test_resume_fails_closed_on_partial_existing_target(tmp_path):
    config = _config(tmp_path)
    target = _target(tmp_path, PipelineStep.STEP0)
    target.mkdir(parents=True)
    (target / "partial").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact|contract|missing|unexpected"):
        SourceStage([{"replacement": True}]).run(_context(config))

    assert (target / "partial").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("failure", ["producer", "append", "finalize"])
def test_new_execution_failure_leaves_no_published_target(
    tmp_path, monkeypatch, failure,
):
    config = _config(tmp_path)

    class FailingStage(SourceStage):
        def produce_rows(self, context, upstream_index, upstream_rows):
            if failure == "producer":
                raise RuntimeError("producer boom")
            yield {"id": 1}

    if failure == "append":
        real_append = ArtifactWriter.append

        def fail_append(self, row):
            real_append(self, row)
            raise RuntimeError("append boom")

        monkeypatch.setattr(ArtifactWriter, "append", fail_append)
    elif failure == "finalize":
        real_finalize = ArtifactWriter.finalize

        def fail_finalize(self):
            self.abort()
            raise RuntimeError("finalize boom")

        monkeypatch.setattr(ArtifactWriter, "finalize", fail_finalize)

    with pytest.raises(RuntimeError, match=f"{failure} boom"):
        FailingStage().run(_context(config))

    assert not _target(tmp_path, PipelineStep.STEP0).exists()
    parent = _target(tmp_path, PipelineStep.STEP0).parent
    assert not list(parent.glob(".alpha.stage-*")) if parent.exists() else True


def test_upstream_reader_failure_aborts_output(tmp_path, monkeypatch):
    config = _config(tmp_path)
    upstream = SourceStage([{"id": 1}]).run(_context(config))
    real_open_reader = stage_base.open_artifact_reader

    def failing_open_reader(*args, **kwargs):
        reader = real_open_reader(*args, **kwargs)

        class FailingReader:
            index = reader.index

            def __init__(self):
                self._seen = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._seen:
                    raise RuntimeError("reader boom")
                self._seen += 1
                return next(reader)

            def close(self):
                reader.close()

        return FailingReader()

    monkeypatch.setattr(stage_base, "open_artifact_reader", failing_open_reader)
    with pytest.raises(RuntimeError, match="reader boom"):
        TransformStage().run(_context(config, upstream=_expectation(upstream)))

    assert not _target(tmp_path, PipelineStep.STEP1).exists()


def test_cleanup_failure_does_not_mask_primary_error(tmp_path, monkeypatch):
    config = _config(tmp_path)

    class FailingStage(SourceStage):
        def produce_rows(self, context, upstream_index, upstream_rows):
            raise RuntimeError("primary boom")
            yield

    real_abort = ArtifactWriter.abort
    cleaned_writers = []

    def failing_abort(self):
        # 先走真实 owner 的资源收敛，再注入 cleanup 报错以验证异常仲裁。
        real_abort(self)
        cleaned_writers.append(self)
        raise OSError("cleanup boom")

    monkeypatch.setattr(ArtifactWriter, "abort", failing_abort)
    with pytest.raises(RuntimeError, match="primary boom") as caught:
        FailingStage().run(_context(config))

    assert "cleanup boom" in " ".join(getattr(caught.value, "__notes__", ()))
    assert len(cleaned_writers) == 1
    writer = cleaned_writers[0]
    assert writer.state == "aborted"
    assert writer._rows is None
    assert writer._temp_fd is None
    assert writer._staging_fd is None
    assert writer._parent_fd is None


def test_stage_instance_is_reusable_across_datasets(tmp_path):
    config = _config(tmp_path, dataset_ids=("alpha", "beta"))
    stage = SourceStage([{"same": True}])
    construction_state = dict(vars(stage))

    alpha = stage.run(_context(config, "alpha"))
    beta = stage.run(_context(config, "beta"))

    assert vars(stage) == construction_state
    assert alpha.dataset_id == "alpha"
    assert beta.dataset_id == "beta"
    assert list(iter_artifact_rows(alpha)) == [{"same": True}]
    assert list(iter_artifact_rows(beta)) == [{"same": True}]


@pytest.mark.parametrize(
    "context,error",
    [
        (lambda root: _config(root, generation=2), "pipeline_generation must be 3"),
        (lambda root: _context(_config(root), "ghost"), "unknown dataset"),
        (
            lambda root: StageRunContext(
                output_root=str(root / "other"),
                dataset_id="alpha",
                config=_config(root),
            ),
            "output_root",
        ),
    ],
)
def test_context_rejects_wrong_generation_dataset_or_output_root(
    tmp_path, context, error,
):
    with pytest.raises(ValueError, match=error):
        SourceStage().run(context(tmp_path))


def test_context_validates_at_construction(tmp_path):
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="unknown dataset"):
        StageRunContext(str(tmp_path), "ghost", config)


def test_context_freezes_relative_output_root_before_cwd_change(
    tmp_path, monkeypatch,
):
    construction_cwd = tmp_path / "construction"
    runtime_cwd = tmp_path / "runtime"
    construction_cwd.mkdir()
    runtime_cwd.mkdir()
    monkeypatch.chdir(construction_cwd)
    frozen_path = construction_cwd.resolve()
    config = _config(frozen_path)
    context = _context(config)
    frozen_root = str(frozen_path)

    assert context.output_root == frozen_root

    monkeypatch.chdir(runtime_cwd)
    index = SourceStage([{"id": 1}]).run(context)

    assert list(iter_artifact_rows(index)) == [{"id": 1}]
    assert _target(frozen_root, PipelineStep.STEP0).is_dir()
    assert not _target(runtime_cwd, PipelineStep.STEP0).exists()


def test_upstream_reader_factory_failure_does_not_leak_resources(
    tmp_path, monkeypatch,
):
    config = _config(tmp_path)
    upstream = SourceStage([{"id": 1}]).run(_context(config))
    target = _target(tmp_path, PipelineStep.STEP1)
    primary = RuntimeError("reader factory boom")
    def snapshot_fds():
        snapshot = {}
        for name in os.listdir("/proc/self/fd"):
            try:
                descriptor = int(name)
                snapshot[descriptor] = os.readlink(f"/proc/self/fd/{descriptor}")
            except (FileNotFoundError, OSError, ValueError):
                continue
        return snapshot

    open_fds_before = snapshot_fds()

    def failing_reader_factory(*args, **kwargs):
        raise primary

    monkeypatch.setattr(
        stage_base, "open_artifact_reader", failing_reader_factory
    )
    with pytest.raises(RuntimeError, match="reader factory boom") as caught:
        TransformStage().run(_context(config, upstream=_expectation(upstream)))

    assert caught.value is primary
    assert not target.exists()
    assert not list(target.parent.glob(".alpha.stage-*"))
    open_fds_after = snapshot_fds()
    # 运行时可能在测试期间正常关闭 transient fd；真正的资源泄漏表现为
    # 新增 fd，或复用既有 fd 编号指向了不同对象。
    assert set(open_fds_after.items()) <= set(open_fds_before.items())


@pytest.mark.parametrize(
    "kind,expected_path",
    [("mapping", "payload.self"), ("list", "payload.nested[0]")],
)
def test_config_digest_rejects_cycles_with_payload_path(
    tmp_path, kind, expected_path,
):
    if kind == "mapping":
        payload = {}
        payload["self"] = payload
    else:
        nested = []
        nested.append(nested)
        payload = {"nested": nested}

    class CyclicStage(SourceStage):
        def config_payload(self, config):
            return payload

    with pytest.raises(ValueError) as caught:
        CyclicStage().config_digest(_config(tmp_path))

    assert "cycle" in str(caught.value).lower()
    assert expected_path in str(caught.value)


def test_config_digest_accepts_shared_acyclic_subobjects(tmp_path):
    shared = {"value": [1, 2]}

    class SharedStage(SourceStage):
        def config_payload(self, config):
            return {"left": shared, "right": shared}

    class CopiedStage(SourceStage):
        def config_payload(self, config):
            return {
                "left": {"value": [1, 2]},
                "right": {"value": [1, 2]},
            }

    config = _config(tmp_path)
    assert SharedStage().config_digest(config) == CopiedStage().config_digest(config)
