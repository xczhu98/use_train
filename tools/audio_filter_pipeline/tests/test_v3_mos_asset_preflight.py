"""Generation-3 Step3 MOS 资产 preflight 契约测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from collections.abc import Mapping

import pytest

from audio_filter_pipeline.processors.stages_v3 import mos_assets
from audio_filter_pipeline.processors.stages_v3.mos_asset_preflight import (
    VerifiedMosExecutionPlan,
    verify_mos_assets,
)
from audio_filter_pipeline.utils import artifacts
from audio_filter_pipeline.processors.stages_v3.mos_execution import (
    MosAssetSpec,
    MosExecutionPlan,
)


pytestmark = pytest.mark.unit

PROVENANCE_MODEL_ORDER = ("scoreq", "urgent_pk", "dnsmos", "nisqa", "utmos")
POLICY_MODEL_ORDER = ("utmos", "scoreq", "nisqa", "urgent_pk", "dnsmos")
CPU_TOPOLOGY = {"kind": "single_process", "device": "cpu"}
TREE_DIGESTS = {
    "scoreq": "1" * 64,
    "dnsmos": "2" * 64,
    "nisqa": "3" * 64,
    "utmos": "4" * 64,
}
CHECKPOINT_IDENTITIES = {
    "nisqa": {"checkpoint_sha256": "5" * 64, "checkpoint_size_bytes": 501},
    "utmos": {"checkpoint_sha256": "6" * 64, "checkpoint_size_bytes": 601},
}
URGENT_IDENTITY = {
    "checkpoint_sha256": "7" * 64,
    "source_commit": "89abcdef0123456789abcdef0123456789abcdef",
    "source_package_sha256": "8" * 64,
}


class _FakeFilesystem:
    """只暴露 preflight 所需路径操作，并记录所有资产访问。"""

    def __init__(self, *, directories=(), files=()):
        self.directories = {self.canonical_path(path) for path in directories}
        self.files = {self.canonical_path(path) for path in files}
        self.calls = []

    @staticmethod
    def canonical_path(path):
        return os.path.realpath(os.path.abspath(path))

    def is_directory(self, path):
        canonical = self.canonical_path(path)
        self.calls.append(("is_directory", canonical))
        return canonical in self.directories

    def is_file(self, path):
        canonical = self.canonical_path(path)
        self.calls.append(("is_file", canonical))
        return canonical in self.files


class _FakeAssetDependencies:
    """提供可观察的 digest、checkpoint snapshot 与 URGENT verifier。"""

    def __init__(self, filesystem):
        self.filesystem = filesystem
        self.digest_calls = []
        self.snapshot_calls = []
        self.verifier_calls = []
        self.snapshot_values = {
            model: dict(identity)
            for model, identity in CHECKPOINT_IDENTITIES.items()
        }
        self.urgent_value = dict(URGENT_IDENTITY)

    def tree_digest(self, path):
        canonical = self.filesystem.canonical_path(path)
        self.digest_calls.append(canonical)
        for model in ("scoreq", "dnsmos", "nisqa", "utmos"):
            marker = f"{os.sep}{model}"
            if marker in canonical:
                return TREE_DIGESTS[model]
        raise AssertionError(f"unexpected digest path: {canonical}")

    def checkpoint_snapshot(self, path):
        canonical = self.filesystem.canonical_path(path)
        self.snapshot_calls.append(canonical)
        for model in ("nisqa", "utmos"):
            if f"{os.sep}{model}{os.sep}" in canonical:
                return self.snapshot_values[model]
        raise AssertionError(f"unexpected checkpoint path: {canonical}")

    def urgent_verifier(
        self, checkpoint, source_dir, expected_sha256, expected_commit,
    ):
        self.verifier_calls.append(
            (checkpoint, source_dir, expected_sha256, expected_commit)
        )
        return self.urgent_value


def _asset_specs():
    score_range = {"min": 1, "max": 5}
    return {
        "scoreq": MosAssetSpec(
            "scoreq", {"model_dir": "assets/scoreq/model"},
        ),
        "urgent_pk": MosAssetSpec(
            "urgent_pk",
            {
                "checkpoint": "assets/urgent/model.ckpt",
                "source_dir": "assets/urgent/source",
            },
            {
                "encoder": "wav2vec2",
                "backbone": "cnn",
                "score_range": score_range,
                "expected_sha256": URGENT_IDENTITY["checkpoint_sha256"],
                "expected_commit": URGENT_IDENTITY["source_commit"],
            },
        ),
        "dnsmos": MosAssetSpec(
            "dnsmos", {"model_dir": "assets/dnsmos/model"},
        ),
        "nisqa": MosAssetSpec(
            "nisqa",
            {"model_path": "assets/nisqa/model.tar"},
            {"model_version": "nisqa-v2", "model_sha256": TREE_DIGESTS["nisqa"]},
        ),
        "utmos": MosAssetSpec(
            "utmos",
            {"model_path": "assets/utmos/model.pt"},
            {"model_version": "utmos-v1", "model_sha256": TREE_DIGESTS["utmos"]},
        ),
    }


def _plan(models=POLICY_MODEL_ORDER):
    assets = _asset_specs()
    return MosExecutionPlan(
        tuple(models), tuple(assets[model] for model in models), CPU_TOPOLOGY,
    )


def _full_filesystem():
    return _FakeFilesystem(
        directories=(
            "assets/scoreq/model",
            "assets/urgent/source",
            "assets/dnsmos/model",
        ),
        files=(
            "assets/urgent/model.ckpt",
            "assets/nisqa/model.tar",
            "assets/utmos/model.pt",
        ),
    )


def _verify(plan, filesystem, dependencies):
    return verify_mos_assets(
        plan,
        filesystem=filesystem,
        tree_digest=dependencies.tree_digest,
        checkpoint_snapshot=dependencies.checkpoint_snapshot,
        urgent_verifier=dependencies.urgent_verifier,
    )


def _expected_options(filesystem):
    canonical = filesystem.canonical_path
    return {
        "utmos": {
            "model_path": canonical("assets/utmos/model.pt"),
            "model_version": "utmos-v1",
            "model_sha256": TREE_DIGESTS["utmos"],
            **CHECKPOINT_IDENTITIES["utmos"],
        },
        "scoreq": {
            "model_dir": canonical("assets/scoreq/model"),
            "verified_model_tree_sha256": TREE_DIGESTS["scoreq"],
        },
        "nisqa": {
            "model_path": canonical("assets/nisqa/model.tar"),
            "model_version": "nisqa-v2",
            "model_sha256": TREE_DIGESTS["nisqa"],
            **CHECKPOINT_IDENTITIES["nisqa"],
        },
        "urgent_pk": {
            "checkpoint": canonical("assets/urgent/model.ckpt"),
            "source_dir": canonical("assets/urgent/source"),
            "encoder": "wav2vec2",
            "backbone": "cnn",
            "score_range": {"min": 1, "max": 5},
            "expected_sha256": URGENT_IDENTITY["checkpoint_sha256"],
            "expected_commit": URGENT_IDENTITY["source_commit"],
        },
        "dnsmos": {
            "model_dir": canonical("assets/dnsmos/model"),
            "verified_model_tree_sha256": TREE_DIGESTS["dnsmos"],
        },
    }


def _expected_provenance(filesystem):
    canonical = filesystem.canonical_path
    return {
        "selected_models": list(PROVENANCE_MODEL_ORDER),
        "models": {
            "scoreq": {
                "model_dir": canonical("assets/scoreq/model"),
                "verified_model_tree_sha256": TREE_DIGESTS["scoreq"],
            },
            "urgent_pk": dict(URGENT_IDENTITY),
            "dnsmos": {
                "model_dir": canonical("assets/dnsmos/model"),
                "verified_model_tree_sha256": TREE_DIGESTS["dnsmos"],
            },
            "nisqa": {
                "model_path": canonical("assets/nisqa/model.tar"),
                "model_version": "nisqa-v2",
                "expected_model_tree_sha256": TREE_DIGESTS["nisqa"],
                "verified_model_tree_sha256": TREE_DIGESTS["nisqa"],
                **CHECKPOINT_IDENTITIES["nisqa"],
            },
            "utmos": {
                "model_path": canonical("assets/utmos/model.pt"),
                "model_version": "utmos-v1",
                "expected_model_tree_sha256": TREE_DIGESTS["utmos"],
                "verified_model_tree_sha256": TREE_DIGESTS["utmos"],
                **CHECKPOINT_IDENTITIES["utmos"],
            },
        },
    }


def _all_keys(value):
    keys = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(key)
            keys.update(_all_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            keys.update(_all_keys(nested))
    return keys


def test_five_model_preflight_options_and_provenance_are_exact():
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    plan = _plan()

    verified = _verify(plan, filesystem, dependencies)

    assert plan.selected_models == POLICY_MODEL_ORDER
    assert verified.selected_models == POLICY_MODEL_ORDER
    assert tuple(verified.options_by_model) == POLICY_MODEL_ORDER
    payload = verified.as_mapping()
    assert payload == {
        "selected_models": list(POLICY_MODEL_ORDER),
        "options_by_model": _expected_options(filesystem),
        "provenance": _expected_provenance(filesystem),
    }
    for model in ("scoreq", "dnsmos"):
        digest = payload["options_by_model"][model][
            "verified_model_tree_sha256"
        ]
        assert len(digest) == 64
        assert digest == digest.lower()
        assert set(digest) <= set("0123456789abcdef")
        assert digest == payload["provenance"]["models"][model][
            "verified_model_tree_sha256"
        ]


def test_provenance_order_is_fixed_and_filtered_from_policy_order():
    selected = ("utmos", "urgent_pk", "scoreq")
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)

    verified = _verify(_plan(selected), filesystem, dependencies)

    assert verified.selected_models == selected
    assert tuple(verified.options_by_model) == selected
    assert verified.provenance["selected_models"] == [
        "scoreq", "urgent_pk", "utmos",
    ]
    assert tuple(verified.provenance["models"]) == (
        "scoreq", "urgent_pk", "utmos",
    )


@pytest.mark.parametrize("selected_model", PROVENANCE_MODEL_ORDER)
def test_preflight_accesses_only_the_selected_models_assets(selected_model):
    plan = _plan((selected_model,))
    specs = _asset_specs()[selected_model]
    directories = tuple(
        path for name, path in specs.paths.items()
        if name in {"model_dir", "source_dir"}
    )
    files = tuple(
        path for name, path in specs.paths.items()
        if name in {"model_path", "checkpoint"}
    )
    filesystem = _FakeFilesystem(directories=directories, files=files)
    dependencies = _FakeAssetDependencies(filesystem)

    verified = _verify(plan, filesystem, dependencies)

    assert verified.selected_models == (selected_model,)
    assert tuple(verified.options_by_model) == (selected_model,)
    assert verified.provenance["selected_models"] == [selected_model]
    assert tuple(verified.provenance["models"]) == (selected_model,)
    canonical = filesystem.canonical_path
    if selected_model in {"scoreq", "dnsmos"}:
        model_dir = canonical(specs.paths["model_dir"])
        assert filesystem.calls == [("is_directory", model_dir)]
        assert dependencies.digest_calls == [model_dir]
        assert dependencies.snapshot_calls == []
        assert dependencies.verifier_calls == []
    elif selected_model in {"nisqa", "utmos"}:
        model_path = canonical(specs.paths["model_path"])
        assert filesystem.calls == [("is_file", model_path)]
        assert dependencies.digest_calls == [os.path.dirname(model_path)]
        assert dependencies.snapshot_calls == [model_path]
        assert dependencies.verifier_calls == []
    else:
        # URGENT preflight 只核验 canonical 路径、checkpoint 与 commit 身份；
        # 它不替代真实 loader 对 source tree（包括 symlink）的安全扫描。
        checkpoint = canonical(specs.paths["checkpoint"])
        source_dir = canonical(specs.paths["source_dir"])
        assert filesystem.calls == [
            ("is_file", checkpoint),
            ("is_directory", source_dir),
        ]
        assert dependencies.digest_calls == []
        assert dependencies.snapshot_calls == []
        assert dependencies.verifier_calls == [(
            checkpoint,
            source_dir,
            specs.settings["expected_sha256"],
            specs.settings["expected_commit"],
        )]


def test_five_model_relative_inputs_become_canonical_execution_paths():
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)

    verified = _verify(_plan(), filesystem, dependencies)

    options = verified.options_by_model
    provenance = verified.provenance["models"]
    canonical = filesystem.canonical_path
    for model in ("scoreq", "dnsmos"):
        expected = canonical(f"assets/{model}/model")
        assert options[model]["model_dir"] == expected
        assert provenance[model]["model_dir"] == expected
    for model, relative_path in (
        ("nisqa", "assets/nisqa/model.tar"),
        ("utmos", "assets/utmos/model.pt"),
    ):
        expected = canonical(relative_path)
        assert options[model]["model_path"] == expected
        assert provenance[model]["model_path"] == expected
    assert options["urgent_pk"]["checkpoint"] == canonical(
        "assets/urgent/model.ckpt"
    )
    assert options["urgent_pk"]["source_dir"] == canonical(
        "assets/urgent/source"
    )
    assert dependencies.verifier_calls == [(
        options["urgent_pk"]["checkpoint"],
        options["urgent_pk"]["source_dir"],
        URGENT_IDENTITY["checkpoint_sha256"],
        URGENT_IDENTITY["source_commit"],
    )]


def test_nisqa_and_utmos_checkpoint_are_snapshotted_exactly_once_each():
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)

    _verify(_plan(("utmos", "nisqa")), filesystem, dependencies)

    assert dependencies.snapshot_calls == [
        filesystem.canonical_path("assets/utmos/model.pt"),
        filesystem.canonical_path("assets/nisqa/model.tar"),
    ]


def test_urgent_score_range_is_not_rewritten_by_preflight():
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    plan = _plan(("urgent_pk",))
    expected_range = {"min": 1, "max": 5}

    verified = _verify(plan, filesystem, dependencies)

    assert verified.options_by_model["urgent_pk"]["score_range"] == expected_range
    assert dependencies.verifier_calls == [(
        filesystem.canonical_path("assets/urgent/model.ckpt"),
        filesystem.canonical_path("assets/urgent/source"),
        URGENT_IDENTITY["checkpoint_sha256"],
        URGENT_IDENTITY["source_commit"],
    )]
    assert plan.assets[0].settings["score_range"] == expected_range


def test_verified_identity_is_deeply_immutable_and_defensively_copied():
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    verified = _verify(_plan(), filesystem, dependencies)

    dependencies.snapshot_values["nisqa"]["checkpoint_size_bytes"] = 999
    dependencies.urgent_value["source_commit"] = "mutated"
    assert verified.options_by_model["nisqa"]["checkpoint_size_bytes"] == 501
    assert verified.provenance["models"]["urgent_pk"] == URGENT_IDENTITY

    with pytest.raises(TypeError):
        verified.options_by_model["urgent_pk"]["score_range"]["min"] = 0
    with pytest.raises(TypeError):
        verified.provenance["models"]["scoreq"]["model_dir"] = "mutated"

    payload = verified.as_mapping()
    payload["options_by_model"]["urgent_pk"]["score_range"]["min"] = 0
    payload["provenance"]["models"]["scoreq"]["model_dir"] = "mutated"
    assert verified.as_mapping() == {
        "selected_models": list(POLICY_MODEL_ORDER),
        "options_by_model": _expected_options(filesystem),
        "provenance": _expected_provenance(filesystem),
    }


def test_verified_result_contains_only_strict_json_asset_identity():
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)

    payload = _verify(_plan(), filesystem, dependencies).as_mapping()

    assert set(payload) == {"selected_models", "options_by_model", "provenance"}
    assert _all_keys(payload).isdisjoint({
        "estimator", "estimators", "fd", "file_descriptor", "process", "popen",
        "provider", "providers", "topology", "worker", "workers",
    })
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_preflight_failure_does_not_return_or_expose_partial_result():
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    sentinel = object()
    result = sentinel

    def fail_snapshot(path):
        dependencies.snapshot_calls.append(filesystem.canonical_path(path))
        raise RuntimeError("snapshot failed")

    dependencies.checkpoint_snapshot = fail_snapshot
    with pytest.raises(RuntimeError, match="snapshot failed"):
        result = _verify(
            _plan(("scoreq", "nisqa", "dnsmos")), filesystem, dependencies,
        )

    assert result is sentinel
    assert dependencies.digest_calls == [
        filesystem.canonical_path("assets/scoreq/model"),
        filesystem.canonical_path("assets/nisqa"),
    ]
    assert dependencies.snapshot_calls == [
        filesystem.canonical_path("assets/nisqa/model.tar"),
    ]


def test_preflight_provenance_has_exact_parity_with_legacy_helper(monkeypatch):
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    verified = _verify(_plan(), filesystem, dependencies)
    options = verified.as_mapping()["options_by_model"]

    monkeypatch.setattr(mos_assets.os.path, "isdir", filesystem.is_directory)
    monkeypatch.setattr(mos_assets.os.path, "isfile", filesystem.is_file)
    monkeypatch.setattr(
        mos_assets,
        "snapshot_local_checkpoint_identity",
        dependencies.checkpoint_snapshot,
    )
    legacy = mos_assets.selected_mos_asset_provenance(
        PROVENANCE_MODEL_ORDER,
        options,
        tree_digest=dependencies.tree_digest,
        urgent_verifier=dependencies.urgent_verifier,
    )

    verified_provenance = verified.as_mapping()["provenance"]
    assert verified_provenance == legacy
    assert artifacts._canonical_json_bytes(verified_provenance) == (
        artifacts._canonical_json_bytes(legacy)
    )


def test_preflight_import_and_scoreq_verification_keep_heavy_dependencies_lazy():
    package_root = Path(__file__).resolve().parents[1]
    tools_root = package_root.parent
    repository_root = tools_root.parent
    probe = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        class BlockHeavyDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, _path, _target=None):
                if (
                    fullname == "audio_filter_pipeline.processors.mos"
                    or fullname == "torch"
                    or fullname.startswith("torch.")
                ):
                    raise AssertionError(f"forbidden eager import: {fullname}")
                return None

        sys.meta_path.insert(0, BlockHeavyDependencies())

        from audio_filter_pipeline.processors.stages_v3.mos_asset_preflight import (
            verify_mos_assets,
        )
        from audio_filter_pipeline.processors.stages_v3.mos_execution import (
            MosAssetSpec,
            MosExecutionPlan,
        )

        forbidden = {"audio_filter_pipeline.processors.mos", "torch"}
        loaded = set(sys.modules)
        assert "audio_filter_pipeline.processors.mos" not in loaded
        assert not any(name == "torch" or name.startswith("torch.") for name in loaded)

        class Filesystem:
            @staticmethod
            def canonical_path(path):
                return os.path.realpath(os.path.abspath(path))

            @staticmethod
            def is_directory(_path):
                return True

            @staticmethod
            def is_file(_path):
                raise AssertionError("scoreq must not query file assets")

        asset = MosAssetSpec("scoreq", {"model_dir": "opaque/scoreq"})
        plan = MosExecutionPlan(
            ("scoreq",), (asset,), {"kind": "single_process", "device": "cpu"},
        )
        verified = verify_mos_assets(
            plan,
            filesystem=Filesystem(),
            tree_digest=lambda _path: "1" * 64,
        )
        assert verified.selected_models == ("scoreq",)
        loaded = set(sys.modules)
        assert "audio_filter_pipeline.processors.mos" not in loaded
        assert not any(name == "torch" or name.startswith("torch.") for name in loaded)
        print("OK")
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(tools_root), str(repository_root),
    ))

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=package_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "OK\n"


def test_verified_plan_cannot_be_forged_through_public_constructor():
    options = {"scoreq": {"model_dir": "unverified/model"}}
    provenance = {
        "selected_models": ["scoreq"],
        "models": {
            "scoreq": {
                "model_dir": "unverified/model",
                "verified_model_tree_sha256": "1" * 64,
            },
        },
    }

    with pytest.raises(TypeError, match="must be created by verify_mos_assets"):
        VerifiedMosExecutionPlan(
            ("scoreq",), options, provenance, _construction_token=object(),
        )


@pytest.mark.parametrize("model", ("scoreq", "dnsmos"))
@pytest.mark.parametrize(
    "digest",
    (
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        b"a" * 64,
    ),
    ids=("short", "long", "uppercase", "nonhex", "bytes"),
)
def test_directory_model_tree_digest_rejects_illegal_actual_values(
    model, digest,
):
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    dependencies.tree_digest = lambda _path: digest

    with pytest.raises(ValueError, match="lowercase SHA256"):
        _verify(_plan((model,)), filesystem, dependencies)


@pytest.mark.parametrize(
    ("source", "digest"),
    (
        ("expected", "a" * 63),
        ("expected", "a" * 65),
        ("expected", "A" * 64),
        ("expected", "g" * 64),
        ("actual", "b" * 63),
        ("actual", "b" * 65),
        ("actual", "B" * 64),
        ("actual", b"b" * 64),
    ),
    ids=(
        "expected-short", "expected-long", "expected-uppercase",
        "expected-nonhex", "actual-short", "actual-long", "actual-uppercase",
        "actual-bytes",
    ),
)
def test_nisqa_tree_digest_rejects_illegal_expected_and_actual_values(
    source, digest,
):
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    expected = digest if source == "expected" else TREE_DIGESTS["nisqa"]
    asset = MosAssetSpec(
        "nisqa",
        {"model_path": "assets/nisqa/model.tar"},
        {"model_version": "nisqa-v2", "model_sha256": expected},
    )
    plan = MosExecutionPlan(("nisqa",), (asset,), CPU_TOPOLOGY)
    if source == "actual":
        dependencies.tree_digest = lambda _path: digest

    with pytest.raises(ValueError, match="lowercase SHA256"):
        _verify(plan, filesystem, dependencies)


def test_nisqa_tree_digest_rejects_valid_but_mismatched_identity():
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    dependencies.tree_digest = lambda _path: "8" * 64

    with pytest.raises(ValueError, match="model tree digest mismatch"):
        _verify(_plan(("nisqa",)), filesystem, dependencies)


@pytest.mark.parametrize(
    "identity",
    (
        None,
        [],
        {"checkpoint_sha256": "5" * 64},
        {
            "checkpoint_sha256": "5" * 64,
            "checkpoint_size_bytes": 501,
            "extra": True,
        },
        {1: "5" * 64, "checkpoint_size_bytes": 501},
        {"checkpoint_sha256": "5" * 63, "checkpoint_size_bytes": 501},
        {"checkpoint_sha256": "G" * 64, "checkpoint_size_bytes": 501},
        {"checkpoint_sha256": "5" * 64, "checkpoint_size_bytes": -1},
        {"checkpoint_sha256": "5" * 64, "checkpoint_size_bytes": True},
        {"checkpoint_sha256": "5" * 64, "checkpoint_size_bytes": 501.0},
    ),
    ids=(
        "none", "sequence", "missing-field", "extra-field", "non-string-field",
        "short-digest", "nonhex-digest", "negative-size", "bool-size",
        "float-size",
    ),
)
def test_checkpoint_snapshot_identity_illegal_matrix_fails_closed(identity):
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    dependencies.snapshot_values["nisqa"] = identity

    with pytest.raises((TypeError, ValueError)):
        _verify(_plan(("nisqa",)), filesystem, dependencies)


@pytest.mark.parametrize(
    "identity",
    (
        None,
        [],
        {"checkpoint_sha256": "7" * 64},
        {
            "checkpoint_sha256": "7" * 64,
            "source_commit": "8" * 40,
            "extra": True,
        },
        {1: "7" * 64, "source_commit": "8" * 40},
        {"checkpoint_sha256": "7" * 63, "source_commit": "8" * 40},
        {"checkpoint_sha256": "Z" * 64, "source_commit": "8" * 40},
        {"checkpoint_sha256": "7" * 64, "source_commit": "8" * 39},
        {"checkpoint_sha256": "7" * 64, "source_commit": "A" * 40},
        {"checkpoint_sha256": "7" * 64, "source_commit": b"8" * 40},
        {**URGENT_IDENTITY, "source_package_sha256": "8" * 63},
        {**URGENT_IDENTITY, "source_package_sha256": "Z" * 64},
    ),
    ids=(
        "none", "sequence", "missing-field", "extra-field", "non-string-field",
        "short-digest", "nonhex-digest", "short-commit", "uppercase-commit",
        "bytes-commit", "short-package-digest", "nonhex-package-digest",
    ),
)
def test_urgent_verified_identity_illegal_matrix_fails_closed(identity):
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    dependencies.urgent_value = identity

    with pytest.raises((TypeError, ValueError)):
        _verify(_plan(("urgent_pk",)), filesystem, dependencies)


@pytest.mark.parametrize(
    "identity",
    (
        {**URGENT_IDENTITY, "checkpoint_sha256": "8" * 64},
        {**URGENT_IDENTITY, "source_commit": "9" * 40},
    ),
    ids=("checkpoint-mismatch", "commit-mismatch"),
)
def test_urgent_verified_identity_must_match_configured_expectation(identity):
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    dependencies.urgent_value = identity

    with pytest.raises(ValueError, match="digest mismatch|commit mismatch"):
        _verify(_plan(("urgent_pk",)), filesystem, dependencies)


@pytest.mark.parametrize(
    ("model", "method_name", "return_value"),
    (
        ("scoreq", "is_directory", None),
        ("scoreq", "is_directory", 0),
        ("scoreq", "is_directory", 1),
        ("scoreq", "is_directory", "yes"),
        ("nisqa", "is_file", None),
        ("nisqa", "is_file", 0),
        ("nisqa", "is_file", 1),
        ("nisqa", "is_file", "yes"),
        ("urgent_pk", "is_file", 1),
        ("urgent_pk", "is_directory", "yes"),
    ),
    ids=(
        "directory-none", "directory-zero", "directory-one", "directory-string",
        "file-none", "file-zero", "file-one", "file-string",
        "urgent-checkpoint-one", "urgent-source-string",
    ),
)
def test_filesystem_predicate_return_type_matrix_fails_closed(
    model, method_name, return_value,
):
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    setattr(filesystem, method_name, lambda _path: return_value)

    with pytest.raises(TypeError, match="filesystem.*bool|must return bool"):
        _verify(_plan((model,)), filesystem, dependencies)


@pytest.mark.parametrize(
    "canonical_value",
    (None, b"path", Path("path"), ""),
    ids=("none", "bytes", "pathlike", "blank"),
)
def test_filesystem_canonical_path_return_type_matrix_fails_closed(
    canonical_value,
):
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    filesystem.canonical_path = lambda _path: canonical_value

    with pytest.raises((TypeError, ValueError), match="canonical path"):
        _verify(_plan(("scoreq",)), filesystem, dependencies)


def test_verified_mapping_has_canonical_json_bytes_parity():
    filesystem = _full_filesystem()
    dependencies = _FakeAssetDependencies(filesystem)
    payload = _verify(_plan(), filesystem, dependencies).as_mapping()
    expected = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    assert artifacts._canonical_json_bytes(payload) == expected
    assert json.loads(expected.decode("utf-8")) == payload
