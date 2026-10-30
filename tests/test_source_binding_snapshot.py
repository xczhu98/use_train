"""公共 source binding 的加载身份、隔离和 fail-closed 合同。"""

import pytest

from core.utils.artifact_identity import SourceBinding
from core.utils.inference_artifacts import build_content_fingerprint

pytestmark = pytest.mark.unit


def test_constructor_rejects_tampered_aggregate_without_rebinding(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"loaded-A")
    fingerprint = build_content_fingerprint(
        {"checkpoint": source}, parameters={"model": {"channels": [1, 2]}}
    )
    fingerprint["parameters"]["model"]["channels"][0] = 99

    with pytest.raises(ValueError, match="sha256"):
        SourceBinding(fingerprint)


@pytest.mark.parametrize("factory", ["constructor", "from_fingerprint"])
def test_binding_keeps_deeply_isolated_loaded_identity(tmp_path, factory):
    source = tmp_path / "source.bin"
    source.write_bytes(b"loaded-A")
    fingerprint = build_content_fingerprint(
        {"checkpoint": source, "optional": None},
        parameters={"model": {"channels": [1, 2]}},
    )
    binding = (
        SourceBinding(fingerprint) if factory == "constructor"
        else SourceBinding.from_fingerprint(fingerprint)
    )
    fingerprint["parameters"]["model"]["channels"][0] = 99
    exported = binding.as_dict()
    exported["files"]["checkpoint"]["size"] = 999
    exported["parameters"]["model"]["channels"].append(3)

    assert binding.as_dict()["parameters"] == {"model": {"channels": [1, 2]}}
    assert binding.as_dict()["files"]["optional"] is None
    binding.verify()
    source.write_bytes(b"loaded-B")
    with pytest.raises(RuntimeError, match="source artifact changed"):
        binding.verify()


def test_binding_accepts_recursive_immutable_json_mappings():
    import json
    from types import MappingProxyType

    parameters = MappingProxyType({"layers": (MappingProxyType({"size": 3}), None)})
    binding = SourceBinding.bind({"optional": None}, parameters=parameters)
    assert json.loads(json.dumps(binding.as_dict()))["parameters"] == {
        "layers": [{"size": 3}, None]
    }
    assert SourceBinding.from_fingerprint(MappingProxyType(binding.as_dict())) == binding


@pytest.mark.parametrize("invalid", [{1: "coerced"}, {"x": float("nan")}, {"x": {1, 2}}])
def test_binding_rejects_non_json_parameters(invalid):
    with pytest.raises(ValueError):
        SourceBinding.bind({"optional": None}, parameters=invalid)


@pytest.mark.parametrize("factory", [SourceBinding, SourceBinding.from_fingerprint])
@pytest.mark.parametrize("change", ["algorithm", "sha256", "extra", "files", "parameters"])
def test_all_constructors_validate_original_schema_and_digest(factory, change):
    fingerprint = build_content_fingerprint({"optional": None})
    if change == "extra":
        fingerprint["extra"] = None
    elif change == "files":
        fingerprint["files"] = {}
    elif change == "parameters":
        fingerprint["parameters"] = []
    else:
        fingerprint[change] = "invalid"
    with pytest.raises(ValueError):
        factory(fingerprint)


@pytest.mark.parametrize("replacement", [False, True])
def test_precomputed_identity_is_not_refreshed_after_source_change(tmp_path, replacement):
    from core.utils.inference_artifacts import fingerprint_file

    source = tmp_path / "source.bin"
    source.write_bytes(b"loaded-A")
    loaded_identity = fingerprint_file(source)
    if replacement:
        other = tmp_path / "other.bin"
        other.write_bytes(b"loaded-B")
        other.replace(source)
    else:
        source.write_bytes(b"loaded-B")
    binding = SourceBinding.bind(
        {"checkpoint": source, "optional": None},
        precomputed={"checkpoint": loaded_identity},
    )
    assert binding.as_dict()["files"]["checkpoint"] == loaded_identity
    with pytest.raises(RuntimeError, match="source artifact changed"):
        binding.verify()


def test_verify_rejects_unknown_selection_and_checks_aggregate():
    binding = SourceBinding.bind({"optional": None})
    with pytest.raises(ValueError, match="unknown source artifact"):
        binding.verify(names=("typo",))
    binding.verify(names=())
    binding.verify(names=("optional",))

    from core.utils.artifact_identity import verify_source_binding

    tampered = binding.as_dict()
    tampered["parameters"]["injected"] = True
    with pytest.raises(ValueError, match="sha256"):
        verify_source_binding(tampered, names=())


def test_binding_rejects_json_cycles():
    parameters = {"nested": []}
    parameters["nested"].append(parameters)
    with pytest.raises(ValueError, match="cycles"):
        SourceBinding.bind({"optional": None}, parameters=parameters)


@pytest.mark.parametrize("absent", [False, True])
def test_precomputed_identity_cannot_override_declared_source(tmp_path, absent):
    from core.utils.inference_artifacts import fingerprint_file

    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"A")
    b.write_bytes(b"B")
    with pytest.raises(ValueError, match="precomputed"):
        SourceBinding.bind(
            {"checkpoint": None if absent else a},
            precomputed={"checkpoint": fingerprint_file(b)},
        )
