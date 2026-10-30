"""tools.shared.scp_provenance 的单元契约测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.shared.scp_provenance import (
    build_scp_provenance,
    parse_scp_snapshot,
    scp_provenance_fresh,
    scp_sidecar_path,
    validate_scp_provenance,
    write_scp_provenance,
)

pytestmark = pytest.mark.unit


def _scp_bytes() -> bytes:
    return b"# comment\n\nu1 /tmp/u1.wav\nu2 16000 32000 /tmp/u2.wav\n"


def test_sidecar_path_naming():
    assert scp_sidecar_path("/a/b.scp").name == "b.scp.provenance.json"


def test_parse_snapshot_counts_entries_and_skips_comments():
    parsed = parse_scp_snapshot(Path("t.scp"), _scp_bytes())
    assert parsed == {"u1": "/tmp/u1.wav", "u2": "/tmp/u2.wav"}


def test_parse_snapshot_rejects_duplicate_uid_and_bad_columns():
    with pytest.raises(ValueError, match="重复 uid"):
        parse_scp_snapshot(Path("t.scp"), b"u1 /a.wav\nu1 /b.wav\n")
    with pytest.raises(ValueError):
        parse_scp_snapshot(Path("t.scp"), b"only-one-column\n")


def test_build_provenance_schema_and_reserved_keys():
    payload = build_scp_provenance(
        _scp_bytes(), kind="generated", source="/src", pattern="*.wav"
    )
    assert payload["version"] == 1
    assert payload["kind"] == "generated"
    assert payload["source"] == "/src"
    assert payload["pattern"] == "*.wav"
    assert payload["entries"] == 2
    assert payload["scp_sha256"] == hashlib.sha256(_scp_bytes()).hexdigest()
    assert payload["created_at"]

    copied = build_scp_provenance("u1 /tmp/a.wav\n", kind="copied", source="/s.scp")
    assert copied["version"] == 2
    assert copied["entries"] == 1

    with pytest.raises(ValueError, match="未知 SCP provenance kind"):
        build_scp_provenance(b"u1 /a.wav\n", kind="linked", source="/s")
    with pytest.raises(TypeError, match="保留键"):
        build_scp_provenance(b"u1 /a.wav\n", kind="generated", source="/s", entries=99)
    with pytest.raises(ValueError):
        build_scp_provenance(b"bad line with too many cols here\n", kind="generated", source="/s")


def test_write_and_validate_roundtrip(tmp_path):
    scp = tmp_path / "a.scp"
    scp.write_bytes(_scp_bytes())
    payload = build_scp_provenance(scp.read_bytes(), kind="generated", source=str(tmp_path))
    sidecar = write_scp_provenance(scp, payload)
    assert json.loads(sidecar.read_text(encoding="utf-8"))["entries"] == 2
    assert validate_scp_provenance(scp)["source"] == str(tmp_path)


def test_validate_fail_closed_on_tamper(tmp_path):
    scp = tmp_path / "a.scp"
    scp.write_bytes(_scp_bytes())
    write_scp_provenance(
        scp, build_scp_provenance(scp.read_bytes(), kind="generated", source="/s")
    )
    sidecar = scp_sidecar_path(scp)

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["entries"] = 99
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="entry count mismatch"):
        validate_scp_provenance(scp)

    write_scp_provenance(
        scp, build_scp_provenance(scp.read_bytes(), kind="generated", source="/s")
    )
    scp.write_bytes(b"u1 /tmp/other.wav\nu2 16000 32000 /tmp/o2.wav\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_scp_provenance(scp)

    sidecar.unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_scp_provenance(scp)

    sidecar.write_text('{"kind": "linked", "entries": 1, "scp_sha256": "x"}\n')
    with pytest.raises(ValueError, match="schema is invalid"):
        validate_scp_provenance(scp)


def test_fresh_subset_semantics(tmp_path):
    scp = tmp_path / "a.scp"
    scp.write_bytes(_scp_bytes())
    write_scp_provenance(
        scp,
        build_scp_provenance(
            scp.read_bytes(), kind="generated", source="/s", pattern="*.wav"
        ),
    )
    assert scp_provenance_fresh(scp, {"kind": "generated", "pattern": "*.wav"})
    assert not scp_provenance_fresh(scp, {"kind": "generated", "pattern": "*.flac"})
    assert not scp_provenance_fresh(tmp_path / "missing.scp", {})

    scp.write_bytes(b"u9 /tmp/u9.wav\n")
    assert not scp_provenance_fresh(scp, {"kind": "generated"})

    scp_sidecar_path(scp).write_text("not json\n", encoding="utf-8")
    assert not scp_provenance_fresh(scp, {})
