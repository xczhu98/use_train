"""实验报告 checkpoint identity v3 生成合同。"""

from __future__ import annotations

import pytest

from tools.ops.reporting.experiment_report_identity import (
    V3_SCHEMA,
    build_identity_record,
    build_payload,
    canonical_json,
    identity_from_payload,
)

pytestmark = pytest.mark.unit


def test_v3_canonical_payload_and_identity_golden() -> None:
    payload = build_payload(
        schema=V3_SCHEMA,
        experiment="实验 α",
        source_path="/tmp/run-一/model 60.tar",
        source_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    assert canonical_json(payload) == (
        '{"effective_config_sha256":"2222222222222222222222222222222222222222222222222222222222222222",'
        '"experiment":"实验 α","schema":"explicit-checkpoint-identity-v3",'
        '"source_name":"model 60.tar","source_path":"/tmp/run-一/model 60.tar",'
        '"source_run":"run-一","source_sha256":"1111111111111111111111111111111111111111111111111111111111111111"}'
    )
    assert identity_from_payload(payload) == "experiment@model-60.tar"


def test_identity_rejects_noncanonical_or_tampered_payload() -> None:
    record = build_identity_record(
        schema=V3_SCHEMA,
        experiment="demo",
        source_path="/tmp/run/model.tar",
        source_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    payload = dict(record["checkpoint_identity_payload"])
    payload["extra"] = "opaque"
    with pytest.raises(ValueError, match="not canonical"):
        identity_from_payload(payload)


def test_legacy_v2_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint identity schema"):
        build_payload(
            schema="explicit-checkpoint-identity-v2",
            experiment="demo",
            source_path="/tmp/run/model.tar",
            source_sha256="a" * 64,
            config_sha256="b" * 64,
        )


def test_legacy_v2_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint identity schema"):
        identity_from_payload(
            {
                "schema": "explicit-checkpoint-identity-v2",
                "experiment": "demo",
                "source_run": "run",
                "source_path": "/tmp/run/model.tar",
                "source_name": "model.tar",
                "source_sha256": "a" * 64,
                "config_sha256": "b" * 64,
            }
        )
