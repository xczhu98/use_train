"""C3 无阈值 top-5 准入与旧契约 fail-closed 测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.data import noise_admission_policy as policy_module
from core.data.noise_admission_policy import (
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    GENERIC_NOISE_MID,
    LEGACY_C2_LABEL_FIELDS,
    LEGACY_C3_FIELDS,
    SCORE_CONTRACT,
    EvidenceBundle,
    EvidenceTopItem,
    decision_to_admission_extra,
    evaluate_noise_admission,
    load_admission_policy,
    parse_c2_evidence,
    validate_admission_contract,
)

pytestmark = pytest.mark.unit

_POLICY = load_admission_policy()


def _c2_stage() -> dict:
    return {
        "status": "passed",
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "score_contract": SCORE_CONTRACT,
        "class_count": CLASSIFIER_LABEL_COUNT,
        "label_space_sha256": _POLICY.label_space_sha256,
    }


def _top5(*mids: str) -> dict:
    if len(mids) != 5:
        raise AssertionError("test evidence must contain exactly five MIDs")
    items = []
    for rank, mid in enumerate(mids):
        index = _POLICY.mid_to_index[mid]
        items.append(
            {
                "index": index,
                "mid": mid,
                "label": _POLICY.label_of(mid),
                "prob": 0.9 - rank * 0.1,
            }
        )
    return {"audioset_top5": items}


def _mids(kind: str, count: int) -> tuple[str, ...]:
    return tuple(
        mid
        for mid in _POLICY.index_to_mid.values()
        if _POLICY.class_of(mid) == kind
    )[:count]


def _decision(c1: str, labels: dict):
    evidence = parse_c2_evidence(labels, _POLICY, _c2_stage())
    return evaluate_noise_admission(
        c1_intelligibility=c1,
        evidence=evidence,
        policy=_POLICY,
    )


def test_calibration_attestation_and_approval_runtime_api_is_removed():
    removed = {
        "AdmissionCalibration",
        "AdmissionAttestation",
        "load_admission_calibration",
        "validate_admission_attestation_binding",
        "default_calibration_path",
        "default_attestation_path",
        "ADMISSION_CALIBRATION_VERSION",
        "ADMISSION_ATTESTATION_VERSION",
    }

    assert not (removed & set(vars(policy_module)))


@pytest.mark.parametrize(
    (
        "c1",
        "mids",
        "expected_decision",
        "expected_reasons",
        "expected_category_mid",
        "expected_category",
        "expected_source",
    ),
    [
        (
            "noise",
            ("/m/06h7j", "/m/07qv_x_", "/m/07pbtc8", "/m/01g90h", "/m/02_nn"),
            "accept_noise",
            ("admitted_noise",),
            "/m/06h7j",
            "Run",
            "noise_top5_hit",
        ),
        (
            "babble",
            ("/m/06h7j", "/m/07qv_x_", "/m/07pbtc8", "/m/01g90h", "/m/02_nn"),
            "accept_babble",
            ("admitted_babble",),
            "/m/06h7j",
            "Run",
            "noise_top5_hit",
        ),
        (
            "noise",
            ("/m/06h7j", "/m/09x0r", "/m/07qv_x_", "/m/07pbtc8", "/m/01g90h"),
            "reject",
            ("speech_mid_in_top5",),
            None,
            None,
            None,
        ),
    ],
)
def test_frozen_admission_matrix_matches_independent_expected_values(
    c1,
    mids,
    expected_decision,
    expected_reasons,
    expected_category_mid,
    expected_category,
    expected_source,
):
    """固定输入与期望均不由生产 evaluator 推导，防止测试同错。"""
    decision = _decision(c1, _top5(*mids))

    assert decision.decision == expected_decision
    assert decision.reason_codes == expected_reasons
    assert decision.balance_category_mid == expected_category_mid
    assert decision.balance_category == expected_category
    assert decision.balance_category_source == expected_source


def test_c1_speech_is_terminal_without_c2_evidence():
    decision = evaluate_noise_admission(
        c1_intelligibility="speech",
        evidence=None,
        policy=_POLICY,
    )

    assert decision.decision == "reject"
    assert decision.reason_codes == ("c1_speech",)
    assert decision.top5_hits == ()


@pytest.mark.parametrize(
    ("c1", "expected"),
    [("noise", "accept_noise"), ("babble", "accept_babble")],
)
def test_c1_noise_and_babble_choose_branch_without_thresholds(c1, expected):
    noise_mids = _mids("noise", 5)
    decision = _decision(c1, _top5(*noise_mids))

    assert decision.decision == expected
    assert decision.admission_allowed is True
    assert decision.balance_category_mid == noise_mids[0]
    assert decision.balance_category_source == "noise_top5_hit"


def test_any_speech_mid_in_top5_vetoes_both_c1_accept_branches():
    speech_mid = _mids("speech", 1)[0]
    noise_mids = _mids("noise", 4)
    labels = _top5(noise_mids[0], speech_mid, *noise_mids[1:])

    for c1 in ("noise", "babble"):
        decision = _decision(c1, labels)
        assert decision.decision == "reject"
        assert decision.reason_codes == ("speech_mid_in_top5",)
        assert decision.admission_allowed is False


def test_babble_mids_do_not_affect_decision_and_use_noise_fallback():
    babble_mids = tuple(_POLICY.index_to_mid.values())[:5]
    synthetic_policy = replace(
        _POLICY,
        mid_to_class={
            mid: ("babble" if mid in babble_mids else kind)
            for mid, kind in _POLICY.mid_to_class.items()
        },
    )
    evidence = parse_c2_evidence(_top5(*babble_mids), synthetic_policy, _c2_stage())
    decision = evaluate_noise_admission(
        c1_intelligibility="babble",
        evidence=evidence,
        policy=synthetic_policy,
    )

    assert decision.decision == "accept_babble"
    assert decision.balance_category_mid == GENERIC_NOISE_MID
    assert decision.balance_category == "Noise"
    assert decision.balance_category_source == "generic_noise_fallback"


def test_balance_category_uses_highest_ranked_noise_mid_not_largest_index():
    babble_mid = _mids("babble", 1)[0]
    noise_mids = _mids("noise", 4)
    labels = _top5(babble_mid, noise_mids[2], noise_mids[0], noise_mids[1], noise_mids[3])

    decision = _decision("noise", labels)

    assert decision.balance_category_mid == noise_mids[2]
    assert decision.balance_category_source == "noise_top5_hit"


@pytest.mark.parametrize("legacy_field", sorted(LEGACY_C2_LABEL_FIELDS))
def test_legacy_c2_evidence_fields_fail_closed(legacy_field):
    labels = _top5(*_mids("noise", 5))
    labels[legacy_field] = []

    with pytest.raises(ValueError, match="legacy C2 label fields are forbidden"):
        parse_c2_evidence(labels, _POLICY, _c2_stage())


def test_labels_original_remains_allowed():
    labels = _top5(*_mids("noise", 5))
    labels["original"] = "upstream-source-label"

    evidence = parse_c2_evidence(labels, _POLICY, _c2_stage())

    assert len(evidence.top5) == 5


@pytest.mark.parametrize("legacy_field", sorted(LEGACY_C3_FIELDS))
def test_legacy_c3_fields_fail_closed(legacy_field):
    labels = _top5(*_mids("noise", 5))
    decision = _decision("noise", labels)
    stage = decision_to_admission_extra(
        decision,
        execution_mode={"c1": "production", "c2": "subprocess_daemon"},
        publishable=True,
        artifact_fingerprints={
            "voice_policy_sha256": _POLICY.sha256,
            "audiotag_label_dict_sha256": _POLICY.label_space_sha256,
        },
    )
    stage[legacy_field] = True

    with pytest.raises(ValueError, match="legacy C3 fields are forbidden"):
        validate_admission_contract(
            c1_intelligibility="noise",
            c2_labels=labels,
            c2_stage=_c2_stage(),
            admission_stage=stage,
            policy=_POLICY,
            require_accepted=True,
            location="unit-test",
        )


def test_c2_labels_and_stage_use_exact_allowlists():
    labels = _top5(*_mids("noise", 5))
    labels["unexpected"] = True
    with pytest.raises(ValueError, match=r"invalid C2 labels fields: unknown=.*unexpected"):
        parse_c2_evidence(labels, _POLICY, _c2_stage())

    c2_stage = _c2_stage()
    c2_stage["unexpected"] = True
    with pytest.raises(ValueError, match=r"invalid stages\.C2 fields: unknown=.*unexpected"):
        parse_c2_evidence(_top5(*_mids("noise", 5)), _POLICY, c2_stage)


def test_c3_stage_uses_exact_allowlist():
    labels = _top5(*_mids("noise", 5))
    decision = _decision("noise", labels)
    stage = decision_to_admission_extra(
        decision,
        execution_mode={"c1": "production", "c2": "subprocess_daemon"},
        publishable=True,
        artifact_fingerprints={
            "voice_policy_sha256": _POLICY.sha256,
            "audiotag_label_dict_sha256": _POLICY.label_space_sha256,
        },
    )
    stage["unexpected"] = True

    with pytest.raises(ValueError, match=r"invalid stages\.C3 fields: unknown=.*unexpected"):
        validate_admission_contract(
            c1_intelligibility="noise",
            c2_labels=labels,
            c2_stage=_c2_stage(),
            admission_stage=stage,
            policy=_POLICY,
            require_accepted=True,
            location="unit-test",
        )


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("reason", lambda stage: stage.__setitem__("reason", "wrong")),
        ("admission_reason_codes", lambda stage: stage.__setitem__("admission_reason_codes", [])),
        ("top5_hits", lambda stage: stage["top5_hits"].pop()),
        ("balance_category_mid", lambda stage: stage.__setitem__("balance_category_mid", GENERIC_NOISE_MID)),
        ("admission_policy_sha256", lambda stage: stage.__setitem__("admission_policy_sha256", "0" * 64)),
        ("evidence_contract_version", lambda stage: stage.__setitem__("evidence_contract_version", 4)),
    ],
)
def test_validate_contract_rejects_each_forged_canonical_field(field, mutate):
    labels = _top5(*_mids("noise", 5))
    decision = _decision("noise", labels)
    stage = decision_to_admission_extra(
        decision,
        execution_mode={"c1": "production", "c2": "subprocess_daemon"},
        publishable=True,
        artifact_fingerprints={
            "voice_policy_sha256": _POLICY.sha256,
            "audiotag_label_dict_sha256": _POLICY.label_space_sha256,
        },
    )
    mutate(stage)

    with pytest.raises(ValueError, match=rf"stages\.C3\.{field} .*does not match"):
        validate_admission_contract(
            c1_intelligibility="noise",
            c2_labels=labels,
            c2_stage=_c2_stage(),
            admission_stage=stage,
            policy=_POLICY,
            require_accepted=True,
            location="unit-test",
        )


def test_evaluator_revalidates_directly_constructed_bundle_top5_integrity():
    parsed = parse_c2_evidence(_top5(*_mids("noise", 5)), _POLICY, _c2_stage())
    bad_items = list(parsed.top5)
    bad_items[1] = EvidenceTopItem(
        index=bad_items[0].index,
        mid=bad_items[0].mid,
        label=bad_items[0].label,
        prob=bad_items[1].prob,
    )
    forged = EvidenceBundle(
        top5=tuple(bad_items),
        evidence_contract_version=parsed.evidence_contract_version,
        score_contract=parsed.score_contract,
        class_count=parsed.class_count,
        label_space_sha256=parsed.label_space_sha256,
    )

    with pytest.raises(ValueError, match="duplicated"):
        evaluate_noise_admission(
            c1_intelligibility="noise",
            evidence=forged,
            policy=_POLICY,
        )


def test_generic_fallback_mid_must_be_noise():
    babble_mids = tuple(_POLICY.index_to_mid.values())[:5]
    synthetic_policy = replace(
        _POLICY,
        mid_to_class={
            mid: (
                "babble"
                if mid in {*babble_mids, GENERIC_NOISE_MID}
                else kind
            )
            for mid, kind in _POLICY.mid_to_class.items()
        },
    )
    evidence = parse_c2_evidence(_top5(*babble_mids), synthetic_policy, _c2_stage())

    with pytest.raises(ValueError, match="generic fallback MID must map to noise"):
        evaluate_noise_admission(
            c1_intelligibility="noise",
            evidence=evidence,
            policy=synthetic_policy,
        )
