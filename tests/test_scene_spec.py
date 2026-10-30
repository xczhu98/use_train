"""SceneSpec 与统一 reference resolver 的纯合同测试。"""

import json

import numpy as np
import pytest

from core.data.simulation.reference_resolver import (
    InvalidReferenceError,
    resolve_erl_reference,
    resolve_ser_reference,
    resolve_sir_reference,
    resolve_snr_reference,
)
from core.data.simulation.scene_spec import SceneSpec


pytestmark = pytest.mark.unit


def test_scene_spec_round_trip_is_json_safe():
    spec = SceneSpec(
        scene_name="prob_target_interference_noise",
        target_present=True,
        interference_present=True,
        background_present=True,
        echo_present=False,
        source_count=3,
        task="tse",
    )

    payload = spec.to_dict()
    assert json.loads(spec.to_json()) == payload
    assert SceneSpec.from_dict(payload) == spec
    assert SceneSpec.from_json(spec.to_json()) == spec


def test_scene_spec_rejects_inconsistent_or_unknown_payload():
    with pytest.raises(ValueError, match="requires at least one"):
        SceneSpec("prob_target_only", True, False, False, False, 0)
    with pytest.raises(ValueError, match="unknown fields"):
        SceneSpec.from_dict(
            {
                "scene_name": "prob_noise_only",
                "target_present": False,
                "interference_present": False,
                "background_present": True,
                "echo_present": False,
                "source_count": 0,
                "unexpected": True,
            }
        )


def test_snr_uses_rendered_sources_and_returns_none_without_background():
    speech = np.array([1.0, -2.0], dtype=np.float32)
    active = SceneSpec("prob_target_noise", True, False, True, False, 1, "se")
    fact = resolve_snr_reference(active, [speech], None)
    assert fact == {
        "role": "rendered_sources",
        "measure_point": "post_rir_pre_background",
        "energy": 5.0,
    }

    no_background = SceneSpec("prob_target_only", True, False, False, False, 1, "se")
    assert resolve_snr_reference(no_background, None, None) is None


def test_snr_uses_latent_reference_only_for_speech_free_background():
    latent = np.array([2.0, 0.0], dtype=np.float32)
    scene = SceneSpec("prob_noise_only", False, False, True, False, 0, "tse")
    fact = resolve_snr_reference(scene, None, {"target": latent})
    assert fact["role"] == "latent_reference"
    assert fact["measure_point"] == "latent_reference"
    assert fact["energy"] == 4.0

    with pytest.raises(InvalidReferenceError, match="requires latent"):
        resolve_snr_reference(scene, None, None)
    with pytest.raises(InvalidReferenceError, match="must not provide"):
        resolve_snr_reference(scene, [latent], {"target": latent})


def test_sir_is_not_applicable_without_interference_and_requires_both_sides():
    target = np.ones(4, dtype=np.float32)
    interference = np.full(4, 2.0, dtype=np.float32)
    no_interference = SceneSpec("prob_target_noise", True, False, True, False, 1, "tse")
    assert resolve_sir_reference(no_interference, target, [interference]) is None

    scene = SceneSpec("prob_interference_only", False, True, False, False, 1, "tse")
    result = resolve_sir_reference(scene, target, [interference])
    assert result["target"]["role"] == "target"
    assert result["interference"]["energy"] == 16.0
    with pytest.raises(InvalidReferenceError, match="target_full_rir"):
        resolve_sir_reference(scene, None, [interference])
    with pytest.raises(InvalidReferenceError, match="at least one"):
        resolve_sir_reference(scene, target, [])


def test_ser_and_erl_distinguish_not_applicable_from_invalid_inputs():
    nearend = np.ones(2, dtype=np.float32)
    echo = np.full(2, 2.0, dtype=np.float32)
    assert resolve_ser_reference(None, None) is None
    assert resolve_erl_reference(None, None) is None
    assert resolve_ser_reference(nearend, echo)["nearend"]["energy"] == 2.0
    assert resolve_erl_reference(nearend, echo)["echo"]["energy"] == 8.0

    with pytest.raises(InvalidReferenceError, match="both nearend"):
        resolve_ser_reference(None, echo)
    with pytest.raises(InvalidReferenceError, match="both farend"):
        resolve_erl_reference(nearend, None)
    with pytest.raises(InvalidReferenceError, match="positive"):
        resolve_ser_reference(np.zeros(2, dtype=np.float32), echo)
