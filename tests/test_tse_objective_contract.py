"""TSE dual self-estimated objective 的配置与模型能力合同。"""

from __future__ import annotations

import pytest

from tasks.tse.objectives import (
    resolve_dual_self_estimated_objective,
    validate_dual_objective_model,
    validate_tse_objective_config,
)


pytestmark = pytest.mark.unit


def _config(*, model_type: str = "TSE_BSRNN_SPK", args: dict | None = None) -> dict:
    return {
        "model": {"type": model_type, "args": args or {"joint_training": True}},
        "trainer": {
            "objectives": {
                "dual_self_estimated": {
                    "enabled": True,
                    "original_loss_weight": 0.4,
                    "self_loss_weight": 0.6,
                }
            }
        },
    }


def test_missing_objective_is_disabled_and_defaults_are_canonical() -> None:
    assert resolve_dual_self_estimated_objective({}) is None
    assert resolve_dual_self_estimated_objective(
        {"objectives": {"dual_self_estimated": {}}}
    ) == (0.4, 0.6)
    assert resolve_dual_self_estimated_objective(
        {"objectives": {"dual_self_estimated": {"enabled": False}}}
    ) is None


@pytest.mark.parametrize(
    ("trainer", "error", "match"),
    [
        ({"objectives": []}, TypeError, "trainer.objectives"),
        ({"objectives": {"unknown": {}}}, KeyError, "unknown"),
        (
            {"objectives": {"dual_self_estimated": {"typo": True}}},
            KeyError,
            "typo",
        ),
        (
            {
                "objectives": {
                    "dual_self_estimated": {
                        "enabled": False,
                        "original_loss_weight": 0.4,
                    }
                }
            },
            ValueError,
            "omit unused weights",
        ),
        (
            {
                "objectives": {"dual_self_estimated": {"enabled": True}},
                "ssa_prob": 0.5,
            },
            ValueError,
            "cannot be combined",
        ),
    ],
)
def test_objective_schema_rejects_ambiguous_values(trainer, error, match) -> None:
    with pytest.raises(error, match=match):
        resolve_dual_self_estimated_objective(trainer)


def test_config_rejects_retired_model_schema_and_unsupported_model() -> None:
    with pytest.raises(ValueError, match="TSE_BSRNN_Multi.*removed"):
        validate_tse_objective_config({**_config(model_type="TSE_BSRNN_Multi")})
    with pytest.raises(ValueError, match="model.args keys are removed"):
        validate_tse_objective_config(
            _config(args={"joint_training": True, "original_loss_weight": 0.4})
        )
    with pytest.raises(ValueError, match="joint_training=true"):
        validate_tse_objective_config(_config(args={"joint_training": False}))
    with pytest.raises(ValueError, match="TSE_Modular"):
        validate_tse_objective_config(_config(model_type="TSE_Modular"))


@pytest.mark.parametrize(
    "config",
    [
        {"model": None, "trainer": {}},
        {"model": [], "trainer": {}},
        {"model": {"type": "TSE_BSRNN_SPK", "args": []}, "trainer": {}},
    ],
)
def test_config_rejects_non_mapping_model_sections(config) -> None:
    with pytest.raises(TypeError, match="mapping"):
        validate_tse_objective_config(config)


def test_runtime_requires_explicit_capability_and_rejects_retired_attribute() -> None:
    class _Supported:
        supports_dual_self_estimated = True

    class _Unsupported:
        supports_dual_self_estimated = False

    class _Retired:
        dual_self_estimated_loss_weights = None

    validate_dual_objective_model(_Supported(), (0.4, 0.6))
    validate_dual_objective_model(_Supported(), None)
    with pytest.raises(ValueError, match="supports_dual_self_estimated"):
        validate_dual_objective_model(_Unsupported(), (0.4, 0.6))
    with pytest.raises(ValueError, match="dual_self_estimated_loss_weights.*removed"):
        validate_dual_objective_model(_Retired(), None)
