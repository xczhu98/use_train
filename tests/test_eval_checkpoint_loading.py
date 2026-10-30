"""SE/AEC 评估与通用推理 checkpoint family 契约。"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from core.utils.inference_artifacts import fingerprint_file
from evaluate.checkpoint_loading import (
    CheckpointFamily,
    apply_checkpoint,
    apply_ema_if_present,
    extract_state_dict,
    load_checkpoint,
    load_state_dict_checked,
    strip_ddp_prefix,
)

pytestmark = [pytest.mark.unit]


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


class _ExtraStateModel(_TinyModel):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_extra_state = None

    def get_extra_state(self):
        return {"version": 1}

    def set_extra_state(self, state) -> None:
        self.loaded_extra_state = state


class _TiedStateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(2, 2, bias=False)
        self.right = nn.Linear(2, 2, bias=False)
        self.right.weight = self.left.weight
        shared = torch.ones(2)
        self.register_buffer("first_buffer", shared)
        self.register_buffer("second_buffer", shared)


class _LoadOverrideFailure(_TiedStateModel):
    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.left.weight.data = torch.full((1,), 91.0, dtype=torch.float64)
        self.left.weight = nn.Parameter(torch.full((2, 2), 92.0), requires_grad=False)
        self.first_buffer.data = torch.full((1,), 93.0, dtype=torch.float64)
        self.first_buffer.requires_grad_(True)
        self.first_buffer = torch.full((2,), 94.0)
        self.left = nn.Linear(2, 2)
        raise RuntimeError("public load failure")


class _LoadFromStateFailure(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.register_buffer("buffer", torch.ones(2))

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.weight.requires_grad_(False)
        raise RuntimeError("private load failure")


class _SetExtraStateFailure(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.extra = {"version": 1}

    def get_extra_state(self):
        return self.extra

    def set_extra_state(self, state) -> None:
        self.extra = state
        if state.get("fail"):
            self.weight = nn.Parameter(torch.full((2,), 99.0), requires_grad=False)
            raise RuntimeError("extra state failure")


def _state(model: nn.Module, *, prefix: str = "") -> dict[str, torch.Tensor]:
    return {f"{prefix}{key}": value.clone() for key, value in model.state_dict().items()}


def test_checkpoint_families_are_explicit_and_do_not_cross_load() -> None:
    model = _TinyModel()
    single = {"model": _state(model)}
    gan = {"model_generator": _state(model)}

    assert set(extract_state_dict(single, CheckpointFamily.SINGLE_MODEL)) == set(
        model.state_dict()
    )
    assert set(extract_state_dict(gan, CheckpointFamily.GAN_GENERATOR)) == set(
        model.state_dict()
    )
    with pytest.raises(RuntimeError, match="model_generator"):
        extract_state_dict(single, CheckpointFamily.GAN_GENERATOR)
    with pytest.raises(RuntimeError, match="single_model checkpoint"):
        extract_state_dict(gan, CheckpointFamily.SINGLE_MODEL)


def test_single_model_rejects_ambiguous_wrappers_and_non_tensor_values() -> None:
    model = _TinyModel()
    state = _state(model)
    with pytest.raises(RuntimeError, match="多个候选"):
        extract_state_dict(
            {"model": state, "state_dict": state},
            CheckpointFamily.SINGLE_MODEL,
        )
    with pytest.raises(RuntimeError, match="non_tensor_keys"):
        extract_state_dict(
            {"model": {**state, "metadata": "bad"}},
            CheckpointFamily.SINGLE_MODEL,
        )


def test_ddp_prefix_is_normalized_per_key_and_collision_fails() -> None:
    model = _TinyModel()
    prefixed = _state(model, prefix="module.")
    normalized = strip_ddp_prefix(prefixed)
    assert set(normalized) == set(model.state_dict())

    collision = {
        "linear.weight": model.linear.weight.detach().clone(),
        "module.linear.weight": model.linear.weight.detach().clone(),
    }
    with pytest.raises(RuntimeError, match="参数键冲突"):
        strip_ddp_prefix(collision)


@pytest.mark.parametrize("invalid_kind", ["base", "key", "shape"])
def test_base_state_validation_fails_before_model_write(invalid_kind: str) -> None:
    model = _TinyModel()
    invalid = _state(model)
    original = _state(model)
    if invalid_kind == "base":
        invalid["metadata"] = "invalid"
        expected = "non_tensor_keys"
    elif invalid_kind == "key":
        invalid["linear.bias_renamed"] = invalid.pop("linear.bias")
        expected = "missing_keys=1"
    else:
        invalid["linear.weight"] = torch.zeros(3, 2)
        expected = "tensor shape"

    with pytest.raises(RuntimeError, match=expected):
        apply_checkpoint(
            model,
            {"model": invalid},
            family=CheckpointFamily.SINGLE_MODEL,
        )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


@pytest.mark.parametrize(
    ("conflicting_key", "kind"),
    (("right.weight", "parameter"), ("second_buffer", "persistent buffer")),
)
def test_tied_state_alias_conflict_fails_before_model_write(
    conflicting_key: str,
    kind: str,
) -> None:
    model = _TiedStateModel()
    original = _state(model)
    state = _state(model)
    state[conflicting_key].add_(1)

    with pytest.raises(RuntimeError, match=rf"同一 {kind}.*alias 值不一致"):
        apply_checkpoint(
            model,
            {"model": state},
            family=CheckpointFamily.SINGLE_MODEL,
        )

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def _compact_dbgnet() -> nn.Module:
    from tasks.aec.models.discriminative.dbgnet import DBGNet

    return DBGNet(
        n_fft=128,
        hop_length=64,
        win_length=128,
        sample_rate=16000,
        erb_low_bins=17,
        erb_bands=16,
        mic_channels=[2, 4, 6, 8],
        far_channels=[1, 2, 3, 4],
        use_state_detector=False,
    )


def _assert_tensor_state_exact(
    model: nn.Module,
    expected: dict[str, torch.Tensor],
) -> None:
    actual = model.state_dict()
    assert set(actual) == set(expected)
    for key, value in actual.items():
        assert value.dtype == expected[key].dtype, key
        assert value.shape == expected[key].shape, key
        assert torch.equal(value, expected[key]), key


def test_dbgnet_erb_transpose_shared_storage_conflict_rolls_back() -> None:
    model = _compact_dbgnet()
    erb_weight = model.erb.erb_fc.weight
    ierb_weight = model.erb.ierb_fc.weight
    original = _state(model)
    original_storage = erb_weight.untyped_storage().data_ptr()
    assert original_storage == ierb_weight.untyped_storage().data_ptr()
    assert erb_weight.shape == ierb_weight.T.shape

    conflicting = _state(model)
    conflicting["erb.erb_fc.weight"].fill_(2.0)
    conflicting["erb.ierb_fc.weight"].fill_(3.0)

    with pytest.raises(RuntimeError, match="提交后 state_dict.*不一致"):
        apply_checkpoint(
            model,
            {"model": conflicting},
            family=CheckpointFamily.SINGLE_MODEL,
            context="DBGNet ERB shared-storage",
        )

    assert model.erb.erb_fc.weight is erb_weight
    assert model.erb.ierb_fc.weight is ierb_weight
    assert erb_weight.untyped_storage().data_ptr() == original_storage
    assert ierb_weight.untyped_storage().data_ptr() == original_storage
    _assert_tensor_state_exact(model, original)


def test_legal_dbgnet_state_with_transpose_shared_storage_loads() -> None:
    source = _compact_dbgnet()
    expected = _state(source)
    target = _compact_dbgnet()

    apply_checkpoint(
        target,
        {"model": expected},
        family=CheckpointFamily.SINGLE_MODEL,
        context="legal DBGNet checkpoint",
    )

    _assert_tensor_state_exact(target, expected)
    assert torch.equal(
        target.erb.erb_fc.weight,
        target.erb.ierb_fc.weight.T,
    )


def test_post_commit_exact_validation_accepts_same_position_nan() -> None:
    model = _TinyModel()
    state = _state(model)
    state["linear.weight"][0, 0] = torch.nan

    load_state_dict_checked(model, state)

    assert torch.isnan(model.linear.weight[0, 0])


def test_post_commit_dtype_mismatch_rolls_back() -> None:
    model = _TinyModel()
    original = _state(model)
    state = _state(model)
    state["linear.weight"] = state["linear.weight"].double()

    with pytest.raises(RuntimeError, match="提交后 state_dict.*不一致"):
        load_state_dict_checked(model, state, context="dtype mismatch")

    _assert_tensor_state_exact(model, original)


def test_load_state_dict_override_failure_restores_aliases_and_registries() -> None:
    model = _LoadOverrideFailure()
    left = model.left
    parameter = model.left.weight
    buffer = model.first_buffer
    original = _state(model)
    state = {key: torch.zeros_like(value) for key, value in original.items()}

    with pytest.raises(RuntimeError, match="public load failure"):
        load_state_dict_checked(model, state)

    assert model.left is left
    assert model.left.weight is parameter
    assert model.right.weight is parameter
    assert model.left.weight.requires_grad
    assert model.first_buffer is buffer
    assert model.second_buffer is buffer
    assert not model.first_buffer.requires_grad
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_load_from_state_dict_failure_restores_values_and_requires_grad() -> None:
    model = _LoadFromStateFailure()
    parameter = model.weight
    buffer = model.buffer
    original = _state(model)
    state = {key: torch.zeros_like(value) for key, value in original.items()}

    with pytest.raises(RuntimeError, match="private load failure"):
        load_state_dict_checked(model, state)

    assert model.weight is parameter
    assert model.buffer is buffer
    assert model.weight.requires_grad
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_set_extra_state_failure_restores_object_value_and_extra_state() -> None:
    model = _SetExtraStateFailure()
    parameter = model.weight
    state = dict(model.state_dict())
    state["weight"] = torch.zeros_like(model.weight)
    state["_extra_state"] = {"version": 2, "fail": True}

    with pytest.raises(RuntimeError, match="extra state failure"):
        load_state_dict_checked(model, state)

    assert model.weight is parameter
    assert model.weight.requires_grad
    torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
    assert model.extra == {"version": 1}


def test_zero_match_fail_closed_and_strict_param_removed() -> None:
    model = _TinyModel()
    with pytest.raises(RuntimeError, match="zero model parameters matched"):
        load_state_dict_checked(
            model,
            {"other.weight": torch.zeros(2, 2)},
        )
    with pytest.raises(TypeError):
        load_state_dict_checked(model, _state(model), strict=False)


def test_buffer_only_checkpoint_is_not_a_parameter_match() -> None:
    model = nn.Sequential(nn.BatchNorm1d(2))
    buffer_only = {
        "0.num_batches_tracked": model.state_dict()["0.num_batches_tracked"].clone()
    }

    with pytest.raises(RuntimeError, match="zero model parameters matched"):
        load_state_dict_checked(model, buffer_only, context="buffer-only")


def test_expected_extra_state_is_loaded_but_unexpected_extra_state_fails() -> None:
    source = _ExtraStateModel()
    state = dict(source.state_dict())
    target = _ExtraStateModel()

    load_state_dict_checked(target, state)
    assert target.loaded_extra_state == {"version": 1}

    plain = _TinyModel()
    with pytest.raises(RuntimeError, match="unexpected_keys=1"):
        load_state_dict_checked(plain, state)


def test_single_and_gan_ddp_checkpoints_load_identical_weights() -> None:
    source = _TinyModel()
    with torch.no_grad():
        source.linear.weight.fill_(2.0)
        source.linear.bias.fill_(3.0)
    expected = _state(source, prefix="module.")

    loaded_models = []
    for family, checkpoint in (
        (CheckpointFamily.SINGLE_MODEL, {"model": expected}),
        (CheckpointFamily.GAN_GENERATOR, {"model_generator": expected}),
    ):
        target = _TinyModel()
        apply_checkpoint(target, checkpoint, family=family)
        loaded_models.append(target)

    for key, value in loaded_models[0].state_dict().items():
        torch.testing.assert_close(value, loaded_models[1].state_dict()[key])
        torch.testing.assert_close(value, source.state_dict()[key])


def test_ema_is_applied_after_base_state(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _TinyModel()
    base_state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    ema_values = [torch.full_like(parameter, 7.0) for parameter in model.parameters()]

    class _FakeEMA:
        def __init__(self, parameters, decay):
            self.parameters = list(parameters)
            self.decay = decay
            self.shadow_params = []

        def load_state_dict(self, state):
            self.shadow_params = state["shadow_params"]

        def copy_to(self, parameters):
            for parameter, value in zip(parameters, self.shadow_params):
                parameter.data.copy_(value)

    module = types.ModuleType("torch_ema")
    module.ExponentialMovingAverage = _FakeEMA
    monkeypatch.setitem(sys.modules, "torch_ema", module)

    apply_checkpoint(
        model,
        {"model": base_state, "ema": {"shadow_params": ema_values}},
        family=CheckpointFamily.SINGLE_MODEL,
    )

    for parameter in model.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 7.0))


def test_ema_overrides_all_tied_parameter_state_dict_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TiedStateModel()
    base_state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    shadow = torch.full_like(model.left.weight, 7.0)

    class _FakeEMA:
        def __init__(self, parameters, decay):
            self.shadow_params = []

        def load_state_dict(self, state):
            self.shadow_params = state["shadow_params"]

    module = types.ModuleType("torch_ema")
    module.ExponentialMovingAverage = _FakeEMA
    monkeypatch.setitem(sys.modules, "torch_ema", module)

    apply_checkpoint(
        model,
        {"model": base_state, "ema": {"shadow_params": [shadow]}},
        family=CheckpointFamily.SINGLE_MODEL,
    )

    assert model.left.weight is model.right.weight
    torch.testing.assert_close(model.state_dict()["left.weight"], shadow)
    torch.testing.assert_close(model.state_dict()["right.weight"], shadow)


def test_public_ema_copy_failure_restores_object_value_and_requires_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    weight = model.linear.weight
    bias = model.linear.bias
    original = _state(model)
    shadow_params = [torch.full_like(parameter, 7.0) for parameter in model.parameters()]

    class _FailingEMA:
        def __init__(self, parameters, decay):
            self.shadow_params = []

        def load_state_dict(self, state):
            self.shadow_params = state["shadow_params"]

        def copy_to(self, parameters):
            parameters[0].data.copy_(self.shadow_params[0])
            parameters[0].requires_grad_(False)
            model.linear.weight = nn.Parameter(
                torch.full_like(parameters[0], 99.0), requires_grad=False
            )
            parameters[1].data.zero_()
            raise RuntimeError("copy failure")

    module = types.ModuleType("torch_ema")
    module.ExponentialMovingAverage = _FailingEMA
    monkeypatch.setitem(sys.modules, "torch_ema", module)

    with pytest.raises(RuntimeError, match="copy failure"):
        apply_ema_if_present({"ema": {"shadow_params": shadow_params}}, model)

    assert model.linear.weight is weight
    assert model.linear.bias is bias
    assert model.linear.weight.requires_grad
    assert model.linear.bias.requires_grad
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


@pytest.mark.parametrize("invalid_kind", ["count", "type", "dtype", "shape"])
def test_ema_shadow_params_must_match_before_any_model_write(
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    model = _TinyModel()
    original = _state(model)
    base_state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    parameters = list(model.parameters())
    shadow_params = [torch.full_like(parameter, 7.0) for parameter in parameters]
    if invalid_kind == "count":
        shadow_params.pop()
    elif invalid_kind == "type":
        shadow_params[0] = "not-a-tensor"
    elif invalid_kind == "dtype":
        shadow_params[0] = shadow_params[0].to(torch.float64)
    else:
        shadow_params[0] = torch.zeros(parameters[0].numel())

    class _FakeEMA:
        def __init__(self, parameters, decay):
            self.shadow_params = []

        def load_state_dict(self, state):
            self.shadow_params = state["shadow_params"]

        def copy_to(self, parameters):
            raise AssertionError("EMA 校验失败时不得提交参数")

    module = types.ModuleType("torch_ema")
    module.ExponentialMovingAverage = _FakeEMA
    monkeypatch.setitem(sys.modules, "torch_ema", module)

    with pytest.raises(RuntimeError, match="EMA shadow_params"):
        apply_checkpoint(
            model,
            {"model": base_state, "ema": {"shadow_params": shadow_params}},
            family=CheckpointFamily.SINGLE_MODEL,
        )

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_ema_state_parse_failure_happens_before_any_model_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    original = _state(model)
    base_state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    shadow_params = [torch.full_like(parameter, 7.0) for parameter in model.parameters()]

    class _RejectingEMA:
        def __init__(self, parameters, decay):
            self.shadow_params = []

        def load_state_dict(self, state):
            raise ValueError("invalid EMA state")

    module = types.ModuleType("torch_ema")
    module.ExponentialMovingAverage = _RejectingEMA
    monkeypatch.setitem(sys.modules, "torch_ema", module)

    with pytest.raises(RuntimeError, match="EMA 状态无法应用"):
        apply_checkpoint(
            model,
            {"model": base_state, "ema": {"shadow_params": shadow_params}},
            family=CheckpointFamily.SINGLE_MODEL,
        )

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_missing_ema_dependency_fails_before_any_model_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    original = _state(model)
    base_state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    shadow_params = [torch.full_like(parameter, 7.0) for parameter in model.parameters()]
    monkeypatch.setitem(sys.modules, "torch_ema", None)

    with pytest.raises(RuntimeError, match="缺少 torch_ema"):
        apply_checkpoint(
            model,
            {"model": base_state, "ema": {"shadow_params": shadow_params}},
            family=CheckpointFamily.SINGLE_MODEL,
        )

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_load_checkpoint_rejects_content_changed_after_prior_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from evaluate import checkpoint_loading

    checkpoint_path = tmp_path / "model.tar"
    checkpoint_path.write_bytes(b"fingerprinted-checkpoint")
    expected_fingerprint = fingerprint_file(checkpoint_path)
    source = _TinyModel()
    checkpoint = {"model": _state(source)}
    target = _TinyModel()
    original = _state(target)

    def _replace_during_load(path, **kwargs):
        checkpoint_path.write_bytes(b"replacement-checkpoint")
        return checkpoint

    monkeypatch.setattr(
        checkpoint_loading.safe_load_module,
        "safe_torch_load",
        _replace_during_load,
    )

    with pytest.raises(RuntimeError, match="加载期间内容发生变化"):
        load_checkpoint(
            target,
            checkpoint_path,
            family=CheckpointFamily.SINGLE_MODEL,
            expected_fingerprint=expected_fingerprint,
        )

    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_load_checkpoint_rejects_atomic_path_replacement_after_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from evaluate import checkpoint_loading

    checkpoint_path = tmp_path / "model.tar"
    replacement_path = tmp_path / "replacement.tar"
    checkpoint_path.write_bytes(b"original-checkpoint")
    expected_fingerprint = fingerprint_file(checkpoint_path)
    source = _TinyModel()
    checkpoint = {"model": _state(source)}
    target = _TinyModel()
    original = _state(target)

    def _replace_during_load(_stream, **_kwargs):
        replacement_path.write_bytes(b"replacement-checkpoint")
        replacement_path.replace(checkpoint_path)
        return checkpoint

    monkeypatch.setattr(
        checkpoint_loading.safe_load_module,
        "safe_torch_load",
        _replace_during_load,
    )

    with pytest.raises(RuntimeError, match="文件身份被替换"):
        load_checkpoint(
            target,
            checkpoint_path,
            family=CheckpointFamily.SINGLE_MODEL,
            expected_fingerprint=expected_fingerprint,
        )

    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, original[key])
