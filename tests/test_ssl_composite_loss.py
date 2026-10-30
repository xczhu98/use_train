"""SSLDOACompositeLoss 的数值、PIT、trainer 与分布式合同验收。"""

from datetime import timedelta
import math

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from core.models.io_contract import ModelIOContract, TensorField
from tasks.ssl.data.batch import SSLBatch
from tasks.ssl.losses import SSLDOACompositeLoss, SSLDPIPDPITLoss
from tasks.ssl.model_io import (
    validate_ssl_loss_output_contract,
    validate_ssl_model_output,
)
from tasks.ssl.trainer.ssl_trainer import SSLTrainer
import tasks.ssl.losses as ssl_losses
from conftest import spawn_ddp_workers


pytestmark = pytest.mark.unit

_NUMERIC_DEVICES = (
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=pytest.mark.gpu),
)


_DPIPD_FIELD = TensorField(
    "dpipd",
    "spectrum",
    ("batch", "frame", "frequency", "track", "pair", "complex"),
    multiple=True,
)
_DOA_FIELD = TensorField(
    "doa_vector", "label", ("batch", "frame", "track", "angle")
)
_VAD_FIELD = TensorField("vad_logits", "logits", ("batch", "frame", "track"))
_RANGE_FIELD = TensorField("range", "label", ("batch", "frame", "track"))


class _AuxiliaryHeadFixture(torch.nn.Module):
    """只用于 trainer 合同验收的辅助头模型，不代表库存 SSL 模型能力。"""

    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"ssl"}),
        inputs=(
            TensorField(
                "features",
                "embedding",
                ("batch", "frame", "frequency", "feature"),
            ),
        ),
        outputs=(_DPIPD_FIELD, _DOA_FIELD, _VAD_FIELD, _RANGE_FIELD),
        forward_kind="estimate",
        frontend_owner="model",
    )

    def __init__(self, *, emit_auxiliary=True, invalid_range=False):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.2))
        self.doa_angles = torch.nn.Parameter(torch.tensor([1.2, 0.3]))
        self.vad_logit = torch.nn.Parameter(torch.tensor(0.1))
        self.range_raw = torch.nn.Parameter(torch.tensor(math.log(math.expm1(2.0))))
        self.emit_auxiliary = emit_auxiliary
        self.invalid_range = invalid_range

    def forward(self, features):
        batch, frames = features.shape[:2]
        dpipd = features.unsqueeze(-2).unsqueeze(-2) * self.gain
        if not self.emit_auxiliary:
            return {"dpipd": dpipd}
        range_output = F.softplus(self.range_raw).expand(batch, frames, 1)
        if self.invalid_range:
            range_output = torch.zeros_like(range_output)
        return {
            "dpipd": dpipd,
            "doa_vector": self.doa_angles.view(1, 1, 1, 2).expand(
                batch, frames, 1, 2
            ),
            "vad_logits": self.vad_logit.expand(batch, frames, 1),
            "range": range_output,
        }


class _DPIPDOnlyContractFixture(torch.nn.Module):
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"ssl"}),
        inputs=(
            TensorField(
                "features",
                "embedding",
                ("batch", "frame", "frequency", "feature"),
            ),
        ),
        outputs=(_DPIPD_FIELD,),
        forward_kind="estimate",
        frontend_owner="model",
    )

    def forward(self, features):
        del features
        raise NotImplementedError


def _loss_case(*, tracks=1, frames=1, near_field=False):
    prediction = torch.zeros(1, frames, 1, tracks, 1, 2)
    target = torch.zeros_like(prediction)
    kwargs = {
        "target_dpipd": target,
        "input_lengths": torch.tensor([frames], dtype=torch.int64),
        "output_lengths": torch.tensor([frames], dtype=torch.int64),
        "pair_index": torch.tensor([[[0, 1]]], dtype=torch.int64),
        "pair_mask": torch.ones(1, 1, dtype=torch.bool),
        "geometry": torch.tensor([[[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]]),
        "frame_origin": torch.zeros(1, dtype=torch.int64),
        "time_stride": 1,
        "input_representation": "reference_ipd",
        "target_representation": (
            "dpipd_near_field" if near_field else "dpipd_far_field"
        ),
        "coordinate_frame": "array_local",
        "doa": torch.zeros(1, frames, tracks, 2),
        "vad": torch.ones(1, frames, tracks),
    }
    if near_field:
        kwargs["source_distances"] = torch.ones(1, frames, tracks)
    return prediction, kwargs


def _loss_case_on_device(*, device, tracks=1, frames=1, near_field=False):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SSL composite numeric acceptance")
    prediction, kwargs = _loss_case(
        tracks=tracks, frames=frames, near_field=near_field
    )
    return prediction.to(device), {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in kwargs.items()
    }


def _ssl_batch(*, near_field=False):
    frames = 2
    batch = SSLBatch(
        features=torch.ones(1, frames, 1, 2),
        target_dpipd=torch.zeros(1, frames, 1, 1, 1, 2),
        input_lengths=torch.tensor([frames]),
        output_lengths=torch.tensor([frames]),
        pair_index=torch.tensor([[[0, 1]]], dtype=torch.int64),
        pair_mask=torch.ones(1, 1, dtype=torch.bool),
        geometry=torch.tensor([[[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]]),
        frame_origin=torch.tensor([0], dtype=torch.int64),
        time_stride=1,
        valid_output_mask=torch.ones(1, frames, dtype=torch.bool),
        doa=torch.tensor([[[[math.pi / 2, 0.2]], [[math.pi / 2, 0.2]]]]),
        vad=torch.ones(1, frames, 1),
        sample_rates=torch.tensor([8000]),
        provenance=({"backend": "composite-loss-fixture"},),
        source_distances=(torch.ones(1, frames, 1) if near_field else None),
        input_representation="reference_ipd",
        target_representation=(
            "dpipd_near_field" if near_field else "dpipd_far_field"
        ),
        coordinate_frame="array_local",
    )
    batch.validate()
    return batch


def _trainer(model, loss, *, rank=0, world_size=1, device="cpu"):
    trainer = object.__new__(SSLTrainer)
    trainer.model = model
    trainer.loss_fn = loss
    trainer.device = torch.device(device)
    trainer.sample_rate = 8000
    trainer.rank = rank
    trainer.world_size = world_size
    trainer._nccl_flow_log = lambda _phase: None
    return trainer


@pytest.mark.parametrize("device", _NUMERIC_DEVICES)
def test_dpipd_only_composite_is_bitwise_equal_to_existing_loss(device):
    torch.manual_seed(91)
    prediction, kwargs = _loss_case_on_device(device=device, tracks=2, frames=3)
    prediction.normal_()
    kwargs["target_dpipd"].normal_()
    kwargs["output_lengths"] = torch.tensor([2], device=device)
    kwargs["vad"][0, 1, 1] = 0.0
    kwargs["pair_mask"] = torch.tensor([[True]], device=device)
    standard = SSLDPIPDPITLoss(max_tracks=2)(prediction, **kwargs)
    composite = SSLDOACompositeLoss(max_tracks=2)(
        {"dpipd": prediction}, **kwargs
    )
    assert torch.equal(composite, standard)


@pytest.mark.parametrize("device", _NUMERIC_DEVICES)
def test_dpipd_weight_matches_hand_computed_complex_mse(device):
    prediction, kwargs = _loss_case_on_device(device=device)
    kwargs["target_dpipd"][..., 0] = 1.0
    value = SSLDOACompositeLoss(dpipd_weight=2.0, max_tracks=1)(
        {"dpipd": prediction}, **kwargs
    )
    torch.testing.assert_close(value, torch.tensor(2.0, device=device), rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("device", _NUMERIC_DEVICES)
def test_doa_classification_matches_hand_computed_cross_entropy(device):
    prediction, kwargs = _loss_case_on_device(device=device)
    kwargs["doa"] = torch.tensor([[[[math.pi / 2, 0.0]]]], device=device)
    bins = [[math.pi / 2, 0.0], [math.pi / 2, math.pi / 2]]
    logits = torch.tensor([[[[math.log(3.0), 0.0]]]], device=device)
    value = SSLDOACompositeLoss(
        dpipd_weight=0.0,
        doa_classification_weight=1.0,
        doa_class_bins=bins,
        max_tracks=1,
    )({"dpipd": prediction, "doa_logits": logits}, **kwargs)
    torch.testing.assert_close(
        value,
        torch.tensor(math.log(4.0 / 3.0), device=device),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("device", _NUMERIC_DEVICES)
def test_doa_regression_matches_spherical_two_angle_reference(device):
    prediction, kwargs = _loss_case_on_device(device=device)
    kwargs["doa"] = torch.tensor([[[[math.pi / 2, 0.0]]]], device=device)
    doa_vector = torch.tensor([[[[math.pi / 2, math.pi / 2]]]], device=device)
    value = SSLDOACompositeLoss(
        dpipd_weight=0.0, doa_regression_weight=1.0, max_tracks=1
    )({"dpipd": prediction, "doa_vector": doa_vector}, **kwargs)
    # 单位向量正交，弦距离平方为 2；三个笛卡尔分量由 loss 取均值。
    torch.testing.assert_close(
        value, torch.tensor(2.0 / 3.0, device=device), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("device", _NUMERIC_DEVICES)
def test_vad_bce_uses_soft_target_reference_value(device):
    prediction, kwargs = _loss_case_on_device(device=device)
    kwargs["vad"] = torch.tensor([[[0.25]]], device=device)
    logits = torch.zeros(1, 1, 1, device=device)
    value = SSLDOACompositeLoss(
        dpipd_weight=0.0, vad_weight=2.0, max_tracks=1
    )({"dpipd": prediction, "vad_logits": logits}, **kwargs)
    torch.testing.assert_close(
        value, torch.tensor(2.0 * math.log(2.0), device=device), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("device", _NUMERIC_DEVICES)
def test_near_field_range_uses_log_distance_smooth_l1_reference(device):
    prediction, kwargs = _loss_case_on_device(device=device, near_field=True)
    kwargs["source_distances"].fill_(1.0)
    range_output = torch.full((1, 1, 1), 2.0, device=device)
    value = SSLDOACompositeLoss(
        dpipd_weight=0.0, range_weight=2.0, max_tracks=1
    )({"dpipd": prediction, "range": range_output}, **kwargs)
    expected = torch.tensor(math.log(2.0) ** 2, device=device)
    torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("device", _NUMERIC_DEVICES)
def test_k3_auxiliary_heads_reuse_the_dpipd_pit_assignment_once(monkeypatch, device):
    tracks = 3
    prediction, kwargs = _loss_case_on_device(
        device=device, tracks=tracks, near_field=True
    )
    target = kwargs["target_dpipd"]
    source_values = torch.tensor([0.0, 1.0, 3.0], device=device)
    target[0, 0, 0, :, 0, 0] = source_values
    permutation = torch.tensor([2, 0, 1], device=device)
    prediction = target[:, :, :, permutation].clone()
    target_angles = torch.tensor(
        [[[[math.pi / 2, 0.0], [math.pi / 2, 1.0], [math.pi / 3, -0.7]]]],
        device=device,
    )
    kwargs["doa"] = target_angles
    kwargs["source_distances"] = torch.tensor(
        [[[1.0, math.exp(0.1), math.exp(0.2)]]], device=device
    )
    # 中间目标声源不活动；有效声源仍唯一确定两个活动槽位的匹配。
    kwargs["vad"] = torch.tensor([[[1.0, 0.0, 1.0]]], device=device)
    doa_output = target_angles[:, :, permutation].clone()
    # 该 range 输出单独偏好恒等匹配，而 DP-IPD 唯一选择 [2,0,1]；
    # 复用 DP-IPD 指派后的 log-Huber range loss 为 0.01。
    range_output = kwargs["source_distances"].clone()

    calls = {"assignment": 0, "hungarian": 0}
    selected_assignment = {}
    original_assignment = ssl_losses._batched_min_assignment
    original_hungarian = ssl_losses._hungarian_min_assignment

    def count_assignment(cost):
        calls["assignment"] += 1
        assignment = original_assignment(cost)
        selected_assignment["value"] = assignment.detach().clone()
        return assignment

    def count_hungarian(cost):
        calls["hungarian"] += 1
        return original_hungarian(cost)

    monkeypatch.setattr(ssl_losses, "_batched_min_assignment", count_assignment)
    monkeypatch.setattr(ssl_losses, "_hungarian_min_assignment", count_hungarian)
    value = SSLDOACompositeLoss(
        dpipd_weight=0.0,
        doa_regression_weight=1.0,
        vad_weight=1.0,
        range_weight=1.0,
        max_tracks=tracks,
    )(
        {
            "dpipd": prediction,
            "doa_vector": doa_output,
            "vad_logits": torch.tensor([[[2.0, 2.0, -2.0]]], device=device),
            "range": range_output,
        },
        **kwargs,
    )
    expected = torch.nn.functional.softplus(torch.tensor(-2.0, device=device)) + 0.0125
    torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6)
    assert calls == {"assignment": 1, "hungarian": 1}
    assert torch.equal(selected_assignment["value"], permutation.view(1, 1, tracks))


@pytest.mark.parametrize("device", _NUMERIC_DEVICES)
@pytest.mark.parametrize("inactive_vad", (0.0, 0.5))
@pytest.mark.parametrize("doa_kind", ("classification", "regression"))
def test_multiframe_auxiliary_masks_match_independent_values_and_gradients(
    device, inactive_vad, doa_kind
):
    prediction, kwargs = _loss_case_on_device(
        device=device, frames=3, near_field=True
    )
    kwargs["output_lengths"] = torch.tensor([2], device=device)
    kwargs["doa"] = torch.tensor(
        [[[[math.pi / 2, 0.0]], [[math.pi / 2, math.pi / 2]], [[1.1, -0.4]]]],
        device=device,
    )
    kwargs["vad"] = torch.tensor([[[1.0], [inactive_vad], [0.25]]], device=device)
    kwargs["source_distances"] = torch.tensor(
        [[[1.0], [1.5], [0.0]]], device=device
    )

    outputs = {
        "dpipd": prediction,
        "vad_logits": torch.tensor([[[0.0], [2.0], [8.0]]], device=device, requires_grad=True),
        "range": torch.tensor([[[2.0], [3.0], [0.0]]], device=device, requires_grad=True),
    }
    if doa_kind == "classification":
        outputs["doa_logits"] = torch.tensor(
            [[[[math.log(3.0), 0.0]], [[0.0, math.log(5.0)]], [[10.0, -10.0]]]],
            device=device,
            requires_grad=True,
        )
        loss = SSLDOACompositeLoss(
            dpipd_weight=0.0,
            doa_classification_weight=1.0,
            doa_class_bins=[[math.pi / 2, 0.0], [math.pi / 2, math.pi / 2]],
            vad_weight=1.0,
            range_weight=1.0,
            max_tracks=1,
        )
        expected_doa = math.log(4.0 / 3.0)
        doa_output_name = "doa_logits"
    else:
        outputs["doa_vector"] = torch.tensor(
            [[[[math.pi / 2, math.pi / 2]], [[0.1, 2.0]], [[2.0, 2.0]]]],
            device=device,
            requires_grad=True,
        )
        loss = SSLDOACompositeLoss(
            dpipd_weight=0.0,
            doa_regression_weight=1.0,
            vad_weight=1.0,
            range_weight=1.0,
            max_tracks=1,
        )
        expected_doa = 2.0 / 3.0
        doa_output_name = "doa_vector"

    value = loss(outputs, **kwargs)
    expected_vad = (
        math.log(2.0)
        + torch.nn.functional.softplus(torch.tensor(2.0, device=device))
        - 2.0 * inactive_vad
    ) / 2.0
    expected_range = 0.5 * math.log(2.0) ** 2
    expected = torch.tensor(expected_doa + expected_range, device=device) + expected_vad
    torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-7)

    value.backward()
    doa_grad = outputs[doa_output_name].grad
    range_grad = outputs["range"].grad
    vad_grad = outputs["vad_logits"].grad
    assert doa_grad is not None and torch.isfinite(doa_grad).all()
    assert doa_grad[0, 0, 0].abs().sum() > 0.0
    assert torch.equal(doa_grad[0, 1:], torch.zeros_like(doa_grad[0, 1:]))
    assert range_grad is not None and torch.isfinite(range_grad).all()
    assert range_grad[0, 0, 0].abs() > 0.0
    assert torch.equal(range_grad[0, 1:], torch.zeros_like(range_grad[0, 1:]))
    assert vad_grad is not None and torch.isfinite(vad_grad).all()
    assert vad_grad[0, 0, 0].abs() > 0.0
    assert vad_grad[0, 1, 0].abs() > 0.0
    assert vad_grad[0, 2, 0] == 0.0


@pytest.mark.parametrize("device", _NUMERIC_DEVICES)
def test_range_padding_and_inactive_sources_are_safe_with_finite_gradients(device):
    prediction, kwargs = _loss_case_on_device(
        device=device, frames=2, near_field=True
    )
    kwargs["input_lengths"] = torch.tensor([2], device=device)
    kwargs["output_lengths"] = torch.tensor([1], device=device)
    kwargs["vad"] = torch.tensor([[[1.0], [0.0]]], device=device)
    kwargs["source_distances"] = torch.tensor([[[1.0], [0.0]]], device=device)
    range_output = torch.tensor([[[2.0], [0.0]]], device=device, requires_grad=True)
    value = SSLDOACompositeLoss(
        dpipd_weight=0.0, range_weight=1.0, max_tracks=1
    )({"dpipd": prediction, "range": range_output}, **kwargs)
    torch.testing.assert_close(
        value, torch.tensor(0.5 * math.log(2.0) ** 2, device=device), rtol=1e-6, atol=1e-7
    )
    value.backward()
    assert range_output.grad is not None
    assert torch.isfinite(range_output.grad).all()
    assert range_output.grad[0, 1, 0] == 0.0

    kwargs["vad"] = torch.zeros_like(kwargs["vad"])
    all_inactive = torch.tensor(
        [[[2.0], [0.0]]], device=device, requires_grad=True
    )
    zero = SSLDOACompositeLoss(
        dpipd_weight=0.0, range_weight=1.0, max_tracks=1
    )({"dpipd": prediction, "range": all_inactive}, **kwargs)
    assert torch.isfinite(zero)
    torch.testing.assert_close(zero, torch.zeros_like(zero))
    zero.backward()
    assert torch.isfinite(all_inactive.grad).all()


def test_pair_mask_cannot_remove_every_pair():
    prediction, kwargs = _loss_case()
    kwargs["pair_mask"] = torch.zeros(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one pair"):
        SSLDOACompositeLoss(max_tracks=1)({"dpipd": prediction}, **kwargs)


@pytest.mark.parametrize(
    "args, error",
    [
        ({"dpipd_weight": 0.0}, "at least one positive"),
        ({"vad_weight": -1.0}, "non-negative"),
        ({"vad_weight": math.inf}, "finite"),
        ({"vad_weight": True}, "real number"),
        ({"max_tracks": None}, "positive integer"),
        (
            {
                "doa_classification_weight": 1.0,
                "doa_regression_weight": 1.0,
                "doa_class_bins": [[math.pi / 2, 0.0], [math.pi / 2, 1.0]],
            },
            "mutually exclusive",
        ),
        ({"doa_classification_weight": 1.0}, "requires explicit doa_class_bins"),
        (
            {"doa_class_bins": [[math.pi / 2, 0.0], [math.pi / 2, 1.0]]},
            "only valid",
        ),
        (
            {
                "doa_classification_weight": 1.0,
                "doa_class_bins": [[True, 0.0], [math.pi / 2, 1.0]],
            },
            "boolean",
        ),
        (
            {
                "doa_classification_weight": 1.0,
                "doa_class_bins": [[math.pi / 2, -math.pi], [math.pi / 2, math.pi]],
            },
            "distinct spherical directions",
        ),
    ],
)
def test_invalid_composite_weights_tracks_and_bins_fail_fast(args, error):
    with pytest.raises((TypeError, ValueError), match=error):
        SSLDOACompositeLoss(**args)


@pytest.mark.parametrize(
    "weight_name, output_name, extra_args",
    [
        ("doa_classification_weight", "doa_logits", {"doa_class_bins": [[1.0, 0.0], [1.0, 1.0]]}),
        ("doa_regression_weight", "doa_vector", {}),
        ("vad_weight", "vad_logits", {}),
        ("range_weight", "range", {}),
    ],
)
def test_enabled_auxiliary_head_and_target_are_required(
    weight_name, output_name, extra_args
):
    prediction, kwargs = _loss_case(near_field=weight_name == "range_weight")
    loss = SSLDOACompositeLoss(
        dpipd_weight=0.0,
        max_tracks=1,
        **{weight_name: 1.0},
        **extra_args,
    )
    with pytest.raises(ValueError, match=output_name):
        loss({"dpipd": prediction}, **kwargs)

    target_name = {
        "doa_classification_weight": "doa",
        "doa_regression_weight": "doa",
        "vad_weight": "vad",
        "range_weight": "source_distances",
    }[weight_name]
    outputs = {
        "doa_logits": torch.zeros(1, 1, 1, 2),
        "doa_vector": torch.zeros(1, 1, 1, 2),
        "vad_logits": torch.zeros(1, 1, 1),
        "range": torch.ones(1, 1, 1),
    }
    if weight_name == "range_weight":
        kwargs["target_representation"] = "dpipd_near_field"
        kwargs["source_distances"] = torch.ones(1, 1, 1)
    kwargs[target_name] = None
    with pytest.raises(ValueError, match=target_name):
        loss({"dpipd": prediction, output_name: outputs[output_name]}, **kwargs)


def test_auxiliary_class_shape_and_output_shape_are_strict():
    prediction, kwargs = _loss_case()
    classification = SSLDOACompositeLoss(
        dpipd_weight=0.0,
        doa_classification_weight=1.0,
        doa_class_bins=[[math.pi / 2, 0.0], [math.pi / 2, 1.0]],
        max_tracks=1,
    )
    with pytest.raises(ValueError, match="class dimension"):
        classification(
            {"dpipd": prediction, "doa_logits": torch.zeros(1, 1, 1, 3)},
            **kwargs,
        )
    regression = SSLDOACompositeLoss(
        dpipd_weight=0.0, doa_regression_weight=1.0, max_tracks=1
    )
    with pytest.raises(ValueError, match="shape"):
        regression(
            {"dpipd": prediction, "doa_vector": torch.zeros(1, 1, 2, 2)},
            **kwargs,
        )


def test_model_contract_rejects_enabled_head_that_model_does_not_declare():
    model = _DPIPDOnlyContractFixture()
    with pytest.raises(ValueError, match="does not declare output head 'doa_vector'"):
        validate_ssl_loss_output_contract(
            model,
            {
                "type": "SSLDOACompositeLoss",
                "args": {"doa_regression_weight": 1.0},
            },
            target_representation="dpipd_far_field",
        )
    with pytest.raises(TypeError, match="args must be a mapping"):
        validate_ssl_loss_output_contract(
            model,
            {"type": "SSLDOACompositeLoss", "args": []},
            target_representation="dpipd_far_field",
        )


def test_model_output_preflight_checks_expected_device_and_declared_heads():
    model = _DPIPDOnlyContractFixture()
    output = torch.zeros(1, 1, 1, 1, 1, 2)
    with pytest.raises(ValueError, match="current feature device"):
        validate_ssl_model_output(
            model,
            output,
            expected_shape=output.shape,
            expected_device="meta",
        )


def test_composite_trainer_fixture_runs_train_backward_and_validation():
    model = _AuxiliaryHeadFixture()
    loss = SSLDOACompositeLoss(
        dpipd_weight=1.0,
        doa_regression_weight=0.5,
        vad_weight=0.5,
        range_weight=0.5,
        max_tracks=1,
    )
    batch = _ssl_batch(near_field=True)
    trainer = _trainer(model, loss)
    train_loss, finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, 8000)
    assert finite and torch.isfinite(train_loss)
    train_loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    prepared = trainer._val_prepare_batch(batch, 0, 1)
    with torch.inference_mode():
        val_loss = trainer._val_forward_prepared(prepared, 0, 1)[0]
    assert torch.isfinite(val_loss)


def _distributed_worker(rank, rendezvous, backend):
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        model = DistributedDataParallel(
            _AuxiliaryHeadFixture().to(device),
            device_ids=[rank] if backend == "nccl" else None,
        )
        loss_fn = SSLDOACompositeLoss(
            dpipd_weight=1.0,
            doa_regression_weight=0.5,
            vad_weight=0.5,
            range_weight=0.5,
            max_tracks=1,
        )
        trainer = _trainer(
            model, loss_fn, rank=rank, world_size=2, device=device
        )
        batch = _ssl_batch(near_field=True).to(device)
        if rank == 1:
            # 此 rank 无活动源；仍执行 loss reduction、DDP backward 与验证。
            batch.vad.zero_()
        train_loss, finite, _ = trainer._forward_and_compute_loss(
            batch, 0, 0, 8000
        )
        assert finite and torch.isfinite(train_loss)
        train_loss.backward()
        for parameter in model.module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

        prepared = trainer._val_prepare_batch(batch, 0, 1)
        with torch.inference_mode():
            val_loss = trainer._val_forward_prepared(prepared, 0, 1)[0]
        assert torch.isfinite(val_loss)

        bad_cases = (
            ("train", _AuxiliaryHeadFixture(emit_auxiliary=(rank == 0))),
            ("validation", _AuxiliaryHeadFixture(emit_auxiliary=(rank == 0))),
            ("train", _AuxiliaryHeadFixture(invalid_range=(rank == 1))),
            ("validation", _AuxiliaryHeadFixture(invalid_range=(rank == 1))),
        )
        for phase, bad_model in bad_cases:
            bad_model = DistributedDataParallel(
                bad_model.to(device),
                device_ids=[rank] if backend == "nccl" else None,
            )
            bad_trainer = _trainer(
                bad_model, loss_fn, rank=rank, world_size=2, device=device
            )
            with pytest.raises(RuntimeError, match="rank 1: ValueError"):
                if phase == "train":
                    bad_trainer._forward_and_compute_loss(batch, 0, 0, 8000)
                else:
                    bad_prepared = bad_trainer._val_prepare_batch(batch, 0, 1)
                    with torch.inference_mode():
                        bad_trainer._val_forward_prepared(bad_prepared, 0, 1)
    finally:
        dist.destroy_process_group()


@pytest.mark.ddp
@pytest.mark.subprocess
def test_ssl_composite_two_rank_gloo_loss_and_preflight_sync(tmp_path):
    spawn_ddp_workers(
        _distributed_worker,
        args=((tmp_path / "gloo").as_uri(), "gloo"),
        nprocs=2,
    )


@pytest.mark.gpu
def test_ssl_composite_cuda_auxiliary_train_and_validation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SSL composite GPU acceptance")
    device = torch.device("cuda", torch.cuda.current_device())
    model = _AuxiliaryHeadFixture().to(device)
    loss = SSLDOACompositeLoss(
        dpipd_weight=1.0,
        doa_regression_weight=0.5,
        vad_weight=0.5,
        range_weight=0.5,
        max_tracks=1,
    )
    trainer = _trainer(model, loss, device=device)
    batch = _ssl_batch(near_field=True).to(device)
    train_loss, finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, 8000)
    assert finite and train_loss.device == device
    train_loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    prepared = trainer._val_prepare_batch(batch, 0, 1)
    with torch.inference_mode():
        val_loss = trainer._val_forward_prepared(prepared, 0, 1)[0]
    assert torch.isfinite(val_loss) and val_loss.device == device


@pytest.mark.gpu
@pytest.mark.ddp
@pytest.mark.subprocess
def test_ssl_composite_two_rank_nccl_loss_and_preflight_sync(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("two visible CUDA devices are required for NCCL acceptance")
    spawn_ddp_workers(
        _distributed_worker,
        args=((tmp_path / "nccl").as_uri(), "nccl"),
        nprocs=2,
    )
