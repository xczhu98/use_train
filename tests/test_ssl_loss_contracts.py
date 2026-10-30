"""SSL loss 门禁：finite、长度、pair、近场与 DOA 输入合同。"""

import pytest
import torch

from tasks.ssl.losses import SSLDPIPDPITLoss, SSLSpatialMSELoss, ipd_mse_loss


pytestmark = pytest.mark.unit


def _tensors(*, tracks=2, frames=2, freqs=3, pairs=1):
    prediction = torch.zeros(1, frames, freqs, tracks, pairs, 2)
    target = torch.zeros_like(prediction)
    return prediction, target


def _metadata(*, tracks=2, pairs=1):
    pair_index = torch.tensor([[[0, 1]]], dtype=torch.int64)
    geometry = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]], dtype=torch.float32
    )
    return {
        "pair_index": pair_index,
        "pair_mask": torch.ones(1, pairs, dtype=torch.bool),
        "geometry": geometry,
        "frame_origin": torch.zeros(1, dtype=torch.int64),
        "time_stride": 1,
        "input_representation": "reference_ipd",
        "target_representation": "dpipd_far_field",
        "coordinate_frame": "array_local",
    }


@pytest.mark.parametrize("which", ["prediction", "target"])
def test_ssl_pit_rejects_nonfinite_dpipd(which):
    prediction, target = _tensors(tracks=1)
    (prediction if which == "prediction" else target)[0, 0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        SSLDPIPDPITLoss(max_tracks=1)(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
        )


def test_ssl_pit_rejects_nonfinite_pairwise_error_after_finite_input_check():
    prediction, target = _tensors(tracks=1)
    # Large finite inputs overflow only when the squared complex error is formed.
    prediction.fill_(1e20)
    with pytest.raises(ValueError, match="pairwise MSE"):
        SSLDPIPDPITLoss(max_tracks=1)(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
        )


def test_ssl_plain_mse_rejects_finite_inputs_that_overflow_the_squared_error():
    prediction, target = _tensors(tracks=1)
    prediction.fill_(1e20)
    with pytest.raises(ValueError, match="DPIPD MSE"):
        ipd_mse_loss(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
        )


def test_ssl_pit_rejects_finite_inputs_that_overflow_assignment_reduction():
    prediction, target = _tensors(tracks=2, freqs=1)
    prediction.fill_(1e19)
    with pytest.raises(ValueError, match="reduction must be finite"):
        SSLDPIPDPITLoss(max_tracks=2)(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
        )


def test_ssl_hungarian_keeps_large_finite_active_costs_as_a_permutation():
    """有限的大代价不能触发伪 inactive 分支并破坏 K>=3 指派。"""
    values = torch.tensor([0.0, 1.0e16, 2.0e16])
    target = torch.zeros(1, 1, 1, 3, 1, 2)
    target[0, 0, 0, :, 0, 0] = values
    prediction = target[:, :, :, [2, 0, 1], :, :].clone()

    loss = SSLDPIPDPITLoss(max_tracks=3)(
        prediction,
        target,
        output_lengths=torch.tensor([1]),
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-5, rtol=0.0)


def test_ssl_plain_mse_rejects_finite_inputs_that_overflow_reduction():
    prediction, target = _tensors(tracks=1, freqs=2)
    prediction.fill_(1e19)
    with pytest.raises(ValueError, match="reduction must be finite"):
        ipd_mse_loss(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
        )


def test_ssl_plain_mse_all_inactive_frames_return_graph_connected_zero():
    prediction, target = _tensors(tracks=1, freqs=2)
    prediction.fill_(1e19)
    prediction.requires_grad_()
    value = ipd_mse_loss(
        prediction,
        target,
        output_lengths=torch.tensor([2]),
        vad=torch.zeros(1, 2, 1),
    )
    assert torch.isfinite(value)
    torch.testing.assert_close(value, torch.zeros_like(value))
    value.backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad, torch.zeros_like(prediction.grad))


def test_ssl_spatial_mse_forwards_canonical_output_lengths_once():
    prediction, target = _tensors(tracks=1)
    value = SSLSpatialMSELoss(allow_no_stft=True)(
        prediction,
        target_dpipd=target,
        output_lengths=torch.tensor([2]),
    )
    assert torch.isfinite(value)


@pytest.mark.parametrize(
    "shape",
    [(0, 2, 3, 1, 1, 2), (1, 0, 3, 1, 1, 2), (1, 2, 0, 1, 1, 2)],
)
def test_ssl_loss_rejects_empty_canonical_axes(shape):
    prediction = torch.zeros(shape)
    with pytest.raises(ValueError, match="at least one"):
        SSLDPIPDPITLoss(max_tracks=1)(prediction, prediction)


def test_ssl_pit_rejects_nonfinite_doa_prediction():
    prediction, target = _tensors(tracks=1)
    doa = torch.zeros(1, 2, 1, 2)
    with pytest.raises(ValueError, match="DOA prediction"):
        SSLDPIPDPITLoss(max_tracks=1, doa_loss_weight=1.0)(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
            doa=doa,
            vad=torch.ones(1, 2, 1),
            doa_prediction=torch.full_like(doa, float("nan")),
        )


def test_ssl_loss_rejects_all_masked_pairs():
    prediction, target = _tensors(tracks=1)
    pair_mask = torch.zeros(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one pair"):
        SSLDPIPDPITLoss(max_tracks=1)(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
            pair_mask=pair_mask,
        )
    with pytest.raises(ValueError, match="at least one pair"):
        ipd_mse_loss(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
            pair_mask=pair_mask,
        )


@pytest.mark.parametrize(
    "lengths, error",
    [(torch.tensor([0]), "positive"), (torch.tensor([2.0]), "integer")],
)
def test_ssl_loss_requires_strict_output_lengths(lengths, error):
    prediction, target = _tensors(tracks=1)
    with pytest.raises((ValueError, TypeError), match=error):
        SSLDPIPDPITLoss(max_tracks=1)(prediction, target, output_lengths=lengths)


def test_ssl_loss_does_not_use_input_lengths_as_output_mask():
    prediction, target = _tensors(tracks=1, frames=2)
    with pytest.raises(ValueError, match="requires output_lengths"):
        SSLDPIPDPITLoss(max_tracks=1)(
            prediction,
            target,
            input_lengths=torch.tensor([1]),
        )


def test_ssl_loss_enforces_near_field_distance_contract():
    prediction, target = _tensors(tracks=1)
    metadata = _metadata(tracks=1)
    near_metadata = {**metadata, "target_representation": "dpipd_near_field"}
    with pytest.raises(ValueError, match="source_distances"):
        SSLDPIPDPITLoss(max_tracks=1)(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
            **near_metadata,
        )
    with pytest.raises(ValueError, match="dpipd_near_field"):
        SSLDPIPDPITLoss(max_tracks=1)(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
            **metadata,
            source_distances=torch.ones(1, 2, 1),
        )


def test_ssl_loss_allows_zero_source_distance_only_on_padded_frames():
    prediction, target = _tensors(tracks=1)
    metadata = {**_metadata(tracks=1), "target_representation": "dpipd_near_field"}
    value = SSLDPIPDPITLoss(max_tracks=1)(
        prediction,
        target,
        output_lengths=torch.tensor([1]),
        source_distances=torch.tensor([[[1.0], [0.0]]]),
        **metadata,
    )
    torch.testing.assert_close(value, torch.zeros_like(value))


def test_ssl_loss_validates_pair_index_and_representation_metadata():
    prediction, target = _tensors(tracks=1)
    metadata = _metadata(tracks=1)
    bad_pair_metadata = {**metadata, "pair_index": torch.tensor([[[0, 0]]])}
    with pytest.raises(ValueError, match="self pairs"):
        SSLDPIPDPITLoss(max_tracks=1)(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
            **bad_pair_metadata,
        )
    bad_rep_metadata = {**metadata, "input_representation": "raw_stft_ri"}
    with pytest.raises(ValueError, match="input_representation"):
        SSLDPIPDPITLoss(max_tracks=1)(
            prediction,
            target,
            output_lengths=torch.tensor([2]),
            **bad_rep_metadata,
        )


def test_strict_ssl_wrapper_accepts_canonical_dpipd_positional_shape():
    """strict wrapper 的 positional 路径必须保留 canonical pair 轴。"""
    from core.losses.build import _StrictLossWrapper

    prediction, target = _tensors(tracks=1)
    prediction.requires_grad_()
    wrapper = _StrictLossWrapper(SSLDPIPDPITLoss(max_tracks=1))
    value = wrapper(prediction, target, torch.tensor([2]))

    assert value.ndim == 0
    assert torch.isfinite(value)


def test_ssl_pit_rejects_legacy_prediction_target_and_length_aliases():
    """loss 入口只接受 canonical dpipd/target_dpipd/output_lengths。"""
    prediction, target = _tensors(tracks=1)
    loss_fn = SSLDPIPDPITLoss(max_tracks=1)
    canonical_kwargs = {
        "target_dpipd": target,
        "output_lengths": torch.tensor([2]),
    }
    for legacy_prediction in ("prediction", "ipd_prediction"):
        with pytest.raises(ValueError, match="canonical.*dpipd"):
            loss_fn({legacy_prediction: prediction}, **canonical_kwargs)
    with pytest.raises(ValueError, match="canonical.*dpipd"):
        loss_fn({"dpipd": prediction, "prediction": prediction}, **canonical_kwargs)
    with pytest.raises(TypeError, match="ipd_target"):
        loss_fn(prediction, ipd_target=target, output_lengths=torch.tensor([2]))
    with pytest.raises(TypeError, match="lengths"):
        loss_fn(prediction, target_dpipd=target, lengths=torch.tensor([2]))


def test_ssl_plain_mse_rejects_legacy_prediction_target_and_length_aliases():
    """plain SSL MSE 与 PIT 共享同一 canonical 输入门禁。"""
    prediction, target = _tensors(tracks=1)
    with pytest.raises(ValueError, match="canonical.*dpipd"):
        ipd_mse_loss(
            {"ipd_prediction": prediction},
            target_dpipd=target,
            output_lengths=torch.tensor([2]),
        )
    with pytest.raises(TypeError, match="ipd_target"):
        ipd_mse_loss(
            prediction,
            ipd_target=target,
            output_lengths=torch.tensor([2]),
        )
    with pytest.raises(TypeError, match="lengths"):
        ipd_mse_loss(
            prediction,
            target_dpipd=target,
            lengths=torch.tensor([2]),
        )


def test_ssl_model_io_rejects_legacy_prediction_output_aliases():
    """模型 I/O 合同也必须只登记 canonical ``dpipd`` 输出。"""
    from dataclasses import replace

    from core.models.io_contract import TensorField
    from tasks.ssl.model_io import validate_ssl_model_io
    from tasks.ssl.models.ipdnet import FNSSL

    class _LegacyOutputFNSSL(FNSSL):
        IO_CONTRACT = replace(
            FNSSL.IO_CONTRACT,
            outputs=(
                TensorField(
                    "prediction",
                    "spectrum",
                    ("batch", "frame", "frequency", "track", "pair", "complex"),
                    multiple=True,
                ),
            ),
        )

    with pytest.raises(ValueError, match="canonical 'dpipd'"):
        validate_ssl_model_io(
            _LegacyOutputFNSSL(input_size=1, hidden_size=4, max_track=1, n_fft=8)
        )
