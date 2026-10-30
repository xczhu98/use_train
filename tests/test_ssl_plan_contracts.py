"""FN-SSL/IPDnet 官方模型的 SSL canonical 合同测试。

这些测试只检查任务边界，不依赖真实 speech/noise 资源。训练配置仍须由
``SSLTrainingEntry`` 在实际启动前执行 strict resource gate。
"""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.unit


def _item(*, input_representation: str = "multichannel_stft_real_imag", target_representation: str = "dpipd_far_field", near_field: bool = False):
    """构造最小的 canonical item，输入和 target 使用独立时间轴。"""
    batch_frames = 5
    target_frames = 3
    freqs = 4
    channels = 2
    tracks = 2
    pairs = 1
    feature_dim = 2 * channels if input_representation == "multichannel_stft_real_imag" else 2 * (channels - 1)
    item = {
        "features": torch.zeros(batch_frames, freqs, feature_dim),
        "target_dpipd": torch.zeros(target_frames, freqs, tracks, pairs, 2),
        "doa": torch.zeros(target_frames, tracks, 2),
        "vad": torch.ones(target_frames, tracks),
        "pair_index": torch.tensor([[0, 1]], dtype=torch.int64),
        "pair_mask": torch.ones(pairs, dtype=torch.bool),
        "geometry": torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]),
        "input_lengths": batch_frames,
        "output_lengths": target_frames,
        "frame_origin": 0,
        "time_stride": 2,
        "sample_rate": 16000,
        "input_representation": input_representation,
        "target_representation": target_representation,
        "coordinate_frame": "array_local",
        "provenance": {"backend": "fixture"},
    }
    if near_field:
        item["source_distances"] = torch.full((target_frames, tracks), 1.0)
    return item


def test_ssl_collate_preserves_raw_stft_pair_axis_and_dual_timebase():
    """多通道 STFT 与 DP-IPD target 不能共享一个伪造的时间长度。"""
    from tasks.ssl.data import ssl_collate_fn

    batch = ssl_collate_fn([_item(), _item()])
    assert batch.features.shape == (2, 5, 4, 4)
    assert batch.target_dpipd.shape == (2, 3, 4, 2, 1, 2)
    assert batch.pair_index.shape == (2, 1, 2)
    assert batch.pair_mask.shape == (2, 1)
    assert batch.input_lengths.tolist() == [5, 5]
    assert batch.output_lengths.tolist() == [3, 3]
    assert batch.frame_origin.tolist() == [0, 0]
    assert batch.time_stride == 2
    batch.validate()


def test_ssl_collate_accepts_reference_ipd_as_explicit_native_contract():
    """native IPD baseline 可用 reference_ipd，但不能伪装成多通道 STFT。"""
    from tasks.ssl.data import ssl_collate_fn

    batch = ssl_collate_fn([_item(input_representation="reference_ipd")])
    assert batch.input_representation == "reference_ipd"
    assert batch.features.shape[-1] == 2
    batch.validate()


@pytest.mark.parametrize(
    "missing",
    ["target_dpipd", "pair_index", "pair_mask", "input_lengths", "output_lengths", "frame_origin", "time_stride"],
)
def test_ssl_collate_rejects_items_missing_canonical_contract_fields(missing):
    """旧的 features/targets/lengths 简化 item 不得作为生产回退入口。"""
    from tasks.ssl.data import ssl_collate_fn

    item = _item()
    item.pop(missing)
    with pytest.raises((KeyError, TypeError, ValueError)):
        ssl_collate_fn([item])


def test_ssl_collate_rejects_pair_index_mismatch():
    from tasks.ssl.data import ssl_collate_fn

    item = _item()
    item["pair_index"] = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    with pytest.raises(ValueError, match="pair_index|pair"):
        ssl_collate_fn([item])


def test_ssl_near_field_requires_source_distance():
    from tasks.ssl.data import ssl_collate_fn

    near = _item(target_representation="dpipd_near_field")
    with pytest.raises(ValueError, match="source_distances|source distance"):
        ssl_collate_fn([near])

def test_ssl_near_field_carries_source_distance_only_as_target_metadata():
    from tasks.ssl.data import ssl_collate_fn

    batch = ssl_collate_fn([_item(target_representation="dpipd_near_field", near_field=True)])
    assert batch.source_distances.shape == (1, 3, 2)
    assert batch.target_representation == "dpipd_near_field"
    batch.validate()


def test_ssl_runtime_forwards_all_canonical_timebase_and_geometry_fields():
    from tasks.ssl.data import ssl_collate_fn
    from tasks.ssl.runtime import SSLRuntimeInputs

    batch = ssl_collate_fn([_item()])
    runtime = SSLRuntimeInputs.from_batch(batch)
    kwargs = runtime.loss_kwargs()
    for key in (
        "target_dpipd",
        "pair_index",
        "pair_mask",
        "geometry",
        "input_lengths",
        "output_lengths",
        "frame_origin",
        "time_stride",
        "input_representation",
        "target_representation",
        "coordinate_frame",
    ):
        assert key in kwargs
    assert kwargs["target_dpipd"].shape == (1, 3, 4, 2, 1, 2)
    assert kwargs["source_distances"] is None


@pytest.mark.parametrize(
    "missing,alias",
    [
        ("target_dpipd", ("targets", "target_dpipd")),
        ("input_lengths", ("lengths", "input_lengths")),
        ("output_lengths", ("lengths", "output_lengths")),
        ("frame_origin", (None, "frame_origin")),
        ("time_stride", ("output_stride", "time_stride")),
        ("coordinate_frame", (None, "coordinate_frame")),
    ],
)
def test_ssl_runtime_rejects_legacy_field_fallbacks(missing, alias):
    """runtime 只消费 canonical SSLBatch 字段，不回退到旧别名或默认值。"""
    from tasks.ssl.data import ssl_collate_fn
    from tasks.ssl.runtime import SSLRuntimeInputs

    batch = ssl_collate_fn([_item()])
    fields = dict(vars(batch))
    fields.pop(missing)
    alias_name, canonical_name = alias
    if alias_name == "targets":
        fields[alias_name] = batch.target_dpipd
    elif alias_name == "lengths":
        fields[alias_name] = batch.output_lengths
    elif alias_name == "output_stride":
        fields[alias_name] = batch.time_stride
    malformed = SimpleNamespace(**fields)
    with pytest.raises(AttributeError, match=canonical_name):
        SSLRuntimeInputs._from_validated_batch(malformed)


def test_ssl_dpipd_pit_is_invariant_to_source_permutation():
    """source track 的排列变化不能改变 PIT 主损失。"""
    from tasks.ssl.losses import SSLDPIPDPITLoss

    target = torch.zeros(1, 2, 3, 2, 1, 2)
    target[:, :, :, 0, :, 0] = 1.0
    target[:, :, :, 1, :, 1] = 1.0
    prediction = target[:, :, :, [1, 0], :, :].clone()
    loss = SSLDPIPDPITLoss(max_tracks=2)(
        prediction,
        target_dpipd=target,
        output_lengths=torch.tensor([2]),
        pair_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0.0)


def test_ssl_dpipd_pit_handles_eight_tracks_without_factorial_enumeration():
    """多轨 PIT 使用 assignment 求解，K=8 仍应快速得到精确零损失。"""
    from tasks.ssl.losses import SSLDPIPDPITLoss

    tracks = 8
    target = torch.zeros(1, 1, 1, tracks, 1, 2)
    for track in range(tracks):
        target[0, 0, 0, track, 0, track % 2] = float(track + 1)
    permutation = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3])
    prediction = target[:, :, :, permutation, :, :].clone().requires_grad_()
    loss = SSLDPIPDPITLoss(max_tracks=tracks)(
        prediction,
        target_dpipd=target,
        output_lengths=torch.tensor([1]),
        vad=torch.ones(1, 1, tracks),
        pair_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0.0)
    loss.backward()
    assert prediction.grad is not None


def test_ssl_model_spec_separates_multichannel_stft_and_native_reference_ipd():
    from tasks.ssl.model_io import ModelSpec, parse_model_spec

    fnssl = parse_model_spec(
        {
            "feature_contract": "multichannel_stft_real_imag",
            "target_contract": "dpipd_pair",
            "pair_layout": "reference_to_all",
            "geometry_mode": "fixed_array",
            "field_mode": "far_field",
            "reduction": "average_pool_12",
            "causal": False,
            "mapping": "far_field_grid",
            "target_layout": "trailing_ri",
        },
        model_type="FNSSL",
    )
    native = parse_model_spec(
        {
            "feature_contract": "reference_ipd",
            "target_contract": "dpipd_pair",
            "pair_layout": "reference_to_all",
            "geometry_mode": "fixed_array",
            "field_mode": "far_field",
            "reduction": "model_defined",
            "causal": False,
            "mapping": "far_field_grid",
            "target_layout": "trailing_ri",
        },
        model_type="FixedArrayIPDNet",
    )
    assert isinstance(fnssl, ModelSpec)
    assert fnssl.feature_contract == "multichannel_stft_real_imag"
    assert fnssl.target_contract == "dpipd_pair"
    assert fnssl.target_layout == "trailing_ri"
    assert native.feature_contract == "reference_ipd"
    assert native.target_contract == "dpipd_pair"
    assert native.target_layout == "trailing_ri"


def test_ssl_model_spec_rejects_unknown_keys_and_official_reference_ipd():
    from tasks.ssl.model_io import parse_model_spec

    spec = {
        "feature_contract": "multichannel_stft_real_imag",
        "target_contract": "dpipd_pair",
        "pair_layout": "reference_to_all",
        "geometry_mode": "fixed_array",
        "field_mode": "far_field",
        "reduction": "average_pool_12",
        "causal": False,
        "mapping": "far_field_grid",
        "target_layout": "trailing_ri",
        "unexpected": True,
    }
    with pytest.raises(ValueError, match="unknown model_spec keys"):
        parse_model_spec(spec, model_type="FNSSL")
    valid = {key: value for key, value in spec.items() if key != "unexpected"}
    valid["feature_contract"] = "reference_ipd"
    with pytest.raises(ValueError, match="official.*multichannel_stft_real_imag"):
        parse_model_spec(valid, model_type="FNSSL")


def test_ssl_model_spec_requires_and_validates_target_layout():
    from tasks.ssl.model_io import parse_model_spec

    spec = {
        "feature_contract": "multichannel_stft_real_imag",
        "target_contract": "dpipd_pair",
        "pair_layout": "reference_to_all",
        "geometry_mode": "fixed_array",
        "field_mode": "far_field",
        "reduction": "average_pool_12",
        "causal": False,
        "mapping": "far_field_grid",
    }
    with pytest.raises(ValueError, match="missing required keys.*target_layout"):
        parse_model_spec(spec, model_type="FNSSL")
    for layout in ("trailing_ri", "freq_concat_ri"):
        parsed = parse_model_spec({**spec, "target_layout": layout}, model_type="FNSSL")
        assert parsed.target_layout == layout
    with pytest.raises(ValueError, match=r"ModelSpec\.target_layout"):
        parse_model_spec({**spec, "target_layout": "flattened"}, model_type="FNSSL")


def test_ssl_entry_rejects_retired_representation_and_distance_model_input():
    from core.utils.config import load_merged_config
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl/fnssl_dynamic.yaml")
    retired = deepcopy(cfg)
    retired["ssl"]["input_representation"] = "raw_stft_ri"
    with pytest.raises(ValueError, match="multichannel_stft_real_imag"):
        SSLTrainingEntry("ssl_contract")._validate_config(retired)

    distance = deepcopy(cfg)
    distance["model"]["args"]["use_source_distance"] = True
    with pytest.raises((TypeError, ValueError), match="source_distance|unknown|unexpected"):
        SSLTrainingEntry("ssl_contract")._validate_config(distance)


def test_ssl_registry_rejects_unknown_only_name_and_keeps_official_names_explicit():
    from tasks.ssl.models import SSL_MODEL_TYPES, register_ssl_models

    assert {"FixedArrayIPDNet", "FNSSL", "IPDNetFixed", "IPDNetVariable", "IPDNet2"} <= set(
        SSL_MODEL_TYPES
    )
    with pytest.raises((KeyError, ValueError), match="unknown|Unknown|not registered"):
        register_ssl_models(only="UnknownSSLModel")


def test_ipdnet2_has_no_ground_truth_source_distance_input_in_official_config():
    """IPDnet2 的配置不能把 GT distance 作为推理观测。"""
    from core.utils.config import load_merged_config

    path = Path(__file__).resolve().parents[1] / "configs/ssl/ipdnet2_dynamic.yaml"
    cfg = load_merged_config(path)
    model_args = cfg.get("model", {}).get("args", {})
    ssl_cfg = cfg.get("ssl", {})
    assert model_args.get("use_source_distance", False) is False
    assert ssl_cfg.get("source_distance_input", False) is False
    assert "source_distance" not in set(ssl_cfg.get("model_inputs", ()))


def test_ssl_entry_aligns_model_spec_and_rejects_unimplemented_pair_layouts():
    """配置不能把 proxy 声明成另一种 pair/time/field 合同。"""
    from core.utils.config import load_merged_config
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl/fnssl_dynamic.yaml")
    mismatch = deepcopy(cfg)
    mismatch["model_spec"]["pair_layout"] = "pair_batched"
    with pytest.raises(ValueError, match=r"model_spec\.pair_layout.*ssl\.pair_layout"):
        SSLTrainingEntry("ssl_contract")._validate_config(mismatch)

    pair_batched = load_merged_config(root / "configs/ssl/ipdnet_variable_dynamic.yaml")
    pair_batched["ssl"]["pair_layout"] = "pair_batched"
    pair_batched["model_spec"]["pair_layout"] = "pair_batched"
    with pytest.raises(ValueError, match="only.*reference_to_all|pair_batched"):
        SSLTrainingEntry("ssl_contract")._validate_config(pair_batched)


def test_ssl_model_io_rejects_configured_spec_different_from_model_declaration():
    from tasks.ssl.model_io import validate_ssl_model_io
    from tasks.ssl.models.ipdnet import FNSSL

    model = FNSSL(input_size=1, hidden_size=4, max_track=1, n_fft=8, stride=12)
    configured = model.MODEL_SPEC.as_dict()
    configured["reduction"] = "causal_conv_pool"
    with pytest.raises(ValueError, match="does not match.*MODEL_SPEC"):
        validate_ssl_model_io(model, configured)
    configured = model.MODEL_SPEC.as_dict()
    configured["target_layout"] = "freq_concat_ri"
    with pytest.raises(ValueError, match="does not match.*MODEL_SPEC"):
        validate_ssl_model_io(model, configured)


def test_ssl_model_io_rejects_internal_target_layout_at_public_boundary():
    from dataclasses import replace

    from tasks.ssl.model_io import validate_ssl_model_io
    from tasks.ssl.models.ipdnet import FNSSL

    class _InternalLayoutFNSSL(FNSSL):
        MODEL_SPEC = replace(FNSSL.MODEL_SPEC, target_layout="freq_concat_ri")

    model = _InternalLayoutFNSSL(input_size=1, hidden_size=4, max_track=1, n_fft=8, stride=12)
    with pytest.raises(ValueError, match="public model I/O.*freq_concat_ri"):
        validate_ssl_model_io(model)


def test_ssl_model_io_requires_explicit_forward_input_name():
    """``**kwargs`` 不能冒充 canonical SSL forward 输入合同。"""
    from tasks.ssl.model_io import validate_ssl_model_io
    from tasks.ssl.models.ipdnet import FNSSL

    class _CatchAllModel(torch.nn.Module):
        MODEL_TYPE = "FNSSL"
        MODEL_SPEC = FNSSL.MODEL_SPEC
        IO_CONTRACT = FNSSL.IO_CONTRACT

        def forward(self, **kwargs):
            del kwargs
            return torch.zeros(1, 1, 1, 1, 1, 2)

    with pytest.raises(ValueError, match="explicitly declare.*multichannel_stft_real_imag"):
        validate_ssl_model_io(_CatchAllModel())


def test_fnssl_uses_official_fn_blocks_and_restores_pair_axis():
    """FNSSL 不能退化成 GRU proxy，且输出必须恢复 canonical pair 轴。"""
    from tasks.ssl.models.ipdnet import FNSSL, _FNBlock

    model = FNSSL(input_size=1, hidden_size=8, max_track=2, n_fft=8, stride=12)
    assert isinstance(model.block_1, _FNBlock)
    assert isinstance(model.block_1.full_lstm, torch.nn.LSTM)
    assert isinstance(model.block_1.narrow_lstm, torch.nn.LSTM)
    assert not hasattr(model, "temporal")
    output = model(torch.randn(2, 24, 5, 4))
    assert output.shape == (2, 2, 5, 2, 1, 2)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("model_name", ["IPDNetFixed", "IPDNetVariable"])
def test_official_ipdnet_names_build_official_structures(model_name):
    """官方名称必须构造官方 FN block / causal convolution 结构，不得退回 GRU 代理。"""
    from tasks.ssl.models import build_model
    from tasks.ssl.models.ipdnet_official import _OfficialFNBlock, _OfficialFNBlockMean

    model = build_model(
        {
            "type": model_name,
            "args": {"input_size": 4, "hidden_size": 8, "max_track": 2, "n_fft": 8, "stride": 12},
        }
    )
    assert model.MODEL_SPEC.model_type == model_name
    assert isinstance(model.block_1, (_OfficialFNBlock, _OfficialFNBlockMean))
    assert not hasattr(model, "temporal")
    features = torch.randn(2, 24, 5, 4)
    if model_name == "IPDNetVariable":
        output = model(features, pair_index=torch.tensor([[[0, 1]], [[0, 1]]]))
    else:
        output = model(features)
    assert output.shape == (2, 2, 5, 2, 1, 2)
    assert torch.isfinite(output).all()


def test_ssl_doa_auxiliary_loss_is_wrap_safe():
    """azimuth 的 -π/π 接缝不能被普通角度 MSE 放大。"""
    from tasks.ssl.losses import SSLDPIPDPITLoss

    target = torch.zeros(1, 1, 2, 1, 1, 2)
    prediction = target.clone()
    target_doa = torch.tensor([[[[torch.pi / 2, -torch.pi + 1e-3]]]])
    prediction_doa = torch.tensor([[[[torch.pi / 2, torch.pi - 1e-3]]]])
    loss = SSLDPIPDPITLoss(max_tracks=1, doa_loss_weight=1.0)(
        prediction,
        target_dpipd=target,
        output_lengths=torch.tensor([1]),
        doa=target_doa,
        doa_prediction=prediction_doa,
        pair_mask=torch.ones(1, 1, dtype=torch.bool),
        pair_index=torch.tensor([[[0, 1]]]),
        geometry=torch.tensor([[[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]]),
        input_representation="multichannel_stft_real_imag",
        target_representation="dpipd_far_field",
        coordinate_frame="array_local",
        time_stride=1,
    )
    assert loss.item() < 1e-4


def test_ssl_doa_auxiliary_loss_reuses_spectral_pit_assignment():
    """DOA 辅助项必须沿用 DP-IPD PIT 选出的 source assignment。"""
    from tasks.ssl.losses import SSLDPIPDPITLoss

    target = torch.zeros(1, 1, 1, 2, 1, 2)
    target[:, :, :, 0, :, 0] = 1.0
    target[:, :, :, 1, :, 1] = 1.0
    prediction = target[:, :, :, [1, 0], :, :].clone()
    target_doa = torch.tensor(
        [[[[torch.pi / 2, 0.0], [torch.pi / 2, torch.pi / 2]]]]
    )
    prediction_doa = target_doa[:, :, [1, 0], :].clone()
    loss = SSLDPIPDPITLoss(max_tracks=2, doa_loss_weight=1.0)(
        prediction,
        target_dpipd=target,
        output_lengths=torch.tensor([1]),
        doa=target_doa,
        vad=torch.ones(1, 1, 2),
        doa_prediction=prediction_doa,
        pair_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    assert loss.item() < 1e-6


def test_ssl_entry_rejects_unknown_ssl_section_keys():
    """ssl: section 的未知键必须在配置边界 fail-fast，不能静默透传。"""
    from core.utils.config import load_merged_config
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl/fnssl_dynamic.yaml")
    cfg["ssl"]["bogus_future_key"] = True
    with pytest.raises(ValueError, match="unknown ssl keys.*bogus_future_key"):
        SSLTrainingEntry("ssl_contract")._validate_config(cfg)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_ssl_entry_validates_freqs_for_each_split(split):
    """train/validation 的频点声明都必须经过 n_fft 合同校验。"""
    from core.utils.config import load_merged_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl/fnssl_dynamic.yaml")
    cfg["data"][split]["freqs"] = 1
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    with pytest.raises(ValueError, match=rf"data\.{split}\.freqs"):
        SSLTrainingEntry("ssl_contract")._validate_config(cfg)


@pytest.mark.parametrize("key,value", [("vad_mode", 2), ("num_tracks", 0)])
def test_ssl_entry_rejects_invalid_vad_mode_and_num_tracks(key, value):
    """已接入的 VAD 模式与声源数仍必须在配置边界拒绝非法值。"""
    from core.utils.config import load_merged_config
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl/fnssl_dynamic.yaml")
    cfg["ssl"][key] = value
    with pytest.raises(ValueError, match=rf"ssl\.{key}"):
        SSLTrainingEntry("ssl_contract")._validate_config(cfg)


@pytest.mark.parametrize(
    "config_name",
    [
        "fnssl_dynamic.yaml",
        "ipdnet_fixed_dynamic.yaml",
        "ipdnet_variable_dynamic.yaml",
        "ipdnet2_dynamic.yaml",
    ],
)
def test_ssl_contract_template_configs_pass_validate_config(config_name):
    """白名单不得误伤已消费键：四份官方合同模板配置仍须通过 _validate_config。"""
    from core.utils.config import load_merged_config
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl" / config_name)
    SSLTrainingEntry("ssl_contract")._validate_config(cfg)


@pytest.mark.parametrize(
    ("config_name", "section", "field", "value", "match"),
    [
        ("fnssl_dynamic.yaml", "ssl", "target_representation", "dpipd_near_field", "FNSSL.*target_representation"),
        ("ipdnet_fixed_dynamic.yaml", "ssl", "geometry_mode", "variable_array", "IPDNetFixed.*geometry_mode"),
        ("ipdnet_variable_dynamic.yaml", "ssl", "mapping", "near_field_doa_range_grid", "IPDNetVariable.*mapping"),
        ("ipdnet2_dynamic.yaml", "ssl", "target_representation", "dpipd_far_field", "IPDNet2.*target_representation"),
        ("fnssl_dynamic.yaml", "model_spec", "target_layout", "freq_concat_ri", r"model_spec\.target_layout"),
    ],
)
def test_ssl_entry_rejects_model_specific_contract_mismatch(
    config_name, section, field, value, match
):
    """模型合同字段必须在配置边界收口，不能延迟到首次 forward。"""
    from core.utils.config import load_merged_config
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl" / config_name)
    cfg[section][field] = value
    with pytest.raises(ValueError, match=match):
        SSLTrainingEntry("ssl_contract")._validate_config(cfg)


def test_ssl_entry_rejects_fnssl_exclude_dc_true():
    """FNSSL 只消费全频 bin（n_fft//2+1），exclude_dc=true 必须 fail-fast。"""
    from core.utils.config import load_merged_config
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl/fnssl_dynamic.yaml")
    cfg["ssl"]["exclude_dc"] = True
    with pytest.raises(ValueError, match="exclude_dc"):
        SSLTrainingEntry("ssl_contract")._validate_config(cfg)


def test_ssl_entry_accepts_fnssl_exclude_dc_false():
    from core.utils.config import load_merged_config
    from tasks.ssl.trainer.entry import SSLTrainingEntry

    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(root / "configs/ssl/fnssl_dynamic.yaml")
    assert cfg["ssl"]["exclude_dc"] is False
    SSLTrainingEntry("ssl_contract")._validate_config(cfg)
