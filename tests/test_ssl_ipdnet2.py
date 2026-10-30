"""IPDnet2（官方 ``OnlineSpatialNet`` + Mamba）结构接入的合同测试。

覆盖范围与边界：

- 模块导入本身不得依赖 ``mamba_ssm``（包导入安全）；
- ``mamba_ssm`` 缺失必须在**构造期** ``ImportError``（不静默退回 GRU/LSTM）；
- 未知 kwargs 与非法的超参组合在构造期 fail-fast；
- 输出 canonical 轴序 ``[B,T_out,F,K,P,2]`` 与形状/有限性；
- ``mamba_ssm`` 为 CUDA-only，真正的 forward 用例标记 ``gpu`` 并在无 CUDA 时跳过；
  其余合同用例必须在 CPU 上跑通。

注意：registry 已由 ``tasks/ssl/models/__init__.py`` 收口到官方
``tasks.ssl.models.ipdnet2``。本文件仍用 ``importlib`` 直接加载模块，隔离
optional ``mamba_ssm`` 的导入副作用，并保持模块级 CUDA-only/fail-fast 合同可独立验收。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "tasks" / "ssl" / "models" / "ipdnet2.py"
)


def _load_module() -> ModuleType:
    """直接按文件加载 ipdnet2 模块，不经过 ``tasks.ssl.models.__init__``。

    先导入 ``tasks.ssl.models.ipdnet`` 以提供共享的 ``_PairwiseIPDModel``，再按
    文件加载本模块；这样可以独立测试 optional import 和构造期 fail-fast，而不依赖
    包 ``__init__`` 在收集阶段触发 Mamba 导入。
    """
    cached = sys.modules.get("_ssl_ipdnet2_under_test")
    if cached is not None:
        return cached
    import tasks.ssl.models.ipdnet  # noqa: F401  提供共享的 _PairwiseIPDModel
    from core.utils.registry import MODEL_REGISTRY

    # 官方模块可能已经通过 canonical lazy registry 导入（例如与
    # ``test_ssl_official_model_smoke`` 同一 pytest 进程中）。直接按别名加载
    # 会再次执行 ``@MODEL_REGISTRY.register('IPDNet2')``，因此暂时摘除现有
    # 条目；执行完成后恢复，避免测试 helper 污染后续 canonical registry。
    previous = MODEL_REGISTRY._registry.pop("IPDNet2", None)

    spec = importlib.util.spec_from_file_location("_ssl_ipdnet2_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # 注册到 sys.modules 后再执行，保证 dataclass/typing 的自省可用。
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        MODEL_REGISTRY._registry.pop("IPDNet2", None)
        if previous is not None:
            MODEL_REGISTRY._registry["IPDNet2"] = previous
    return module


# 小规模但满足官方整除链的配置：num_freqs//2//8 * fre_compression_ratio == num_freqs
# 与 n_fft 解耦（n_fft=64 -> full_freqs=33），用于 CPU 合同测试。
_SMALL_KWARGS = dict(
    input_size=4,
    hidden_size=16,
    max_track=2,
    n_fft=64,
    num_freqs=32,
    num_layers=2,
    dim_squeeze=16,
    num_heads=2,
    attention="mamba(8,4)",
    fre_compression_ratio=16,
    time_compression_ratio=5,
    stride=5,
    arrangement="alternating",
    forgetting_norm_sample_length=None,
)


def _cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


@pytest.mark.unit
def test_module_import_does_not_require_mamba_ssm():
    """包导入安全：加载本模块本身不得触发 ``mamba_ssm`` 导入。"""
    module = _load_module()
    assert hasattr(module, "IPDNet2")
    assert module.IPDNet2.MODEL_TYPE == "IPDNet2"
    # 模块级不得残留已导入的 mamba 对象（只在构造期按需导入）。
    assert not hasattr(module, "Mamba")


@pytest.mark.unit
def test_missing_mamba_ssm_raises_at_construction(monkeypatch):
    """``mamba_ssm`` 缺失时必须在构造期显式 ``ImportError``，不静默回退。"""
    module = _load_module()
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(ImportError, match="mamba-ssm"):
        module.IPDNet2(**_SMALL_KWARGS)


@pytest.mark.unit
def test_unknown_kwargs_raise_type_error():
    """未知构造参数必须抛 ``TypeError``，不能静默吞掉 YAML 拼写错误。"""
    module = _load_module()
    with pytest.raises(TypeError, match="unexpected model arguments"):
        module.IPDNet2(**_SMALL_KWARGS, not_a_real_argument=1)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({'causal': False}, "causal"),
        ({'stride': 3}, "stride must equal time_compression_ratio"),
        ({'output_chunk_frames': 7}, "output_chunk_frames must be divisible"),
        ({'num_layers': 0}, "num_layers"),
        ({'time_compression_layer': 5}, "time_compression_layer"),
        ({'num_freqs': 30}, "num_freqs"),
        ({'exclude_dc': False}, "exclude_dc=false requires"),
        ({'exclude_dc': True, 'num_freqs': 33}, "exclude_dc=true requires"),
        ({'attention': 'mhsa(251)'}, "mamba"),
        ({'attention': 'mamba(16)'}, "mamba"),
        ({'attention': 'mamba(0,4)'}, "mamba d_state and d_conv must be positive"),
        ({'attention': 'mamba(16,0)'}, "mamba d_state and d_conv must be positive"),
        ({'conv_groups': (5, 8)}, "groups"),
        ({'norms': ("LN", "LN", "GN", "LN", "LN", "BAD")}, "norm"),
        ({'dropout': (0.0, 1.5, 0.0)}, "dropout"),
        ({'arrangement': "nope"}, "arrangement"),
        ({'mamba_d_state': 7}, "conflicts with the attention string"),
    ],
)
def test_invalid_hyperparameters_fail_fast(overrides, message):
    """非法超参组合必须在构造期 fail-fast。"""
    module = _load_module()
    kwargs = dict(_SMALL_KWARGS)
    kwargs.update(overrides)
    with pytest.raises((ValueError, TypeError), match=message):
        module.IPDNet2(**kwargs)


@pytest.mark.unit
def test_declared_model_spec_and_io_contract():
    """``MODEL_SPEC`` 必须是官方 IPDnet2 的显式合同。"""
    module = _load_module()
    spec = module.IPDNet2.MODEL_SPEC
    assert spec.model_type == "IPDNet2"
    assert spec.feature_contract == "multichannel_stft_real_imag"
    assert spec.target_contract == "dpipd_pair"
    assert spec.pair_layout == "reference_to_all"
    assert spec.geometry_mode == "variable_array"
    assert spec.field_mode == "near_field"
    assert spec.reduction == "mamba_time_pool"
    assert spec.causal is True
    assert spec.mapping == "near_field_doa_range_grid"
    assert spec.target_layout == "trailing_ri"
    contract = module.IPDNet2.IO_CONTRACT
    assert contract.tasks == frozenset({"ssl"})


@pytest.mark.unit
def test_forward_rejects_malformed_input_on_cpu():
    """CPU 上可校验的输入合同（无需 mamba_ssm）。"""
    module = _load_module()
    model = module.IPDNet2(**_SMALL_KWARGS)
    with pytest.raises(ValueError, match="feature dimension"):
        model(torch.randn(1, 20, 32, 8))  # 2C=8 与配置 2C=10 不符
    with pytest.raises(ValueError, match="frequency bins"):
        model(torch.randn(1, 20, 40, 10))  # F 既不是 32 也不是 33


@pytest.mark.unit
def test_official_layout_places_explicit_reference_channel_first():
    """IPDNet2 的 canonical 输入要按 reference_channel 对齐官方 pair 轴。"""
    module = _load_module()
    model = module.IPDNet2(**{**_SMALL_KWARGS, "reference_channel": 1})
    value = torch.arange(1 * 2 * 33 * 10, dtype=torch.float32).reshape(1, 2, 33, 10)
    official = model._official_layout(value)
    canonical = value.reshape(1, 2, 33, 5, 2)
    order = [1, 0, 2, 3, 4]
    reordered = canonical[..., order, :]
    expected = torch.cat((reordered[..., 0], reordered[..., 1]), dim=-1)
    expected = expected.permute(0, 3, 2, 1)
    assert torch.equal(official, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="mamba_ssm is CUDA-only")
def test_official_forward_shapes_and_finiteness_on_cuda():
    """官方 8 层配置的 forward：canonical 轴序、形状与有限性。

    ``T_out = T // time_compression_ratio``；输入是 canonical
    ``[B,T,F,2C]``（含 DC 的 n_fft//2+1 频点），模块内部删 DC 再重排成官方
    ``[B,2C,F,T]``。
    """
    module = _load_module()
    model = module.IPDNet2(
        input_size=4,
        hidden_size=96,
        max_track=2,
        n_fft=512,
        num_freqs=256,
        num_layers=8,
        dim_squeeze=8,
        num_heads=4,
        attention="mamba(16,4)",
        fre_compression_ratio=16,
        time_compression_ratio=5,
        stride=5,
        forgetting_norm_sample_length=None,
    ).cuda()
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    frames, channels = 40, 10
    features = torch.randn(1, frames, 257, channels, device="cuda")
    with torch.no_grad():
        output = model(features)
    assert output.shape == (1, frames // 5, 256, 2, 4, 2), output.shape
    assert torch.isfinite(output).all()
    assert output.device.type == "cuda"
    peak = torch.cuda.max_memory_allocated()
    assert peak > 0


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="mamba_ssm is CUDA-only")
def test_output_is_deterministic_in_eval_mode():
    """eval 模式下同一输入必须给出同一输出（无 dropout、无随机分支）。"""
    module = _load_module()
    model = module.IPDNet2(**_SMALL_KWARGS).cuda().eval()
    features = torch.randn(1, 20, 33, 10, device="cuda")
    with torch.no_grad():
        first = model(features)
        second = model(features)
    assert torch.equal(first, second)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="mamba_ssm is CUDA-only")
def test_output_scale_maps_tanh_range():
    """``output_scale`` 必须显式作用于输出（默认 1.0 = 纯官方 tanh 输出）。"""
    module = _load_module()
    features = torch.randn(1, 20, 33, 10, device="cuda")
    plain = module.IPDNet2(**_SMALL_KWARGS).cuda().eval()
    scaled = module.IPDNet2(**{**_SMALL_KWARGS, "output_scale": 3.14159}).cuda().eval()
    scaled.load_state_dict(plain.state_dict())
    with torch.no_grad():
        base = plain(features)
        wide = scaled(features)
    assert torch.allclose(wide, base * 3.14159, atol=1e-5)
