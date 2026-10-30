"""PASE 家族 WavLM 前端在 autocast 下的 apply_mask dtype 回归测试。

bf16 autocast 时特征 ``x`` 为 bf16，而 ``mask_emb`` 是 fp32 参数；
``x[mask_indices] = self.mask_emb`` 会触发
``Index put requires the source and destination dtypes match``。
UniPASE Stage1/3/4 与 GAP-URGENet Phase2 的 enable_plc=True 路径都会经过这里。
"""

import pytest
import torch

from tasks.se.models.pase_family.common.wavlm.WavLM import WavLM

pytestmark = [pytest.mark.unit]


class _DummyMaskHost:
    """只承载 ``apply_mask`` 所需的属性，避免构造完整 WavLM。"""

    def __init__(
        self,
        mask_emb: torch.Tensor,
        mask_prob: float = 0.65,
        mask_length: int = 2,
        mask_selection: str = "static",
        mask_other: float = 0.0,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        mask_channel_prob: float = 0.0,
    ) -> None:
        self.mask_emb = mask_emb
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_prob = mask_channel_prob


def test_apply_mask_bf16_autocast_with_indices() -> None:
    """bf16 x + fp32 mask_emb 在 autocast 下不得因 dtype 不一致报错。"""
    torch.manual_seed(0)
    x_orig = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    x = x_orig.clone()
    mask_emb = torch.randn(8, dtype=torch.float32)
    # 与真实调用一致：compute_mask_indices 产出 (B, T) 布尔掩码
    mask_indices = torch.zeros(2, 5, dtype=torch.bool)
    mask_indices[0, :2] = True
    mask_indices[1, 3:] = True

    host = _DummyMaskHost(mask_emb=mask_emb)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, returned_indices = WavLM.apply_mask(host, x, mask_indices=mask_indices)

    assert out is x
    assert out.dtype == torch.bfloat16
    assert returned_indices is mask_indices

    expected_emb = mask_emb.to(dtype=torch.bfloat16)
    assert torch.equal(out[mask_indices], expected_emb.expand_as(out[mask_indices]))
    assert torch.equal(out[~mask_indices], x_orig[~mask_indices])


def test_apply_mask_no_mask_returns_none_indices() -> None:
    """mask_indices=None 且 mask_prob=0 时不改写 x，返回的 indices 为 None。"""
    torch.manual_seed(1)
    x = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    x_before = x.clone()
    mask_emb = torch.randn(8, dtype=torch.float32)

    host = _DummyMaskHost(mask_emb=mask_emb, mask_prob=0.0)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, returned_indices = WavLM.apply_mask(host, x)

    assert returned_indices is None
    assert torch.equal(out, x_before)
