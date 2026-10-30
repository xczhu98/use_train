"""StreamingGTCRN 转换与离线 parity 回归。

历史回归：
1. ``convert_gtcrn_to_streaming`` 在框架 GTCRN 容器改名为 ``layers`` 后未同步，
   官方命名的 ``en_convs``/``de_convs`` 查找全部失败。
2. 流式 DPGRNN 长期缺失 ``inter_fc`` 投影，导致流式/离线数值发散
   （旧发布资产 gtcrn_offical.onnx 同样携带该 bug）。
"""

import torch
import pytest

pytestmark = pytest.mark.unit


def _build_pair():
    from onnx.models.streaming_gtcrn import StreamingGTCRN, convert_gtcrn_to_streaming
    from tasks.se.models.discriminative.gtcrn import GTCRN

    torch.manual_seed(42)
    base = GTCRN(n_fft=512).eval()
    stream = StreamingGTCRN(n_fft=512, hop_length=256, win_length=512).eval()
    convert_gtcrn_to_streaming(stream, base)
    return base, stream


def test_convert_maps_all_streaming_keys():
    """转换必须填充全部 streaming 权重（无未映射 key、无残留零初始化）。"""
    base, stream = _build_pair()
    for name, tensor in stream.state_dict().items():
        assert torch.isfinite(tensor).all(), f"{name} 含非有限值"
    # 抽查 dpgrnn 权重逐位一致
    for key, value in base.dpgrnn1.state_dict().items():
        assert torch.equal(stream.dpgrnn1.state_dict()[key], value), key


def test_streaming_matches_offline_frame_by_frame():
    """随机初始化权重下，逐帧流式输出必须与整段离线输出一致。"""
    base, stream = _build_pair()
    torch.manual_seed(7)
    frames = 8
    spec = torch.randn(1, 257, frames, 2)
    with torch.inference_mode():
        offline = base(spec)
        cache = stream.init_cache(1, torch.device("cpu"), torch.float32)
        outs = []
        caches = [cache[key] for key in stream.CACHE_KEYS]
        for index in range(frames):
            out, *caches = stream(spec[:, :, index : index + 1, :], *caches)
            outs.append(out)
        online = torch.cat(outs, dim=2)
    max_err = (offline - online).abs().max().item()
    assert max_err < 1e-5, f"流式/离线发散: max_abs={max_err:.3e}"


def test_streaming_dpgrnn_applies_inter_fc():
    """流式 DPGRNN inter 路径必须应用 inter_fc 投影（防历史缺失回归）。"""
    from onnx.models.streaming_gtcrn import DPGRNN

    torch.manual_seed(0)
    mod = DPGRNN(16, 33, 16).eval()
    x = torch.randn(1, 16, 1, 33)
    h = torch.zeros(1, 33, 16)
    with torch.inference_mode():
        out_with, _ = mod(x, h)
        # 手动绕过 inter_fc 的参考实现；若 forward 缺失 inter_fc，两者会相等
        xp = x.permute(0, 2, 3, 1)
        intra = mod.intra_ln(
            mod.intra_fc(mod.intra_rnn(xp.reshape(1, 33, 16))[0]).reshape(1, 1, 33, 16)
        )
        intra_out = xp + intra
        x2 = intra_out.permute(0, 2, 1, 3).reshape(33, 1, 16)
        inter_x, _ = mod.inter_rnn(x2, h)  # 故意不经过 inter_fc
        inter_x = inter_x.reshape(1, 33, 1, 16).permute(0, 2, 1, 3)
        out_without = (intra_out + mod.inter_ln(inter_x)).permute(0, 3, 1, 2)
    assert not torch.allclose(out_with, out_without), "forward 似乎未应用 inter_fc"
