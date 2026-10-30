"""Numerical parity test: ported GAP-URGENet modules vs. the official gap-urgenet repo.

tasks/se/models/pase_family/gap_urgenet/ is a 1:1 ported port of
the official gap-urgenet repo checkout (Predictor / PostNet；生成支路复用
unipase 包，其 parity 由 tests/test_unipase_official_parity.py 覆盖)。
This test loads the official pretrained checkpoints into BOTH implementations
and compares outputs on identical random inputs (bit-exact expected: Phase 1
移植时实测 max_abs_diff=0.0)。

Skipped unless both are present:
  - official repo checkout: env var GAP_URGENET_SOURCE_REPO (default
    /opt/nonexistent/gap-urgenet)
  - official weights:       env var GAP_URGENET_CKPT_DIR (default
    tasks/se/models/pase_family/gap_urgenet/open_model_checkpoint)
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_SOURCE_REPO = Path(os.environ.get("GAP_URGENET_SOURCE_REPO", "/opt/nonexistent/gap-urgenet"))
_CKPT_DIR = Path(os.environ.get(
    "GAP_URGENET_CKPT_DIR", "tasks/se/models/pase_family/gap_urgenet/open_model_checkpoint"
))

_REQUIRED_CKPTS = ["Predictor.pt", "PostNet.pt"]

pytestmark = [pytest.mark.external, pytest.mark.skipif(
    not _SOURCE_REPO.is_dir() or not all((_CKPT_DIR / f).is_file() for f in _REQUIRED_CKPTS),
    reason=(
        f"need official repo (GAP_URGENET_SOURCE_REPO, got {_SOURCE_REPO}) and "
        f"official weights (GAP_URGENET_CKPT_DIR, got {_CKPT_DIR})"
    ),
)]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _deterministic_backends():
    """bit-exact parity 断言要求确定性 CUDA 后端（同 test_unipase_official_parity）。"""
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.benchmark = prev_benchmark
    torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32


@pytest.fixture(autouse=True)
def _clean_sys_path():
    """测试后从 sys.path 移除官方 repo，避免其顶层 ``models`` 包对同 session
    其他测试造成 import 阴影（review P3-9）。sys.modules 里已加载的
    ``models`` / ``models.*`` 模块一并清除——否则后续 unipase parity 测试
    import ``models.postnet.tfgrid_cws_res`` 时会命中本 repo 的 models 包
    而报 ModuleNotFoundError。``_stub_optional_deps`` 注入过的 stub 模块
    （espnet2/omegaconf/einops，仅在实际注入时）同样弹出。"""
    yield
    p = str(_SOURCE_REPO)
    while p in sys.path:
        sys.path.remove(p)
    for name in [m for m in sys.modules if m == "models" or m.startswith("models.")]:
        sys.modules.pop(name, None)
    for name in _STUBBED_MODS:
        sys.modules.pop(name, None)
    _STUBBED_MODS.clear()


# _stub_optional_deps 实际注入过 stub 的模块名（teardown 时清理）
_STUBBED_MODS: list[str] = []


def _stub_optional_deps() -> None:
    """官方文件 import 的 espnet2 / omegaconf / einops 缺失时注入最小 stub。

    官方 models/predictor/tfgridnet.py 实际只用到 AbsSeparator（空基类）与
    get_layer("prelu")（即 nn.PReLU）；models/postnet/tfgrid_cws.py 额外用到
    einops.rearrange 的一个固定 pattern。stub 与原依赖数值行为一致。
    """

    def _ensure_mod(name, **attrs):
        mod = sys.modules.get(name) or types.ModuleType(name)
        if name not in sys.modules:
            _STUBBED_MODS.append(name)
        sys.modules[name] = mod
        for k, v in attrs.items():
            setattr(mod, k, v)

    try:
        import espnet2  # noqa: F401
    except ImportError:
        _ensure_mod("espnet2")
        _ensure_mod("espnet2.enh")
        _ensure_mod("espnet2.enh.separator")
        _ensure_mod("espnet2.enh.separator.abs_separator", AbsSeparator=nn.Module)

        def get_layer(name):
            assert name == "prelu"
            return nn.PReLU

        _ensure_mod("espnet2.torch_utils")
        _ensure_mod("espnet2.torch_utils.get_layer_from_string", get_layer=get_layer)
    try:
        import omegaconf  # noqa: F401
    except ImportError:
        _ensure_mod("omegaconf", OmegaConf=None)
    try:
        import einops  # noqa: F401
    except ImportError:
        def rearrange(x, pattern):
            assert pattern == "b c f t ri -> b (c ri) t f"
            b, c, f, t, ri = x.shape
            return x.permute(0, 1, 4, 3, 2).reshape(b, c * ri, t, f)

        _ensure_mod("einops", rearrange=rearrange)


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.shape[-1], b.shape[-1])
    return (a[..., :n] - b[..., :n]).abs().max().item()


def _assert_parity(a: torch.Tensor, b: torch.Tensor) -> None:
    """CPU 要求逐比特一致（已实测 max_abs_diff == 0.0）；CUDA 上 10 层双向 LSTM
    堆叠存在 cuDNN 算法选择的 ULP 级噪声（实测 ~1.5e-8，2^-26），放宽到 1e-6。"""
    assert a.shape == b.shape, f"输出形状不一致: {tuple(a.shape)} vs {tuple(b.shape)}"
    diff = _max_diff(a, b)
    if _DEVICE.type == "cuda":
        assert diff < 1e-6, f"CUDA parity diff={diff:.3e} 超出 ULP 容差"
    else:
        assert diff == 0.0, f"CPU parity diff={diff:.3e} != 0.0"


def test_gap_urgenet_predictor_parity():
    _stub_optional_deps()
    if str(_SOURCE_REPO) not in sys.path:
        sys.path.insert(0, str(_SOURCE_REPO))
    from models.predictor.tfgridnet import TFGridNet as OfficialPredictor  # noqa: E402
    from tasks.se.models.pase_family.gap_urgenet.predictor.tfgridnet import TFGridNet  # noqa: E402

    ckpt = str(_CKPT_DIR / "Predictor.pt")
    off = OfficialPredictor.from_pretrained(ckpt).to(_DEVICE)
    ours = TFGridNet.from_pretrained(ckpt).to(_DEVICE)

    torch.manual_seed(0)
    wav16k = torch.randn(1, 16000, device=_DEVICE) * 0.1
    with torch.inference_mode():
        _assert_parity(off(wav16k, sr=16000), ours(wav16k, sr=16000))


def test_gap_urgenet_postnet_parity():
    _stub_optional_deps()
    if str(_SOURCE_REPO) not in sys.path:
        sys.path.insert(0, str(_SOURCE_REPO))
    from models.postnet.tfgrid_cws import TFGridNet as OfficialPostNet  # noqa: E402
    from tasks.se.models.pase_family.gap_urgenet.postnet.tfgrid_cws import TFGridNet  # noqa: E402

    ckpt = str(_CKPT_DIR / "PostNet.pt")
    off = OfficialPostNet.from_pretrained(ckpt).to(_DEVICE)
    ours = TFGridNet.from_pretrained(ckpt).to(_DEVICE)

    torch.manual_seed(0)
    wav2ch48k = torch.randn(1, 2, 48000, device=_DEVICE) * 0.1
    with torch.inference_mode():
        _assert_parity(off(wav2ch48k, sr_in=48000), ours(wav2ch48k, sr_in=48000))
