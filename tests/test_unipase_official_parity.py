"""Numerical parity test: ported UniPASE modules vs. the official unipase repo.

tasks/se/models/pase_family/unipase/ is a 1:1 vendored port of
the official unipase repo checkout (Adapter / Vocoder / PostNet).
This test loads the official pretrained checkpoints into BOTH implementations
and compares outputs on identical random inputs (bit-exact expected: our
verify script tools/verify/verify_unipase_official_weights.py measured
max_abs_diff=0.0 end-to-end).

Skipped unless both are present:
  - official repo checkout: env var UNIPASE_SOURCE_REPO (default
    /opt/nonexistent/unipase)
  - official weights:       env var UNIPASE_CKPT_DIR (default
    tasks/se/models/pase_family/unipase/open_model_checkpoint)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_SOURCE_REPO = Path(os.environ.get("UNIPASE_SOURCE_REPO", "/opt/nonexistent/unipase"))
_CKPT_DIR = Path(os.environ.get(
    "UNIPASE_CKPT_DIR", "tasks/se/models/pase_family/unipase/open_model_checkpoint"
))

_REQUIRED_CKPTS = ["Adapter.pt", "Vocoder_DWO-L1.pt", "PostNet.pt"]

pytestmark = [pytest.mark.external, pytest.mark.skipif(
    not _SOURCE_REPO.is_dir() or not all((_CKPT_DIR / f).is_file() for f in _REQUIRED_CKPTS),
    reason=(
        f"need official repo (UNIPASE_SOURCE_REPO, got {_SOURCE_REPO}) and "
        f"official weights (UNIPASE_CKPT_DIR, got {_CKPT_DIR})"
    ),
)]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _deterministic_backends():
    """bit-exact parity 断言要求确定性 CUDA 后端。

    前序 trainer smoke 测试（stupase/pase/unipase GAN smoke 等）经
    ``BaseTrainer.__init__`` 把 ``cudnn.benchmark`` / ``matmul.allow_tf32``
    全局打开（core/trainer/base_trainer.py:101-110 的 CUDA 加速默认），
    benchmark 的 conv 算法自动选择带时序噪声，会让两个同权重 TFGridNet
    实例在 GPU 上产生 ~1 ulp 的差异，使 ``== 0.0`` 断言偶发失败。
    这里固定为默认确定性配置，测试后恢复原值。
    """
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
    ``models`` / ``models.*`` 模块一并清除——否则后续 gap-urgenet parity 测试
    import ``models.predictor.tfgridnet`` 时会命中本 repo 的 models 包
    而报 ModuleNotFoundError（与 test_gap_urgenet_official_parity 对称）。"""
    yield
    p = str(_SOURCE_REPO)
    while p in sys.path:
        sys.path.remove(p)
    for name in [m for m in sys.modules if m == "models" or m.startswith("models.")]:
        sys.modules.pop(name, None)


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.shape[-1], b.shape[-1])
    return (a[..., :n] - b[..., :n]).abs().max().item()


def test_unipase_adapter_parity():
    if str(_SOURCE_REPO) not in sys.path:
        sys.path.insert(0, str(_SOURCE_REPO))
    from models.adapter.vocos.adapter import VocosAdapter as OfficialAdapter  # noqa: E402
    from tasks.se.models.pase_family.unipase.adapter import VocosAdapter  # noqa: E402

    ckpt = str(_CKPT_DIR / "Adapter.pt")
    off = OfficialAdapter.from_pretrained(ckpt).to(_DEVICE)
    ours = VocosAdapter.from_pretrained(ckpt).to(_DEVICE)

    torch.manual_seed(0)
    ea = torch.randn(1, 100, 1024, device=_DEVICE)
    ep = torch.randn(1, 100, 1024, device=_DEVICE)
    with torch.inference_mode():
        assert _max_diff(off(ea, ep), ours(ea, ep)) == 0.0


def test_unipase_vocoder_parity():
    if str(_SOURCE_REPO) not in sys.path:
        sys.path.insert(0, str(_SOURCE_REPO))
    from models.vocoder.vocos.vocoder import VocosVocoder as OfficialVocoder  # noqa: E402
    from tasks.se.models.pase_family.unipase.vocoder import VocosVocoder  # noqa: E402

    ckpt = str(_CKPT_DIR / "Vocoder_DWO-L1.pt")
    off = OfficialVocoder.from_pretrained(ckpt).to(_DEVICE)
    ours = VocosVocoder.from_pretrained(ckpt).to(_DEVICE)

    torch.manual_seed(0)
    feat = torch.randn(1, 1024, 100, device=_DEVICE)
    with torch.inference_mode():
        assert _max_diff(off(feat), ours(feat)) == 0.0


def test_unipase_postnet_parity():
    if str(_SOURCE_REPO) not in sys.path:
        sys.path.insert(0, str(_SOURCE_REPO))
    from models.postnet.tfgrid_cws_res import TFGridNet as OfficialPostNet  # noqa: E402
    from tasks.se.models.pase_family.unipase.postnet import TFGridNet  # noqa: E402

    ckpt = str(_CKPT_DIR / "PostNet.pt")
    off = OfficialPostNet.from_pretrained(ckpt).to(_DEVICE)
    ours = TFGridNet.from_pretrained(ckpt).to(_DEVICE)

    torch.manual_seed(0)
    wav16k = torch.randn(1, 16000, device=_DEVICE) * 0.1
    with torch.inference_mode():
        # 推理路径为软门控残差（infer），训练路径 forward 也要对齐
        assert _max_diff(off.infer(wav16k, sr_in=16000, sr_out=48000),
                         ours.infer(wav16k, sr_in=16000, sr_out=48000)) == 0.0
        assert _max_diff(off(wav16k), ours(wav16k)) == 0.0
