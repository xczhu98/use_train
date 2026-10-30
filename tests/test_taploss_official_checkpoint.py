"""真实官方 TAP estimator checkpoint 的安全加载与前向验证。

默认不联网：设置 ``TAPLOSS_OFFICIAL_CHECKPOINT`` 指向已下载 checkpoint；或显式
设置 ``TAPLOSS_DOWNLOAD_OFFICIAL_CHECKPOINT=1``，测试才会从固定官方 commit 下载到
pytest 临时目录。无论来源，均在反序列化前校验 SHA256。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

import pytest
import torch

from core.losses.tap import TAPLoss, _TAPAcousticEstimator
from core.utils.safe_load import safe_torch_load
from tools.model_tools.convert_taploss_official_checkpoint import convert_checkpoint

pytestmark = pytest.mark.external

_OFFICIAL_COMMIT = "d1fa133d80161bb837f2874d0b18eed89d5bf554"
_OFFICIAL_URL = (
    "https://raw.githubusercontent.com/YunyangZeng/TAPLoss/"
    f"{_OFFICIAL_COMMIT}/TAPLoss/TAP_estimator_model_v0.pt"
)
_OFFICIAL_SHA256 = "b6455bba8f282207f8dba8c38ef4c0bc410f76ab8f732d1d6f2de85ee05c4b72"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def official_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    configured_path = os.environ.get("TAPLOSS_OFFICIAL_CHECKPOINT")
    if configured_path:
        checkpoint_path = Path(configured_path).expanduser()
        if not checkpoint_path.is_file():
            pytest.fail(
                "TAPLOSS_OFFICIAL_CHECKPOINT does not point to a file: "
                f"{checkpoint_path}"
            )
    elif os.environ.get("TAPLOSS_DOWNLOAD_OFFICIAL_CHECKPOINT") == "1":
        checkpoint_path = (
            tmp_path_factory.mktemp("taploss_official")
            / "TAP_estimator_model_v0.pt"
        )
        with urlopen(_OFFICIAL_URL, timeout=120) as response:  # noqa: S310
            checkpoint_path.write_bytes(response.read())
    else:
        pytest.skip(
            "set TAPLOSS_OFFICIAL_CHECKPOINT to a local file, or explicitly set "
            "TAPLOSS_DOWNLOAD_OFFICIAL_CHECKPOINT=1 to download the official checkpoint"
        )

    actual_sha256 = _sha256(checkpoint_path)
    assert actual_sha256 == _OFFICIAL_SHA256, (
        "official TAP checkpoint SHA256 mismatch: "
        f"expected {_OFFICIAL_SHA256}, got {actual_sha256}"
    )
    return checkpoint_path


def test_official_checkpoint_converts_safely_and_backpropagates(
    official_checkpoint: Path,
    tmp_path: Path,
) -> None:
    converted_path = tmp_path / "tap_estimator_tensor_only.pt"
    source_sha256, _output_sha256, tensor_count, parameter_count = convert_checkpoint(
        official_checkpoint, converted_path
    )
    assert source_sha256 == _OFFICIAL_SHA256
    assert tensor_count == 28
    assert parameter_count == 4_872_729
    assert 19_000_000 < converted_path.stat().st_size < 20_000_000

    state_dict = safe_torch_load(converted_path, map_location="cpu")
    assert isinstance(state_dict, dict)
    assert all(type(value) is torch.Tensor for value in state_dict.values())
    _TAPAcousticEstimator().load_state_dict(state_dict, strict=True)

    # TAPLoss 再次使用普通 safe_torch_load，并以 strict=True 构造 estimator。
    loss = TAPLoss(acoustic_model_path=converted_path, device="cpu")
    generator = torch.Generator().manual_seed(0)
    enhanced = torch.randn(1, 768, generator=generator, requires_grad=True)
    clean = torch.randn(1, 768, generator=generator)
    value = loss(enhanced, clean)
    value.backward()

    assert value.shape == ()
    assert torch.isfinite(value)
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()
    assert all(parameter.grad is None for parameter in loss.estimate_acoustics.parameters())
