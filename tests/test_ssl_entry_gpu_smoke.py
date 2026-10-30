"""官方 SSL 模型经真实 train.py 入口的 CUDA 训练验收。"""

from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from core.utils.safe_load import safe_torch_load
from tasks.ssl.model_io import resolve_ssl_model_spec
from tasks.ssl.models import build_model
from test_multitask_entry_audit_smoke import _ssl_config
from test_ssl_official_model_smoke import _ipdnet_fixed_cfg, _ipdnet_variable_cfg


pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.subprocess]
_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model_type", ["IPDNetFixed", "IPDNetVariable"])
def test_official_ssl_cuda_root_entry_epoch(model_type, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SSL entry acceptance")
    exp_path = tmp_path / "experiment"
    config = _ssl_config(tmp_path, exp_path)
    model_cfg = (
        _ipdnet_fixed_cfg() if model_type == "IPDNetFixed" else _ipdnet_variable_cfg()
    )
    model = build_model(model_cfg, stft_cfg=config["stft"])
    config["model_spec"] = asdict(resolve_ssl_model_spec(model))
    config["model"] = model_cfg
    # 让真实入口根据 canonical 声源数注入模型 K，不能依赖构模默认值。
    config["model"]["args"].pop("max_track")
    config["ssl"] = {
        "input_representation": "multichannel_stft_real_imag",
        "target_representation": "dpipd_far_field",
        "num_tracks": 2,
        "vad_mode": "direct_path",
        "time_stride": 12,
        "normalization": "none",
    }
    config["loss"] = {
        "type": "SSLDOACompositeLoss", "strict": True, "args": {"max_tracks": 2}
    }
    for split in ("train", "validation"):
        config["data"][split]["segment_seconds"] = 200 / 8000
    config["data"]["dataloader"]["pin_memory"] = True
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    result = subprocess.run(
        [sys.executable, "train.py", "-C", str(config_path)],
        cwd=_ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=120,
    )
    assert result.returncode == 0, result.stdout
    assert "Training complete." in result.stdout
    assert "[Val] epoch 1" in result.stdout
    checkpoints = list((exp_path / "checkpoints").glob("*/latest_model.tar"))
    assert len(checkpoints) == 1, result.stdout
    # 仅加载本测试刚刚生成的可信训练状态，包含 optimizer 与 RNG 的 pickle 数据。
    state = safe_torch_load(
        checkpoints[0], map_location="cpu", allow_unsafe=True,
        purpose="Inspect the trusted SSL CUDA smoke checkpoint created by this test",
    )
    assert state["n_steps"] > 0
    assert state["curve_epochs"] == [1]
    assert torch.isfinite(torch.tensor(state["train_losses"] + state["val_losses"])).all()
    assert state["rng_state"]["torch_cuda"]
    assert state["config"]["ssl"]["num_tracks"] == 2
    assert state["config"]["loss"]["type"] == "SSLDOACompositeLoss"
