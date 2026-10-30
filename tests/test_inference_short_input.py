"""inference.py 根 CLI 辅助函数的回归测试。

覆盖与具体输入无关的 root CLI 契约：

- AEC 模型只声明实际消费的谱输入（`aec_input_capabilities`）；
- AEC 采样率解析只接受与配置精确匹配的正整数；
- AEC frontend 三态 CLI 覆盖只改 `delay_alignment.enabled`，并要求显式配置，
  旧的 `aec_stream_align` 配置在边界被拒绝；
- runtime 设备迁移在部分迁移失败时回滚到请求设备。

短输入补齐、空输入拒绝、AEC batch 分帧与 frontend routing 等逻辑随旧 batch 循环
（`run_se_inference` / `run_aec_inference_batch`）一并移除，现由 canonical 文件推理路径
（`core.inference` + `tasks/*/inference/adapter.py`）承担，覆盖见
`test_se_inference_adapter.py`、`test_core_inference.py` 与
`test_eval_public_inference_entrypoints.py`。
"""

from __future__ import annotations

import pytest
import torch

import inference
from tasks.aec.inference.device_policy import move_aec_runtime as _move_aec_runtime
from tasks.aec.models.discriminative.cagcrn import CAGCRN
from tasks.aec.models.discriminative.dbgnet import DBGNet
from tasks.aec.models.discriminative.deepvqe import DeepVQE
from tasks.aec.models.discriminative.localvqe import LocalVQE


class _TrackedDeviceComponent:
    def __init__(self, device="cuda:0", *, fail_on=None):
        self.device = torch.device(device)
        self.fail_on = None if fail_on is None else torch.device(fail_on)
        self.moves = []

    def to(self, device):
        target = torch.device(device)
        self.moves.append(target)
        if target == self.fail_on:
            raise RuntimeError(f"cannot move to {target}")
        self.device = target
        return self


@pytest.mark.parametrize("model_type", [DBGNet, CAGCRN, LocalVQE, DeepVQE])
def test_aec_models_declare_only_consumed_inference_spectra(model_type):
    assert model_type.aec_input_capabilities == frozenset(
        {"mic_spectra", "farend_spectra"}
    )


def test_aec_sample_rate_only_accepts_exact_config_value():
    assert inference._resolve_aec_sample_rate(16000) == 16000
    assert inference._resolve_aec_sample_rate(32000, 32000) == 32000
    with pytest.raises(ValueError, match="must match configured"):
        inference._resolve_aec_sample_rate(32000, 16000)


@pytest.mark.parametrize("configured_sr", [None, True, 0, -1, 16000.0, "16000"])
def test_aec_sample_rate_requires_positive_config_integer(configured_sr):
    with pytest.raises(ValueError, match="data.sample_rate"):
        inference._resolve_aec_sample_rate(configured_sr)


def test_aec_frontend_cli_default_reuses_configured_value():
    """未给 CLI flag 时完整保留 frontend 对齐开关。"""
    cfg = {
        "aec_frontend": {
            "delay_alignment": {"enabled": True, "frame_size": 80},
        }
    }

    resolved = inference._resolve_aec_frontend_with_cli(cfg)
    assert resolved.delay_alignment.enabled is True
    assert resolved.delay_alignment.frame_size == 80
    assert resolved.input_routing == "mic_farend"


def test_aec_frontend_cli_override_does_not_accept_legacy_config():
    with pytest.raises(ValueError, match="legacy aec_stream_align"):
        inference._resolve_aec_frontend_with_cli(
            {"inference": {"aec_stream_align": {"enabled": False}}},
            force_enable=True,
        )


@pytest.mark.parametrize(
    ("force_enable", "force_disable", "expected"),
    [(True, False, True), (False, True, False)],
)
def test_aec_frontend_cli_only_overrides_delay_alignment_enabled(
    force_enable, force_disable, expected
):
    """CLI 三态覆盖只改 enabled，不漂移 routing、LinearAEC 或其他对齐参数。"""
    cfg = {
        "aec_frontend": {
            "delay_alignment": {"enabled": not expected, "frame_size": 80},
            "linear_aec": {"backend": "pfdkf"},
            "input_routing": "error_farend",
        }
    }
    baseline = inference.resolve_aec_frontend_config(cfg).to_dict()

    resolved = inference._resolve_aec_frontend_with_cli(
        cfg,
        force_enable=force_enable,
        force_disable=force_disable,
    ).to_dict()

    baseline["delay_alignment"]["enabled"] = expected
    assert resolved == baseline


def test_aec_frontend_cli_rejects_conflicting_flags():
    with pytest.raises(ValueError, match="mutually exclusive"):
        inference._resolve_aec_frontend_with_cli(
            {}, force_enable=True, force_disable=True
        )


def test_aec_runtime_device_migration_rolls_back_partial_move():
    model = _TrackedDeviceComponent()
    stft = _TrackedDeviceComponent(fail_on="cpu")

    with pytest.raises(RuntimeError, match="cannot move to cpu"):
        _move_aec_runtime(model, stft, "cpu", "cuda:0")

    assert model.device == torch.device("cuda:0")
    assert stft.device == torch.device("cuda:0")
    assert model.moves == [torch.device("cpu"), torch.device("cuda:0")]
    assert stft.moves == [torch.device("cpu"), torch.device("cuda:0")]


def test_aec_runtime_device_migration_can_restore_requested_device():
    model = _TrackedDeviceComponent()
    stft = _TrackedDeviceComponent()

    _move_aec_runtime(model, stft, "cpu", "cuda:0")
    _move_aec_runtime(model, stft, "cuda:0", "cpu")

    assert model.device == torch.device("cuda:0")
    assert stft.device == torch.device("cuda:0")


pytestmark = [pytest.mark.unit]
