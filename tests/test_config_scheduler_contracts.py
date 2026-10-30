"""仓库配置中的调度器、数值类型与公开 mode 文档合同。"""

import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from core.trainer.entry_components import build_single_scheduler
from core.utils.config import load_merged_config
from tasks.registry import iter_tasks


pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = sorted(
    path for path in (_ROOT / "configs").rglob("*")
    if path.suffix in {".yaml", ".yml"}
)


def _scheduler_sections(config):
    scheduler = config.get("scheduler", {})
    if "type" in scheduler:
        return [("default", scheduler)]
    return list(scheduler.items())


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda path: str(path.relative_to(_ROOT)))
def test_training_config_schedulers_build_and_step(path):
    config = load_merged_config(path)
    meta = config.get("meta", {})
    if meta.get("inference_only") or meta.get("evaluation_only"):
        return
    sections = _scheduler_sections(config)
    assert sections, f"{path}: training configuration needs a scheduler"
    for name, section in sections:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer_config = config["optimizer"]
        if "type" not in optimizer_config:
            optimizer_config = optimizer_config[name]
        optimizer = torch.optim.SGD(
            [parameter], lr=optimizer_config.get("args", {}).get("lr", 0.001)
        )
        scheduler = build_single_scheduler(
            SimpleNamespace(), section, 32, optimizer, config,
            world_size=1, scheduler_name=name,
        )
        for _ in range(3):
            optimizer.zero_grad()
            parameter.sum().backward()
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(1.0)
            else:
                scheduler.step()
            assert all(
                math.isfinite(group["lr"]) and group["lr"] >= 0.0
                for group in optimizer.param_groups
            ), (path, name)


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda path: str(path.relative_to(_ROOT)))
def test_raw_scheduler_learning_rates_are_numeric(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, section in _scheduler_sections(config):
        for key, value in section.get("args", {}).items():
            if key in {"max_lr", "min_lr", "initial_lr", "final_lr", "eta_min"}:
                assert type(value) in (int, float) and math.isfinite(value), (
                    path, name, key, value
                )


def test_ssl_recipe_keeps_fixed_learning_rate():
    config = load_merged_config(
        _ROOT / "configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml"
    )
    parameter = torch.nn.Parameter(torch.ones(1))
    initial_lr = config["optimizer"]["args"]["lr"]
    optimizer = torch.optim.Adam([parameter], lr=initial_lr)
    scheduler = build_single_scheduler(
        SimpleNamespace(), config["scheduler"], 32, optimizer, config,
        world_size=1, scheduler_name="default",
    )
    for _ in range(10):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr)


def test_config_readme_mode_list_matches_task_registry():
    text = (_ROOT / "configs/README.md").read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if "**`data.mode` 是必填路由字段**" in line)
    documented = set(re.findall(r"`((?:dynamic|offline)(?:_[a-z]+)?)`", line))
    supported = set().union(*(task.data_modes for task in iter_tasks()))
    assert documented == supported
