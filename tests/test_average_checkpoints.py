"""tools/model_tools/average_checkpoints.py 的单元测试。

合成 3 个同构 checkpoint（不同权重），验证：
  - best 模式按文件名 val_loss 升序选 top N；
  - final 模式按 epoch 序号选最后 N 个 model_XXXX.pth；
  - float 键均值正确、int/buffer 键取第一个；
  - EMA shadow_params 均值正确；
  - 输出可被 torch.load 且键齐全（含 config/resolved_config，剔除 optimizer 等）。
"""

from __future__ import annotations
import pytest

import sys
import unittest
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.model_tools.average_checkpoints import (  # noqa: E402
    _average_ema,
    _average_state_dicts,
    average_checkpoints,
    select_checkpoints,
)
from core.utils.safe_load import safe_torch_load  # noqa: E402


def _make_state_dict(scale: float) -> dict:
    """合成一个含 float 参数与 int buffer 的 state_dict。"""
    return {
        "fc.weight": torch.full((2, 2), scale),
        "fc.bias": torch.full((2,), scale * 10),
        "bn.num_batches_tracked": torch.tensor(7, dtype=torch.long),  # int buffer：不平均
    }


def _make_ema(scale: float) -> dict:
    """合成 torch_ema 风格的 state_dict（shadow_params 为参数列表）。"""
    return {
        "decay": 0.999,
        "num_updates": 100,
        "collected_params": None,
        "shadow_params": [torch.full((2, 2), scale), torch.full((2,), scale * 10)],
    }


def _make_ckpt(scale: float, with_ema: bool = True) -> dict:
    """合成 BaseTrainer 同构 ckpt（仅保留本测试关心的键）。"""
    ckpt = {
        "epoch": 1,
        "n_steps": 100,
        "model": _make_state_dict(scale),
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"last_epoch": 1},
        "config": {"model": {"type": "Toy"}, "meta": {"scale": scale}},
        "resolved_config": {"model": {"type": "Toy"}, "meta": {"scale": scale}},
    }
    if with_ema:
        ckpt["ema"] = _make_ema(scale)
    return ckpt


class _Base(unittest.TestCase):
    """公共脚手架：tmp 目录下落 3 个同构 ckpt。"""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        # 3 个 best_step ckpt：val_loss 分别 0.30 / 0.10 / 0.20，权重 scale 1/2/3
        torch.save(_make_ckpt(1.0), self.dir / "best_step=000100-val_loss=0.300.tar")
        torch.save(_make_ckpt(2.0), self.dir / "best_step=000200-val_loss=0.100.tar")
        torch.save(_make_ckpt(3.0), self.dir / "best_step=000300-val_loss=0.200.tar")
        # 3 个 epoch ckpt：权重 scale 4/5/6
        torch.save(_make_ckpt(4.0), self.dir / "model_0001.pth")
        torch.save(_make_ckpt(5.0), self.dir / "model_0002.pth")
        torch.save(_make_ckpt(6.0), self.dir / "model_0003.pth")

    def tearDown(self):
        self._tmp.cleanup()


class TestSelectCheckpoints(_Base):
    def test_best_mode_picks_lowest_val_loss(self):
        """best 模式按 val_loss 升序取 top 2：0.100 与 0.200。"""
        selected = select_checkpoints(self.dir, "best", 2)
        self.assertEqual(
            [p.name for p in selected],
            ["best_step=000200-val_loss=0.100.tar", "best_step=000300-val_loss=0.200.tar"],
        )

    def test_final_mode_picks_last_epochs(self):
        """final 模式按 epoch 序号取最后 2 个 model_XXXX.pth。"""
        selected = select_checkpoints(self.dir, "final", 2)
        self.assertEqual([p.name for p in selected], ["model_0002.pth", "model_0003.pth"])

    def test_best_mode_no_match_raises(self):
        """目录内无匹配文件时应报错。"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                select_checkpoints(d, "best", 2)

    def test_final_mode_epoch_ckpt_sorts_after_same_epoch_mid(self):
        """final 模式：同 epoch 的 epoch 末 ckpt 必须排在该 epoch 的中间 ckpt 之后。

        回归：旧键 (epoch, -1) 把 model_XXXX.pth 排在 model_E*_B*.pth 之前，
        取"最后 N 个"时会把 epoch 末 ckpt 漏掉。
        """
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            torch.save(_make_ckpt(1.0), d / "model_0001.pth")
            torch.save(_make_ckpt(2.0), d / "model_E0001_B0050.pth")
            torch.save(_make_ckpt(3.0), d / "model_E0001_B0100.pth")
            # 最后 2 个 = 最晚的中间 ckpt + epoch 末 ckpt
            selected = select_checkpoints(d, "final", 2)
            self.assertEqual(
                [p.name for p in selected],
                ["model_E0001_B0100.pth", "model_0001.pth"],
            )
            # num=1 时应取 epoch 末 ckpt（而非最后一个 mid ckpt）
            selected = select_checkpoints(d, "final", 1)
            self.assertEqual([p.name for p in selected], ["model_0001.pth"])

    def test_best_mode_negative_and_scientific_val_loss(self):
        """best 模式：val_loss 支持负数与科学计数法，并按数值升序排序。"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            torch.save(_make_ckpt(1.0), d / "best_step=000100-val_loss=-15.231.tar")
            torch.save(_make_ckpt(2.0), d / "best_step=000200-val_loss=1.5e-3.tar")
            torch.save(_make_ckpt(3.0), d / "best_step=000300-val_loss=0.01.tar")
            selected = select_checkpoints(d, "best", 2)
            self.assertEqual(
                [p.name for p in selected],
                [
                    "best_step=000100-val_loss=-15.231.tar",
                    "best_step=000200-val_loss=1.5e-3.tar",
                ],
            )

    def test_best_mode_unparseable_glob_hit_warns_and_skips(self):
        """best 模式：glob 命中但文件名无法解析时打 warning 并跳过，不静默。"""
        import contextlib
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            torch.save(_make_ckpt(1.0), d / "best_step=000100-val_loss=0.100.tar")
            # glob 命中但 step/val_loss 字段非法（如手工改名），regex 不匹配
            torch.save(_make_ckpt(2.0), d / "best_step=abc-val_loss=xyz.tar")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                selected = select_checkpoints(d, "best", 1)
            self.assertEqual(
                [p.name for p in selected], ["best_step=000100-val_loss=0.100.tar"]
            )
            self.assertIn("[WARN]", buf.getvalue())
            self.assertIn("best_step=abc-val_loss=xyz.tar", buf.getvalue())


class TestAverageStateDicts(unittest.TestCase):
    def test_float_mean_and_int_passthrough(self):
        """float 键取均值；int buffer 取第一个。"""
        sds = [_make_state_dict(s) for s in (1.0, 2.0, 3.0)]
        avg = _average_state_dicts(sds)
        self.assertTrue(torch.allclose(avg["fc.weight"], torch.full((2, 2), 2.0)))
        self.assertTrue(torch.allclose(avg["fc.bias"], torch.full((2,), 20.0)))
        # int buffer 不参与平均，取第一个 ckpt 的值
        self.assertEqual(avg["bn.num_batches_tracked"].item(), 7)
        self.assertEqual(avg["bn.num_batches_tracked"].dtype, torch.long)

    def test_mismatched_keys_raise(self):
        """键集合不一致的 ckpt 不允许平均。"""
        sd1 = _make_state_dict(1.0)
        sd2 = _make_state_dict(2.0)
        sd2.pop("fc.bias")
        with self.assertRaises(ValueError):
            _average_state_dicts([sd1, sd2])


class TestAverageEma(unittest.TestCase):
    def test_shadow_params_averaged(self):
        """ema.shadow_params 逐元素求均值；decay/num_updates 取第一个。"""
        ckpts = [_make_ckpt(s) for s in (1.0, 2.0, 3.0)]
        avg = _average_ema(ckpts)
        self.assertIsNotNone(avg)
        self.assertTrue(torch.allclose(avg["shadow_params"][0], torch.full((2, 2), 2.0)))
        self.assertEqual(avg["decay"], 0.999)

    def test_missing_ema_returns_none(self):
        """部分 ckpt 缺 ema 键时返回 None（输出不写 ema）。"""
        ckpts = [_make_ckpt(1.0), _make_ckpt(2.0, with_ema=False)]
        self.assertIsNone(_average_ema(ckpts))


class TestAverageCheckpointsEndToEnd(_Base):
    def test_best_mode_output_loadable(self):
        """best 模式端到端：top-2（scale 2,3）均值 2.5；输出键齐全可安全加载。"""
        out = average_checkpoints(self.dir, "best", 2, self.dir / "avg_best2.tar")
        ckpt = safe_torch_load(out, map_location="cpu")
        # 均值正确（scale 2 与 3 的均值）
        self.assertTrue(torch.allclose(ckpt["model"]["fc.weight"], torch.full((2, 2), 2.5)))
        self.assertTrue(torch.allclose(ckpt["model"]["fc.bias"], torch.full((2,), 25.0)))
        # 元数据键齐全：config/resolved_config 取自第一个 ckpt
        self.assertIn("config", ckpt)
        self.assertIn("resolved_config", ckpt)
        self.assertIn("ema", ckpt)
        self.assertTrue(torch.allclose(ckpt["ema"]["shadow_params"][0], torch.full((2, 2), 2.5)))
        # optimizer/scheduler 已剔除；溯源信息存在
        self.assertNotIn("optimizer", ckpt)
        self.assertNotIn("scheduler", ckpt)
        self.assertEqual(ckpt["averaged_num"], 2)
        self.assertEqual(len(ckpt["averaged_from"]), 2)

    def test_final_mode_output(self):
        """final 模式端到端：最后 2 个 epoch ckpt（scale 5,6）均值 5.5。"""
        out = average_checkpoints(self.dir, "final", 2, self.dir / "avg_final2.tar")
        ckpt = safe_torch_load(out, map_location="cpu")
        self.assertTrue(torch.allclose(ckpt["model"]["fc.weight"], torch.full((2, 2), 5.5)))
        self.assertEqual(ckpt["averaged_mode"], "final")


if __name__ == "__main__":
    unittest.main()

pytestmark = [pytest.mark.unit]
