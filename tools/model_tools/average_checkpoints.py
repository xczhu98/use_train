#!/usr/bin/env python3
"""Checkpoint 平均工具（SWA-like weight averaging，SE/AEC/TSE 三任务通用）。

训练结束后对 best/final N 个 checkpoint 的 raw 模型权重逐键求均值
（wesep 惯例；对 flat-minimum 有一定正则化效果，常小幅提升泛化）。

两种选 ckpt 模式（对应 BaseTrainer 在 ``exp_path/checkpoints/<ts>/`` 下的命名）：
  - best ：按 ``best_step=*-val_loss=*.tar`` 文件名中的 val_loss 升序取 top N；
  - final：按 epoch/batch 序号取最后 N 个 ``model_XXXX.pth``（含 ``model_E*_B*.pth`` 中间 ckpt）。

平均规则：
  - float/complex 类型的 tensor 键：逐键求算术平均；
  - int/bool 等其他类型键（如 BatchNorm 的 ``num_batches_tracked``）：直接取第一个 ckpt 的值
    （计数器无平均语义，保持确定性与形状一致）；
  - EMA：所有入选 ckpt 均含 ``ema`` 键时，对其 ``shadow_params`` 逐元素求平均后写入输出。
    加载侧（``inference.py`` / ``tasks/tse/inference/extractor.py``）会先 load ``model`` 再用
    ``ema.copy_to`` 覆盖为 EMA 权重，因此 EMA 存在时平均后的 EMA 权重自动优先生效，
    与 BaseTrainer/inference 的 EMA 语义一致。

输出写入同目录临时文件后 ``os.replace`` 原子发布，并在发布前 fsync；
读取统一走 ``core.utils.safe_load.safe_torch_load``（仓库禁止新增裸 ``torch.load``）。

输出与 BaseTrainer ckpt 同构：保留第一个 ckpt 的顶层元数据（含 ``config`` /
``resolved_config``，保证自包含可追溯），替换模型权重为平均值；剔除
optimizer/scheduler/scaler/rng_state/sampler_state（平均权重无可 resume 的优化状态），
并追加 ``averaged_from`` / ``averaged_mode`` / ``averaged_num`` 溯源信息。

用法：
    # 平均 val_loss 最小的 5 个 step-best ckpt
    python tools/model_tools/average_checkpoints.py \
        --ckpt_dir exp/se/xxx/checkpoints/<ts> --mode best --num 5 \
        --output exp/se/xxx/checkpoints/<ts>/avg_best5.tar

    # 平均最后 3 个 epoch ckpt
    python tools/model_tools/average_checkpoints.py \
        --ckpt_dir exp/se/xxx/checkpoints/<ts> --mode final --num 3 \
        --output exp/se/xxx/checkpoints/<ts>/avg_final3.tar
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

# 脚本位于 tools/model_tools/，需回退两级到 use_train 根目录（供 core.utils 导入）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


# BaseTrainer 的 ckpt 命名（见 core/trainer/base_trainer.py::_save_checkpoint 等）
# val_loss 允许负数（如 -15.231）与科学计数法（如 1.5e-3）
_BEST_STEP_RE = re.compile(r"^best_step=(\d+)-val_loss=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\.tar$")
_EPOCH_CKPT_RE = re.compile(r"^model_(\d+)\.pth$")
_MID_CKPT_RE = re.compile(r"^model_E(\d+)_B(\d+)\.pth$")


def select_checkpoints(ckpt_dir: str | Path, mode: str, num: int) -> list[Path]:
    """按模式挑选待平均的 checkpoint 文件，返回有序列表（平均顺序无关，仅用于日志/溯源）。"""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"ckpt_dir 不存在或不是目录: {ckpt_dir}")

    if mode == "best":
        # best_step=<step>-val_loss=<loss>.tar，按 val_loss 升序取 top N
        cands = []
        unparsed: list[str] = []
        for p in ckpt_dir.glob("best_step=*.tar"):
            m = _BEST_STEP_RE.match(p.name)
            if m:
                cands.append((float(m.group(2)), int(m.group(1)), p))
            else:
                # glob 命中但文件名解析失败（如手写/改名文件），显式告警而非静默跳过
                unparsed.append(p.name)
        if unparsed:
            print(f"[WARN] {len(unparsed)} 个 best_step=*.tar 文件名无法解析，已跳过: {unparsed[:5]}")
        cands.sort(key=lambda t: (t[0], t[1]))
        selected = [p for _, _, p in cands[:num]]
    else:  # final
        # model_XXXX.pth（epoch 序号）与 model_E*_B*.pth（中间 ckpt），按 (epoch, batch) 取最后 N 个
        cands = []
        for p in ckpt_dir.glob("model_*.pth"):
            m = _EPOCH_CKPT_RE.match(p.name)
            if m:
                # epoch 末 ckpt 视为该 epoch 的最后位置（同 epoch 的中间 ckpt
                # model_E*_B*.pth 键为 (epoch, batch_idx)，必须排在它之前）
                cands.append(((int(m.group(1)), sys.maxsize), p))
                continue
            m = _MID_CKPT_RE.match(p.name)
            if m:
                cands.append(((int(m.group(1)), int(m.group(2))), p))
        cands.sort(key=lambda t: t[0])
        selected = [p for _, p in cands[-num:]]

    if not selected:
        raise RuntimeError(
            f"在 {ckpt_dir} 未找到 mode={mode} 匹配的 checkpoint"
            f"（best 需要 best_step=*-val_loss=*.tar；final 需要 model_XXXX.pth / model_E*_B*.pth）"
        )
    if len(selected) < num:
        print(f"[WARN] 仅找到 {len(selected)} 个候选 ckpt（要求 {num} 个），按实际数量平均")
    return selected


def _find_model_key(ckpt: dict) -> str:
    """定位 raw 模型权重键：单模型为 'model'；多模型训练仅支持恰有一个 model_* 键的情况。"""
    if "model" in ckpt:
        return "model"
    model_keys = [k for k in ckpt if k.startswith("model_") and isinstance(ckpt[k], dict)]
    if len(model_keys) == 1:
        return model_keys[0]
    raise KeyError(
        f"无法定位模型权重键（多模型 ckpt 含 model_* 键 {model_keys}，请指定单模型 ckpt 目录）"
    )


def _average_state_dicts(sds: list[dict]) -> OrderedDict:
    """逐键平均：float/complex tensor 求均值；其余类型取第一个（见模块 docstring）。"""
    ref_keys = list(sds[0].keys())
    for sd in sds[1:]:
        if list(sd.keys()) != ref_keys:
            raise ValueError("入选 ckpt 的模型权重键集合/顺序不一致，无法平均（可能混入了不同模型或训练阶段）")
    out = OrderedDict()
    n = float(len(sds))
    for k in ref_keys:
        v0 = sds[0][k]
        if isinstance(v0, torch.Tensor) and (v0.is_floating_point() or v0.is_complex()):
            acc = v0.to(torch.float64)
            for sd in sds[1:]:
                acc += sd[k].to(torch.float64)
            # float64 累加避免大 N 时精度损失，再 cast 回原 dtype
            out[k] = (acc / n).to(v0.dtype)
        else:
            # int/bool buffer（如 num_batches_tracked）：计数器无平均语义，取第一个 ckpt 的值
            out[k] = v0
    return out


def _average_ema(ckpts: list[dict]) -> dict | None:
    """平均 torch_ema 的 state_dict：shadow_params 逐元素求均值，decay/num_updates 取第一个。

    所有入选 ckpt 均含 ema 时才平均；否则返回 None（输出不写 ema 键）。
    """
    if not all(isinstance(c.get("ema"), dict) and "shadow_params" in c["ema"] for c in ckpts):
        if any("ema" in c for c in ckpts):
            print("[WARN] 部分 ckpt 缺少 ema 键，输出不含 EMA 权重（将使用平均后的 raw 权重）")
        return None
    ema0 = ckpts[0]["ema"]
    n = float(len(ckpts))
    shadow = []
    for i, p0 in enumerate(ema0["shadow_params"]):
        acc = p0.to(torch.float64)
        for c in ckpts[1:]:
            acc += c["ema"]["shadow_params"][i].to(torch.float64)
        shadow.append((acc / n).to(p0.dtype))
    avg = dict(ema0)
    avg["shadow_params"] = shadow
    # collected_params 为 copy_to 暂存槽（通常为 None），与 decay/num_updates 一样取第一个
    return avg


def average_checkpoints(ckpt_dir: str | Path, mode: str, num: int, output: str | Path) -> Path:
    """挑选并平均 checkpoint，输出与 BaseTrainer ckpt 同构的 tar，返回输出路径。"""
    from core.utils.safe_load import safe_torch_load

    paths = select_checkpoints(ckpt_dir, mode, num)
    print(f"[INFO] 入选 {len(paths)} 个 ckpt（mode={mode}）:")
    for p in paths:
        print(f"  - {p.name}")

    # ckpt 内含 rng_state 的 numpy 标量，safe_torch_load 默认安全模式可处理（numpy 白名单）
    ckpts = [safe_torch_load(p) for p in paths]

    model_key = _find_model_key(ckpts[0])
    avg_sd = _average_state_dicts([c[model_key] for c in ckpts])
    avg_ema = _average_ema(ckpts)

    # 以第一个 ckpt 为骨架：保留 config/resolved_config 等元数据（pickle 共享引用，不占额外空间）
    out = dict(ckpts[0])
    out[model_key] = avg_sd
    # 平均权重无对应的优化器/调度器/AMP/RNG 状态，剔除以避免误导 resume
    for k in list(out.keys()):
        if (
            k in ("optimizer", "scheduler", "scaler", "rng_state", "sampler_state")
            or k.startswith(("optimizer_", "scheduler_"))
        ):
            out.pop(k)
    if avg_ema is not None:
        out["ema"] = avg_ema
    else:
        out.pop("ema", None)
    # 溯源信息
    out["averaged_from"] = [p.name for p in paths]
    out["averaged_mode"] = mode
    out["averaged_num"] = len(paths)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 先写同目录临时文件再 os.replace 发布：中断只会留下 tmp，不会截断已有 ckpt
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(output.parent), prefix=f".{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "wb") as handle:
            torch.save(out, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, output)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    print(f"[OK] 已保存平均 ckpt: {output}")
    print("     加载: from core.utils.safe_load import safe_torch_load")
    print(f"           ckpt = safe_torch_load('{output}', purpose='averaged checkpoint model weights')")
    print(f"           model.load_state_dict(ckpt['{model_key}'])")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint 平均（SWA-like）：对 best/final N 个 ckpt 的模型权重逐键求均值"
    )
    parser.add_argument("--ckpt_dir", required=True,
                        help="checkpoint 目录（exp_path/checkpoints/<ts>）")
    parser.add_argument("--mode", required=True, choices=["best", "final"],
                        help="best=按文件名 val_loss 升序取 top N；final=按 epoch/batch 取最后 N 个 model_*.pth")
    parser.add_argument("--num", type=int, default=5,
                        help="参与平均的 ckpt 数量（默认 5）")
    parser.add_argument("--output", required=True, help="输出 tar 路径")
    args = parser.parse_args()

    if args.num < 1:
        print("[ERROR] --num 必须 >= 1")
        sys.exit(1)

    average_checkpoints(args.ckpt_dir, args.mode, args.num, args.output)


if __name__ == "__main__":
    main()
