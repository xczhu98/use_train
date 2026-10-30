"""评估 subprocess 的仓库定位与执行适配。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 评估 subprocess 的统一超时（秒）；超时后 subprocess 直接杀子进程并抛 TimeoutExpired。
EVAL_SUBPROCESS_TIMEOUT_S = float(
    os.environ.get("USE_TRAIN_EVAL_SUBPROCESS_TIMEOUT_S", str(12 * 3600))
)


def _set_parent_death_signal() -> None:
    """子进程 preexec：父进程死亡（含 SIGKILL）时子进程收 SIGKILL，避免孤儿。"""
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, 9)  # PR_SET_PDEATHSIG=1, SIGKILL=9
    except Exception:
        pass


# 仅 Linux 提供；preexec_fn 禁止在多线程调用点使用，调用方须保证主线程单线程 fork。
death_signal_preexec = (
    _set_parent_death_signal if sys.platform.startswith("linux") else None
)


def repository_script(relative_path: str | Path) -> Path:
    """解析仓库自有脚本，拒绝超出仓库根的路径。"""
    script = (PROJECT_ROOT / relative_path).resolve(strict=True)
    try:
        script.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"脚本路径超出仓库根: {script}") from exc
    if not script.is_file():
        raise FileNotFoundError(f"仓库脚本不存在: {script}")
    return script


def absolute_path(value: str | Path) -> Path:
    """按调用进程当前语义把用户路径立即解析为绝对路径。"""
    return Path(value).expanduser().resolve()


def run_evaluate_urgent(
    inf_scp: str | Path,
    output_dir: str | Path,
    ref_scp: Optional[str | Path] = None,
    audio_quality: bool = False,
    downstream: bool = False,
    device: str = "cuda:0",
) -> Path:
    """用绝对脚本和参数执行 ``evaluate_urgent.py``，不改变 cwd。"""
    output = absolute_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(repository_script("evaluate/evaluate_urgent.py")),
        "--inf_scp",
        str(absolute_path(inf_scp)),
        "--output_dir",
        str(output),
        "--device",
        device,
    ]
    if ref_scp:
        command += ["--ref_scp", str(absolute_path(ref_scp))]
    if audio_quality:
        command.append("--audio_quality")
    if downstream:
        command.append("--downstream")
    logger.info("[Evaluate-URGENT] %s", " ".join(command))
    subprocess.run(
        command,
        check=True,
        timeout=EVAL_SUBPROCESS_TIMEOUT_S,
        start_new_session=True,
        preexec_fn=death_signal_preexec,
    )
    return output
