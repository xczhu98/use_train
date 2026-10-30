"""非侵入式 MOS 指标包装与批处理。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)


def compute_dnsmos(model, audio: np.ndarray, fs: int) -> Optional[dict]:
    """计算 DNSMOS 分数。model = load_dnsmos() 返回的可调用对象。

    Returns: {"OVRL": float, "SIG": float, "BAK": float}
    """
    try:
        return model(audio, fs)
    except Exception as e:
        _log.warning(f"DNSMOS failed: {e}")
        return None


def compute_nisqa(model, audio: np.ndarray, fs: int) -> Optional[dict]:
    """计算 NISQA 分数。model = load_nisqa() 返回的可调用对象。

    Returns: {"MOS": float, "NOI": float, "COL": float, "DIS": float, "LOUD": float}
    """
    try:
        return model(audio, fs)
    except Exception as e:
        _log.warning(f"NISQA failed: {e}")
        return None


def compute_utmos(model, audio: np.ndarray, fs: int) -> Optional[dict]:
    """计算 UTMOS 分数。model = load_utmos() 返回的可调用对象。

    Returns: {"MOS": float}
    """
    try:
        return model(audio, fs)
    except Exception as e:
        _log.warning(f"UTMOS failed: {e}")
        return None


def compute_scoreq(model, audio: np.ndarray, fs: int) -> Optional[dict]:
    """计算 SCOREQ 分数。model = load_scoreq() 返回的可调用对象。

    Returns: {"MOS": float}
    """
    try:
        return model(audio, fs)
    except Exception as e:
        _log.warning(f"SCOREQ failed: {e}")
        return None


_MOS_MODELS = {
    "DNSMOS": (compute_dnsmos, "OVRL"),
    "NISQA":  (compute_nisqa, "MOS"),
    "UTMOS":  (compute_utmos, "MOS"),
    "SCOREQ": (compute_scoreq, "MOS"),
}

_MOS_SUB_SCORES = {
    "DNSMOS": ["OVRL", "SIG", "BAK"],
    "NISQA":  ["MOS", "NOI", "COL", "DIS", "LOUD"],
    "UTMOS":  ["MOS"],
    "SCOREQ": ["MOS"],
}

def _compute_mos_one(args):
    """无原生 batch 能力模型的逐条 MOS 并行工作函数。"""
    model, model_name, enh_np, fs = args
    compute_fn, _ = _MOS_MODELS.get(model_name, (None, None))
    if compute_fn is None:
        return None
    try:
        return compute_fn(model, enh_np, fs)
    except Exception:
        return None


def compute_mos_batch(
    model,
    model_name: str,
    items: list[tuple[np.ndarray, int]],
    batch_size: int = 16,
    device: str = "cpu",
    parallel_per_item: bool = True,
) -> list[Optional[dict]]:
    """对 (enh_audio, fs) 列表运行 MOS 模型。

    - UTMOS: 当可用时调用原生 model.batch() GPU 批处理。
    - DNSMOS/NISQA/SCOREQ: 逐条处理；当 parallel_per_item=True 且 len(items)>=4 时，
      使用 ThreadPoolExecutor 加速。返回每条一个 dict。
    """
    if not items:
        return []
    compute_fn, _ = _MOS_MODELS.get(model_name, (None, None))
    if compute_fn is None:
        return [None] * len(items)

    # UTMOS 支持 GPU 上的 batch()
    if model_name == "UTMOS" and hasattr(model, "batch"):
        try:
            return model.batch(items, batch_size=batch_size)
        except Exception as e:
            _log.warning(f"UTMOS batch failed ({e}), falling back to sequential")

    # 逐条：对 CPU 密集型模型（DNSMOS、SCOREQ）可选并行。
    # NISQA 的 nisqaModel 非线程安全；mos_models._NISQAScorer 内部已加锁，因此跳过线程池。
    if parallel_per_item and len(items) >= 4 and model_name != "NISQA":
        try:
            import os as _os
            n_workers = min(max(1, (_os.cpu_count() or 1) - 1), len(items), 8)
            tasks = [(model, model_name, enh_np, fs) for enh_np, fs in items]
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                out = list(pool.map(_compute_mos_one, tasks, chunksize=max(1, len(tasks) // n_workers)))
            return out
        except Exception as e:
            _log.warning(f"MOS parallel_per_item failed ({e}), falling back to sequential")

    out = []
    for enh_np, fs in items:
        out.append(compute_fn(model, enh_np, fs))
    return out
