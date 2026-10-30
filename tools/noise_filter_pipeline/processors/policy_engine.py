"""
PolicyEngine — VAD 策略引擎。

依据 B4 的 VAD 结果（fail_count）把 chunk 分为 noise / suspected / rejected，
并按 keep_* 开关决定是否保留。

原第二层 AED 标签过滤（人声 / 音乐 / babble）已随旧 B5 AED 分类一并删除：
B 阶段不再产出 AED 标签；当前 Stage C 按 C1→C2→C3 执行，由 C3 汇总
C1 内容分类与 C2 AudioSet canonical evidence 完成准入判定。

用法:
    engine = PolicyEngine(config)          # config 可以是 PipelineConfig 或 dict
    pr = engine.apply(chunk)              # 返回 PolicyResult
"""

import logging
import math
from dataclasses import dataclass
from numbers import Real
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# rejected 归因子目录名
REJECTED_REASON_VAD = 'vad_rejected'


# ---------------------------------------------------------------------------
# VAD 辅助函数（从 pipeline_v2 移入，消除循环导入）
# ---------------------------------------------------------------------------
def _finite_probability(value: Any) -> bool:
    """判断比例/概率是否为有限的 [0, 1] 数值。"""
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _valid_frame_probs(value: Any) -> bool:
    """严格校验帧级概率序列，避免 NaN/Inf 或标量伪装成序列。"""
    if not isinstance(value, (list, tuple, np.ndarray)):
        return False
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return (
        array.ndim == 1
        and array.size > 0
        and bool(np.isfinite(array).all())
        and bool(((array >= 0.0) & (array <= 1.0)).all())
    )


def _vad_result_has_ratios(vad_result: Any) -> bool:
    """严格判断 VAD 结果是否含有可重算的有限比例证据。"""
    if not isinstance(vad_result, dict) or not vad_result:
        return False
    source_count = 0

    if "individual_results" in vad_result:
        individual_results = vad_result["individual_results"]
        if not isinstance(individual_results, list) or not individual_results:
            return False
        for individual in individual_results:
            if not isinstance(individual, dict):
                return False
            ratio = individual.get("speech_ratio")
            if not _finite_probability(ratio):
                return False
            frame_probs = individual.get("frame_probs")
            if frame_probs is not None and not _valid_frame_probs(frame_probs):
                return False
        source_count += 1

    if "vad_results" in vad_result:
        vad_results = vad_result["vad_results"]
        if not isinstance(vad_results, dict) or not vad_results:
            return False
        if any(not _finite_probability(value) for value in vad_results.values()):
            return False
        source_count += 1

    if "speech_ratio" in vad_result:
        if not _finite_probability(vad_result["speech_ratio"]):
            return False
        source_count += 1

    if "frame_probs" in vad_result:
        frame_probs = vad_result["frame_probs"]
        if frame_probs is not None and not _valid_frame_probs(frame_probs):
            return False

    return source_count > 0


def _speech_ratio_from_individual(ind: Dict, speech_threshold: float | None) -> float:
    """从单个 VAD 的 individual 结果得到 speech_ratio。"""
    if speech_threshold is not None:
        fp = ind.get("frame_probs")
        if fp is not None:
            arr = np.asarray(fp, dtype=np.float64)
            return float(np.mean(arr > float(speech_threshold)))
    return float(ind["speech_ratio"])


def _speech_ratios_from_vad_result(vad_result: Dict, speech_threshold: float | None = None) -> List[float]:
    """从已校验的 vad_result 得到各 VAD 的 speech_ratio 列表。"""
    ind_list = vad_result.get("individual_results")
    if ind_list is not None:
        return [_speech_ratio_from_individual(item, speech_threshold) for item in ind_list]
    vad_results = vad_result.get("vad_results")
    if vad_results is not None:
        return [float(value) for value in vad_results.values()]
    if "speech_ratio" in vad_result:
        return [float(vad_result["speech_ratio"])]
    return []


def _fail_count_from_vad_result(
    vad_result: Dict,
    vad_speech_ratio_threshold: float,
    speech_threshold: float | None = None,
) -> int:
    """从 vad_result 用给定阈值计算 fail_count。"""
    ratios = _speech_ratios_from_vad_result(vad_result, speech_threshold)
    return sum(1 for ratio in ratios if ratio > float(vad_speech_ratio_threshold))


def _final_category_and_kept_from_vad_result(
    chunk: Dict,
    speech_ratio_threshold: float,
    keep_noise: bool = True,
    keep_suspected: bool = False,
    keep_rejected: bool = False,
    speech_threshold: float | None = None,
    defer_speech_to_stage_c: bool = False,
) -> tuple:
    """根据 vad_result + 阈值计算 final_category 与 kept。
    返回 (final_category, kept, fail_count)。

    ``defer_speech_to_stage_c=True`` 时，任何真实 VAD 语音票
    （fail_count > 0）都归一为 ``speech_suspected`` 并放行进 Stage C，
    由 C1/C2/C3 做内容级最终裁决；严格技术失败（strict fallback /
    failed_vad_types / 无有效结果）仍 fail-closed 拒绝，不参与延期。
    """
    vad_result = chunk.get("vad_result")
    # 没有真实、可重算的 VAD 证据时，禁止信任输入携带的 final_category。
    # 这同时覆盖 None、非 object、空 object、非法比例及 NaN/Inf。
    if not isinstance(vad_result, dict) or not _vad_result_has_ratios(vad_result):
        return ("rejected", keep_rejected, 2)
    # B3 的硬性判定必须被 B4 尊重，不得因阈值重算而翻案：
    # strict_vad_models 下 VAD 退化为 energy fallback 时，B3 已判 rejected
    if vad_result.get('strict_vad_models'):
        ind_list = vad_result.get('individual_results') or []
        if any(isinstance(r, dict) and r.get('is_real_model') is False for r in ind_list):
            return ('rejected', keep_rejected, 2)
        # strict 模式下指定 VAD 推理抛异常被跳过同样视为失败（与 B3 口径一致）
        if vad_result.get('failed_vad_types'):
            return ('rejected', keep_rejected, 2)

    fail_count = _fail_count_from_vad_result(vad_result, speech_ratio_threshold, speech_threshold)

    # 内容型延期：任意真实语音票进入 speech_suspected，由 Stage C 裁决。
    # 该分支位于 strict 技术失败检查之后，因此不会放行 fallback/失败证据。
    if defer_speech_to_stage_c and fail_count > 0:
        return ('speech_suspected', keep_suspected, fail_count)

    # B3 已按自身（更严格的）阈值/规则判为 rejected 时，直接尊重该结论，
    # 不得用本层阈值重算翻案（belt & braces：即使 B4 阈值配置偏宽也不放行）
    if chunk.get('vad_category') == 'rejected':
        return ('rejected', keep_rejected, 2)
    # reject_on_any_speech：任一 VAD 检出语音即拒绝，与 B3 的 MultiVADResult 判定一致
    if vad_result.get('reject_on_any_speech') and fail_count > 0:
        return ('rejected', keep_rejected, fail_count)
    if fail_count == 0:
        final_category = 'noise'
        kept = keep_noise
    elif fail_count == 1:
        final_category = 'suspected'
        kept = keep_suspected
    else:
        final_category = 'rejected'
        kept = keep_rejected
    return (final_category, kept, fail_count)


# ---------------------------------------------------------------------------
# PolicyResult & PolicyEngine
# ---------------------------------------------------------------------------
@dataclass
class PolicyResult:
    """单次策略应用的结果。"""

    final_category: str  # noise | suspected | rejected
    kept: bool
    rejected_reason: Optional[str] = None
    fail_count: int = 0
    layer1_kept: bool = True   # VAD 层结果


def _config_get(cfg: Any, key: str, default=None):
    """兼容 config 为 dict 或对象。"""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class PolicyEngine:
    """VAD 策略引擎：按 VAD fail_count 分档并应用 keep_* 保留策略。"""

    def __init__(self, config: Any):
        """config 可以是 PipelineConfig 对象或 dict（多进程 worker 传 dict）。"""
        self.vad_thr = _config_get(config, 'vad_speech_ratio_threshold', 0.05)
        self.speech_thr = _config_get(config, 'speech_threshold', None)
        self.keep_noise = _config_get(config, 'keep_noise', True)
        self.keep_suspected = _config_get(config, 'keep_suspected', False)
        self.keep_rejected = _config_get(config, 'keep_rejected', False)
        # 内容型延期默认关闭：只有显式配置才把 VAD 语音票延后给 Stage C
        self.defer_speech_to_stage_c = bool(
            _config_get(config, 'defer_speech_to_stage_c', False)
        )

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def apply(self, chunk: Dict) -> PolicyResult:
        """对单个 chunk 应用 VAD 策略，返回 PolicyResult。"""
        cat, kept, fc = _final_category_and_kept_from_vad_result(
            chunk, self.vad_thr,
            self.keep_noise, self.keep_suspected, self.keep_rejected,
            self.speech_thr,
            self.defer_speech_to_stage_c,
        )
        rejected_reason = REJECTED_REASON_VAD if cat == 'rejected' else None

        # 同步 vad_result.fail_count
        vr = chunk.get('vad_result')
        if isinstance(vr, dict):
            vr['fail_count'] = fc

        return PolicyResult(
            final_category=cat,
            kept=kept,
            rejected_reason=rejected_reason,
            fail_count=fc,
            layer1_kept=kept,
        )

    def apply_to_chunk(self, chunk: Dict) -> PolicyResult:
        """对单个 chunk 应用策略并写回 chunk 的 final_category/kept/rejected_reason。
        返回 PolicyResult 供调用方做额外统计。"""
        pr = self.apply(chunk)
        chunk['final_category'] = pr.final_category
        chunk['kept'] = pr.kept
        if pr.rejected_reason:
            chunk['rejected_reason'] = pr.rejected_reason
        vr = chunk.get('vad_result')
        if isinstance(vr, dict):
            vr['fail_count'] = pr.fail_count
        return pr
