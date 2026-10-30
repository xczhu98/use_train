"""Layer C 分支 → CompositeLoss 门控：无有效语音参考时不算 PESQ/STOI/SI-SNR/Loud。

- SE DynamicMixingDataset：仅 ``prob_noise_only`` → ``is_pure_noise``。
- AEC：``prob_noise_only`` / ``prob_echo_only`` / ``prob_noise_echo_only`` 分别对应三分支 bool。
  ``has_echo`` 独立标记：所有含回声的分支为 True，用于 ERLE 样本收集。
"""

from __future__ import annotations


def branch_to_pure_flags_se(branch: str | None) -> tuple[bool, bool, bool]:
    """(is_pure_noise, is_pure_echo, is_pure_echo_noise)。SE 无 echo 分支，后两者恒 False。"""
    if branch == "prob_noise_only":
        return (True, False, False)
    return (False, False, False)


def branch_to_pure_flags_aec(branch: str | None) -> tuple[bool, bool, bool, bool]:
    """(is_pure_noise, is_pure_echo, is_pure_echo_noise, has_echo).

    ``has_echo`` is True for branches that contain a disk-manifest echo component:
    prob_echo_only, prob_noise_echo_only, prob_target_echo, prob_target_noise_echo.
    Used by the trainer to collect ERLE samples without false positives from
    prob_target_noise / prob_target_only / prob_noise_only branches.
    """
    if branch == "prob_noise_only":
        return (True, False, False, False)   # noise, no echo
    if branch == "prob_echo_only":
        return (False, True, False, True)    # echo, no nearend
    if branch == "prob_noise_echo_only":
        return (False, False, True, True)    # echo + noise, no nearend
    if branch == "prob_target_noise":
        return (False, False, False, False)  # nearend + noise, NO echo
    if branch == "prob_target_only":
        return (False, False, False, False)  # nearend only, no echo
    # prob_target_echo / prob_target_noise_echo (and any future echo-containing branch)
    # These have echo by construction — only dispatched when the branch needs C-aec.
    _has_echo = branch in ("prob_target_echo", "prob_target_noise_echo")
    return (False, False, False, _has_echo)
