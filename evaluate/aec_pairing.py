"""AEC mic ↔ farend path pairing (shared by ``inference.py`` and ``evaluate_aec.py``).

Challenge-style layouts use ``nearend_mic_*`` / ``farend_speech_*`` naming; flat dirs may use
``*_mic`` / ``*_lpb`` stems. Resolution rules live here so evaluation and inference stay consistent.
"""

from __future__ import annotations

from pathlib import Path


def farend_uid_from_mic_uid(uid: str) -> str:
    """Map mic utterance id to far-end id (Microsoft / internal naming)."""
    return uid.replace("nearend_mic_", "farend_speech_")


def build_farend_index(far_files: list[tuple[str, str]]) -> dict[str, str]:
    """把 ``(uid, path)`` 列表预建成 lookup dict，供批量配对复用。"""
    return {u: p for u, p in far_files}


def resolve_farend_path(
    mic_path: Path | str,
    far_files: list[tuple[str, str]],
    *,
    mic_uid: str | None = None,
    far_index: dict[str, str] | None = None,
) -> str | None:
    """Return far-end audio path for a mic file, using the same rules as ``inference.py``.

    Args:
        mic_path: Path to mic wav (used for stem heuristics when ``mic_uid`` is omitted).
        far_files: List of ``(uid, path)`` as built by directory or SCP listing.
        mic_uid: Optional utterance id when it differs from ``Path(mic_path).stem`` (e.g. SCP).
        far_index: Optional pre-built :func:`build_farend_index` result；按 mic 逐个
            调用时传入可避免每次重建 dict（``far_files`` 仍用于 stem 兜底扫描）。
    """
    mic_path = Path(mic_path)
    uid = mic_uid if mic_uid is not None else mic_path.stem
    far_dict = far_index if far_index is not None else build_farend_index(far_files)
    far_uid = farend_uid_from_mic_uid(uid)
    far_path = far_dict.get(far_uid) or far_dict.get(uid)
    if far_path:
        return far_path
    stem = uid
    far_candidates = [
        p
        for fu, p in far_files
        if fu == stem
        or stem.replace("mic", "farend") == fu
        or stem.replace("mic", "lpb") == fu
    ]
    if far_candidates:
        return far_candidates[0]
    return None
