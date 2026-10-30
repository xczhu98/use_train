"""TSE sidecar 发布 Module。

该 Module 的唯一输入是公共 ``InferenceReport`` 和每个 UID 的期望 artifact
metadata。它不决定 WAV 是否复用；WAV 的执行和 reuse 已由公共 runner 完成，
sidecar 只为 report 中成功/复用的结果生成并校验 TSE 专属 sidecar。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.inference import InferenceReport
from tasks.tse.inference import artifacts


def _shared_artifact_fingerprint(expected: Mapping[str, Any]) -> str | None:
    """共享 artifact 身份（``inputs`` 之外）的指纹，用作单次 run 内的验证缓存键。"""
    expected_artifacts = expected.get("artifacts")
    if not isinstance(expected_artifacts, Mapping):
        return None
    shared = {key: value for key, value in expected_artifacts.items() if key != "inputs"}
    try:
        return artifacts.fingerprint_json(shared)
    except Exception:  # 结构异常时回退完整校验，由 verify_artifact_meta 负责报错
        return None


def _verify_already_checked_shared_artifacts(expected: Mapping[str, Any]) -> None:
    """共享身份已在本 run 内完整复核；这里只复核 metadata 结构、指纹与自身输入。"""
    if expected.get("schema") != artifacts.ARTIFACT_SCHEMA:
        raise ValueError("TSE artifact metadata schema is invalid")
    expected_artifacts = expected["artifacts"]
    if expected.get("fingerprint") != artifacts.fingerprint_json(dict(expected_artifacts)):
        raise ValueError("TSE artifact metadata fingerprint is invalid")
    inputs = expected_artifacts.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("TSE artifact metadata has no inputs")
    artifacts.verify_input_identities(inputs)
    cues = expected_artifacts.get("cues")
    if cues is not None:
        artifacts.verify_cue_binding(cues, inputs)


def publish_sidecars_from_report(
    report: InferenceReport,
    expected_by_uid: Mapping[str, Mapping[str, Any]],
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """根据公共 report 发布 TSE sidecar，并返回 UID 到 sidecar 的映射。

    ``report`` 是 WAV 发布的 source of truth：failed UID 不会生成 sidecar；
    ``ok`` 和 ``skipped`` UID 都必须存在于 expected metadata，且其 output
    identity 会在 sidecar 写出前重新验证。共享 artifact 身份（checkpoint、
    配置、实现指纹、数据集）在同一 run 内只完整复核一次，避免逐 UID 重算
    整包指纹并反序列化 checkpoint；后续 UID 只复核自身输入与 output stat。
    """
    root = Path(output_dir).expanduser().resolve()
    published: dict[str, Path] = {}
    with artifacts.exclusive_output_run(root):
        verified_shared: set[str] = set()
        for result in report.results:
            if result.status not in {"ok", "skipped"}:
                continue
            expected = expected_by_uid.get(result.uid)
            if expected is None:
                raise ValueError(f"公共 inference report 缺少 TSE metadata: {result.uid}")
            if not result.output_path:
                raise ValueError(f"公共 inference report 缺少 output path: {result.uid}")
            output = Path(result.output_path).expanduser().resolve()
            try:
                output.relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError(f"TSE output escapes sidecar output_dir: {result.uid}") from exc
            if result.output_identity != artifacts.stat_audio_artifact(output):
                raise RuntimeError(f"公共 inference output identity 已变化: {result.uid}")
            sidecar = output.with_name(f"{output.stem}.meta.json")
            shared_key = _shared_artifact_fingerprint(expected)
            if shared_key is not None and shared_key in verified_shared:
                _verify_already_checked_shared_artifacts(expected)
                artifacts.atomic_write_json(
                    sidecar,
                    {**expected, "output": artifacts.stat_audio_artifact(output)},
                )
            else:
                artifacts.write_output_sidecar_locked(sidecar, output, expected)
                if shared_key is not None:
                    verified_shared.add(shared_key)
            published[result.uid] = sidecar
    return published
