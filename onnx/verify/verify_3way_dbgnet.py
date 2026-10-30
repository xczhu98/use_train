#!/usr/bin/env python3
"""三路精度对齐验证：离线 PyTorch vs 流式 PyTorch vs 导出的 ONNX Runtime (DBGNet).

流程：
1. 构建离线 DBGNet + 流式 StreamingDBGNet，共享权重
2. 导出 StreamingDBGNetFlat ONNX
3. 逐帧对比三路输出（enhanced spec + state probs）

``--save-onnx`` 正式发布要求 config、checkpoint、float32、未放宽的预注册阈值，
并且三路检查全部 PASS。

Usage:
    python onnx/verify/verify_3way_dbgnet.py \
        --config configs/aec/microsoft_challenge/dbgnet/dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml \
        --checkpoint exp/aec/dbgnet_small_16k_10ms_delay80ms_loss_hybrid/.../best_model.tar \
        [--frames 80]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


from onnx.utils import export_onnx
from onnx.verify.manifest import runtime_contract_from_spec
from onnx.verify.verify_common import (
    OwnedOnnxLifecycle as _OwnedOnnxLifecycle,
    VerifiedOnnxArtifact as _VerifiedOnnxArtifact,
    fingerprint_fd as _fingerprint_fd,
    open_regular_file as _open_regular_file,
    owned_onnx_cleanup,
    require_expected_fingerprint as _require_expected_fingerprint,
)
from core.utils.config import load_merged_config


MIN_WIDENED_FRAMES = 1000
MIN_WIDENED_RESETS = 2
DEFAULT_MSE_THRESHOLDS = {"float32": 1e-5, "float16": 1e-3}
DEFAULT_MAX_ABS_THRESHOLDS = {"float32": 5e-3, "float16": 1e-1}
EXTERNAL_ISTFT_CAPABILITY_LABEL = "External iSTFT synthesis capability"


def stft_lookahead_samples(*, center: bool, n_fft: int) -> int:
    """返回前端 center STFT 的未来样本需求；神经谱图路径不计入此值。"""
    return n_fft // 2 if center else 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    """构建 CLI parser，便于测试 widened 默认口径。"""
    ap = argparse.ArgumentParser(
        description="三路精度对齐：离线 PT / 流式 PT / 流式 ORT (DBGNet)"
    )
    ap.add_argument("--config", "-C", type=Path, required=True,
                    help="DBGNet training YAML config")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="模型 checkpoint (.tar / .pth)")
    ap.add_argument(
        "--frames", type=_positive_int, default=1200,
        help=f"STFT 帧数（widened 口径要求 ≥{MIN_WIDENED_FRAMES}）",
    )
    ap.add_argument(
        "--resets", type=_nonnegative_int, default=2,
        help=f"同步 cold reset 次数（widened 口径要求 ≥{MIN_WIDENED_RESETS}）",
    )
    ap.add_argument(
        "--inference-dtype", default="float32",
        choices=list(DEFAULT_MSE_THRESHOLDS), help="推理精度档位",
    )
    ap.add_argument("--mse-threshold", type=float, default=None,
                    help="PASS/FAIL MSE 阈值（默认按 dtype 自适应）")
    ap.add_argument("--max-abs-threshold", type=float, default=None,
                    help="PASS/FAIL max_abs 阈值（默认按 dtype 自适应）")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--onnx", type=Path, default=None,
                    help="Optional pre-exported ONNX file (skip export step)")
    ap.add_argument("--save-onnx", type=Path, default=None,
                    help="全部三路检查通过后发布 ONNX 到此路径")
    return ap


def publish_gate_passed(statuses, *, widened: bool, save_requested: bool) -> bool:
    """发布必须使用 widened 口径；手工 narrow 仍可诊断。"""
    return all(status == "PASS" for status in statuses) and (
        widened or not save_requested
    )


def verification_gate_passed(
    statuses,
    *,
    widened: bool,
    save_requested: bool,
    numeric_gate: bool,
    state_pass: bool,
    external_istft_pass: bool,
) -> bool:
    """三路数值、state 与外部 iSTFT synthesis capability 缺一不可。"""
    return (
        numeric_gate
        and state_pass
        and external_istft_pass
        and publish_gate_passed(
            statuses, widened=widened, save_requested=save_requested
        )
    )


def require_matching_runtime_contract(runtime_spec, wrapper_spec) -> None:
    """artifact 必须与本次 config 构建的 wrapper 完整 ABI 一致。"""
    checks = {
        "input names": (runtime_spec.input_names, wrapper_spec.input_names),
        "output names": (runtime_spec.output_names, wrapper_spec.output_names),
        "cache shapes": (dict(runtime_spec.cache_shapes), dict(wrapper_spec.cache_shapes)),
        "dynamic axes": (
            {name: dict(axes) for name, axes in runtime_spec.dynamic_axes.items()},
            {name: dict(axes) for name, axes in wrapper_spec.dynamic_axes.items()},
        ),
        "STFT metadata": (runtime_spec.stft, wrapper_spec.stft),
        "DBGNet metadata": (runtime_spec.dbgnet, wrapper_spec.dbgnet),
        "deployment metadata": (
            dict(runtime_spec.deployment_config or {}),
            dict(wrapper_spec.deployment_config or {}),
        ),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise RuntimeError(
            "DBGNet ONNX runtime contract does not match config-built wrapper: "
            + ", ".join(mismatches)
        )


def require_batch_one_graph_contract(session, runtime_spec) -> None:
    """正式 DBGNet artifact 的所有 batch-bearing graph 轴必须静态为 1。"""
    input_infos = {info.name: info for info in session.get_inputs()}
    output_infos = {info.name: info for info in session.get_outputs()}
    cache_shapes = dict(runtime_spec.cache_shapes)
    expected_shapes = dict(cache_shapes)
    expected_shapes.update({f"{name}_out": shape for name, shape in cache_shapes.items()})

    for name in ("mic_spec", "far_spec", "enh", "state_probs"):
        info = input_infos.get(name) or output_infos.get(name)
        if info is None:
            continue
        shape = getattr(info, "shape", None)
        if not shape or shape[0] != 1:
            raise RuntimeError(
                "DBGNet formal ONNX deployment only supports static batch_size=1; "
                f"graph tensor {name!r} has shape {shape!r}"
            )

    for name, expected in expected_shapes.items():
        info = input_infos.get(name) or output_infos.get(name)
        shape = getattr(info, "shape", None) if info is not None else None
        if shape is None or tuple(shape) != tuple(expected):
            raise RuntimeError(
                "DBGNet formal ONNX deployment only supports static batch_size=1; "
                f"graph cache {name!r} has shape {shape!r}, expected {tuple(expected)!r}"
            )


@dataclass(frozen=True)
class _FrozenConfigSidecar:
    name: str
    content: bytes

    def load(self) -> dict[str, object]:
        """从 frozen bytes 解析 resolved config，拒绝仍含继承指令的 sidecar。"""
        from omegaconf import OmegaConf

        try:
            cfg = OmegaConf.load(io.StringIO(self.content.decode("utf-8")))
        except (UnicodeDecodeError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"invalid frozen resolved config sidecar: {self.name}"
            ) from exc
        if "base_config" in cfg:
            raise RuntimeError(
                f"resolved config sidecar must not contain base_config: {self.name}"
            )
        value = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"resolved config sidecar must contain a mapping: {self.name}"
            )
        return dict(value)


@dataclass(frozen=True)
class _VerifierCheckpointSnapshot:
    """绑定 caller 显示路径与内存中不可变的 checkpoint/sidecar bytes。"""

    source_path: Path
    sha256: str
    size_bytes: int
    _content: bytes
    _sidecars: tuple[_FrozenConfigSidecar, ...]

    def open_checkpoint(self) -> io.BytesIO:
        """为每次 safe loader 调用返回同一 frozen bytes 的独立游标。"""
        stream = io.BytesIO(self._content)
        stream.name = f"{self.source_path} (immutable verifier snapshot)"
        return stream

    def load_model(self, model: torch.nn.Module) -> None:
        """从 frozen bytes 严格加载权重，禁止重新解析任何 checkpoint pathname。"""
        from onnx.export import load_checkpoint

        with self.open_checkpoint() as checkpoint:
            load_checkpoint(model, checkpoint, family="single_model")

    def require_config_provenance(
        self, config: Mapping[str, object], model_id: str
    ) -> dict[str, object]:
        """从同一 frozen checkpoint 与 frozen sidecar 校验训练配置绑定。"""
        from core.trainer.checkpoint_state import (
            parse_resolved_config_provenance,
            resolved_config_identity,
        )
        from core.utils.safe_load import safe_torch_load
        from onnx.runtime_spec import deployment_config_binding
        from onnx.verify.verify_common import checkpoint_embedded_training_config

        with self.open_checkpoint() as checkpoint:
            checkpoint_data = safe_torch_load(
                checkpoint,
                purpose="verify ONNX checkpoint config provenance",
            )
        if not isinstance(checkpoint_data, Mapping):
            raise RuntimeError("checkpoint must contain a mapping")
        embedded_config = checkpoint_embedded_training_config(checkpoint_data)
        frozen_sidecars = {sidecar.name: sidecar for sidecar in self._sidecars}
        provenance_value = checkpoint_data.get("resolved_config_provenance")

        if provenance_value is not None:
            provenance = parse_resolved_config_provenance(provenance_value)
            source_name = str(provenance["source"])
            if source_name not in frozen_sidecars:
                raise RuntimeError(
                    "checkpoint resolved config provenance source is missing: "
                    f"{source_name}"
                )
            training_config = frozen_sidecars[source_name].load()
            expected_identity = str(provenance["sha256"])
            if resolved_config_identity(training_config) != expected_identity:
                raise RuntimeError("checkpoint resolved config sidecar identity mismatch")
            for key in ("resolved_config", "config"):
                value = checkpoint_data.get(key)
                if not isinstance(value, Mapping):
                    raise RuntimeError(f"bound checkpoint must embed mapping '{key}'")
                if resolved_config_identity(value) != expected_identity:
                    raise RuntimeError(
                        f"checkpoint embedded {key} identity does not match provenance"
                    )
        elif embedded_config is not None:
            embedded_identity = resolved_config_identity(embedded_config)
            if frozen_sidecars and not any(
                resolved_config_identity(sidecar.load()) == embedded_identity
                for sidecar in frozen_sidecars.values()
            ):
                raise RuntimeError(
                    "checkpoint resolved_config sidecar does not match embedded training config"
                )
            training_config = embedded_config
        else:
            if not frozen_sidecars:
                raise RuntimeError(
                    "checkpoint config provenance is missing: expected a resolved_config "
                    "sidecar or embedded resolved_config/config"
                )
            by_identity = {
                resolved_config_identity(config): config
                for config in (sidecar.load() for sidecar in frozen_sidecars.values())
            }
            if len(by_identity) != 1:
                raise RuntimeError(
                    "legacy checkpoint config provenance is ambiguous across "
                    "resolved_config sidecars"
                )
            training_config = next(iter(by_identity.values()))

        training_binding = deployment_config_binding(training_config, model_id)
        verifier_binding = deployment_config_binding(config, model_id)
        if verifier_binding["resolved_config"] != training_binding["resolved_config"]:
            raise RuntimeError(
                f"{model_id} verifier config fingerprint does not match checkpoint "
                "training resolved_config"
            )
        if verifier_binding["config"] != training_binding["config"]:
            raise RuntimeError(
                f"{model_id} verifier deployment semantics do not match checkpoint "
                "training config"
            )
        return training_binding

    @property
    def report_binding(self) -> dict[str, object]:
        """报告保留 caller source，摘要严格来自 frozen bytes。"""
        return {
            "source_path": str(self.source_path),
            "snapshot_sha256": self.sha256,
            "snapshot_size_bytes": self.size_bytes,
        }

    def verify(self) -> None:
        """复核内存中不可变内容与创建时摘要一致，不信任 pathname/inode。"""
        if len(self._content) != self.size_bytes:
            raise RuntimeError("DBGNet immutable checkpoint snapshot size changed")
        if hashlib.sha256(self._content).hexdigest() != self.sha256:
            raise RuntimeError("DBGNet immutable checkpoint snapshot digest changed")

    def close(self) -> None:
        """内存 snapshot 无 checkpoint/sidecar 临时路径需要清理。"""


def _freeze_file(source: Path, *, label: str):
    """从单一 source FD 固化 bytes，并复核读取前后 inode 与内容身份。"""
    source_fd = _open_regular_file(source, label=label)
    try:
        source_before = _fingerprint_fd(source_fd, label=label)
        os.lseek(source_fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(source_fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        source_after = _fingerprint_fd(source_fd, label=f"{label} after freeze")
        _require_expected_fingerprint(
            source_after,
            expected_sha256=source_before.sha256,
            expected_size_bytes=source_before.size_bytes,
            label=f"{label} after freeze",
        )
        if (
            source_after.device != source_before.device
            or source_after.inode != source_before.inode
        ):
            raise RuntimeError(f"{label} identity changed")
        if hashlib.sha256(content).hexdigest() != source_before.sha256:
            raise RuntimeError(f"{label} frozen bytes digest mismatch")
        return source_before, content
    finally:
        os.close(source_fd)


class _VerifierArtifactTransaction:
    """统一持有并 finally 清理 checkpoint 与 ONNX verifier snapshots。"""

    def __init__(self) -> None:
        self._checkpoint: _VerifierCheckpointSnapshot | None = None
        self._onnx_artifacts: list[_VerifiedOnnxArtifact] = []

    def __enter__(self):
        return self

    def freeze_checkpoint(self, checkpoint) -> _VerifierCheckpointSnapshot:
        if self._checkpoint is not None:
            raise RuntimeError("DBGNet verifier checkpoint is already snapshotted")
        source = Path(checkpoint).expanduser().absolute()
        fingerprint, content = _freeze_file(
            source, label="DBGNet checkpoint source"
        )
        sidecars = []
        for sidecar in sorted({
            *source.parent.glob("resolved_config*.yaml"),
            *source.parent.glob("resolved_config*.yml"),
            *source.parent.glob("resolved_config*.json"),
        }):
            if not sidecar.is_file():
                continue
            _sidecar_fingerprint, sidecar_content = _freeze_file(
                sidecar, label=f"DBGNet checkpoint sidecar {sidecar.name}"
            )
            sidecars.append(
                _FrozenConfigSidecar(name=sidecar.name, content=sidecar_content)
            )
        snapshot = _VerifierCheckpointSnapshot(
            source_path=source,
            sha256=fingerprint.sha256,
            size_bytes=fingerprint.size_bytes,
            _content=content,
            _sidecars=tuple(sidecars),
        )
        snapshot.verify()
        self._checkpoint = snapshot
        return snapshot

    def bind_onnx(self, artifact: _VerifiedOnnxArtifact) -> _VerifiedOnnxArtifact:
        self._onnx_artifacts.append(artifact)
        return artifact

    def close(self) -> None:
        failures = []
        for artifact in reversed(self._onnx_artifacts):
            try:
                artifact.close()
            except OSError as exc:
                failures.append(exc)
        self._onnx_artifacts.clear()
        if self._checkpoint is not None:
            try:
                self._checkpoint.close()
            except RuntimeError as exc:
                failures.append(exc)
            self._checkpoint = None
        if failures:
            raise RuntimeError("DBGNet verifier artifact cleanup failed") from failures[0]

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
        return False


def _copy_caller_onnx_for_verification(caller_path, lifecycle) -> _VerifiedOnnxArtifact:
    """复制 caller-owned ONNX，并把 copy 登记为脚本 owned。"""
    import shutil
    fd, owned_path = tempfile.mkstemp(suffix=".onnx")
    lifecycle.own(owned_path)
    os.close(fd)
    shutil.copyfile(caller_path, owned_path)
    return _VerifiedOnnxArtifact(owned_path)


def state_gate_passed(
    *,
    use_state: bool,
    expected_frames: int,
    pt_frames: int,
    ort_frames: int,
    mse_off_pt: float | None,
    mse_pt_ort: float | None,
    max_abs_off_pt: float | None = None,
    max_abs_pt_ort: float | None = None,
    threshold: float,
    max_abs_threshold: float | None = None,
) -> bool:
    """state detector 启用时要求输出完整且 MSE/max_abs 均通过。"""
    if not use_state:
        return True
    return (
        pt_frames == expected_frames
        and ort_frames == expected_frames
        and mse_off_pt is not None
        and mse_pt_ort is not None
        and max_abs_off_pt is not None
        and max_abs_pt_ort is not None
        and max_abs_threshold is not None
        and np.isfinite(mse_off_pt)
        and np.isfinite(mse_pt_ort)
        and np.isfinite(max_abs_off_pt)
        and np.isfinite(max_abs_pt_ort)
        and mse_off_pt < threshold
        and mse_pt_ort < threshold
        and max_abs_off_pt < max_abs_threshold
        and max_abs_pt_ort < max_abs_threshold
    )


def _concat_state_frames(frames):
    if not frames:
        return None
    try:
        return torch.cat(frames, dim=1)
    except RuntimeError:
        return None


def _build_model_from_config(
    config_path: str,
    checkpoint_snapshot: _VerifierCheckpointSnapshot | None = None,
):
    """Build DBGNet model from config YAML and optionally load frozen checkpoint."""
    cfg = load_merged_config(config_path)
    model_cfg = cfg.get("model", {})
    if model_cfg.get("type") != "DBGNet":
        raise ValueError(f"DBGNet verifier requires model.type=DBGNet, got {model_cfg.get('type')!r}")
    model_args = dict(model_cfg.get("args", {}))
    stft_cfg = cfg.get("stft", {})

    # Inject STFT params into model args
    for key in ("n_fft", "hop_length", "win_length"):
        if key in stft_cfg and key not in model_args:
            model_args[key] = stft_cfg[key]
    sample_rate = int((cfg.get("data", {}) or {})["sample_rate"])
    if int(model_args.get("sample_rate", sample_rate)) != sample_rate:
        raise ValueError("DBGNet model/data sample_rate mismatch")
    model_args["sample_rate"] = sample_rate

    from tasks.aec.models.discriminative.dbgnet import DBGNet

    base = DBGNet(**model_args).eval()

    if checkpoint_snapshot is not None:
        # strict loader 必须消费冻结的 checkpoint bytes，禁止按路径重新解析。
        checkpoint_snapshot.load_model(base)

    return base, model_args, cfg


def _build_streaming_pt_reference(base, model_args, cfg, device, inference_dtype):
    """构建独立的流式 PT 精度档位，避免转换时连带修改离线锚点。"""
    from tasks.aec.models.discriminative.dbgnet import DBGNet
    from onnx.models.streaming_dbgnet import StreamingDBGNet

    stft_cfg = cfg.get("stft", {}) or {}
    dtype = getattr(torch, inference_dtype)
    if dtype == torch.float32:
        model = base
    else:
        model = DBGNet(**model_args).eval().to(device)
        model.load_state_dict(base.state_dict())
    stream = StreamingDBGNet(
        model,
        window=str(stft_cfg.get("window", "hann")),
        center=bool(stft_cfg.get("center", True)),
        deployment_config=cfg,
    ).eval().to(device=device, dtype=dtype)
    return stream, dtype


def _run_with_artifacts(
    lifecycle: _OwnedOnnxLifecycle,
    transaction: _VerifierArtifactTransaction,
) -> int:
    from onnx.export import write_streaming_runtime_metadata
    from onnx.infer.streaming_ort_runner import StreamingORTFrameRunner
    from onnx.models.streaming_dbgnet import StreamingDBGNet, StreamingDBGNetFlat
    from onnx.verify.verify_common import (
        freeze_release_source_binding,
        print_kept_onnx_artifact,
        publish_verified_onnx,
        release_results_pass,
        require_release_preflight,
        require_strict_runtime_spec,
    )

    # 用户决策：只验证 fp32/fp16，int8 不在精度验证范围内（不提供该档位）。
    ns = build_arg_parser().parse_args()
    if ns.mse_threshold is None:
        ns.mse_threshold = DEFAULT_MSE_THRESHOLDS[ns.inference_dtype]
    if ns.max_abs_threshold is None:
        ns.max_abs_threshold = DEFAULT_MAX_ABS_THRESHOLDS[ns.inference_dtype]
    cfg = load_merged_config(str(ns.config))
    require_release_preflight(
        save_onnx=ns.save_onnx,
        checkpoint=ns.checkpoint,
        config=cfg,
        require_config=True,
        inference_dtype=ns.inference_dtype,
        thresholds={
            "mse_threshold": (
                ns.mse_threshold, DEFAULT_MSE_THRESHOLDS["float32"]
            ),
            "max_abs_threshold": (
                ns.max_abs_threshold, DEFAULT_MAX_ABS_THRESHOLDS["float32"]
            ),
        },
    )
    checkpoint_snapshot = (
        transaction.freeze_checkpoint(ns.checkpoint)
        if ns.checkpoint is not None else None
    )
    source_binding = None
    if checkpoint_snapshot is not None:
        checkpoint_snapshot.verify()
        checkpoint_report = checkpoint_snapshot.report_binding
        source_binding = freeze_release_source_binding(
            model_id="dbgnet",
            checkpoint=ns.checkpoint,
            config=ns.config,
            checkpoint_binding={
                "path": checkpoint_report["source_path"],
                "size": checkpoint_report["snapshot_size_bytes"],
                "sha256": checkpoint_report["snapshot_sha256"],
            },
        )
        checkpoint_snapshot.require_config_provenance(cfg, "dbgnet")
        checkpoint_report = checkpoint_snapshot.report_binding
        print(
            f"Checkpoint source: {checkpoint_report['source_path']} "
            f"(verified snapshot sha256={checkpoint_report['snapshot_sha256']}, "
            f"size={checkpoint_report['snapshot_size_bytes']} bytes)"
        )

    dev = torch.device(ns.device)

    # ---- 1. 构建模型 ----
    base, model_args, cfg = _build_model_from_config(
        str(ns.config), checkpoint_snapshot
    )
    if checkpoint_snapshot is not None:
        checkpoint_snapshot.verify()
    base = base.to(dev)
    stft_cfg = cfg.get("stft", {}) or {}

    stream = StreamingDBGNet(
        base,
        window=str(stft_cfg.get("window", "hann")),
        center=bool(stft_cfg.get("center", True)),
        deployment_config=cfg,
    ).eval().to(dev)
    flat = StreamingDBGNetFlat(stream).eval()

    n_fft = base.n_fft
    hop = base.hop_length
    F_bins = n_fft // 2 + 1
    T = ns.frames
    use_state = base.use_state_detector
    state_nclass = base.state_nclass

    print(f"Config: n_fft={n_fft}, hop={hop}, freq_bins={F_bins}, "
          f"use_state={use_state}, state_nclass={state_nclass}, T={T}")
    print(f"Model: mic_channels={[base.conv_mic1[0].out_channels, base.conv_mic2[0].out_channels, base.conv_mic3[0].out_channels, base.conv_mic4[0].out_channels]}, "
          f"far_channels={[base.conv_far1[0].out_channels, base.conv_far2[0].out_channels, base.conv_far3[0].out_channels, base.conv_far4[0].out_channels]}")

    # ---- 2. 导出 ONNX（或加载已有文件）----
    in_names = flat.get_input_names()
    out_names = flat.get_output_names()
    if ns.onnx:
        if ns.save_onnx:
            # 发布 helper 会消费 source；先复制，再让 ORT 加载并验证这份 owned copy。
            onnx_artifact = _copy_caller_onnx_for_verification(ns.onnx, lifecycle)
            onnx_path = str(onnx_artifact.path)
            print(f"\nUsing copied ONNX for verification: {onnx_path}")
        else:
            onnx_path = str(ns.onnx)
            onnx_artifact = _VerifiedOnnxArtifact(onnx_path)
            print(f"\nUsing existing ONNX: {onnx_path}")
    else:
        # ``mkstemp`` 原子创建临时文件，避免 TOCTOU 竞态。
        _fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
        lifecycle.own(onnx_path)
        os.close(_fd)

        cache_tensors = flat.init_cache(1, str(dev), torch.float32)
        mic_dummy = torch.randn(1, F_bins, 1, 2, device=dev, dtype=torch.float32)
        ref_dummy = torch.randn(1, F_bins, 1, 2, device=dev, dtype=torch.float32)
        dummy_inputs = (mic_dummy, ref_dummy) + cache_tensors

        print(f"\nExporting ONNX to {onnx_path} ...")
        export_onnx(
            flat, dummy_inputs, onnx_path,
            input_names=in_names, output_names=out_names,
            dynamic_axes=flat.get_dynamic_axes(),
            opset_version=ns.opset, do_constant_folding=True, verbose=False, dynamo=False,
        )
        write_streaming_runtime_metadata(onnx_path, flat)
        print(f"  ONNX exported ({Path(onnx_path).stat().st_size / 1024 / 1024:.1f} MB)")
        onnx_artifact = _VerifiedOnnxArtifact(onnx_path)

    onnx_artifact = transaction.bind_onnx(onnx_artifact)
    sess = onnx_artifact.create_session()
    runtime_spec = require_strict_runtime_spec(sess, "dbgnet")
    from onnx.runtime_spec import runtime_spec_from_wrapper

    wrapper_spec = runtime_spec_from_wrapper(flat, "dbgnet")
    require_matching_runtime_contract(runtime_spec, wrapper_spec)
    require_batch_one_graph_contract(sess, runtime_spec)
    print(f"  ONNX inputs: {len(runtime_spec.input_names)} (expected {len(in_names)})")
    print(f"  ONNX outputs: {len(runtime_spec.output_names)} (expected {len(out_names)})")
    ort_runner = StreamingORTFrameRunner(sess, spec=runtime_spec)

    # ORT artifact 保持 fp32；低精度档位只转换独立的流式 PT 副本。
    stream_pt, spec_dtype = _build_streaming_pt_reference(
        base, model_args, cfg, dev, ns.inference_dtype
    )

    # ---- 3. 生成测试数据（waveform -> 配置 STFT）----
    torch.manual_seed(42)
    win = int(base.win_length)
    sample_rate = int(base.sample_rate)
    center = bool(stft_cfg.get("center", True))
    window_name = str(stft_cfg.get("window", "hann"))
    if window_name != "hann":
        raise ValueError(f"DBGNet verifier requires window='hann', got {window_name!r}")
    window = torch.hann_window(win, device=dev)
    n_samples = (T - 1) * hop + n_fft
    mic_wav = torch.randn(1, n_samples, device=dev)
    ref_wav = torch.randn(1, n_samples, device=dev)
    mic_full = torch.view_as_real(torch.stft(
        mic_wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=center, return_complex=True,
    ))[:, :, :T, :]
    ref_full = torch.view_as_real(torch.stft(
        ref_wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=center, return_complex=True,
    ))[:, :, :T, :]
    T_actual = min(T, mic_full.shape[2], ref_full.shape[2])
    mic_full = mic_full[:, :, :T_actual, :]
    ref_full = ref_full[:, :, :T_actual, :]

    # ---- 4/5. 分段离线 PT + 流式 PT/ORT，同步 cold reset ----
    n_resets = max(0, ns.resets)
    n_segments = n_resets + 1
    bounds = sorted({round(i * T_actual / n_segments) for i in range(n_segments + 1)})
    bounds[0], bounds[-1] = 0, T_actual
    off_enh_segments = []
    off_ps_segments = []
    pt_enh_frames = []
    pt_ps_frames = []
    ort_enh_frames = []
    ort_ps_frames = []
    max_pt_ort_mse = float("nan")
    max_pt_ort_abs = float("nan")
    worst_pt_ort_step = -1
    n_ort_compared = 0
    frame_metrics_finite = True

    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        with torch.inference_mode():
            off_out = base.forward_spectra(
                mic_full[:, :, b0:b1, :], ref_full[:, :, b0:b1, :]
            )
        if use_state:
            seg_off, seg_ps_off = off_out
            off_ps_segments.append(seg_ps_off)
        else:
            seg_off = off_out
        off_enh_segments.append(seg_off)
        cache_pt = stream_pt.init_cache(1, str(dev), spec_dtype)
        ort_runner.reset()

        for t in range(b0, b1):
            mic_f = mic_full[:, :, t:t+1, :]
            far_f = ref_full[:, :, t:t+1, :]
            with torch.inference_mode():
                pt_enh, pt_ps, cache_pt = stream_pt.forward_streaming(
                    mic_f.to(spec_dtype), far_f.to(spec_dtype), cache_pt
                )
            pt_enh_frames.append(pt_enh.float())
            if use_state and pt_ps is not None:
                pt_ps_frames.append(pt_ps.float())

            ort_value, _ = ort_runner.run_frame(mic_f, far_f)
            ort_enh = torch.from_numpy(ort_value).to(
                device=dev, dtype=mic_full.dtype
            )
            ort_enh_frames.append(ort_enh)
            if use_state and "state_probs" in ort_runner.last_outputs:
                ort_ps_frames.append(torch.from_numpy(
                    ort_runner.last_outputs["state_probs"]
                ).to(device=dev, dtype=mic_full.dtype))

            mse_e = float((pt_enh - ort_enh).pow(2).mean().item())
            abs_e = float((pt_enh - ort_enh).abs().max().item())
            finite = bool(np.isfinite(mse_e) and np.isfinite(abs_e))
            frame_metrics_finite = frame_metrics_finite and finite
            n_ort_compared += 1
            if finite and (not np.isfinite(max_pt_ort_mse) or mse_e > max_pt_ort_mse):
                max_pt_ort_mse = mse_e
                worst_pt_ort_step = t
            if finite and (not np.isfinite(max_pt_ort_abs) or abs_e > max_pt_ort_abs):
                max_pt_ort_abs = abs_e

    y_off = torch.cat(off_enh_segments, dim=2)
    y_pt = torch.cat(pt_enh_frames, dim=2)
    y_ort = torch.cat(ort_enh_frames, dim=2)
    ps_off = torch.cat(off_ps_segments, dim=1) if use_state else None
    ps_pt = _concat_state_frames(pt_ps_frames) if use_state else None
    ps_ort = _concat_state_frames(ort_ps_frames) if use_state else None

    # ---- 6. 计算三路 MSE/max_abs ----
    mse_off_pt = float((y_off - y_pt).pow(2).mean().item())
    mx_off_pt = float((y_off - y_pt).abs().max().item())
    mse_pt_ort = float((y_pt - y_ort).pow(2).mean().item())
    mx_pt_ort = float((y_pt - y_ort).abs().max().item())
    mse_off_ort = float((y_off - y_ort).pow(2).mean().item())
    mx_off_ort = float((y_off - y_ort).abs().max().item())

    pt_state_frames = int(ps_pt.shape[1]) if ps_pt is not None and ps_pt.ndim >= 2 else 0
    ort_state_frames = int(ps_ort.shape[1]) if ps_ort is not None and ps_ort.ndim >= 2 else 0
    mse_state_off_pt = None
    mse_state_pt_ort = None
    mx_state_off_pt = None
    mx_state_pt_ort = None
    if use_state and ps_off is not None and ps_pt is not None and pt_state_frames == T_actual:
        if ps_off.shape == ps_pt.shape:
            mse_state_off_pt = float((ps_off - ps_pt).pow(2).mean().item())
            mx_state_off_pt = float((ps_off - ps_pt).abs().max().item())
        if ps_ort is not None and ort_state_frames == T_actual and ps_pt.shape == ps_ort.shape:
            mse_state_pt_ort = float((ps_pt - ps_ort).pow(2).mean().item())
            mx_state_pt_ort = float((ps_pt - ps_ort).abs().max().item())

    # ---- 7. 汇总 ----
    thr = ns.mse_threshold
    abs_thr = ns.max_abs_threshold

    status_A = (
        "PASS"
        if n_ort_compared > 0 and np.isfinite(mse_off_pt)
        and np.isfinite(mx_off_pt) and mse_off_pt < thr and mx_off_pt < abs_thr
        else "FAIL"
    )
    status_B = (
        "PASS"
        if n_ort_compared > 0 and frame_metrics_finite
        and np.isfinite(max_pt_ort_mse) and np.isfinite(max_pt_ort_abs)
        and max_pt_ort_mse < thr and max_pt_ort_abs < abs_thr
        else "FAIL"
    )
    status_C = (
        "PASS"
        if n_ort_compared > 0 and np.isfinite(mse_off_ort)
        and np.isfinite(mx_off_ort) and mse_off_ort < thr and mx_off_ort < abs_thr
        else "FAIL"
    )
    state_pass = state_gate_passed(
        use_state=use_state,
        expected_frames=T_actual,
        pt_frames=pt_state_frames,
        ort_frames=ort_state_frames,
        mse_off_pt=mse_state_off_pt,
        mse_pt_ort=mse_state_pt_ort,
        max_abs_off_pt=mx_state_off_pt,
        max_abs_pt_ort=mx_state_pt_ort,
        threshold=thr,
        max_abs_threshold=abs_thr,
    )
    try:
        reconstructed = torch.istft(
            torch.view_as_complex(y_ort.contiguous()),
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=center,
        )
    except RuntimeError:
        external_istft_pass = False
    else:
        external_istft_pass = bool(
            reconstructed.numel() > 0 and torch.isfinite(reconstructed).all().item()
        )
    widened = T_actual >= MIN_WIDENED_FRAMES and n_resets >= MIN_WIDENED_RESETS
    scope = "widened" if widened else "narrow"
    lookahead_samples = stft_lookahead_samples(center=center, n_fft=n_fft)
    stft_lookahead_ms = 1000.0 * lookahead_samples / sample_rate

    print("\n" + "=" * 70)
    print("  三路精度对齐汇总 (DBGNet)")
    print("=" * 70)
    print("  神经谱图路径: future_frames=0 (causal)")
    print(f"  center STFT lookahead: {lookahead_samples} samples "
          f"({stft_lookahead_ms:.3f} ms)")
    print(f"  验证口径: {scope} waveform synthetic | frames={T_actual}, "
          f"resets={n_resets}, segments={len(bounds) - 1}")
    print(f"  STFT: sr={sample_rate}, n_fft={n_fft}, hop={hop}, win={win}, "
          f"window={window_name}, center={center}")
    print(f"  use_state_detector={use_state}, state_nclass={state_nclass}")
    print(f"  阈值: MSE<{thr:g}, max_abs<{abs_thr:g}")
    print()
    print(f"  (A) 离线 PT vs 流式 PT:")
    print(f"      MSE={mse_off_pt:.6e}  max_abs={mx_off_pt:.6e}  [{status_A}]")
    if mse_state_off_pt is not None and mx_state_off_pt is not None:
        print(f"      State MSE={mse_state_off_pt:.6e}  "
              f"max_abs={mx_state_off_pt:.6e}")
    print()
    print(f"  (B) 流式 PT vs 流式 ORT (逐帧, {n_ort_compared} frames):")
    print(f"      overall_MSE={mse_pt_ort:.6e}  max_abs={mx_pt_ort:.6e}")
    print(f"      max_frame_MSE={max_pt_ort_mse:.6e}  "
          f"max_frame_abs={max_pt_ort_abs:.6e}  worst_step={worst_pt_ort_step}  "
          f"[{status_B}]")
    if mse_state_pt_ort is not None and mx_state_pt_ort is not None:
        print(f"      State MSE={mse_state_pt_ort:.6e}  "
              f"max_abs={mx_state_pt_ort:.6e}")
    print()
    print(f"  (C) 离线 PT vs 流式 ORT:")
    print(f"      MSE={mse_off_ort:.6e}  max_abs={mx_off_ort:.6e}  [{status_C}]")
    print("=" * 70)

    if use_state:
        print(f"  State gate: {'PASS' if state_pass else 'FAIL'} "
              f"(PT frames={pt_state_frames}/{T_actual}, "
              f"ORT frames={ort_state_frames}/{T_actual})")
    print(f"  {EXTERNAL_ISTFT_CAPABILITY_LABEL}: "
          f"{'PASS' if external_istft_pass else 'FAIL'} "
          "(finite non-empty waveform only; not streaming waveform equivalence)")
    numeric_gate = release_results_pass(
        [status_A, status_B, status_C],
        compared_counts=[T_actual, n_ort_compared, T_actual],
        metrics=[mse_off_pt, max_pt_ort_mse, mse_off_ort],
        thresholds=[thr, thr, thr],
    )
    all_pass = verification_gate_passed(
        [status_A, status_B, status_C],
        widened=widened,
        save_requested=ns.save_onnx is not None,
        numeric_gate=numeric_gate,
        state_pass=state_pass,
        external_istft_pass=external_istft_pass,
    )
    print(f"  总结: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)

    if ns.save_onnx:
        published = onnx_artifact.publish(
            ns.save_onnx, all_pass=all_pass, publisher=publish_verified_onnx,
            manifest={
                "model_id": "dbgnet",
                "source_files": {"checkpoint": ns.checkpoint, "config": ns.config},
                "source_binding": source_binding,
                "runtime_contract": runtime_contract_from_spec(runtime_spec),
                "verification": {"statuses": [status_A, status_B, status_C]},
            },
            require_manifest=True,
        )
        if published:
            size_mb = published["size_bytes"] / 1024 / 1024
            print(f"  ONNX artifact kept: {published['path']} ({size_mb:.1f} MB, "
                  f"sha256={published['sha256']})")
    elif not ns.onnx:
        # 手工验证模式保留本脚本导出的临时 ONNX，不处理 caller-owned 文件。
        print_kept_onnx_artifact(onnx_path)
        lifecycle.keep()
    return 0 if all_pass else 1


def _run(lifecycle: _OwnedOnnxLifecycle) -> int:
    with _VerifierArtifactTransaction() as transaction:
        return _run_with_artifacts(lifecycle, transaction)


def main() -> int:
    with owned_onnx_cleanup() as lifecycle:
        return _run(lifecycle)


if __name__ == "__main__":
    raise SystemExit(main())
