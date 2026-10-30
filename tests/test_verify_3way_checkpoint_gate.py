"""JAE-M05（checkpoint 完整性）+ JAE-M10（DeepVQE gate / widened 口径）的单元测试。

不依赖真实训练 checkpoint 或 onnxruntime：用小模型 + 临时文件验证加载校验逻辑，
用预注册常量与纯函数 gate 验证 PASS/FAIL 判定逻辑。
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import multiprocessing
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from onnx.verify import verify_3way_cagcrn as v_cagcrn
from onnx.verify import verify_3way_dbgnet as v_dbgnet
from onnx.verify import verify_3way_deepvqe as v_deepvqe
from onnx.verify import verify_3way_fastenhancer as v_fastenhancer
from onnx.verify import verify_3way_ultra_dpc as v_ultra_dpc
from onnx.verify import verify_common
from onnx.verify.verify_common import (
    VerifiedOnnxArtifact,
    load_checkpoint_strict,
    load_checkpoint_training_config,
    owned_onnx_cleanup,
    publish_verified_onnx,
    require_checkpoint_config_provenance,
    require_exact_onnx_io,
    sha256_file,
)

_ROOT = Path(__file__).resolve().parents[1]
_VERIFY_SCRIPTS = (
    "onnx/verify/verify_3way_deepvqe.py",
    "onnx/verify/verify_3way_dbgnet.py",
    "onnx/verify/verify_3way_cagcrn.py",
    "onnx/verify/verify_3way_fastenhancer.py",
    "onnx/verify/verify_3way_gtcrn.py",
    "onnx/verify/verify_3way_tse.py",
    "onnx/verify/verify_3way_ultra_dpc.py",
    "onnx/verify/verify_3way_ulunas.py",
)


def _acquire_lock_in_child(directory, target_name, ready, acquired):
    parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    lock_fd = None
    try:
        ready.set()
        lock_fd, _, _ = verify_common._acquire_target_lock(parent_fd, target_name)
        acquired.set()
    finally:
        if lock_fd is not None:
            verify_common.fcntl.flock(lock_fd, verify_common.fcntl.LOCK_UN)
            os.close(lock_fd)
        os.close(parent_fd)


@pytest.fixture(autouse=True)
def _isolated_onnx_release_root(monkeypatch, tmp_path):
    """发布测试使用隔离根目录，不在仓库产物目录留下测试 artifact。"""
    monkeypatch.setattr(verify_common, "_ONNX_RELEASE_ROOT", tmp_path)


class _TinyNet(torch.nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.fc = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.fc(x)


def _save_ckpt(tmp_path: Path, sd: dict) -> Path:
    p = tmp_path / "ckpt.tar"
    torch.save({"model_state_dict": sd}, p)
    return p


def _aec_training_config(model_id: str, *, align_delay: int = 20) -> dict:
    model_type = "DeepVQE" if model_id == "deepvqe" else "DBGNet"
    model_args = (
        {"variant": "small", "align_delay": align_delay, "compression": 0.3}
        if model_id == "deepvqe"
        else {"sample_rate": 16000, "use_state_detector": True, "state_nclass": 3}
    )
    return {
        "model": {"type": model_type, "args": model_args},
        "stft": {
            "n_fft": 320, "hop_length": 160, "win_length": 320,
            "window": "hann", "center": True,
        },
        "data": {"sample_rate": 16000},
        "aec_frontend": {"input_routing": "mic_farend"},
        "loss": {"type": "training-semantic-marker"},
    }


# ---- M05：checkpoint 完整性校验（复用 onnx.export.load_checkpoint 逻辑）----

def test_strict_load_ok(tmp_path):
    model = _TinyNet()
    load_checkpoint_strict(model, _save_ckpt(tmp_path, model.state_dict()))  # 不抛异常即通过


def test_strict_load_strips_module_prefix(tmp_path):
    """DDP 保存的 ``module.`` 前缀是已知唯一 legacy 前缀，显式迁移后应正常加载。"""
    model = _TinyNet()
    sd = {f"module.{k}": v for k, v in model.state_dict().items()}
    load_checkpoint_strict(model, _save_ckpt(tmp_path, sd))


def test_strict_load_missing_unexpected_keys_raise(tmp_path):
    """等参数量换 key 也必须由共享 strict loader 拒绝。"""
    model = _TinyNet()
    sd = dict(model.state_dict())
    sd["fc.bias_renamed"] = sd.pop("fc.bias")
    with pytest.raises(RuntimeError, match="state_dict 不匹配"):
        load_checkpoint_strict(model, _save_ckpt(tmp_path, sd))


def test_strict_load_shape_mismatch_raises(tmp_path):
    """同名 tensor shape 与模型构造不一致时必须在写入前报错。"""
    model = _TinyNet(width=4)
    big = _TinyNet(width=64)
    with pytest.raises(RuntimeError, match="tensor shape"):
        load_checkpoint_strict(model, _save_ckpt(tmp_path, big.state_dict()))


def test_strict_load_rejects_ambiguous_single_model_wrappers(tmp_path):
    """ONNX loader 与 evaluate loader 一致拒绝多个单模型权重候选。"""
    model = _TinyNet()
    state = model.state_dict()
    checkpoint = tmp_path / "ambiguous.tar"
    torch.save({"model": state, "model_state_dict": state}, checkpoint)

    with pytest.raises(RuntimeError, match="多个候选"):
        load_checkpoint_strict(model, checkpoint)


@pytest.mark.parametrize("model_id", ("deepvqe", "dbgnet"))
def test_checkpoint_provenance_accepts_matching_embedded_resolved_config(
    tmp_path, model_id,
):
    config = _aec_training_config(model_id)
    checkpoint = tmp_path / "best_model.tar"
    torch.save({"model": _TinyNet().state_dict(), "resolved_config": config}, checkpoint)

    binding = require_checkpoint_config_provenance(checkpoint, config, model_id)

    assert binding["resolved_config"]["algorithm"] == "sha256-json-v1"
    assert len(binding["resolved_config"]["sha256"]) == 64


def test_deepvqe_checkpoint_provenance_rejects_shape_compatible_align_delay_drift(
    tmp_path,
):
    training_config = _aec_training_config("deepvqe", align_delay=100)
    verifier_config = _aec_training_config("deepvqe", align_delay=20)
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {"model": _TinyNet().state_dict(), "resolved_config": training_config},
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="fingerprint does not match"):
        require_checkpoint_config_provenance(
            checkpoint, verifier_config, "deepvqe"
        )


def test_checkpoint_provenance_prefers_sidecar_and_rejects_embedded_drift(tmp_path):
    import yaml

    sidecar_config = _aec_training_config("deepvqe", align_delay=100)
    embedded_config = _aec_training_config("deepvqe", align_delay=20)
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {"model": _TinyNet().state_dict(), "resolved_config": embedded_config},
        checkpoint,
    )
    (tmp_path / "resolved_config.yaml").write_text(
        yaml.safe_dump(sidecar_config, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="sidecar does not match embedded"):
        load_checkpoint_training_config(checkpoint, "deepvqe")


def _config_provenance(config: dict, source: str) -> dict:
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )
    return {
        "version": 1,
        "algorithm": "sha256-json-v1",
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "source": source,
    }


def test_checkpoint_provenance_uses_bound_resumed_sidecar(tmp_path):
    import yaml

    original = _aec_training_config("deepvqe", align_delay=100)
    resumed = _aec_training_config("deepvqe", align_delay=20)
    source = "resolved_config_resumed_bound.yaml"
    (tmp_path / "resolved_config.yaml").write_text(
        yaml.safe_dump(original, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / source).write_text(
        yaml.safe_dump(resumed, sort_keys=False), encoding="utf-8"
    )
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {
            "model": _TinyNet().state_dict(),
            "config": resumed,
            "resolved_config": resumed,
            "resolved_config_provenance": _config_provenance(resumed, source),
        },
        checkpoint,
    )

    assert load_checkpoint_training_config(checkpoint, "deepvqe") == resumed


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("../resolved_config.yaml", "source"),
        ("resolved_config_missing.yaml", "missing"),
    ),
)
def test_checkpoint_provenance_rejects_invalid_bound_source(tmp_path, source, expected):
    config = _aec_training_config("deepvqe")
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {
            "model": _TinyNet().state_dict(),
            "resolved_config": config,
            "resolved_config_provenance": _config_provenance(config, source),
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match=expected):
        load_checkpoint_training_config(checkpoint, "deepvqe")


@pytest.mark.parametrize(
    ("mutate", "expected"),
    (
        (lambda value: {**value, "version": True}, "version"),
        (lambda value: {**value, "unexpected": 1}, "fields mismatch"),
        (lambda value: {**value, "algorithm": "sha1"}, "algorithm"),
        (lambda value: {**value, "sha256": "0"}, "SHA-256"),
    ),
)
def test_checkpoint_provenance_rejects_malformed_binding(tmp_path, mutate, expected):
    import yaml

    config = _aec_training_config("deepvqe")
    source = "resolved_config_bound.yaml"
    (tmp_path / source).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {
            "model": _TinyNet().state_dict(),
            "config": config,
            "resolved_config": config,
            "resolved_config_provenance": mutate(
                _config_provenance(config, source)
            ),
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match=expected):
        load_checkpoint_training_config(checkpoint, "deepvqe")


def test_checkpoint_provenance_rejects_tampered_bound_sidecar(tmp_path):
    import yaml

    config = _aec_training_config("deepvqe", align_delay=20)
    tampered = _aec_training_config("deepvqe", align_delay=100)
    source = "resolved_config_resumed_bound.yaml"
    (tmp_path / source).write_text(
        yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8"
    )
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {
            "model": _TinyNet().state_dict(),
            "resolved_config": config,
            "resolved_config_provenance": _config_provenance(config, source),
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="identity"):
        load_checkpoint_training_config(checkpoint, "deepvqe")


def test_checkpoint_provenance_rejects_conflicting_embedded_config_aliases(tmp_path):
    current = _aec_training_config("deepvqe", align_delay=20)
    stale = _aec_training_config("deepvqe", align_delay=100)
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {
            "model": _TinyNet().state_dict(),
            "config": stale,
            "resolved_config": current,
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="embedded.*conflict"):
        load_checkpoint_training_config(checkpoint, "deepvqe")


def test_legacy_checkpoint_selects_unique_sidecar_matching_embedded_config(tmp_path):
    import yaml

    original = _aec_training_config("deepvqe", align_delay=100)
    resumed = _aec_training_config("deepvqe", align_delay=20)
    (tmp_path / "resolved_config.yaml").write_text(
        yaml.safe_dump(original, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "resolved_config_resumed.yaml").write_text(
        yaml.safe_dump(resumed, sort_keys=False), encoding="utf-8"
    )
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {"model": _TinyNet().state_dict(), "resolved_config": resumed},
        checkpoint,
    )

    assert load_checkpoint_training_config(checkpoint, "deepvqe") == resumed


def test_legacy_checkpoint_without_embedded_config_rejects_sidecar_ambiguity(tmp_path):
    import yaml

    (tmp_path / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            _aec_training_config("deepvqe", align_delay=100), sort_keys=False
        ),
        encoding="utf-8",
    )
    (tmp_path / "resolved_config_resumed.yaml").write_text(
        yaml.safe_dump(
            _aec_training_config("deepvqe", align_delay=20), sort_keys=False
        ),
        encoding="utf-8",
    )
    checkpoint = _save_ckpt(tmp_path, _TinyNet().state_dict())

    with pytest.raises(RuntimeError, match="ambiguous"):
        load_checkpoint_training_config(checkpoint, "deepvqe")


def test_checkpoint_provenance_rejects_missing_training_config(tmp_path):
    checkpoint = _save_ckpt(tmp_path, _TinyNet().state_dict())

    with pytest.raises(RuntimeError, match="config provenance is missing"):
        require_checkpoint_config_provenance(
            checkpoint, _aec_training_config("deepvqe"), "deepvqe"
        )


@pytest.mark.unit
def test_safe_torch_load_accepts_rewound_seekable_stream():
    """file-like 安全加载每次从 frozen 内容起点读取。"""
    import io

    from core.utils.safe_load import safe_torch_load

    stream = io.BytesIO()
    torch.save({"value": torch.tensor([1.0])}, stream)
    stream.seek(0, os.SEEK_END)

    loaded = safe_torch_load(stream)

    torch.testing.assert_close(loaded["value"], torch.tensor([1.0]))


@pytest.mark.unit
def test_safe_torch_load_stream_keeps_unsafe_purpose_fail_closed():
    """file-like 扩展不能绕过 allow_unsafe 的 purpose 合同。"""
    import io

    from core.utils.safe_load import safe_torch_load

    stream = io.BytesIO()
    torch.save({"value": torch.tensor([1.0])}, stream)

    with pytest.raises(ValueError, match="non-empty purpose"):
        safe_torch_load(stream, allow_unsafe=True)


@pytest.mark.unit
def test_dbgnet_checkpoint_snapshot_binds_provenance_weights_and_report_after_replace(
    tmp_path,
):
    """路径换 inode 后，provenance、权重与报告仍来自同一 frozen bytes。"""
    config = _aec_training_config("dbgnet")
    original = _TinyNet()
    replacement = _TinyNet()
    with torch.no_grad():
        original.fc.weight.fill_(1.0)
        original.fc.bias.fill_(2.0)
        replacement.fc.weight.fill_(7.0)
        replacement.fc.bias.fill_(8.0)
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {"model_state_dict": original.state_dict(), "resolved_config": config},
        checkpoint,
    )

    with v_dbgnet._VerifierArtifactTransaction() as transaction:
        snapshot = transaction.freeze_checkpoint(checkpoint)
        replacement_path = tmp_path / "replacement.tar"
        torch.save(
            {"model_state_dict": replacement.state_dict(), "resolved_config": config},
            replacement_path,
        )
        os.replace(replacement_path, checkpoint)

        binding = snapshot.require_config_provenance(config, "dbgnet")
        loaded = _TinyNet()
        snapshot.load_model(loaded)
        with snapshot.open_checkpoint() as frozen:
            frozen_content = frozen.read()

        assert binding["resolved_config"]["sha256"]
        assert snapshot.report_binding == {
            "source_path": str(checkpoint),
            "snapshot_sha256": hashlib.sha256(frozen_content).hexdigest(),
            "snapshot_size_bytes": len(frozen_content),
        }
        assert sha256_file(checkpoint) != snapshot.sha256
        for key, value in original.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value)

    assert checkpoint.is_file()


@pytest.mark.unit
def test_dbgnet_checkpoint_snapshot_binds_relative_source_path(tmp_path, monkeypatch):
    """相对 checkpoint 路径也必须与 release source binding 的绝对路径一致。"""
    config = _aec_training_config("dbgnet")
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {"model_state_dict": _TinyNet().state_dict(), "resolved_config": config},
        checkpoint,
    )

    monkeypatch.chdir(tmp_path)
    with v_dbgnet._VerifierArtifactTransaction() as transaction:
        snapshot = transaction.freeze_checkpoint(Path("best_model.tar"))
        report = snapshot.report_binding
        binding = verify_common.freeze_release_source_binding(
            model_id="dbgnet",
            checkpoint=Path("best_model.tar"),
            checkpoint_binding={
                "path": report["source_path"],
                "size": report["snapshot_size_bytes"],
                "sha256": report["snapshot_sha256"],
            },
        )

    assert report["source_path"] == str(checkpoint)
    assert binding["files"]["checkpoint"]["path"] == str(checkpoint)


@pytest.mark.unit
def test_dbgnet_checkpoint_snapshot_survives_root_writable_same_inode_overwrite(
    tmp_path,
):
    """不依赖 mode：同 UID/root 成功覆写原 inode 后仍只加载 frozen bytes。"""
    config = _aec_training_config("dbgnet")
    original = _TinyNet()
    replacement = _TinyNet()
    with torch.no_grad():
        original.fc.weight.fill_(1.0)
        replacement.fc.weight.fill_(9.0)
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {"model_state_dict": original.state_dict(), "resolved_config": config},
        checkpoint,
    )

    with v_dbgnet._VerifierArtifactTransaction() as transaction:
        snapshot = transaction.freeze_checkpoint(checkpoint)
        inode_before = checkpoint.stat().st_ino
        replacement_bytes = io.BytesIO()
        torch.save(
            {"model_state_dict": replacement.state_dict(), "resolved_config": config},
            replacement_bytes,
        )
        with checkpoint.open("wb") as writable:
            writable.write(replacement_bytes.getvalue())
        assert checkpoint.stat().st_ino == inode_before

        snapshot.verify()
        snapshot.require_config_provenance(config, "dbgnet")
        loaded = _TinyNet()
        snapshot.load_model(loaded)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value)
        assert sha256_file(checkpoint) != snapshot.sha256


@pytest.mark.unit
def test_dbgnet_checkpoint_snapshot_freezes_bound_sidecar_before_same_inode_overwrite(
    tmp_path,
):
    """bound sidecar 被同 UID/root 覆写后，provenance 仍只读取 frozen bytes。"""
    import yaml

    config = _aec_training_config("dbgnet")
    tampered = _aec_training_config("dbgnet")
    tampered["stft"]["center"] = False
    source = "resolved_config_bound.yaml"
    sidecar = tmp_path / source
    sidecar.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {
            "model_state_dict": _TinyNet().state_dict(),
            "config": config,
            "resolved_config": config,
            "resolved_config_provenance": _config_provenance(config, source),
        },
        checkpoint,
    )

    with v_dbgnet._VerifierArtifactTransaction() as transaction:
        snapshot = transaction.freeze_checkpoint(checkpoint)
        inode_before = sidecar.stat().st_ino
        sidecar.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")
        assert sidecar.stat().st_ino == inode_before

        binding = snapshot.require_config_provenance(config, "dbgnet")
        assert binding["resolved_config"]["sha256"] == _config_provenance(
            config, source
        )["sha256"]


@pytest.mark.unit
def test_dbgnet_artifact_transaction_cleans_onnx_snapshot_on_exception(tmp_path):
    """checkpoint/sidecar 无临时路径，后续异常仍清理 verifier-owned ONNX。"""
    checkpoint = tmp_path / "best_model.tar"
    torch.save(
        {
            "model_state_dict": _TinyNet().state_dict(),
            "resolved_config": _aec_training_config("dbgnet"),
        },
        checkpoint,
    )
    onnx_source = tmp_path / "caller.onnx"
    onnx_source.write_bytes(b"caller-owned-onnx")
    onnx_snapshot = None

    with pytest.raises(RuntimeError, match="injected failure"):
        with v_dbgnet._VerifierArtifactTransaction() as transaction:
            transaction.freeze_checkpoint(checkpoint)
            artifact = transaction.bind_onnx(
                v_dbgnet._VerifiedOnnxArtifact(onnx_source)
            )
            artifact.create_session(lambda *_args, **_kwargs: object())
            onnx_snapshot = artifact._snapshot_path
            assert onnx_snapshot is not None and onnx_snapshot.is_file()
            raise RuntimeError("injected failure")

    assert onnx_snapshot is not None and not onnx_snapshot.exists()
    assert checkpoint.is_file()
    assert onnx_source.read_bytes() == b"caller-owned-onnx"


def test_onnx_loader_requires_explicit_gan_generator_family(tmp_path):
    """GAN checkpoint 不得被单模型路径猜测加载，但显式 family 可正常加载。"""
    from onnx.export import load_checkpoint

    source = _TinyNet()
    checkpoint = tmp_path / "gan.tar"
    torch.save({"model_generator": source.state_dict()}, checkpoint)

    with pytest.raises(RuntimeError, match="single_model checkpoint"):
        load_checkpoint(_TinyNet(), checkpoint)
    target = _TinyNet()
    load_checkpoint(target, checkpoint, family="gan_generator")
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_onnx_loader_applies_ema_after_base_weights(monkeypatch, tmp_path):
    """ONNX 导出与 inference/evaluate 保持相同的 EMA 覆盖顺序。"""
    from onnx.export import load_checkpoint

    class _FakeEMA:
        def __init__(self, parameters, decay):
            self.parameters = list(parameters)
            self.shadow_params = []

        def load_state_dict(self, state):
            self.shadow_params = state["shadow_params"]

        def copy_to(self, parameters):
            for parameter, value in zip(parameters, self.shadow_params):
                parameter.data.copy_(value)

    fake_ema = ModuleType("torch_ema")
    fake_ema.ExponentialMovingAverage = _FakeEMA
    monkeypatch.setitem(sys.modules, "torch_ema", fake_ema)

    target = _TinyNet()
    base = {key: torch.zeros_like(value) for key, value in target.state_dict().items()}
    ema_values = [torch.full_like(parameter, 7.0) for parameter in target.parameters()]
    checkpoint = tmp_path / "ema.tar"
    torch.save({"model": base, "ema": {"shadow_params": ema_values}}, checkpoint)

    load_checkpoint(target, checkpoint)
    for parameter in target.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 7.0))


def test_onnx_loader_resolves_ddp_compile_dbgnet_wrapper_and_ddp_prefix(
    monkeypatch, tmp_path,
):
    """受控 resolver 必须递归解开 DDP、compile 与 DBGNet flat wrapper。"""
    from tasks.aec.models.discriminative.dbgnet import DBGNet
    from onnx.export import load_checkpoint
    from onnx.models.streaming_dbgnet import StreamingDBGNet, StreamingDBGNetFlat

    class _FakeEMA:
        def __init__(self, parameters, decay):
            self.shadow_params = []

        def load_state_dict(self, state):
            self.shadow_params = state["shadow_params"]

    class _CompileWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    class _FakeDDP(torch.nn.parallel.DistributedDataParallel):
        def __init__(self, module):
            torch.nn.Module.__init__(self)
            self.module = module

    fake_ema = ModuleType("torch_ema")
    fake_ema.ExponentialMovingAverage = _FakeEMA
    monkeypatch.setitem(sys.modules, "torch_ema", fake_ema)

    target = DBGNet(
        n_fft=128,
        hop_length=64,
        win_length=128,
        erb_low_bins=17,
        erb_bands=16,
        mic_channels=[2, 4, 6, 8],
        far_channels=[1, 2, 3, 4],
        use_state_detector=False,
    )
    flat = StreamingDBGNetFlat(StreamingDBGNet(target))
    wrapped = _FakeDDP(_CompileWrapper(flat))
    base = {
        f"module.{key}": torch.zeros_like(value)
        for key, value in target.state_dict().items()
    }
    ema_values = [torch.full_like(parameter, 9.0) for parameter in target.parameters()]
    checkpoint = tmp_path / "wrapped_ema.tar"
    torch.save({"model": base, "ema": {"shadow_params": ema_values}}, checkpoint)

    load_checkpoint(wrapped, checkpoint)
    for parameter in target.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 9.0))


def test_onnx_loader_resolves_streaming_bsrnn_wrapper(tmp_path):
    """BSRNN wrapper 只允许沿 ``model`` 边定位训练模型。"""
    from onnx.export import load_checkpoint
    from onnx.models.streaming_bsrnn import StreamingBSRNNWrapper

    class _TinyBSRNN(_TinyNet):
        def __init__(self):
            super().__init__()
            self.bsrnn = torch.nn.Module()
            self.bsrnn.causal = True

    source = _TinyBSRNN()
    target = _TinyBSRNN()
    checkpoint = _save_ckpt(tmp_path, source.state_dict())

    load_checkpoint(StreamingBSRNNWrapper(target), checkpoint)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_onnx_checkpoint_target_resolver_rejects_wrapper_cycle():
    """compile wrapper 循环引用必须 fail-fast，不能无限递归。"""
    from onnx.export import _resolve_checkpoint_target

    class _CompileCycle(torch.nn.Module):
        @property
        def _orig_mod(self):
            return self

    with pytest.raises(RuntimeError, match="wrapper cycle"):
        _resolve_checkpoint_target(_CompileCycle())


def test_onnx_loader_rejects_legacy_compact_state_off_dbgnet_topology(tmp_path):
    """真实 compact state-off DBGNet 缺拓扑版本 key 时正式 loader 必须阻断。"""
    from tasks.aec.models.discriminative.dbgnet import DBGNet
    from onnx.export import load_checkpoint
    from onnx.models.streaming_dbgnet import StreamingDBGNet, StreamingDBGNetFlat

    model_args = {
        "n_fft": 128,
        "hop_length": 64,
        "win_length": 128,
        "sample_rate": 16000,
        "erb_low_bins": 17,
        "erb_bands": 16,
        "mic_channels": [2, 4, 6, 8],
        "far_channels": [1, 2, 3, 4],
        "use_state_detector": False,
    }
    source = DBGNet(**model_args)
    legacy_state = dict(source.state_dict())
    legacy_state.pop("_recursive_topology_v1")
    checkpoint = _save_ckpt(tmp_path, legacy_state)
    target = DBGNet(**model_args)
    original = {key: value.clone() for key, value in target.state_dict().items()}
    wrapped = StreamingDBGNetFlat(StreamingDBGNet(target))

    with pytest.raises(RuntimeError, match="_recursive_topology_v1"):
        load_checkpoint(wrapped, checkpoint)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, original[key])


# ---- M10：DeepVQE 预注册阈值与双指标 gate ----

def test_preregistered_thresholds_separate_jitter_from_normal():
    """预注册阈值必须落在"正常 sibling"与"jitter 异常"之间（数值来源见脚本头注释）。"""
    assert 1e-12 < v_deepvqe.PREREG_MSE_THRESHOLD_F32 < 1.8e-7
    assert 1e-5 < v_deepvqe.PREREG_MAX_ABS_THRESHOLD_F32 < 2.8e-3


def test_gate_status_requires_both_metrics():
    mse_thr = v_deepvqe.PREREG_MSE_THRESHOLD_F32
    abs_thr = v_deepvqe.PREREG_MAX_ABS_THRESHOLD_F32
    # 正常 sibling 量级 → PASS
    assert v_deepvqe.gate_status(1e-12, 1e-5, mse_thr, abs_thr) == "PASS"
    # jitter 异常量级 → FAIL（旧口径只看 MSE<1e-6 会 PASS，正是 JAE-M10 要堵的口子）
    assert v_deepvqe.gate_status(1.8e-7, 2.8e-3, mse_thr, abs_thr) == "FAIL"
    # 单项超标即 FAIL
    assert v_deepvqe.gate_status(1e-12, 2.8e-3, mse_thr, abs_thr) == "FAIL"
    assert v_deepvqe.gate_status(1.8e-7, 1e-5, mse_thr, abs_thr) == "FAIL"


@pytest.mark.unit
def test_deepvqe_cold_start_gate_requires_all_frames_and_both_ac_paths():
    """稳态 PASS 不得掩盖 cold-start/reset 帧缺失或 A/C 任一路失败。"""
    assert v_deepvqe.cold_start_gate_passed(
        "PASS", "PASS", compared_frames=12, expected_frames=12,
    )
    assert not v_deepvqe.cold_start_gate_passed(
        "PASS", "PASS", compared_frames=11, expected_frames=12,
    )
    assert not v_deepvqe.cold_start_gate_passed(
        "FAIL", "PASS", compared_frames=12, expected_frames=12,
    )
    assert not v_deepvqe.cold_start_gate_passed(
        "PASS", "FAIL", compared_frames=12, expected_frames=12,
    )


@pytest.mark.unit
@pytest.mark.parametrize("center", [False, True])
def test_deepvqe_waveform_reconstruction_gate_enforces_nola(center):
    """Hann + center=False 不满足边界 NOLA，必须阻断 waveform 发布门禁。"""
    n_fft = 16
    hop = 8
    window = torch.hann_window(n_fft)
    spec = torch.zeros(1, n_fft // 2 + 1, 8, 2)

    passed = v_deepvqe.waveform_reconstruction_passed(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=center,
    )

    assert passed is center


@pytest.mark.unit
@pytest.mark.parametrize("failed_gate", ["cold_start_pass", "waveform_pass"])
def test_deepvqe_final_gate_requires_cold_start_and_waveform(failed_gate):
    """任一新增硬门禁失败都不得被 widened 三路数值 PASS 绕过。"""
    gates = {"cold_start_pass": True, "waveform_pass": True}
    assert v_deepvqe.verification_gate_passed(
        ["PASS", "PASS", "PASS"],
        widened=True,
        save_requested=True,
        numeric_gate=True,
        **gates,
    )

    gates[failed_gate] = False
    assert not v_deepvqe.verification_gate_passed(
        ["PASS", "PASS", "PASS"],
        widened=True,
        save_requested=True,
        numeric_gate=True,
        **gates,
    )


def test_deepvqe_parser_defaults_are_widened():
    """默认口径必须满足 widened：frames ≥1000 且 resets ≥2。"""
    ns = v_deepvqe.build_arg_parser().parse_args([])
    assert ns.frames >= v_deepvqe.MIN_WIDENED_FRAMES
    assert ns.resets >= v_deepvqe.MIN_WIDENED_RESETS


def test_deepvqe_publish_gate_rejects_narrow_numeric_pass():
    """数值 A/B/C 全 PASS 也不能把 narrow 验证用于发布。"""
    assert not v_deepvqe.publish_gate_passed(
        ["PASS", "PASS", "PASS"], widened=False, save_requested=True,
    )
    assert v_deepvqe.publish_gate_passed(
        ["PASS", "PASS", "PASS"], widened=True, save_requested=True,
    )
    # 无发布目标时保持原有手工验证的数值退出语义。
    assert v_deepvqe.publish_gate_passed(
        ["PASS", "PASS", "PASS"], widened=False, save_requested=False,
    )


def test_dbgnet_parser_defaults_are_widened():
    """DBGNet 默认口径必须满足 widened：frames ≥1000 且 resets ≥2。"""
    ns = v_dbgnet.build_arg_parser().parse_args(["--config", "dummy.yaml"])
    assert ns.frames >= v_dbgnet.MIN_WIDENED_FRAMES
    assert ns.resets >= v_dbgnet.MIN_WIDENED_RESETS


@pytest.mark.parametrize(
    ("module", "prefix"),
    ((v_deepvqe, []), (v_dbgnet, ["--config", "dummy.yaml"])),
)
def test_aec_verifier_parsers_reject_empty_or_negative_workloads(module, prefix):
    """空帧与负 reset 必须在昂贵模型/导出步骤前 fail-closed。"""
    parser = module.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*prefix, "--frames", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args([*prefix, "--resets", "-1"])


def test_dbgnet_publish_gate_rejects_narrow_numeric_pass():
    """DBGNet narrow 数值全 PASS 仍不得用于发布。"""
    assert not v_dbgnet.publish_gate_passed(
        ["PASS", "PASS", "PASS"], widened=False, save_requested=True,
    )
    assert v_dbgnet.publish_gate_passed(
        ["PASS", "PASS", "PASS"], widened=True, save_requested=True,
    )
    assert v_dbgnet.publish_gate_passed(
        ["PASS", "PASS", "PASS"], widened=False, save_requested=False,
    )


def _runtime_contract(**overrides):
    values = {
        "input_names": ("mic_spec", "far_spec", "cache"),
        "output_names": ("enh", "cache_out"),
        "cache_shapes": {"cache": (1, 2)},
        "dynamic_axes": {},
        "stft": "stft",
        "deepvqe": "deepvqe",
        "dbgnet": "dbgnet",
        "deployment_config": {"sha256": "same"},
    }
    values.update(overrides)
    return type("_RuntimeContract", (), values)()


@pytest.mark.parametrize(
    "module", [v_deepvqe, v_dbgnet],
)
def test_aec_verifier_runtime_contract_rejects_io_and_cache_drift(module):
    """相同 model metadata 不能掩盖 graph I/O 或 cache shape 漂移。"""
    expected = _runtime_contract()
    module.require_matching_runtime_contract(_runtime_contract(), expected)

    with pytest.raises(RuntimeError, match="input names"):
        module.require_matching_runtime_contract(
            _runtime_contract(input_names=("mic_spec", "far_spec", "other_cache")),
            expected,
        )
    with pytest.raises(RuntimeError, match="cache shapes"):
        module.require_matching_runtime_contract(
            _runtime_contract(cache_shapes={"cache": (1, 3)}), expected
        )


def test_dbgnet_verifier_rejects_dynamic_or_non_one_graph_batch():
    """metadata 是 batch-one 模板时，verifier 仍须拒绝动态图和 batch>1 图。"""
    class _Info:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class _Session:
        def __init__(self, batch):
            self.batch = batch

        def get_inputs(self):
            return [
                _Info("mic_spec", [self.batch, 161, 1, 2]),
                _Info("far_spec", [self.batch, 161, 1, 2]),
                _Info("cache", [1, 2]),
            ]

        def get_outputs(self):
            return [
                _Info("enh", [self.batch, 161, 1, 2]),
                _Info("cache_out", [1, 2]),
            ]

    spec = _runtime_contract()
    v_dbgnet.require_batch_one_graph_contract(_Session(1), spec)
    with pytest.raises(RuntimeError, match="static batch_size=1"):
        v_dbgnet.require_batch_one_graph_contract(_Session("batch"), spec)
    with pytest.raises(RuntimeError, match="static batch_size=1"):
        v_dbgnet.require_batch_one_graph_contract(_Session(2), spec)


def test_dbgnet_float16_reference_uses_independent_half_precision_model():
    """float16 档位必须真实运行 half，且不得连带修改离线 fp32 锚点。"""
    from tasks.aec.models.discriminative.dbgnet import DBGNet

    model_args = {
        "n_fft": 320,
        "hop_length": 160,
        "win_length": 320,
        "sample_rate": 16000,
        "erb_low_bins": 41,
        "erb_bands": 40,
        "mic_channels": [16, 32, 48, 64],
        "far_channels": [8, 16, 24, 32],
        "use_state_detector": False,
    }
    cfg = {"stft": {"window": "hann", "center": True}}
    base = DBGNet(**model_args).eval()

    stream, dtype = v_dbgnet._build_streaming_pt_reference(
        base, model_args, cfg, torch.device("cpu"), "float16"
    )
    assert dtype == torch.float16
    assert next(base.parameters()).dtype == torch.float32
    assert next(stream.parameters()).dtype == torch.float16

    frame = torch.randn(1, 161, 1, 2, dtype=dtype)
    cache = stream.init_cache(1, "cpu", dtype)
    with torch.inference_mode():
        enhanced, state_logits, updated = stream.forward_streaming(
            frame, frame, cache
        )
    assert enhanced.dtype == dtype
    assert state_logits is None
    assert all(value.dtype == dtype for value in updated.values())


def test_dbgnet_state_gate_requires_complete_finite_mse_and_max_abs():
    """state 输出缺失、非有限或任一 MSE/max_abs 超阈值都必须阻断。"""
    valid = {
        "use_state": True,
        "expected_frames": 4,
        "pt_frames": 4,
        "ort_frames": 4,
        "mse_off_pt": 1e-6,
        "mse_pt_ort": 1e-6,
        "max_abs_off_pt": 1e-4,
        "max_abs_pt_ort": 1e-4,
        "threshold": 1e-5,
        "max_abs_threshold": 5e-3,
    }
    assert v_dbgnet.state_gate_passed(**valid)
    for overrides in (
        {"ort_frames": 3},
        {"mse_pt_ort": None},
        {"max_abs_off_pt": None},
        {"max_abs_pt_ort": float("nan")},
        {"mse_off_pt": 1e-4},
        {"max_abs_pt_ort": 1e-2},
    ):
        assert not v_dbgnet.state_gate_passed(**(valid | overrides))
    assert v_dbgnet.state_gate_passed(
        use_state=False, expected_frames=4, pt_frames=0, ort_frames=0,
        mse_off_pt=None, mse_pt_ort=None, threshold=1e-5,
    )


def test_dbgnet_causality_contract_separates_neural_and_center_stft_lookahead():
    """神经谱图零未来帧不能掩盖 center STFT 的前端 lookahead。"""
    assert v_dbgnet.stft_lookahead_samples(center=True, n_fft=320) == 160
    assert v_dbgnet.stft_lookahead_samples(center=False, n_fft=320) == 0


def test_dbgnet_external_istft_capability_has_exact_bounded_name():
    """能力名称必须明确是外部 synthesis，而非 waveform 等价验证。"""
    assert v_dbgnet.EXTERNAL_ISTFT_CAPABILITY_LABEL == (
        "External iSTFT synthesis capability"
    )


@pytest.mark.parametrize("failed_gate", ["state_pass", "external_istft_pass"])
def test_dbgnet_final_gate_requires_state_and_external_istft_capability(failed_gate):
    """state 或外部 iSTFT synthesis capability 失败均不得发布。"""
    gates = {"state_pass": True, "external_istft_pass": True}
    assert v_dbgnet.verification_gate_passed(
        ["PASS", "PASS", "PASS"],
        widened=True,
        save_requested=True,
        numeric_gate=True,
        **gates,
    )
    gates[failed_gate] = False
    assert not v_dbgnet.verification_gate_passed(
        ["PASS", "PASS", "PASS"],
        widened=True,
        save_requested=True,
        numeric_gate=True,
        **gates,
    )


def test_cagcrn_publish_gate_rejects_missing_io():
    assert not v_cagcrn.publish_gate_passed(
        ["PASS", "PASS", "PASS"], missing_in=["cache_0"], missing_out=[],
    )
    assert not v_cagcrn.publish_gate_passed(
        ["PASS", "PASS", "PASS"], missing_in=[], missing_out=["enh_spec"],
    )
    assert v_cagcrn.publish_gate_passed(
        ["PASS", "PASS", "PASS"], missing_in=[], missing_out=[],
    )


def _record_session_source(seen, source, providers):
    seen.append(source)
    assert isinstance(source, bytes)
    assert providers == ["CPUExecutionProvider"]
    return object()


def _recording_publisher(
    seen,
    source,
    target,
    *,
    all_pass,
    expected_sha256,
    expected_size_bytes,
    expected_device,
    expected_inode,
    source_fd,
):
    source_path = Path(source)
    assert source_path.is_file()
    seen.append((source_path, all_pass, expected_sha256, expected_size_bytes))
    return publish_verified_onnx(
        source_path,
        target,
        all_pass=all_pass,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        expected_device=expected_device,
        expected_inode=expected_inode,
        source_fd=source_fd,
    )


def test_deepvqe_verifier_wires_narrow_gate_to_verified_source(monkeypatch, tmp_path):
    """narrow save 必须把 ORT 同一 source 以 all_pass=False 交给 helper。"""
    source = tmp_path / "deepvqe-owned.onnx"
    source.write_bytes(b"deepvqe-verified-by-ort")
    target = tmp_path / "published.onnx"
    target.write_bytes(b"old-published")
    session_sources = []
    publish_calls = []
    artifact = v_deepvqe._VerifiedOnnxArtifact(source)
    monkeypatch.setattr(
        v_deepvqe,
        "publish_verified_onnx",
        lambda src, dst, *, all_pass, expected_sha256, expected_size_bytes, expected_device, expected_inode, source_fd: (
            _recording_publisher(
                publish_calls,
                src,
                dst,
                all_pass=all_pass,
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
                expected_device=expected_device,
                expected_inode=expected_inode,
                source_fd=source_fd,
            )
        ),
        raising=False,
    )

    artifact.create_session(
        lambda src, providers: _record_session_source(session_sources, src, providers),
    )
    all_pass = v_deepvqe.publish_gate_passed(
        ["PASS", "PASS", "PASS"], widened=False, save_requested=True,
    )
    published = artifact.publish(
        target, all_pass=all_pass, publisher=v_deepvqe.publish_verified_onnx,
    )

    assert session_sources == [b"deepvqe-verified-by-ort"]
    assert len(publish_calls) == 1
    assert publish_calls[0][1:] == (
        False,
        hashlib.sha256(b"deepvqe-verified-by-ort").hexdigest(),
        len(b"deepvqe-verified-by-ort"),
    )
    assert published is None
    assert target.read_bytes() == b"old-published"
    assert not source.exists()


def test_dbgnet_verifier_copies_caller_source_before_session_and_publish(monkeypatch, tmp_path):
    """caller ONNX + save 必须由 ORT 验证 owned copy，helper 也只消费该 copy。"""
    caller = tmp_path / "caller.onnx"
    caller.write_bytes(b"caller-owned")
    owned = tmp_path / "dbgnet-owned-copy.onnx"
    target = tmp_path / "published.onnx"
    target.write_bytes(b"old-published")
    real_mkstemp = v_dbgnet.tempfile.mkstemp
    session_sources = []
    publish_calls = []

    def mkstemp_in_test_dir(*, suffix):
        assert suffix == ".onnx"
        return real_mkstemp(dir=tmp_path, suffix=suffix)

    monkeypatch.setattr(v_dbgnet.tempfile, "mkstemp", mkstemp_in_test_dir)
    monkeypatch.setattr(
        v_dbgnet,
        "publish_verified_onnx",
        lambda src, dst, *, all_pass, expected_sha256, expected_size_bytes, expected_device, expected_inode, source_fd: (
            _recording_publisher(
                publish_calls,
                src,
                dst,
                all_pass=all_pass,
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
                expected_device=expected_device,
                expected_inode=expected_inode,
                source_fd=source_fd,
            )
        ),
        raising=False,
    )

    with v_dbgnet.owned_onnx_cleanup() as lifecycle:
        artifact = v_dbgnet._copy_caller_onnx_for_verification(caller, lifecycle)
        owned = artifact.path
        artifact.create_session(
            lambda src, providers: _record_session_source(session_sources, src, providers),
        )
        published = artifact.publish(
            target, all_pass=False, publisher=v_dbgnet.publish_verified_onnx,
        )

    assert owned != caller
    assert session_sources == [b"caller-owned"]
    assert len(publish_calls) == 1
    assert publish_calls[0][1:] == (
        False,
        hashlib.sha256(b"caller-owned").hexdigest(),
        len(b"caller-owned"),
    )
    assert published is None
    assert caller.read_bytes() == b"caller-owned"
    assert target.read_bytes() == b"old-published"
    assert not owned.exists()


def test_cagcrn_verifier_wires_io_gate_to_verified_source(monkeypatch, tmp_path):
    """缺 IO 必须把 ORT 同一 source 以 all_pass=False 交给 helper。"""
    source = tmp_path / "cagcrn-owned.onnx"
    source.write_bytes(b"cagcrn-verified-by-ort")
    target = tmp_path / "published.onnx"
    target.write_bytes(b"old-published")
    session_sources = []
    publish_calls = []
    artifact = v_cagcrn._VerifiedOnnxArtifact(source)
    monkeypatch.setattr(
        v_cagcrn,
        "publish_verified_onnx",
        lambda src, dst, *, all_pass, expected_sha256, expected_size_bytes, expected_device, expected_inode, source_fd: (
            _recording_publisher(
                publish_calls,
                src,
                dst,
                all_pass=all_pass,
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
                expected_device=expected_device,
                expected_inode=expected_inode,
                source_fd=source_fd,
            )
        ),
        raising=False,
    )

    artifact.create_session(
        lambda src, providers: _record_session_source(session_sources, src, providers),
    )
    all_pass = v_cagcrn.publish_gate_passed(
        ["PASS", "PASS", "PASS"], missing_in=["cache_0"], missing_out=[],
    )
    published = artifact.publish(
        target, all_pass=all_pass, publisher=v_cagcrn.publish_verified_onnx,
    )

    assert session_sources == [b"cagcrn-verified-by-ort"]
    assert len(publish_calls) == 1
    assert publish_calls[0][1:] == (
        False,
        hashlib.sha256(b"cagcrn-verified-by-ort").hexdigest(),
        len(b"cagcrn-verified-by-ort"),
    )
    assert published is None
    assert target.read_bytes() == b"old-published"
    assert not source.exists()


@pytest.mark.parametrize("module", [
    v_deepvqe,
    v_dbgnet,
    v_cagcrn,
    v_fastenhancer,
    v_ultra_dpc,
])
def test_owned_onnx_cleanup_removes_temp_after_exception(module, tmp_path):
    """五个 verifier 的 owned temp 在异常退出时均不得泄漏。"""
    owned = tmp_path / f"{module.__name__.rsplit('.', 1)[-1]}.onnx"
    owned.write_bytes(b"partial-or-unverified")

    with pytest.raises(RuntimeError, match="injected failure"):
        with module.owned_onnx_cleanup() as lifecycle:
            lifecycle.own(owned)
            raise RuntimeError("injected failure")

    assert not owned.exists()


def test_owned_onnx_cleanup_can_transfer_keep_ownership(tmp_path):
    """无 save 且原行为要求保留时，显式 keep 后 finally 不删除。"""
    owned = tmp_path / "kept.onnx"
    owned.write_bytes(b"verified")

    with v_deepvqe.owned_onnx_cleanup() as lifecycle:
        lifecycle.own(owned)
        lifecycle.keep()

    assert owned.read_bytes() == b"verified"


# ---- 共享小工具与三脚本 --help 冒烟 ----

def test_sha256_file(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_require_exact_onnx_io_rejects_extra_missing_and_reordered_names():
    """发布门禁必须拒绝额外、缺失或顺序变化的图 I/O。"""
    require_exact_onnx_io(["x"], ["y"], ["x"], ["y"])
    for actual_inputs, actual_outputs in (
        (["x", "extra"], ["y"]),
        ([], ["y"]),
        (["x"], ["y", "extra"]),
        (["x"], []),
        (["cache", "x"], ["y"]),
    ):
        with pytest.raises(RuntimeError, match="I/O ABI mismatch"):
            require_exact_onnx_io(["x"], ["y"], actual_inputs, actual_outputs)


# ---- PASS-gated ONNX artifact 发布 ----


def _publish_expected(source: Path, target: Path):
    payload = source.read_bytes()
    return publish_verified_onnx(
        source,
        target,
        all_pass=True,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )


def test_publish_verified_onnx_pass_creates_nested_target_with_digest(tmp_path):
    """漏建父目录或写错摘要时应失败。"""
    source = tmp_path / "verified.onnx"
    payload = b"verified-onnx-bytes"
    source.write_bytes(payload)
    target = tmp_path / "nested" / "artifacts" / "model.onnx"

    published = _publish_expected(source, target)

    assert target.read_bytes() == payload
    assert published == {
        "path": str(target),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    assert not source.exists()


def test_publish_verified_onnx_pass_replaces_existing_target(tmp_path):
    """缺少原子替换时，已发布 artifact 不会被新验证结果更新。"""
    source = tmp_path / "verified.onnx"
    source.write_bytes(b"new-verified-artifact")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"old-artifact")

    _publish_expected(source, target)

    assert target.read_bytes() == b"new-verified-artifact"
    assert not source.exists()


def test_publish_verified_onnx_pass_rejects_normalized_source_target_alias(tmp_path):
    """缺少 alias guard 会在发布后删除同一路径的目标 artifact。"""
    source = tmp_path / "model.onnx"
    payload = b"verified-artifact"
    source.write_bytes(payload)
    target = tmp_path / "." / "model.onnx"

    with pytest.raises(ValueError, match="must not alias"):
        _publish_expected(source, target)

    assert source.exists()
    assert source.read_bytes() == payload
    assert target.read_bytes() == payload


def test_publish_verified_onnx_fail_rejects_source_target_alias_without_cleanup(tmp_path):
    source = tmp_path / "failed.onnx"
    payload = b"failed-artifact"
    source.write_bytes(payload)

    with pytest.raises(ValueError, match="must not alias"):
        publish_verified_onnx(source, source, all_pass=False)

    assert source.read_bytes() == payload


def test_publish_verified_onnx_pass_stages_in_target_parent_and_replaces(tmp_path, monkeypatch):
    """新建发布必须使用同目录 staging 与原子 no-replace CAS。"""
    source = tmp_path / "verified.onnx"
    source.write_bytes(b"verified-artifact")
    target = tmp_path / "published" / "model.onnx"
    rename_calls = []
    real_rename = verify_common._rename_noreplace

    def record_rename(parent_fd, staging, destination):
        rename_calls.append((parent_fd, staging, destination))
        real_rename(parent_fd, staging, destination)

    monkeypatch.setattr(verify_common, "_rename_noreplace", record_rename)

    _publish_expected(source, target)

    assert len(rename_calls) == 1
    parent_fd, staging, destination = rename_calls[0]
    assert staging.startswith(".onnx-stage-")
    assert destination == target.name
    assert parent_fd >= 0
    assert target.read_bytes() == b"verified-artifact"


def test_verified_artifact_uses_snapshot_after_source_aba(tmp_path):
    source = tmp_path / "verified.onnx"
    source.write_bytes(b"session-verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous-publication")
    seen = []
    artifact = VerifiedOnnxArtifact(source)

    def session_factory(model_bytes, providers):
        seen.append(model_bytes)
        source.write_bytes(b"malicious-middle")
        source.write_bytes(b"session-verified")
        return object()

    artifact.create_session(session_factory)
    source.write_bytes(b"mutated-after-session")
    artifact.publish(target, all_pass=True)

    assert seen == [b"session-verified"]
    assert target.read_bytes() == b"session-verified"


def test_verified_artifact_preserves_replaced_source_and_publishes_snapshot(tmp_path):
    source = tmp_path / "verified.onnx"
    source.write_bytes(b"session-verified")
    replacement = tmp_path / "replacement.onnx"
    replacement.write_bytes(b"replaced-content")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous-publication")
    artifact = VerifiedOnnxArtifact(source)
    artifact.create_session(lambda *_args, **_kwargs: object())

    os.replace(replacement, source)
    artifact.publish(target, all_pass=True)

    assert source.read_bytes() == b"replaced-content"
    assert target.read_bytes() == b"session-verified"


def test_publish_verified_onnx_rejects_source_drift_during_fd_copy(
    monkeypatch, tmp_path,
):
    source = tmp_path / "verified.onnx"
    payload = b"session-verified"
    source.write_bytes(payload)
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous-publication")
    real_copy_fd = verify_common._copy_fd

    def copy_then_mutate(source_fd, staging_fd, chunk_size=1 << 20):
        real_copy_fd(source_fd, staging_fd, chunk_size)
        source.write_bytes(b"copy-race-drifts")

    monkeypatch.setattr(verify_common, "_copy_fd", copy_then_mutate)

    with pytest.raises(RuntimeError, match="source after copy fingerprint mismatch"):
        publish_verified_onnx(
            source,
            target,
            all_pass=True,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size_bytes=len(payload),
        )
    assert target.read_bytes() == b"previous-publication"


def test_publish_verified_onnx_requires_bound_fingerprint_and_preserves_target(tmp_path):
    source = tmp_path / "verified.onnx"
    source.write_bytes(b"session-verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous-publication")

    with pytest.raises(RuntimeError, match="requires expected_sha256"):
        publish_verified_onnx(source, target, all_pass=True)
    assert source.read_bytes() == b"session-verified"
    assert target.read_bytes() == b"previous-publication"


def test_publish_verified_onnx_fail_preserves_existing_target(tmp_path):
    """FAIL 分支若替换目标，会销毁此前通过验证的 artifact。"""
    source = tmp_path / "failed.onnx"
    source.write_bytes(b"failed-artifact")
    target = tmp_path / "model.onnx"
    original = b"previously-published-artifact"
    target.write_bytes(original)

    published = publish_verified_onnx(source, target, all_pass=False)

    assert published is None
    assert target.read_bytes() == original
    assert not source.exists()


def test_publish_verified_onnx_fail_without_target_creates_none(tmp_path):
    """FAIL 分支若意外发布，会生成不可信的 artifact。"""
    source = tmp_path / "failed.onnx"
    source.write_bytes(b"failed-artifact")
    target = tmp_path / "nested" / "model.onnx"

    published = publish_verified_onnx(source, target, all_pass=False)

    assert published is None
    assert not target.exists()
    assert not source.exists()


def test_publish_verified_onnx_fail_preserves_source_aba_replacement(
    monkeypatch, tmp_path,
):
    source = tmp_path / "failed.onnx"
    replacement = tmp_path / "replacement.onnx"
    source.write_bytes(b"failed-artifact")
    replacement.write_bytes(b"replacement")
    target = tmp_path / "model.onnx"
    real_unlink_source = verify_common._unlink_source_if_owned

    def replace_before_cleanup(path, source_stat):
        os.replace(replacement, source)
        real_unlink_source(path, source_stat)

    monkeypatch.setattr(
        verify_common, "_unlink_source_if_owned", replace_before_cleanup,
    )

    assert publish_verified_onnx(source, target, all_pass=False) is None
    assert source.read_bytes() == b"replacement"
    assert not target.exists()


@pytest.mark.parametrize("target_factory", [
    lambda root, outside: outside / "absolute.onnx",
    lambda root, outside: root / ".." / "outside" / "traversal.onnx",
])
def test_publish_verified_onnx_rejects_external_targets_and_cleans_source(
    tmp_path, target_factory,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    outside = tmp_path.parent / "outside"
    target = target_factory(tmp_path, outside)

    with pytest.raises(ValueError, match="publication target"):
        _publish_expected(source, target)

    assert not source.exists()
    assert not target.exists()


def test_publish_verified_onnx_rejects_symlink_parent_escape_and_preserves_target(tmp_path):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    outside = tmp_path.parent / "outside_symlink"
    outside.mkdir()
    escaped_parent = tmp_path / "escape"
    escaped_parent.symlink_to(outside, target_is_directory=True)
    target = escaped_parent / "model.onnx"

    with pytest.raises(ValueError, match="publication target"):
        _publish_expected(source, target)

    assert not source.exists()
    assert not target.exists()


def test_publish_verified_onnx_rolls_back_parent_symlink_swap(
    monkeypatch, tmp_path,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    parent = tmp_path / "release"
    parent.mkdir()
    target = parent / "model.onnx"
    target.write_bytes(b"previous")
    moved_parent = tmp_path / "release-moved"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_replace = verify_common.os.replace
    swapped = False

    def swap_parent_before_replace(src, dst, **kwargs):
        nonlocal swapped
        if not swapped and str(src).startswith(".onnx-stage-"):
            swapped = True
            parent.rename(moved_parent)
            parent.symlink_to(outside, target_is_directory=True)
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(verify_common.os, "replace", swap_parent_before_replace)

    with pytest.raises(RuntimeError, match="previous target restored"):
        _publish_expected(source, target)

    assert not (outside / "model.onnx").exists()
    assert (moved_parent / "model.onnx").read_bytes() == b"previous"


@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_publish_verified_onnx_rejects_staging_entry_replacement(
    monkeypatch, tmp_path, attack,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    attacker = tmp_path / "attacker.onnx"
    attacker.write_bytes(b"attacker")
    real_require = verify_common._require_anchored_entry
    injected = False

    def replace_staging(parent_fd, name, held_stat, *, label):
        nonlocal injected
        if not injected and label == "staged ONNX artifact":
            injected = True
            if attack == "symlink":
                os.unlink(name, dir_fd=parent_fd)
                os.symlink(attacker, name, dir_fd=parent_fd)
            else:
                os.link(
                    name,
                    ".attacker-hardlink.onnx",
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
        return real_require(parent_fd, name, held_stat, label=label)

    monkeypatch.setattr(verify_common, "_require_anchored_entry", replace_staging)

    with pytest.raises(RuntimeError, match="staged ONNX artifact"):
        _publish_expected(source, target)

    assert target.read_bytes() == b"previous"


def test_publish_verified_onnx_rejects_staging_mutation_after_hash(
    monkeypatch, tmp_path,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    real_fingerprint = verify_common._fingerprint_fd
    injected = False

    def mutate_after_hash(fd, *, label, chunk_size=1 << 20):
        nonlocal injected
        result = real_fingerprint(fd, label=label, chunk_size=chunk_size)
        if not injected and label == "staged ONNX artifact before replace":
            injected = True
            os.fchmod(fd, 0o600)
            writer = os.open(f"/proc/self/fd/{fd}", os.O_WRONLY)
            try:
                os.pwrite(writer, b"tampered", 0)
            finally:
                os.close(writer)
        return result

    monkeypatch.setattr(verify_common, "_fingerprint_fd", mutate_after_hash)

    with pytest.raises(RuntimeError, match="at commit fingerprint mismatch"):
        _publish_expected(source, target)

    assert target.read_bytes() == b"previous"


@pytest.mark.parametrize("has_previous", [False, True])
def test_publish_verified_onnx_rolls_back_directory_fsync_failure(
    monkeypatch, tmp_path, has_previous,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    if has_previous:
        target.write_bytes(b"previous")
    real_fsync = verify_common.os.fsync
    failed = False

    def fail_commit_fsync(fd):
        nonlocal failed
        if not failed and stat.S_ISDIR(os.fstat(fd).st_mode):
            failed = True
            raise OSError("injected directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(verify_common.os, "fsync", fail_commit_fsync)

    with pytest.raises(RuntimeError, match="previous target restored"):
        _publish_expected(source, target)

    if has_previous:
        assert target.read_bytes() == b"previous"
    else:
        assert not target.exists()


def test_publish_existing_target_uses_atomic_exchange_without_missing_window(
    monkeypatch, tmp_path,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    real_exchange = verify_common._rename_exchange
    observations = []

    def observe_exchange(parent_fd, left_name, right_name):
        assert left_name.startswith(".onnx-stage-")
        assert right_name == target.name
        assert os.stat(right_name, dir_fd=parent_fd, follow_symlinks=False)
        real_exchange(parent_fd, left_name, right_name)
        observations.append(target.read_bytes())

    monkeypatch.setattr(verify_common, "_rename_exchange", observe_exchange)

    _publish_expected(source, target)

    assert observations == [b"verified"]
    assert target.read_bytes() == b"verified"


def test_publish_atomic_exchange_restores_noncooperative_replacement(
    monkeypatch, tmp_path,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    attacker = tmp_path / "attacker.onnx"
    attacker.write_bytes(b"concurrent")
    real_exchange = verify_common._rename_exchange
    injected = False

    def replace_before_exchange(parent_fd, left_name, right_name):
        nonlocal injected
        if not injected:
            injected = True
            os.replace(attacker, target)
        real_exchange(parent_fd, left_name, right_name)

    monkeypatch.setattr(verify_common, "_rename_exchange", replace_before_exchange)

    with pytest.raises(RuntimeError, match="exchange CAS conflict"):
        _publish_expected(source, target)

    assert target.read_bytes() == b"concurrent"


def test_publish_exchange_swapback_failure_uses_identity_checked_rollback(
    monkeypatch, tmp_path,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    attacker = tmp_path / "attacker.onnx"
    attacker.write_bytes(b"concurrent")
    real_exchange = verify_common._rename_exchange
    calls = 0

    def fail_second_exchange(parent_fd, left_name, right_name):
        nonlocal calls
        calls += 1
        if calls == 1:
            os.replace(attacker, target)
            return real_exchange(parent_fd, left_name, right_name)
        raise OSError("injected swapback failure")

    monkeypatch.setattr(verify_common, "_rename_exchange", fail_second_exchange)

    with pytest.raises(RuntimeError, match="previous target restored"):
        _publish_expected(source, target)

    assert calls == 2
    assert target.read_bytes() == b"concurrent"


def test_publish_create_cas_preserves_noncooperative_new_target(monkeypatch, tmp_path):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    real_rename = verify_common._rename_noreplace
    injected = False

    def create_before_noreplace(parent_fd, left_name, right_name):
        nonlocal injected
        if not injected:
            injected = True
            target.write_bytes(b"concurrent")
        real_rename(parent_fd, left_name, right_name)

    monkeypatch.setattr(verify_common, "_rename_noreplace", create_before_noreplace)

    with pytest.raises(RuntimeError, match="create CAS conflict"):
        _publish_expected(source, target)

    assert target.read_bytes() == b"concurrent"


def test_publish_post_fsync_validation_failure_reports_committed(
    monkeypatch, tmp_path,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    real_require = verify_common._require_anchored_entry

    def fail_after_fsync(parent_fd, name, held_stat, *, label):
        if label == "published ONNX artifact after fsync":
            raise RuntimeError("injected post-fsync validation failure")
        return real_require(parent_fd, name, held_stat, label=label)

    monkeypatch.setattr(verify_common, "_require_anchored_entry", fail_after_fsync)

    with pytest.raises(RuntimeError, match="committed.*post-fsync"):
        _publish_expected(source, target)

    assert target.read_bytes() == b"verified"


def test_publish_staging_cleanup_failure_is_observable(monkeypatch, tmp_path):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    real_fingerprint = verify_common._fingerprint_fd
    real_unlink = verify_common.os.unlink

    def fail_fingerprint(fd, *, label, chunk_size=1 << 20):
        if label == "staged ONNX artifact before replace":
            raise RuntimeError("injected precommit failure")
        return real_fingerprint(fd, label=label, chunk_size=chunk_size)

    def retain_staging(path, **kwargs):
        if str(path).startswith(".onnx-stage-cleanup-"):
            raise OSError("injected staging cleanup failure")
        return real_unlink(path, **kwargs)

    monkeypatch.setattr(verify_common, "_fingerprint_fd", fail_fingerprint)
    monkeypatch.setattr(verify_common.os, "unlink", retain_staging)

    with pytest.raises(RuntimeError, match="staging_cleanup=retained"):
        _publish_expected(source, target)

    assert any(path.name.startswith(".onnx-stage-cleanup-") for path in tmp_path.iterdir())


def test_publish_persistent_directory_fsync_failure_rolls_back_only_once(
    monkeypatch, tmp_path,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    real_fsync = verify_common.os.fsync
    restore_calls = 0
    real_replace = verify_common.os.replace

    def fail_all_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("persistent directory fsync failure")
        return real_fsync(fd)

    def count_restore(src, dst, **kwargs):
        nonlocal restore_calls
        if str(src).startswith(".onnx-backup-") and dst == target.name:
            restore_calls += 1
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(verify_common.os, "fsync", fail_all_directory_fsync)
    monkeypatch.setattr(verify_common.os, "replace", count_restore)

    with pytest.raises(RuntimeError, match="rollback durability or identity is unconfirmed"):
        _publish_expected(source, target)

    assert restore_calls == 1
    assert target.read_bytes() == b"previous"


def test_publish_staging_create_failure_closes_parent_fds(monkeypatch, tmp_path):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    before = len(os.listdir("/proc/self/fd"))

    def fail_create(*_args, **_kwargs):
        raise OSError("injected staging create failure")

    monkeypatch.setattr(verify_common, "_create_anchored_file", fail_create)

    with pytest.raises(OSError, match="staging create failure"):
        _publish_expected(source, target)

    assert len(os.listdir("/proc/self/fd")) == before
    assert not target.exists()


def test_verified_snapshot_create_failure_closes_fd_and_directory(
    monkeypatch, tmp_path,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    artifact = VerifiedOnnxArtifact(source)
    snapshot_root = tmp_path / "snapshot-private"
    real_open = verify_common.os.open
    before = len(os.listdir("/proc/self/fd"))

    monkeypatch.setattr(
        verify_common.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(snapshot_root.mkdir() or snapshot_root),
    )

    def fail_snapshot_open(path, flags, *args, **kwargs):
        if Path(path).name == "snapshot.onnx":
            raise OSError("injected snapshot create failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(verify_common.os, "open", fail_snapshot_open)

    with pytest.raises(OSError, match="snapshot create failure"):
        artifact.create_session(lambda *_args, **_kwargs: object())

    assert len(os.listdir("/proc/self/fd")) == before
    assert not snapshot_root.exists()


def test_owned_onnx_cleanup_preserves_aba_replacement(tmp_path):
    owned = tmp_path / "owned.onnx"
    replacement = tmp_path / "replacement.onnx"
    owned.write_bytes(b"owned")
    replacement.write_bytes(b"replacement")

    with owned_onnx_cleanup() as lifecycle:
        lifecycle.own(owned)
        os.replace(replacement, owned)

    assert owned.read_bytes() == b"replacement"


def test_unlink_source_if_owned_preserves_aba_replacement(tmp_path):
    source = tmp_path / "owned.onnx"
    replacement = tmp_path / "replacement.onnx"
    source.write_bytes(b"owned")
    source_stat = source.stat()
    replacement.write_bytes(b"replacement")
    os.replace(replacement, source)

    verify_common._unlink_source_if_owned(source, source_stat)

    assert source.read_bytes() == b"replacement"


@pytest.mark.parametrize("attack", ["symlink", "different-inode"])
def test_rollback_rejects_replaced_backup(tmp_path, attack):
    target = tmp_path / "model.onnx"
    target.write_bytes(b"staged")
    staged_stat = target.stat()
    backup = tmp_path / ".onnx-backup-test.onnx"
    backup.write_bytes(b"previous")
    backup_stat = backup.stat()
    replacement = tmp_path / "replacement.onnx"
    replacement.write_bytes(b"attacker")
    if attack == "symlink":
        backup.unlink()
        backup.symlink_to(replacement)
        expected_error = "must be a regular file"
    else:
        os.replace(replacement, backup)
        expected_error = "backup identity changed"

    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RuntimeError, match=expected_error):
            verify_common._rollback_publication(
                parent_fd,
                target.name,
                backup.name,
                staged_stat=staged_stat,
                backup_stat=backup_stat,
            )
    finally:
        os.close(parent_fd)

    assert target.read_bytes() == b"staged"


def test_publish_backup_cleanup_failure_is_explicit_and_target_remains_verified(
    monkeypatch, tmp_path,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    real_unlink = verify_common.os.unlink

    def retain_backup(path, **kwargs):
        if str(path).startswith(".onnx-backup-"):
            raise OSError("injected backup cleanup failure")
        return real_unlink(path, **kwargs)

    monkeypatch.setattr(verify_common.os, "unlink", retain_backup)

    published = _publish_expected(source, target)

    assert published["backup_cleanup"] == "retained"
    assert target.read_bytes() == b"verified"
    assert any(path.name.startswith(".onnx-backup-") for path in tmp_path.iterdir())


def test_target_lock_serializes_same_target_but_not_different_target(tmp_path):
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    lock_fd = None
    other_lock_fd = None
    process = None
    try:
        lock_fd, lock_name, _ = verify_common._acquire_target_lock(
            parent_fd, "model.onnx",
        )
        other_lock_fd, other_lock_name, _ = verify_common._acquire_target_lock(
            parent_fd, "other.onnx",
        )
        assert lock_name != other_lock_name

        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        acquired = context.Event()
        process = context.Process(
            target=_acquire_lock_in_child,
            args=(tmp_path, "model.onnx", ready, acquired),
        )
        process.start()
        assert ready.wait(5)
        assert not acquired.wait(0.2)

        verify_common.fcntl.flock(lock_fd, verify_common.fcntl.LOCK_UN)
        os.close(lock_fd)
        lock_fd = None
        assert acquired.wait(5)
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(5)
        for held_fd in (other_lock_fd, lock_fd):
            if held_fd is not None:
                verify_common.fcntl.flock(held_fd, verify_common.fcntl.LOCK_UN)
                os.close(held_fd)
        os.close(parent_fd)


def test_publish_target_replacement_after_fsync_reports_committed_conflict(
    monkeypatch, tmp_path,
):
    source = tmp_path / "owned.onnx"
    source.write_bytes(b"verified")
    target = tmp_path / "model.onnx"
    target.write_bytes(b"previous")
    attacker = tmp_path / "attacker.onnx"
    attacker.write_bytes(b"attacker")
    real_fsync = verify_common.os.fsync
    injected = False

    def replace_target_during_fsync(fd):
        nonlocal injected
        if not injected and stat.S_ISDIR(os.fstat(fd).st_mode):
            injected = True
            os.replace(attacker, target)
        return real_fsync(fd)

    monkeypatch.setattr(verify_common.os, "fsync", replace_target_during_fsync)

    with pytest.raises(RuntimeError, match="committed.*post-fsync"):
        _publish_expected(source, target)

    assert target.read_bytes() == b"attacker"


@pytest.mark.parametrize("script", _VERIFY_SCRIPTS)
def test_verify_scripts_help(script):
    """全部三路 verifier 的 CLI 都应暴露统一的 artifact 发布参数。"""
    result = subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True, text=True, timeout=120, cwd=_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "--save-onnx" in result.stdout


def test_all_verify_scripts_use_verified_artifact_for_session_and_publish():
    for relative_path in _VERIFY_SCRIPTS:
        script = _ROOT / relative_path
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        attribute_calls = {
            node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
        }
        direct_calls = {
            node.func.id for node in calls if isinstance(node.func, ast.Name)
        }
        assert "create_session" in attribute_calls, script
        assert "publish" in attribute_calls, script
        assert "create_ort_session" not in direct_calls, script
        assert "publish_verified_onnx" not in direct_calls, script


class _FakeOrtSession:
    def __init__(self, source, providers, seen, fail, input_names, output_names):
        assert isinstance(source, bytes)
        seen.append(source)
        assert providers == ["CPUExecutionProvider"]
        self.fail = fail
        self.input_names = list(input_names)
        self.output_names = list(output_names)

    def get_modelmeta(self):
        from types import SimpleNamespace

        from onnx.runtime_spec import RUNTIME_METADATA_KEY, RUNTIME_METADATA_VERSION

        model_id = "fastenhancer" if "spec_in" in self.input_names else "ultra_dpc"
        runtime_metadata = {
            "version": RUNTIME_METADATA_VERSION,
            "model_id": model_id,
            "stft": {
                "n_fft": None,
                "hop_length": None,
                "win_length": None,
                "sample_rate": None,
                "window": "hann",
                "center": None,
            },
        }
        return SimpleNamespace(custom_metadata_map={
            "fastenhancer_cache_cfg": '{"cache": [1]}',
            "ultra_dpc_cache_cfg": '{"cache": [1]}',
            RUNTIME_METADATA_KEY: json.dumps(runtime_metadata),
        })

    def get_inputs(self):
        from types import SimpleNamespace
        return [
            SimpleNamespace(name=name, shape=[1])
            for name in self.input_names
        ]

    def get_outputs(self):
        from types import SimpleNamespace
        return [SimpleNamespace(name=name) for name in self.output_names]

    def run(self, _outputs, feeds):
        primary = next(value for key, value in feeds.items() if key in {"spec", "spec_in", "mic_spec"})
        if self.fail:
            primary = primary + 1.0
        return [primary] + [value for key, value in feeds.items() if key not in {"spec", "spec_in", "mic_spec", "ref_spec", "err_spec"}]


def _install_late_verifier_fakes(monkeypatch, module, tmp_path, fail, seen):
    """注入最小模型/export/ORT 依赖，但由真实 verifier `_run` 计算 gate 并发布。"""
    import torch

    class _Base(torch.nn.Module):
        def eval(self): return self
        def to(self, *_args): return self
        def forward(self, x): return x
        def forward_spectra(self, mic, _ref, _err): return mic

    class _Inner:
        CACHE_KEYS = ("cache",)
        conv_lookahead = total_delay = 0
        freq_bins = 2
        n_fft = df_lookahead = df_order = 0
        def init_cache(self, *_args):
            return {"cache": torch.zeros(1), "step": torch.tensor(0)}
        def __call__(self, spec, cache):
            return spec, cache

    class _Stream:
        def __init__(self, *_args): self.inner = _Inner()
        def eval(self): return self
        def to(self, *_args): return self
        def init_cache(self, *_args): return {"cache": torch.zeros(1)}
        def __call__(self, first, *_args):
            return first, _args[-1]

    class _Flat:
        def __init__(self, stream): self.inner = stream.inner
        def eval(self): return self
        def get_dynamic_axes(self): return {}
        def init_cache_flat(self, *_args): return [torch.zeros(1)]
        def get_input_names(self):
            if module is v_fastenhancer: return ["spec_in", "cache"]
            return ["mic_spec", "ref_spec", "err_spec", "cache"]
        def get_output_names(self):
            if module is v_fastenhancer: return ["spec_out", "out_cache"]
            return ["enh_spec", "out_cache"]

    def fake_export(_model, _inputs, path, **_kwargs):
        Path(path).write_bytes(b"verified-onnx")

    fake_ort = ModuleType("onnxruntime")

    def fake_session(path, providers):
        if module is v_fastenhancer:
            input_names = ["spec_in", "cache"]
            output_names = ["spec_out", "out_cache"]
        else:
            input_names = ["mic_spec", "ref_spec", "err_spec", "cache"]
            output_names = ["enh_spec", "out_cache"]
        return _FakeOrtSession(
            path, providers, seen["sessions"], fail, input_names, output_names,
        )

    fake_ort.InferenceSession = fake_session
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(torch.onnx, "export", fake_export)
    monkeypatch.setattr(module, "export_onnx", fake_export)
    # fake exporter 不是合法 ONNX；隔离 public metadata writer，但保留后续 strict ABI gate。
    import importlib
    export_module = importlib.import_module("onnx.export")
    monkeypatch.setattr(
        export_module, "write_streaming_runtime_metadata", lambda *_args, **_kwargs: {},
    )
    real_mkstemp = module.tempfile.mkstemp

    def fake_mkstemp(**kwargs):
        if "dir" in kwargs:
            return real_mkstemp(**kwargs)
        owned = tmp_path / "owned.onnx"
        return os.open(owned, os.O_CREAT | os.O_RDWR), str(owned)

    monkeypatch.setattr(module.tempfile, "mkstemp", fake_mkstemp)

    if module is v_fastenhancer:
        streaming = ModuleType("onnx.models.streaming_fastenhancer")
        streaming.StreamingFastEnhancer = _Stream
        streaming.StreamingFastEnhancerFlat = _Flat
        model = ModuleType("tasks.se.models.discriminative.fastenhancer")
        model.FastEnhancer = lambda **_kwargs: _Base()
        monkeypatch.setitem(sys.modules, "onnx.models.streaming_fastenhancer", streaming)
        monkeypatch.setitem(sys.modules, "tasks.se.models.discriminative.fastenhancer", model)
    else:
        streaming = ModuleType("onnx.models.streaming_ultra_dpc")
        streaming.StreamingUltraDPCAEC = _Stream
        streaming.StreamingUltraDPCAECFlat = _Flat
        model = ModuleType("tasks.aec.models.discriminative.ultra_dpc_aec")
        model.UltraDPCAEC = lambda **_kwargs: _Base()
        monkeypatch.setitem(sys.modules, "onnx.models.streaming_ultra_dpc", streaming)
        monkeypatch.setitem(sys.modules, "tasks.aec.models.discriminative.ultra_dpc_aec", model)


@pytest.mark.parametrize("module", [v_fastenhancer, v_ultra_dpc])
@pytest.mark.parametrize("fail", [False, True])
def test_late_verifier_run_wires_ort_source_to_actual_publication_gate(
    monkeypatch, tmp_path, module, fail,
):
    """真实 `_run` 必须将 ORT 所验 owned 文件与真实 PASS/FAIL gate 一起交给 publisher。"""
    import os

    target = tmp_path / "target.onnx"
    target.write_bytes(b"previous-publication")
    checkpoint = tmp_path / "checkpoint.tar"
    checkpoint.write_bytes(b"fake-checkpoint")
    config = tmp_path / "resolved_config.yaml"
    config.write_text("model: verifier-fixture\n", encoding="utf-8")
    seen = {"sessions": [], "publish": []}
    _install_late_verifier_fakes(monkeypatch, module, tmp_path, fail, seen)
    monkeypatch.setattr(verify_common, "load_checkpoint_strict", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify.py",
            "--frames",
            "1",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
            "--save-onnx",
            str(target),
        ],
    )
    def record_publish(
        source,
        destination,
        *,
        all_pass,
        expected_sha256,
        expected_size_bytes,
        expected_device,
        expected_inode,
        source_fd,
        manifest,
    ):
        source_path = Path(source)
        assert source_path.is_file()
        seen["publish"].append((
            source_path,
            Path(destination),
            all_pass,
            expected_sha256,
            expected_size_bytes,
        ))
        return publish_verified_onnx(
            source_path,
            destination,
            all_pass=all_pass,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            expected_device=expected_device,
            expected_inode=expected_inode,
            source_fd=source_fd,
            manifest=manifest,
        )

    monkeypatch.setattr(verify_common, "publish_verified_onnx", record_publish)

    with module.owned_onnx_cleanup() as lifecycle:
        rc = module._run(lifecycle)

    source, destination, all_pass, expected_sha256, expected_size_bytes = (
        seen["publish"].pop()
    )
    assert seen["sessions"] == [b"verified-onnx"]
    assert destination == target
    assert all_pass is (not fail)
    assert expected_sha256 == hashlib.sha256(b"verified-onnx").hexdigest()
    assert expected_size_bytes == len(b"verified-onnx")
    assert rc == (1 if fail else 0)
    if fail:
        assert target.read_bytes() == b"previous-publication"
    else:
        assert target.read_bytes() == b"verified-onnx"


def test_ulunas_gate_rejects_nonfinite_or_incomplete_frames():
    """逐帧 NaN 或缺帧时，UL-UNAS 三路 gate 必须全部 FAIL。"""
    from onnx.verify import verify_3way_ulunas as v_ulunas

    statuses, _ = v_ulunas._evaluate_three_way_statuses(
        [0.0, float("nan")], 0.0, 0.0, expected_frames=2, threshold=1e-4,
    )
    assert statuses == [False, False, False]

    statuses, _ = v_ulunas._evaluate_three_way_statuses(
        [0.0], 0.0, 0.0, expected_frames=2, threshold=1e-4,
    )
    assert statuses == [False, False, False]


def test_materialize_cache_shapes_uses_declared_dynamic_axis():
    """cache metadata 应按输入图声明的 axis 0/1/2 扩展，并拒绝未知轴。"""
    from types import SimpleNamespace

    from onnx.utils import materialize_cache_shapes

    infos = [
        SimpleNamespace(name="axis0", shape=["batch", 2]),
        SimpleNamespace(name="axis1", shape=[1, "batch_size", 3]),
        SimpleNamespace(name="axis2", shape=[2, 1, "batch", 4]),
    ]
    assert materialize_cache_shapes(
        {"axis0": [1, 2], "axis1": [1, 1, 3], "axis2": [2, 1, 1, 4]},
        infos,
        batch_size=4,
    ) == {
        "axis0": [4, 2],
        "axis1": [1, 4, 3],
        "axis2": [2, 1, 4, 4],
    }
    with pytest.raises(RuntimeError, match="unknown dynamic axes"):
        materialize_cache_shapes(
            {"bad": [1, 2, 2]},
            [SimpleNamespace(name="bad", shape=[1, "time", 2])],
            batch_size=2,
        )


pytestmark = [pytest.mark.unit]
