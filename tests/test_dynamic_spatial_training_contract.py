"""CPU 合同链：真实 SE 数据读取/渲染 → collate → SpatialNet → loss/backward。

只替换在线 RIR 生成边界，不替换 dataset、卷积、STFT、模型或 loss。
此测试不覆盖 gpuRIR 数值正确性、IPC 或完整 Trainer 生命周期。
"""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from core.data.collate import collate_fn
from core.data.simulation import (
    MovingSpatialRIRTrackSet,
    SpatialRIRProvider,
    SpatialRIRResult,
    SpatialRIRTrackSet,
)
from core.data.simulation.rir_provider import MovingRIRCapabilityError, OnlineRIRProvider
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.data.simulation.trajectory import TrajectoryTimebase
from core.losses.time_domain import L1WavLoss
from core.models.io_contract import validate_spatial_model_geometry
from core.stft import build_stft_istft
from tasks.se.data.batch import SEBatch
from tasks.se.data.dynamic_dataset import DynamicMixingDataset
from tasks.se.models import build_model
from tasks.se.trainer.se_trainer import SETrainer
from test_aec_spatial_dynamic_branches import _spatial_aec_dataset

pytestmark = pytest.mark.unit


class _MovingProvider:
    """两段不同脉冲响应，保证测试确实经过随时间变化的卷积。"""

    def __init__(self):
        self.calls = []

    def sample_spatial_scene(self, sample_rate, source_count, rng, **kwargs):
        self.calls.append((sample_rate, source_count, kwargs))
        return SpatialRIRResult(
            np.array([[[1.0, 0.0], [0.0, 0.5]]], dtype=np.float32),
            {"positions_m": [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]},
            {"source_order": list(kwargs["source_uids"])}, "cpu-fixture",
        )

    def sample_scene_tracks(self, scene_plan, *, normalize=True, track_kinds=("full",)):
        del normalize, track_kinds
        source_uids = tuple(actor.actor_uid for actor in scene_plan.actors)
        base = self.sample_spatial_scene(
            scene_plan.sample_rate,
            len(source_uids),
            None,
            source_uids=source_uids,
        )
        provenance = {
            **base.provenance,
            "source_order": list(source_uids),
            "actor_order": list(source_uids),
            "scene_uid": scene_plan.identity["scene_uid"],
            "room": scene_plan.to_dict()["room"],
        }
        geometry = {"positions_m": np.asarray(scene_plan.array.positions_m).tolist()}
        full = SpatialRIRResult(
            base.rirs, geometry, provenance, base.generator_version
        )
        early = SpatialRIRResult(
            base.rirs * np.float32(0.8), geometry, provenance, base.generator_version
        )
        return SpatialRIRTrackSet(full=full, early=early)

    def sample_moving_scene_tracks(self, request, *, normalize=True):
        self.calls.append((request.sample_rate, request.actor_count, {"source_uids": request.source_uids}))
        timebase = request.timebase
        positions = request.array["positions_m"]
        trajectory = np.concatenate([t.positions_m for t in request.actor_trajectories.values()], axis=1)
        full = np.zeros((timebase.frame_count, 1, 2, 4), dtype=np.float32)
        full[:, 0, 0, 0] = 1.0
        full[:, 0, 1, 1] = np.linspace(.25, .75, timebase.frame_count)
        direct = full.copy()
        full[..., 3] = 0.05
        return MovingSpatialRIRTrackSet(
            full=full, direct=direct,
            plan={
                "room_dim": list(request.room["room_dim_m"]),
                "sample_rate": request.sample_rate, "num_samples": 4,
                "mic_loc": positions, "source_loc": trajectory[0].tolist(),
                "trajectory": trajectory.tolist(),
                "trajectory_timebase": timebase.to_dict(),
                "source_ids": list(request.source_uids),
                "scene_uid": request.scene_uid,
                "room": dict(request.room),
                "geometry": {"positions_m": positions},
            },
            track_facts={"group_uid": "cpu-moving-scene"},
        )


@pytest.mark.parametrize("moving", [False, True], ids=["static", "moving"])
@pytest.mark.parametrize(
    "model_type,model_extra",
    (
        ("SpatialNet_SE", {}),
        ("OnlineSpatialNet_SE", {"attention": "mhsa(16)"}),
        ("OnlineSpatialNet_SE", {"attention": "ret(2,not_share_qk)", "decay": [4, 5]}),
        ("OnlineSpatialNet_SE", {"attention": "mamba(4,4)"}),
    ),
    ids=["offline", "online-mhsa", "online-ret", "online-mamba"],
)
def test_se_dynamic_spatial_sample_to_native_model_loss(
    tmp_path, monkeypatch, moving, model_type, model_extra
):
    provider = _MovingProvider()
    # 在正式配置边界启用 online，替换服务构造以保持 CPU-only。
    monkeypatch.setattr(OnlineRIRProvider, "from_simulation", lambda *_a, **_kw: provider)
    waveform = (0.2 * np.sin(np.linspace(0, 5, 128))).astype(np.float32)
    for name in ("speech", "noise"):
        sf.write(tmp_path / f"{name}.wav", waveform, 8000, subtype="FLOAT")
        (tmp_path / f"{name}.scp").write_text(
            f"{name} 8000 128 {name}.wav\n", encoding="utf-8"
        )
    dataset = DynamicMixingDataset(
        str(tmp_path / "speech.scp"), str(tmp_path / "noise.scp"), None,
        max_duration_seconds=128 / 8000, filter_by_duration=False,
        strict_load=True, simulation_cfg={
            "spatial_enabled": True, "array_profile": "linear_2ch_80mm",
            "reference_channel": 0, "rir_backend": "online",
            "prob_moving_rir": float(moving), "prob_reverberation": 1.0,
            "prob_target_only": 1.0, "prob_target_noise": 0.0,
            "prob_noise_only": 0.0, "prob_channel_enabled": False,
            "peak_normalization_mode": "off", "noisy_post_mix_degradations_enabled": False,
        },
    )
    traced = dataset.get_sample_with_trace(0)
    sample = traced.sample
    batch = collate_fn([sample], allow_spatial_noisy=True)
    assert isinstance(batch, SEBatch)
    assert batch.noisy.shape == (1, 2, 128)
    assert batch.clean.shape == (1, 128)
    assert batch.lengths.tolist() == [128]
    assert not torch.equal(batch.noisy[:, 0], batch.noisy[:, 1])
    if moving:
        assert len(provider.calls) == 1
        assert provider.calls[0][1] == 1
        assert provider.calls[0][2]["source_uids"] == ("speech",)
        # 两段 RIR 的非参考通道增益不同，不能退回静态或广播路径。
        first = batch.noisy[0, 1, 8:56].norm() / batch.noisy[0, 0, 8:56].norm()
        second = batch.noisy[0, 1, 72:120].norm() / batch.noisy[0, 0, 72:120].norm()
        assert second > first * 1.5
        facts = traced.trace.metadata_dict()["sources"]["scene"]["scene"]["spatial"]
        assert facts["rir_shape"] is None
        provenance = facts["pre_rendered_provenance"]
        assert provenance["mode"] == "moving"
        assert provenance["source_order"] == ["speech"]
        assert provenance["trajectory_timebase"]["num_samples"] == 128
        assert provenance["trajectory_timebase"]["hop_samples"] == 32
        assert provenance["track_identity"]["track_set"]["full"] != provenance["track_identity"]["track_set"]["direct"]

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = build_model({"type": model_type, "args": {
            "n_fft": 64, "n_channel": 2, "ref_channel": 0,
            "num_layers": 1, "dim_hidden": 8, "dim_ffn": 16,
            "dim_squeeze": 4, "num_heads": 2, "conv_groups": (1, 1),
            **model_extra,
        }})
    validate_spatial_model_geometry(model, dataset)
    # 跳过 Trainer 的日志、设备和分布式生命周期，执行真实 forward/loss 方法。
    trainer = object.__new__(SETrainer)
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.stft_module = build_stft_istft({
        "n_fft": 64, "win_length": 64, "hop_length": 32,
        "window": "hamming", "center": True,
    })
    trainer.execution_plan = SimpleNamespace(input_domain="spectrum")
    trainer._model_accepts_fs = True
    trainer._use_lsnr_loss = False
    trainer._strict_checks = True
    trainer.loss_fn = L1WavLoss()
    received = []
    handle = model.register_forward_pre_hook(lambda _model, args: received.append(args[0].detach().clone()))
    try:
        loss, finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, 8000)
    finally:
        handle.remove()
    assert received[0].shape == (1, 2, 33, 5, 2)
    expected = trainer.stft_module(batch.noisy.reshape(2, 128)).reshape(1, 2, 33, 5, 2)
    torch.testing.assert_close(received[0], expected)
    assert finite and loss.ndim == 0 and loss.item() > 0
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) > 0 for g in grads)
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_aec_real_dynamic_sample_preserves_capability_failure(tmp_path):
    dataset = _spatial_aec_dataset(tmp_path)
    dataset.sim_cfg.prob_moving_rir = 1.0
    dataset.sim_cfg.rir_backend = "online"
    calls = []

    class UnsupportedProvider(SpatialRIRProvider):
        def sample_moving_scene_tracks(self, request, *, normalize=True):
            calls.append(request)
            return super().sample_moving_scene_tracks(request, normalize=normalize)

    dataset._spatial_rir_provider = UnsupportedProvider()
    dataset.spatial_rir_provider = dataset._spatial_rir_provider
    with pytest.raises(RuntimeError) as error:
        dataset.get_sample_with_trace(0, force_branch="prob_target_noise_echo")
    assert isinstance(error.value.__cause__, MovingRIRCapabilityError)
    assert len(calls) == 1
    assert calls[0].source_uids == ("speech", "__echo__")
    assert calls[0].microphone_count == 2


def test_pre_rendered_scene_rejects_wrong_caller_source_identity():
    from core.data.simulation.scene_renderer import render_speech_scene
    from core.data.simulation_configs import SimulationConfigs
    from core.data.spatial import render_spatial_scene

    cfg = SimulationConfigs.from_dict({
        "spatial_enabled": True, "array_profile": "linear_2ch_80mm",
        "reference_channel": 0, "prob_reverberation": 1.0,
    })
    source = np.ones((1, 128), dtype=np.float32)
    rendered = render_spatial_scene(
        source, np.ones((1, 2, 1), dtype=np.float32),
        early_boundary=1, num_samples=128, provenance={"source_order": ["speaker-a"]},
    )
    with pytest.raises(ValueError, match="source_order does not match"):
        render_speech_scene(
            SimpleNamespace(sim_cfg=cfg),
            (
                AcousticActor(
                    uid="speaker-b",
                    role=AcousticRole.TARGET_SPEECH,
                    rir_policy=RIRPolicy.SPATIAL_SOURCE,
                    waveform=source[0],
                ),
            ),
            8000,
            np.random.default_rng(0),
            sir_range=None, sir_mode="none", need_noise=False,
            pre_rendered_spatial=rendered, sources_preplaced=True,
        )
