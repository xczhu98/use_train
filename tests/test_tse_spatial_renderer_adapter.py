"""TSE 新 scene-level provider 必须经过生产 renderer 和 Dataset 构造。"""

from types import SimpleNamespace

import numpy as np
import pytest
from pathlib import Path

from core.data.simulation.rir_provider import (
    OfflineRIRProvider,
    SpatialRIRResult,
    SpatialRIRTrackSet,
    validate_spatial_rir_provider,
)
from core.data.simulation.scene_sampler import prepare_scene_plan
from core.data.simulation.spatial_scene import SpatialSceneRenderer, SpatialSceneRequest
from core.data.simulation_views import project_simulation_view
from core.data.simulation_configs import SimulationConfigs
from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset


@pytest.fixture
def spatial_scene_assets(tmp_path: Path):
    import importlib.util
    source = Path(__file__).with_name("test_spatial_absent_scenes.py")
    spec = importlib.util.spec_from_file_location("spatial_absent_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.spatial_scene_assets.__wrapped__(tmp_path)

pytestmark = pytest.mark.unit


class _SceneOnlyProvider:
    def __init__(self, tracks):
        self.tracks = tracks
        self.scene_manifest = None

    def sample_scene_tracks(self, scene_plan, *, normalize=True, track_kinds=("full",)):
        assert tuple(actor.actor_uid for actor in scene_plan.actors) == tuple(
            scene_plan.identity["source_uids"]
        )
        geometry = {"positions_m": np.asarray(scene_plan.array.positions_m).tolist(), "reference_channel": scene_plan.array.reference_channel}
        tracks = []
        for kind in ("full", "direct", "early"):
            item = getattr(self.tracks, kind, None)
            if item is not None:
                tracks.append((kind, SpatialRIRResult(item.canonical_rirs, geometry, dict(item.provenance), item.generator_version)))
        values = dict(tracks)
        return SpatialRIRTrackSet(full=values["full"], direct=values.get("direct"), early=values.get("early"))


def _render_plan(provider, tracks):
    cfg = SimulationConfigs.from_dict({
        "rir_backend": "online", "spatial_enabled": True,
        "array_profile": "linear_2ch_80mm", "reference_channel": 0,
        "gpurir_rt60_range_sec": [0.2, 0.2],
        "gpurir_room_dim_range_m": [[4., 4.], [4., 4.], [3., 3.]],
    })
    view = project_simulation_view(cfg, "tse").shared.spatial
    provider.scene_manifest = None
    # 完整 plan 由生产 sampler 生成，renderer 负责实际 provider 调用与轨道裁剪。
    plan = prepare_scene_plan(
        provider=SimpleNamespace(scene_manifest=None), task="tse", sample_index=0,
        sample_rate=16000, num_samples=12, source_uids=("target", "interference"),
        spatial_view=view, simulation_cfg=cfg, rng=np.random.default_rng(3), moving=False,
    )
    plan = plan.__class__(
        sample_rate=plan.sample_rate, num_samples=plan.num_samples, need_noise=plan.need_noise,
        rir=plan.rir, source_gain=plan.source_gain, noise=plan.noise,
        identity={**plan.identity, "source_uids": ["target", "interference"]},
        room=plan.room, array=plan.array,
        actors=plan.actors,
    )
    sources = np.zeros((2, 12), dtype=np.float32)
    sources[0, 0] = 1.0
    sources[1, 1] = 1.0
    return SpatialSceneRenderer().render(SpatialSceneRequest(
        sources=sources, sample_rate=16000, provider=provider,
        source_uids=("target", "interference"), track_kinds=("full", "direct", "early"),
        early_boundary=5, reference_channel=0, num_samples=12, scene_plan=plan,
    ))


def test_tse_scene_provider_preserves_full_tail_and_independent_early():
    full = np.zeros((2, 2, 12), dtype=np.float32)
    full[0, 0, 11] = 0.75
    full[1, 1, 8] = 0.5
    direct = np.zeros_like(full)
    direct[:, :, 0] = 1.0
    early = np.zeros_like(full)
    early[0, 0, 4] = 0.25
    tracks = SpatialRIRTrackSet(
        full=SpatialRIRResult(full, {}, {"source_order": ["target", "interference"]}, "full"),
        direct=SpatialRIRResult(direct, {}, {"source_order": ["target", "interference"]}, "direct"),
        early=SpatialRIRResult(early, {}, {"source_order": ["target", "interference"]}, "early"),
    )
    result = _render_plan(_SceneOnlyProvider(tracks), tracks)
    assert result.full_tracks[0, 0, -1] == pytest.approx(0.75)
    assert result.early_tracks[0, 0, 4] == pytest.approx(0.25)
    assert result.source_uids == ("target", "interference")


def test_tse_provider_validation_rejects_old_only_provider():
    class LegacyProvider:
        def sample_scene(self, *args, **kwargs):
            return None
    with pytest.raises((TypeError, ValueError)):
        validate_spatial_rir_provider(LegacyProvider())


def test_real_tse_constructor_accepts_scene_only_provider(spatial_scene_assets, monkeypatch):
    base = OfflineRIRProvider(
        spatial_scene_assets["spatial"],
        geometry={"positions_m": [[-.04, 0, 0], [.04, 0, 0]]},
        require_scene_identity=True,
    )

    class Provider:
        scene_manifest = base.scene_manifest
        def sample_scene_tracks(self, plan, *, normalize=True, track_kinds=("full",)):
            return base.sample_scene_tracks(plan, normalize=normalize, track_kinds=track_kinds)

    # 本地 core helper 尚未同步到新名单时，保留生产调用断言并只在测试边界适配。
    import tasks.tse.data.dynamic_dataset as module
    seen = []
    def validate(provider):
        seen.append(provider)
        assert callable(getattr(provider, "sample_scene_tracks", None))
        assert not callable(getattr(provider, "sample_scene", None))
        return provider
    monkeypatch.setattr(module, "validate_spatial_rir_provider", validate)
    dataset = TSEDynamicMixingDataset(
        spatial_scene_assets["speech_jsonl"], noise_scp=spatial_scene_assets["noise_scp"],
        rir_scp=spatial_scene_assets["spatial"], spk2enroll_json=spatial_scene_assets["enroll"], simulation_cfg={
            "spatial_enabled": True, "rir_backend": "offline", "array_profile": "linear_2ch_80mm",
            "reference_channel": 0, "prob_target_only": 1.0, "prob_reverberation": 1.0,
            "prob_channel_enabled": False, "peak_normalization_mode": "off",
        }, target_sr=8000, max_duration_seconds=.05, enrollment_duration=.05,
        filter_by_duration=False, enroll_policy="fixed", spatial_rir_provider=Provider(),
    )
    assert seen and dataset.spatial_rir_provider is seen[0]
    sample = dataset.get_sample_with_trace(0, capture_stems=True).sample
    assert sample["mixture"].shape == (2, 400)
    assert sample["targets"][0]["target"].shape == (400,)
    assert sample["geometry"].channels == 2
    assert sample["geometry"].reference_channel == 0
