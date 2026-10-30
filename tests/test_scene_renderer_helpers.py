"""scene renderer 内部 background 编排 helper 的合同测试。"""

from types import SimpleNamespace

import numpy as np
import pytest

import core.data.simulation.scene_renderer as scene_renderer
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.data.simulation.scene_context import SceneRenderContext
from core.data.waveform_bundle import WaveformBundle


pytestmark = pytest.mark.unit


class _UniformRecorder:
    def __init__(self, value=3.0):
        self.value = value
        self.calls = []

    def uniform(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.value


def _dataset():
    return SimpleNamespace(sim_cfg=SimpleNamespace(snr_low=-5.0, snr_high=5.0))


def test_mono_background_helper_preserves_source_branch_snr_draw(monkeypatch):
    captured = {}

    def fake_mix_background(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "background"

    monkeypatch.setattr(scene_renderer, "mix_background", fake_mix_background)
    rng = _UniformRecorder()
    background, spec = scene_renderer._render_mono_background(
        _dataset(),
        np.ones(4, dtype=np.float32),
        np.full(4, 2.0, dtype=np.float32),
        8000,
        4,
        rng,
        False,
        None,
        has_sources=True,
    )

    assert background == "background"
    assert spec.policy.value == "additive_mono"
    assert len(rng.calls) == 1
    assert captured["args"][1].shape == (1, 4)
    assert captured["args"][2].shape == (1, 4)
    assert captured["args"][7] is False
    assert captured["args"][8] == pytest.approx(3.0)
    assert captured["kwargs"]["background_spec"] is spec


def test_mono_background_helper_rejects_empty_scene_without_noise():
    rng = _UniformRecorder()
    with pytest.raises(ValueError, match="scene must contain speech or background"):
        scene_renderer._render_mono_background(
            _dataset(),
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            8000,
            4,
            rng,
            False,
            None,
            has_sources=False,
        )
    assert not rng.calls


def test_prepare_speech_scene_uses_typed_context_and_places_sources_once(monkeypatch):
    cfg = SimpleNamespace(
        spatial_enabled=False,
        rir_backend="offline",
        prob_point_source_background=0.0,
    )
    dataset = SimpleNamespace(sim_cfg=cfg)
    actor = AcousticActor(
        uid="target-0",
        role=AcousticRole.TARGET_SPEECH,
        rir_policy=RIRPolicy.MONO_SOURCE,
        waveform=np.arange(4, dtype=np.float32),
    )
    geometry = np.zeros((1, 3), dtype=np.float32)
    placed = np.full(6, 0.25, dtype=np.float32)
    timeline = ("timeline",)
    rng = object()
    captured = {}

    monkeypatch.setattr(
        scene_renderer,
        "_resolve_observation_geometry",
        lambda *_args: ("profile", "custom", geometry),
    )

    def place(sources, temporal, actual_rng):
        captured["sources"] = sources
        captured["temporal"] = temporal
        captured["rng"] = actual_rng
        return (placed,), timeline

    monkeypatch.setattr(scene_renderer, "place_scene_sources", place)
    prepared = scene_renderer._prepare_speech_scene(
        dataset,
        (actor,),
        rng,
        sir_mode="none",
        need_noise=False,
        num_samples=None,
        spatial_rirs=None,
        pre_rendered_spatial=None,
        sources_preplaced=False,
        timeline=None,
        background_spec=None,
        observation_geometry=None,
    )

    assert isinstance(prepared.context, SceneRenderContext)
    assert len(prepared.sources) == 1
    assert prepared.sources[0] is placed
    assert prepared.source_uids == ("target-0",)
    assert prepared.actor_facts == [actor.to_dict()]
    assert prepared.timeline is timeline
    assert prepared.length == placed.size
    assert prepared.microphone_positions is geometry
    assert captured["sources"][0] is actor.waveform
    assert captured["temporal"] is cfg
    assert captured["rng"] is rng


def test_speech_scene_post_mix_request_projects_roles_in_bundle_order():
    tracks = {
        "mixture": np.ones(4, dtype=np.float32),
        "background": np.full(4, 0.5, dtype=np.float32),
        "source_full_rir/0": np.full(4, 0.25, dtype=np.float32),
        "target/0": np.full(4, 0.75, dtype=np.float32),
    }
    scene = scene_renderer.SpeechScene(
        WaveformBundle(tracks), {"scene": "test"}, used_sidechain_wind=False
    )
    dataset = SimpleNamespace(
        sim_cfg=SimpleNamespace(),
        _simulation_view=SimpleNamespace(),
        _apply_channel_degradation_result=lambda *args: None,
    )

    request = scene_renderer._build_speech_scene_post_mix_request(
        dataset, scene, 8000, np.random.default_rng(0)
    )

    assert request.observation_roles == ("mixture", "background")
    assert request.protected_roles == ("target/0",)
    assert request.amplitude_roles == ("source_full_rir/0",)
    assert request.role_contract.observation_roles == request.observation_roles
    assert request.retain_stage_states is False


def test_speech_scene_post_mix_request_marks_sidechain_as_non_decomposable():
    scene = scene_renderer.SpeechScene(
        WaveformBundle(
            {
                "mixture": np.ones(4, dtype=np.float32),
                "target/0": np.full(4, 0.75, dtype=np.float32),
            }
        ),
        {"scene": "test"},
        used_sidechain_wind=True,
    )
    dataset = SimpleNamespace(
        sim_cfg=SimpleNamespace(),
        _simulation_view=SimpleNamespace(),
        _apply_channel_degradation_result=lambda *args: None,
    )

    request = scene_renderer._build_speech_scene_post_mix_request(
        dataset, scene, 8000, np.random.default_rng(0)
    )

    assert request.non_decomposable_contract is not None
    assert request.non_decomposable_contract.transform_id == "wind_noise_sidechain_v1"
    assert request.non_decomposable_contract.control_roles == ()
    assert request.non_decomposable_contract.mutable_stage == "C-2"


def test_finished_speech_scene_facts_preserve_post_mix_contract_fields():
    scene = scene_renderer.SpeechScene(
        WaveformBundle({"mixture": np.ones(2, dtype=np.float32)}),
        {"scene": "test"},
        used_sidechain_wind=False,
    )
    post_mix = SimpleNamespace(
        trace=SimpleNamespace(stage_order=("pre-finite", "D-normalize", "post-finite")),
        facts={
            "layer_d_facts": {"normalize": {"scale": 1.0}},
            "layer_e": {"applied_ops": []},
            "layer_f": {"applied_ops": []},
            "final_peak_guard": {"scale": 1.0},
            "post_mix_marker": "kept",
            "role_contract": {"observation_roles": ["mixture"]},
            "decomposable_final": True,
        },
    )

    facts = scene_renderer._build_finished_speech_scene_facts(scene, post_mix)

    assert facts["scene"] == "test"
    assert facts["post_mix_stage_order"] == ["pre-finite", "D-normalize", "post-finite"]
    assert facts["post_mix_facts"]["post_mix_marker"] == "kept"
    assert facts["role_contract"] == {"observation_roles": ["mixture"]}
    assert facts["decomposable_final"] is True
