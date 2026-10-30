"""AEC post-mix Adapter 的纯 seam 合同测试。"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import tasks.aec.data.post_mix_adapter as adapter_module
from tasks.aec.data.dynamic_dataset import _build_aec_delivery_bundle
from tasks.aec.data.post_mix_adapter import (
    ECHO,
    FAREND,
    MIC,
    NEAREND,
    NOISE,
    AECPostMixContext,
    MicExtraNoiseResult,
    TaskPostMixAdapter,
    apply_scene_time_transform,
    project_training_tuple,
)
from core.data.dynamic_dataset_base import LayerEOperationResult, LayerEResult
from core.data.post_mix_pipeline import (
    POST_MIX_STAGE_ORDER,
    PostMixContext,
    run_post_mix_orchestration,
)
from core.data.simulation_configs import SimulationConfigs
from core.data.time_transform import TileOp, plan_time_transform
from core.data.waveform_bundle import WaveformBundle

pytestmark = pytest.mark.unit


def test_aec_delivery_bundle_projects_layer_d_exports_only_when_captured():
    roles = {
        NEAREND: np.array([1.0, 2.0], dtype=np.float32),
        MIC: np.array([3.0, 4.0], dtype=np.float32),
        FAREND: np.array([5.0, 6.0], dtype=np.float32),
        ECHO: np.array([7.0, 8.0], dtype=np.float32),
        NOISE: np.array([9.0, 10.0], dtype=np.float32),
    }
    bundle = WaveformBundle(roles, metadata={"branch": "prob_target_noise"})
    d_state = SimpleNamespace(
        waveforms={
            MIC: np.array([11.0, 12.0], dtype=np.float32),
            NEAREND: np.array([13.0, 14.0], dtype=np.float32),
            NOISE: np.array([15.0, 16.0], dtype=np.float32),
            ECHO: np.array([17.0, 18.0], dtype=np.float32),
            FAREND: np.array([19.0, 20.0], dtype=np.float32),
        }
    )
    post_mix_result = SimpleNamespace(
        bundle=bundle,
        trace=SimpleNamespace(
            results=(SimpleNamespace(stage="D-guard", state=d_state),)
        ),
    )

    plain = _build_aec_delivery_bundle(post_mix_result, capture_stems=False)
    assert set(plain.tracks) == set(roles)
    assert dict(plain.metadata) == {"branch": "prob_target_noise"}

    exported = _build_aec_delivery_bundle(post_mix_result, capture_stems=True)
    assert set(exported.tracks) == set(roles) | {
        "export/layer_d_noisy",
        "export/layer_d_clean",
        "export/layer_d_noise",
        "export/layer_d_echo",
        "export/farend_ref",
        "export/final_noise",
    }
    np.testing.assert_array_equal(
        exported.require("export/layer_d_noisy"), d_state.waveforms[MIC]
    )
    np.testing.assert_array_equal(
        exported.require("export/final_noise"), bundle.require(NOISE)
    )


class _FakeDataset:
    target_sr = None

    def __init__(
        self,
        *,
        layer_e_changed: bool = True,
        layer_e_clipping: bool = False,
        extra_noise: bool = True,
        d_normalize_mode: str = "off",
    ):
        self.sim_cfg = SimulationConfigs.from_dict(
            {
                "peak_normalization_mode": d_normalize_mode,
                "peak_normalization_scale_range": [0.5, 0.5],
            },
            strict=True,
        )
        self.calls: list[str] = []
        self.layer_e_changed = layer_e_changed
        self.layer_e_clipping = layer_e_clipping
        self.extra_noise = extra_noise
        self.decomposable_seen_before_extra = None
        self.exported: dict[str, np.ndarray] = {}
        self._last_mic_extra_noise_policy = "stale-invalid-policy"

    def _guard_finite(self, mic, nearend, _uid, label):
        self.calls.append(label)
        return np.asarray(mic, dtype=np.float32), np.asarray(nearend, dtype=np.float32)

    def _export_layer_d_stems(self, **stems):
        self.calls.append("export-D")
        self.exported = {key: np.asarray(value).copy() for key, value in stems.items()}

    def _apply_channel_degradation_result(self, mic, *_args):
        self.calls.append("E")
        waveform = (
            np.asarray(mic, dtype=np.float32) + np.float32(1.0)
            if self.layer_e_changed
            else np.asarray(mic, dtype=np.float32)
        )
        operation = (
            LayerEOperationResult("clipping", True, self.layer_e_changed, None),
        ) if self.layer_e_clipping else ()
        return LayerEResult(
            waveform=waveform,
            enabled=True,
            num_augmentations_drawn=len(operation),
            selected=("clipping",) if operation else (),
            operations=operation,
            observation_changed=self.layer_e_changed,
        )

    def _maybe_mic_extra_noise(self, mic, _sr, _index, _clean):
        self.calls.append("task-post-EF")
        self.decomposable_seen_before_extra = self.adapter.decomposable
        if not self.extra_noise:
            return mic, None
        extra = np.full_like(mic, 0.25, dtype=np.float32)
        return (mic + extra).astype(np.float32), extra

    def _maybe_mic_extra_noise_result(self, mic, sr, index, clean):
        mic_out, extra_noise = self._maybe_mic_extra_noise(mic, sr, index, clean)
        return MicExtraNoiseResult(mic_out, extra_noise, None)

def _run(
    monkeypatch,
    *,
    layer_e_changed=True,
    layer_e_clipping=False,
    layer_f_changes=True,
    extra_noise=True,
    d_normalize_mode="off",
):
    dataset = _FakeDataset(
        layer_e_changed=layer_e_changed,
        layer_e_clipping=layer_e_clipping,
        extra_noise=extra_noise,
        d_normalize_mode=d_normalize_mode,
    )
    rng = np.random.default_rng(23)

    def _layer_f(mic, *_args, **kwargs):
        dataset.calls.append("F")
        assert kwargs["skip_clip"] is layer_e_clipping
        if layer_e_clipping:
            assert _args[1].prob_clip == 0.0
        if layer_f_changes:
            kwargs["applied_ops_out"].append("post_eq" if layer_e_clipping else "test-op")
            if not layer_e_clipping:
                kwargs["clipping_facts_out"].update(
                    {"variant": "fixed_rail", "applied": True}
                )
            return (np.asarray(mic) * np.float32(2.0)).astype(np.float32)
        return np.asarray(mic, dtype=np.float32)

    monkeypatch.setattr(adapter_module, "apply_post_augmentations", _layer_f)
    context = AECPostMixContext.from_aec_disk(
        branch="prob_target_echo",
        task="aec",
        aec_disk={
            "prob_target_echo_disk": {
                "farend": np.array([2.0, -2.0], dtype=np.float32),
                "echo": np.array([0.5, -0.5], dtype=np.float32),
                "c3_meta": {"delay_ms": 10.0},
            }
        },
        used_sidechain_wind=False,
        joint_target_rir_mismatch=False,
        sidechain_wind_mismatch_reason=None,
        index=7,
    )
    adapter = TaskPostMixAdapter(
        dataset=dataset,
        rng=rng,
        sample_rate=16_000,
        actual_len=2,
        speech_uid="speaker-a",
        context=context,
        metadata={"branch": "prob_target_echo"},
    )
    dataset.adapter = adapter
    bundle = adapter.prepare_initial_bundle(
        nearend=np.array([1.0, -1.0], dtype=np.float32),
        mic=np.array([2.0, -2.0], dtype=np.float32),
        farend=np.array([2.0, -2.0], dtype=np.float32),
        echo=np.array([0.5, -0.5], dtype=np.float32),
        noise=np.array([0.25, -0.25], dtype=np.float32),
    )
    state_before_d_normalize = rng.bit_generator.state.copy()
    runtime = run_post_mix_orchestration(
        bundle,
        adapter,
        context=PostMixContext({"branch": "prob_target_echo", "sample_rate": 16_000}),
    )
    return dataset, rng, state_before_d_normalize, adapter, runtime.trace


def test_fixed_pipeline_preserves_aec_roles_and_explicit_facts(monkeypatch):
    dataset, _, _, adapter, trace = _run(monkeypatch)
    result = adapter.build_result(trace)

    assert trace.stage_order == POST_MIX_STAGE_ORDER
    assert trace.stage_order == (
        "pre-finite",
        "D-normalize",
        "D-guard",
        "E",
        "F",
        "task-post-EF",
        "final-guard",
        "post-finite",
    )
    assert trace.results[1].stage == "D-normalize"
    assert trace.results[1].applied is False
    assert trace.results[2].stage == "D-guard"
    assert trace.results[2].applied is True
    assert dataset.calls == [
        "mix pipeline",
        "export-D",
        "E",
        "F",
        "task-post-EF",
        "after augmentation",
    ]
    assert set(result.bundle.tracks) == {NEAREND, MIC, FAREND, ECHO, NOISE}
    assert result.role_contract.observation_roles == (MIC,)
    assert result.role_contract.protected_roles == (FAREND, ECHO)
    assert result.role_contract.mutable_stages[FAREND] == ("D-normalize", "D-guard")
    assert result.role_contract.mutable_stages[NOISE] == (
        "D-normalize",
        "D-guard",
        "task-post-EF",
    )
    assert result.decomposable is False
    assert result.decomposable_reason_codes == (
        "layer_e_nonlinear",
        "layer_f_nonlinear",
        "task_post_ef_mic_extra_noise",
        "final_guard_observation_only_scale",
    )
    assert dataset.decomposable_seen_before_extra is False
    assert result.layer_f_applied_ops == ("test-op",)
    assert result.layer_f_clipping_facts == {
        "variant": "fixed_rail",
        "applied": True,
        "event": True,
        "actual_changed": True,
        "skipped": False,
        "skip_reason": None,
    }
    threshold = float(np.float32(0.99))
    assert result.layer_d_facts == {
        "stage_order": ("D-normalize", "D-guard"),
        "normalize": {
            "mode": "off",
            "target_peak": None,
            "scale": 1.0,
            "preserves_decomposition": True,
        },
        "guard": {
            "mic_related": {
                "roles": [MIC, NEAREND, ECHO, NOISE],
                "threshold": threshold,
                "peak": 2.0,
                "scale": pytest.approx(threshold / 2.0),
                "applied": True,
            },
            "farend": {
                "roles": [FAREND],
                "threshold": threshold,
                "peak": 2.0,
                "scale": pytest.approx(threshold / 2.0),
                "applied": True,
            },
        },
    }
    assert not hasattr(result, "layer_d_normalize_facts")
    assert not hasattr(result, "farend_guard_facts")
    assert not hasattr(result, "peak_normalization")
    assert result.final_guard_facts["roles"] == [MIC]
    assert result.final_guard_facts["threshold"] == threshold
    assert result.final_guard_facts["peak"] > threshold
    assert result.final_guard_facts["scale"] < 1.0
    assert result.final_guard_facts["applied"] is True
    assert result.aec_disk["prob_target_echo_disk"]["c3_meta"] == {"delay_ms": 10.0}
    assert "c3_meta" not in result.bundle.metadata


def test_explicit_mic_extra_result_ignores_stale_policy_sidecar(monkeypatch):
    dataset, _, _, adapter, trace = _run(monkeypatch)
    result = adapter.build_result(trace)

    assert result.mic_extra_noise_policy is None
    assert dataset._last_mic_extra_noise_policy == "stale-invalid-policy"


def test_final_guard_noop_preserves_decomposability(monkeypatch):
    dataset, _, _, adapter, trace = _run(
        monkeypatch,
        layer_e_changed=False,
        layer_f_changes=False,
        extra_noise=False,
    )
    result = adapter.build_result(trace)

    assert result.final_guard_facts["applied"] is False
    assert result.decomposable_reason_codes == ()
    assert result.decomposable is True
    assert result.layer_f_applied_ops == ()
    assert result.layer_f_clipping_facts == {}
    assert dataset.decomposable_seen_before_extra is True


def test_layer_e_applied_clipping_skips_only_layer_f_clipping(monkeypatch):
    _, _, _, adapter, trace = _run(
        monkeypatch,
        layer_e_clipping=True,
        extra_noise=False,
    )
    result = adapter.build_result(trace)

    assert result.layer_e.operation_applied("clipping") is True
    assert result.layer_f_applied_ops == ("post_eq",)
    assert result.layer_f_clipping_facts == {
        "event": False,
        "applied": False,
        "actual_changed": False,
        "skipped": True,
        "skip_reason": "layer_e_clipping_applied",
    }
    assert result.decomposable_reason_codes == (
        "layer_e_nonlinear",
        "layer_f_nonlinear",
        "final_guard_observation_only_scale",
    )


def test_d_normalize_is_strict_noop_and_consumes_no_rng(monkeypatch):
    _, _, state_before_d_normalize, adapter, _ = _run(monkeypatch)
    rng = np.random.default_rng(31)
    adapter._rng = rng
    state = adapter.state_from_bundle(
        WaveformBundle(
            {
                NEAREND: np.ones(2, dtype=np.float32),
                MIC: np.ones(2, dtype=np.float32),
                FAREND: np.ones(2, dtype=np.float32),
                ECHO: np.ones(2, dtype=np.float32),
                NOISE: np.ones(2, dtype=np.float32),
            }
        )
    )
    cursor = rng.bit_generator.state.copy()
    stage = adapter.apply_stage("D-normalize", state)
    assert stage.stage == "D-normalize"
    assert stage.applied is False
    assert rng.bit_generator.state == cursor
    assert state_before_d_normalize != cursor


def test_aec_d_guard_threshold_is_adapter_noop_with_clear_facts(monkeypatch):
    _, _, _, adapter, _ = _run(monkeypatch)
    rng = np.random.default_rng(53)
    adapter._rng = rng
    cursor = rng.bit_generator.state.copy()
    threshold = np.float32(0.99)
    state = adapter.state_from_bundle(
        WaveformBundle(
            {
                NEAREND: np.array([threshold, -threshold], dtype=np.float32),
                MIC: np.array([threshold, -threshold], dtype=np.float32),
                FAREND: np.array([threshold, -threshold], dtype=np.float32),
                ECHO: np.array([0.5, -0.5], dtype=np.float32),
                NOISE: np.array([0.25, -0.25], dtype=np.float32),
            }
        )
    )

    stage = adapter.apply_stage("D-guard", state)

    assert stage.stage == "D-guard"
    assert stage.applied is False
    assert stage.state is state
    assert rng.bit_generator.state == cursor
    assert adapter._layer_d_facts == {
        "stage_order": ("D-normalize", "D-guard"),
        "normalize": {
            "mode": "off",
            "target_peak": None,
            "scale": 1.0,
            "preserves_decomposition": True,
        },
        "guard": {
            "mic_related": {
                "roles": [MIC, NEAREND, ECHO, NOISE],
                "threshold": float(threshold),
                "peak": float(threshold),
                "scale": 1.0,
                "applied": False,
            },
            "farend": {
                "roles": [FAREND],
                "threshold": float(threshold),
                "peak": float(threshold),
                "scale": 1.0,
                "applied": False,
            },
        },
    }


def test_joint_d_normalize_uses_mic_peak_and_scales_all_aec_roles(monkeypatch):
    _, rng, _, adapter, _ = _run(monkeypatch, d_normalize_mode="joint")
    rng = np.random.default_rng(41)
    adapter._rng = rng
    state = adapter.state_from_bundle(
        WaveformBundle(
            {
                NEAREND: np.array([1.0, -1.0], dtype=np.float32),
                MIC: np.array([2.0, -2.0], dtype=np.float32),
                FAREND: np.array([100.0, -100.0], dtype=np.float32),
                ECHO: np.array([0.5, -0.5], dtype=np.float32),
                NOISE: np.array([0.25, -0.25], dtype=np.float32),
            }
        )
    )

    stage = adapter.apply_stage("D-normalize", state)

    assert stage.stage == "D-normalize"
    assert stage.applied is True
    assert adapter._layer_d_normalize_stage_facts == {
        "mode": "joint",
        "target_peak": 0.5,
        "scale": 0.25,
        "preserves_decomposition": True,
    }
    np.testing.assert_array_equal(stage.state.waveforms[MIC], [0.5, -0.5])
    np.testing.assert_array_equal(stage.state.waveforms[NEAREND], [0.25, -0.25])
    np.testing.assert_array_equal(stage.state.waveforms[NOISE], [0.0625, -0.0625])
    np.testing.assert_array_equal(stage.state.waveforms[ECHO], [0.125, -0.125])
    np.testing.assert_array_equal(stage.state.waveforms[FAREND], [25.0, -25.0])


@pytest.mark.parametrize("mode", ["separate", "single"])
def test_aec_d_normalize_rejects_non_follower_modes_without_rng_draw(monkeypatch, mode):
    _, _, _, adapter, _ = _run(monkeypatch)
    # 非 follower 模式在 SimulationConfigs 校验即被拒绝，视图边界只可能收到 off/joint。
    with pytest.raises(ValueError, match=r"peak_normalization_mode.*one of"):
        SimulationConfigs.from_dict(
            {
                "peak_normalization_mode": mode,
                "peak_normalization_scale_range": [0.5, 0.5],
            },
            strict=True,
        )
    # 这里直接改写已解析的冻结视图，验证 Layer D 自身兜底门禁同样 fail-closed
    # 且不消耗 RNG。
    adapter._simulation_view = replace(
        adapter._simulation_view,
        shared=replace(
            adapter._simulation_view.shared,
            peak=replace(
                adapter._simulation_view.shared.peak,
                peak_normalization_mode=mode,
            ),
        ),
    )
    rng = np.random.default_rng(43)
    adapter._rng = rng
    state = adapter.state_from_bundle(
        WaveformBundle({role: np.ones(2, dtype=np.float32) for role in (NEAREND, MIC, FAREND, ECHO, NOISE)})
    )
    cursor = rng.bit_generator.state.copy()

    with pytest.raises(
        ValueError,
        match=r"Layer D normalization supports only \['off', 'joint'\]",
    ):
        adapter.apply_stage("D-normalize", state)

    assert rng.bit_generator.state == cursor


def test_adapter_replacement_shares_unchanged_tracks_and_fast_paths_identity(monkeypatch):
    _, _, _, adapter, _ = _run(monkeypatch)
    state = adapter.state_from_bundle(
        WaveformBundle(
            {
                NEAREND: np.array([1.0, -1.0], dtype=np.float32),
                MIC: np.array([2.0, -2.0], dtype=np.float32),
                FAREND: np.array([3.0, -3.0], dtype=np.float32),
                ECHO: np.array([4.0, -4.0], dtype=np.float32),
                NOISE: np.array([5.0, -5.0], dtype=np.float32),
            }
        )
    )

    next_state = adapter._state(
        state, mic=np.array([6.0, -6.0], dtype=np.float32)
    )

    assert next_state.waveforms[NEAREND] is state.waveforms[NEAREND]
    assert next_state.waveforms[FAREND] is state.waveforms[FAREND]
    assert next_state.waveforms[ECHO] is state.waveforms[ECHO]
    assert next_state.waveforms[NOISE] is state.waveforms[NOISE]
    assert adapter._state(next_state, mic=next_state.waveforms[MIC]) is next_state


def test_missing_required_waveform_fails_closed():
    bundle = WaveformBundle(
        {
            NEAREND: np.ones(2, dtype=np.float32),
            MIC: np.ones(2, dtype=np.float32),
        }
    )
    with pytest.raises(ValueError, match="missing roles"):
        TaskPostMixAdapter.state_from_bundle(bundle)


def test_scene_time_transform_replays_one_plan_on_all_aec_roles():
    class _TimeDataset:
        target_sr = 8

        def _build_final_time_plan(
            self,
            speech,
            noisy,
            fs_out,
            rng,
            *,
            fixed_silence_canvas_active,
        ):
            assert len(speech) == len(noisy) == 4
            assert fs_out == 8
            assert fixed_silence_canvas_active is False
            gap = int(rng.integers(1, 2))
            transform = plan_time_transform(
                input_length=4,
                operations=(TileOp(output_length=7, gap_length=gap),),
                policy="test_tile_with_gap",
            )
            return SimpleNamespace(
                transform=transform,
                output_length=7,
                fixed_silence_canvas_active=False,
            )

    calls: list[tuple[int, int, int]] = []

    def _resample(waveform, source_sr, target_sr):
        calls.append((len(waveform), source_sr, target_sr))
        return np.repeat(np.asarray(waveform, dtype=np.float32), 2)

    tracks = {
        NEAREND: np.array([1.0, 2.0], dtype=np.float32),
        MIC: np.array([3.0, 4.0], dtype=np.float32),
        FAREND: np.array([5.0, 6.0], dtype=np.float32),
        ECHO: np.array([7.0, 8.0], dtype=np.float32),
        NOISE: np.array([9.0, 10.0], dtype=np.float32),
        "export/layer_d_noisy": np.array([11.0, 12.0], dtype=np.float32),
    }
    result = apply_scene_time_transform(
        dataset=_TimeDataset(),
        bundle=WaveformBundle(tracks),
        source_sample_rate=4,
        rng=np.random.default_rng(17),
        fixed_silence_canvas_active=False,
        resample_adapter=_resample,
    )

    assert result.plan.policy == "aec_target_sr+test_tile_with_gap"
    assert [type(op).__name__ for op in result.plan.operations] == [
        "ResampleOp",
        "TileOp",
    ]
    assert calls == [(2, 4, 8)] + [(2, 4, 8)] * len(tracks)
    for role, source in tracks.items():
        repeated = np.repeat(source, 2)
        expected = np.concatenate(
            [repeated, np.zeros(1, dtype=np.float32), repeated]
        )[:7]
        np.testing.assert_array_equal(result.bundle.require(role), expected)



def test_final_tuple_projection_uses_bundle_and_task_context_only():
    observed = {}

    class _ProjectionDataset:
        def _assemble_training_tuple(self, *args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            return ("projected",)

    context = AECPostMixContext.from_aec_disk(
        branch="prob_target_echo",
        task="joint",
        aec_disk={
            "prob_target_echo_disk": {
                "farend": np.ones(2, dtype=np.float32),
                "echo": np.ones(2, dtype=np.float32),
                "c3_meta": {"delay_samples": 3},
            }
        },
        used_sidechain_wind=False,
        joint_target_rir_mismatch=False,
        sidechain_wind_mismatch_reason=None,
        index=5,
    )
    bundle = WaveformBundle(
        {
            NEAREND: np.array([1.0, 2.0], dtype=np.float32),
            MIC: np.array([3.0, 4.0], dtype=np.float32),
            FAREND: np.array([5.0, 6.0], dtype=np.float32),
            ECHO: np.array([7.0, 8.0], dtype=np.float32),
            NOISE: np.array([9.0, 10.0], dtype=np.float32),
        }
    )
    output = project_training_tuple(
        dataset=_ProjectionDataset(),
        bundle=bundle,
        sample_rate=8_000,
        index=5,
        uid="speaker-a",
        rng=np.random.default_rng(29),
        context=context,
        resampled_from=16_000,
    )

    assert output == ("projected",)
    args = observed["args"]
    np.testing.assert_array_equal(args[0], bundle.require(NEAREND))
    np.testing.assert_array_equal(args[1], bundle.require(MIC))
    np.testing.assert_array_equal(args[7], bundle.require(NOISE))
    assert args[2:5] == (8_000, 5, "speaker-a")
    assert args[5]["prob_target_echo_disk"]["c3_meta"]["sample_rate"] == 16_000
    assert observed["kwargs"] == {"branch": "prob_target_echo"}


def test_scene_time_wrapper_delegates_task_policy_and_projects_legacy_result(monkeypatch):
    class _TimeDataset:
        target_sr = 8

        def _build_final_time_plan(self, *_args, **_kwargs):
            raise AssertionError("shared seam must own planner invocation")

    bundle = WaveformBundle(
        {
            NEAREND: np.array([1.0, 2.0], dtype=np.float32),
            MIC: np.array([3.0, 4.0], dtype=np.float32),
            FAREND: np.array([5.0, 6.0], dtype=np.float32),
            ECHO: np.array([7.0, 8.0], dtype=np.float32),
            NOISE: np.array([9.0, 10.0], dtype=np.float32),
        }
    )
    rng = np.random.default_rng(47)
    calls = {}
    expected_plan = plan_time_transform(
        input_length=2,
        operations=(),
        policy="aec_target_sr+delegated",
        required_tracks=tuple(bundle.tracks),
    )

    def _shared_scene_time_transform(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            bundle=bundle,
            plan=expected_plan,
            sample_rate=8,
            fixed_silence_canvas_active=True,
        )

    monkeypatch.setattr(
        adapter_module, "apply_shared_scene_time_transform", _shared_scene_time_transform
    )
    result = apply_scene_time_transform(
        dataset=_TimeDataset(),
        bundle=bundle,
        source_sample_rate=4,
        rng=rng,
        fixed_silence_canvas_active=True,
    )

    assert isinstance(result, adapter_module.AECSceneTimeResult)
    assert result.bundle is bundle
    assert result.plan is expected_plan
    assert result.sample_rate == 8
    assert result.fixed_silence_canvas_active is True
    assert calls["bundle"] is bundle
    assert calls["source_sample_rate"] == 4
    assert calls["target_sample_rate"] == 8
    assert calls["rng"] is rng
    assert calls["fixed_silence_canvas_active"] is True
    assert calls["build_final_time_plan"].__self__.__class__ is _TimeDataset
    assert calls["resample_adapter"] is adapter_module._resample_audio
    assert calls["policy_prefix"] == "aec_target_sr"
