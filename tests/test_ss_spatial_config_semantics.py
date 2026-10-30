"""SS 的 timeline 与 post-RIR 音量经过真实 WAV/provider/卷积/trace/collate。"""

import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from core.data.simulation.scene_manifest import trajectory_digest
from tasks.ss.data.collate import ss_collate_fn
from tasks.ss.data.dynamic_dataset import SSDynamicMixingDataset

pytestmark = pytest.mark.unit

FS, LENGTH, HOP = 8000, 400, 200
UIDS = ("a", "b")


def _assets(tmp_path, *, moving=False, silent_reference=False):
    waves, banks, entries, identities = {}, {}, {}, {}
    for index, uid in enumerate(UIDS):
        waves[uid] = (0.03 + 0.01 * np.sin(np.arange(LENGTH) * (0.05 + index * 0.02))).astype(np.float32)
        sf.write(tmp_path / f"{uid}.wav", waves[uid], FS, subtype="FLOAT")
        full = np.zeros((2, 4), dtype=np.float32)
        full[:, 0] = np.array([1., 0.5]) * (index + 1)
        full[:, 1] = full[:, 0] * 0.15
        full[:, 3] = full[:, 0] * 0.25
        if silent_reference:
            full[0] = 0
        early = full.copy()
        early[:, 2:] = 0
        direct = full.copy()
        direct[:, 1:] = 0
        tracks = {"full": full, "early": early, "direct": direct}
        entries[uid], identities[uid], banks[uid] = {}, {}, {}
        for kind, array in tracks.items():
            if moving:
                array = np.stack([array, array * 0.5])
            array = np.ascontiguousarray(array, dtype="<f4")
            banks[uid][kind] = array
            path = tmp_path / f"{uid}_{kind}.npy"
            np.save(path, array)
            entries[uid][kind] = str(path)
            identities[uid][kind] = hashlib.sha256(array.tobytes()).hexdigest()
    speech = tmp_path / "speech.scp"
    speech.write_text("".join(f"{uid} {FS} {LENGTH} {tmp_path / (uid + '.wav')}\n" for uid in UIDS))
    noise_path = tmp_path / "noise.wav"
    sf.write(noise_path, np.full(LENGTH, 0.01, dtype=np.float32), FS, subtype="FLOAT")
    noise = tmp_path / "noise.scp"
    noise.write_text(f"n {FS} {LENGTH} {noise_path}\n")
    metadata = {
        "schema_version": 1, "scene_uid": "ss-semantics", "sample_rate": FS,
        "actor_order": list(UIDS),
        "room": {"room_uid": "fixture-room", "room_dim_m": [4., 4., 3.],
                 "rt60": 0.3, "sound_speed": 340., "rir_num_samples": 4},
        "actor_positions_m": {uid: [i + 1., 1., 1.5] for i, uid in enumerate(UIDS)},
        "geometry": {"positions_m": [[1., 1., 1.5], [1.08, 1., 1.5]],
                     "reference_channel": 0, "channel_order": ["mic_0", "mic_1"]},
        "generator_version": "ss-semantics-fixture-v1", "track_identities": identities,
    }
    if moving:
        trajectories = {uid: [[[i + 1., 1., 1.5]], [[i + 1.1, 1., 1.5]]] for i, uid in enumerate(UIDS)}
        clock = {"sample_rate": FS, "num_samples": LENGTH, "hop_samples": HOP, "frame_count": 2, "start_sample": 0}
        metadata.update({
            "schema_version": 3, "moving": True, "actor_roles": list(UIDS),
            "actor_trajectories": trajectories, "trajectory_timebase": clock,
            "trajectory_sha256": trajectory_digest(trajectories, clock),
        })
    manifest = tmp_path / "scene.json"
    manifest.write_text(json.dumps({"entries": entries, "metadata": metadata}))
    return {"speech": speech, "noise": noise, "manifest": manifest, "waves": waves,
            "banks": banks, "metadata": metadata, "moving": moving}


def _sample(assets, *, sir_range=(2., 2.), **overrides):
    cfg = {
        "spatial_enabled": True, "array_profile": "linear_2ch_80mm", "reference_channel": 0,
        "prob_reverberation": 1., "prob_moving_rir": 0.,
        "early_rir_sec": 2 / FS, "peak_normalization_mode": "off",
        "prob_channel_enabled": False, "use_high_pass": False,
        "noisy_post_mix_degradations_enabled": False,
        "source_volume_normalization_enabled": False, "prob_timeline": 0.,
        "source_volume_use_rms": False, "source_volume_db_range": [-40., -40.],
        "snr_low": 5., "snr_high": 5., **overrides,
    }
    dataset = SSDynamicMixingDataset(
        str(assets["speech"]), noise_scp=str(assets["noise"]), rir_scp=str(assets["manifest"]),
        num_sources=2, sample_rate=FS, segment_seconds=LENGTH / FS, seed=19,
        sir_range=sir_range, simulation_cfg=cfg,
    )
    traced = dataset.get_sample_with_trace(0, capture_stems=True)
    batch = ss_collate_fn([traced.sample])
    batch.validate()
    assert batch.mixture.shape == (1, 2, LENGTH)
    assert batch.sources.shape == (1, 2, LENGTH)
    assert batch.source_mask.all()
    return traced


def _convolution(assets, uid, wave, kind):
    bank = assets["banks"][uid][kind]
    if not assets["moving"]:
        return np.stack([np.convolve(wave, rir)[:LENGTH] for rir in bank]).astype(np.float32)
    out = np.zeros((2, LENGTH), dtype=np.float32)
    for frame in range(2):
        start = frame * HOP
        for channel in range(2):
            value = np.convolve(wave[start:start + HOP], bank[frame, channel])
            count = min(len(value), LENGTH - start)
            out[channel, start:start + count] += value[:count]
    return out


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("target_type", ["dry", "early"])
def test_volume_changes_real_spatial_audio_with_common_gain(tmp_path, moving, target_type):
    assets = _assets(tmp_path, moving=moving)
    before = _sample(assets, target_type=target_type)
    after = _sample(assets, target_type=target_type, source_volume_normalization_enabled=True)
    assert not np.array_equal(before.sample["mixture"], after.sample["mixture"])
    facts = after.trace.metadata_dict()["scene"]
    assert before.sample["source_ids"] == after.sample["source_ids"]
    for index, uid in enumerate(after.sample["source_ids"]):
        full = _convolution(assets, uid, assets["waves"][uid], "full")
        early = _convolution(assets, uid, assets["waves"][uid], "early")
        gain = 0.01 / (float(np.max(np.abs(early[0]))) + 1.01e-6)
        calibration = facts["source_volume_calibration"][index]
        assert calibration["gain"] == pytest.approx(gain, rel=1e-6)
        assert calibration["reference_db_after"] == pytest.approx(-40., abs=0.001)
        assert calibration["stage"] == "post_rir_pre_noise_echo"
        if index:
            gain *= facts["sir"][index - 1]["gain"]
        np.testing.assert_allclose(after.trace.sources["scene"].stems[f"source_full_rir/{index}"], full * gain, rtol=2e-6, atol=1e-8)
        target = assets["waves"][uid] if target_type == "dry" else early[0]
        np.testing.assert_allclose(after.sample["sources"][index], target * gain, rtol=2e-6, atol=1e-8)
    assert facts["sir"][0]["selected_db"] == before.trace.metadata_dict()["scene"]["sir"][0]["selected_db"]
    assert facts["decomposable_final"]


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("target_type", ["dry", "early"])
def test_zero_overlap_keeps_fixed_sources_but_only_one_active(tmp_path, moving, target_type):
    assets = _assets(tmp_path, moving=moving)
    before = _sample(assets, target_type=target_type)
    after = _sample(assets, target_type=target_type, prob_timeline=1., overlap_ratio_range=[0., 0.])
    assert np.count_nonzero(after.sample["sources"][1]) == 0
    stems = after.trace.sources["scene"].stems
    assert np.count_nonzero(stems["source_full_rir/1"]) == 0
    np.testing.assert_array_equal(after.sample["mixture"], stems["source_full_rir/0"])
    assert after.sample["source_ids"] == before.sample["source_ids"]
    facts = after.trace.metadata_dict()["scene"]
    assert facts["source_count"] == 2
    assert facts["activity"]["active_source_count"] == 1
    assert facts["activity"]["active_ratios"] == [1., 0.]
    assert facts["activity"]["measure_point"] == "post_rir_pre_background"
    assert facts["sir"][0]["reason"] == "inactive_source"
    assert facts["sir"][0]["selected_db"] == before.trace.metadata_dict()["scene"]["sir"][0]["selected_db"]
    assert facts["timeline"][0]["turns"] == []
    assert facts["spatial_scene"] == before.trace.metadata_dict()["scene"]["spatial_scene"]


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("target_type", ["dry", "early"])
@pytest.mark.parametrize("mode", ["head", "tail"])
def test_half_overlap_aligns_dry_and_reverberant_targets(tmp_path, moving, target_type, mode):
    assets = _assets(tmp_path, moving=moving)
    baseline = _sample(assets, target_type=target_type)
    after = _sample(
        assets, target_type=target_type, prob_timeline=1., overlap_ratio_range=[0.5, 0.5],
        timeline_mode_weights={name: float(name == mode) for name in ("head", "tail", "middle", "crossing", "containment")},
        source_volume_normalization_enabled=True,
    )
    facts = after.trace.metadata_dict()["scene"]
    uid = after.sample["source_ids"][1]
    placed = np.zeros(LENGTH, dtype=np.float32)
    start = 0 if mode == "head" else HOP
    placed[start:start + HOP] = assets["waves"][uid][:HOP]
    full = _convolution(assets, uid, placed, "full")
    early = _convolution(assets, uid, placed, "early")[0]
    gain = 0.01 / (float(np.max(np.abs(early))) + 1.01e-6)
    assert facts["source_volume_calibration"][1]["gain"] == pytest.approx(gain, rel=1e-6)
    gain *= facts["sir"][0]["gain"]
    expected_target = placed if target_type == "dry" else early
    np.testing.assert_allclose(after.sample["sources"][1], expected_target * gain, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(after.trace.sources["scene"].stems["source_full_rir/1"], full * gain, rtol=2e-6, atol=1e-8)
    if mode == "head":
        assert placed[HOP] == 0
        assert early[HOP] > 0
        assert full[0, HOP + 2] > 0
    spatial = facts["spatial_scene"]
    assert spatial == baseline.trace.metadata_dict()["scene"]["spatial_scene"]
    assert spatial["source_order"] == list(UIDS)
    assert spatial["geometry"]["positions_m"] == assets["metadata"]["geometry"]["positions_m"]
    if moving:
        assert spatial["trajectory_timebase"] == assets["metadata"]["trajectory_timebase"]
        for actor in spatial["scene_plan"]["actors"]:
            np.testing.assert_allclose(actor["trajectory"]["positions_m"], assets["metadata"]["actor_trajectories"][actor["actor_uid"]])
        wave = assets["waves"][UIDS[0]]
        static_image = np.convolve(wave, assets["banks"][UIDS[0]]["full"][0, 0])[:LENGTH]
        moving_image = baseline.trace.sources["scene"].stems["source_full_rir/0"][0]
        np.testing.assert_allclose(moving_image[HOP + 4:], static_image[HOP + 4:] * 0.5, atol=1e-8)


@pytest.mark.parametrize("moving", [False, True])
def test_noise_only_volume_changes_background_without_latent_speech(tmp_path, moving):
    assets = _assets(tmp_path, moving=moving)
    kwargs = dict(prob_target_only=0., prob_noise_only=1., prob_timeline=1.)
    before = _sample(assets, **kwargs)
    after = _sample(assets, **kwargs, source_volume_normalization_enabled=True)
    assert not np.array_equal(before.sample["mixture"], after.sample["mixture"])
    assert np.count_nonzero(after.sample["mixture"]) > 0
    assert after.sample["source_ids"] == ("absent", "absent")
    for source in after.sample["sources"]:
        assert np.count_nonzero(source) == 0
    stems = after.trace.sources["scene"].stems
    assert not any(role.startswith("source_full_rir/") for role in stems)
    np.testing.assert_array_equal(after.sample["mixture"], stems["background"])
    facts = after.trace.metadata_dict()["scene"]
    gain = facts["source_volume_calibration"][0]["gain"]
    np.testing.assert_allclose(after.sample["mixture"], before.sample["mixture"] * gain, rtol=2e-5, atol=1e-8)
    assert facts["activity"]["active_source_count"] == 0
    assert facts["latent_snr_reference_removed_from_observation"]
    assert facts["decomposable_final"]


@pytest.mark.parametrize("moving", [False, True])
def test_activity_and_sir_use_nonreference_channels(tmp_path, moving):
    assets = _assets(tmp_path, moving=moving, silent_reference=True)
    result = _sample(assets)
    assert np.count_nonzero(result.sample["mixture"][0]) == 0
    assert np.count_nonzero(result.sample["mixture"][1]) > 0
    facts = result.trace.metadata_dict()["scene"]
    assert facts["activity"]["active_ratios"] == [1., 1.]
    assert facts["activity"]["active_source_count"] == 2
    assert facts["sir"][0]["reason"] is None
    assert facts["sir"][0]["realized_db"] == pytest.approx(2.)


@pytest.mark.parametrize("moving,expected", [
    (False, {
        "mixture": "e645037acd086644a532b0466be6539e6bca7aaa9bac5fff50292fb36c92b43b",
        "background": "b17df3bb7bb2b57a377e91bdcad21d6a858c18dafa95a8543827e27bf1af3867",
        "source_full_rir/0": "76bd970ebddf75213eab7a1c8d389ff94c0f3b29c2e10a737eb60244e2bd4db1",
        "source_full_rir/1": "434a9ec759701a75efbe2642b365cad61dd3062d2a784f797bd2fc8ad1f63a4f",
        "target/0": "f0810e85015445cf1b8c59bb5fc50fb1a219f9b4e66db91ce2263f3140fe370a",
        "target/1": "951467898ad7e44975ac74b4b754c5ad838587a87af95d996fd5e35791400775",
    }),
    (True, {
        "mixture": "27d2cd1c7d8769cc728ebfb271646508e25dae8917eac490ceef3d38af314187",
        "background": "62ebb0f942cf24475f35957d4f7927916607335d0b691dd62a00f4755edcd0ab",
        "source_full_rir/0": "dcd2d8d0d2fd6fc8bedc5265b23fb462737d69e7d8e8e0734854c52f3ea49261",
        "source_full_rir/1": "a0806e9d478c8c0212862247997b67213de31a35d9d908d3d7bb72ca73f52faa",
        "target/0": "f0810e85015445cf1b8c59bb5fc50fb1a219f9b4e66db91ce2263f3140fe370a",
        "target/1": "bfc5bdea85a6439d45617f722c771f2185975c109076f0f18fb4483575a3c09e",
    }),
])
def test_disabled_stages_match_prechange_real_audio_baseline(tmp_path, moving, expected):
    """固定摘要采自接入 timeline/volume 前的真实 SS 路径。"""
    traced = _sample(_assets(tmp_path, moving=moving), prob_target_only=0., prob_target_noise=1., target_type="dry")
    assert {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in traced.trace.sources["scene"].stems.items()} == expected
    facts = traced.trace.metadata_dict()["scene"]
    assert facts["source_volume_calibration"] is None
    assert facts["timeline"] == []
    assert facts["sir"][0]["selected_db"] == 2.
    assert facts["sir"][0]["gain"] == pytest.approx(0.4040210200263749 if moving else 0.39177472498581534, rel=1e-12)
    assert facts["spatial_scene"]["scene_uid"] == "ss-semantics"
    assert facts["spatial_scene"]["source_order"] == list(UIDS)


def test_shared_renderer_preserves_preextraction_audio_and_rng():
    """公共 renderer 的 beta 抽样和逐源 SIR 使用提取 helper 前的数值基线。"""
    from types import SimpleNamespace
    from core.data.simulation.scene_renderer import render_speech_scene
    from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
    from core.data.simulation_configs import SimulationConfigs
    from core.data.simulation_views import project_simulation_view

    cfg = SimulationConfigs.from_dict(dict(
        spatial_enabled=True, array_profile="linear_2ch_80mm", reference_channel=0,
        prob_reverberation=1., source_volume_normalization_enabled=True,
        source_volume_distribution="beta",
    ))
    dataset = SimpleNamespace(sim_cfg=cfg, export_simulation_stems=False, _simulation_view=project_simulation_view(cfg, "ss"))
    rng = np.random.default_rng(65)
    actors = (
        AcousticActor(
            uid="a",
            role=AcousticRole.TARGET_SPEECH,
            rir_policy=RIRPolicy.SPATIAL_SOURCE,
            waveform=np.linspace(.01, .05, 32, dtype=np.float32),
        ),
        AcousticActor(
            uid="b",
            role=AcousticRole.INTERFERENCE_SPEECH,
            rir_policy=RIRPolicy.SPATIAL_SOURCE,
            waveform=np.linspace(.05, .01, 32, dtype=np.float32),
        ),
    )
    scene = render_speech_scene(
        dataset, actors,
        FS, rng, sir_range=[-5., 5.], sir_mode="per_source", need_noise=False,
        spatial_rirs=np.array([[[1., .2], [.5, .1]], [[.8, .4], [.4, .2]]], np.float32),
    )
    expected = {
        "mixture": "716ff2891b0246f51d85e8797a5a45ab0b0095c4813367bfeca459a729b4fc38",
        "background": "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1",
        "source_full_rir/0": "0069ad60e8cb0c878eeab1d22b86fde3c8cc3d0b84f5530738a11226d37521b9",
        "source_full_rir/1": "c04fe72d3c070e6392b21fc9d710f7f235e81b213dd4dce7195963e815524b95",
        "target/0": "ae5c0e184bdd4c54c5c1e71c5bce97b36cb6c8079f7b8d0b24b017062e3c7a29",
        "target/1": "45f38a7ca95ef1af4d1cec47a885da74b6270fcaa2b5e12ae4cf71e1b3755985",
    }
    assert {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in scene.bundle.tracks.items()} == expected
    assert rng.bit_generator.state["state"]["state"] == 127175890178605156487036964822185441878


@pytest.mark.parametrize("enabled", [False, True])
def test_volume_helper_preserves_inputs_and_disabled_rng(enabled):
    from core.data.simulation.scene_renderer import calibrate_spatial_source_volume
    from core.data.simulation_configs import SimulationConfigs

    cfg = SimulationConfigs.from_dict({"source_volume_normalization_enabled": enabled, "reference_channel": 0})
    rng = np.random.default_rng(71)
    state = rng.bit_generator.state
    full = np.full((2, 2, 8), 0.1, dtype=np.float32)
    early = full * 0.5
    targets = full[:, 0].copy()
    originals = tuple(array.copy() for array in (full, early, targets))
    output = calibrate_spatial_source_volume(full, early, targets, cfg, FS, rng)
    for array, original in zip((full, early, targets), originals):
        np.testing.assert_array_equal(array, original)
    if enabled:
        for index, calibration in enumerate(output[3]):
            for array, original in zip(output[:3], originals):
                np.testing.assert_array_equal(array[index], original[index] * calibration["gain"])
    else:
        assert rng.bit_generator.state == state
        assert output[3] == []
        assert all(left is right for left, right in zip(output[:3], (full, early, targets)))


@pytest.mark.parametrize("moving", [False, True])
def test_new_stages_do_not_shift_existing_random_streams(tmp_path, moving):
    assets = _assets(tmp_path, moving=moving)
    kwargs = dict(sir_range=(-5., 5.), prob_target_only=0., prob_target_noise=1., snr_low=2., snr_high=8.)
    before = _sample(assets, **kwargs)
    after = _sample(assets, **kwargs, prob_timeline=1., overlap_ratio_range=[.2, .8],
                    source_volume_normalization_enabled=True, source_volume_db_range=[-40., -30.])
    old, new = (result.trace.metadata_dict()["scene"] for result in (before, after))
    assert old["sir"][0]["selected_db"] == new["sir"][0]["selected_db"]
    assert old["spatial_scene"] == new["spatial_scene"]
    for key in ("component_identities", "snr_nominal_db"):
        assert old["noise_image"][key] == new["noise_image"][key]
    assert old["noise_image"]["snr_gain"] != new["noise_image"]["snr_gain"]
    realized_snr = []
    for result in (before, after):
        stems = result.trace.sources["scene"].stems
        speech = stems["source_full_rir/0"] + stems["source_full_rir/1"]
        noise = stems["background"]
        realized_snr.append(10 * np.log10(np.sum(speech.astype(np.float64) ** 2) / np.sum(noise.astype(np.float64) ** 2)))
    assert realized_snr[0] == pytest.approx(realized_snr[1], abs=1e-4)
    old_noise = before.trace.sources["scene"].stems["background"]
    new_noise = after.trace.sources["scene"].stems["background"]
    np.testing.assert_allclose(old_noise / np.max(np.abs(old_noise)), new_noise / np.max(np.abs(new_noise)), rtol=1e-6)
