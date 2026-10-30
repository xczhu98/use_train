"""四任务共享的 Layer C 背景混合；语音声学渲染与背景能量参考显式分离。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.data.post_augment import colored_noise
from core.data.simulation.engine import mix_noise
from core.data.simulation_audio_utils import _align_noise_2d_to_length
from core.data.simulation_utils import rng_random, rng_uniform
from core.data.simulation.noise_image import (
    NoiseImage,
    adapt_noise_waveform,
    scale_noise_image,
)
from core.data.simulation.role_policy import BackgroundPolicy, BackgroundSpec
from core.data.simulation.scene_context import coerce_background_runtime


def wind_pool_required(
    prob_wind_noise_sidechain,
    scenario_noise_probs=(),
    prob_wind_noise_background=0.0,
):
    """四任务共享的风噪池需求判定。

    sidechain/background 两个概率都 ≤ 0 时永远不需要风噪池；未提供场景概率
    （SE/AEC）时只看概率；
    提供场景概率（TSE/SS）时要求至少一个含 noise 的场景概率 > 0，
    否则风噪不会被触发，不应 fail-fast 要求风噪池。
    """
    if prob_wind_noise_sidechain <= 0 and prob_wind_noise_background <= 0:
        return False
    if not scenario_noise_probs:
        return True
    return any(p > 0 for p in scenario_noise_probs)


@dataclass(frozen=True)
class BackgroundMix:
    observation: np.ndarray
    additive: np.ndarray
    snr_values: list[float]
    used_sidechain_wind: bool
    mix_two: bool
    want_sidechain: bool
    wind_snr: float
    primary_snr: float | None
    wind_candidate_snr: float | None
    audit: dict | None
    noise_image_provenance: dict | None = None
    used_wind_background: bool = False
    used_colored_noise: bool = False


def mix_background(runtime, noisy_speech, mix_reference, sr, actual_len, max_dur_sr, rng, need_noise, snr,
                   *, image_geometry=None, background_spec=None):
    """按显式背景策略混合背景；``runtime`` 是注入的资源端口。

    ``point_source_rir`` 由 ``BackgroundSpec.point_source_provider`` 负责把
    mono noise waveform 渲染成当前 scene 的 ``NoiseImage``；空间 wind background
    也通过同一 ``BackgroundSpec`` 走 diffuse 或 point-source 投影，不进入
    ``wind_noise`` 的非加性 sidechain 分支。
    """
    runtime = coerce_background_runtime(runtime)
    cfg = runtime.config
    if not isinstance(background_spec, BackgroundSpec):
        raise TypeError("background_spec must be a BackgroundSpec")
    policy = background_spec.policy
    if image_geometry is None and policy in {
        BackgroundPolicy.DIFFUSE_COHERENCE,
        BackgroundPolicy.DIRECT_MULTICHANNEL,
        BackgroundPolicy.POINT_SOURCE_RIR,
    }:
        raise ValueError(f"background policy {policy.value!r} requires image_geometry")
    if image_geometry is not None and policy == BackgroundPolicy.ADDITIVE_MONO:
        raise ValueError("additive_mono cannot be used for a spatial observation")
    if policy == BackgroundPolicy.NONE and need_noise:
        raise ValueError("background policy 'none' conflicts with need_noise=True")
    snr_primary_drawn_db = None
    wind_noise_snr_candidate_db = None
    wn_snr = 0.0
    additive_noise_2d = np.zeros_like(noisy_speech, dtype=np.float32)
    additive_snr_db_list: list[float] = []
    acoustic_audit = {} if runtime.export_simulation_stems else None

    reuse_noise = getattr(cfg, "reuse_noise", False)
    used_sidechain_wind = False
    used_wind_background = False
    used_colored_noise = False
    colored_facts = None
    mix_two = False
    want_sidechain = False
    want_wind_background = False
    image_facts = []
    wind_background_facts = None
    wind_sidechain_facts: dict = {}

    def _record_component_identities(start=0):
        record = runtime.acoustic_plan_record()
        if record is None:
            return []
        sources = record.get("noise_sources", ())
        identities = []
        for source in tuple(sources)[int(start):]:
            uid = getattr(source, "uid", None)
            if not uid:
                continue
            family = str(getattr(source, "family", "ordinary"))
            family = family.rsplit(".", 1)[-1].lower()
            prefix = "noise" if family == "ordinary" else family
            identities.append(f"{prefix}_scp:{uid}")
        return identities

    def add_ordinary(
        observation, waveform, selected_snr, audit, source_identity=None,
        component_identities=(),
    ):
        value = np.asarray(waveform)
        component_identities = list(component_identities)
        if not component_identities and source_identity is not None:
            component_identities = [str(source_identity)]
        if source_identity is None and component_identities:
            source_identity = (
                component_identities[0]
                if len(component_identities) == 1
                else "noise_mix:" + "+".join(component_identities)
            )
        if policy == BackgroundPolicy.DIRECT_MULTICHANNEL:
            if value.ndim != 2:
                raise ValueError("direct_multichannel background requires a [C,T] waveform")
        if policy == BackgroundPolicy.POINT_SOURCE_RIR:
            validate_geometry = getattr(
                background_spec.point_source_provider, "validate_geometry", None
            )
            if not callable(validate_geometry):
                raise TypeError(
                    "point_source_rir provider must implement validate_geometry"
                )
            validate_geometry(image_geometry)
            if value.ndim == 2:
                if value.shape[0] != 1:
                    raise ValueError(
                        "point_source_rir background requires a mono [T] or [1,T] waveform"
                    )
                value = value[0]
            if value.ndim != 1:
                raise ValueError("point_source_rir background requires a mono waveform")
            rendered = background_spec.point_source_provider.render(
                value, sample_rate=int(sr)
            )
            if not isinstance(rendered, NoiseImage):
                raise TypeError("point source provider must return a NoiseImage")
            if rendered.waveform.shape != observation.shape:
                raise ValueError(
                    "point source RIR image shape does not match observation: "
                    f"{rendered.waveform.shape} vs {observation.shape}"
                )
            rendered_provenance = dict(getattr(rendered, "provenance", {}) or {})
            resolved_source_identity = (
                source_identity
                or rendered_provenance.get("source_identity")
                or rendered_provenance.get("source_uid")
            )
            point_source_provenance = {
                "snr_reference": cfg.snr_reference,
                "stage": "post_point_source_rir_pre_mix",
                "component_identities": component_identities,
                "coherence_policy": "point_source_rir",
            }
            if resolved_source_identity is not None:
                point_source_provenance["source_identity"] = str(resolved_source_identity)
            scaled = scale_noise_image(
                mix_reference,
                rendered,
                snr_db=selected_snr,
                provenance=point_source_provenance,
            )
            image_facts.append(dict(scaled.provenance))
            image_facts[-1]["background_policy"] = policy.value
            image_facts[-1]["background_shared_scene"] = bool(background_spec.shared_scene)
            if audit is not None:
                audit.update(scaled.provenance)
            return observation + scaled.waveform, scaled.waveform
        effective_policy = (
            BackgroundPolicy.DIRECT_MULTICHANNEL
            if policy == BackgroundPolicy.DIFFUSE_COHERENCE
            and value.ndim == 2 and value.shape[0] > 1
            else policy
        )
        # spatial 调用在唯一标定点前适配原始 noise，禁止事后重标定整个背景。
        if image_geometry is None:
            return mix_noise(observation, waveform, snr=selected_snr, rng=rng,
                             snr_reference=mix_reference, audit_out=audit)
        image = adapt_noise_waveform(
            waveform, geometry=image_geometry, sample_rate=sr, rng=rng,
            coherence_policy=cfg.spatial_diffuse_coherence_model,
            reference_channel=cfg.reference_channel, source_identity=source_identity,
            provenance={"component_identities": component_identities},
        )
        scaled = scale_noise_image(
            mix_reference, image, snr_db=selected_snr,
            provenance={"snr_reference": cfg.snr_reference, "stage": "post_projection_pre_mix"},
        )
        image_facts.append(dict(scaled.provenance))
        image_facts[-1]["background_policy"] = effective_policy.value
        image_facts[-1]["background_shared_scene"] = bool(background_spec.shared_scene)
        if audit is not None:
            audit.update(scaled.provenance)
        return observation + scaled.waveform, scaled.waveform

    def align_noise_sample(noise_sample):
        value = np.asarray(noise_sample)
        if policy == BackgroundPolicy.POINT_SOURCE_RIR and value.ndim == 1:
            value = value[None, :]
        return _align_noise_2d_to_length(value, noisy_speech.shape[1], rng)

    if need_noise:
        snr_primary_drawn_db = float(snr)
        _wcfg = cfg.wind_noise_config
        _wn_snr_lo, _wn_snr_hi = _wcfg["wind_noise_snr_range"]
        wc_wn = float(rng_uniform(rng, float(_wn_snr_lo), float(_wn_snr_hi)))
        wind_noise_snr_candidate_db = float(wc_wn)
        mix_two = rng_random(rng) < cfg.noise_mix_prob
        want_sidechain = rng_random(rng) < cfg.prob_wind_noise_sidechain
        if image_geometry is not None and float(getattr(cfg, "prob_wind_noise_background", 0.0)) > 0.0:
            want_wind_background = (
                rng_random(rng) < float(cfg.prob_wind_noise_background)
            )
        # Wind-noise SNR must match the actual side-chain branch to keep metadata consistent.
        wn_snr = wc_wn if (want_sidechain or want_wind_background) else snr

        if want_sidechain:
            noisy_speech, additive_noise_2d, used_sidechain_wind = (
                runtime.apply_wind(
                    noisy_speech,
                    mix_reference,
                    additive_noise_2d,
                    sr,
                    actual_len,
                    max_dur_sr,
                    wn_snr,
                    rng,
                    reuse_noise,
                    acoustic_audit,
                    image_geometry,
                    background_spec,
                    wind_sidechain_facts,
                )
            )
        if want_wind_background and not want_sidechain:
            record = runtime.acoustic_plan_record()
            source_start = len(record.get("noise_sources", ())) if record is not None else 0
            wind_sample = runtime.get_wind_sample(
                sr, max_dur_sr, rng, reuse_noise
            )
            if wind_sample is not None:
                wind_sample = align_noise_sample(wind_sample)
                wind_audit = {} if acoustic_audit is not None else None
                wind_component_identities = _record_component_identities(source_start)
                if not wind_component_identities:
                    source_identity = runtime.last_wind_source_identity
                    if source_identity is not None:
                        wind_component_identities = [source_identity]
                noisy_speech, scaled_wind = add_ordinary(
                    noisy_speech,
                    wind_sample,
                    wc_wn,
                    wind_audit,
                    component_identities=wind_component_identities,
                )
                additive_noise_2d += scaled_wind.astype(np.float32, copy=False)
                additive_snr_db_list.append(wc_wn)
                used_wind_background = True
                wind_background_facts = {
                    "shape": list(scaled_wind.shape),
                    "snr_db": float(wc_wn),
                    "source_identity": (
                        wind_component_identities[0]
                        if wind_component_identities
                        else None
                    ),
                    "background_policy": policy.value,
                    "geometry_bound": bool(background_spec.geometry_bound),
                }
                if acoustic_audit is not None:
                    acoustic_audit.setdefault("additive", []).append(
                        {"source": "wind_background", **wind_audit}
                    )
                    acoustic_audit["wind_background"] = {
                        "applied": True,
                        "waveform_shape": list(scaled_wind.shape),
                        "snr_db": float(wc_wn),
                        "source_identities": wind_component_identities,
                    }
        record = runtime.acoustic_plan_record()
        wind_active = (
            (want_sidechain and used_sidechain_wind)
            or (want_wind_background and used_wind_background)
        )
        if record is not None:
            if want_wind_background and used_wind_background:
                record["noise_mode"] = (
                    "wind_background_plus_single" if mix_two else "wind_background"
                )
            elif want_sidechain and used_sidechain_wind:
                record["noise_mode"] = "wind_plus_single" if mix_two else "wind"
            else:
                record["noise_mode"] = "dual" if mix_two else "single"

        if mix_two and wind_active:
            source_start = len(record.get("noise_sources", ())) if record is not None else 0
            source_identity = None
            if image_geometry is None:
                noise_sample = runtime.read_additive_noise(sr, max_dur_sr, rng)
            else:
                source_identity, noise_sample = runtime.read_additive_noise_with_identity(sr, max_dur_sr, rng)
            if reuse_noise:
                runtime.set_noise_single_reuse_cache(
                    (
                        sr,
                        np.ascontiguousarray(noise_sample, dtype=np.float32).copy(),
                        runtime.reuse_noise_source_sidecar(record, source_start),
                    )
                )
            noise_sample = align_noise_sample(noise_sample)
            drawn_snr = float(rng_uniform(rng, cfg.snr_low, cfg.snr_high))
            mix_audit = {} if acoustic_audit is not None else None
            noisy_speech, scaled_n = add_ordinary(
                noisy_speech, noise_sample, drawn_snr, mix_audit, source_identity,
                _record_component_identities(source_start),
            )
            if acoustic_audit is not None:
                acoustic_audit.setdefault("additive", []).append(
                    {"source": "ordinary_after_wind", **mix_audit}
                )
            additive_noise_2d += scaled_n.astype(np.float32, copy=False)
            additive_snr_db_list.append(drawn_snr)
        elif not wind_active:
            mode = "dual" if mix_two else "single"
            record_start = len(record.get("noise_sources", ())) if record is not None else 0
            source_identity = None
            if image_geometry is not None and mode == "single" and not reuse_noise:
                source_identity, noise_sample = runtime.read_additive_noise_with_identity(sr, max_dur_sr, rng)
            else:
                noise_sample = runtime.read_additive_mode(mode, sr, max_dur_sr, rng, reuse_noise)
            noise_sample = align_noise_sample(noise_sample)
            mix_audit = {} if acoustic_audit is not None else None
            noisy_speech, scaled_n = add_ordinary(
                noisy_speech, noise_sample, snr, mix_audit, source_identity,
                _record_component_identities(record_start),
            )
            if acoustic_audit is not None:
                acoustic_audit.setdefault("additive", []).append(
                    {"source": mode, **mix_audit}
                )
            additive_noise_2d += scaled_n.astype(np.float32, copy=False)
            additive_snr_db_list.append(snr)

        if image_geometry is None:
            noisy_speech, additive_noise_2d = runtime.apply_colored_noise(
                noisy_speech,
                mix_reference,
                additive_noise_2d,
                additive_snr_db_list,
                sr,
                rng,
                acoustic_audit,
            )
        elif float(getattr(cfg, "prob_colored_noise", 0.0)) > 0.0:
            # C-3 spatial：合成 colored waveform 作为 noise actor，与 wind_background
            # 走同一 BackgroundSpec 渲染链（diffuse / point_source 由 spec.policy 决定），
            # 在 scale_noise_image 唯一标定点按 spatial SNR 标定；target/direct 监督
            # 轨迹不变。prob=0 时整段跳过、零随机数消耗。
            want_colored = rng_random(rng) < float(cfg.prob_colored_noise)
            if want_colored:
                lo, hi = cfg.colored_noise_snr_range
                colored_snr = float(rng_uniform(rng, float(lo), float(hi)))
                reference_mono = np.asarray(mix_reference)[0]
                colored_audit = {} if acoustic_audit is not None else None
                augmented = colored_noise(
                    reference_mono,
                    sr,
                    colored_snr,
                    colored_snr,
                    cfg.colored_noise_types,
                    rng=rng,
                    audit_out=colored_audit,
                )
                colored_component = (augmented - reference_mono).astype(np.float32)
                colored_sample = align_noise_sample(colored_component[None, :])
                colored_mix_audit = {} if acoustic_audit is not None else None
                noisy_speech, scaled_colored = add_ordinary(
                    noisy_speech,
                    colored_sample,
                    colored_snr,
                    colored_mix_audit,
                    "colored_noise:synthetic",
                )
                additive_noise_2d += scaled_colored.astype(np.float32, copy=False)
                additive_snr_db_list.append(colored_snr)
                used_colored_noise = True
                colored_facts = {
                    "shape": list(scaled_colored.shape),
                    "snr_db": float(colored_snr),
                    "background_policy": policy.value,
                    "color": (
                        str(colored_audit.get("color")) if colored_audit else None
                    ),
                }
                if record is not None:
                    record["colored_enabled"] = True
                    record["colored_snr_db"] = float(colored_snr)
                    record["colored_type"] = colored_facts["color"]
                if acoustic_audit is not None:
                    acoustic_audit["colored_noise"] = {
                        "applied": True,
                        "spatial": True,
                        "waveform_shape": list(scaled_colored.shape),
                        "snr_db": float(colored_snr),
                        "color": colored_facts["color"],
                        "background_policy": policy.value,
                    }

    # side-chain 风噪请求但没有可用风噪样本时，记录实际普通噪声混合的 SNR。
    if want_sidechain and not used_sidechain_wind:
        wn_snr = float(snr)
    if want_wind_background and not used_wind_background and not want_sidechain:
        wn_snr = float(snr)


    record = runtime.acoustic_plan_record()
    source_policies = []
    if record is not None:
        source_policies = [
            source.to_dict()
            for source in record.get("noise_sources", ())
            if hasattr(source, "to_dict")
        ]
    ordered_component_identities = [
        identity
        for facts in image_facts
        for identity in facts.get("component_identities", ())
    ]
    return BackgroundMix(
        noisy_speech, additive_noise_2d, additive_snr_db_list,
        used_sidechain_wind, mix_two, want_sidechain, wn_snr,
        snr_primary_drawn_db, wind_noise_snr_candidate_db, acoustic_audit,
        ({**(
            image_facts[-1] if image_facts else {
                "projection": "absent", "rir_applied": False,
                "background_policy": policy.value,
                "background_shared_scene": bool(background_spec.shared_scene),
            }
        ),
          "component_identities": ordered_component_identities,
          "ordinary_components": image_facts,
          "wind_background_applied": bool(used_wind_background),
          "wind_background": wind_background_facts,
          "colored_noise_applied": bool(used_colored_noise),
          "colored_noise": (colored_facts or None),
          "wind_sidechain_applied": bool(used_sidechain_wind),
          "wind_sidechain": (wind_sidechain_facts or None),
          "source_policies": source_policies} if image_geometry is not None else None),
        used_wind_background,
        used_colored_noise,
    )
