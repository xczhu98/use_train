"""ordinary/music 背景源的共享选择与读取；task 注入资源池和读盘函数。"""

from __future__ import annotations

from core.data.noise_source_policy import NoiseApplicationRole
from core.data.simulation_utils import read_source_scp, rng_random, rng_uniform


def initialize_music_pool(owner, music_scp, *, extract_embedded):
    owner.music_source, owner.music_uids, owner.music_flt, owner._music_lengths = (
        read_source_scp(music_scp) if music_scp is not None else ({}, {}, {}, {})
    )
    if extract_embedded:
        owner._extract_jsonl_role(role="music", prefix="jsonl_music_",
            target_source=owner.music_source, target_uids=owner.music_uids,
            target_flat=owner.music_flt, target_lengths=owner._music_lengths)
    owner._build_noise_sampling_state()
    ratio = float(owner.sim_cfg.music_ratio)
    if ratio > 0 and not owner.music_flt:
        raise ValueError("music_ratio > 0 but no music entries available; provide non-empty music_scp or embedded music entries")
    if not owner.noise_flt and not (ratio == 1 and owner.music_flt):
        raise ValueError("ordinary noise pool is empty after Music/Wind role routing; use music_ratio=1 with a music pool")
    owner._music_routing_enabled = True


def select_source(owner, sr, rng, *, exclude_category_id=None, exclude_uid=None, force_ordinary=False):
    if not force_ordinary and rng_random(rng) < float(owner.sim_cfg.music_ratio):
        uid = owner._select_sample(sr, owner.music_source, rng=rng)
        return ("music", uid, owner.music_flt[uid]) if uid is not None else None
    if owner.balanced_noise_sampling_enabled:
        uid = owner._select_balanced_additive_noise_uid(sr, rng, exclude_category_id=exclude_category_id, exclude_uid=exclude_uid)
    elif exclude_category_id is not None:
        uid = owner._select_natural_different_category_uid(sr, rng, exclude_category_id=exclude_category_id, exclude_uid=exclude_uid or "")
    else:
        uid = owner._select_sample(sr, owner.noise_source, rng=rng)
    return ("ordinary", uid, owner.noise_flt[uid]) if uid is not None else None


def select_dual_sources(owner, sr, rng, *, selector=None):
    selector = selector or (lambda *args, **kwargs: select_source(owner, *args, **kwargs))
    first = selector(sr, rng)
    if first is None:
        return ()
    family, uid, _ = first
    category = (owner._noise_category_id_by_uid[uid]
                if family == "ordinary" and owner._ordinary_noise_categories_complete() else None)
    second = selector(sr, rng, exclude_category_id=category, exclude_uid=uid, force_ordinary=family == "music")
    return (first,) if second is None else (first, second)


def read_source(owner, source, sr, max_duration, rng, *,
                application_role=NoiseApplicationRole.LAYER_C_ADDITIVE,
                strict_load=None, spatial=None):
    family, uid, path = source
    crop = {}
    if spatial is None:
        spatial = (
            application_role == NoiseApplicationRole.LAYER_C_ADDITIVE
            and bool(getattr(owner.sim_cfg, "spatial_enabled", False))
        )
    sample = owner.resource_store.read_noise(
        path,
        sr,
        max_duration if max_duration is not None else owner._max_duration_for_sr(sr),
        spatial=bool(spatial),
        strict=owner.strict_load if strict_load is None else strict_load,
        rng=rng,
        crop_record=crop,
    )
    owner._record_noise_source(family=family, uid=uid, path=path, crop=crop.get("crop"), application_role=application_role)
    record = owner._acoustic_plan_record()
    if record is not None:
        record["noise_assets"].append(sample.copy())
    return sample


def read_one(owner, sr, max_duration, rng):
    source = select_source(owner, sr, rng)
    if source is None:
        raise RuntimeError(f"No additive noise source for sr={sr}.")
    return read_source(owner, source, sr, max_duration, rng)


def read_dual(owner, sr, max_duration, rng, *, selector=None, spatial=None):
    sources = (selector(sr, rng) if selector else select_dual_sources(owner, sr, rng))
    if not sources:
        raise RuntimeError(f"No additive noise source for sr={sr}.")
    reader = lambda source, sample_rate, duration, source_rng: read_source(
        owner,
        source,
        sample_rate,
        duration,
        source_rng,
        spatial=spatial,
    )
    noise = reader(sources[0], sr, max_duration, rng)
    if len(sources) == 1:
        return noise
    noise2 = owner._extend_noise_to_length(reader(sources[1], sr, max_duration, rng), noise.shape[-1])
    ratio = rng_uniform(rng, *owner.sim_cfg.noise_mix_ratio_range)
    record = owner._acoustic_plan_record()
    if record is not None:
        record["mix_ratio"] = float(ratio)
    return noise + ratio * noise2[:, :noise.shape[-1]]
