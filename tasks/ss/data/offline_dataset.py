"""Offline mixture/source dataset for Speech Separation."""

from __future__ import annotations

import random
import math
from pathlib import Path

import numpy as np

from torch.utils.data import Dataset
import torch

from core.data.offline_dataset import validate_loaded_audio_metadata
from core.data.offline_multitrack import (
    ensure_finite,
    ensure_mono,
    load_audio,
    resampled_num_frames,
)
from core.data.separation_manifest import SeparationRecord, parse_separation_manifest
from core.data.simulation.array_profiles import resolve_microphone_array
from core.data.simulation.rir_provider import build_spatial_rir_provider
from core.data.simulation_configs import SimulationConfigs
from core.data.spatial import render_spatial_scene
from tasks.ss.data.config import build_ss_dataset_contract
from core.data.errors import RecoverableResourceError, is_recoverable_resource_error


class SSOfflineDataset(Dataset):
    """Load fixed-source-count mixture/source records from strict JSONL.

    Random training crops read the shared epoch/base-seed token, so persistent
    DataLoader workers observe every ``set_epoch`` update instead of freezing the
    first epoch's crop positions.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        num_sources: int,
        rir_scp: str | None = None,
        target_sr: int | None = None,
        strict_load: bool = True,
        placeholder_num_samples: int = 1024,
        max_duration_seconds: float | None = None,
        seed: int = 0,
        training: bool = False,
        simulation_cfg: dict | None = None,
        simulation_cfg_strict: bool = True,
    ) -> None:
        if type(num_sources) is not int or num_sources < 1:
            raise ValueError("num_sources must be an exact positive int")
        if target_sr is not None and (type(target_sr) is not int or target_sr <= 0):
            raise ValueError("target_sr must be an exact positive int when provided")
        if type(strict_load) is not bool:
            raise TypeError("strict_load must be an exact bool")
        if type(placeholder_num_samples) is not int or placeholder_num_samples <= 0:
            raise ValueError("placeholder_num_samples must be an exact positive int")
        if type(seed) is not int:
            raise TypeError("seed must be an exact int")
        if type(training) is not bool:
            raise TypeError("training must be an exact bool")
        if type(simulation_cfg_strict) is not bool:
            raise TypeError("simulation_cfg_strict must be an exact bool")
        if max_duration_seconds is not None and (
            isinstance(max_duration_seconds, bool)
            or not isinstance(max_duration_seconds, (int, float))
            or not math.isfinite(max_duration_seconds)
            or max_duration_seconds <= 0
        ):
            raise ValueError(
                "max_duration_seconds must be a finite positive number when provided"
            )
        self.records: tuple[SeparationRecord, ...] = parse_separation_manifest(manifest)
        self.num_sources = num_sources
        self.target_sr = target_sr
        self.strict_load = strict_load
        self.placeholder_num_samples = placeholder_num_samples
        self.max_duration_seconds = (
            float(max_duration_seconds) if max_duration_seconds is not None else None
        )
        self.seed = seed
        self._base_seed = 0
        self.training = training
        self.epoch = 0
        self._worker_rng_state = torch.zeros(2, dtype=torch.int64).share_memory_()
        self._worker_last_seen_epoch = -1
        self.sim_cfg = SimulationConfigs.from_dict(
            simulation_cfg or {}, strict=simulation_cfg_strict
        )
        self.spatial = self.sim_cfg.spatial_enabled
        if self.spatial and self.sim_cfg.rir_backend != "offline":
            raise ValueError("SS offline spatial reconstruction requires rir_backend='offline'")
        self._spatial_rir_provider = (
            build_spatial_rir_provider(self.sim_cfg, rir_scp=rir_scp)
            if self.spatial
            else None
        )
        contract_sample_rate = target_sr or self.records[0].sample_rate
        self.contract = build_ss_dataset_contract(
            sample_rate=contract_sample_rate,
            num_sources=num_sources,
        )
        if any(len(record.sources) != num_sources for record in self.records):
            raise ValueError("all SS records must match ss.num_sources")

    def __len__(self) -> int:
        return len(self.records)

    def get_srs(self) -> list[int]:
        return [self.target_sr or record.sample_rate for record in self.records]

    def get_source_length(self) -> list[int]:
        lengths = []
        for record in self.records:
            sample_rate = self.target_sr or record.sample_rate
            length = (
                resampled_num_frames(record.length, record.sample_rate, sample_rate)
                if sample_rate != record.sample_rate
                else record.length
            )
            if self.max_duration_seconds is not None:
                length = min(length, int(round(self.max_duration_seconds * sample_rate)))
            lengths.append(length)
        return lengths

    def set_epoch(self, epoch: int, base_seed: int | None = None) -> None:
        if type(epoch) is not int:
            raise TypeError("epoch must be an exact int")
        if base_seed is not None:
            if type(base_seed) is not int:
                raise TypeError("base_seed must be an exact int when provided")
            self._base_seed = base_seed
            self._worker_rng_state[1] = base_seed
        self.epoch = epoch
        self._worker_rng_state[0] = epoch

    def _placeholder(self, index: int, reason: str) -> dict:
        record = self.records[index]
        sample_rate = int(self.target_sr or record.sample_rate)
        length = int(self.placeholder_num_samples)
        if self.max_duration_seconds is not None:
            length = min(length, max(1, int(round(self.max_duration_seconds * sample_rate))))
        zeros = torch.zeros(length, dtype=torch.float32)
        if self.spatial:
            resolved = resolve_microphone_array(
                array_profile=self.sim_cfg.array_profile,
                custom_geometry=self.sim_cfg.custom_geometry,
                reference_channel=self.sim_cfg.reference_channel,
            )
            if resolved is None:
                raise RuntimeError("spatial SS placeholder requires resolved microphone geometry")
            mixture = torch.zeros((resolved.channels, length), dtype=torch.float32)
        else:
            mixture = zeros.clone()
        return {
            "uid": record.uid,
            "mixture": mixture,
            "sources": torch.zeros(
                (self.num_sources, length), dtype=torch.float32
            ),
            "length": length,
            "sr": sample_rate,
            "source_ids": ["absent"] * self.num_sources,
            "placeholder": reason,
        }

    def __getitem__(self, index: int) -> dict:
        try:
            return self._load_item(index)
        except Exception as exc:
            if self.strict_load or not is_recoverable_resource_error(exc):
                raise
            return self._placeholder(index, f"{type(exc).__name__}: {exc}")

    def _load_item(self, index: int) -> dict:
        record = self.records[index]
        mixture, mixture_sr = load_audio(str(record.mixture))
        mixture = ensure_finite(
            ensure_mono(mixture), strict=True, track="mixture", sample=record.uid
        )
        validate_loaded_audio_metadata(
            mixture,
            mixture_sr,
            expected_sr=record.sample_rate,
            expected_length=record.length,
            source=f"SS mixture {record.uid}",
        )
        source_wavs = []
        for source in record.sources:
            waveform, source_sr = load_audio(str(source.path))
            waveform = ensure_finite(
                ensure_mono(waveform),
                strict=True,
                track=f"source:{source.source_id}",
                sample=record.uid,
            )
            if source_sr != mixture_sr:
                raise ValueError(
                    f"{record.uid}: source sample rate {source_sr} does not match mixture {mixture_sr}"
                )
            validate_loaded_audio_metadata(
                waveform,
                source_sr,
                expected_sr=record.sample_rate,
                expected_length=source.length,
                source=f"SS source {record.uid}/{source.source_id}",
            )
            if source.length != record.length:
                raise ValueError(
                    f"{record.uid}: source {source.source_id} declared length "
                    f"{source.length} does not match mixture length {record.length}"
                )
            source_wavs.append(waveform)
        if self.spatial:
            source_uids = tuple(source.source_id for source in record.sources)
            rir_result = self._spatial_rir_provider.get_rirs(
                mixture_sr,
                len(source_uids),
                None,
                source_uids=source_uids,
            )
            rendered = render_spatial_scene(
                np.stack([source[0].numpy() for source in source_wavs]),
                rir_result.canonical_rirs,
                early_boundary=max(0, int(round(self.sim_cfg.early_rir_sec * mixture_sr))),
                reference_channel=self.sim_cfg.reference_channel,
                num_samples=record.length,
            )
            mixture = torch.from_numpy(rendered.mixture.copy())
            source_wavs = [torch.from_numpy(row.copy()) for row in rendered.reference_tracks]
        if self.target_sr is not None and mixture_sr != self.target_sr:
            from core.data.offline_multitrack import resample

            mixture = resample(mixture, mixture_sr, self.target_sr)
            source_wavs = [resample(wav, mixture_sr, self.target_sr) for wav in source_wavs]
            mixture_sr = self.target_sr
        if self.target_sr is not None:
            expected_length = resampled_num_frames(
                record.length, record.sample_rate, self.target_sr
            )
        else:
            expected_length = record.length
        actual_lengths = [mixture.shape[-1], *(source.shape[-1] for source in source_wavs)]
        if any(length != expected_length for length in actual_lengths):
            raise ValueError(
                f"{record.uid}: resampled waveform length mismatch; "
                f"expected={expected_length}, actual={actual_lengths}"
            )
        length = expected_length
        start = 0
        if self.max_duration_seconds is not None:
            crop_length = int(round(self.max_duration_seconds * mixture_sr))
            if length > crop_length:
                if self.training:
                    rng = random.Random(
                        self.seed * 1_000_003
                        + int(self._worker_rng_state[1]) * 100_003
                        + int(self._worker_rng_state[0]) * 10_007
                        + int(index) * 31
                    )
                    start = rng.randint(0, length - crop_length)
                length = crop_length
        return {
            "uid": record.uid,
            "mixture": (
                mixture[:, start : start + length].contiguous()
                if self.spatial else mixture[0, start : start + length].contiguous()
            ),
            "sources": [
                (source[:, start : start + length].contiguous() if self.spatial and source.ndim == 2
                 else source[start : start + length].contiguous() if self.spatial
                 else source[0, start : start + length].contiguous())
                for source in source_wavs
            ],
            "length": int(length),
            "sr": int(mixture_sr),
            "source_ids": [source.source_id for source in record.sources],
        }
