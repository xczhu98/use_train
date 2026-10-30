"""自定义 Batch 必须参与 DataLoader 的锁页内存递归。"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import partial

import pytest
import torch
from torch.utils.data._utils.pin_memory import pin_memory

from core.data.source_separation_batch import SSBatch
from core.data.collate import collate_fn
from tasks.aec.data.collate import aec_collate_fn
from tasks.tse.cue_contract import TSECue, TSECueSpec, TSECueSpecs, TSECues
from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn
from tasks.ss.data.collate import ss_collate_fn
from tasks.ssl.data.batch import SSLBatch
from tasks.ssl.data.collate import ssl_collate_fn


def _ss_batch() -> SSBatch:
    return ss_collate_fn([dict(
        mixture=torch.zeros(8), sources=[torch.ones(8), torch.zeros(8)],
        sr=8000, length=8, source_ids=["source-a", "source-b"],
    )])


def _se_batch(flags=True):
    wav = torch.arange(8, dtype=torch.float32)
    sample = (wav, torch.stack((wav, wav + 1)), 8000, 8)
    if flags:
        sample += (True, False, False)
    return collate_fn([sample], allow_spatial_noisy=True)


def _aec_batch(state=True):
    wav = torch.arange(8, dtype=torch.float32)
    sample = (wav, torch.stack((wav, wav + 1)), wav, 8000, 8, wav, True)
    if state:
        sample += (False, True, False, True, torch.tensor([0, 1, -1]))
    return aec_collate_fn([sample])


def _tse_batch():
    specs = TSECueSpecs(speaker_waveform=TSECueSpec("waveform", (4,)))
    cues = TSECues(speaker_waveform=TSECue(
        representation="waveform", value=torch.ones(4), present=True, length=4, timebase=None,
    ))
    item = make_tse_sample(
        key="pin-test", mixture=torch.zeros(8), sr=8000, length=8,
        targets=[make_tse_target(target=torch.ones(8), cues=cues, speaker_id="speaker-a", spk_label=0)],
        provenance={"source": "fixture"},
    )
    return tse_collate_fn([item], cue_specs=specs)


def _ssl_batch() -> SSLBatch:
    return ssl_collate_fn([dict(
        features=torch.zeros(2, 3, 4),
        target_dpipd=torch.zeros(2, 3, 1, 1, 2),
        input_lengths=2,
        output_lengths=2,
        pair_index=torch.tensor([[0, 1]], dtype=torch.int64),
        pair_mask=torch.ones(1, dtype=torch.bool),
        geometry=torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]),
        frame_origin=0,
        time_stride=1,
        doa=torch.zeros(2, 1, 2),
        vad=torch.ones(2, 1),
        sample_rate=8000,
        source_distances=torch.ones(2, 1),
        input_frame_index=torch.tensor([0, 1], dtype=torch.int64),
        output_frame_index=torch.tensor([0, 1], dtype=torch.int64),
        provenance={"backend": "fixture"},
        input_representation="multichannel_stft_real_imag",
        target_representation="dpipd_near_field",
        coordinate_frame="room_xyz_m",
    )])


_FACTORIES = {"se": _se_batch, "aec": _aec_batch, "tse": _tse_batch, "ss": _ss_batch, "ssl": _ssl_batch}


def _tensor_fields(value, prefix=""):
    """测试独立枚举字段，避免与生产显式字段列表共用而漏检。"""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _tensor_fields(getattr(value, field.name), f"{prefix}.{field.name}")
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from _tensor_fields(item, f"{prefix}.{name}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensor_fields(item, f"{prefix}[{index}]")


@pytest.mark.unit
@pytest.mark.parametrize("factory", [
    *_FACTORIES.values(), partial(_se_batch, flags=False), partial(_aec_batch, state=False),
])
def test_batch_pin_memory_recurses_into_all_tensor_fields(monkeypatch, factory):
    calls = []

    def fake_pin(self, *args, **kwargs):
        calls.append(id(self))
        return self.clone()

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin)
    batch = factory()
    batch.validate()
    pinned = pin_memory(batch)
    assert pinned is not batch
    assert type(pinned) is type(batch)
    assert pinned.batch_size == batch.batch_size
    original = dict(_tensor_fields(batch))
    actual = dict(_tensor_fields(pinned))
    assert original.keys() == actual.keys()
    assert sorted(calls) == sorted(id(tensor) for tensor in original.values())
    for name, tensor in original.items():
        assert actual[name] is not tensor
        torch.testing.assert_close(actual[name], tensor)
    pinned.validate()


def _collate_task(items, *, task):
    assert len(items) == 1
    return _FACTORIES[task]()


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("task", _FACTORIES)
@pytest.mark.parametrize("workers", [0, 1])
def test_real_dataloader_pinning_and_cuda_transfer(task, workers):
    """真实 DataLoader -> pin_memory -> CUDA，覆盖 worker pickle 和可选字段。"""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for real pinned memory")
    kwargs = dict(
        batch_size=1, num_workers=workers, pin_memory=True,
        collate_fn=partial(_collate_task, task=task),
    )
    if workers:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True)
    loader = torch.utils.data.DataLoader([0, 1], **kwargs)
    iterator = None
    try:
        for _ in range(2):
            iterator = iter(loader)
            for batch in iterator:
                batch.validate()
                tensors = dict(_tensor_fields(batch))
                assert all(value.is_pinned() for value in tensors.values())
                moved = batch.to("cuda", non_blocking=True)
                assert type(moved) is type(batch)
                assert moved.batch_size == batch.batch_size
                moved.validate()
                for name, value in _tensor_fields(moved):
                    assert value.device.type == "cuda"
                    torch.testing.assert_close(value.cpu(), tensors[name])
                for name in ("provenance", "source_ids", "keys", "speaker_ids", "cue_specs"):
                    if hasattr(batch, name):
                        assert getattr(moved, name) == getattr(batch, name)
    finally:
        if workers and iterator is not None:
            iterator._shutdown_workers()
