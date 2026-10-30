"""cue 校验保持可变输入 fail-fast，标量同步次数不随目标数增长。"""

import pytest
import torch
from torch.profiler import profile, ProfilerActivity

from tasks.tse.cue_contract import TSECue, TSECueSpec, TSECueSpecs, TSECues, collate_tse_cues

def _batch(rows):
    specs = TSECueSpecs(speaker_waveform=TSECueSpec("waveform", (16,)))
    cue = TSECues(speaker_waveform=TSECue("waveform", torch.ones(8), True, 8, None))
    return collate_tse_cues([cue] * rows, specs), specs


@pytest.mark.unit
def test_validation_scalar_sync_count_is_independent_of_target_rows():
    counts = []
    for rows in (1, 8, 32, 128):
        batch, specs = _batch(rows)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            batch.validate(specs, rows)
        counts.append(sum(event.count for event in prof.key_averages()
                          if event.key == "aten::_local_scalar_dense"))
    print(f"cue validation scalar counts R=1/8/32/128: {counts}")
    assert max(counts) <= 4


@pytest.mark.unit
def test_revalidation_detects_inplace_tail_mutation():
    batch, specs = _batch(8)
    batch.validate(specs, 8)
    batch.speaker_waveform.value[5, 12] = 1
    with pytest.raises(ValueError, match="after valid length"):
        batch.validate(specs, 8)


@pytest.mark.unit
def test_collate_validates_each_input_and_materialized_cue_once(monkeypatch):
    from tasks.tse.cue_contract import TSECueBatch
    from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn
    counts = {"input": 0, "batch": 0}
    original_input, original_batch = TSECue.validate, TSECueBatch.validate

    def input_validate(self, *args):
        counts["input"] += 1
        return original_input(self, *args)

    def batch_validate(self, *args):
        counts["batch"] += 1
        return original_batch(self, *args)

    monkeypatch.setattr(TSECue, "validate", input_validate)
    monkeypatch.setattr(TSECueBatch, "validate", batch_validate)
    specs = TSECueSpecs(speaker_waveform=TSECueSpec("waveform", (16,)))
    cue = TSECues(speaker_waveform=TSECue("waveform", torch.ones(8), True, 8, None))
    samples = [make_tse_sample(key=str(i), mixture=torch.ones(32), sr=16000, length=32,
                              targets=[make_tse_target(target=torch.ones(32), cues=cue, speaker_id=None)])
               for i in range(8)]
    tse_collate_fn(samples, cue_specs=specs)
    assert counts == {"input": 8, "batch": 1}


@pytest.mark.parametrize("slot,representation,shape", [
    ("speaker_waveform", "waveform", (6,)),
    ("speaker_embedding", "embedding", (3,)),
    ("spatial", "static", (2,)),
    ("spatial", "track", (2, 4)),
    ("visual", "features", (3, 4)),
    ("visual", "raw", (2, 3, 3, 4)),
    ("textual", "phoneme_ids", (6,)),
])
@pytest.mark.unit
def test_all_representations_reject_mutated_missing_rows(slot, representation, shape):
    spec = TSECueSpec(representation, shape)
    specs = TSECueSpecs(**{slot: spec})
    temporal = spec.temporal_axis is not None
    cue = TSECues(**{slot: TSECue(representation, None, False, 0 if temporal else None, None)})
    batch = collate_tse_cues([cue], specs)
    batch.validate(specs, 1)
    getattr(batch, slot).value.flatten()[0] = 1
    with pytest.raises(ValueError, match="zero value placeholders"):
        batch.validate(specs, 1)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_validation_sync_budget_and_measurement():
    from statistics import median
    from time import perf_counter
    results = []
    for rows in (1, 8, 32, 128):
        batch, specs = _batch(rows)
        batch = batch.to("cuda")
        for _ in range(10):
            batch.validate(specs, rows)
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = perf_counter()
            batch.validate(specs, rows)
            torch.cuda.synchronize()
            times.append((perf_counter() - start) * 1000)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            batch.validate(specs, rows)
        scalar_count = sum(event.count for event in prof.key_averages()
                           if event.key == "aten::_local_scalar_dense")
        d2h_count = sum(event.count for event in prof.key_averages() if "DtoH" in event.key)
        results.append((rows, scalar_count, d2h_count, round(median(times), 3)))
        assert scalar_count == 0
        assert d2h_count == 1
    print(f"CUDA cue validation (rows,scalar,D2H,median_ms): {results}")
