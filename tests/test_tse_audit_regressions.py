"""TSE 审计修复的回归测试。"""

from __future__ import annotations

import pytest
import torch

from tasks.tse.cue_contract import TSECueSpecs, TSECues
from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn
from tasks.tse.model_io import mask_waveform_tail
from tasks.tse.models.modules.fusion.speech import CrossFuse
from tasks.tse.models.modules.speaker.spk_frontend import TFMapFeature
from tasks.tse.models.modules.speaker.usef import USEF_attentionblock

pytestmark = pytest.mark.unit


def test_mask_waveform_tail_rejects_broadcastable_single_length() -> None:
    waveform = torch.ones(2, 8)
    with pytest.raises(ValueError, match=r"shape \[2\]"):
        mask_waveform_tail(waveform, torch.tensor([4], dtype=torch.int64))


def test_collate_rejects_float_metadata_without_truncation() -> None:
    sample = make_tse_sample(
        key="float-meta",
        mixture=torch.ones(8),
        sr=16000.0,
        length=8,
        targets=[
            make_tse_target(
                target=torch.ones(8),
                cues=TSECues(),
                speaker_id=None,
                spk_label=-1,
            )
        ],
    )
    with pytest.raises(TypeError, match="not an int"):
        tse_collate_fn([sample], cue_specs=TSECueSpecs())


def test_cross_fuse_rejects_single_enrollment_batch_broadcast() -> None:
    layer = CrossFuse(embed_dim=2, atten_dim=2, mix_dim=2, batch_first=True)
    with pytest.raises(ValueError, match="matching non-empty batch"):
        layer(torch.randn(2, 2, 3), torch.randn(1, 2, 4))


def test_tfmap_attention_guard_rejects_pathological_lengths() -> None:
    feature = TFMapFeature({})
    with pytest.raises(ValueError, match="16,000,000"):
        feature.compute(torch.ones(1, 1, 4001), torch.ones(1, 1, 4000))


def test_usef_attention_guard_rejects_pathological_lengths() -> None:
    attention = USEF_attentionblock(2, 1, 1, 1)
    with pytest.raises(ValueError, match="16,000,000"):
        attention(torch.ones(1, 2, 4001, 1), torch.ones(1, 2, 4000, 1))
