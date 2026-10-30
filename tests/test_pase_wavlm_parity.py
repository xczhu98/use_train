"""Numerical parity test: ported WavLM vs. the source repo's vendored WavLM.

tasks/se/models/pase_family/common/wavlm/WavLM.py is a structural port of
${PASE_SOURCE_REPO}/pase/models/wavlm/WavLM.py (same class/function set,
reformatted). This test verifies they are numerically equivalent -- not just
structurally similar -- by loading the same state_dict into both and comparing
outputs on identical input. Skipped if PASE_SOURCE_REPO does not point to an
available source checkout (e.g. CI without the external repository).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# 审计 T-02: 源码路径经环境变量 PASE_SOURCE_REPO 配置；未配置时使用
# ``<external>`` 占位并跳过，避免把任一开发机 checkout 写入仓库。
_SOURCE_REPO = Path(os.environ.get("PASE_SOURCE_REPO", "<external>"))
_SOURCE_PASE_ROOT = _SOURCE_REPO / "pase"

pytestmark = [pytest.mark.external, pytest.mark.skipif(
    not _SOURCE_PASE_ROOT.is_dir(),
    reason=(
        f"source repo not found at {_SOURCE_REPO}; set PASE_SOURCE_REPO env var "
        f"to the local checkout of cisco-open/pase (审计 T-02)"
    ),
)]


def _tiny_cfg_kwargs() -> dict:
    return {
        "encoder_layers": 2,
        "encoder_embed_dim": 32,
        "encoder_ffn_embed_dim": 64,
        "encoder_attention_heads": 2,
        "layer_norm_first": True,
        "gru_rel_pos": True,
        "relative_position_embedding": True,
    }


def test_wavlm_forward_parity():
    from tasks.se.models.pase_family.common.wavlm.WavLM import (
        WavLM as PortedWavLM,
        WavLMConfig as PortedWavLMConfig,
    )

    if str(_SOURCE_PASE_ROOT) not in sys.path:
        sys.path.insert(0, str(_SOURCE_PASE_ROOT))
    from models.wavlm.WavLM import WavLM as SourceWavLM, WavLMConfig as SourceWavLMConfig  # noqa: E402

    torch.manual_seed(0)
    source_cfg = SourceWavLMConfig(_tiny_cfg_kwargs())
    source_model = SourceWavLM(source_cfg)
    source_model.eval()

    ported_cfg = PortedWavLMConfig(_tiny_cfg_kwargs())
    ported_model = PortedWavLM(ported_cfg)
    ported_model.eval()

    source_keys = set(source_model.state_dict().keys())
    ported_keys = set(ported_model.state_dict().keys())
    assert source_keys == ported_keys, (
        f"state_dict key mismatch.\n"
        f"Only in source: {sorted(source_keys - ported_keys)}\n"
        f"Only in ported: {sorted(ported_keys - source_keys)}"
    )

    # Copy source weights into the ported model so both start from identical
    # parameters (rather than relying on identical RNG consumption order).
    ported_model.load_state_dict(source_model.state_dict(), strict=True)

    torch.manual_seed(1)
    wav = torch.randn(2, 3200)  # short waveform, well past pad_to_multiple's 320-sample stride

    with torch.no_grad():
        source_out = source_model.extract_features(wav, output_layer=1, mask=False)[0]
        ported_out = ported_model.extract_features(wav, output_layer=1, mask=False)[0]

    for key in ("x", "layer_reps"):
        if key not in source_out:
            continue
        s_val = source_out[key]
        p_val = ported_out[key]
        if isinstance(s_val, list):
            assert len(s_val) == len(p_val)
            for s_t, p_t in zip(s_val, p_val):
                torch.testing.assert_close(s_t, p_t, atol=1e-5, rtol=1e-4)
        else:
            torch.testing.assert_close(s_val, p_val, atol=1e-5, rtol=1e-4)
