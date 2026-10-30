"""TSE 推理 adapter 的 canonical speaker cue 绑定测试。"""

import numpy as np
import pytest
from types import SimpleNamespace
from pathlib import Path

from core.inference import FileInferenceItem
from tasks.tse.inference import TSEInferenceAdapter


class _Extractor:
    sample_rate = 16000

    def declared_cue_roles(self):
        return {
            "speaker_waveform": SimpleNamespace(
                required=True, representation="waveform"
            )
        }

    def extract_speech_with_cues(self, mixture, sr, cues):
        return np.ones(32, dtype=np.float32)


@pytest.mark.unit
def test_tse_adapter_binds_mixture_and_canonical_speaker_waveform_role():
    adapter = TSEInferenceAdapter(_Extractor())
    files, declaration = adapter.prepare_item(
        FileInferenceItem(
            "utt",
            {"mixture": "mix.wav", "speaker_waveform": "enroll.wav"},
            "out.wav",
        )
    )
    assert files["speaker_waveform"] == Path("enroll.wav")
    assert declaration == {
        "speaker_waveform": {"kind": "file", "path": "enroll.wav"}
    }
