"""DeepFilterNet 构造参数合同回归测试。"""

from __future__ import annotations

import pytest

from tasks.se.models.discriminative.deepfilternet import DeepFilterNet


pytestmark = pytest.mark.unit


def test_deepfilternet_consumes_runtime_stft_context():
    model = DeepFilterNet(
        n_fft=320,
        hop_length=160,
        win_length=320,
        window="hann",
        center=True,
    )

    assert model.n_fft == 320


def test_deepfilternet_rejects_unknown_constructor_kwargs():
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        DeepFilterNet(n_frt=320)
