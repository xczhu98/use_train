"""AEC 模型构造参数与频谱输入对齐合同。"""

from __future__ import annotations

import pytest
import torch

from tasks.aec.models.discriminative.cagcrn import CAGCRN
from tasks.aec.models.discriminative.localvqe import LocalVQE

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_fft", "512"),
        ("hop_length", 256.0),
        ("win_length", True),
        ("sample_rate", 16000.0),
        ("window", 123),
        ("center", 1),
    ],
)
def test_cagcrn_rejects_coercible_constructor_parameter_types(name, value):
    with pytest.raises(TypeError):
        CAGCRN(**{name: value})


@pytest.mark.parametrize("model_class", [LocalVQE])
def test_vqe_rejects_boolean_compression(model_class):
    with pytest.raises(TypeError, match="compression"):
        model_class(compression=True)


@pytest.mark.parametrize("model_class", [LocalVQE])
def test_vqe_rejects_spectra_dtype_different_from_model_parameters(model_class):
    model = model_class().eval()
    n_freqs = model.n_fft // 2 + 1
    spectra = torch.zeros(1, n_freqs, 1, 2, dtype=torch.float64)

    with pytest.raises(TypeError, match="model parameters"):
        model.forward_spectra(spectra, spectra)


@pytest.mark.parametrize("model_class", [LocalVQE])
def test_vqe_rejects_spectra_device_different_from_model_parameters(model_class):
    model = model_class().eval()
    n_freqs = model.n_fft // 2 + 1
    spectra = torch.zeros(1, n_freqs, 1, 2, device="meta")

    with pytest.raises(ValueError, match="model parameters"):
        model.forward_spectra(spectra, spectra)
