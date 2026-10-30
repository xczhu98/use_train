import pytest
import torch

from tasks.ssl.models import DPIPD, FixedArrayIPDNet, IPDFeatureContract, build_model


@pytest.mark.unit
def test_fixed_array_ipdnet_synthetic_contract():
    spectra = torch.randn(2, 3, 33, 10, dtype=torch.complex64)
    features = IPDFeatureContract().extract(spectra)
    assert features.shape == (2, 10, 33, 4)

    model = build_model({"type": "FixedArrayIPDNet", "args": {"input_size": 2, "hidden_size": 8, "n_fft": 64}})
    output = model(features)
    assert output.shape == (2, 10, 33, 2, 2, 2)
    assert torch.isfinite(output).all()

    templates = DPIPD(
        torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]]),
        (2, 5),
        n_fft=64,
    )
    spatial = templates(features)
    assert spatial.shape == (2, 10, 2, 5)


@pytest.mark.unit
def test_ipd_contract_rejects_invalid_layout():
    with pytest.raises(ValueError, match="at least two"):
        IPDFeatureContract().extract(torch.randn(1, 1, 33, 4, dtype=torch.complex64))
    with pytest.raises(ValueError, match="last dimension"):
        IPDFeatureContract().extract(torch.randn(1, 2, 33, 4, 3))
