"""SE 模型 P2/P3 审计条目的回归测试。"""

import pytest
import torch

pytestmark = [pytest.mark.unit]


def test_zipenhancer_bypass_layers_hold_independent_parameters():
    from tasks.se.models.discriminative.zipenhancer import ZipEnhancer

    model = ZipEnhancer()
    paths = list(model.ts_conformer.encoders[0].bypass_layers)
    assert len(paths) > 1
    assert len({id(path) for path in paths}) == len(paths)
    assert len({id(path.bypass_scale) for path in paths}) == len(paths)


def test_frcrn_waveform_paths_fail_fast_and_spectrum_path_still_serves():
    from tasks.se.models.discriminative.clearervoice import FRCRN_SE_16K

    model = FRCRN_SE_16K(n_fft=640, hop_length=320)
    with torch.no_grad():
        enhanced = model(torch.zeros(1, 321, 4, 2))
    assert enhanced.shape == (1, 321, 4, 2)

    core = model._net.model
    with pytest.raises(RuntimeError, match="forward_enhance_spec"):
        core(torch.zeros(1, 1600))
    with pytest.raises(RuntimeError, match="forward_enhance_spec"):
        core.inference(torch.zeros(1, 1600))


def test_conv_module_dilated_is_instantiable_after_super_fix():
    from tasks.se.models.modules.clearervoice_impl.mossformer2.conv_module import (
        ConvModule_Dilated,
    )

    block = ConvModule_Dilated(in_channels=8)
    with torch.no_grad():
        out = block(torch.zeros(2, 5, 8))
    assert out.shape == (2, 5, 8)


def test_conv_stft_version_branch_compares_numeric_parts():
    from tasks.se.models.modules.clearervoice_impl.frcrn import conv_stft

    assert conv_stft._version_tuple("1.9.3") < conv_stft._version_tuple("1.10.1")
    assert conv_stft._version_tuple("1.16.0rc1") >= conv_stft._version_tuple("1.10.1")
    assert conv_stft._version_tuple("2.0") == (2, 0, 0)


def test_channel_layer_norm_reports_class_name_on_wrong_rank():
    from tasks.se.models.modules.clearervoice_impl.mossformer2.layer_norm import (
        CLayerNorm,
    )

    with pytest.raises(RuntimeError, match="CLayerNorm"):
        CLayerNorm(4)(torch.zeros(2, 4))
