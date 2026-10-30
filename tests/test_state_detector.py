"""Unit tests for the reusable AEC state detector head."""

from __future__ import annotations
import pytest

import unittest

import torch

from tasks.aec.models.discriminative.dbgnet import DBGNet
from tasks.aec.models.state_detector import (
    AECStateDetector,
    AECStateDetectorHead,
    AECStateFeatureExtractor,
)


class TestAECStateFeatureExtractor(unittest.TestCase):
    """Tests for AECStateFeatureExtractor."""

    def test_output_shape(self):
        """Output has shape (B, T, output_size)."""
        extractor = AECStateFeatureExtractor(input_size=20, output_size=64)
        mic = torch.randn(2, 8, 10, 32)  # (B, C, F, T)
        far = torch.randn(2, 8, 10, 32)
        out = extractor(mic, far)
        self.assertEqual(out.shape, (2, 32, 64))

    def test_different_input_sizes(self):
        """Extractor works with different frequency sizes."""
        for freq_size in (8, 16, 32):
            extractor = AECStateFeatureExtractor(input_size=2 * freq_size, output_size=32)
            mic = torch.randn(1, 4, freq_size, 10)
            far = torch.randn(1, 4, freq_size, 10)
            out = extractor(mic, far)
            self.assertEqual(out.shape, (1, 10, 32))

    def test_t_f_layout(self):
        """Extractor supports (B, C, T, F) T_F layout."""
        extractor = AECStateFeatureExtractor(
            input_size=40, output_size=32, spatial_layout="T_F"
        )
        mic = torch.randn(1, 4, 10, 20)  # (B, C, T, F)
        far = torch.randn(1, 4, 10, 20)
        out = extractor(mic, far)
        self.assertEqual(out.shape, (1, 10, 32))

    def test_invalid_layout_raises(self):
        """Invalid spatial_layout raises ValueError."""
        with self.assertRaises(ValueError):
            AECStateFeatureExtractor(input_size=20, spatial_layout="invalid")


class TestAECStateDetector(unittest.TestCase):
    """Tests for AECStateDetector classifier."""

    def test_output_shape(self):
        """Output logits have shape (B, T, nclass)."""
        detector = AECStateDetector(input_size=256, hidden_size=128, nclass=8)
        z = torch.randn(2, 20, 256)
        out = detector(z)
        self.assertEqual(out.shape, (2, 20, 8))

    def test_nclass_3(self):
        """3-class head works."""
        detector = AECStateDetector(input_size=64, hidden_size=32, nclass=3)
        z = torch.randn(1, 10, 64)
        out = detector(z)
        self.assertEqual(out.shape, (1, 10, 3))


class TestAECStateDetectorHead(unittest.TestCase):
    """Tests for the combined AECStateDetectorHead."""

    def test_multi_resolution_forward(self):
        """Head accepts multiple resolution feature pairs."""
        head = AECStateDetectorHead([32, 16, 8], nclass=3)
        pairs = [
            (torch.randn(2, 64, 32, 20), torch.randn(2, 64, 32, 20)),
            (torch.randn(2, 128, 16, 20), torch.randn(2, 128, 16, 20)),
            (torch.randn(2, 256, 8, 20), torch.randn(2, 256, 8, 20)),
        ]
        logits = head(pairs)
        self.assertEqual(logits.shape, (2, 20, 3))

    def test_single_resolution(self):
        """Head works with a single feature pair."""
        head = AECStateDetectorHead([16], nclass=8)
        pairs = [(torch.randn(1, 32, 16, 10), torch.randn(1, 32, 16, 10))]
        logits = head(pairs)
        self.assertEqual(logits.shape, (1, 10, 8))

    def test_multi_resolution_different_t_upsample(self):
        """Head interpolates lower-resolution embeddings to the finest T.

        Simulates DBGNet encoder output where stride=2 downsampling halves the
        time dimension at each level: T values are [20, 10, 5, 3].
        The F.interpolate upsampling path must correctly align them to T_max=20.
        """
        head = AECStateDetectorHead([32, 16, 8, 4], nclass=3)
        pairs = [
            (torch.randn(2, 32, 32, 20), torch.randn(2, 32, 32, 20)),  # T=20 (finest)
            (torch.randn(2, 64, 16, 10), torch.randn(2, 64, 16, 10)),  # T=10
            (torch.randn(2, 128, 8, 5), torch.randn(2, 128, 8, 5)),   # T=5
            (torch.randn(2, 192, 4, 3), torch.randn(2, 192, 4, 3)),   # T=3 (coarsest)
        ]
        logits = head(pairs)
        self.assertEqual(logits.shape, (2, 20, 3))  # T_max=20

    def test_mismatched_pair_count_raises(self):
        """Mismatch between feature pairs and configured sizes raises."""
        head = AECStateDetectorHead([16, 8], nclass=3)
        with self.assertRaises(ValueError):
            head([(torch.randn(1, 4, 16, 10), torch.randn(1, 4, 16, 10))])

    def test_empty_feature_sizes_raises(self):
        """Empty feature_freq_sizes raises ValueError."""
        with self.assertRaises(ValueError):
            AECStateDetectorHead([])


class TestAECStateDetectorHeadLayouts(unittest.TestCase):
    """Tests for AECStateDetectorHead with different spatial layouts."""

    def test_f_t_layout(self):
        """Head works with (B, C, F, T) DBGNet-style features."""
        head = AECStateDetectorHead([16, 8], nclass=3, spatial_layout="F_T")
        pairs = [
            (torch.randn(2, 32, 16, 20), torch.randn(2, 32, 16, 20)),
            (torch.randn(2, 64, 8, 20), torch.randn(2, 64, 8, 20)),
        ]
        logits = head(pairs)
        self.assertEqual(logits.shape, (2, 20, 3))

    def test_t_f_layout(self):
        """Head works with (B, C, T, F) T_F-layout features."""
        head = AECStateDetectorHead([16, 8], nclass=3, spatial_layout="T_F")
        pairs = [
            (torch.randn(2, 32, 20, 16), torch.randn(2, 32, 20, 16)),
            (torch.randn(2, 64, 20, 8), torch.randn(2, 64, 20, 8)),
        ]
        logits = head(pairs)
        self.assertEqual(logits.shape, (2, 20, 3))


class TestDBGNetStateDetectorBackwardCompat(unittest.TestCase):
    """Ensure DBGNet state-detector state-dict keys are unchanged."""

    def test_state_dict_keys_with_state_detector(self):
        """State-detector keys use the AECStateDetectorHead naming scheme."""
        model = DBGNet(use_state_detector=True, state_nclass=8)
        keys = sorted(k for k in model.state_dict().keys() if "state_head" in k)
        expected_prefixes = [
            "state_head.detector.linear.bias",
            "state_head.detector.linear.weight",
            "state_head.detector.rnn.",
            "state_head.feature_extractors.0.",
            "state_head.feature_extractors.1.",
            "state_head.feature_extractors.2.",
            "state_head.feature_extractors.3.",
        ]
        for prefix in expected_prefixes:
            self.assertTrue(
                any(k.startswith(prefix) for k in keys),
                f"Missing state-dict keys for prefix {prefix!r}",
            )

    def test_forward_outputs_state_probs_when_enabled(self):
        """DBGNet outputs state_probs when use_state_detector=True."""
        model = DBGNet(use_state_detector=True, state_nclass=3).eval()
        mic = torch.randn(1, 257, 20, 2)
        far = torch.randn(1, 257, 20, 2)
        with torch.inference_mode():
            out = model({"mic_spectra": mic, "farend_spectra": far})
        self.assertIn("state_probs", out)
        self.assertEqual(out["state_probs"].shape, (1, 20, 3))

    def test_forward_no_state_probs_when_disabled(self):
        """DBGNet does not output state_probs when use_state_detector=False."""
        model = DBGNet(use_state_detector=False).eval()
        mic = torch.randn(1, 257, 20, 2)
        far = torch.randn(1, 257, 20, 2)
        with torch.inference_mode():
            out = model({"mic_spectra": mic, "farend_spectra": far})
        self.assertNotIn("state_probs", out)

    def test_forward_320_stft_variant(self):
        """DBGNet supports 320/160 STFT with erb_low_bins=41/erb_bands=40."""
        model = DBGNet(
            n_fft=320,
            hop_length=160,
            win_length=320,
            erb_low_bins=41,
            erb_bands=40,
            use_state_detector=True,
            state_nclass=3,
        ).eval()
        mic = torch.randn(1, 161, 20, 2)
        far = torch.randn(1, 161, 20, 2)
        with torch.inference_mode():
            out = model({"mic_spectra": mic, "farend_spectra": far})
        self.assertIn("state_probs", out)
        self.assertEqual(out["state_probs"].shape, (1, 20, 3))
        self.assertEqual(out["nearend_estimate_spectra"].shape, (1, 161, 20, 2))

    def test_forward_640_stft_32k_variant(self):
        """32 kHz DBGNet preserves 321 bins and emits three-class state logits."""
        model = DBGNet(
            n_fft=640,
            hop_length=320,
            win_length=640,
            sample_rate=32000,
            erb_low_bins=81,
            erb_bands=80,
            use_state_detector=True,
            state_nclass=3,
        ).eval()
        mic = torch.randn(1, 321, 20, 2)
        far = torch.randn(1, 321, 20, 2)
        with torch.inference_mode():
            out = model({"mic_spectra": mic, "farend_spectra": far})
        self.assertEqual(model._freq_sizes, (161, 161, 80, 39, 19))
        self.assertEqual(out["nearend_estimate_spectra"].shape, (1, 321, 20, 2))
        self.assertEqual(out["state_probs"].shape, (1, 20, 3))


if __name__ == "__main__":
    unittest.main()

pytestmark = [pytest.mark.unit]
