"""失败样本不能污染上一个已发布样本的可变快照。"""

import numpy as np
import pytest

from core.data.dynamic_dataset_base import BaseDynamicMixingDataset


pytestmark = pytest.mark.unit


def test_failed_sample_restores_nested_scalar_values_and_container_types():
    dataset = object.__new__(BaseDynamicMixingDataset)
    original = {"gain": 1.0, "facts": ["codec", {"applied": False}], "shape": [1, 8]}
    dataset._last_custom = original
    snapshot = dataset._legacy_last_snapshot()

    original["gain"] = 9.0
    original["facts"][0] = "clipping"
    original["facts"][1]["applied"] = True
    original["shape"] = {"wrong": "type"}
    original["extra"] = "partial"
    dataset._last_unpublished = "partial"
    dataset._restore_legacy_last_snapshot(snapshot)

    assert dataset._last_custom is original
    assert original == {"gain": 1.0, "facts": ["codec", {"applied": False}], "shape": [1, 8]}
    assert not hasattr(dataset, "_last_unpublished")


def test_failed_sample_restores_arrays_without_replacing_compatible_references():
    dataset = object.__new__(BaseDynamicMixingDataset)
    waveform = np.arange(8, dtype=np.float32)
    nested = [waveform, {"count": 1}]
    dataset._last_custom = {"tracks": nested}
    snapshot = dataset._legacy_last_snapshot()
    waveform[:] = -1
    nested[1]["count"] = 8
    nested.append("partial")

    dataset._restore_legacy_last_snapshot(snapshot)

    assert dataset._last_custom["tracks"] is nested
    assert nested[0] is waveform
    np.testing.assert_array_equal(waveform, np.arange(8, dtype=np.float32))
    assert nested[1] == {"count": 1}
    assert len(nested) == 2


@pytest.mark.parametrize("replacement", [np.zeros((2, 4)), np.zeros(8, dtype=np.int16)])
def test_failed_sample_restores_replaced_array_shape_and_dtype(replacement):
    dataset = object.__new__(BaseDynamicMixingDataset)
    dataset._last_custom = {"waveform": np.arange(8, dtype=np.float32)}
    snapshot = dataset._legacy_last_snapshot()
    dataset._last_custom["waveform"] = replacement

    dataset._restore_legacy_last_snapshot(snapshot)

    restored = dataset._last_custom["waveform"]
    assert restored.shape == (8,)
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, np.arange(8, dtype=np.float32))
