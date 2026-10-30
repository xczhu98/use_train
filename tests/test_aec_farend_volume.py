"""AEC 已移除 far-end 独立音量键的拒绝测试。"""

from __future__ import annotations

import pytest

from tasks.aec.data.dynamic_dataset import validate_aec_dataset_config


_RETIRED_FAREND_VOLUME_KEYS = (
    "farend_random_volume",
    "farend_volume_db_range",
    "farend_use_rms",
    "farend_volume_distribution",
    "farend_volume_beta_a",
    "farend_volume_beta_b",
)


@pytest.mark.parametrize("key", _RETIRED_FAREND_VOLUME_KEYS)
def test_aec_dataset_rejects_retired_farend_volume_keys(key):
    with pytest.raises(ValueError, match=rf"{key}.*far_db_range.*erl_range"):
        validate_aec_dataset_config(simulation_cfg={key: False})


pytestmark = [pytest.mark.unit]
