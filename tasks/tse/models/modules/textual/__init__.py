# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""文本 cue 前端模块（KCE / dae_kce）的导出门面。

来源：WeSep commit ``dc11cb18`` 的 ``wesep/modules/textual/``（Apache-2.0）。
"""

from tasks.tse.models.modules.textual.kce import (
    DEFAULT_KCE_FBANK_CONFIG,
    DEFAULT_KCE_MODEL_CONFIG,
    KCE,
    KCE_SAMPLE_RATE,
    DAEPhonemeTokenizer,
    KCEBackbone,
    KCEWeightsError,
    validate_kce_model_config,
)
from tasks.tse.models.modules.textual.textual_frontend import (
    DAEKCEFeature,
    DEFAULT_TEXTUAL_CONFIG,
    TextualFrontend,
    TextualFrontendOutput,
)

__all__ = [
    "DAEKCEFeature",
    "DAEPhonemeTokenizer",
    "DEFAULT_KCE_FBANK_CONFIG",
    "DEFAULT_KCE_MODEL_CONFIG",
    "DEFAULT_TEXTUAL_CONFIG",
    "KCE",
    "KCEBackbone",
    "KCEWeightsError",
    "KCE_SAMPLE_RATE",
    "TextualFrontend",
    "TextualFrontendOutput",
    "validate_kce_model_config",
]
