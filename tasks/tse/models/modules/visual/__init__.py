"""MuSE 视觉 cue frontend（WeSep commit ``dc11cb18`` 移植）。

只暴露视觉 frontend 与其 kernel：不注册模型、不持有 ``CueComposer`` 融合层、
不持有 separator STFT，也不解码视频文件。模型接线由后续 scope 在
``TSE_Modular`` 侧完成。
"""

from tasks.tse.models.modules.visual.muse import (
    MUSE_FRONTEND_FEATURE_DIM,
    build_visual_adapter,
    load_muse_frontend,
)
from tasks.tse.models.modules.visual.visual_frontend import (
    VisualFrontend,
    VisualFrontendOutput,
    VisualRepresentation,
)


__all__ = [
    "MUSE_FRONTEND_FEATURE_DIM",
    "VisualFrontend",
    "VisualFrontendOutput",
    "VisualRepresentation",
    "build_visual_adapter",
    "load_muse_frontend",
]
