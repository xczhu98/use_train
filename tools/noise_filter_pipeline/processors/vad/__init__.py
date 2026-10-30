"""VAD 子包：由 processors/vad.py 拆分而来。

原 ``noise_filter_pipeline.processors.vad`` 模块的全部公开符号与进程池
worker 函数在此 re-export，保证 ``from ...processors.vad import X`` 与
pickle 按属性路径解析（spawn 进程池 worker）继续可用。
"""
from .bootstrap import (  # noqa: F401
    _ensure_fireredvad_path,
    _get_default_nemo_onnx_path,
    _get_or_download_fireredvad_root,
    _suppress_stdout_stderr,
)
from .wrappers import (  # noqa: F401
    FallbackVAD,
    FireRedVADWrapper,
    FunASRVADWrapper,
    NeMoONNXVADWrapper,
    NeMoVADWrapper,
    SileroVADWrapper,
    VADResult,
    WebRTCVADWrapper,
    _NeMoMelPreprocessor,
    create_vad_detector,
)
from .multi import (  # noqa: F401
    MultiVADDetector,
    MultiVADResult,
    _vad_worker_init,
)
