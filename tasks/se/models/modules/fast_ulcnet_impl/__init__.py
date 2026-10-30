"""Fast-ULCNet implementation: Comfi-FastGRNN, CRM, and core blocks.

方法来源: Fast-ULCNet（Comfi-FastGRNN、CRM 等），与公开仓库对齐。
固定上游 commit、复制/适配范围、MIT 版权及许可全文见本目录 ``PROVENANCE``、
``LICENSE.Fast-ULCNet`` 与 ``LICENSE.CRM_pytorch``。

官方/上游: https://github.com/narrietal/Fast-ULCNet
- ComfiFastGRNN: paper Comfi-FastGRNN with trainable complementary filter (gamma, lambda).
- ComplexRatioMask: R/C/E masking modes for complex spectrogram.
- No dependency on libsegmenter; STFT uses standard Hann window.
"""

from tasks.se.models.modules.fast_ulcnet_impl.comfi_fast_grnn import ComfiFastGRNN
from tasks.se.models.modules.fast_ulcnet_impl.crm import ComplexRatioMask

__all__ = ["ComfiFastGRNN", "ComplexRatioMask"]
