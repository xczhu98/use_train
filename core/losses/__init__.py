"""Core loss registration and build.

All concrete loss implementations live in ``core/losses/*`` and register
themselves into ``LOSS_REGISTRY`` as a side effect of import.  Importing this
package triggers those registrations, so ``core.losses.build_loss`` works
standalone.

``CompositeLoss`` is special-cased by ``build_loss`` and is intentionally not in
``LOSS_REGISTRY``.
"""

from core.losses.time_domain import *           # noqa: F401,F403
from core.losses.pairwise import *              # noqa: F401,F403
from core.losses.permutation import *           # noqa: F401,F403
from core.losses.pit import *                   # noqa: F401,F403
from core.losses.freq_domain import *           # noqa: F401,F403
from core.losses.complex_mask import *           # noqa: F401,F403
from core.losses.hybrid import *                # noqa: F401,F403
from core.losses.se_train_mr import *           # noqa: F401,F403
from core.losses.perceptual_metric import *     # noqa: F401,F403
from core.losses.perceptual_lib import *        # noqa: F401,F403
from core.losses.spectral_distance import *     # noqa: F401,F403
from core.losses.model_specific.deepfilternet import *  # noqa: F401,F403
from core.losses.model_specific.metricgan import *      # noqa: F401,F403
from core.losses.model_specific.cofi_lite import *      # noqa: F401,F403
from core.losses.fw_sdr import *                # noqa: F401,F403
from core.losses.plcpa_asymmetric import *      # noqa: F401,F403
from core.losses.scalar_passthrough import *    # noqa: F401,F403
try:
    from core.losses.pase_family import *  # noqa: F401,F403
except ImportError as exc:  # pragma: no cover - 环境可选依赖边界。
    import logging

    logging.getLogger(__name__).warning(
        "PASE-family losses unavailable during core.losses import: %s", exc
    )
except RuntimeError as exc:  # pragma: no cover - 已知 numba cache 环境边界。
    message = str(exc)
    if "cannot cache function" not in message or "no locator available" not in message:
        raise
    import logging

    logging.getLogger(__name__).warning(
        "PASE-family losses unavailable because numba cannot locate its cache: %s",
        exc,
    )
from core.losses.state_ce import *              # noqa: F401,F403
from core.losses.speaker_ce import *            # noqa: F401,F403
from core.losses.tap import *                   # noqa: F401,F403
from core.losses.mbrcl import *                 # noqa: F401,F403
from core.losses.composite import CompositeLoss  # noqa: F401
from core.losses.build import build_loss         # noqa: F401
