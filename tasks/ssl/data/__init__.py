"""SSL 动态仿真数据管道。"""

from .batch import (
    SSLBatch,
    SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG,
    SSL_INPUT_REFERENCE_IPD,
    SSL_TARGET_DPIPD_FAR_FIELD,
    SSL_TARGET_DPIPD_NEAR_FIELD,
)
from .collate import ssl_collate_fn
from .doa_mapping import (
    DOAMappingConfig,
    DOAMappingResult,
    DPIPDToDOAMapper,
    map_dpipd_to_doa,
)
from .projection import (
    SSL_INACTIVE_MANIFOLD_BESSEL_J0,
    SSL_INACTIVE_MANIFOLD_ZERO,
    SSLProjectionConfig,
    SSLSceneInput,
    prepare_ssl_scene,
    project_ssl_scene,
)
from .recipe import (
    SSL_RECIPE_VERSION,
    SSL_SIMULATION_PROFILES,
    SSL_SIMULATION_RECIPES,
    SSLSimulationRecipe,
    apply_overlap_policy,
    clean_speech_with_webrtcvad,
    resolve_ssl_recipe,
)

__all__ = [
    "SSLBatch",
    "SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG",
    "SSL_INPUT_REFERENCE_IPD",
    "SSL_TARGET_DPIPD_FAR_FIELD",
    "SSL_TARGET_DPIPD_NEAR_FIELD",
    "ssl_collate_fn",
    "DOAMappingConfig",
    "DOAMappingResult",
    "DPIPDToDOAMapper",
    "map_dpipd_to_doa",
    "SSLSceneInput",
    "SSLProjectionConfig",
    "SSL_INACTIVE_MANIFOLD_BESSEL_J0",
    "SSL_INACTIVE_MANIFOLD_ZERO",
    "prepare_ssl_scene",
    "project_ssl_scene",
    "SSL_RECIPE_VERSION",
    "SSL_SIMULATION_PROFILES",
    "SSL_SIMULATION_RECIPES",
    "SSLSimulationRecipe",
    "apply_overlap_policy",
    "clean_speech_with_webrtcvad",
    "resolve_ssl_recipe",
]
