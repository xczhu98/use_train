"""共享仿真组件。"""

from core.data.simulation.array_profiles import (
    ARRAY_PROFILES,
    ArrayTopology,
    MicrophoneArrayProfile,
    ResolvedMicrophoneArray,
    build_geometry_metadata,
    get_array_profile,
    resolve_geometry_positions,
    resolve_microphone_array,
    resolve_scene_microphone_array,
)
from core.data.simulation.rir_provider import (
    OnlineRIRProvider,
    SpatialRIRProvider,
    SpatialRIRResult,
    SpatialRIRTrackSet,
    MovingSpatialRIRTrackSet,
    OfflineRIRProvider,
    build_spatial_rir_provider,
    MovingSceneRIRRequest,
    MovingRIRCapabilityError,
    validate_moving_scene_rir_request,
)
from core.data.simulation.gpurir_backend import (
    generate_moving_rir_group,
    generate_moving_rir_tracks,
)
from core.data.simulation.spatial_background import (
    SpatialBackground,
    SpatialBackgroundAssetSpec,
    parse_spatial_background_manifest_record,
    render_diffuse_background,
    validate_spatial_background,
)
from core.data.simulation.capabilities import (
    SPATIAL_FEATURE_CAPABILITIES,
    SUPPORTED_SPATIAL_TASKS,
    SpatialFeatureCapability,
    TEMPORAL_FEATURE_CAPABILITIES,
    TemporalFeatureCapability,
    triggered_spatial_features,
    triggered_temporal_features,
)
from core.data.simulation.noise_image import NoiseImage, adapt_noise_waveform
from core.data.simulation.contracts import (
    LayerEFOutput,
    LayerEFResult,
    LayerEOperationResult,
    LayerEResult,
    PeakGuardFacts,
    PeakGuardResult,
    PreparedSpeech,
    apply_deterministic_peak_guard,
)
from core.data.simulation.scene_context import (
    AcousticRuntimePort,
    BackgroundRuntimePort,
    DatasetAcousticRuntimeAdapter,
    SceneRenderContext,
    coerce_background_runtime,
    coerce_scene_render_context,
)
from core.data.simulation.role_policy import (
    AcousticActor,
    AcousticRole,
    BackgroundPolicy,
    BackgroundSpec,
    RIRPolicy,
    build_background_spec,
    validate_scene_actor_render_mode,
    validate_actor_policy,
)
from core.data.simulation.scene_manifest import (
    SCENE_MANIFEST_SCHEMA_VERSION,
    OfflineSceneManifest,
    build_track_identity,
    trajectory_digest,
)

__all__ = [
    "SPATIAL_FEATURE_CAPABILITIES",
    "SUPPORTED_SPATIAL_TASKS",
    "SpatialFeatureCapability",
    "triggered_spatial_features",
    "TEMPORAL_FEATURE_CAPABILITIES",
    "TemporalFeatureCapability",
    "triggered_temporal_features",
    "SpatialBackground",
    "SpatialBackgroundAssetSpec",
    "parse_spatial_background_manifest_record",
    "render_diffuse_background",
    "validate_spatial_background",
    "NoiseImage",
    "adapt_noise_waveform",
    "LayerEFOutput",
    "LayerEFResult",
    "LayerEOperationResult",
    "LayerEResult",
    "PeakGuardFacts",
    "PeakGuardResult",
    "PreparedSpeech",
    "apply_deterministic_peak_guard",
    "AcousticRuntimePort",
    "BackgroundRuntimePort",
    "DatasetAcousticRuntimeAdapter",
    "SceneRenderContext",
    "coerce_background_runtime",
    "coerce_scene_render_context",
    "MovingPointSourceRIRProvider",
    "PointSourceRIRProvider",
    "AcousticActor",
    "AcousticRole",
    "BackgroundPolicy",
    "BackgroundSpec",
    "build_background_spec",
    "RIRPolicy",
    "validate_scene_actor_render_mode",
    "validate_actor_policy",
    "SCENE_MANIFEST_SCHEMA_VERSION",
    "OfflineSceneManifest",
    "build_track_identity",
    "trajectory_digest",
    "ARRAY_PROFILES",
    "ArrayTopology",
    "MicrophoneArrayProfile",
    "ResolvedMicrophoneArray",
    "build_geometry_metadata",
    "get_array_profile",
    "resolve_geometry_positions",
    "resolve_microphone_array",
    "resolve_scene_microphone_array",
    "SpatialSceneRenderer",
    "SpatialSceneRequest",
    "SpatialSceneResult",
    "SpatialRIRProvider",
    "SpatialRIRResult",
    "SpatialRIRTrackSet",
    "MovingSpatialRIRTrackSet",
    "OfflineRIRProvider",
    "OnlineRIRProvider",
    "build_spatial_rir_provider",
    "MovingSceneRIRRequest",
    "MovingRIRCapabilityError",
    "validate_moving_scene_rir_request",
    "generate_moving_rir_group",
    "generate_moving_rir_tracks",
]

from core.data.simulation.trajectory import (
    SourceTrajectory,
    TrajectoryTimebase,
    plan_linear_source_trajectory,
    sample_source_trajectories_by_speed,
    validate_trajectory_timebase,
)

__all__ += [
    "SourceTrajectory",
    "TrajectoryTimebase",
    "plan_linear_source_trajectory",
    "sample_source_trajectories_by_speed",
    "validate_trajectory_timebase",
]

_LAZY_SPATIAL_SCENE = frozenset({
    "SpatialSceneRenderer", "SpatialSceneRequest", "SpatialSceneResult",
})
_LAZY_POINT_SOURCE = frozenset({
    "MovingPointSourceRIRProvider", "PointSourceRIRProvider",
})
_LAZY_SPATIAL_EXECUTOR = frozenset({"SpatialSceneExecutor", "SpatialSceneRendererLike"})
_LAZY_OBSERVATION_POST_MIX = frozenset({
    "ObservationPostMixRenderer", "ObservationPostMixRequest", "ObservationPostMixResult",
    "PostMixRoleContract", "NonDecomposableObservationContract",
    "infer_post_mix_role_contract",
})
_LAZY_DYNAMIC_COMPONENTS = frozenset({
    "SimulationRNG",
    "ResourceStore",
    "SpeechPreparation",
    "AcousticBundleResult",
    "AcousticRenderer",
    "ScenarioSampler",
})
# background_resolver 依赖 acoustic_scene_plan，而后者经 post_augment 反向依赖
# 本包；急切导入会形成初始化循环，因此与 spatial_scene 一样按需加载。
_LAZY_BACKGROUND_RESOLVER = frozenset({
    "BACKGROUND_NOISE_ACTOR_ROLE",
    "BACKGROUND_NOISE_ACTOR_UID",
    "build_background_noise_plan",
    "resolve_and_mix_background",
    "resolve_background_spec",
    "resolve_point_source_provider",
    "sample_background_noise_actor",
})


def __getattr__(name):
    """按需加载依赖 ``core.data.spatial`` 的模块，避免初始化循环。"""
    if name in _LAZY_POINT_SOURCE:
        from core.data.simulation.point_source import (
            MovingPointSourceRIRProvider,
            PointSourceRIRProvider,
        )
        values = {
            "MovingPointSourceRIRProvider": MovingPointSourceRIRProvider,
            "PointSourceRIRProvider": PointSourceRIRProvider,
        }
        return values[name]
    if name in _LAZY_BACKGROUND_RESOLVER:
        from core.data.simulation import background_resolver
        return getattr(background_resolver, name)
    if name in _LAZY_DYNAMIC_COMPONENTS:
        from core.data.simulation.acoustic_renderer import AcousticBundleResult, AcousticRenderer
        from core.data.simulation.resource_store import ResourceStore
        from core.data.simulation.rng import SimulationRNG
        from core.data.simulation.scenario_sampler import ScenarioSampler
        from core.data.simulation.speech_preparation import SpeechPreparation
        values = {
            "SimulationRNG": SimulationRNG,
            "ResourceStore": ResourceStore,
            "SpeechPreparation": SpeechPreparation,
            "AcousticBundleResult": AcousticBundleResult,
            "AcousticRenderer": AcousticRenderer,
            "ScenarioSampler": ScenarioSampler,
        }
        return values[name]
    if name in _LAZY_SPATIAL_SCENE:
        from core.data.simulation.spatial_scene import (
            SpatialSceneRenderer, SpatialSceneRequest, SpatialSceneResult,
        )
        values = {
            "SpatialSceneRenderer": SpatialSceneRenderer,
            "SpatialSceneRequest": SpatialSceneRequest,
            "SpatialSceneResult": SpatialSceneResult,
        }
        return values[name]
    if name in _LAZY_SPATIAL_EXECUTOR:
        from core.data.simulation.spatial_executor import (
            SpatialSceneExecutor,
            SpatialSceneRendererLike,
        )
        values = {
            "SpatialSceneExecutor": SpatialSceneExecutor,
            "SpatialSceneRendererLike": SpatialSceneRendererLike,
        }
        return values[name]
    if name in _LAZY_OBSERVATION_POST_MIX:
        from core.data.simulation.observation_post_mix import (
            ObservationPostMixRenderer, ObservationPostMixRequest, ObservationPostMixResult,
            NonDecomposableObservationContract, PostMixRoleContract,
            infer_post_mix_role_contract,
        )
        values = {
            "ObservationPostMixRenderer": ObservationPostMixRenderer,
            "ObservationPostMixRequest": ObservationPostMixRequest,
            "ObservationPostMixResult": ObservationPostMixResult,
            "NonDecomposableObservationContract": NonDecomposableObservationContract,
            "PostMixRoleContract": PostMixRoleContract,
            "infer_post_mix_role_contract": infer_post_mix_role_contract,
        }
        return values[name]
    raise AttributeError(name)


__all__ += sorted(_LAZY_SPATIAL_SCENE)
__all__ += sorted(_LAZY_POINT_SOURCE)
__all__ += sorted(_LAZY_SPATIAL_EXECUTOR)
__all__ += sorted(_LAZY_OBSERVATION_POST_MIX)
__all__ += sorted(_LAZY_BACKGROUND_RESOLVER)
__all__ += sorted(_LAZY_DYNAMIC_COMPONENTS)
