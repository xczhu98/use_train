"""模型输入输出的共享描述语言。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


TensorDomain = Literal[
    "waveform",
    "spectrum",
    "embedding",
    "logits",
    "label",
    "mask",
    "loss",
]
ForwardKind = Literal[
    "estimate",
    "loss_returning",
    "gan_generator",
    "gan_discriminator",
    "streaming",
]
FrontendOwner = Literal["runtime", "model"]

_DOMAINS = {
    "waveform",
    "spectrum",
    "embedding",
    "logits",
    "label",
    "mask",
    "loss",
}
_FORWARD_KINDS = {
    "estimate",
    "loss_returning",
    "gan_generator",
    "gan_discriminator",
    "streaming",
}
_FRONTEND_OWNERS = {"runtime", "model"}


@dataclass(frozen=True)
class TensorField:
    """一个具名 tensor role 的 domain、axes 与可选性。"""

    name: str
    domain: TensorDomain
    axes: tuple[str, ...]
    required: bool = True
    multiple: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("TensorField.name must be a non-empty string")
        if self.domain not in _DOMAINS:
            raise ValueError(f"unsupported tensor domain: {self.domain!r}")
        if not isinstance(self.axes, tuple) or not self.axes:
            raise ValueError("TensorField.axes must be a non-empty tuple")
        if any(not isinstance(axis, str) or not axis for axis in self.axes):
            raise ValueError("TensorField axes must be non-empty strings")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("TensorField axes must be unique")
        if type(self.required) is not bool or type(self.multiple) is not bool:
            raise TypeError("TensorField required/multiple must be bool")


@dataclass(frozen=True)
class ModelIOContract:
    """模型固有 task、字段和 forward/frontend 语义。"""

    tasks: frozenset[str]
    inputs: tuple[TensorField, ...]
    outputs: tuple[TensorField, ...]
    forward_kind: ForwardKind
    frontend_owner: FrontendOwner

    def __post_init__(self) -> None:
        if not self.tasks or any(task not in {"se", "aec", "tse", "ss", "ssl"} for task in self.tasks):
            raise ValueError("ModelIOContract.tasks must contain supported task names")
        if not self.inputs or not self.outputs:
            raise ValueError("ModelIOContract requires non-empty inputs and outputs")
        if len({field.name for field in self.inputs}) != len(self.inputs):
            raise ValueError("ModelIOContract input names must be unique")
        if len({field.name for field in self.outputs}) != len(self.outputs):
            raise ValueError("ModelIOContract output names must be unique")
        if self.forward_kind not in _FORWARD_KINDS:
            raise ValueError(f"unsupported forward_kind: {self.forward_kind!r}")
        if self.frontend_owner not in _FRONTEND_OWNERS:
            raise ValueError(f"unsupported frontend_owner: {self.frontend_owner!r}")


def resolve_dataset_microphone_array(dataset: Any):
    """动态/已物化 observation 共用同一个已解析 microphone 几何入口。"""
    from core.data.simulation.array_profiles import ResolvedMicrophoneArray, resolve_microphone_array

    resolved = getattr(dataset, "resolved_microphone_array", None)
    if resolved is not None:
        if not isinstance(resolved, ResolvedMicrophoneArray):
            raise TypeError("dataset.resolved_microphone_array must be ResolvedMicrophoneArray")
        return resolved
    simulation = getattr(dataset, "sim_cfg", None)
    simulated_spatial = bool(
        simulation is not None
        and getattr(simulation, "spatial_enabled", False) is True
    )
    offline_spatial = getattr(dataset, "spatial_enabled", False) is True
    if not simulated_spatial and not offline_spatial:
        return None

    geometry = None
    if simulated_spatial:
        geometry = resolve_microphone_array(
            array_profile=getattr(simulation, "array_profile", None),
            custom_geometry=getattr(simulation, "custom_geometry", None),
            reference_channel=getattr(simulation, "reference_channel", 0),
        )
        if geometry is None:
            raise ValueError("spatial training dataset requires resolved microphone geometry")
        return geometry
    raise ValueError("offline spatial dataset requires resolved microphone geometry; unknown C is not supported")


def inject_dataset_model_geometry(cfg: dict, dataset: Any) -> None:
    """在构造前注入并保存 geometry 参数；不改已构造模型，不隐式投影 observation。"""
    import inspect
    from core.models.build import resolve_model_class

    geometry = resolve_dataset_microphone_array(dataset)
    if geometry is None:
        return
    model_cfg = cfg.get("model", {})
    if "generator" in model_cfg:
        model_cfg = model_cfg["generator"]
    cls = resolve_model_class(model_cfg)
    contract = getattr(cls, "IO_CONTRACT", None)
    native = getattr(cls, "supports_multichannel_input", False) is True or any(
        "microphone" in field.axes for field in getattr(contract, "inputs", ())
    )
    if geometry.channels > 1 and not native:
        raise ValueError(f"spatial dataset C={geometry.channels} requires native microphone-axis model; got {cls.__name__}")
    parameters = inspect.signature(cls).parameters
    args = dict(model_cfg.get("args") or {})
    for names, expected in ((("ref_channel", "reference_channel"), geometry.reference_channel),
                            (("n_channel",), geometry.channels)):
        for name in names:
            if name not in parameters:
                continue
            if name in args and (type(args[name]) is not int or args[name] != expected):
                raise ValueError(f"model.args.{name}={args[name]!r} conflicts with Dataset geometry {expected}")
            args[name] = expected
    model_cfg["args"] = args


def validate_split_microphone_contract(train_dataset: Any, validation_dataset: Any) -> None:
    """同一模型训练/验证时必须保持 C 和 reference 索引；不要求两 split 同一房间。"""
    train = resolve_dataset_microphone_array(train_dataset)
    validation = resolve_dataset_microphone_array(validation_dataset)
    train_shape = (1, 0) if train is None else (train.channels, train.reference_channel)
    validation_shape = (1, 0) if validation is None else (validation.channels, validation.reference_channel)
    if train_shape != validation_shape:
        raise ValueError(
            f"train/validation microphone contract mismatch: C/reference={train_shape} vs {validation_shape}"
        )


def validate_spatial_model_geometry(model: Any, dataset: Any) -> None:
    """在模型构造边界校验 Dataset geometry 与原生多通道模型合同。"""
    geometry = resolve_dataset_microphone_array(dataset)
    if geometry is None:
        return

    primary = model
    if isinstance(model, dict):
        primary = model.get("generator")
        if primary is None:
            raise ValueError(
                "spatial multi-model training requires a primary 'generator' model"
            )
    contract = getattr(primary, "IO_CONTRACT", None)
    accepts_microphones = getattr(primary, "supports_multichannel_input", False) is True
    accepts_microphones = accepts_microphones or any(
        field.domain == "spectrum" and "microphone" in field.axes
        for field in getattr(contract, "inputs", ())
    )
    if geometry.channels > 1 and not accepts_microphones:
        channel_text = f"C={geometry.channels}"
        raise ValueError(
            f"spatial dataset has {channel_text}, but the model does not "
            "declare a native microphone-axis input contract"
        )

    declared_channels = getattr(primary, "n_channel", None)
    if declared_channels is not None and declared_channels != geometry.channels:
        raise ValueError(
            f"model n_channel={declared_channels} conflicts with resolved "
            f"microphone geometry C={geometry.channels}"
        )
    declared_reference = getattr(primary, "ref_channel", getattr(primary, "reference_channel", None))
    if declared_reference is not None and declared_reference != geometry.reference_channel:
        raise ValueError(
            f"model ref_channel={declared_reference} conflicts with simulation "
            f"reference_channel={geometry.reference_channel}"
        )
