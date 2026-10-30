"""MOS 配置、资产与 Step3 wire 的轻量 canonical contract。"""

from __future__ import annotations

from types import MappingProxyType


__all__ = [
    "MOS_ASSET_LOAD_BINDING",
    "MOS_ASSET_PATH_FIELDS",
    "MOS_ASSET_PROVENANCE_FIELDS",
    "MOS_ASSET_SETTING_FIELDS",
    "MOS_MODEL_ORDER",
    "MOS_MODEL_REQUEST_OPTION_FIELDS",
    "MOS_PROVIDER_ATTESTATION_FIELDS",
    "MOS_RUNTIME_OPTION_FIELDS",
    "MOS_SCORING_EXECUTION_POLICY",
    "MOS_SHIPPED_MODELS",
    "MOS_THRESHOLD_DIMENSIONS",
    "MOS_WORKER_ATTESTATION_FIELDS",
    "MOS_WORKER_MODEL_TUPLE_ATTESTATION_FIELDS",
    "MOS_WORKER_ORDINAL_SIDECAR_FIELDS",
    "MOS_WORKER_PROTOCOL",
    "MOS_WORKER_REQUEST_FIELDS",
    "MOS_WORKER_SUBSHARD_FIELDS",
    "MOS_WORKER_TUPLE_PROVIDER_ATTESTATION_FIELDS",
    "is_mos_device_attestation",
    "validate_mos_provider_attestation",
]


# 五个条目是 registry/canonical wire 的 option slots 与 provenance 顺序；实际执行
# 集合始终由配置策略选择，未选择的 slot 为 None，也不会参与资产 preflight。
MOS_MODEL_ORDER = ("scoreq", "urgent_pk", "dnsmos", "nisqa", "utmos")
# 导出名为兼容保留，不等同于 shipped 生产 route；route 仍由配置策略决定。
MOS_SHIPPED_MODELS = MOS_MODEL_ORDER[:3]

MOS_THRESHOLD_DIMENSIONS = MappingProxyType({
    "scoreq": (),
    "urgent_pk": (),
    "dnsmos": ("ovrl", "sig", "bak", "p808"),
    "nisqa": (),
    "utmos": (),
})

MOS_ASSET_PATH_FIELDS = MappingProxyType({
    "scoreq": frozenset({"model_dir"}),
    "urgent_pk": frozenset({"checkpoint", "source_dir"}),
    "dnsmos": frozenset({"model_dir"}),
    "nisqa": frozenset({"model_path"}),
    "utmos": frozenset({"model_path"}),
})
MOS_ASSET_SETTING_FIELDS = MappingProxyType({
    "scoreq": frozenset(),
    "urgent_pk": frozenset({
        "encoder", "backbone", "score_range", "expected_sha256",
        "expected_commit",
    }),
    "dnsmos": frozenset(),
    "nisqa": frozenset({"model_version", "model_sha256"}),
    "utmos": frozenset({"model_version", "model_sha256"}),
})

MOS_RUNTIME_OPTION_FIELDS = MappingProxyType({
    "scoreq": frozenset({"model_dir", "verified_model_tree_sha256"}),
    "urgent_pk": frozenset({
        "checkpoint", "source_dir", "encoder", "backbone", "score_range",
        "expected_sha256", "expected_commit",
    }),
    "dnsmos": frozenset({"model_dir", "verified_model_tree_sha256"}),
    "nisqa": frozenset({
        "model_path", "model_version", "model_sha256", "checkpoint_sha256",
        "checkpoint_size_bytes",
    }),
    "utmos": frozenset({
        "model_path", "model_version", "model_sha256", "checkpoint_sha256",
        "checkpoint_size_bytes",
    }),
})
MOS_ASSET_PROVENANCE_FIELDS = MappingProxyType({
    "scoreq": frozenset({"model_dir", "verified_model_tree_sha256"}),
    "urgent_pk": frozenset({
        "checkpoint_sha256", "source_commit", "source_package_sha256",
    }),
    "dnsmos": frozenset({"model_dir", "verified_model_tree_sha256"}),
    "nisqa": frozenset({
        "model_path", "model_version", "expected_model_tree_sha256",
        "verified_model_tree_sha256", "checkpoint_sha256",
        "checkpoint_size_bytes",
    }),
    "utmos": frozenset({
        "model_path", "model_version", "expected_model_tree_sha256",
        "verified_model_tree_sha256", "checkpoint_sha256",
        "checkpoint_size_bytes",
    }),
})
MOS_MODEL_REQUEST_OPTION_FIELDS = MappingProxyType({
    "scoreq": "scoreq_options",
    "urgent_pk": "urgent_options",
    "dnsmos": "dnsmos_options",
    "nisqa": "nisqa_options",
    "utmos": "utmos_options",
})

# 评分执行语义独立于资产加载绑定：修改 waveform padding、跨 source 聚合、
# source session 或批处理资源策略时必须递增 version，令 Step3 provenance/resume
# 与 worker wire 精确失配；threshold-only policy 不得进入此 payload。
MOS_SCORING_EXECUTION_POLICY = MappingProxyType({
    "version": 1,
    "waveform": MappingProxyType({
        "channel": "first",
        "dtype": "float32",
        "padding": "none",
    }),
    "batching": MappingProxyType({
        "max_rows": 32,
        "max_float32_bytes": 64 * 1024 * 1024,
        "decode_reservation": "interval_channels_plus_mono_float32",
        "cross_source": True,
        "flush": "tuple_change_or_limit_or_eof",
    }),
    "source_session": MappingProxyType({
        "reuse": "same_source_across_batches",
        "switch": "close_reverify_then_open",
    }),
    "worker": MappingProxyType({"physical_subshard_boundary": "flush"}),
})

# 资产加载绑定版本与 worker wire protocol 各自独立演进、精确匹配；v3/v6
# 组合不表示版本协商，也不接纳缺字段或旧版本 artifact。完整 subshard
# schema 包含稀疏 ordinal sidecar；连续 shard 从该集合去除 sidecar 字段推导。
MOS_ASSET_LOAD_BINDING = MappingProxyType({"version": 3})
MOS_WORKER_PROTOCOL = "audio_filter_pipeline.step3_mos_worker.v6"
MOS_WORKER_REQUEST_FIELDS = frozenset({
    "protocol", "asset_load_binding", "scoring_execution_policy", "private_dir",
    "attestation_path", "subshards", *MOS_MODEL_REQUEST_OPTION_FIELDS.values(),
})
MOS_WORKER_ORDINAL_SIDECAR_FIELDS = frozenset({
    "ordinal_sidecar_path", "ordinal_sidecar_sha256", "ordinal_sidecar_size_bytes",
})
MOS_WORKER_SUBSHARD_FIELDS = frozenset({
    "models", "input_path", "output_path", "row_count", "input_sha256",
    "start_ordinal", "end_ordinal", *MOS_WORKER_ORDINAL_SIDECAR_FIELDS,
})
MOS_WORKER_ATTESTATION_FIELDS = frozenset({
    "protocol", "asset_load_binding", "scoring_execution_policy", "worker_models",
    "model_tuple_attestations",
})
MOS_WORKER_MODEL_TUPLE_ATTESTATION_FIELDS = frozenset({
    "models", "attestation",
})
# 单个 model tuple 的 provider attestation 信封字段；writer（mos_runtime
# _runtime_provider_provenance）与 reader（mos_gpu_job _read_runtime_attestation）
# 必须以这份字段集为准，任一侧单独增删字段都会 fail-closed。
MOS_WORKER_TUPLE_PROVIDER_ATTESTATION_FIELDS = frozenset({
    "requested_device", "actual_providers", "fallback",
    "selected_models", "asset_load_binding", "scoring_execution_policy",
})

MOS_PROVIDER_ATTESTATION_FIELDS = MappingProxyType({
    "scoreq": frozenset({"execution_mode", "providers", "device"}),
    "urgent_pk": frozenset({"device", "parameters"}),
    "dnsmos": MappingProxyType({
        "cpu": frozenset({"onnx_sess", "p808_onnx_sess"}),
        "cuda": frozenset({
            "onnx_sess", "p808_onnx_sess", "cpu_ep_fallback_disabled",
        }),
    }),
    "nisqa": frozenset({"device", "parameters", "checkpoint_sha256"}),
    "utmos": frozenset({"device", "parameters", "checkpoint_sha256"}),
})


def is_mos_device_attestation(value, *, requested):
    """按完整值校验 CPU/CUDA device 证明，禁止子串伪装。"""
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported requested MOS provider device: {requested!r}")
    if type(value) is not str:
        return False
    if requested == "cpu":
        return value == "cpu"
    if value in {"cuda", "gpu"}:
        return True
    kind, separator, index = value.partition(":")
    return (
        separator == ":"
        and kind == "cuda"
        and index.isascii()
        and index.isdigit()
        and (index == "0" or not index.startswith("0"))
    )


def validate_mos_provider_attestation(
    model, value, *, requested, schema_label=None,
):
    """严格校验 SCOREQ/DNSMOS provider entry 的 canonical 值语义。"""
    if model not in {"scoreq", "dnsmos"}:
        raise ValueError(f"unsupported canonical MOS provider model: {model!r}")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported requested MOS provider device: {requested!r}")
    if schema_label is not None and type(schema_label) is not str:
        raise TypeError("MOS provider schema_label must be a string or None")
    fields = MOS_PROVIDER_ATTESTATION_FIELDS[model]
    expected_fields = fields[requested] if model == "dnsmos" else fields
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or set(value) != expected_fields
    ):
        label = schema_label or f"MOS {model} provider schema"
        raise ValueError(f"{label} is invalid")

    if model == "scoreq":
        providers = value["providers"]
        device = value["device"]
        execution_mode = value["execution_mode"]
        if (
            type(providers) is not list
            or any(type(provider) is not str for provider in providers)
            or type(execution_mode) is not str
        ):
            raise ValueError("MOS scoreq provider attestation values are invalid")
        if requested == "cuda":
            cuda_only = providers == ["CUDAExecutionProvider"]
            cuda_hybrid = providers == [
                "CUDAExecutionProvider", "CPUExecutionProvider",
            ]
            expected_mode = (
                "cuda_only_ep" if cuda_only
                else "cuda_primary_hybrid_ep" if cuda_hybrid
                else None
            )
            valid = (
                execution_mode == expected_mode
                and is_mos_device_attestation(device, requested=requested)
            )
        else:
            valid = (
                providers == ["CPUExecutionProvider"]
                and execution_mode == "cpu_ep"
                and is_mos_device_attestation(device, requested=requested)
            )
        if not valid:
            suffix = "cuda-primary" if requested == "cuda" else "cpu"
            raise ValueError(f"MOS scoreq provider is not {suffix}")
        return

    fallback_disabled = value.get("cpu_ep_fallback_disabled", False)
    if requested == "cuda" and type(fallback_disabled) is not bool:
        raise ValueError("MOS dnsmos provider attestation values are invalid")
    for name in ("onnx_sess", "p808_onnx_sess"):
        providers = value[name]
        if (
            type(providers) is not list
            or any(type(provider) is not str for provider in providers)
        ):
            raise ValueError("MOS dnsmos provider attestation values are invalid")
        if requested == "cuda":
            cuda_only = providers == ["CUDAExecutionProvider"]
            strict_hybrid = (
                providers == [
                    "CUDAExecutionProvider", "CPUExecutionProvider",
                ]
                and fallback_disabled is True
            )
            valid = cuda_only or strict_hybrid
        else:
            valid = providers == ["CPUExecutionProvider"]
        if not valid:
            raise ValueError(f"MOS dnsmos {name} provider is not {requested}")
