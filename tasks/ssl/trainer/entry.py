"""SSL 训练入口：formal dynamic_ssl 共享仿真配置边界。"""

from collections.abc import Mapping

from core.trainer.entry import UnifiedTrainingEntry
from tasks.ssl.config import resolve_ssl_num_tracks

# SSLDynamicMixingDataset 构造器显式消费的 builder-only split 键。
_SSL_DATASET_BUILDER_KEYS = frozenset(
    {
        "placeholder_num_samples",
        "simulation_cfg_strict",
        "strict_load",
        "simulation_profile",
        "vad_cleaning",
        "segment_sampling",
        "overlap_policy",
    }
)
# ssl: section 的合法键白名单，与 _validate_config/_dataset_config 的真实消费点一一对应。
# 已退役的 output_stride 不在其中：它走专门的"已退役"报错，不被当作未知键。
_SSL_SECTION_KEYS = frozenset(
    {
        "input_representation",
        "target_representation",
        "vad_mode",
        "num_tracks",
        "time_stride",
        "exclude_dc",
        "normalization",
        "forgetting_norm_sample_length",
        "inactive_target_manifold",
        "phase_sign",
        "coordinate_frame",
        "model_inputs",
        "source_distance_input",
        "mapping",
        "pair_layout",
        "geometry_mode",
        "time_reduction",
    }
)
# 这些字段属于模型的实际结构合同，不允许仅靠 ``model_spec`` 重新声明成
# 另一条输入/目标路径。配置边界先校验一次，避免把错误推迟到模型构造或首个
# batch 的 forward。
_SSL_MODEL_CONTRACTS = {
    "FixedArrayIPDNet": {
        "input_representation": "reference_ipd",
        "target_representation": "dpipd_far_field",
        "geometry_mode": "fixed_array",
        "mapping": "far_field_grid",
        "time_reduction": "model_defined",
        "causal": False,
        "target_layout": "trailing_ri",
    },
    "FNSSL": {
        "input_representation": "multichannel_stft_real_imag",
        "target_representation": "dpipd_far_field",
        "geometry_mode": "fixed_array",
        "mapping": "far_field_grid",
        "time_reduction": "average_pool_12",
        "causal": False,
        "target_layout": "trailing_ri",
    },
    "IPDNetFixed": {
        "input_representation": "multichannel_stft_real_imag",
        "target_representation": "dpipd_far_field",
        "geometry_mode": "fixed_array",
        "mapping": "far_field_grid",
        "time_reduction": "causal_conv_pool",
        # IPDnet fixed exposes online (causal=True) and offline
        # (causal=False) narrow-band variants. The convolutional block stays
        # causal in both variants; the model argument selects the LSTM mode.
        "causal": None,
        "target_layout": "trailing_ri",
    },
    "IPDNetVariable": {
        "input_representation": "multichannel_stft_real_imag",
        "target_representation": "dpipd_far_field",
        "geometry_mode": "variable_array",
        "mapping": "far_field_grid",
        "time_reduction": "causal_conv_pool",
        # See IPDNetFixed: causal is an explicit model variant switch.
        "causal": None,
        "target_layout": "trailing_ri",
    },
    "IPDNet2": {
        "input_representation": "multichannel_stft_real_imag",
        "target_representation": "dpipd_near_field",
        "geometry_mode": "variable_array",
        "mapping": "near_field_doa_range_grid",
        "time_reduction": "mamba_time_pool",
        "causal": True,
        "target_layout": "trailing_ri",
    },
}


class SSLTrainingEntry(UnifiedTrainingEntry):
    @staticmethod
    def _dataset_config(cfg, split):
        data = cfg["data"]
        section = data.get(split, {})
        if not isinstance(section, dict) or not section:
            raise ValueError(f"data.{split} must be a non-empty mapping")
        from core.data.config_flags import (
            resolve_placeholder_num_samples,
            resolve_simulation_cfg_strict,
            resolve_strict_load,
        )
        from core.data.dataset_build import (
            BUILDER_ONLY_SECTION_KEYS,
            validate_layer_a_runtime_config,
        )
        from core.data.resource_paths import resolve_data_section_resource_paths

        section = dict(section)
        num_tracks = resolve_ssl_num_tracks(cfg)
        seed = (cfg.get("meta") or {}).get("seed", 42)
        if type(seed) is not int:
            raise ValueError("meta.seed must be an exact integer")
        if "seed" in section and (type(section["seed"]) is not int or section["seed"] != seed):
            raise ValueError(f"data.{split}.seed must match meta.seed")
        section["seed"] = seed
        strict_load = resolve_strict_load(cfg)
        if not strict_load or section.get("strict_load", True) is not True:
            raise ValueError("dynamic_ssl requires strict_load=True; placeholder supervision is unsupported")
        section["strict_load"] = strict_load
        section.setdefault("simulation_cfg_strict", resolve_simulation_cfg_strict(cfg))
        section.setdefault("placeholder_num_samples", resolve_placeholder_num_samples(cfg))
        validate_layer_a_runtime_config(section, config_path=f"data.{split}")
        if "max_duration" in section:
            raise ValueError("dynamic_ssl uses segment_seconds, not max_duration")
        duration = section.pop("max_duration_seconds", None)
        if duration is not None:
            if "segment_seconds" in section and section["segment_seconds"] != duration:
                raise ValueError(f"data.{split}.segment_seconds conflicts with max_duration_seconds")
            section["segment_seconds"] = duration
        section.pop("num_per_epoch", None)
        # SSLDynamicMixingDataset 只消费下列 builder-only 键；其余（mode/use_dynamic/
        # sampling/return_noise 等）必须在入口显式拒绝，不能透传给严格构造器。
        residual_builder_keys = sorted(
            key
            for key in BUILDER_ONLY_SECTION_KEYS
            if key in section and key not in _SSL_DATASET_BUILDER_KEYS
        )
        if residual_builder_keys:
            raise ValueError(
                f"data.{split} contains builder-only keys unsupported by dynamic_ssl: "
                f"{residual_builder_keys}"
        )
        resolve_data_section_resource_paths(section)
        # Dataset 特征维度取决于仿真阵列的实际麦克风数 C；不能从模型
        # input_size 反推，因为 IPDNetFixed 使用 2C，IPDNetVariable 固定为 4。
        section.setdefault(
            "simulation_cfg",
            section.get("simulation_cfg", data.get("simulation_cfg", cfg.get("simulation_cfg"))),
        )
        simulation = SSLTrainingEntry._resolve_simulation_cfg(section, split)
        from core.data.simulation.array_profiles import resolve_microphone_array

        geometry = resolve_microphone_array(
            array_profile=simulation.array_profile,
            custom_geometry=simulation.custom_geometry,
            reference_channel=simulation.reference_channel,
        )
        if geometry is None or geometry.channels < 2:
            raise ValueError(
                f"data.{split}.simulation_cfg must resolve an array with at least two microphones"
            )
        channels = geometry.channels
        model_args = dict(cfg.get("model", {}).get("args", {}))
        stft = dict(cfg.get("stft") or {})
        model_type = cfg.get("model", {}).get("type")
        ssl_cfg = cfg.get("ssl")
        if ssl_cfg is None and model_type in {None, "FixedArrayIPDNet"}:
            ssl_cfg = {
                "input_representation": "reference_ipd",
                "target_representation": "dpipd_far_field",
                "time_stride": model_args.get("downsample", 1),
            }
        if not isinstance(ssl_cfg, Mapping):
            raise ValueError("dynamic_ssl requires an explicit ssl mapping")
        input_representation = ssl_cfg.get("input_representation")
        target_representation = ssl_cfg.get(
            "target_representation", "dpipd_far_field"
        )
        vad_mode = ssl_cfg.get("vad_mode", "direct_path")
        if vad_mode != "direct_path":
            raise ValueError("ssl.vad_mode currently supports only 'direct_path'")
        if "output_stride" in ssl_cfg:
            raise ValueError("ssl.output_stride is retired; use ssl.time_stride")
        output_stride = ssl_cfg.get("time_stride", model_args.get("stride", 1))
        if input_representation not in {"multichannel_stft_real_imag", "reference_ipd"}:
            raise ValueError(
                "dynamic_ssl dataset input_representation must be multichannel_stft_real_imag or reference_ipd"
            )
        if target_representation not in {"dpipd_far_field", "dpipd_near_field"}:
            raise ValueError("dynamic_ssl dataset target_representation is unsupported")
        if type(output_stride) is not int or output_stride < 1:
            raise ValueError("dynamic_ssl dataset output_stride must be a positive integer")
        exclude_dc = ssl_cfg.get("exclude_dc", False)
        if type(exclude_dc) is not bool:
            raise TypeError("ssl.exclude_dc must be a bool")
        normalization = ssl_cfg.get("normalization", "none")
        if normalization not in {"none", "forgetting_laplace", "global_mean"}:
            raise ValueError(
                "ssl.normalization must be none, forgetting_laplace or global_mean"
            )
        forgetting_norm_sample_length = ssl_cfg.get(
            "forgetting_norm_sample_length", 298
        )
        if (
            type(forgetting_norm_sample_length) is not int
            or forgetting_norm_sample_length < 1
        ):
            raise ValueError(
                "ssl.forgetting_norm_sample_length must be an exact integer >= 1"
            )
        # inactive manifold 显式登记官方两种形式：FN-SSL 零向量 / IPDnet·IPDnet2 J0 Bessel。
        inactive_manifold = ssl_cfg.get("inactive_target_manifold", "zero")
        if inactive_manifold not in {"zero", "bessel_j0"}:
            raise ValueError("ssl.inactive_target_manifold must be zero or bessel_j0")
        # 默认 +1 与 projection 的 others*conj(ref) 约定绑定（DOAMappingConfig 同值），
        # 镜像时必须由配置显式改回 -1.0。
        phase_sign = ssl_cfg.get("phase_sign", 1.0)
        if isinstance(phase_sign, bool) or phase_sign not in {-1, -1.0, 1, 1.0}:
            raise ValueError("ssl.phase_sign must be -1 or +1")
        coordinate_frame = ssl_cfg.get("coordinate_frame", "room_xyz_m")
        if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
            raise ValueError("ssl.coordinate_frame must be a non-empty string")
        model_input_size = model_args.get("input_size")
        if type(model_input_size) is not int or model_input_size < 1:
            raise ValueError("SSL model.args.input_size must be a positive integer")
        # 模型 input_size 保持各模型自己的合同；Dataset input_size 是由阵列 C
        # 与输入表示共同决定的最终实虚特征维度。
        feature_size = (
            2 * channels
            if input_representation == "multichannel_stft_real_imag"
            else 2 * (channels - 1)
        )
        section.setdefault("sample_rate", data.get("sample_rate", 16000))
        section.setdefault("n_fft", stft.get("n_fft", model_args.get("n_fft", 512)))
        section.setdefault("hop_length", stft.get("hop_length", section["n_fft"] // 2))
        section.setdefault("win_length", stft.get("win_length", section["n_fft"]))
        section.setdefault("window", stft.get("window", "hamming"))
        section.setdefault("center", stft.get("center", False))
        if "input_size" in section and section["input_size"] != feature_size:
            raise ValueError(
                f"data.{split}.input_size must equal feature dimension {feature_size} "
                f"for input_representation={input_representation!r}"
            )
        section["input_size"] = feature_size
        section["tracks"] = num_tracks
        section["max_track"] = num_tracks
        section["input_representation"] = input_representation
        section["target_representation"] = target_representation
        section["vad_mode"] = vad_mode
        section["output_stride"] = output_stride
        section["exclude_dc"] = exclude_dc
        section["normalization"] = normalization
        section["forgetting_norm_sample_length"] = forgetting_norm_sample_length
        section["inactive_target_manifold"] = inactive_manifold
        section["phase_sign"] = float(phase_sign)
        section["coordinate_frame"] = coordinate_frame
        if "sound_speed" in section:
            raise ValueError(
                f"data.{split}.sound_speed is derived from simulation_cfg.gpurir_sound_speed_mps"
            )
        section.setdefault("speech_scp", data.get("speech_scp"))
        section.setdefault("noise_scp", data.get("noise_scp"))
        section.setdefault("rir_scp", data.get("rir_scp"))
        # ``meta.ssl_profile`` remains a compatibility marker for the checked-in
        # official templates.  Only the three registered official values are
        # promoted into the task-local recipe; moving/shared markers are not
        # silently reinterpreted as an official recipe.
        from tasks.ssl.data.recipe import resolve_ssl_recipe, SSL_SIMULATION_PROFILES

        if "simulation_profile" not in section:
            marker = (cfg.get("meta") or {}).get("ssl_profile")
            if marker in SSL_SIMULATION_PROFILES:
                section["simulation_profile"] = marker
        recipe = resolve_ssl_recipe(
            section.get("simulation_profile"),
            vad_cleaning=section.get("vad_cleaning"),
            segment_sampling=section.get("segment_sampling"),
            overlap_policy=section.get("overlap_policy"),
        )
        if recipe is not None:
            section.setdefault("vad_cleaning", recipe.vad_cleaning)
            section.setdefault("segment_sampling", recipe.segment_sampling)
            section.setdefault("overlap_policy", recipe.overlap_policy)
        return section

    @staticmethod
    def _validate_composite_loss_config(loss_cfg, *, target_representation):
        """复用 loss 构造校验参数，并在 entry 收口目标表示兼容性。"""

        raw_args = loss_cfg.get("args")
        args = {} if raw_args is None else raw_args
        if not isinstance(args, Mapping):
            raise TypeError("loss.args must be a mapping")
        from tasks.ssl.losses import SSLDOACompositeLoss

        SSLDOACompositeLoss(**dict(args))
        range_weight = args.get("range_weight", 0.0)
        if range_weight > 0.0 and target_representation != "dpipd_near_field":
            raise ValueError(
                "SSLDOACompositeLoss range_weight requires ssl.target_representation='dpipd_near_field'"
            )

    def build_loaders(self, cfg, rank, local_rank, world_size):
        from tasks.task_api import load_task_data_api

        data_api = load_task_data_api("ssl")
        SSLDynamicMixingDataset = data_api.resolve(
            "tasks.ssl.data.dynamic_dataset:SSLDynamicMixingDataset"
        )

        data = cfg["data"]
        mode = data["mode"].strip().lower()
        train_cfg = self._dataset_config(cfg, "train")
        val_cfg = self._dataset_config(cfg, "validation")
        self._validate_split_resources(mode, train_cfg, "train")
        self._validate_split_resources(mode, val_cfg, "validation")
        # DP-IPD near-field manifold 与 RIR 物理渲染必须使用同一个声速；不从
        # projection 默认值重新猜测，直接绑定 canonical SimulationConfigs。
        train_simulation = self._resolve_simulation_cfg(train_cfg, "train")
        val_simulation = self._resolve_simulation_cfg(val_cfg, "validation")
        train_cfg["sound_speed"] = float(train_simulation.gpurir_sound_speed_mps)
        val_cfg["sound_speed"] = float(val_simulation.gpurir_sound_speed_mps)
        train_cfg.pop("provider", None)
        train_cfg["simulation_split"] = "train"
        val_cfg.pop("provider", None)
        val_cfg["simulation_split"] = "validation"
        train = SSLDynamicMixingDataset(**train_cfg)
        val = SSLDynamicMixingDataset(**val_cfg)
        seed = train_cfg["seed"]
        train.set_epoch(0, base_seed=seed)
        val.set_epoch(0, base_seed=seed)
        return self._build_loaders_common(
            cfg, rank, local_rank, world_size, train_ds=train, val_ds=val,
            train_collate_fn=data_api.collate,
            train_num_per_epoch=data["train"].get("num_per_epoch"),
            val_num_per_epoch=data["validation"].get("num_per_epoch"),
        )

    @staticmethod
    def _validate_split_resources(mode, section, split):
        from core.data.constants import DATA_MODE_DYNAMIC_SSL

        if mode != DATA_MODE_DYNAMIC_SSL:
            raise ValueError(
                f"SSL supports only data.mode={DATA_MODE_DYNAMIC_SSL}, got {mode!r}"
            )
        online = (section.get("simulation_cfg") or {}).get("rir_backend") == "online"
        if online and section.get("rir_scp") is not None:
            raise ValueError(
                f"data.{split}.rir_scp is not consumed when rir_backend=online"
            )
        required = (
            ("speech_scp", "noise_scp") if online else ("speech_scp", "noise_scp", "rir_scp")
        )
        missing = [name for name in required if not section.get(name)]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"formal dynamic_ssl requires data.{split}.{joined}")
        if not isinstance(section.get("simulation_cfg"), Mapping):
            raise ValueError(f"data.{split}.simulation_cfg is required for dynamic_ssl")

    @staticmethod
    def _resolve_simulation_cfg(section, split):
        from core.data.simulation_config_contract import validate_simulation_config_contract
        from core.data.simulation_configs import SimulationConfigs

        raw = section.get("simulation_cfg")
        if not isinstance(raw, Mapping):
            raise ValueError(f"data.{split}.simulation_cfg is required for formal dynamic SSL")
        if split == "train":
            validate_simulation_config_contract(
                task="ssl", split="train", train_section={"simulation_cfg": raw}
            )
        else:
            validate_simulation_config_contract(
                task="ssl",
                split="val",
                train_section={"simulation_cfg": {}},
                validation_section={"simulation_cfg": raw},
            )
        simulation = SimulationConfigs.from_dict(dict(raw), strict=True)
        points = section.get("trajectory_points", 1)
        if type(points) is not int or not 1 <= points <= 128:
            raise ValueError("trajectory_points must be an exact integer in [1,128]")
        if points > 1 and not (simulation.spatial_enabled and simulation.rir_backend == "online"):
            raise ValueError("moving SSL requires spatial online RIR")
        if simulation.prob_moving_rir > 0 and points < 2:
            raise ValueError("moving SSL requires trajectory_points >= 2")
        return simulation

    def build_model(self, cfg, rank):
        import tasks.ssl.losses  # noqa: F401 -- register SSL losses
        from tasks.ssl.models import build_model
        from tasks.ssl.model_io import (
            validate_ssl_loss_output_contract,
            validate_ssl_model_io,
        )

        model_cfg = dict(cfg["model"])
        model_args = dict(model_cfg.get("args", {}))
        model_args["max_track"] = resolve_ssl_num_tracks(cfg)
        model_cfg["args"] = model_args
        model = build_model(model_cfg, stft_cfg=cfg.get("stft"))
        validate_ssl_model_io(model, cfg.get("model_spec"))
        ssl_cfg = cfg.get("ssl") or {}
        validate_ssl_loss_output_contract(
            model,
            cfg.get("loss") or {},
            target_representation=ssl_cfg.get(
                "target_representation", "dpipd_far_field"
            ),
        )
        return model

    def get_trainer_cls(self):
        from tasks.ssl.trainer.ssl_trainer import SSLTrainer
        return SSLTrainer

    def _validate_config(self, cfg):
        from core.data.constants import normalize_data_mode

        super()._validate_config(cfg)
        mode = normalize_data_mode(cfg.get("data", {}).get("mode"))
        if mode.task != "ssl":
            raise ValueError("SSLTrainingEntry requires data.mode dynamic_ssl")
        model_cfg = cfg.get("model", {})
        model_type = model_cfg.get("type")
        official_models = {"FNSSL", "IPDNetFixed", "IPDNetVariable", "IPDNet2"}
        native_models = {"FixedArrayIPDNet"}
        if model_type not in official_models | native_models:
            raise ValueError(
                "SSL model.type must be one of FixedArrayIPDNet, FNSSL, "
                "IPDNetFixed, IPDNetVariable, IPDNet2"
            )
        model_args = model_cfg.get("args", {})
        if not isinstance(model_args, Mapping):
            raise TypeError("model.args must be a mapping")
        if type(model_args.get("input_size")) is not int:
            raise ValueError("SSL models require integer args.input_size")
        num_tracks = resolve_ssl_num_tracks(cfg)
        ssl_cfg = cfg.get("ssl")
        if ssl_cfg is None and model_type in native_models:
            # 旧 native smoke 配置可暂时省略 ssl 段；它的默认合同仍然明确，
            # 不会把它解释为官方 FN-SSL。
            ssl_cfg = {
                "input_representation": "reference_ipd",
                "target_representation": "dpipd_far_field",
                "time_stride": model_args.get("downsample", 1),
                "pair_layout": "reference_to_all",
                "geometry_mode": "fixed_array",
                "mapping": "far_field_grid",
                "coordinate_frame": "array",
            }
        if not isinstance(ssl_cfg, Mapping):
            raise ValueError("SSL model requires an explicit ssl mapping")
        unknown_ssl_keys = sorted(set(ssl_cfg) - _SSL_SECTION_KEYS - {"output_stride"})
        if unknown_ssl_keys:
            raise ValueError(f"unknown ssl keys: {unknown_ssl_keys}")
        input_rep = ssl_cfg.get("input_representation")
        target_rep = ssl_cfg.get("target_representation")
        allowed_inputs = {"multichannel_stft_real_imag", "reference_ipd"}
        allowed_targets = {"dpipd_far_field", "dpipd_near_field"}
        if input_rep not in allowed_inputs:
            raise ValueError(
                "ssl.input_representation must be multichannel_stft_real_imag or reference_ipd; "
                "legacy ipd is rejected"
            )
        if target_rep not in allowed_targets:
            raise ValueError(
                "ssl.target_representation must be dpipd_far_field or dpipd_near_field"
            )
        vad_mode = ssl_cfg.get("vad_mode", "direct_path")
        if vad_mode != "direct_path":
            raise ValueError("ssl.vad_mode currently supports only 'direct_path'")
        model_contract = _SSL_MODEL_CONTRACTS[model_type]
        expected_input = model_contract["input_representation"]
        if input_rep != expected_input:
            raise ValueError(
                f"{model_type} requires ssl.input_representation={expected_input!r}; "
                f"got {input_rep!r}"
            )
        expected_target = model_contract["target_representation"]
        if target_rep != expected_target:
            raise ValueError(
                f"{model_type} requires ssl.target_representation={expected_target!r}; "
                f"got {target_rep!r}"
            )
        model_inputs = ssl_cfg.get("model_inputs")
        if model_inputs is not None:
            if not isinstance(model_inputs, (list, tuple)) or list(model_inputs) != [expected_input]:
                raise ValueError(
                    f"ssl.model_inputs must be exactly [{expected_input!r}]"
                )
        source_distance_input = ssl_cfg.get("source_distance_input", False)
        if type(source_distance_input) is not bool:
            raise TypeError("ssl.source_distance_input must be a bool")
        if source_distance_input:
            raise ValueError(
                "ground-truth source distance cannot be an SSL model input; use target/mapping metadata"
            )
        if "causal" in model_args and type(model_args["causal"]) is not bool:
            raise TypeError("model.args.causal must be a bool")
        expected_causal = model_contract["causal"]
        actual_causal = model_args.get(
            "causal", True if expected_causal is None else expected_causal
        )
        if expected_causal is not None and actual_causal != expected_causal:
            raise ValueError(
                f"{model_type} requires model.args.causal={expected_causal!r}"
            )
        if "source_distance" in model_args or model_args.get("use_source_distance") is True:
            raise ValueError(
                "source_distance is target/mapping metadata and cannot be an IPDNet2 model input"
            )
        pair_layout = ssl_cfg.get("pair_layout", "reference_to_all")
        if pair_layout not in {"reference_to_all", "all_pairs", "pair_batched"}:
            raise ValueError("ssl.pair_layout must be reference_to_all, all_pairs or pair_batched")
        if pair_layout != "reference_to_all":
            raise ValueError(
                "dynamic_ssl currently supports only ssl.pair_layout='reference_to_all'; "
                "all_pairs and pair_batched require their dedicated projection/model implementation"
            )
        geometry_mode = ssl_cfg.get("geometry_mode", model_contract["geometry_mode"])
        if geometry_mode not in {"fixed_array", "variable_array"}:
            raise ValueError("ssl.geometry_mode must be fixed_array or variable_array")
        if geometry_mode != model_contract["geometry_mode"]:
            raise ValueError(
                f"{model_type} requires ssl.geometry_mode={model_contract['geometry_mode']!r}; "
                f"got {geometry_mode!r}"
            )
        mapping = ssl_cfg.get("mapping", model_contract["mapping"])
        if mapping not in {"far_field_grid", "near_field_doa_range_grid", "learned_range_head"}:
            raise ValueError("ssl.mapping is unsupported")
        if mapping != model_contract["mapping"]:
            raise ValueError(
                f"{model_type} requires ssl.mapping={model_contract['mapping']!r}; "
                f"got {mapping!r}"
            )
        time_reduction = ssl_cfg.get("time_reduction", model_contract["time_reduction"])
        if time_reduction is not None and time_reduction not in {
            "model_defined", "average_pool_12", "causal_conv_pool", "mamba_time_pool"
        }:
            raise ValueError("ssl.time_reduction is unsupported")
        if time_reduction != model_contract["time_reduction"]:
            raise ValueError(
                f"{model_type} requires ssl.time_reduction={model_contract['time_reduction']!r}; "
                f"got {time_reduction!r}"
            )
        coordinate_frame = ssl_cfg.get("coordinate_frame", "room_xyz_m")
        if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
            raise ValueError("ssl.coordinate_frame must be a non-empty string")
        exclude_dc = ssl_cfg.get("exclude_dc", False)
        if type(exclude_dc) is not bool:
            raise TypeError("ssl.exclude_dc must be a bool")
        if model_type == "FNSSL" and exclude_dc:
            # FNSSL._canonical_stft 要求完整 n_fft//2+1 频 bin，exclude_dc 会使频率维
            # 少 1 而在 forward 时才崩溃；本地限制在配置边界显式登记并 fail-fast。
            raise ValueError(
                "FNSSL requires full-band STFT bins (n_fft//2+1) and does not support "
                "ssl.exclude_dc=true"
            )
        normalization = ssl_cfg.get("normalization", "none")
        if normalization not in {"none", "forgetting_laplace", "global_mean"}:
            raise ValueError(
                "ssl.normalization must be none, forgetting_laplace or global_mean"
            )
        forgetting_norm_sample_length = ssl_cfg.get(
            "forgetting_norm_sample_length", 298
        )
        if (
            type(forgetting_norm_sample_length) is not int
            or forgetting_norm_sample_length < 1
        ):
            raise ValueError(
                "ssl.forgetting_norm_sample_length must be an exact integer >= 1"
            )
        inactive_manifold = ssl_cfg.get("inactive_target_manifold", "zero")
        if inactive_manifold not in {"zero", "bessel_j0"}:
            raise ValueError("ssl.inactive_target_manifold must be zero or bessel_j0")
        # 默认 +1 与 projection 的 others*conj(ref) 约定绑定（DOAMappingConfig 同值），
        # 镜像时必须由配置显式改回 -1.0。
        phase_sign = ssl_cfg.get("phase_sign", 1.0)
        if isinstance(phase_sign, bool) or phase_sign not in {-1, -1.0, 1, 1.0}:
            raise ValueError("ssl.phase_sign must be -1 or +1")
        from tasks.ssl.model_io import parse_model_spec

        model_spec_value = cfg.get("model_spec")
        if model_type in official_models and not isinstance(model_spec_value, Mapping):
            raise ValueError(
                f"{model_type} requires an explicit top-level model_spec mapping"
            )
        model_spec = (
            None
            if model_spec_value is None
            else parse_model_spec(model_spec_value, model_type=model_type)
        )
        if model_spec is not None:
            expected_field_mode = (
                "far_field" if target_rep == "dpipd_far_field" else "near_field"
            )
            expected_values = {
                "feature_contract": input_rep,
                "pair_layout": pair_layout,
                "geometry_mode": geometry_mode,
                "field_mode": expected_field_mode,
                "reduction": time_reduction,
                "causal": actual_causal,
                "mapping": mapping,
                # 公共 forward 始终返回末尾 real/imag 维；官方内部的
                # freq-concat 只属于实现细节，不能透出为运行时布局。
                "target_layout": model_contract["target_layout"],
            }
            ssl_field_names = {
                "feature_contract": "input_representation",
                "pair_layout": "pair_layout",
                "geometry_mode": "geometry_mode",
                "field_mode": "target_representation",
                "reduction": "time_reduction",
                "causal": "model.args.causal",
                "mapping": "mapping",
                "target_layout": "public model I/O target_layout",
            }
            for field, expected in expected_values.items():
                actual = getattr(model_spec, field)
                if actual != expected:
                    source = ssl_field_names[field]
                    source_label = (
                        source
                        if field in {"causal", "target_layout"}
                        else f"ssl.{source}"
                    )
                    raise ValueError(
                        f"model_spec.{field} must match {source_label}: "
                        f"expected {expected!r}, got {actual!r}"
                    )
            if time_reduction is not None and model_spec.reduction != time_reduction:
                raise ValueError(
                    "model_spec.reduction must match ssl.time_reduction: "
                    f"expected {time_reduction!r}, got {model_spec.reduction!r}"
                )
            if model_spec.target_contract != "dpipd_pair":
                raise ValueError(
                    "SSL dynamic projection requires model_spec.target_contract='dpipd_pair'"
                )
            if "causal" in model_args and model_spec.causal != model_args["causal"]:
                raise ValueError(
                    "model_spec.causal must match model.args.causal"
                )
        if "output_stride" in ssl_cfg:
            raise ValueError("ssl.output_stride is retired; use ssl.time_stride")
        stride = ssl_cfg.get("time_stride", model_args.get("stride", 1))
        if type(stride) is not int or stride < 1:
            raise ValueError("ssl.time_stride must be an exact positive integer")
        for key in ("stride", "time_stride", "output_stride"):
            if key in model_args and model_args[key] != stride:
                raise ValueError(f"ssl.time_stride must match model.args.{key}")
        loss_cfg = cfg.get("loss") or {}
        if not isinstance(loss_cfg, Mapping):
            raise TypeError("loss must be a mapping")
        loss_type = loss_cfg.get("type")
        allowed_losses = (
            {"SSLSpatialMSELoss", "SSLDOACompositeLoss"}
            if model_type in native_models
            else {"SSLDPIPDPITLoss", "SSLDOACompositeLoss"}
        )
        if loss_type not in allowed_losses:
            expected = " or ".join(sorted(allowed_losses))
            raise ValueError(f"{model_type} requires loss.type to be one of: {expected}")
        if loss_type == "SSLDOACompositeLoss":
            self._validate_composite_loss_config(
                loss_cfg, target_representation=target_rep
            )
        val_metrics = (cfg.get("trainer") or {}).get("val_metrics") or {}
        if val_metrics.get("enabled") is True:
            raise ValueError(
                "SSL does not support waveform trainer.val_metrics; "
                "set trainer.val_metrics.enabled=false"
            )
        for split in ("train", "validation"):
            section = self._dataset_config(cfg, split)
            self._validate_split_resources(mode.task_mode, section, split)
            simulation = self._resolve_simulation_cfg(section, split)
            from core.data.simulation.array_profiles import resolve_microphone_array

            geometry = resolve_microphone_array(
                array_profile=simulation.array_profile,
                custom_geometry=simulation.custom_geometry,
                reference_channel=simulation.reference_channel,
            )
            if geometry is None:
                raise ValueError(f"data.{split}.simulation_cfg requires microphone geometry")
            channels = geometry.channels
            if "channels" in section:
                raise ValueError(
                    f"data.{split}.channels is removed for dynamic SSL; "
                    "C is derived from simulation_cfg microphone geometry"
                )
            # 各模型输入通道口径：native/FNSSL/IPDNet2 以 C-1 登记（IPDNet2 内部
            # 再换算 dim_input=2C）；IPDnet fixed 消费 multichannel STFT（real+imag 共 2C）；
            # IPDnet variable 消费 pair-batched 2 麦 re/im（固定 4）。
            expected_input_size = {
                "FixedArrayIPDNet": channels - 1,
                "FNSSL": channels - 1,
                "IPDNet2": channels - 1,
                "IPDNetFixed": 2 * channels,
                "IPDNetVariable": 4,
            }[model_type]
            if channels < 2 or model_args["input_size"] != expected_input_size:
                raise ValueError(
                    f"data.{split} microphone geometry and model.args.input_size "
                    f"must satisfy input_size={expected_input_size} for {model_type}"
                )
            tracks = section.get("tracks", section.get("max_track", num_tracks))
            if type(tracks) is not int or tracks != num_tracks:
                raise ValueError(
                    f"data.{split} source count must match canonical ssl.num_tracks={num_tracks}"
                )
            expected_freqs = section["n_fft"] // 2 + 1 - int(ssl_cfg.get("exclude_dc", False))
            if "freqs" in section and section["freqs"] != expected_freqs:
                raise ValueError(f"data.{split}.freqs must equal n_fft//2+1-int(exclude_dc)")
