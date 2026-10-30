#!/usr/bin/env python3
"""AEC training entry point.

Hosts ``AECTrainingEntry``, the AEC-specific concrete subclass of the
task-agnostic ``core.trainer.entry.UnifiedTrainingEntry``.
"""

from __future__ import annotations

from core.data.constants import DATA_MODE_DYNAMIC_AEC
from core.data.constants import DATA_MODE_OFFLINE_AEC
from core.data.constants import normalize_data_mode
from core.trainer.entry import (
    UnifiedTrainingEntry,
    detect_task,
    set_spawn_start_method,
)


# Re-export detect_task / set_spawn_start_method so callers can import everything
# they need from either ``core.trainer.entry`` or ``tasks.aec.trainer.entry``.
__all__ = ["AECTrainingEntry", "detect_task", "set_spawn_start_method"]


def _aec_dataset_is_multichannel(dataset) -> bool:
    """从已解析 geometry 判断 AEC dataset 是否真正有 microphone 轴。"""
    from core.models.io_contract import resolve_dataset_microphone_array
    geometry = resolve_dataset_microphone_array(dataset)
    return geometry is not None and geometry.channels > 1


class AECTrainingEntry(UnifiedTrainingEntry):
    def build_model_for_training(self, cfg: dict, rank: int, *, train_dataset):
        import tasks.aec.models
        from core.models.io_contract import inject_dataset_model_geometry, validate_spatial_model_geometry
        from tasks.aec.model_io import validate_aec_multichannel_contract

        inject_dataset_model_geometry(cfg, train_dataset)
        model = self.build_model(cfg, rank)
        validate_spatial_model_geometry(model, train_dataset)
        if _aec_dataset_is_multichannel(train_dataset):
            primary = model.get("generator") if isinstance(model, dict) else model
            validate_aec_multichannel_contract(primary)
        return model

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------
    def _validate_config(self, cfg: dict) -> None:
        """Fail-fast validation of required config keys.

        校验 AEC frontend 与 dynamic_aec Dataset 配置；frontend 在 STFT 前由
        Entry/Trainer 运行，不属于 Dataset 仿真。
        """
        if "aec_multichannel_adapter" in cfg:
            raise ValueError(
                "aec_multichannel_adapter is removed; use a native model IO "
                "contract with a microphone axis"
            )
        super()._validate_config(cfg)
        from tasks.aec.inference.aec_frontend import resolve_aec_frontend_config

        frontend_cfg = resolve_aec_frontend_config(cfg)
        data_cfg = cfg["data"]
        data_mode = normalize_data_mode(data_cfg.get("mode")).task_mode
        if data_mode == DATA_MODE_OFFLINE_AEC:
            if frontend_cfg.linear_aec is not None:
                for section_name in ("train", "validation"):
                    section = data_cfg.get(section_name)
                    if isinstance(section, dict) and section.get("error_scp"):
                        raise ValueError(
                            "runtime LinearAEC conflicts with pre-generated "
                            f"data.{section_name}.error_scp; disable runtime linear_aec "
                            "or remove error_scp"
                        )
            return
        if data_mode != DATA_MODE_DYNAMIC_AEC:
            return
        from tasks.aec.data.dynamic_aec_build import validate_aec_dynamic_aec_config

        validate_aec_dynamic_aec_config(cfg)

    # ------------------------------------------------------------------
    # State-detector config derivation / validation
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_state_detector_cfg(cfg: dict) -> None:
        """为 DBGNet 推导并校验 ``state_nclass`` / ``use_state_detector``。

        ``data.train.state_detector_label_type`` 是类别数的单一事实来源，
        ``StateCrossEntropyLoss`` 是否存在决定是否启用 state-detector head。
        未启用 StateCE 时移除 Dataset 的标签生成入口，避免关闭 state head 后
        继续计算并 collate 无人消费的逐帧标签。普通 model 与 GAN generator
        使用同一规则；其他模型不注入 DBGNet 专属参数，并在配置
        StateCrossEntropyLoss 时明确失败。
        """
        data_cfg = cfg.get("data", {})
        train_cfg = data_cfg.get("train") or {}
        label_type = train_cfg.get("state_detector_label_type")

        loss_cfg = cfg.get("loss", {})
        loss_type = loss_cfg.get("type")
        if loss_type == "StateCrossEntropyLoss":
            raise ValueError(
                "StateCrossEntropyLoss must be a CompositeLoss component, not the top-level loss."
            )
        components = loss_cfg.get("components") if loss_type == "CompositeLoss" else []
        has_state_ce = any(
            isinstance(component, dict)
            and component.get("type") == "StateCrossEntropyLoss"
            and float(component.get("weight", 1.0)) != 0.0
            for component in (components or [])
        )
        has_primary_loss = any(
            isinstance(component, dict)
            and component.get("type") != "StateCrossEntropyLoss"
            and float(component.get("weight", 1.0)) != 0.0
            for component in (components or [])
        )
        if has_state_ce and not has_primary_loss:
            raise ValueError(
                "StateCrossEntropyLoss is auxiliary-only and requires at least one "
                "non-zero non-StateCrossEntropyLoss CompositeLoss component."
            )

        if not has_state_ce:
            # validation 默认继承 train；只保留阈值等惰性配置，不允许 hop/win
            # 单独触发标签生成。
            for section_name in ("train", "validation"):
                section = data_cfg.get(section_name)
                if not isinstance(section, dict):
                    continue
                section.pop("state_detector_label_type", None)
                section.pop("state_detector_hop_length", None)
                section.pop("state_detector_win_length", None)
            data_cfg.pop("state_detector_label_type", None)
            data_cfg.pop("state_detector_hop_length", None)
            data_cfg.pop("state_detector_win_length", None)
            label_type = None

        model_cfg = cfg.get("model", {})
        target_cfg = model_cfg.get("generator", {}) if "generator" in model_cfg else model_cfg
        if "path" in target_cfg:
            model_name = str(target_cfg.get("path", "")).rsplit(".", 1)[-1]
        else:
            model_name = target_cfg.get("type")
        if model_name != "DBGNet":
            if has_state_ce:
                raise ValueError(
                    "Loss contains StateCrossEntropyLoss, but the configured model "
                    f"{model_name or '<unknown>'!r} is not DBGNet and has no state-detector head."
                )
            return

        target_args = target_cfg.get("args") or {}
        expected_nclass = None
        if label_type is not None:
            label_to_nclass = {"fst_dt_nst": 3, "scenario": 8}
            if label_type not in label_to_nclass:
                raise ValueError(
                    f"Unknown state_detector_label_type={label_type!r}. "
                    f"Supported: {list(label_to_nclass.keys())}."
                )
            expected_nclass = label_to_nclass[label_type]
            if (
                "state_nclass" in target_args
                and int(target_args["state_nclass"]) != expected_nclass
            ):
                raise ValueError(
                    f"model.args.state_nclass={target_args['state_nclass']} does not match "
                    f"data.train.state_detector_label_type={label_type!r} (expected {expected_nclass})."
                )

        if has_state_ce and label_type is None:
            raise ValueError(
                "Loss contains StateCrossEntropyLoss but data.train.state_detector_label_type "
                "is not set. Cannot generate state labels."
            )

        if (
            "use_state_detector" in target_args
            and bool(target_args["use_state_detector"]) != bool(has_state_ce)
        ):
            raise ValueError(
                f"model.args.use_state_detector={target_args['use_state_detector']} is "
                f"inconsistent with StateCrossEntropyLoss presence in loss (has_state_ce={has_state_ce})."
            )

        resolved_args = dict(target_args)
        if expected_nclass is not None:
            resolved_args.setdefault("state_nclass", expected_nclass)
        resolved_args.setdefault("use_state_detector", bool(has_state_ce))
        target_cfg["args"] = resolved_args

    def build_loaders(self, cfg: dict, rank: int, local_rank: int, world_size: int):
        from functools import partial

        from tasks.task_api import load_task_data_api
        from core.data.config_flags import (
            resolve_placeholder_num_samples,
            resolve_strict_load,
        )

        data_api = load_task_data_api("aec")
        aec_collate_fn = data_api.resolve("tasks.aec.data.collate:aec_collate_fn")
        build_dynamic_aec_dataset = data_api.resolve(
            "tasks.aec.data.dynamic_aec_build:build_dynamic_aec_dataset"
        )

        self._resolve_state_detector_cfg(cfg)
        data_cfg = cfg["data"]
        train_cfg = data_cfg.get("train") or {}
        val_cfg = data_cfg.get("validation") or {}
        stft_cfg = cfg.get("stft")
        data_mode = normalize_data_mode(data_cfg.get("mode")).task_mode
        if data_mode not in (DATA_MODE_DYNAMIC_AEC, DATA_MODE_OFFLINE_AEC):
            raise ValueError(
                f"AEC training only supports data.mode 'dynamic_aec' / 'offline_aec' "
                f"(got {data_mode!r}). Use SE train.py for offline denoising."
            )

        ph_aec = max(1, int(resolve_placeholder_num_samples(cfg)))

        if (data_cfg.get("train") or {}).get("sources"):
            raise ValueError("data.train.sources is not supported (only dynamic_aec; use a single data.train block).")

        if data_mode == DATA_MODE_OFFLINE_AEC:
            # 离线预仿真三元组：mic/farend/nearend [+error] scp 配对
            if rank == 0:
                self.logger.info(
                    "data.mode=offline_aec: mic/farend/nearend pre-simulated triplets from scp manifests."
                )
            build_offline_aec_dataset = data_api.resolve(
                "tasks.aec.data.offline_dataset:build_offline_aec_dataset"
            )
            train_ds = build_offline_aec_dataset(cfg, "train")
            val_ds = build_offline_aec_dataset(cfg, "val")
        else:
            if rank == 0:
                self.logger.info(
                    "data.mode=dynamic_aec: nearend = DynamicMixingDataset (Layer A–F); "
                    "farend+echo = real_farend_echo_pair_json (.json; default stereo ch0=echo ch1=farend); mic = noisy_nearend + echo. "
                    "AEC input/deployment frontend runs before STFT."
                )
            train_ds = build_dynamic_aec_dataset(cfg, "train", stft_cfg=stft_cfg)
            val_ds = build_dynamic_aec_dataset(cfg, "val", stft_cfg=stft_cfg)
        base_seed = int(cfg.get("meta", {}).get("seed", 0))
        if hasattr(val_ds, "set_epoch"):
            val_ds.set_epoch(0, base_seed=base_seed)

        sr_collate = int(data_cfg.get("sample_rate", 16000))
        md_raw = data_cfg.get("max_duration", 0) or 0
        md_sec = float(md_raw)
        if md_sec > 0:
            collate_placeholder_T = max(1024, int(md_sec * sr_collate))
        else:
            collate_placeholder_T = max(1024, int(ph_aec))
        collate_raise = bool(data_cfg.get("aec_collate_raise_on_all_failed", False))
        if world_size > 1:
            # DDP 下某个 rank 的全失败 placeholder 会改变有效样本权重；
            # 与部分 None 的 fail-fast 语义保持一致，统一在 collate 边界终止。
            collate_raise = True
        if collate_raise and rank == 0:
            self.logger.info("data.aec_collate_raise_on_all_failed=true: empty batches will raise instead of silent placeholder.")
        model_cfg = cfg.get("model", {})
        target_cfg = model_cfg.get("generator", {}) if "generator" in model_cfg else model_cfg
        target_args = target_cfg.get("args") or {}
        aec_collate_bound = partial(
            aec_collate_fn,
            placeholder_sr=sr_collate,
            placeholder_samples=collate_placeholder_T,
            raise_on_all_failed=collate_raise,
            expect_state=bool(target_args.get("use_state_detector", False)),
            # strict_load=True 时 Dataset 已校验波形有限性，跳过 collate 的逐样本
            # O(T) 兜底扫描；strict_load=False 保留兜底。
            check_finite=not resolve_strict_load(cfg),
        )

        return self._build_loaders_common(
            cfg, rank, local_rank, world_size,
            train_ds=train_ds,
            val_ds=val_ds,
            train_collate_fn=aec_collate_bound,
            train_num_per_epoch=train_cfg.get("num_per_epoch"),
            val_num_per_epoch=val_cfg.get("num_per_epoch"),
        )

    def build_model(self, cfg: dict, rank: int) -> "torch.nn.Module | dict[str, torch.nn.Module]":
        import tasks.aec.models as aec_models
        import torch
        from core.models.build import build_model
        import core.losses  # noqa: F401 -- registers shared loss classes before
        # run_training() calls
        # core.losses.build.build_loss().

        self._resolve_state_detector_cfg(cfg)
        model_cfg = cfg.get("model", {})
        stft_cfg = cfg.get("stft")

        if "generator" in model_cfg:
            models = {}
            gen_cfg = model_cfg["generator"]
            _mt = gen_cfg.get("type")
            if _mt:
                aec_models.register_aec_models(only=str(_mt))
            else:
                aec_models.register_aec_models()
            gen = build_model(gen_cfg, stft_cfg=stft_cfg)
            models["generator"] = gen
            disc_cfg = model_cfg.get("discriminator")
            if disc_cfg:
                disc = build_model(disc_cfg)
                models["discriminator"] = disc
            if rank == 0:
                gen_desc = gen_cfg.get("type") or gen_cfg.get("path", "generator")
                total_params = sum(p.numel() for m in models.values() for p in m.parameters())
                self.logger.info(
                    f"Models (GAN): generator={gen_desc} | "
                    f"total_params={total_params/1e6:.2f}M | "
                    f"components={list(models.keys())}"
                )
            return models

        _mt = model_cfg.get("type")
        if _mt:
            aec_models.register_aec_models(only=str(_mt))
        else:
            aec_models.register_aec_models()
        model = build_model(model_cfg, stft_cfg=stft_cfg)
        if rank == 0:
            model_desc = model_cfg.get("type") or model_cfg.get("path", "model")
            self.logger.info(f"Model: {model_desc} | {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
        return model

    def get_trainer_cls(self):
        from tasks.aec.trainer.aec_trainer import AECTrainer

        return AECTrainer
