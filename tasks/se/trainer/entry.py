#!/usr/bin/env python3
"""SE training entry point.

The unified, task-agnostic base class lives in ``core.trainer.entry``
(``UnifiedTrainingEntry``, ``TaskInfo``, ``detect_task``,
``set_spawn_start_method``); this module hosts only the SE-specific
concrete subclass.
"""

from __future__ import annotations

from functools import partial

import torch

from core.trainer.entry import UnifiedTrainingEntry


# ===================================================================
# SE Training Entry
# ===================================================================
class SETrainingEntry(UnifiedTrainingEntry):
    def build_loaders(self, cfg: dict, rank: int, local_rank: int, world_size: int):
        from tasks.task_api import load_task_data_api

        data_api = load_task_data_api("se")

        train_ds = data_api.build_dataset(cfg["data"], mode="train")
        val_ds = data_api.build_dataset(cfg["data"], mode="val", meta=cfg.get("meta"))

        def _collate_for_dataset(dataset):
            """仅为显式 spatial_enabled 的 Dataset 开启空间 noisy 合同。"""
            simulation_cfg = getattr(dataset, "sim_cfg", None)
            spatial_enabled = getattr(simulation_cfg, "spatial_enabled", False)
            if type(spatial_enabled) is not bool:
                raise TypeError("Dataset sim_cfg.spatial_enabled must be an exact bool")
            return (
                partial(data_api.collate, allow_spatial_noisy=True)
                if spatial_enabled
                else data_api.collate
            )

        return self._build_loaders_common(
            cfg, rank, local_rank, world_size,
            train_ds=train_ds,
            val_ds=val_ds,
            train_collate_fn=_collate_for_dataset(train_ds),
            val_collate_fn=_collate_for_dataset(val_ds),
            train_num_per_epoch=cfg["data"].get("train", {}).get("num_per_epoch"),
            val_num_per_epoch=cfg["data"].get("validation", {}).get("num_per_epoch"),
        )

    def build_model(self, cfg: dict, rank: int) -> torch.nn.Module | dict[str, torch.nn.Module]:
        from tasks.se.models import build_model
        import core.losses  # noqa: F401 -- registers loss classes into LOSS_REGISTRY
        # before run_training() calls core.losses.build.build_loss().

        model_cfg = cfg.get("model", {})
        stft_cfg = cfg.get("stft")

        # GAN mode: dict of models (generator + discriminator)
        if "generator" in model_cfg:
            models = {}
            gen_cfg = model_cfg["generator"]
            gen = build_model(gen_cfg, stft_cfg=stft_cfg)
            models["generator"] = gen
            disc_cfg = model_cfg.get("discriminator")
            if disc_cfg:
                # Discriminator typically does not need STFT params injected
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

        # Single model mode
        model = build_model(model_cfg, stft_cfg=stft_cfg)
        if rank == 0:
            model_desc = model_cfg.get("type") or model_cfg.get("path", "model")
            self.logger.info(f"Model: {model_desc} | {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
        return model

    def build_model_for_training(self, cfg: dict, rank: int, *, train_dataset):
        import tasks.se.models
        from core.models.io_contract import inject_dataset_model_geometry, validate_spatial_model_geometry

        inject_dataset_model_geometry(cfg, train_dataset)
        model = self.build_model(cfg, rank)
        validate_spatial_model_geometry(model, train_dataset)
        return model

    def get_trainer_cls(self):
        from tasks.se.trainer import SETrainer

        return SETrainer

    def _validate_config(self, cfg: dict) -> None:
        """校验 SE 配置，并为交替 GAN 显式推导 DDP reducer 选项。"""
        super()._validate_config(cfg)
        model_cfg = cfg.get("model", {})
        trainer_type = str(cfg.get("trainer", {}).get("type", "")).strip()
        is_gan = "generator" in model_cfg or trainer_type.endswith("GANTrainer")
        if not is_gan:
            return

        data_cfg = cfg.get("data", {})
        data_sections = [
            data_cfg,
            data_cfg.get("train", {}) if isinstance(data_cfg, dict) else {},
            data_cfg.get("validation", {}) if isinstance(data_cfg, dict) else {},
        ]
        if any(
            isinstance(section, dict)
            and isinstance(section.get("simulation_cfg"), dict)
            and section["simulation_cfg"].get("spatial_enabled") is True
            for section in data_sections
        ):
            raise ValueError(
                "SE GAN trainers do not support spatial_enabled=True; "
                "use a spatial-capable GAN trainer or a non-GAN SpatialNet trainer."
            )

        distributed_cfg = cfg.setdefault("distributed", {})
        configured = distributed_cfg.get("find_unused_parameters")
        if configured is False:
            raise ValueError(
                "SE GAN trainers require distributed.find_unused_parameters=true: "
                "alternating generator/discriminator freezing leaves one DDP "
                "wrapper unused during each backward pass."
            )
        distributed_cfg["find_unused_parameters"] = True

    def _default_gan_trainer_cls(self):
        from tasks.se.trainer.gan_trainer import GANTrainer

        return GANTrainer
