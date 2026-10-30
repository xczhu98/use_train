"""Speech Separation training entry point."""

from __future__ import annotations

from core.trainer.entry import UnifiedTrainingEntry


class SSTrainingEntry(UnifiedTrainingEntry):
    """Build the SS-specific dataset, model, and trainer components."""

    def build_loaders(self, cfg: dict, rank: int, local_rank: int, world_size: int):
        from tasks.task_api import load_task_data_api

        data_api = load_task_data_api("ss")

        train_ds = data_api.build_dataset(cfg, mode="train")
        val_ds = data_api.build_dataset(cfg, mode="val")
        return self._build_loaders_common(
            cfg,
            rank,
            local_rank,
            world_size,
            train_ds=train_ds,
            val_ds=val_ds,
            train_collate_fn=data_api.collate,
            train_num_per_epoch=cfg["data"].get("train", {}).get("num_per_epoch"),
            val_num_per_epoch=cfg["data"].get("validation", {}).get("num_per_epoch"),
        )

    def build_model(self, cfg: dict, rank: int):
        import core.losses  # noqa: F401 -- register shared losses before build_loss
        from tasks.ss.models import build_model

        return build_model(
            cfg["model"],
            stft_cfg=cfg.get("stft"),
            num_sources=int(cfg["ss"]["num_sources"]),
        )

    def build_model_for_training(self, cfg: dict, rank: int, *, train_dataset):
        import tasks.ss.models
        from core.models.io_contract import inject_dataset_model_geometry, validate_spatial_model_geometry

        inject_dataset_model_geometry(cfg, train_dataset)
        model = self.build_model(cfg, rank)
        validate_spatial_model_geometry(model, train_dataset)
        return model

    def get_trainer_cls(self):
        from tasks.ss.trainer.ss_trainer import SSTrainer

        return SSTrainer

    def _validate_config(self, cfg: dict) -> None:
        super()._validate_config(cfg)
        from tasks.ss.data.config import SSDatasetConfig

        ss_cfg = cfg.get("ss")
        if not isinstance(ss_cfg, dict):
            raise ValueError("SS configs require a top-level 'ss' mapping")
        num_sources = ss_cfg.get("num_sources")
        if type(num_sources) is not int or num_sources < 1:
            raise ValueError("ss.num_sources must be an exact positive int")
        explicit_num_sources = (cfg.get("model", {}).get("args") or {}).get(
            "num_sources"
        )
        if explicit_num_sources is not None and explicit_num_sources != num_sources:
            raise ValueError(
                "model.args.num_sources conflicts with ss.num_sources"
            )
        SSDatasetConfig.from_config(cfg, "train")
        SSDatasetConfig.from_config(cfg, "validation")
