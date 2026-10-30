"""Regression coverage for the MetricGAN epoch-level scheduler recipe."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from core.trainer.base_trainer import BaseTrainer
from core.utils.config import load_merged_config
from tasks.aec.trainer.entry import AECTrainingEntry


pytestmark = pytest.mark.unit

_CONFIGS = (
    "configs/se/urgent2026/gan/mossformer_gan_metricgan_dynamic.yaml",
    "configs/se/urgent2026/gan/mp_senet_metricgan_dynamic.yaml",
)


def _entry() -> AECTrainingEntry:
    return object.__new__(AECTrainingEntry)


class _RecipeTrainer(BaseTrainer):
    def _save_experiment_context(self):
        pass

    def _train_epoch(self, epoch, step_callback=None):
        expected_lr = 0.1 * 0.99 ** (epoch - 1)
        for _ in range(3):
            for name in ("discriminator", "generator"):
                model = self.models[name]
                model(torch.ones(1, 1, device=self.device)).sum().backward()
                assert self._optimizer_step(
                    name,
                    update_scaler=name == "generator",
                    increment_step=name == "generator",
                )
                assert self.optimizers[name].param_groups[0]["lr"] == pytest.approx(
                    expected_lr
                )
        return 0.0

    def _save_checkpoint(self, epoch, **_kwargs):
        self.epoch_lrs.append(
            (epoch, {name: opt.param_groups[0]["lr"] for name, opt in self.optimizers.items()})
        )


@pytest.mark.parametrize("config_path", _CONFIGS)
def test_metricgan_configs_use_epoch_exponential_decay_on_both_gan_optimizers(
    config_path: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = deepcopy(load_merged_config(config_path))
    config["meta"].update(use_amp=False, use_ema=False, compile_model=False)
    config["trainer"].update(
        exp_path=str(tmp_path / Path(config_path).stem),
        epochs=2,
        validation_epoch_interval=0,
        validation_steps_interval=0,
        save_checkpoint_interval=1,
    )
    models = {
        "generator": torch.nn.Linear(1, 1),
        "discriminator": torch.nn.Linear(1, 1),
    }
    optimizers = {
        name: torch.optim.SGD(model.parameters(), lr=0.1)
        for name, model in models.items()
    }
    schedulers = _entry().build_scheduler(config, 3, optimizers, world_size=1)
    trainer = _RecipeTrainer(
        rank=0,
        world_size=1,
        model=models,
        optimizer=optimizers,
        scheduler=schedulers,
        loss_fn=torch.nn.MSELoss(),
        config=config,
    )
    trainer.epoch_lrs = []
    try:
        trainer.train()
    finally:
        if trainer.writer is not None:
            trainer.writer.close()

    assert trainer.n_steps == 6
    assert [epoch for epoch, _ in trainer.epoch_lrs] == [1, 2]
    for epoch, lrs in trainer.epoch_lrs:
        assert set(lrs) == {"generator", "discriminator"}
        for lr in lrs.values():
            assert lr == pytest.approx(0.1 * 0.99**epoch)
