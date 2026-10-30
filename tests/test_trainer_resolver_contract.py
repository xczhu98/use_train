"""Task-local Trainer resolver contract tests."""

import pytest

from tasks.aec.trainer.entry import AECTrainingEntry
from tasks.se.trainer.entry import SETrainingEntry


pytestmark = pytest.mark.unit


def test_se_entry_resolves_default_trainer_short_name():
    trainer_cls = SETrainingEntry("test")._import_trainer_class("SETrainer")

    assert trainer_cls.__module__ == "tasks.se.trainer.se_trainer"
    assert trainer_cls.__name__ == "SETrainer"


def test_entry_validates_default_factory_result(monkeypatch):
    entry = SETrainingEntry("test")
    monkeypatch.setattr(entry, "get_trainer_cls", lambda: object)

    with pytest.raises(ValueError, match="must resolve to a BaseTrainer subclass"):
        entry._import_trainer_class("default")


def test_se_entry_resolves_task_local_extra_trainer_short_name():
    trainer_cls = SETrainingEntry("test")._import_trainer_class(
        "MetricGANCompositeTrainer"
    )

    assert trainer_cls.__module__ == "tasks.se.trainer.metricgan_trainer"
    assert trainer_cls.__name__ == "MetricGANCompositeTrainer"


def test_entry_resolves_fully_qualified_trainer_name():
    trainer_cls = SETrainingEntry("test")._import_trainer_class(
        "tasks.se.trainer.dual_input_loss_trainer.DualInputLossTrainer"
    )

    assert trainer_cls.__module__ == "tasks.se.trainer.dual_input_loss_trainer"
    assert trainer_cls.__name__ == "DualInputLossTrainer"


def test_fully_qualified_resolution_does_not_call_default_factory(monkeypatch):
    entry = SETrainingEntry("test")

    def fail_default_factory():
        raise AssertionError("FQCN resolution must not call the default factory")

    monkeypatch.setattr(entry, "get_trainer_cls", fail_default_factory)
    trainer_cls = entry._import_trainer_class(
        "tasks.se.trainer.dual_input_loss_trainer.DualInputLossTrainer"
    )

    assert trainer_cls.__name__ == "DualInputLossTrainer"


@pytest.mark.parametrize(
    "trainer_name",
    ["core.trainer.base_trainer.logger", "torch.nn.Linear"],
)
def test_entry_rejects_fqcn_that_is_not_a_trainer_class(trainer_name):
    with pytest.raises(ValueError, match="must resolve to a BaseTrainer subclass"):
        SETrainingEntry("test")._import_trainer_class(trainer_name)


def test_aec_entry_resolves_only_aec_task_short_names():
    entry = AECTrainingEntry("test")

    trainer_cls = entry._import_trainer_class("AECTrainer")
    assert trainer_cls.__module__ == "tasks.aec.trainer.aec_trainer"
    assert trainer_cls.__name__ == "AECTrainer"

    with pytest.raises(ValueError, match="Unknown trainer class 'SETrainer'"):
        entry._import_trainer_class("SETrainer")


def test_entry_rejects_unknown_task_local_short_name():
    with pytest.raises(ValueError, match="Unknown trainer class 'MissingTrainer'"):
        SETrainingEntry("test")._import_trainer_class("MissingTrainer")
