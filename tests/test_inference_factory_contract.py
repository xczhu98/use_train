"""任务 descriptor 驱动的文件推理 factory 合同。"""

import pytest


pytestmark = pytest.mark.unit


def test_task_descriptor_inference_factory_dispatches_task_owned_factory(monkeypatch):
    import tasks.registry as registry

    class FactoryModule:
        @staticmethod
        def build_inference_adapter(*args, **kwargs):
            return args, kwargs

    monkeypatch.setattr(
        registry.importlib,
        "import_module",
        lambda module_path: FactoryModule
        if module_path == "tasks.se.inference.adapter"
        else pytest.fail(module_path),
    )

    assert registry.build_inference_adapter("se", "model", sample_rate=16000) == (
        ("model",),
        {"sample_rate": 16000},
    )


@pytest.mark.parametrize(
    ("task", "pattern"),
    [
        ("ss", "SS root file inference is unavailable"),
        ("ssl", "SSL root file inference is unavailable"),
    ],
)
def test_task_descriptor_factory_fails_fast_for_non_public_file_tasks(task, pattern):
    from tasks.registry import build_inference_adapter

    with pytest.raises(ValueError, match=pattern):
        build_inference_adapter(task, object())


def test_descriptor_factory_builds_aec_from_root_stft_cfg():
    import torch

    from tasks.aec.inference import AECInferenceAdapter
    from tasks.registry import build_inference_adapter

    class Model(torch.nn.Module):
        def forward(self, batch):
            return {"nearend_estimate_spectra": batch["mic_spectra"]}

    adapter = build_inference_adapter(
        "aec",
        Model(),
        stft_cfg={"n_fft": 32, "hop_length": 16, "win_length": 32},
        model_fixed_sr=16000,
    )

    assert isinstance(adapter, AECInferenceAdapter)
    assert adapter.sample_rate == 16000
