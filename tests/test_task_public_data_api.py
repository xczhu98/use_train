import pickle
from functools import partial

import pytest


@pytest.mark.unit
def test_task_data_api_pickle_safe_for_spawn_dataloader():
    """train.py 强制 spawn start method，DataLoader worker 会 pickle entry 传入的
    ``partial(api.collate, ...)``；TaskDataAPI 不得因持有 module 引用而不可 pickle。"""
    from tasks.registry import iter_tasks
    from tasks.task_api import load_task_data_api

    for descriptor in iter_tasks():
        api = load_task_data_api(descriptor.name)
        pickle.dumps(api)
        pickle.dumps(partial(api.collate))


@pytest.mark.unit
def test_all_task_descriptors_expose_data_modules():
    from tasks.registry import iter_tasks
    from tasks.task_api import load_task_data_api

    for descriptor in iter_tasks():
        api = load_task_data_api(descriptor.name)
        assert api.module.__name__ == descriptor.data_module
        assert api.collate_function is not None
        if descriptor.data_builder is not None:
            assert callable(getattr(api.module, descriptor.data_builder))


@pytest.mark.unit
def test_tse_data_module_unifies_dynamic_and_offline_entrypoints():
    from tasks.task_api import load_task_data_api

    api = load_task_data_api("tse")
    assert api.task == "tse"
    assert api.dataset_builder == "build_dataset"
    assert api.collate_function == "tasks.tse.data.collate:tse_collate_fn"
    assert api.dataset_validator == "tasks.tse.data.interface:validate_tse_dataset_interface"


@pytest.mark.unit
def test_cue_frontend_registry_is_explicit_and_kind_scoped():
    from tasks.tse.models.modular_spec import resolve_modular_spec

    spec = resolve_modular_spec(
        backbone="bsrnn",
        separator={"feature_dim": 8, "num_repeat": 1, "causal": False},
        cues={
            "speaker": {
                "frontend": "speaker",
                "representation": "embedding",
                "features": {"spkemb": {}},
            }
        },
        composition=None,
        sample_rate=16000,
        n_fft=16,
        hop_length=4,
        win_length=16,
        window="hann",
        center=True,
        n_channel=1,
        ref_channel=0,
        require_implemented=False,
    )
    assert spec.cues["speaker"].frontend == "speaker"

    with pytest.raises(ValueError, match="belongs to cue kind"):
        resolve_modular_spec(
            backbone="bsrnn",
            separator={"feature_dim": 8, "num_repeat": 1, "causal": False},
            cues={
                "speaker": {
                    "frontend": "spatial",
                    "representation": "embedding",
                    "features": {"spkemb": {}},
                }
            },
            composition=None,
            sample_rate=16000,
            n_fft=16,
            hop_length=4,
            win_length=16,
            window="hann",
            center=True,
            n_channel=1,
            ref_channel=0,
            require_implemented=False,
        )
