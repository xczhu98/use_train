import pytest
import torch

from core.losses.permutation import _permutation_table, solve_source_permutation


@pytest.mark.unit
def test_source_permutation_solver_aligns_estimates_from_pairwise_cost():
    cost = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    estimates = torch.tensor([[[10.0], [20.0]]])
    result = solve_source_permutation(cost, estimates=estimates)
    assert result.assignment.tolist() == [[1, 0]]
    assert result.aligned_estimates.tolist() == [[[20.0], [10.0]]]
    assert result.per_sample_cost.item() == pytest.approx(0.0)


@pytest.fixture()
def isolated_permutation_table_cache():
    """测试前后清空模块级 lru_cache，避免测试之间共享或污染缓存状态。"""
    _permutation_table.cache_clear()
    yield
    _permutation_table.cache_clear()


@pytest.mark.unit
def test_source_permutation_table_is_cached_per_shape_and_device(
    isolated_permutation_table_cache,
):
    first = _permutation_table(3, torch.device("cpu"))
    second = _permutation_table(3, torch.device("cpu"))
    assert first.shape == (6, 3)
    assert torch.equal(first, second)
    # 命中次数是可观察的缓存行为，替代对返回对象同一性的断言。
    assert _permutation_table.cache_info().hits == 1
