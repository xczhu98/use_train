"""权重感知 batch 切分器。

Step2 MOS 批处理中，不同模型的显存峰值与「批内条目」的度量不同：
- DNSMOS：显存 ∝ ``run_batch_audio`` 的 segment 数（9.01s 窗 / 1s hop / 短音频补长）
- UTMOS / NISQA / SCOREQ：显存 ∝ 批内总时长（秒）

本切分器把「按条目权重累加切批」的逻辑从 ``compute_mos_scores_batch``
中抽出为独立可测组件：estimator 声明 ``batch_weight``（单条权重）与
``batch_budget``（批预算，权重单位），切分器按权重累加切批，单条超限
时独占一批（无法再拆分，交由外层 OOM 兜底）。
"""

from typing import Callable, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class WeightedBatchSplitter(Iterator[List[T]]):
    """按条目权重累加切分 items 为多个 batch。

    语义（与旧版时长感知切批一致）：
    - 第一条无条件接受（单条超限无法再拆分，交由外层 OOM 减半/逐条兜底）
    - 后续条目仅当「已累计权重 + 本条权重 ≤ 预算」时加入
    - ``max_items`` 作为条数上限（防止权重过小时 batch 无限大）

    ``budget`` 是实例可变属性：外层在 OOM 时下调 ``splitter.budget``，
    后续迭代自动按新预算切批（显存自适应下限）。
    """

    def __init__(
        self,
        items: List[T],
        weight_fn: Callable[[T], float],
        budget: float,
        max_items: Optional[int] = None,
    ):
        self.items = items
        self.weight_fn = weight_fn
        self.budget = max(1.0, float(budget))
        self.max_items = max(1, int(max_items)) if max_items else None
        self._pos = 0

    def __iter__(self) -> "WeightedBatchSplitter":
        return self

    def __next__(self) -> List[T]:
        n = len(self.items)
        if self._pos >= n:
            raise StopIteration
        i = self._pos
        j = i
        w = 0.0
        while j < n and (self.max_items is None or (j - i) < self.max_items):
            item_w = self.weight_fn(self.items[j])
            if j > i and w + item_w > self.budget:
                break
            w += item_w
            j += 1
        batch = self.items[i:j]
        self._pos = j
        return batch
