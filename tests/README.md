# 测试组织与 marker 约定

本目录是仓库的自动化测试套件：每个 `test_*.py` 文件里是一组断言，跑一遍就知道改动有没有弄坏别的地方。

**为什么需要这篇文档**：这里的测试有的几毫秒跑完，有的要起子进程、要两张显卡、要下载外部模型。默认只跑「快而稳定」的那一批，
其余用 pytest 的 *marker*（标签，写在测试函数上的 `@pytest.mark.xxx`）按需挑选。看懂标签语义，才知道自己该跑哪一批。

## 快速对照

| 我想…… | 命令 |
|---|---|
| 跑默认测试集合 | `python -m pytest` |
| 只跑不依赖子进程 / 显卡 / 外部资源的测试 | `python -m pytest -m "unit and not subprocess and not ddp and not ort and not pipeline"` |
| 跑单个文件 | `python -m pytest tests/test_loss_build.py` |
| 跑多卡 DDP 测试（需真实 GPU） | `python -m pytest -m ddp` |
| 跑 ONNX Runtime 测试（需 onnxruntime） | `python -m pytest -m ort` |

## 默认收集

在仓库根目录执行 `python -m pytest` 时，pytest 按默认文件模式递归收集 `tests/`，
并按根目录 `pyproject.toml` 里的 `addopts = '-m "unit"'` 只选 `unit` 标记的测试。

`subprocess`、`ddp`、`ort` 等资源 marker 只提供显式筛选能力，不会从默认集合中
静默排除已标记为 `unit` 的既有测试。需要纯 hermetic 子集时，由调用方显式排除
资源 marker。非 pytest 的运维、比较或验证 CLI 应放在 `tools/ops/` 或 `tools/verify/`；
`tests/` 不存放 CLI shim。

任务推理合同测试统一由本目录收集：

| 任务 | 测试文件 |
|---|---|
| SE | `test_se_inference_adapter.py` |
| AEC | `test_aec_inference_adapter.py` |
| SS | `test_ss_inference_adapter.py` |
| TSE | `test_tse_inference_adapter.py`、`test_tse_root_manifest.py`、`test_tse_inference_sidecar.py` |

这些用例均声明 `unit` marker；任务源码目录不再保留重复测试副本。

## Marker 语义

- `unit`：仓库默认集合，包含纯逻辑、契约测试及少量资源型测试。
- `contract`：公共 API、配置、CLI 或兼容性契约；通常与 `unit` 组合。
- `integration`：跨模块、真实文件系统、daemon 或端到端行为。
- `subprocess`：真实启动子进程；可由调用方显式排除。
- `ddp`：初始化 distributed worker 或调用 `torchrun`，即多卡分布式训练；可由调用方显式排除。
- `ort`：需要或执行 ONNX Runtime（ONNX 模型的推理引擎）；可由调用方显式排除。
- `gpu`、`external`、`pipeline`：分别表示需要显卡、需要外部资源（模型/权重/网络）、需要独立的 pipeline 环境。

资源 marker 可叠加，例如真实两卡 DDP 测试使用 `integration + gpu + ddp + subprocess`。
`tests/conftest.py` 暂时集中补充已核实的资源 marker，避免为分类目的修改并行代理
正在维护的大型测试文件。

## 更多筛选组合

```bash
# loader 契约（忽略根目录 pyproject.toml 的默认 -m）
python -m pytest -m "unit and contract" tests/test_legacy_loader_contracts.py

# 仅运行会启动子进程、但不含 DDP/ORT 的 unit
python -m pytest -m "unit and subprocess and not ddp and not ort"
```

命令行 `-m` 会覆盖根目录 `pyproject.toml` 的默认 marker 表达式；命令行的 `-m`
也是选择非 `unit` 测试（如 `integration`、`gpu`、`external`）的唯一方式。

## 目录演进

当前 `tests/` 仍采用平铺布局，没有 `tests/contract/`、`tests/integration/` 等子目录。
新测试可逐步放入这类子目录，但 marker 才是执行环境的事实来源；目录名
不自动隐含 marker。任务级测试也应放在默认收集目录内，并使用任务前缀区分文件名。
