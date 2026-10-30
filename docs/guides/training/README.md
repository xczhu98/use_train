# 训练指南

本文是训练链路的导航层：入口怎么路由、entry 要兑现什么合同、trainer 家族提供哪些能力、优化器与 scheduler 在哪配。具体合同（DDP 红线、批合同、loss 路由、仿真层级）分别以根 `AGENTS.md`、各任务 guides 和 [`../simulation/Simulation_Flow_Logic.md`](../simulation/Simulation_Flow_Logic.md) 为准，本文不复制。

## 入口路由

统一入口是根目录 `train.py -C <config.yaml>`，路由链固定为三段：

1. 配置里的 `data.mode`（九种取值，清单见 [`../simulation/README.md`](../simulation/README.md)）决定协议与任务；
2. `tasks/registry.py` 按 descriptor 把 mode 映射到任务；
3. 任务的 `tasks/<task>/trainer/entry.py::<Task>TrainingEntry`（`core/trainer/entry.py::UnifiedTrainingEntry` 子类）构建 loader、模型与 trainer，然后开始训练。

配置的结构语义（字段名、`base_config` 继承、list 整体替换、`exp_path` 命名）以 [`../../../configs/README.md`](../../../configs/README.md) 为准；写配置前先读它。

## TrainingEntry 合同

每个任务的 entry 继承 `UnifiedTrainingEntry` 并实现三个抽象方法：

- `build_loaders`：返回 `(train_loader, val_loader, train_sampler, train_ds)`；
- `build_model`：构造模型，显式消费 `build_model` 注入的 STFT 上下文（与有效 STFT 冲突会被 `core/models/build.py` 拒绝），未知 kwargs 抛 `TypeError`；
- `get_trainer_cls`：返回 trainer 类。

可选覆写：支持 GAN 时覆写 `_default_gan_trainer_cls`；需要 train dataset 事实（如通道数、参考通道）构造模型时覆写 `build_model_for_training`。`find_unused_parameters` 在 entry/config 层显式推导与校验，存在条件分支的模型必须推导，无条件单路径模型不得随意设为 `true`。

## Trainer 能力地图

核心实现在 `core/trainer/`：

- `base_trainer.py`：训练主循环、AMP/EMA、checkpoint 保存、`resume` 事务（可信训练状态恢复允许显式 `allow_unsafe=True` 并注明 purpose；model-only pretrained 初始化保持 `weights_only=True`，两条路径严格校验模型 state）。
- `metrics_trainer.py`：`MetricsCapableTrainer` 模板，消费 YAML `trainer.val_metrics` 的验证指标编排。
- `optimizer_step.py`：GAN / 多优化器 step 的统一语义（discriminator 先 `update_scaler=False, increment_step=False`，generator 后 `update_scaler=True, increment_step=True`）。
- `validation_orchestration.py` / `validation_metrics_config.py`：验证编排与指标配置解析。

架构红线（违反即阻断，只列名字与位置）：DDP collective 一致性（空 rank 以零值参与聚合，不得提前 return；validation 非空门禁在任何 collective 之前按全局分布对称判定）；epoch-tail 残差梯度必须经 `_handle_epoch_tail_gradients()`；NaN/Inf loss 用 `_graph_preserving_zero_loss()`；`torch.compile` 先于 DDP wrapping，`_raw_model()` 同时解包 `DDP.module` 与 `_orig_mod`。细节见根 `AGENTS.md`「架构红线」。

## 优化器与 scheduler

optimizer 由 YAML `optimizer` 段配置（类型、学习率、weight decay 等）；scheduler 统一在 [`core/optim/scheduler.py`](../../../core/optim/scheduler.py)，经 YAML `scheduler.type` 选择，提供 `LinearWarmupCosineAnnealing`、`ExponentialDecrease`、`CosineAnnealingWarmupRestarts` 等。scheduler 初始化前会拒绝破坏比例缩放的 optimizer 基准学习率；初始化 `--checkpoint`（model-only）同时支持安全 torch 与 `.safetensors`，复制参数前执行完整 key/dtype/shape 校验。

## 各任务入口

跑通某个任务的具体命令不在这里复制，按任务进：

| 任务 | 指南 | 任务 README（可复制命令） |
|---|---|---|
| SE | [`../se/`](../se/README.md) | [`tasks/se/README.md`](../../../tasks/se/README.md) |
| AEC | [`../aec/`](../aec/README.md) | [`tasks/aec/README.md`](../../../tasks/aec/README.md) |
| TSE | [`../tse/`](../tse/README.md) | [`tasks/tse/README.md`](../../../tasks/tse/README.md) |
| SS | [`../ss/`](../ss/README.md) | [`tasks/ss/README.md`](../../../tasks/ss/README.md) |
| SSL | [`../ssl/`](../ssl/README.md) | [`tasks/ssl/README.md`](../../../tasks/ssl/README.md) |

## 权威口径

- 训练配置合同：[`../../../configs/README.md`](../../../configs/README.md)。
- 训练目标配置：[`../loss/`](../loss/README.md)。
- checkpoint 内容 schema 与加载语义：[`Checkpoint_Format.md`](Checkpoint_Format.md)。
- 代码 SSOT：`core/trainer/`、`train.py`、根 `AGENTS.md`（架构红线与熔断机制）。
