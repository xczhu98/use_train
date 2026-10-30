# Checkpoint 格式与加载语义

checkpoint 是训练过程写出的文件，用于之后恢复训练，或复用其中的权重做微调、推理和导出。本文是 checkpoint 内容 schema 与三条加载路径的参考页；checkpoint 的保存事务（`latest_model.tar`、epoch checkpoint、best checkpoint 经同一 generation 事务原子发布）由 `core/trainer/base_trainer.py` 实现。

## 内容 schema

下面的示例只列出主要字段，不是完整 schema。基础 checkpoint 包含 epoch/step 计数、best-score 与 patience 状态、loss 曲线、RNG 状态、model/optimizer/scheduler 状态，以及解析后配置和 provenance 字段 `config`、`resolved_config`、`resolved_config_provenance`、`git_info`。根据 Trainer 类型和保存时机，checkpoint 还可能包含 `sampler_state`、`scaler`、`ema`、validation history、`rng_states_by_rank`、`rng_world_size`、多模型状态字段，以及 `batch_cursor`/`epoch_incomplete`。

```python
{
    "epoch": int, "n_steps": int, "best_score": float,
    "patience_counter": int,
    "model": OrderedDict, "optimizer": OrderedDict,
    "scheduler": OrderedDict, "scaler": OrderedDict,
    "ema": OrderedDict,  # 可选
    "config": dict, "resolved_config": dict,  # 同一份完整解析后配置的兼容键与显式别名
    "git_info": dict,
}
```

新 checkpoint 将训练入口解析后的完整配置同时保存为兼容键 `config` 与显式别名 `resolved_config`（同一对象），并保存 `git_info`，因此 artifact 自包含训练配置与代码版本 provenance；旧 checkpoint 缺少显式别名仍保持兼容。

## 加载语义按用途隔离

- **SE/AEC 评估与推理共用的公共 loader**（`evaluate/checkpoint_loading.py`）fail-closed：命中模型参数为 0、`missing_keys`、`unexpected_keys`、tensor shape 不匹配，以及 EMA `shadow_params` 数量、类型、dtype、shape 与 `model.parameters()` 不一致，全部在复制权重前报错；全部校验通过后才复制 EMA。
- **Trainer 完整 resume** 是独立的可信训练状态路径：optimizer/scheduler state 可能依赖 pickle，因此显式使用带 `purpose` 的 `allow_unsafe=True`；resume 恢复模型、optimizer、scheduler、AMP/EMA 与进度状态。
- **model-only pretrained 初始化**（`--checkpoint`）始终保持 `weights_only=True` 安全模式。

两条路径都会严格校验模型 state；resume 的 unsafe 权限不得复用于预训练权重加载。

## 推理 fingerprint 绑定

推理会把记录的 fingerprint 与实际 loader 使用的 artifact 绑定：普通 checkpoint 在 loader 前后重新计算内容身份；waveform-native 模型构造完成后，逐个复核预先记录的 `*_ckpt_path` artifact。推理 source boundary 包含 `core/utils/inference_artifacts.py`，因此 fingerprint、manifest 或 `--skip-existing` 校验实现变化时，现有 manifest 规则会阻止复用旧产物。
