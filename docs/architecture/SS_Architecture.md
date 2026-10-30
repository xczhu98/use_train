# Speech Separation（SS）架构

`tasks/ss/` 是仓库中的多源语音分离任务包。它与 `tasks/se/` 的单目标 Speech Enhancement 合同不同：SS 输入一个 mixture，输出固定数量的 source 集合，并通过 PIT 处理 source permutation。

## 任务边界

| 任务 | 输入 | 输出 | 训练核心 |
|---|---|---|---|
| SE | noisy waveform | `[B, T]` enhanced waveform | 一一对应 waveform/spectral loss |
| TSE | mixture + enrollment | `[B, T]` target waveform | speaker-conditioned extraction |
| SS | mixture | `[B, K, T]` source waveforms | pairwise cost + PIT assignment |

SS 不复用 `SEBatch`、`SETrainer` 或 `BaseSEModel` 的单目标 contract。公共训练骨架仍复用 `core.trainer`，数据/DSP 原语复用 `core.data`，模型和 source-aware loss 由 `tasks/ss/` 所有。

## 路由与目录

```text
data.mode: dynamic_ss  -> SSTrainingEntry -> SSDynamicMixingDataset
data.mode: offline_ss  -> SSTrainingEntry -> SSOfflineDataset

tasks/ss/
├── data/       typed config、source pool、Dataset Adapter、collate
├── models/     TDANet waveform-native、TIGER external-STFT、SpatialNet/OnlineSpatialNet 多通道谱进谱出
├── model_io.py SSModelRuntime canonical source seam
├── trainer/    SSTrainingEntry 与 SSTrainer
└── inference/  SSExtractor 多源 waveform 输出与 SSInferenceAdapter
```

## Batch Contract

`SSBatch` 是固定 source count 的 typed batch：

```text
mixture:     [B, T]
sources:     [B, K, T]
lengths:     [B]
source_mask: [B, K]
sr:          [B] 或 scalar
```

第一版要求每个 batch 的 `K` 固定，`source_mask` 全部有效。variable-K、空 source 和 source identity ordering 不属于当前合同。

## Model Contract

所有 SS 模型在 trainer 边界输出以下二者之一：

```python
{"est_sources": [B, K, T]}
{"est_sources_spectra": [B, K, F, T, 2]}
```

TDANet 使用 waveform-native 路径。TIGER 使用顶层 `core.stft` 注入的 external-STFT 路径；模型内部保留频带拆分、separator 和 complex mask，但不执行 STFT/iSTFT。模型输出 source spectra 后由 `SSTrainer` 统一 iSTFT。

`SSModelRuntime.separate()` 是 trainer、extractor 和 verifier 共用的 model Interface。它消费启动期解析出的 `ResolvedExecutionPlan`，根据模型声明的 `input_domain` / `output_domain` 隐藏 STFT、source-axis reshape 和 iSTFT；caller 统一获得 `[B,K,T]` canonical source waveform。模型采样率、source count、谱 shape、输出时间长度和有限性必须精确匹配，runtime 不再通过裁剪或补零隐藏模型错误。

## Loss Contract

公开配置仍使用仓库统一的 `loss:` 块：

```yaml
loss:
  type: PITLoss
  args:
    pairwise_loss: PairwiseNegSISDRLoss
    assignment: exhaustive
    reduction: mean
```

PIT 在语义上是训练目标构造策略，在代码中实现为 scalar loss wrapper。内部先计算 `[B, K_est, K_ref]` pairwise cost，再调用 `core.losses.permutation.solve_source_permutation()`，最后返回 scalar loss。Metrics 复用同一 permutation Module，不依赖 PIT training objective。

`ss.num_sources` 是 Dataset、Model Runtime、PIT 和 inference 的单一事实来源。模型构建 Adapter 会注入该值；`model.args.num_sources` 若显式存在且冲突则 fail-fast。模型输入域由模型能力声明，不再使用 `tasks.ss.model_io` 配置字符串。

训练计划要求 loss 显式声明 permutation invariant、length-aware 和 source-mask capability。validation 固定执行 `prepare → model/output preflight → loss → task hooks → metric pairs`；每个阶段的普通异常均跨 rank 同步后再退出，避免某个 rank 在其他 rank 进入 DDP forward 或 loss collective 时提前失败。

`dynamic_ss` source pool 使用严格五字段 `uid/speaker_id/fs/length/path` JSONL。采样单位先按 `speaker_id` 去重，保证同一 mixture 不会因随机抽到同一说话人的不同 utterance 而退化为非标准分离样本。

## 官方模型参数边界

- TDANet EchoSet baseline 使用可严格加载 `TDANetBest-2ms-LRS2` checkpoint 的 topology：`out_channels=128`、`in_channels=512`、`num_blocks=16`、`upsampling_depth=5`、`enc_kernel_size=2 ms`、`num_sources=2`、`sample_rate=16000`。
- TIGER 配置使用官方 EchoSet small 口径：`out_channels=128`、`in_channels=256`、`num_blocks=4`、`upsampling_depth=5`、`win=640`、`stride=160`、`num_sources=2`、`sample_rate=16000`。
- 上述参数位于 `configs/ss/`；模型类不携带 “official/small/large” 配置身份，只消费显式构造参数。
- 2026-09-10 已完成 TDANet/TIGER 官方 checkpoint strict load、reference waveform parity、EchoSet 16 条抽样评估，以及 SSTrainer DataLoader→forward→PIT→backward 最小训练 smoke；ONNX 发布门禁仍未建立。
