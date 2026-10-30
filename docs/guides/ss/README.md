# SS 使用指南

SS（Speech Separation，语音分离）要解决的是：把一段多人混音拆成固定数量的语音源。与 SE 的区别在输出：SE 返回一路增强后的语音；SS 输入 mixture（不指定要哪一路的混音），输出固定 K 路 source，「哪一路输出对应哪个说话人」不预先指定，由训练时的 PIT（Permutation Invariant Training，排列不变训练：枚举所有「输出—监督」配对、保留代价最小的一种）处理。K 由顶层配置 `ss.num_sources` 固定，当前不支持 variable-K。

本目录收录 SS 的任务指南与训练入门。训练入口统一是根目录 `train.py`（按 `data.mode: dynamic_ss` / `offline_ss` 路由），配置在 `configs/ss/`；可直接运行的训练命令与 EchoSet 清单说明见任务入口 [`../../../tasks/ss/README.md`](../../../tasks/ss/README.md)。

## 主指南

- [SS Training Primer](SS_Training_Primer.md)：三种模型 I/O 形态（waveform-native / external-STFT / 多通道谱进谱出）、执行计划校验、PIT 与 loss 的关系、动态仿真的固定 K 合同。适合第一次读 SS 代码。
- [任务 README](../../../tasks/ss/README.md)：当前边界清单、EchoSet 入口、官方 checkpoint parity 验证与推理示例。跑通训练以该文为准。

## 关键事实（先过一遍，避免误读后面的细节）

- **数据模式**：`data.mode: offline_ss` 读取严格 JSONL mixture/source manifest（每行一条混音与 K 路源）；`dynamic_ss` 从共享 source pool 采样固定数量 source，在线执行共享 Layer A、SS Layer B（语音 / 语音+背景）、逐源 RIR、post-RIR SIR、背景混合与 D–F。Layer B、sampling、timeline、音乐路由与 source-scene audit 见 [`../simulation/Multi_Source_Simulation.md`](../simulation/Multi_Source_Simulation.md)。
- **批合同**：dict item（`mixture` [T]/[C,T]、`sources`、`sr`、`length`，可选 `source_ids`）→ `ss_collate_fn` → `SSBatch`（定义在 `core/data/source_separation_batch.py`）+ `validate()`。批内固定 source 数、固定麦克风数、统一 sr；`source_mask` 全 True，拒绝 partial mask；item 的 `sr`/`length` 是正整数；`source_ids` 全批提供或全批省略且每项为非空字符串。
- **模型 I/O 三种形态**：waveform-native（`TDANet`，`[B,T] → [B,K,T]`，模型内部不挂 STFT）；external-STFT（`TIGER`，消费框架 `mixture_spectra`，输出 `[B,K,F,T,2]` source spectra，必须传 `lengths`）；多通道谱进谱出（`SpatialNet` 离线 / `OnlineSpatialNet` 因果，`[B,C,F,T,2] → est_sources_spectra [B,K,F,T,2]`）。`SSModelRuntime.separate()` 是 trainer、推理与 verifier 共用的模型调用与 source waveform 输出入口。
- **PIT 由 trainer 管理**：loss 通过统一 `loss:` 配置构造 `PITLoss`，内部包装 `PairwiseL1Loss` 或 `PairwiseNegSISDRLoss`；合同细节见 [`../loss/README.md`](../loss/README.md) 的 SS/PIT 节。
- **执行计划**：`ResolvedExecutionPlan` 在启动期统一校验 Dataset、Model、Loss、采样率、source count 与 STFT 所有权，训练和验证不再各自从 YAML 重新推断模型输入域。
- **推理**：SS 没有根文件推理入口，把 SS 配置交给根 `inference.py` 会在模型构造前直接报错。分离推理走 `SSExtractor`（`tasks/ss/inference/extractor.py`，构造需要 `model, plan, stft_module`，调用返回 `[K,T]` 波形），adapter 在同一边界提供。

## 权威口径

- 模型路由与合同：[`../../architecture/SS_Architecture.md`](../../architecture/SS_Architecture.md)。
- SS 论文精读（SpatialNet / NBSS-moving lineage 见 [`paper/se/Audio-WestlakeU/`](../../paper/se/Audio-WestlakeU/)）：[`paper/se/`](../../paper/se/README.md)。
- 仿真主合同：[`../simulation/Simulation_Flow_Logic.md`](../simulation/Simulation_Flow_Logic.md)。
- 配置命名与继承语义：[`../../../configs/README.md`](../../../configs/README.md)。
