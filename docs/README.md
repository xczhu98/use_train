# use_train 文档中心

use_train 是一个语音训练与推理框架，覆盖五个任务：**SE**（Speech Enhancement，语音增强：降噪、去混响）、**AEC**（Acoustic Echo Cancellation，回声消除）、**TSE**（Target Speaker Extraction，目标说话人提取）、**SS**（Speech Separation，语音分离）和 **SSL**（Spatial Sound Localization，空间声源定位）。训练哪个任务由配置里的 `data.mode` 决定：`dynamic` 表示训练时在线仿真合成数据，`offline` 表示直接读取现成的配对数据，五个任务共九种取值。

`docs/` 只放文档，不放代码，按内容职责划分为架构、指南、论文、报告和图表五层。模型接口以代码和测试为准；文档用于解释设计、操作流程、实验记录和研究证据。

第一次来可以按目的选路径：[先跑起来](../README.md) → [看懂模型](architecture/README.md) → [改数据、仿真或 Loss](guides/README.md) → [查论文](paper/README.md)。

仿真（simulation：训练时按配置在线合成带混响、噪声、回声的训练样本）语义的唯一文档入口是 [`guides/simulation/Simulation_Flow_Logic.md`](guides/simulation/Simulation_Flow_Logic.md)。任务 README 只保留任务差异、配置入口和链接，不复制完整的 A–F 合同（A–F 指仿真流程的六个固定阶段：A 语音预处理、B 场景选择、C 声学混合、D 幅度整形、E 信道退化、F 观测/设备退化）。

## 目录结构

```text
docs/
├── architecture/   # 模型架构 SSOT 与跨模型机制分析
├── guides/         # 五任务与框架级主题的训练、仿真、推理、评估、部署和学习指南
├── paper/          # 论文 Markdown、PDF、图解与论文分析归档
├── reports/        # 本机实验与评估记录（个人归档，不纳入开源发布）
└── diagrams/       # 跨模块流程图与 SE 模型图
```

## 主要入口

- [架构文档](architecture/README.md)：SE、AEC、TSE、SS 架构与模型族分析，SSL 任务入口。
- [使用指南](guides/README.md)：五任务导读与训练、推理、评估、仿真、Loss、部署等框架级主题入口。
- [预训练权重](guides/reference/Pretrained_Weights.md)与[第三方资产](guides/reference/Third_Party_Assets.md)：未入库权重、第三方模型与源码的下载入口和放置路径。
- [Audio Filter Pipeline](../tools/audio_filter_pipeline/README.md)：音频质量筛选与数据整理。
- [Noise Filter Pipeline](../tools/noise_filter_pipeline/README.md)：噪声准入、发布与训练源路由。
- [论文归档](paper/README.md)：AEC、SE（含原 MCSE 多通道增强）、TSE、Survey、Loss、ASR、fullduplex 与 sv 八个方向的 canonical 精读（canonical：纳入版本控制的正式精读版，正文不依赖本地 PDF 或缓存图片）。
- [实验报告](reports/)：本机实验与评估记录，个人归档，不纳入开源发布（由 `.gitignore` 隔离，仓库内不跟踪）。
- [图表索引](diagrams/README.md)：跨模块流程图与 SE/PASE 模型图。

## 快速开始

按目的选一条：

- 安装、训练、推理、评估的总流程：[项目根 README](../README.md)
- SE 从训练到部署的完整示例（Voicebank-DEMAND 数据集、GTCRN 模型）：[VoiceBank Deployment Quickstart](guides/se/VoiceBank_Deployment_Quickstart.md)
- AEC 训练第一站（把教科书里的回声消除框图对到 `AECTrainer` 的代码流程）：[AEC Training Primer](guides/aec/AEC_Training_Primer.md)
- 训练数据是怎么合成的：[Simulation Flow Logic](guides/simulation/Simulation_Flow_Logic.md)
- 让仿真在线生成房间脉冲响应（RIR，用来模拟混响）：[Online gpuRIR](guides/simulation/Online_gpuRIR.md)
- 训练用的目标函数（Loss）怎么配：[Loss Configuration Guide](guides/loss/Loss_Configuration_Guide.md)
- URGENT2026 竞赛（训练 → 推理 → 评估）：[URGENT2026 Guide](guides/se/URGENT2026_Guide.md)
- TSE 任务导读（cue、批合同、训练与评测入口）：[TSE 使用指南](guides/tse/README.md)
- SS 任务导读（固定 K、PIT、三种模型 I/O 形态）：[SS 使用指南](guides/ss/README.md)
- SSL 任务导读（IPD 特征、动态仿真 seam、训练约束）：[SSL 使用指南](guides/ssl/README.md)
- 训练 / 推理 / 评估的框架级链路：[Training](guides/training/README.md)、[Inference](guides/inference/README.md)、[Evaluation](guides/evaluation/README.md)
- 运行时环境变量全表与 checkpoint 格式语义：[Runtime Environment Variables](guides/reference/Runtime_Environment_Variables.md)、[Checkpoint Format](guides/training/Checkpoint_Format.md)

## 事实来源（Canonical SSOT）

SSOT = Single Source of Truth，唯一事实来源。遇到「这件事以哪份文档为准」时按下表查：

| 主题 | 事实来源 |
|---|---|
| 仿真层级、五任务九种 `data.mode`、spatial observation、SSL 动态 scene 能力与 clean target 边界 | [`guides/simulation/Simulation_Flow_Logic.md`](guides/simulation/Simulation_Flow_Logic.md) |
| 物理阵列 profiles、在线 gpuRIR 与 geometry provenance | [`guides/simulation/Online_gpuRIR.md`](guides/simulation/Online_gpuRIR.md) |
| 仿真分层图与可重绘源文件 | [`diagrams/se_aec_tse_simulation_layers_current.dot`](diagrams/se_aec_tse_simulation_layers_current.dot) |
| 配置 schema、ownership 和生成器版本 | [`../configs/README.md`](../configs/README.md) |
| 运行时实现与兼容性 | `core/data/`、`tasks/se/data/`、`tasks/aec/data/`、`tasks/tse/data/`、`tasks/ss/`、`tasks/ssl/` 及对应 unit tests |

## 维护边界

- `architecture/` 回答“模型是什么”，不保存论文正文或操作手册。
- `guides/` 回答“怎么运行、验证和学习”，不重复归档论文正文。
- `paper/` 是论文 Markdown、PDF、相邻 figures 和论文分析的唯一归档。
- `reports/` 是本机实验与评估记录（个人归档），不随仓库发布；它不替代代码/测试 SSOT。
- `diagrams/` 只保存可重绘的图表源文件和生成图，不独立定义接口语义。
- 文档与实现不一致时，以当前源码、配置和测试结果为准，并同步修正文档。
