# 🧩 语音分离（Speech Separation, SS）训练配置

**解决的问题**：一段录音里多个人同时说话时，把混合信号拆成固定数量的单人语音（通用语音分离框架，简称 `ss` 任务）。本目录维护 `ss` 的训练配置，每份 YAML 都能直接用 `python train.py -C <配置>` 启动。

读配置前先认识本文用到的词：

| 术语 | 一句话解释 |
|---|---|
| source count（`K`） | 每段混合音固定要分出几路源。当前配置统一 `K=2`，Dataset、模型与 loss 三处必须一致；本仓不支持可变 K |
| waveform-native | 模型直接吃波形，STFT/iSTFT 由框架或模型内部管理 |
| external STFT | 模型吃外部传进来的复数谱（`mixture_spectra`），因此必须同时给出 `lengths` |
| PIT（Permutation Invariant Training，置换不变训练） | 分离出的 K 路没有固定编号，训练时自动挑误差最小的“第 1 路对第 1 路”配对，避免因输出顺序不同白扣分 |
| SI-SDR（Scale-Invariant Signal-to-Distortion Ratio，尺度不变信号失真比，dB，越大越好） | 衡量分离质量的常用客观指标；“尺度不变”指整体音量缩放不影响得分 |
| A–F 仿真 | 训练时在线合成带噪、带混响混合信号的六个处理层，完整语义见 [仿真流程逻辑](../../docs/guides/simulation/Simulation_Flow_Logic.md) |

## 配置一览

| 配置 | 模型 | 输入/输出 | 采样率 | 数据口径 |
|---|---|---|---:|---|
| `tdanet/tdanet_dynamic_sisdr.yaml` | TDANetBest 2 ms | waveform → `[B, K, T]` | 16 kHz | URGENT clean source pool，共享 A–F，`K=2` |
| `tdanet/tdanet_echoset_offline.yaml` | TDANetBest 2 ms | waveform → `[B, K, T]` | 16 kHz | EchoSet，`K=2`，train 3 秒随机裁剪 |
| `tiger/tiger_echoset_external_stft.yaml` | TIGER | external STFT → source spectra | 16 kHz | EchoSet，`K=2`，3 秒片段 |
| `tiger/tiger_echoset_external_stft_large.yaml` | TIGER large | external STFT → source spectra | 16 kHz | 官方 `TIGER-speech` checkpoint parity |
| `spatial/online_spatialnet_dynamic_{mhsa,ret,mamba}.yaml` | OnlineSpatialNet | 多通道复谱 `[B,C,F,T,2]` → `est_sources_spectra` | 8 kHz | NBSS 论文对齐（6 通道、在线 gpuRIR、moving），`K=2` |

offline 配置消费严格 JSONL separation manifest，dynamic 配置从共享 source pool 在线混音。训练产物写入各配置的 `trainer.exp_path`，约定为 `exp/ss/<配置文件名去 .yaml>`（例如 `tdanet_dynamic_sisdr.yaml` → `exp/ss/tdanet_dynamic_sisdr/`），checkpoint 位于其中的 `checkpoints/<run-id>/`。

上游论文仓库只作为模型结构和参数参考；训练入口、manifest、PIT loss、checkpoint 与评测均使用本仓库的 `tasks/ss/` 任务实现。

动态配置的纯背景场景保持 K 路有效零 target；默认 pairwise SI-SDR 在静音参考上
切换为预测均方能量抑制，避免零梯度。shared speech sampler 决定第一路 anchor，
其余 K−1 路按不同身份抽样；sampling quota 只约束 anchor 曝光。

## EchoSet 基线

数据准备（生成 JSONL 清单，输出在 `--output-dir` 下）：

```bash
python -m tools.prepare_data_tools.generators.ss.prepare_echoset \
  --input-root data/ss/EchoSet \
  --output-dir data/ss/EchoSet
```

官方 checkpoint 的 EchoSet 抽样评估（`--max-samples 16` 表示只评 16 条，便于快速验证；报告写入 `--output` 指定的 JSON）：

```bash
python tools/verify/verify_ss_echoset.py \
  -C configs/ss/tiger/tiger_echoset_external_stft_large.yaml \
  --checkpoint /path/TIGER-speech/model.safetensors \
  --manifest data/ss/EchoSet/test.jsonl \
  --max-samples 16 \
  --output data/ss/EchoSet/reports/tiger_test16.json
```

上面的 `--input-root`、`--checkpoint` 都是示例路径，换成实际机器上的 EchoSet 目录与官方权重位置即可。数据准备命令生成 `train.jsonl` / `val.jsonl` / `test.jsonl` 与 `echoset_metadata.json`；评估命令输出一份 JSON 报告，并把汇总指标打到 stdout。两个脚本都会自动创建所需的输出目录。
