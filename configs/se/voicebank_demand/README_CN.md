# 🎤 VoiceBank-DEMAND — GTCRN 训练与部署指南

<p align="center">
  <strong>基于 GTCRN (ICASSP 2024) 复现 VoiceBank-DEMAND 基线结果</strong><br>
  含完整数据准备、训练、评估与 ONNX 部署流程
</p>

---

本目录存放 VoiceBank-DEMAND 的训练配置。VoiceBank-DEMAND 是一份小型配对语料：每条 noisy 音频都有逐条对应的 clean 音频，因此适合作为端到端的第一条流程——生成 SCP 清单、训练、在官方 test 集上评估，最后按需导出 ONNX。训练、推理、评估与导出都复用与其它 SE recipe 相同的入口（`train.py`、`inference.py`、`evaluate/eval_pipeline_se.py`、`onnx/`）。

下文用到的词：

| 术语 | 一句话解释 |
|---|---|
| SCP | 以空格分列的清单文件，每行四列 `uid fs length path` |
| PESQ / STOI / ESTOI / SI-SDR | 需要干净参考的侵入式客观指标，数值越高越好 |
| ONNX | 一种模型交换格式，用于把训练好的模型部署到 PyTorch 之外 |
| WOLA | weighted overlap-add，官方实现所用的分析/合成窗配对方式 |
| RTF | 实时率：处理耗时 ÷ 音频时长；小于 1.0 表示快于实时 |
| THOP | 本仓库用来测 MACs 的 hook 式 profiler |

本目录当前提供以下训练配置：

| 配置 | 模型与边界 | 独立实验目录 |
|------|------------|--------------|
| [`gtcrn_causal_offline.yaml`](gtcrn_causal_offline.yaml) | GTCRN 因果谱模型，离线预混训练 | `exp/se/gtcrn_causal_offline` |
| [`zipenhancer_s.yaml`](zipenhancer_s.yaml) | ZipEnhancer-S | `exp/se/zipenhancer_s` |
| [`fast_ulcnet_offline.yaml`](fast_ulcnet_offline.yaml) | FastULCNet；标准 Hann 不等同于官方 `hann75`/WOLA，尚无 checkpoint 或真实 90 秒质量复现 | `exp/se/fast_ulcnet_offline` |
| [`meanflowse_ncsnpp_offline.yaml`](meanflowse_ncsnpp_offline.yaml) | MeanFlowSE；纯 PyTorch NCSN++ 风格项目实现，不兼容官方 checkpoint，尚未训练复现论文指标 | `exp/se/meanflowse_ncsnpp_offline` |
| [`dccrn_offline.yaml`](dccrn_offline.yaml) | DCCRN（原始 DCCRN-CL：complex encoder / LSTM / decoder，E 模式复掩码）；仅移植模型拓扑，STFT/iSTFT 由本仓框架管理，不声明官方训练配置、checkpoint 或论文指标 | `exp/se/dccrn_offline` |
| [`meanse_ncsnpp_offline.yaml`](meanse_ncsnpp_offline.yaml) | MeanSE（arXiv:2509.21214）的最终 curriculum stage（`max_interval=1.0`）；与 MeanFlowSE 不是同一论文或实现，许可未确认且无公开 checkpoint，不声明官方结构或 checkpoint 兼容 | `exp/se/meanse_ncsnpp_offline` |
| [`mtfaa_net_causal_offline.yaml`](mtfaa_net_causal_offline.yaml) | MTFAA-Net 风格模型（Phase Encoder + multi-scale TFCM + axial attention + 两阶段 MTFAA mask）；改自非官方实现，上游明示细节可能与论文不一致 | `exp/se/mtfaa_net_causal_offline` |
| [`rt_semamba_kd8_causal_offline.yaml`](rt_semamba_kd8_causal_offline.yaml) | RT-SEMamba（8 个 cTF-Mamba block，谱进谱出）；只训练判别式 generator，不含官方 metric discriminator / PESQ GAN recipe | `exp/se/rt_semamba_kd8_causal_offline` |
| [`rt_semamba_kd1_distill_causal_offline.yaml`](rt_semamba_kd1_distill_causal_offline.yaml) | RT-SEMamba KD1 渐进蒸馏：冻结 8-block teacher → 1-block student；teacher checkpoint 必须先经严格转换或校验 | `exp/se/rt_semamba_kd1_distill_causal_offline` |
| [`tf_skimnet_causal_offline_mag_complex_l1.yaml`](tf_skimnet_causal_offline_mag_complex_l1.yaml) | TF-SkiMNet（Interspeech 2025）；官方仓库无训练代码、checkpoint、评估代码与许可证，当前 loss 是可用的 consistency surrogate，不等同于论文逐项配方 | `exp/se/tf_skimnet_causal_offline_mag_complex_l1` |

两份 RT-SEMamba 配置的前端与官方不同：官方使用 `center=False` + Hann，共享 `torch.istft` 会因 NOLA 拒绝该组合，因此本目录改用同为 400/100/400 的 Hamming 窗，不声称前端数值精确复现。每份 YAML 头部都重复各自的审计摘要（`Params` / `MACs` / `FLOPs` / `Trainer MOS boundary`）。

上述目录只属于 VoiceBank-DEMAND recipe；DNS Challenge 3 的同名 GTCRN 配置使用 `exp/se/dns_challenge3_gtcrn_causal_offline`，不得跨数据集 `--resume` 或复用 checkpoint。

参数量/MACs 与未测项统一参考 [SE 架构文档](../../../docs/architecture/SE_Architecture.md) 和 `tools/model_tools/audit_metrics.py`。THOP MACs 仅为 partial/unverified：hook 可能漏计 functional、complex、custom 或 unsupported operator；独立 FLOPs 为 not measured。DNSMOS/NISQA/UTMOS/SCOREQ 等 MOS 指标统一按 [评估流水线文档](../../../evaluate/README.md) 离线计算，不写入 Trainer validation。

---

## 📑 目录

- [论文来源](#-论文来源)
- [1. 原始数据集](#1-原始数据集)
- [2. 数据准备](#2-数据准备)
- [3. 论文超参数](#3-论文超参数-gtcrn-section-3)
- [4. 运行命令](#4-运行命令)
- [5. 部署](#deployment-edge)
- [6. 文件结构](#6-文件结构)
- [7. 常见问题](#7-常见问题)
- [8. 参考文献](#8-参考文献)

---

## 📖 论文来源

基于 **GTCRN** (ICASSP 2024) Section 3 的配置，在 VCTK-DEMAND 数据集上复现基线结果。

> 本目录**仅用于基线论文对比与框架验证**；DNS Challenge 3 对应 `configs/se/dns_challenge3/gtcrn_causal_offline.yaml`，同样是 GTCRN 的 `offline` 配对 recipe。其他训练配置见 `configs/se/urgent2026/`。

---

## 1. 原始数据集

| 项 | 值 |
|---|---|
| 来源 | Edinburgh DataShare (handle 10283/1942) |
| 训练集 | 28 说话人 / 11,572 条 / 16 kHz |
| 测试集 | 2 说话人 (p232, p257) / 824 条 / 48 kHz |

### 噪声与 SNR 协议

| 划分 | SNR（dB） | 噪声条件 |
|---|---|---|
| 官方 train | 0、5、10、15 | 训练噪声条件 |
| 官方 test | 2.5、7.5、12.5、17.5 | 训练阶段未见的 5 种噪声条件 |

train 与 test 刻意采用不同的噪声和交错 SNR；官方 test 用于检验对 unseen noises 的泛化，而不是复用 train 的噪声/SNR 组合。

> **评估协议警告：**本目录全部 YAML 当前都把完整官方 test 集写入 `voicebank_*_val.scp`。若据此选择 checkpoint、early stopping 或调超参数，就会把最终 test 信息泄漏到模型开发阶段。严格实验应从官方 train 的 clean/noisy 配对中另划独立 validation，并保持官方 test 完全保留；本地训练时将 YAML 指向该 validation，最终评估时再通过命令行显式传入保留的 test SCP。

---

## 2. 数据准备

以下命令中的 `-d data/se/voicebank_demand` 是相对路径，因此必须从仓库根目录运行；若当前 shell 位于其他目录，请改用仓库中 `data/se/voicebank_demand` 的绝对路径，或相应调整相对路径：

```bash
unzip clean_trainset_wav.zip -d data/se/voicebank_demand
unzip noisy_trainset_wav.zip -d data/se/voicebank_demand
unzip clean_testset_wav.zip -d data/se/voicebank_demand
unzip noisy_testset_wav.zip -d data/se/voicebank_demand
```

压缩包目录层级可能因分发版本而异。下列命令采用 test 压缩包含同名内层目录的常见布局；若 WAV 直接位于外层目录，应改为传入实际包含 WAV 的目录。每条轨道分别使用 `prepare_scp.py` 单目录模式生成四列 SCP；该模式扫描**所给目录的全部文件且不做 train/val 随机切分**：

```bash
python tools/prepare_data_tools/scp/prepare_scp.py \
    data/se/voicebank_demand/clean_trainset_wav \
    -o data/se/voicebank_demand/voicebank_clean_train
python tools/prepare_data_tools/scp/prepare_scp.py \
    data/se/voicebank_demand/noisy_trainset_wav \
    -o data/se/voicebank_demand/voicebank_noisy_train
python tools/prepare_data_tools/scp/prepare_scp.py \
    data/se/voicebank_demand/clean_testset_wav/clean_testset_wav \
    -o data/se/voicebank_demand/voicebank_clean_val
python tools/prepare_data_tools/scp/prepare_scp.py \
    data/se/voicebank_demand/noisy_testset_wav/noisy_testset_wav \
    -o data/se/voicebank_demand/voicebank_noisy_val
```

本目录全部 YAML 精确使用以下路径：

- `data/se/voicebank_demand/voicebank_clean_train.scp`
- `data/se/voicebank_demand/voicebank_noisy_train.scp`
- `data/se/voicebank_demand/voicebank_clean_val.scp`
- `data/se/voicebank_demand/voicebank_noisy_val.scp`

每行格式为 `uid fs length path`。全量 train 的 clean/noisy 各 **11,572 条**（每轨约 9.391 小时，16 kHz）；validation 直接使用完整官方 test 集，clean/noisy 各 **824 条**（每轨约 0.576 小时，磁盘采样率 48 kHz），由 offline dataset 在加载时按配置的 `data.sample_rate: 16000` 重采样。`trainer.val_metrics.max_utterances` 只是全局上限：GTCRN YAML 中的 `872` 不会生成或代表 872 条 validation，实际最多仍只有 824 对。

`prepare_scp.py` 单目录模式用 `soundfile` 探测 metadata 并完整写出目录，不做划分。若原始来源未经验证，可在四条命令后加 `--validate_audio`，对空音频、NaN/Inf、越界峰值、过静音和高 DC offset 做全解码检查。生成后应核对四份 SCP 的行数，再通过构造 offline dataset（或启动训练）校验 clean/noisy UID 集完全一致，并将声明的 `fs` / `length` 绑定到实际解码 header。

clean/noisy 必须按相同 UID 配对，而不能只依赖行号。两条 validation 命令必须从对应的同级内层目录生成，使 `p232_001` 等 UID 在两轨一致。VoiceBank 是官方预混配对语料：`voicebank_clean_*` 是监督 target，`voicebank_noisy_*` 是对应 degraded input；本 recipe 不采样 RIR，也不构造 early-RIR target。预混 noisy mixture SCP 也不能作为 dynamic 模式的 `noise_scp` 噪声源。

论文 GTCRN 使用 2 秒随机片段，而本仓库当前 GTCRN VoiceBank 配置使用 `max_duration: 128000`（16 kHz 下 8 秒）训练片段，两者不是同一复现口径。

---

## 3. 论文超参数 (GTCRN, Section 3)

| 参数 | 值 |
|------|-----|
| 采样率 | 16,000 Hz |
| Batch Size | 4 |
| 优化器 | Adam, lr=0.001 |
| 调度器 | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss | 论文原名为 HybridLoss（α=0.01, β=0.3）；当前 YAML 已迁移/替换为与官方公式对应的显式组件 `GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`，并由 `CompositeLoss` 组合 |
| 模型大小 | 23.67K trainable / ~48K total |
| 1 秒输入 MACs | 约 34M（THOP，partial/unverified；通常不含 STFT/iSTFT） |
| FLOPs | not measured |
| 帧移 | 16 ms（`hop_length=256` @ 16 kHz，不是端到端系统 latency） |

### 历史复现记录

| 指标 | GTCRN 论文 | use_train 历史记录 |
|------|-----------:|-------------------:|
| PESQ | 2.87 | **2.8031** |
| STOI | 0.940 | **0.9331** |
| ESTOI | 0.848 | **0.8322** |
| SI-SDR | 18.8 dB | **18.2864 dB** |

**Provenance：**以上四个数值保留自仓库历史评估 transcript，协议为完整 824 条 VoiceBank-DEMAND 官方 test。历史记录标注 checkpoint `exp/gtcrn_vb/checkpoints/2026-03-05_10-31-00/best_model.tar`，并使用旧入口 `evaluate/evaluate_voicebank.py --no_mos`。当前工作树不含该 checkpoint 或原始评估 artifact，因此本次更新**没有重新测量**这些数值，也不能将其表述为当前 run 结果。重新评估时应记录当前 checkpoint 路径或内容 hash、配置、精确的 824 对 SCP、依赖版本、命令和 `evaluate/eval_pipeline_se.py` 原始输出。

当前 GTCRN YAML 的 Trainer validation 指标列表是 `[SI-SDR, PESQ, STOI]`，训练期间不记录 ESTOI；下方配对离线评估命令在存在 clean reference SCP 时，会通过 `evaluate.common.requested_standard_metrics` 请求 PESQ、STOI、ESTOI 与 SI-SDR 四项。

---

## 4. 运行命令

```bash
# 训练
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/voicebank_demand/gtcrn_causal_offline.yaml

# 推理
python inference.py \
    -C configs/se/voicebank_demand/gtcrn_causal_offline.yaml \
    --checkpoint "exp/se/gtcrn_causal_offline/checkpoints/<run-id>/best_model.tar" \
    --input_dir path/to/noisy/wavs --output_dir results/enhanced

# 最终评估：显式限定 VoiceBank，并传入保留的官方 test SCP
python evaluate/eval_pipeline_se.py \
    -C configs/se/voicebank_demand/gtcrn_causal_offline.yaml \
    --checkpoint "exp/se/gtcrn_causal_offline/checkpoints/<run-id>/best_model.tar" \
    --datasets voicebank \
    --voicebank_clean_scp data/se/voicebank_demand/voicebank_clean_val.scp \
    --voicebank_noisy_scp data/se/voicebank_demand/voicebank_noisy_val.scp
```

---

<a id="deployment-edge"></a>
## 5. 部署准备（ONNX 调试/通用产物）

### ONNX 调试导出与正式发布

```bash
# GTCRN 流式 debug/general 导出；--smoke 仅做一次 PT/ORT 冒烟检查
python onnx/export.py \
    --model streaming_gtcrn \
    --checkpoint "exp/se/gtcrn_causal_offline/checkpoints/<run-id>/best_model.tar" \
    --output runs/onnx_debug/gtcrn_streaming.onnx \
    --smoke

# 正式发布：离线 PT / 逐帧流式 PT / 逐帧 ORT 三路必须全部 PASS
python onnx/verify/verify_3way_gtcrn.py \
    --checkpoint "exp/se/gtcrn_causal_offline/checkpoints/<run-id>/best_model.tar" \
    --save-onnx onnx/onnx_model/gtcrn_streaming.onnx
```

`onnx/export.py` 的 `--output` 只生成 debug/general artifact；`--smoke`（兼容别名
`--verify`）不是三路发布门禁。正式产物必须由 `verify_3way_gtcrn.py` 严格加载 checkpoint、
校验 runtime metadata，并在三路比较全部通过后写入 `onnx/onnx_model/`。该 verifier 的
GTCRN 结构固定为 `n_fft=512`，不接受 `-C/--config`；传入的 checkpoint 必须与该结构匹配。

### 历史性能参考（未复测）

下表没有可追溯的测试脚本、环境与原始记录，仅保留为历史参考；当前版本未重新复测。

| 硬件 | RTF |
|------|-----|
| Intel i7 | ~0.04 (ONNX) |
| ARM Cortex-A76 | ~0.13 |

> RTF < 1.0 通常表示处理速度快于音频时长；上表数值不是当前版本的性能承诺。

`hop_length=256` 在 16 kHz 下对应 **16 ms 帧移**，不是端到端系统 latency。`center=True` 与 `center=False` 都不是通用部署规则；必须显式固定 STFT/iSTFT framing、buffering 与边界处理，并用同一输入对拍训练路径和部署路径。

---

## 6. 文件结构

```
configs/se/voicebank_demand/
├── gtcrn_causal_offline.yaml
├── zipenhancer_s.yaml
├── fast_ulcnet_offline.yaml
├── meanflowse_ncsnpp_offline.yaml
├── dccrn_offline.yaml
├── meanse_ncsnpp_offline.yaml
├── mtfaa_net_causal_offline.yaml
├── rt_semamba_kd8_causal_offline.yaml
├── rt_semamba_kd1_distill_causal_offline.yaml
├── tf_skimnet_causal_offline_mag_complex_l1.yaml
├── README.md
└── README_CN.md
data/se/voicebank_demand/
├── voicebank_clean_train.scp      # 11,572 条
├── voicebank_noisy_train.scp      # 11,572 条
├── voicebank_clean_val.scp        # 824 条（官方 test clean）
├── voicebank_noisy_val.scp        # 824 条（官方 test noisy）
├── {clean,noisy}_trainset_wav/
└── {clean,noisy}_testset_wav/
exp/se/gtcrn_causal_offline/
└── checkpoints/<run-id>/{best_model.tar, latest_model.tar}
```

---

## 7. 常见问题

<details>
<summary>Q1: 测试集必须重采样吗？</summary>

GTCRN 的模型输入必须是 16 kHz。当前 offline dataset 会根据 SCP 中的真实 48 kHz metadata，在加载时自动重采样到配置的 `data.sample_rate: 16000`；若绕过该 loader 或生成独立 16 kHz 评估资产，则必须在外部显式重采样并保留对应 provenance。
</details>

<details>
<summary>Q2: lamda_ri=30, lamda_mag=70 的含义？</summary>

对应 β×100 和 (1-β)×100，与官方 `30*(rl+il) + 70*ml + sisnr` 一致。
</details>

<details>
<summary>Q3: 指标不匹配？</summary>

将重采样、optimizer/scheduler、训练片段长度、实际停止 epoch，以及 PESQ 实现版本与模式作为对照排查项，而不是已证实原因。改用 Adam、训练约 80–100 epochs 或更换 PESQ 实现都不保证复现论文指标；现有证据不足以把任一差异认定为指标偏差的因果来源。
</details>

---

## 8. 参考文献

1. GTCRN (ICASSP 2024)
2. VoiceBank-DEMAND: Valentini-Botinhao et al., ISCA SSW 2016
3. [GTCRN 仓库](https://github.com/AkenoSyuRi/GTCRN)

---

<p align="center">
  <sub>最后更新：2026-09-21 · 感谢 VoiceBank-DEMAND 与 GTCRN 作者</sub>
</p>
