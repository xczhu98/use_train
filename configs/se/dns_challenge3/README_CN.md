# 🌐 DNS Challenge 3 — GTCRN 实验配置

<p align="center">
  <strong>基于 DNS Challenge 3 数据源的 GTCRN 离线配对论文协议复现</strong>
</p>

---

## 📑 目录

- [说明](#-说明)
- [配置文件](#-配置文件)
- [论文参数](#-论文参数-gtcrn-section-3)
- [使用流程](#-使用流程)
- [相关文档](#-相关文档)

---

## 📖 说明

本目录**仅用于 DNS Challenge 3 数据源上的 [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) 论文协议复现与框架验证**。当前 recipe 是由专用生成器产出 clean/noisy 配对的 `offline` 训练，不是动态混合。模型与损失和 VoiceBank-DEMAND GTCRN recipe 对齐；调度器、batch size 与数据协议不同。

完整流程共五步：下载 DNS3 原始数据 → 物化配对波形与 SCP 清单 → 训练 → 评估生成的配对 test split → 按需对 noisy-only challenge 集做推理。下文命令中 `/path/to/DNS-Challenge` 指你自己的 DNS Challenge checkout，`data/...` 均相对 `use_train` 仓库根目录解析，其余无需改动。

> ONNX 导出位于 `onnx/`，详见 [onnx/README.md](../../../onnx/README.md)。参数量/MACs 统一参考 [SE 架构文档](../../../docs/architecture/SE_Architecture.md) 和 `tools/model_tools/audit_metrics.py`；THOP MACs 仅为 partial/unverified，独立 FLOPs 为 `not measured`。DNSMOS/NISQA/UTMOS/SCOREQ 等 MOS 指标按 [评估流水线文档](../../../evaluate/README.md) 离线计算，不写入 Trainer validation。

---

## 📋 配置文件

| 文件 | 说明 |
|------|------|
| [`gtcrn_causal_offline.yaml`](gtcrn_causal_offline.yaml) | 当前唯一 DNS3 YAML；GTCRN 训练/评估（离线预混合，因果），实验根目录为 `exp/se/dns_challenge3_gtcrn_causal_offline` |

---

## 📐 论文参数 (GTCRN, Section 3)

| 参数 | 值 |
|------|-----|
| 采样率 | 16 kHz |
| 生成数量 | train 720,000 / val 840 / test 800，每条 10 秒 |
| 源池划分 | train/val/test 从同一 clean/noise/RIR 文件池独立打乱并循环采样；无源级 hold-out，仅用于论文协议复现 |
| 混合策略 | full RIR 混响语音 + noise；生成器从 `[-5, 15]` dB 连续均匀采样 `u`，再对能量归一化后的 noise 施加 `alpha = 10^(-1.5u/20)`。因此最终 speech/noise 的功率 SNR 为 `1.5u`，约为 `[-7.5, 22.5]` dB；`u` 本身不是最终功率 SNR |
| 训练目标 | 从 RIR 直达峰起截取前 100 ms，与 clean 卷积得到 early-RIR target |
| 训练读取 | 生成文件为 10 秒；YAML `max_duration: 128000` 在 16 kHz 下随机截取最多 8 秒 |
| Batch Size | 16 |
| 优化器 | 论文：AdamW, `lr=1e-3`, `wd=0.01`；本配置：Adam, `lr=1e-3`（无 weight decay） |
| 调度器 | 论文：ReduceLROnPlateau (`patience=5`, `factor=0.5`)；本配置：LinearWarmupCosineAnnealing（warmup 25,000 / decay 250,000 optimizer steps） |
| Loss | 论文原名为 HybridLoss；当前 YAML 已迁移/替换为与官方公式对应的显式组件 `GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`，并由 `CompositeLoss` 组合 |

---

## 🚀 使用流程

### 1. 下载 DNS3 原始数据

```bash
git clone https://github.com/microsoft/DNS-Challenge.git
cd DNS-Challenge
python download-dns-challenge-3.py
```

### 2. 生成配对波形与 SCP

以下路径与当前 YAML 一致。两个目标目录都必须可写；生成器使用 owned staging 与事务发布，拒绝覆盖不属于该生成器的目录。DNS Challenge 下载脚本需在其 checkout 内运行；下载完成后必须先回到 `use_train` 仓库根目录，再调用本仓生成器。

```bash
cd /path/to/use_train
python -m tools.prepare_data_tools.generators.se.generate_dns3_offline_dataset \
    --data-root /path/to/DNS-Challenge \
    --output-dir data/se/dns_challenge3_generated \
    --scp-dir data/se/dns_challenge3 \
    --num-workers 8
```

每个生成 split 都包含 clean/noisy 配对波形目录，以及严格四列 `clean.scp` / `noisy.scp`（`uid fs length path`）。schema v2 manifest 用同一个 generation identity 绑定 waveform root 与 index root。YAML 消费 train/val 清单；生成器的 test 是配对 protocol test，官方 DNS V2/V3/V5 challenge 集仍是 noisy-only 评估输入。

### 3. 训练

```bash
python train.py -C configs/se/dns_challenge3/gtcrn_causal_offline.yaml
```

### 4. 评估生成器的配对 test split

训练产物写入 `trainer.exp_path` 下的 run 目录。当前统一 SE CLI 将通用配对 SCP 路由暴露为 `voicebank` dataset tag，因此必须显式传入 DNS3 test SCP：

```bash
python evaluate/eval_pipeline_se.py \
    -C configs/se/dns_challenge3/gtcrn_causal_offline.yaml \
    --checkpoint exp/se/dns_challenge3_gtcrn_causal_offline/checkpoints/<run_id>/best_model.tar \
    --datasets voicebank \
    --voicebank_clean_scp data/se/dns_challenge3/test/clean.scp \
    --voicebank_noisy_scp data/se/dns_challenge3/test/noisy.scp
```

存在 clean reference 时，该路由请求 PESQ、STOI、ESTOI 与 SI-SDR。`voicebank` 仅用于选择配对评估器；报告结果时必须注明这是生成器的 DNS3 protocol test，不能写成 VoiceBank-DEMAND。

### 5. 盲测

当前 DNS Challenge checkout 中，V3 noisy-only blind set 位于 `datasets_fullband/V2_V3_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k`。若实际 checkout 的目录层级不同，请传入实际包含 noisy WAV 文件的目录。

```bash
python inference.py \
    -C configs/se/dns_challenge3/gtcrn_causal_offline.yaml \
    --checkpoint exp/se/dns_challenge3_gtcrn_causal_offline/checkpoints/<run_id>/best_model.tar \
    --input_dir /path/to/DNS-Challenge/datasets_fullband/V2_V3_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k \
    --output_dir results/dns3_blind
```

---

## 📚 相关文档

- [VoiceBank-DEMAND](../voicebank_demand/README_CN.md)
- [URGENT2026](../urgent2026/README_CN.md)
- [项目根 README_CN.md](../../../README_CN.md)

---

## 🙏 致谢

感谢 Microsoft DNS Challenge 与 GTCRN 作者。

<p align="center">
  <sub>最后更新：2026-09-21</sub>
</p>
