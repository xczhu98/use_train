# 🏆 URGENT2026 Challenge — 实验配置

<p align="center">
  <strong>BSRNN / DPDFNet2 · 离线 & 动态混合</strong>
</p>

---

## 📑 目录

- [说明](#-说明)
- [术语速查](#-术语速查)
- [配置文件](#-配置文件)
- [数据集 (Track 1)](#-数据集-track-1)
- [数据准备](#-数据准备)
- [训练命令](#-训练命令)
- [DDP 注意事项](#multirate-bsrnn-ddp)
- [相关文档](#-相关文档)

---

## 📖 说明

本目录用于与 **[URGENT2026 Challenge](https://urgent-challenge.github.io/urgent2026/)** 对齐，包含 BSRNN baseline 及因果/动态变体。

Track 1 要求一个模型同时完成降噪、去混响与通道修复，并覆盖多种采样率，因此这里的每份 recipe 都是"一个模型、多种录音条件"。第一张表里的四份是监督 baseline，`gan/` 下的五份是同一任务的对抗训练变体。第一次接触该任务建议先从下面的非因果 BSRNN baseline 开始：它的部件最少，也不涉及动态混音。

> 上游 baseline 来源为 [`urgent-challenge/urgent2026_challenge_track1`](https://github.com/urgent-challenge/urgent2026_challenge_track1)，对应上游仓库中的 `baseline_code/models/bsrnn.py` 与 `conf/models/BSRNN_baseline.yaml`；这两个不是本仓路径。本仓移植入口为 `tasks/se/models/discriminative/bsrnn.py`，核心实现在 `tasks/se/models/modules/bsrnn_impl/`，本仓 baseline 配置为 `usr_bsrnn_noncausal_urgent26_offline_multires_l1_spec.yaml`。
>
> 子目录：[`pase/`](pase/README.md)（PASE/StuPASE，以及已索引的 [`unipase/`](pase/unipase/README.md) 与 [`gap_urgenet/`](pase/gap_urgenet/README.md)）、`gan/`（下表列出的五份动态混合 GAN 配置）。
>
> `configs/se/dns_challenge3/` 当前仅提供 GTCRN 离线配置；其他 SE 训练配置见本目录的模型与数据配方。

### 🔤 术语速查

| 术语 | 一句话解释 |
|---|---|
| Track 1 | 本仓库对标的赛道：通用语音增强（降噪 + 去混响 + 通道修复） |
| BSRNN | band-split RNN：模型把频谱切成若干子带，再用循环层分别建模 |
| 因果 / 非因果 | 因果模型只能看过去帧（流式部署必需）；非因果模型可以看完整句 |
| DPDFNet2 | 轻量增强模型，固定 2 帧算法前瞻（16 kHz 下约 20 ms）；文件名里的 `causal` 不代表零前瞻 |
| offline recipe | 直接在磁盘上已有的 clean/noisy 配对数据上训练 |
| dynamic recipe | 训练时用干净语音 + 噪声/RIR 清单在线合成带噪输入 |
| MetricGAN 系列 | 用判别器估计感知质量指标的对抗训练 |
| PESQ | 需要干净参考的感知质量指标，数值越高越好；`gan/gtcrn_metricdiscriminator_dynamic.yaml` 是唯一不把 PESQ 注入判别器的配置 |
| nonblind | 自带 clean 标签的评估音频；它永远不作为训练数据 |
| DDP | 多卡数据并行训练 |

---

## 📋 配置文件

| 文件 | 模型 | 数据模式 | Loss |
|------|------|----------|------|
| `usr_bsrnn_noncausal_urgent26_offline_multires_l1_spec.yaml` | BSRNN 非因果 | 离线 | MultiResL1SpecLoss |
| `usr_bsrnn_causal_urgent26_offline_hybrid.yaml` | BSRNN 因果 | 离线 | 显式 CompositeLoss（`GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`） |
| `dpdfnet2_causal_urgent26_offline_hybrid.yaml` | DPDFNet2 近因果（2 帧前瞻） | 离线 | 显式 CompositeLoss（`GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`） |
| `dpdfnet2_causal_urgent26_dynamic_hybrid.yaml` | DPDFNet2 近因果（2 帧前瞻） | 动态 | 显式 CompositeLoss（`GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`） |

以上四份文件都直接位于 `configs/se/urgent2026/`。`gan/` 下另有五份配置：

| 文件 | Generator / discriminator | Trainer | 数据模式 |
|------|---------------------------|---------|----------|
| `gan/gtcrn_metricdiscriminator_dynamic.yaml` | `GTCRN` / `MetricDiscriminator` | `GANTrainer`（普通 LSGAN，不注入 PESQ） | 动态 |
| `gan/mambattention_metricgan_dynamic.yaml` | `MambAttentionGenerator` / `MetricMagDiscriminator` | `MetricGANCompositeTrainer` | 动态 |
| `gan/mossformer_gan_metricgan_dynamic.yaml` | `MossFormerGANGenerator` / `MetricMagDiscriminator` | `MetricGANCompositeTrainer` | 动态 |
| `gan/mp_senet_metricgan_dynamic.yaml` | `MPSEnetGenerator` / `MetricMagDiscriminator` | `MetricGANCompositeTrainer` | 动态 |
| `gan/rwsa_mamba_unet_metricgan_dynamic.yaml` | `RWSAMambaUNetGenerator` / `MetricMagDiscriminator` | `MetricGANCompositeTrainer` | 动态 |

> DPDFNet2 配置保留 `df_lookahead=2` 与 `conv_lookahead=2`；cache 可复用历史状态并支持实时逐帧处理，但不能消除这 2 帧前瞻。16 kHz、`hop_length=160` 时固定算法前瞻约为 20 ms。文件名中的 `causal` 是既有命名，不代表零前瞻。

---

## 📊 数据集 (Track 1)

| 项 | 值 |
|----|-----|
| 任务 | 通用语音增强（降噪 + 去混响 + 通道修复） |
| 磁盘采样率 | train 含 8 / 16 / 22.05 / 24 / 32 / 44.1 / 48 kHz；validation 不含 8 kHz |
| 离线训练音频 | `data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/simulation_train/{clean,noisy}/` |
| 离线验证音频 | `data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/simulation_validation/{clean,noisy}/` |
| 原始 validation manifest | 2,200 组 clean/noisy 配对 |
| DPDFNet/GAN 等带时长过滤的配置 | Dataset 经 2–30 秒闭区间过滤后保留 2,167 组 validation 配对 |
| 两份 BSRNN 配置 | Dataset 仍保留全部 2,200 组 validation 配对；`trainer.val_metrics.max_utterances: 2167` 将指标计算限制为 2,167 组 |
| nonblind 评估音频 | `data/shared/urgent26_track1_se_dataset/urgent26_nonblind_with_label/{clean,noisy}/` |
| BSRNN baseline | ~37.8M 参数，非因果，内部 48 kHz 处理 |

---

## 📐 数据准备

### 离线配对 SCP

每条轨道分别使用 `prepare_scp.py` 单目录模式生成四列 SCP。单目录模式扫描**全部文件且不做 train/val 随机切分**：

```bash
DATA_ROOT=data/shared/urgent26_track1_se_dataset
UNIVERSAL_ROOT="$DATA_ROOT/urgent26_track1_universal_se"
NONBLIND_ROOT="$DATA_ROOT/urgent26_nonblind_with_label"

python tools/prepare_data_tools/scp/prepare_scp.py \
    "$UNIVERSAL_ROOT/simulation_train/clean" \
    -o "$UNIVERSAL_ROOT/urgent26_clean_train"
python tools/prepare_data_tools/scp/prepare_scp.py \
    "$UNIVERSAL_ROOT/simulation_train/noisy" \
    -o "$UNIVERSAL_ROOT/urgent26_noisy_train"
python tools/prepare_data_tools/scp/prepare_scp.py \
    "$UNIVERSAL_ROOT/simulation_validation/clean" \
    -o "$UNIVERSAL_ROOT/urgent26_clean_val"
python tools/prepare_data_tools/scp/prepare_scp.py \
    "$UNIVERSAL_ROOT/simulation_validation/noisy" \
    -o "$UNIVERSAL_ROOT/urgent26_noisy_val"

# nonblind 标签仅供评估，禁止写入训练或 Trainer validation 配置。
python tools/prepare_data_tools/scp/prepare_scp.py \
    "$NONBLIND_ROOT/clean" \
    -o "$NONBLIND_ROOT/urgent26_clean_nonblind"
python tools/prepare_data_tools/scp/prepare_scp.py \
    "$NONBLIND_ROOT/noisy" \
    -o "$NONBLIND_ROOT/urgent26_noisy_nonblind"
```

离线 YAML 精确使用以下四列（`uid fs length path`）路径：

- train clean：`data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/urgent26_clean_train.scp`
- train noisy：`data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/urgent26_noisy_train.scp`
- validation clean：`data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/urgent26_clean_val.scp`
- validation noisy：`data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/urgent26_noisy_val.scp`

全量语料包含 **436,240 组训练配对**（每轨约 709.349 小时）；原始 validation manifest 包含 **2,200 组配对**（每轨约 5.402 小时）。DPDFNet、GAN 等显式配置 `2.0 <= duration <= 30.0` 秒 Dataset 闭区间过滤的 recipe 实际保留 **2,167 组 validation 配对**：31 组短于 2 秒，2 组长于 30 秒。两份 BSRNN recipe 未配置该 Dataset filter，因此其 validation Dataset 仍有全部 **2,200 组配对**；其中 `trainer.val_metrics.max_utterances: 2167` 只把全局指标计算限制为前 2,167 组，不会删除 Dataset item。nonblind 集包含 **1,023 组评估配对**（每轨约 2.232 小时）。clean/noisy 两轨必须具有完全相同的 UID 集：clean 是监督 target，noisy 是对应的预混 degraded input。四列 SCP 已内嵌各轨真实采样率和长度，不再需要额外的 `utt2fs` 或 `speech_length.scp`。

nonblind clean/noisy SCP **仅用于评估**，不得用于模型训练、超参数选择或 Trainer validation。

目录形式的 `--urgent_noisy_dir` / `--urgent_clean_dir` 仍按目录规则扫描，不能处理 nonblind `noisy/0/*.flac` 等嵌套 FLAC，因此不要把该 noisy 根目录直接交给这两个参数。AEC `experiment_report` 默认把 `urgent26_noisy_nonblind.scp` 与 `urgent26_clean_nonblind.scp` 成对传给 AEC pipeline 的 `--urgent_noisy_scp` / `--urgent_clean_scp`：pipeline 会保留四列 metadata、校验 UID 集一致，并以 zero-farend 方式评估。两项 SCP 必须成对提供；它们仍不得用于训练、超参数选择或 Trainer validation。

### 动态混合

当前 dynamic recipe 使用以下独立 source manifest：

- `speech_scp`：`data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/urgent26_clean_train.scp`
- `noise_scp`：`data/shared/noise/filter/v2/train_noise_sources.scp`
- `windnoise_scp`：`data/shared/noise/filter/v2/train_wind_noise_sources.scp`
- `rir_scp`：`data/shared/rirs/gpurir_static/gpurir/fs16000/train_rirs.scp`

`urgent26_noisy_train.scp` 保存的是**预混 noisy mixture**，不是可独立叠加的噪声。它只能作为 offline clean/noisy 配对的 input，绝不能填入 dynamic `noise_scp`；dynamic `noise_scp` 必须是可与 clean speech 混合的独立噪声录音。

如需从其他语料准备 dynamic source，可明确使用 batch 模式；该命令与上面的全量离线配对 SCP 生成相互独立：

```bash
SPEECH_CORPUS=/path/to/speech-corpus
NOISE_CORPUS=/path/to/noise-corpus
RIR_CORPUS=/path/to/rir-corpus
WIND_NOISE_CORPUS=/path/to/wind-noise-corpus
python tools/prepare_data_tools/scp/prepare_scp.py \
    --speech "$SPEECH_CORPUS" --noise "$NOISE_CORPUS" \
    --rir "$RIR_CORPUS" --wind "$WIND_NOISE_CORPUS" \
    -o data/train_sources --val_ratio 0
```

---

## 🚀 训练命令

```bash
# BSRNN 非因果离线
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/usr_bsrnn_noncausal_urgent26_offline_multires_l1_spec.yaml

# BSRNN 因果离线
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/usr_bsrnn_causal_urgent26_offline_hybrid.yaml

# DPDFNet2 近因果离线（2 帧、约 20 ms 前瞻）
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/dpdfnet2_causal_urgent26_offline_hybrid.yaml

# DPDFNet2 近因果动态（2 帧、约 20 ms 前瞻）
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/dpdfnet2_causal_urgent26_dynamic_hybrid.yaml

# 动态 GAN 配置
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/gan/gtcrn_metricdiscriminator_dynamic.yaml
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/gan/mambattention_metricgan_dynamic.yaml
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/gan/mossformer_gan_metricgan_dynamic.yaml
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/gan/mp_senet_metricgan_dynamic.yaml
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/gan/rwsa_mamba_unet_metricgan_dynamic.yaml
```

---

<a id="multirate-bsrnn-ddp"></a>
## ⚠️ 多采样率 BSRNN 与 DDP

`distributed.find_unused_parameters: true` 在多采样率 BSRNN 训练时通常必需。

原因：低采样率 batch 中部分子带没有梯度，DDP 需要被告知"这一步某些参数没有梯度是正常的"；不加该开关，多采样率训练会直接中断。只有确实存在整条支路未被使用的配置才需要开启，单路径模型保持默认值。

---

## 📚 相关文档

- [主 README_CN](../../../README_CN.md)
- [DNS Challenge 3 / GTCRN](../dns_challenge3/README_CN.md)
- [URGENT2026 Track 1 官方仓库](https://github.com/urgent-challenge/urgent2026_challenge_track1)

<p align="center">
  <sub>最后更新: 2026-09-21 · 感谢 URGENT2026 组委会与 baseline 作者</sub>
</p>
