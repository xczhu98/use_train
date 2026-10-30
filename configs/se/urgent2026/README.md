# 🏆 URGENT2026 Challenge — Experiment Configurations

<p align="center">
  <strong>BSRNN / DPDFNet2 · Offline pairs and dynamic mixing</strong>
</p>

---

## 📑 Contents

- [Overview](#-overview)
- [Key terms](#-key-terms)
- [Configurations](#-configurations)
- [Dataset (Track 1)](#-dataset-track-1)
- [Non-causal BSRNN baseline](#-non-causal-bsrnn-baseline)
- [Data preparation](#-data-preparation)
- [Training commands](#-training-commands)
- [Multi-rate BSRNN and DDP](#multirate-bsrnn-ddp)
- [Related documentation](#-related-documentation)

---

## 📖 Overview

This directory aligns with the **[URGENT2026 Challenge](https://urgent-challenge.github.io/urgent2026/)** and contains the BSRNN baseline plus causal and dynamic variants.

Track 1 asks a single model to denoise, dereverberate, and repair channel damage across many sample rates, so each recipe here is "one model, many recording conditions". The four recipes in the first table are the supervised baselines; the five recipes under `gan/` are an adversarial variant of the same task. A new reader should start with the non-causal BSRNN baseline below: it has the fewest moving parts and no dynamic mixing.

> Upstream baseline reference: [`urgent-challenge/urgent2026_challenge_track1`](https://github.com/urgent-challenge/urgent2026_challenge_track1), especially upstream `baseline_code/models/bsrnn.py` and `conf/models/BSRNN_baseline.yaml`. Those are upstream repository paths, not paths in this checkout. The corresponding implementation in this repository is `tasks/se/models/discriminative/bsrnn.py` with the core under `tasks/se/models/modules/bsrnn_impl/`; the local baseline recipe is `usr_bsrnn_noncausal_urgent26_offline_multires_l1_spec.yaml`.
>
> Subdirectories: [`pase/`](pase/README.md) for PASE/StuPASE plus the indexed [`unipase/`](pase/unipase/README.md) and [`gap_urgenet/`](pase/gap_urgenet/README.md) families; `gan/` contains the five dynamically mixed GAN recipes indexed below.
>
> The DNS Challenge 3 directory currently provides a GTCRN offline recipe only. Other SE recipes are documented in this directory.

### 🔤 Key terms

| Term | Meaning |
|---|---|
| Track 1 | the challenge track this repository targets: universal speech enhancement (denoising + dereverberation + channel restoration) |
| BSRNN | band-split RNN: the model splits the spectrum into frequency sub-bands and processes them with recurrent layers |
| causal / non-causal | a causal model may only use past frames (required for streaming); a non-causal model may look at the whole utterance |
| DPDFNet2 | a lightweight enhancer with a fixed algorithmic lookahead of 2 frames (about 20 ms at 16 kHz); the existing `causal` filename does not mean zero lookahead |
| offline recipe | trains on clean/noisy pairs that already exist on disk |
| dynamic recipe | synthesizes the noisy input while training, from clean speech plus noise/RIR manifests |
| MetricGAN family | adversarial training where the discriminator is trained to estimate a perceptual quality metric |
| PESQ | an intrusive perceptual quality metric (higher is better); `gan/gtcrn_metricdiscriminator_dynamic.yaml` is the one recipe that does **not** inject it into the discriminator |
| nonblind | evaluation audio that ships with clean labels; it is never training data |
| DDP | distributed data parallel multi-GPU training |

---

## 📋 Configurations

| File | Model | Data mode | Loss |
|---|---|---|---|
| `usr_bsrnn_noncausal_urgent26_offline_multires_l1_spec.yaml` | Non-causal BSRNN | Offline | MultiResL1SpecLoss |
| `usr_bsrnn_causal_urgent26_offline_hybrid.yaml` | Causal BSRNN | Offline | Explicit CompositeLoss (`GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`) |
| `dpdfnet2_causal_urgent26_offline_hybrid.yaml` | Near-causal DPDFNet2 (2-frame lookahead) | Offline | Explicit CompositeLoss (`GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`) |
| `dpdfnet2_causal_urgent26_dynamic_hybrid.yaml` | Near-causal DPDFNet2 (2-frame lookahead) | Dynamic | Explicit CompositeLoss (`GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`) |

All four files above are directly under `configs/se/urgent2026/`. The five recipes under `gan/` are:

| File | Generator / discriminator | Trainer | Data mode |
|---|---|---|---|
| `gan/gtcrn_metricdiscriminator_dynamic.yaml` | `GTCRN` / `MetricDiscriminator` | `GANTrainer` (ordinary LSGAN; no PESQ injection) | Dynamic |
| `gan/mambattention_metricgan_dynamic.yaml` | `MambAttentionGenerator` / `MetricMagDiscriminator` | `MetricGANCompositeTrainer` | Dynamic |
| `gan/mossformer_gan_metricgan_dynamic.yaml` | `MossFormerGANGenerator` / `MetricMagDiscriminator` | `MetricGANCompositeTrainer` | Dynamic |
| `gan/mp_senet_metricgan_dynamic.yaml` | `MPSEnetGenerator` / `MetricMagDiscriminator` | `MetricGANCompositeTrainer` | Dynamic |
| `gan/rwsa_mamba_unet_metricgan_dynamic.yaml` | `RWSAMambaUNetGenerator` / `MetricMagDiscriminator` | `MetricGANCompositeTrainer` | Dynamic |

> DPDFNet2 keeps `df_lookahead=2` and `conv_lookahead=2`. Its cache reuses history and supports real-time frame-by-frame processing, but does not remove the two-frame lookahead. At 16 kHz with `hop_length=160`, the fixed algorithmic lookahead is about 20 ms. The existing `causal` filename does not mean zero lookahead.

---

## 📊 Dataset (Track 1)

| Item | Value |
|---|---|
| Task | Universal speech enhancement: denoising, dereverberation, and channel restoration |
| On-disk sample rates | 8 / 16 / 22.05 / 24 / 32 / 44.1 / 48 kHz in training; validation excludes 8 kHz |
| Offline training audio | `data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/simulation_train/{clean,noisy}/` |
| Offline validation audio | `data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/simulation_validation/{clean,noisy}/` |
| Raw validation manifest | 2,200 clean/noisy pairs |
| DPDFNet/GAN and other duration-filtered recipes | Dataset retains 2,167 validation pairs after the inclusive 2–30 s filter |
| Two BSRNN recipes | Dataset retains all 2,200 validation pairs; `trainer.val_metrics.max_utterances: 2167` caps metric computation to 2,167 |
| Nonblind evaluation audio | `data/shared/urgent26_track1_se_dataset/urgent26_nonblind_with_label/{clean,noisy}/` |

---

## 📐 Non-causal BSRNN baseline

| Parameter | Value |
|---|---|
| `num_channel` | 196 |
| `num_layer` | 6 |
| `causal` | false |
| Parameters | ~37.8M |
| STFT (top-level `stft` block) | 960 / 480 at 48 kHz |

---

## 📐 Data preparation

### Offline paired SCPs

Run `prepare_scp.py` once per track in single-directory mode. This mode scans the **complete directory without a random train/validation split**:

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

# Nonblind labels are evaluation-only; do not add these SCPs to training or validation YAML.
python tools/prepare_data_tools/scp/prepare_scp.py \
    "$NONBLIND_ROOT/clean" \
    -o "$NONBLIND_ROOT/urgent26_clean_nonblind"
python tools/prepare_data_tools/scp/prepare_scp.py \
    "$NONBLIND_ROOT/noisy" \
    -o "$NONBLIND_ROOT/urgent26_noisy_nonblind"
```

The offline YAML files consume exactly these four-column (`uid fs length path`) paths:

- train clean: `data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/urgent26_clean_train.scp`
- train noisy: `data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/urgent26_noisy_train.scp`
- validation clean: `data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/urgent26_clean_val.scp`
- validation noisy: `data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/urgent26_noisy_val.scp`

The complete prepared corpus contains **436,240 training pairs** (709.349 h per track) and a raw validation manifest of **2,200 pairs** (5.402 h per track). DPDFNet, GAN, and other recipes that configure the inclusive `2.0 <= duration <= 30.0` second Dataset filter expose **2,167 validation pairs**; 31 pairs are shorter than 2 seconds and 2 pairs are longer than 30 seconds. The two BSRNN recipes do not configure this Dataset filter, so their validation Dataset still contains all **2,200 pairs**; their `trainer.val_metrics.max_utterances: 2167` only limits metric computation to the first 2,167 globally and does not remove Dataset items. The nonblind set contains **1,023 evaluation pairs** (2.232 h per track). Clean and noisy tracks must have identical UID sets: clean is the supervised target, and noisy is the corresponding premixed degraded input.

The nonblind clean/noisy SCPs are **evaluation-only**. They must not be used for model training, hyperparameter selection, or Trainer validation.

目录形式的 `--urgent_noisy_dir` / `--urgent_clean_dir` 仍采用扫描契约，不能处理 nonblind `noisy/0/*.flac` 等嵌套 FLAC。AEC `experiment_report` 默认将 `urgent26_noisy_nonblind.scp` 与 `urgent26_clean_nonblind.scp` 成对传给 AEC pipeline 的 `--urgent_noisy_scp` / `--urgent_clean_scp`；pipeline 会保留四列 metadata、校验 UID 集一致，并以 zero-farend 方式评估。两项 SCP 必须成对提供；它们仍不得用于训练、超参数选择或 Trainer validation。

### Dynamic mixing

Dynamic recipes use the clean training manifest as their speech source:

- `speech_scp`: `data/shared/urgent26_track1_se_dataset/urgent26_track1_universal_se/urgent26_clean_train.scp`
- `noise_scp`: `data/shared/noise/filter/v2/train_noise_sources.scp`
- `windnoise_scp`: `data/shared/noise/filter/v2/train_wind_noise_sources.scp`
- `rir_scp`: `data/shared/rirs/gpurir_static/gpurir/fs16000/train_rirs.scp`

SE Urgent2026 动态配置统一使用按 split 发布的 16 kHz 合并 RIR 清单；train
清单包含 DNS5 RIR 与静态 gpuRIR，validation 使用对应 `val_rirs.scp`。
`urgent26_noisy_train.scp` contains **premixed noisy mixtures**, not isolated noise. It is valid only as the input side of an offline clean/noisy pair and must never be passed as dynamic `noise_scp`. Dynamic `noise_scp` must contain standalone noise recordings suitable for mixing with clean speech.

If preparing separate dynamic source corpora, use batch mode explicitly and choose the desired split policy; this is independent of the full offline-pair commands above:

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

## 🚀 Training commands

```bash
# Non-causal offline BSRNN (baseline style)
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/usr_bsrnn_noncausal_urgent26_offline_multires_l1_spec.yaml

# Causal offline BSRNN
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/usr_bsrnn_causal_urgent26_offline_hybrid.yaml

# Offline DPDFNet2
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/dpdfnet2_causal_urgent26_offline_hybrid.yaml

# Dynamic DPDFNet2
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/urgent2026/dpdfnet2_causal_urgent26_dynamic_hybrid.yaml

# Dynamic GAN recipes
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
## ⚠️ Multi-rate BSRNN and DDP

`distributed.find_unused_parameters: true` is normally required for multi-rate BSRNN training because some subbands receive no gradients in lower-sample-rate batches.

In plain words: DDP must be told that a parameter without a gradient in the current step is expected, not a bug; without this flag the multi-rate run aborts. Set it only for recipes whose bands genuinely go unused, and leave single-path models at the default.

---

## 📚 Related documentation

- [Main README](../../../README.md)
- [DNS Challenge 3 / GTCRN](../dns_challenge3/README.md)
- [VoiceBank-DEMAND](../voicebank_demand/README.md)
- [Official URGENT2026 Track 1 repository](https://github.com/urgent-challenge/urgent2026_challenge_track1)

<p align="center">
  <sub>Last updated: 2026-09-21 · Thanks to the URGENT2026 organizers and baseline authors</sub>
</p>
