# 🎤 VoiceBank-DEMAND — GTCRN Training and Deployment Guide

<p align="center">
  <strong>Reproducing the VoiceBank-DEMAND baseline with GTCRN (ICASSP 2024)</strong><br>
  End-to-end data preparation, training, evaluation, and ONNX deployment
</p>

---

> 💡 For a short training/evaluation/inference/ONNX walkthrough, see the [VoiceBank deployment quickstart](../../../docs/guides/se/VoiceBank_Deployment_Quickstart.md).

This directory contains the VoiceBank-DEMAND recipes. VoiceBank-DEMAND is a small paired corpus in which every noisy utterance has an exactly matching clean utterance, which makes it a convenient first end-to-end run: build the SCP manifests, train, evaluate on the official test set, and optionally export to ONNX. Training, inference, evaluation, and export all go through the same entry points (`train.py`, `inference.py`, `evaluate/eval_pipeline_se.py`, `onnx/`) as every other SE recipe.

Words used below:

| Term | Meaning |
|---|---|
| SCP | a whitespace-separated manifest whose rows are `uid fs length path` |
| PESQ / STOI / ESTOI / SI-SDR | intrusive quality metrics computed against the clean reference; higher is better |
| ONNX | an interchange format for deploying a trained model outside PyTorch |
| WOLA | weighted overlap-add, the specific analysis/synthesis window pairing used by the official implementation |
| RTF | real-time factor: processing time divided by audio duration; below 1.0 means faster than real time |
| THOP | the hook-based profiler used for the MAC counts in this repository |

This directory currently provides the following training configurations:

| Configuration | Model and boundary | Dedicated experiment directory |
|---|---|---|
| [`gtcrn_causal_offline.yaml`](gtcrn_causal_offline.yaml) | Causal spectral GTCRN trained on offline premixed pairs | `exp/se/gtcrn_causal_offline` |
| [`zipenhancer_s.yaml`](zipenhancer_s.yaml) | ZipEnhancer-S | `exp/se/zipenhancer_s` |
| [`fast_ulcnet_offline.yaml`](fast_ulcnet_offline.yaml) | FastULCNet; the standard Hann window is not the official `hann75`/WOLA setup, and no checkpoint or real 90-second quality reproduction is available yet | `exp/se/fast_ulcnet_offline` |
| [`meanflowse_ncsnpp_offline.yaml`](meanflowse_ncsnpp_offline.yaml) | Project-native pure-PyTorch NCSN++-style MeanFlowSE; incompatible with official checkpoints and not yet trained to reproduce paper metrics | `exp/se/meanflowse_ncsnpp_offline` |
| [`dccrn_offline.yaml`](dccrn_offline.yaml) | DCCRN (original DCCRN-CL: complex encoder / LSTM / decoder, E-mode complex mask); model topology only — the framework manages STFT/iSTFT, so no official training config, checkpoint, or paper metric is claimed | `exp/se/dccrn_offline` |
| [`meanse_ncsnpp_offline.yaml`](meanse_ncsnpp_offline.yaml) | MeanSE (arXiv:2509.21214) final curriculum stage (`max_interval=1.0`); a different paper and implementation from MeanFlowSE, with no confirmed licence or public checkpoint, so official structure/checkpoint compatibility is not claimed | `exp/se/meanse_ncsnpp_offline` |
| [`mtfaa_net_causal_offline.yaml`](mtfaa_net_causal_offline.yaml) | MTFAA-Net-style model (phase encoder + multi-scale TFCM + axial attention + two-stage MTFAA mask), adapted from a non-official implementation whose upstream states details may differ from the paper | `exp/se/mtfaa_net_causal_offline` |
| [`rt_semamba_kd8_causal_offline.yaml`](rt_semamba_kd8_causal_offline.yaml) | RT-SEMamba with eight cTF-Mamba blocks (spectra in, spectra out); trains the discriminative generator only, without the official metric discriminator / PESQ GAN recipe | `exp/se/rt_semamba_kd8_causal_offline` |
| [`rt_semamba_kd1_distill_causal_offline.yaml`](rt_semamba_kd1_distill_causal_offline.yaml) | RT-SEMamba KD1 progressive distillation: frozen 8-block teacher → 1-block student; the teacher checkpoint must be strictly converted or verified first | `exp/se/rt_semamba_kd1_distill_causal_offline` |
| [`tf_skimnet_causal_offline_mag_complex_l1.yaml`](tf_skimnet_causal_offline_mag_complex_l1.yaml) | TF-SkiMNet (Interspeech 2025); the upstream repository ships no training code, checkpoint, evaluation code, or licence, and the current loss is a usable consistency surrogate rather than the paper's exact RI + magnitude recipe | `exp/se/tf_skimnet_causal_offline_mag_complex_l1` |

The two RT-SEMamba recipes deviate from the official front end: the official setup uses `center=False` with a Hann window, which the shared `torch.istft` rejects (NOLA), so these configs use the same 400/100/400 geometry with a Hamming window and do not claim numerically exact front-end reproduction. Each YAML header repeats its own audit summary (`Params` / `MACs` / `FLOPs` / `Trainer MOS boundary`).

These experiment directories belong only to the VoiceBank-DEMAND recipe. The DNS Challenge 3 GTCRN recipe uses `exp/se/dns_challenge3_gtcrn_causal_offline`; do not resume from or reuse checkpoints across datasets.

For parameter/MAC counts and unmeasured items, use the [SE architecture document](../../../docs/architecture/SE_Architecture.md) and `tools/model_tools/audit_metrics.py`. The reported THOP MAC count is partial/unverified because hooks may omit functional, complex, custom, or unsupported operators; independent FLOPs are not measured. Compute DNSMOS/NISQA/UTMOS/SCOREQ and other MOS metrics offline through the [evaluation pipeline](../../../evaluate/README.md), not in Trainer validation.

---

## 📑 Contents

- [Paper source](#-paper-source)
- [1. Original dataset](#1-original-dataset)
- [2. Data preparation](#2-data-preparation)
- [3. Paper hyperparameters](#3-paper-hyperparameters-gtcrn-section-3)
- [4. Commands](#4-commands)
- [5. Model architecture](#5-model-architecture)
- [6. Deployment](#voicebank-deployment)
- [7. File layout](#7-file-layout)
- [8. FAQ](#8-faq)
- [9. References](#9-references)

---

## 📖 Paper source

The recipe follows the **GTCRN** (ICASSP 2024) Section 3 setup on VCTK-DEMAND.

> C. Valentini-Botinhao et al., "Investigating RNN-based speech enhancement methods for noise-robust text-to-speech," Proc. ISCA SSW, 2016.

---

## 1. Original dataset

| Item | Value |
|---|---|
| Source | Edinburgh DataShare (handle 10283/1942) |
| Archives | `clean_trainset_wav.zip`, `noisy_trainset_wav.zip`, `clean_testset_wav.zip`, `noisy_testset_wav.zip` |

### Dataset partitions

| Partition | Speakers | Files | On-disk sample rate |
|---|---:|---:|---:|
| Official training set | 28 | 11,572 | 16 kHz |
| Official test set, used as validation by these configs | 2 (p232, p257) | 824 | 48 kHz |

### Noise types and SNRs

| Partition | SNRs (dB) | Noise conditions |
|---|---|---|
| Official training set | 0, 5, 10, 15 | Training noise conditions |
| Official test set | 2.5, 7.5, 12.5, 17.5 | Five noise conditions unseen during training |

The train and test conditions are deliberately different: the official test set evaluates generalization to unseen noises and interleaved SNRs, rather than reusing the training noise/SNR combinations.

> **Evaluation protocol warning:** every YAML in this directory currently points `voicebank_*_val.scp` at the complete official test set. Using that validation loss for checkpoint selection, early stopping, or hyperparameter tuning leaks final-test information into model development. For a strict experiment, split an independent clean/noisy validation partition from the official training set, keep the official test set untouched, update the YAML paths locally for that validation split, and pass the reserved test SCPs explicitly to the final evaluation command.

---

## 2. Data preparation

### Step 1: Extract the archives

```bash
unzip clean_trainset_wav.zip -d data/se/voicebank_demand
unzip noisy_trainset_wav.zip -d data/se/voicebank_demand
unzip clean_testset_wav.zip -d data/se/voicebank_demand
unzip noisy_testset_wav.zip -d data/se/voicebank_demand
```

Because `data/se/voicebank_demand` is a relative `-d` path, run these commands from the repository root. If your shell is elsewhere, use the repository's absolute `data/se/voicebank_demand` path or adjust the relative path accordingly. Archive layouts vary by distribution: the commands below use the common layout where each test archive contains a same-named inner directory. If your archive extracts directly to WAV files, pass that directory instead; `prepare_scp.py` always indexes the directory supplied on the command line.

### Step 2: Generate the exact SCP files

Run `prepare_scp.py` once per track in single-directory mode. Single-directory mode indexes the entire directory and performs **no train/validation split**:

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

The four output paths consumed by every YAML in this directory are exactly:

- `data/se/voicebank_demand/voicebank_clean_train.scp`
- `data/se/voicebank_demand/voicebank_noisy_train.scp`
- `data/se/voicebank_demand/voicebank_clean_val.scp`
- `data/se/voicebank_demand/voicebank_noisy_val.scp`

Each SCP row is `uid fs length path`. The prepared full training tracks contain **11,572 matched clean/noisy pairs** (9.391 h per track, 16 kHz). The validation tracks are the complete official test set: **824 matched pairs** (0.576 h per track, stored at 48 kHz). The offline dataset resamples validation audio to the configured `data.sample_rate: 16000` at load time. `trainer.val_metrics.max_utterances` is only a global cap: the GTCRN value `872` does not create or imply 872 validation examples; at most the actual 824 pairs are available.

`prepare_scp.py` single-directory mode probes metadata with `soundfile` and writes the complete directory without splitting. Add `--validate_audio` to each command when the source has not already been verified and a full decode check for empty, non-finite, clipped, silent, or high-DC files is required. After generation, verify all four counts and then instantiate the offline dataset (or start training) to enforce identical UID sets and bind declared `fs` / `length` to decoded headers.

Clean and noisy files are paired by identical UID, not merely by line number. Generate both tracks from corresponding directory levels so that `p232_001`, for example, appears under the same UID in both validation SCPs. VoiceBank-DEMAND is already a paired premixed corpus: `voicebank_clean_*` is the supervised target and `voicebank_noisy_*` is its degraded input. It does not sample RIRs or construct an early-RIR target, and a noisy-mixture SCP is never a dynamic `noise_scp` source.

The GTCRN paper uses random 2-second segments, while the current GTCRN VoiceBank configuration uses `max_duration: 128000` (8 seconds at 16 kHz); these are different reproduction settings.

---

## 3. Paper hyperparameters (GTCRN, Section 3)

| Parameter | Value | Notes |
|---|---|---|
| Sample rate | 16,000 Hz | Test audio is resampled from 48 kHz |
| Training segment | 2 seconds | Random crop |
| Batch size | 4 | Section 3.2 |
| Optimizer | Adam | `lr=0.001` in the paper |
| Scheduler | ReduceLROnPlateau | `factor=0.5`, `patience=5` |
| Loss | HybridLoss (paper terminology) | α=0.01, β=0.3; the current YAML migrates/replaces the removed monolithic API with the explicit official-formula components `GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss` in `CompositeLoss` |

### Loss (paper Eq. 1)

```
L = α·L_SISNR + (1-β)·L_mag + β·(L_real + L_imag)
α = 0.01,  β = 0.3
```

### Historical reproduction record

| Metric | GTCRN paper | Historical use_train record |
|---|---:|---:|
| PESQ | 2.87 | **2.8031** |
| STOI | 0.940 | **0.9331** |
| ESTOI | 0.848 | **0.8322** |
| SI-SDR | 18.8 dB | **18.2864 dB** |

**Provenance:** these four values are preserved from the repository's legacy evaluation transcript for all 824 official VoiceBank-DEMAND test utterances. That transcript identified checkpoint `exp/gtcrn_vb/checkpoints/2026-03-05_10-31-00/best_model.tar` and the former `evaluate/evaluate_voicebank.py --no_mos` command. The checkpoint and raw evaluation artifact are not present in the current working tree, so the numbers were **not remeasured in this update** and must not be treated as a current-run result. For a new result, record the current checkpoint path or content hash, config, exact 824-pair SCPs, dependency versions, command, and raw output from `evaluate/eval_pipeline_se.py`.

The current GTCRN YAML's Trainer validation list is `[SI-SDR, PESQ, STOI]`; it does not log ESTOI during training. The paired offline evaluation command below requests PESQ, STOI, ESTOI, and SI-SDR because `evaluate.common.requested_standard_metrics` enables all four whenever a clean reference SCP is present.

---

## 4. Commands

### Training

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python train.py -C configs/se/voicebank_demand/gtcrn_causal_offline.yaml

# Multi-GPU DDP
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
    -C configs/se/voicebank_demand/gtcrn_causal_offline.yaml

# Resume
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/voicebank_demand/gtcrn_causal_offline.yaml --resume
```

### Inference

```bash
python inference.py \
    -C configs/se/voicebank_demand/gtcrn_causal_offline.yaml \
    --checkpoint "exp/se/gtcrn_causal_offline/checkpoints/<run-id>/best_model.tar" \
    --input_dir path/to/noisy/wavs --output_dir results/enhanced
```

### Evaluation

```bash
python evaluate/eval_pipeline_se.py \
    -C configs/se/voicebank_demand/gtcrn_causal_offline.yaml \
    --checkpoint "exp/se/gtcrn_causal_offline/checkpoints/<run-id>/best_model.tar" \
    --datasets voicebank \
    --voicebank_clean_scp data/se/voicebank_demand/voicebank_clean_val.scp \
    --voicebank_noisy_scp data/se/voicebank_demand/voicebank_noisy_val.scp
```

---

## 5. Model architecture

| Parameter | Value |
|---|---|
| `n_fft` | 512 |
| `hop_length` | 256 |
| `win_length` | 512 |
| Parameters | 23.67K trainable / ~48K total |
| MACs | ~34M for a 1-second, 16 kHz input (THOP, partial/unverified; STFT/iSTFT normally excluded) |
| FLOPs | not measured |
| Frame shift | 16 ms (`hop=256` at 16 kHz; not end-to-end system latency) |

### GTCRN architecture

```
Input: noisy waveform (16 kHz)
  ↓ STFT (n_fft=512, hop=256)
  ↓ ERB filter bank (frequency compression: 257 → 32)
  ↓ SFE (Shallow Feature Extraction) + Encoder
  ↓ DPGRNN × 2 (band-split + inter-band RNN)
  ↓ Decoder + iERB + Mask
  ↓ iSTFT → enhanced waveform
```

---

<a id="voicebank-deployment"></a>
## 6. Deployment preparation (ONNX debug/general artifacts)

### Debug export and verified publication

```bash
# Streaming GTCRN debug/general export; --smoke runs one PT/ORT smoke comparison
python onnx/export.py \
    --model streaming_gtcrn \
    --checkpoint "exp/se/gtcrn_causal_offline/checkpoints/<run-id>/best_model.tar" \
    --output runs/onnx_debug/gtcrn_streaming.onnx \
    --smoke

# Formal publication: offline PT, frame-wise streaming PT, and frame-wise ORT must all PASS
python onnx/verify/verify_3way_gtcrn.py \
    --checkpoint "exp/se/gtcrn_causal_offline/checkpoints/<run-id>/best_model.tar" \
    --save-onnx onnx/onnx_model/gtcrn_streaming.onnx
```

`onnx/export.py --output` creates only a debug/general artifact. `--smoke` (compatibility alias `--verify`) is not the three-way publication gate. A formal artifact must be written by `verify_3way_gtcrn.py` only after strict checkpoint loading, runtime metadata validation, and all three comparisons pass. The verifier fixes GTCRN to `n_fft=512`, does not accept `-C/--config`, and therefore requires a matching checkpoint.

### Historical performance reference (not remeasured)

The following figures have no traceable script, environment, or raw record and are retained only as historical context. They are not a current performance commitment.

| Hardware | Precision | RTF | Notes |
|---|---|---:|---|
| Intel i7-11800H | FP32 | ~0.06 | PyTorch eager |
| Intel i7-11800H | FP32 | ~0.04 | ONNX Runtime |
| ARM Cortex-A76 | FP32 | ~0.13 | Edge device |

> RTF below 1.0 normally means processing is faster than audio duration; the values above have not been remeasured on the current version.

### Deployment checklist

| Item | Training mode | Deployment mode |
|---|---|---|
| STFT `center` | Follow the actual training config | Neither `True` nor `False` is universally correct; preserve training framing semantics |
| Inference | Full-sequence `forward()` | Explicit chunk framing, buffering, and boundary handling |
| Batch size | 4+ | 1 |
| Model format | PyTorch `.tar` | ONNX `.onnx` |

Before deployment, freeze the STFT/iSTFT framing and buffering strategy and compare the offline and deployment paths on identical inputs. Switching `center` alone does not prove causality, streaming behavior, or equivalence.

---

## 7. File layout

```
use_train/
├── configs/se/voicebank_demand/
│   ├── gtcrn_causal_offline.yaml
│   ├── zipenhancer_s.yaml
│   ├── fast_ulcnet_offline.yaml
│   ├── meanflowse_ncsnpp_offline.yaml
│   ├── dccrn_offline.yaml
│   ├── meanse_ncsnpp_offline.yaml
│   ├── mtfaa_net_causal_offline.yaml
│   ├── rt_semamba_kd8_causal_offline.yaml
│   ├── rt_semamba_kd1_distill_causal_offline.yaml
│   ├── tf_skimnet_causal_offline_mag_complex_l1.yaml
│   ├── README.md
│   └── README_CN.md
├── data/se/voicebank_demand/
│   ├── voicebank_clean_train.scp
│   ├── voicebank_noisy_train.scp
│   ├── voicebank_clean_val.scp
│   ├── voicebank_noisy_val.scp
│   ├── {clean,noisy}_trainset_wav/
│   └── {clean,noisy}_testset_wav/
├── tools/prepare_data_tools/scp/prepare_scp.py
├── onnx/{export.py, models/streaming_gtcrn.py}
└── exp/se/gtcrn_causal_offline/
    └── checkpoints/<run-id>/{best_model.tar, latest_model.tar}
```

---

## 8. FAQ

<details>
<summary>Q1: Must the test set be resampled to 16 kHz?</summary>

Yes. The files are stored at 48 kHz, while GTCRN is designed for 16 kHz. The current offline loader resamples them to the configured 16 kHz target at read time.
</details>

<details>
<summary>Q2: What do `lamda_ri=30` and `lamda_mag=70` mean?</summary>

They correspond to β×100 and (1-β)×100 and match the official `30*(rl+il) + 70*ml + sisnr` implementation.
</details>

<details>
<summary>Q3: Why do validation metrics differ from the paper?</summary>

Treat resampling, optimizer and scheduler settings, segment length, the actual stopping epoch, and PESQ implementation/version/mode as comparison points, not proven causes. In particular, switching to Adam, training for roughly 80–100 epochs, or changing the PESQ implementation does not by itself guarantee the paper metrics; the available evidence does not establish that any one of these differences caused the gap.
</details>

---

## 9. References

1. GTCRN: "A Speech Enhancement Model Requiring Ultralow Computational Resources" (ICASSP 2024)
2. VoiceBank-DEMAND: Valentini-Botinhao et al., ISCA SSW 2016
3. [Official GTCRN repository](https://github.com/AkenoSyuRi/GTCRN)
4. [DEMAND database](https://zenodo.org/record/1227121)

---

<p align="center">
  <sub>Last updated: 2026-09-21 · Thanks to the VoiceBank-DEMAND and GTCRN authors</sub>
</p>
