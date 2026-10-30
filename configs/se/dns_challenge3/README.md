# 🌐 DNS Challenge 3 — GTCRN Recipe

<p align="center">
  <strong>Offline premixed GTCRN paper-reproduction recipe on DNS Challenge 3 sources</strong>
</p>

---

## 📑 Contents

- [Scope](#-scope)
- [Configuration](#-configuration)
- [Protocol](#-gtcrn-section-3-protocol)
- [Workflow](#-workflow)
- [Related documentation](#-related-documentation)

---

## 📖 Scope

This directory is limited to GTCRN paper-protocol reproduction and framework validation using DNS Challenge 3 source pools. The canonical generator materializes paired clean/noisy waveforms for `data.mode: offline`; this is not dynamic mixing. The model and loss match the VoiceBank-DEMAND GTCRN recipe, while the scheduler, batch size, and data protocol differ.

The whole workflow is five steps: download the DNS3 sources, materialize paired waveforms plus SCP manifests, train, evaluate the generated paired test split, and optionally run inference on a noisy-only challenge set. In every command below, `/path/to/DNS-Challenge` stands for your DNS Challenge checkout and `data/...` paths resolve relative to the `use_train` repository root; nothing else needs editing.

> ONNX export lives under `onnx/`; see [onnx/README.md](../../../onnx/README.md). Use the [SE architecture document](../../../docs/architecture/SE_Architecture.md) and `tools/model_tools/audit_metrics.py` for parameter/MAC reports. THOP MACs are partial/unverified and independent FLOPs are not measured. Compute DNSMOS/NISQA/UTMOS/SCOREQ and other MOS metrics offline through the [evaluation pipeline](../../../evaluate/README.md), not in Trainer validation.

---

## 📋 Configuration

| File | Purpose |
|------|---------|
| [`gtcrn_causal_offline.yaml`](gtcrn_causal_offline.yaml) | The only DNS3 YAML: causal GTCRN on offline premixed pairs; experiment root `exp/se/dns_challenge3_gtcrn_causal_offline` |

---

## 📐 GTCRN Section 3 protocol

| Item | Value |
|------|-------|
| Sample rate | 16 kHz |
| Generated pairs | train 720,000 / val 840 / test 800; 10 seconds each |
| Source-pool split | train/val/test independently shuffle and cycle through the same clean/noise/RIR pools; there is no source-level hold-out, so this is a paper-reproduction protocol rather than a generalization benchmark |
| Mixing | full-RIR reverberant speech + noise; the generator samples `u` continuously and uniformly from `[-5, 15]` dB, then applies `alpha = 10^(-1.5u/20)` to energy-normalized noise. Therefore the resulting speech/noise power SNR is `1.5u`, approximately `[-7.5, 22.5]` dB; `u` itself is not the resulting power SNR |
| Target | clean speech convolved with the first 100 ms of the RIR starting at its direct-path peak |
| Training read length | generated files are 10 seconds; YAML `max_duration: 128000` randomly crops to at most 8 seconds at 16 kHz |
| Batch size | 16 |
| Optimizer | paper: AdamW, `lr=1e-3`, `wd=0.01`; current YAML: Adam, `lr=1e-3`, no weight decay |
| Scheduler | paper: ReduceLROnPlateau (`patience=5`, `factor=0.5`); current YAML: LinearWarmupCosineAnnealing (25,000 warmup / 250,000 decay optimizer steps) |
| Loss | Paper terminology: HybridLoss; the current YAML migrates/replaces the removed monolithic API with the explicit official-formula components `GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss` in `CompositeLoss` |

---

## 🚀 Workflow

### 1. Download DNS Challenge 3 sources

```bash
git clone https://github.com/microsoft/DNS-Challenge.git
cd DNS-Challenge
python download-dns-challenge-3.py
```

### 2. Materialize paired waveforms and SCP manifests

The output paths below match the current YAML. Both destinations must be writable; the generator publishes them transactionally and refuses to replace unowned directories. After running the DNS Challenge downloader inside its checkout, return to the `use_train` repository root before invoking this repository's generator.

```bash
cd /path/to/use_train
python -m tools.prepare_data_tools.generators.se.generate_dns3_offline_dataset \
    --data-root /path/to/DNS-Challenge \
    --output-dir data/se/dns_challenge3_generated \
    --scp-dir data/se/dns_challenge3 \
    --num-workers 8
```

Each generated split contains paired waveform directories plus strict four-column `clean.scp` / `noisy.scp` records (`uid fs length path`). A verified schema-v2 manifest binds the waveform and index roots to one generation identity. The YAML consumes the train and val manifests; the generated test split is a paired protocol test set, while official DNS V2/V3/V5 challenge sets remain noisy-only evaluation inputs. SCP 中的相对音频路径相对于对应 SCP 文件目录解析，因此迁移项目时应同时迁移生成目录与其音频文件。

### 3. Train

```bash
python train.py -C configs/se/dns_challenge3/gtcrn_causal_offline.yaml
```

### 4. Evaluate the generated paired test split

Checkpoints are stored below `trainer.exp_path`; substitute the actual run ID. The current unified SE CLI exposes its generic paired-SCP route under the `voicebank` dataset tag, so pass the DNS3 generated test SCPs explicitly:

```bash
python evaluate/eval_pipeline_se.py \
    -C configs/se/dns_challenge3/gtcrn_causal_offline.yaml \
    --checkpoint exp/se/dns_challenge3_gtcrn_causal_offline/checkpoints/<run_id>/best_model.tar \
    --datasets voicebank \
    --voicebank_clean_scp data/se/dns_challenge3/test/clean.scp \
    --voicebank_noisy_scp data/se/dns_challenge3/test/noisy.scp
```

With a clean reference, this route requests PESQ, STOI, ESTOI, and SI-SDR. The dataset tag only selects the paired evaluator; report the result as the generated DNS3 protocol test, not as VoiceBank-DEMAND.

### 5. Run inference on a noisy-only challenge set

The current DNS Challenge checkout places the V3 noisy-only blind set under `datasets_fullband/V2_V3_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k`. If your checkout uses a different layout, pass the directory that actually contains its noisy WAV files.

```bash
python inference.py \
    -C configs/se/dns_challenge3/gtcrn_causal_offline.yaml \
    --checkpoint exp/se/dns_challenge3_gtcrn_causal_offline/checkpoints/<run_id>/best_model.tar \
    --input_dir /path/to/DNS-Challenge/datasets_fullband/V2_V3_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k \
    --output_dir results/dns3_blind
```

---

## 📚 Related documentation

- [VoiceBank-DEMAND](../voicebank_demand/README.md)
- [URGENT2026](../urgent2026/README.md)
- [Project README](../../../README.md)

---

## 🙏 Acknowledgements

Thanks to the Microsoft DNS Challenge and GTCRN authors. This recipe is limited to paper-protocol comparison and framework validation.

<p align="center">
  <sub>Last updated: 2026-09-21</sub>
</p>
