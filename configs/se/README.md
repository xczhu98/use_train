# 🎧 Speech Enhancement Configs (`configs/se`)

<p align="center">
  <strong>Training recipes for VoiceBank-DEMAND, DNS Challenge 3, and URGENT2026</strong>
</p>

---

Speech enhancement (SE) removes noise from a recording and returns a cleaned-up version of the same speaker; some recipes additionally handle reverberation or channel damage. Each subdirectory below holds the recipes for one dataset or challenge, and every recipe starts the same way:

```bash
python train.py -C configs/se/<subdirectory>/<recipe>.yaml
```

Training writes everything into the recipe's `trainer.exp_path` (contract: `exp/se/<recipe filename without .yaml>`), so two recipes never share a run directory.

## 📑 Contents

- [Key terms used below](#-key-terms-used-below)
- [Configuration overview](#-configuration-overview)
- [Dataset documentation](#-dataset-documentation)
- [Validation simulation policy](#inheritance-semantics)
- [Dynamic balanced sampling (optional)](#balanced-sampling)
- [Quick links](#-quick-links)

---

## 🔤 Key terms used below

| Term | Meaning |
|---|---|
| recipe | one YAML file describing a complete training run |
| `data.mode` | the routing field that decides which task and dataset builder reads the config. Repo-wide there are nine legal values; SE uses `dynamic` and `offline` |
| `offline` | the noisy input is already premixed on disk, so you supply a clean/noisy SCP pair |
| `dynamic` | the noisy input is synthesized while training, from clean speech plus noise/RIR manifests |
| A–F layers | the six synthesis layers a dynamic recipe runs: speech preprocessing, scene branch, acoustic mixing, post-mix amplitude conditioning, channel degradation, device degradation. Full contract: [`Simulation_Flow_Logic.md`](../../docs/guides/simulation/Simulation_Flow_Logic.md) |
| SCP | a whitespace-separated manifest whose rows are `uid fs length path` |
| standalone recipe | every shipped YAML declares its complete model, data, loss, optimizer, scheduler, trainer, and distributed sections |
| STFT | short-time Fourier transform; the framing step that turns a waveform into a spectrogram before the model sees it |
| SI-SDR / PESQ / STOI / ESTOI | intrusive quality metrics computed during validation against a clean reference (higher is better) |
| DNSMOS / NISQA / UTMOS / SCOREQ | MOS-style quality predictors that the Trainer never computes; run them offline through [`evaluate/`](../../evaluate/README.md) |
| DDP | distributed data parallel multi-GPU training |

---

## 📁 Configuration overview

| Directory | Scope |
|------|------|
| **urgent2026/** | URGENT2026 BSRNN (causal/noncausal) and DPDFNet2 offline/dynamic recipes, plus `pase/` (PASE/StuPASE/UniPASE/GAP-URGENet) and `gan/` (MetricGAN-family) subdirectories |
| **dns_challenge3/** | DNS Challenge 3 source-pool protocol; currently provides `gtcrn_causal_offline.yaml` |
| **voicebank_demand/** | Ten VoiceBank-DEMAND recipes: GTCRN, ZipEnhancer-S, FastULCNet, MeanFlowSE, MeanSE, DCCRN, MTFAANet, RT-SEMamba (KD8 and KD1 distillation) and TFSkiMNet |
| **spatial/** | NBSS SpatialNet/OnlineSpatialNet paper-aligned recipes at 8 kHz: one offline reference plus three online streaming variants (`mhsa`/`ret`/`mamba`) and their 32 s LF-stage configs (including a 36-epoch MHSA variant) |

Data-preparation tools live under `tools/prepare_data_tools/`. The complete `model.type` mapping is `tasks/se/models/__init__.py::_MODEL_REGISTRY_MAP`; this index lists only YAML examples shipped in the repository.

> **Reproduction boundaries:** RWSA-MambaUNet is aligned to upstream commit `02635ab7e92a0584fde729f8a55f398c2379da82`, but official checkpoints and paper metrics have not been verified. FastULCNet implements the CRM/Comfi path and provides `tools/verify/verify_fast_ulcnet_state_drift.py`; the shared standard Hann STFT differs from official `hann75`/WOLA, and no checkpoint or real 90-second quality reproduction is available. MeanFlowSE is a project-native pure-PyTorch NCSN++-style implementation with complex STFT, MeanFlow objective, and finite-difference JVP; it is incompatible with official checkpoints and has not reproduced paper metrics.
>
> **Reporting contract:** use [`docs/architecture/SE_Architecture.md`](../../docs/architecture/SE_Architecture.md) and `tools/model_tools/audit_metrics.py` for parameter/MAC reports. THOP results are partial/unverified; independent FLOPs remain `not measured` unless separately profiled. Trainer validation computes intrusive metrics only; run DNSMOS/NISQA/UTMOS/SCOREQ and other MOS metrics through the offline [`evaluate/`](../../evaluate/README.md) pipelines.

---

## 📖 Dataset documentation

| Dataset | English | Chinese |
|--------|---------|---------|
| VoiceBank-DEMAND | [README.md](voicebank_demand/README.md) | [README_CN.md](voicebank_demand/README_CN.md) |
| DNS Challenge 3 | [README.md](dns_challenge3/README.md) | [README_CN.md](dns_challenge3/README_CN.md) |
| URGENT2026 | [README.md](urgent2026/README.md) | [README_CN.md](urgent2026/README_CN.md) |
| PASE / StuPASE | [README.md](urgent2026/pase/README.md) | — |
| UniPASE | [README.md](urgent2026/pase/unipase/README.md) | — |
| GAP-URGENet | [README.md](urgent2026/pase/gap_urgenet/README.md) | — |

### Data SCP index

- **VoiceBank offline:** the four supervised manifests are `voicebank_{clean,noisy}_{train,val}.scp`; see [VoiceBank-DEMAND data preparation](voicebank_demand/README.md#step-2-generate-the-exact-scp-files).
- **URGENT offline / dynamic:** offline training and validation use paired `urgent26_{clean,noisy}_{train,val}.scp`; dynamic training and validation use `urgent26_clean_{train,val}.scp` as `speech_scp`, plus separate noise/RIR manifests. See [URGENT2026 data preparation](urgent2026/README.md#-data-preparation).
- **DNS synthetic protocol:** `generate_dns3_offline_dataset` materializes paired train/val/test waveforms and strict four-column `clean.scp` / `noisy.scp` manifests for paper-protocol reproduction. Official DNS V2/V3/V5 challenge sets remain noisy-only evaluation inputs and do not enter supervised training or Trainer validation. See [DNS Challenge 3](dns_challenge3/README.md).
- **URGENT nonblind：** clean/noisy SCP 仅用于评估。AEC `experiment_report` 默认使用该 nonblind 成对清单，并将 `--urgent-noisy-scp` / `--urgent-clean-scp` 传给 AEC pipeline，以 zero-farend 方式评估；目录参数仍不适用于嵌套的 nonblind FLAC 布局。参见 [URGENT2026 nonblind 说明](urgent2026/README.md#offline-paired-scps)。

---

<a id="inheritance-semantics"></a>
## Standalone configuration vs. validation simulation inheritance

The remaining inheritance is a runtime data policy inside one complete configuration:

- `data.validation.inherit_simulation` is a **dynamic-validation build policy inside the already merged configuration**. With the default `true`, validation inherits train simulation parameters and may override only data-source keys such as `speech_scp`, `noise_scp`, `rir_scp`, and `windnoise_scp`. With `false`, validation may override simulation parameters; omitted fields still come from train, and nested `simulation_cfg` dictionaries are recursively merged.
- `inherit_simulation` does not load another YAML. Offline validation builds directly from `clean_scp` / `noisy_scp` and does not use dynamic simulation inheritance.
- Dynamic validation strips balanced-sampling controllers inherited from train; an explicitly enabled validation `sampling` controller is rejected as a capability error.

---

<a id="balanced-sampling"></a>
## ⚖️ Dynamic balanced sampling (optional)

SE/AEC dynamic training enables balanced sampling through `data.train.sampling`. Speech and noise controllers are independent; each controller is opt-in through the presence of its corresponding subsection. The examples below retain YAML comments in Chinese to match the repository's configuration-comment convention:

```yaml
# language-only：只控制语言比例（全局 exact margin）
sampling:
  speech:
    language:
      probs: {zh: 0.5, en: 0.4, other: 0.1}

# dataset-only：全局 dataset 池；alpha/max_ratio 省略时分别取 0.5/0.3
sampling:
  speech:
    dataset:
      alpha: 0.5
      max_ratio: 0.3

# 联合：language 与全局 dataset 两组 exact margin 在
# (language, dataset_id) transportation 上联合精确满足
sampling:
  speech:
    language:
      probs: {zh: 0.5, en: 0.4, other: 0.1}
    dataset:
      alpha: 0.5
      max_ratio: 0.3

# noise 类别均衡（与 speech 独立 opt-in）
sampling:
  noise:
    balance_categories: true
```

- `language.probs` 只接受 `zh`、`en` 和 `other`，正概率之和必须为 1；每个候选 `actual` 都重新执行 Hamilton 取整。
- dataset 是全局 exact margin：同名 `dataset_id` 跨语言、跨采样率合并；权重来自过滤后的有效时长；单 dataset cap 为 `min(dataset_capacity, ceil(max_ratio * actual))`。
- speech strict JSONL 为 eligible 音频提供 `language` 和 `dataset_id`。manifest 在构造期读取，epoch 内 UID 无放回，DDP 不通过 padding 复制 balanced UID。
- `speech.speaker` 只用于 TSE；SE/AEC 配置该子块时报告 capability error。
- `noise.balance_categories` 只影响 ordinary additive noise，并与 additive `reuse_noise` 互斥；wind 和 wind reuse 不变。SE/AEC/TSE 均支持 Category / Temperature controller。
- selector 派生缓存采用双预算 LRU（16 entries / 65,536 UID references）；超大 Natural entry 使用零复制 view。缓存不是 UID hard cap，也不改变 RNG。
- **噪声采样与角色路由的完整合同**（Natural / Category / Temperature 的 SR tier、双噪声异类规则、`roles` 推导、Wind/Music 分流、发布侧 strict JSONL v6 与 admission policy v4 / AudioTag evidence v5 / noise source parser v10 / generator `6.1` / promotion marker v4）统一维护在 [仿真流程逻辑 §0.1.1](../../docs/guides/simulation/Simulation_Flow_Logic.md)，配置侧对照表见 [configs/README.md 六种训练噪声策略对比](../README.md#六种训练噪声策略对比)。配置入口速记：`sampling.noise.balance_categories` 打开 Controller，`balance_mode: category|temperature` 选择类间等权或 Temperature，`alpha` 默认 `0.5` 且仅在 Temperature 下生效；旧键 `prob_wind_noise_additive` 与 `allow_unverified_noise_manifest` 已删除，遗留配置会 fail-fast。
- **TSE 音乐来源**：Music 只在 TSE 从 ordinary pool 路由到 music pool；SE/AEC 仍将 Music 与乐器类视为 ordinary additive noise。`music_scp` 可省略并与 metadata 中的 `music` role 合并；`music_ratio > 0` 但合并后的 music pool 为空时报错。

`tools/prepare_data_tools/generators/noise/build_stage_d_noise_manifests.py` 的 split 模式发布 train/val
JSONL 与 `noise_publication_metadata.json` 三件套；`category_id`、`category_name` 与 `roles`
已内嵌每条 JSONL，不再生成重复的 `noise_categories.json` / `noise_category_roles.json`
sidecar。metadata 用于构建 provenance、内容完整性与顶层 `manifest_summary` 审计，不是训练
输入；摘要不代表 runtime 均衡。训练实际 category/temperature 分布使用
`tools/verify/audit_noise_sampling.py` 审计，README 不再硬编码数量快照。

- AEC speech 只平衡 near-end `speech_scp` anchor；far-end、echo、wind 和 RIR 不受影响。
- validation 构建时主动剥离继承的动态均衡 controller；validation 显式配置有效 controller
  时报告 capability error。

---

## 🔗 Quick links

| Goal | Directory |
|------|-----------|
| Reproduce the GTCRN VoiceBank-DEMAND baseline | [voicebank_demand/](voicebank_demand/) |
| Reproduce the GTCRN DNS3 protocol or run noisy-only challenge inference | [dns_challenge3/](dns_challenge3/) |
| Train an URGENT2026 baseline | [urgent2026/](urgent2026/) |
| Train a PASE / StuPASE generative enhancer | [urgent2026/pase/](urgent2026/pase/) |
| Train a MetricGAN-family enhancer | [urgent2026/gan/](urgent2026/gan/) |
| Train a multichannel / moving-speaker (NBSS) recipe | [spatial/](spatial/) |

---

<p align="center">
  <sub>Last updated: 2026-09-21</sub>
</p>
