# Data Preparation Tools

`tools/prepare_data_tools/` turns raw audio directories into the indexes the training datasets read, and generates simulated assets (RIRs, wind noise, echo pairs, offline clean/noisy pairs). It contains no training or inference code.

- `scp/` — index (SCP/JSONL) producers: `prepare_scp.py`, `prepare_offline.py`, `prepare_voicebank.py`, `prepare_testset_scps.py`, `prepare_aec_offline.py`, `prepare_aec_challenge_offline.py`.
- `generators/` — simulated-data producers grouped by asset kind: `aec/`, `se/`, `tse/`, `ss/`, `noise/`, `echo/`.
- `inspect_manifest.py` — read-only inspector for any manifest a producer emits.
- `generation_manifest.py`, `offline_publication.py`, `offline_reload_gate.py`, `manifest_contracts.py`, `se_index_publication.py` — shared publication/reload contracts used by the producers.
- `_common.py` — internal helper module; its API is **not stable** (see [Canonical Entry Points](#canonical-entry-points)).

> **Note:** Generators are grouped by asset kind under `tools/prepare_data_tools/generators/`: `aec/`, `se/`, `tse/`, `ss/`, `noise/`, and `echo/`. The RIR generator lives under `generators/se/`.

## Glossary

- **SCP** — a plain-text index, one record per line. The strict form used here is four columns: `uid fs length path` (`uid` = unique id, `fs` = sample rate, `length` = frame count, `path` = audio path).
- **manifest** — the general word for any index file. Serialization (SCP vs JSONL vs JSON) and capability (what a dataset can do with the records) are independent; see [Serialization and Capability Are Orthogonal](#serialization-and-capability-are-orthogonal).
- **RIR** — Room Impulse Response: how a room colors and reverberates sound. Simulation convolves clean speech with an RIR to create reverberant training data.
- **GenerationManifest v2** — the publication contract for a generated offline dataset: which generator produced it, the content SHA-256 of every resource, and how to reload it. Schema v1 collections must be regenerated.
- **VAD** — Voice Activity Detection (used by the Noise Filter pipeline, not by this directory).
- **fail-closed** — stop with an error instead of guessing. Most checks here are fail-closed, so an error usually means a check did its job.

## How to read this page

1. New to the directory? Start with [Quick start](#quick-start).
2. Generating simulated assets (RIR / wind noise / echo): read the generator section you need.
3. Choosing which index format your training config should consume: read [SCP Index & Manifest Scripts](#scp-index--manifest-scripts) — the consumer → producer → inspector flow.
4. Auditing a publication or migrating old artifacts: read [Important Producers and Their Boundaries](#important-producers-and-their-boundaries).

## Quick start

```bash
# 1. Index one audio root as a single strict four-column SCP (single-directory mode)
#    -o is an output file prefix without ".scp": writes speech_sources.scp
python tools/prepare_data_tools/scp/prepare_scp.py /data/speech \
    -o data/train_sources/speech_sources

#    Batch mode instead splits every resource 95:5 and writes two files
#    (train_speech_sources.scp + val_speech_sources.scp); --val_ratio 0 disables the split
python tools/prepare_data_tools/scp/prepare_scp.py --speech /data/speech \
    -o data/train_sources

# 2. Check what you just wrote (read-only; never opens the referenced audio)
python tools/prepare_data_tools/inspect_manifest.py \
    data/train_sources/train_speech_sources.scp --profile dynamic-source

# 3. Generate simulated RIRs with the default gpuRIR recipe
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --output-dir /data/rir/gpurir_se_16k
```

Step 2 prints `serialization_format`, `record_count`, `sample_rates` and a `yaml_hint` telling you which training config key to use. Step 3 refuses a non-empty output directory and writes `rirs.scp` + `rirs_meta.json` plus the RIR files; the training config then points `data.train.rir_scp` at that `rirs.scp`.

---

# RIR Simulation Data Preparation

This section documents RIR asset generation only. Training-time Layer A–F semantics and the distinction between Dynamic Dataset simulation and AEC model-input preprocessing are defined in [`docs/guides/simulation/Simulation_Flow_Logic.md`](../../docs/guides/simulation/Simulation_Flow_Logic.md).

## Files

- `generators/se/generate_simulated_rirs.py` — unified CLI entry point.
- `generators/se/rir_backends/` — implementation of the three RIR backends.

## Supported Backends

| Backend | Method | Dependency | Hardware |
|---------|--------|------------|----------|
| `rirgen` | image-source (`rir_generator`) | `pip install rir_generator` | CPU |
| `fram`   | Fast Random Approximation (FRAM-RIR) | PyTorch + torchaudio | CPU/GPU |
| `gpurir` | GPU-accelerated image-source (`gpuRIR`) | `pip install gpuRIR` | GPU |

All backends are optional. The script detects available ones at runtime and reports missing dependencies clearly; gpuRIR additionally requires at least one usable GPU.

## RIR Generation Defaults

`generators/se/generate_simulated_rirs.py` is the canonical RIR generator. The output directory is completely user-selectable, including its name; for example, `--output-dir /data/rir/gpurir_se_16k` writes the complete dataset and manifests below that directory. There is no separate naming option because the output directory itself is the dataset name. The target must be new or empty; a non-empty directory is rejected to prevent mixing assets from different manifests.

Canonical generator profiles and their field documentation live beside the generator code under
`generators/se/`: `rir_gpurir.yaml`, `rir_fram.yaml`, and `rir_rirgen.yaml` are the three
backend profiles. Fields use unit-bearing
`snake_case` names; see `generators/se/README.md`. These files are generator runtime profiles,
not training/inference configs, and do not belong under the repository-level `configs/` tree.

The canonical profile files define the complete generation defaults. Use `rir_gpurir.yaml`
for gpuRIR, `rir_fram.yaml` for FRAM-RIR, and `rir_rirgen.yaml` for rir_generator. Stress
runs override RT60, room, distance, and motion fields in the same backend YAML. All generation
parameters use unit-bearing `snake_case` names.
The CLI accepts `--config` plus canonical override options such as `--method`,
`--sample-rate-hz`, and `--output-dir`; old option aliases are not supported.

Selecting `--method fram` or `--method rirgen` generates static RIRs only. `--method all`
runs every runnable backend as static generation; when gpuRIR is installed but automatic
detection finds no GPU, gpuRIR is skipped and the CPU backends continue.

## Output Naming

Each generated RIR file and SCP `uid` encodes the key metadata:

```
{method}_fs{fs}_{field_or_field_moving}_rt{rt60:.2f}_dist{dist:.2f}_room{L:.1f}x{W:.1f}x{H:.1f}_{idx:08d}.{ext}
```

For example:

```
fram_fs16000_near_rt0.50_dist0.32_room4.5x5.2x3.1_00000123.wav
gpurir_fs16000_far_moving_rt0.80_dist1.20_room6.0x5.0x3.0_00000123.pkl
```

The profile's `output.format` defaults to `.pkl`; **PKL is recommended for production datasets**. Static RIRs may also use FLOAT WAV or 24-bit FLAC. PKL and FLOAT WAV preserve float32 samples outside `[-1, 1]`; integer-PCM FLAC is written only when every sample is within `[-1, 1]` and otherwise fails instead of clipping silently. Moving RIRs must use `.pkl` because a multi-segment sequence cannot be represented as mono audio.

## Output Directory Layout

RIR files are organized by simulation method, sample rate, static/moving mode, near/far field, and shard:

```
out_dir/
├── rirs.scp
├── rirs_meta.json
└── {method}/
    └── fs{fs}/
        ├── static/
        │   ├── near/
        │   │   ├── shard_0000/
        │   │   └── ...
        │   └── far/
        │       ├── shard_0000/
        │       └── ...
        └── moving/                 # gpuRIR only
            ├── near/
            │   ├── shard_0000/
            │   └── ...
            └── far/
                ├── shard_0000/
                └── ...
```

A mode or field with zero requested samples is not created. The output directory must be new or empty; generation rejects a non-empty directory rather than mixing assets from different runs. `--shards` controls the number of shard directories for each `(method, fs, mode, field)` combination. With multiple `--sample-rate-hz` values, each rate gets its own `fs{fs}/` tree and an additional `rirs_fs{fs}.scp`. The `rirs.scp` path is relative to the manifest directory and always has four columns: `uid fs num_samples path`. `rirs_meta.json` records the shared RT60 range, ratio weights, requested/valid counts, effective formats, and an output inventory with SHA-256 identities. Per-file `.json` metadata is additionally written when `--write-metadata-json` is set.

## Usage Examples

### Default gpuRIR recipe: 100,000 static RIRs

```bash
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --output-dir /data/rir/gpurir_se_16k
```

This requests 50,000 near-field and 50,000 far-field RIRs. The default uses `static_moving_ratio=[1,0]` and `motion.enabled=false`, so it writes static PKL only. `--gpus` may be omitted to auto-detect available GPUs.

### Explicit static-only FRAM-RIR generation

```bash
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_fram.yaml \
    --output-dir /data/rir/sim_rirs_fram_8k \
    --field near --total-rirs 10000 --validate
```

An explicit `fram` or `rirgen` selection never creates moving RIRs. `--method all` is also explicit: it runs every available backend as static RIR generation, so the requested count is produced once per available backend.

### Custom ratios and shared RT60 range

```bash
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --output-dir /data/rir/gpurir_custom --total-rirs 20000 --gpus 0 1
```

### Moving-source profile

The source moves while the microphone remains fixed. The trajectory advances through `n_segments` time steps of `segment_sec` seconds each; each position produces a static RIR, yielding a piecewise time-varying `[T_seg, n_samples]` sequence. **Only gpuRIR supports this mode**; FRAM/RIRGenerator reject moving configurations. The output is PKL because a multi-segment sequence cannot be represented as mono WAV:

```bash
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --output-dir /data/rir/gpurir_moving --motion-enabled --gpus 0 1 2 3
```

Moving pkl payloads contain `{"rir": [T_seg, n_samples] float32, "fs": int, "moving": true, "n_segments": int, "segment_sec": float, "trajectory": [T_seg, 3], "speed": float, "rt60": float, "dist_min": float, "dist_max": float, ...}`. `dist` is the start-point distance; `dist_min`/`dist_max` are the closest/farthest source-to-mic distances across the whole trajectory. The `moving: true` marker is consumed automatically by the dynamic dataset.

## RT60, Absorption, and RIR Length

RT60 is a room property, not an intrinsic near/far label. Near/far differences should primarily come from source-microphone distance, direct-path level, and the resulting direct-to-reverberant ratio (DRR). Using one RT60 range for both fields therefore avoids coupling the room acoustics to the distance class.

The default RT60 range is `0.1–1.0 s`, shared by near and far fields. Wider long-reverberation stress coverage should be configured explicitly in the selected backend YAML.

The script does not use one fixed reverberation coefficient. For each sampled room and RT60, `rir_backends/` derives the Sabine mean absorption coefficient approximately as:

```
alpha = 24 * ln(10) * V / (c * S * RT60)
```

Samples are retried until `alpha < 1`. The gpuRIR backend then converts the target RT60 and room geometry into wall reflection coefficients (`beta`) through `beta_SabineEstimation`; `beta` is therefore also not a single global constant. With the default room ranges (`L,W=3–10 m`, `H=2.5–4.5 m`, `c=340 m/s`) and RT60 range, the corner-case Sabine `alpha` is approximately `0.059–0.962`, so the default recipe stays below the validity boundary. The coefficient is derived per sampled room and is not an independent CLI parameter.

When `near_rir_length_samples` / `far_rir_length_samples` are omitted, the generator computes a 256-sample-aligned length from the upper RT60 bound, the maximum configured propagation distance, and a 50 ms safety margin:

```
n_samples = ceil(fs * (rt60_hi + dist_hi / c + 0.05) / 256) * 256
```

For the default ranges, 16 kHz uses 16,896 near-field samples and 17,152 far-field samples; 32 kHz uses 33,792 and 34,048 samples. A shorter explicit length emits a warning and can truncate the late reverberation tail.

## Reproducibility and Validation Gate

Each sample's scene seed is a stable BLAKE2b digest of the base `--seed`, backend, field/mode domain, and sample index. It never depends on the GPU assignment, worker count, or task scheduling, so two runs with identical CLI arguments sample the same rooms, source/microphone positions, RT60 targets, and moving trajectories even when executed on a different GPU subset or with a different number of workers. FRAM backend randomness is isolated per task: the global NumPy/Torch states are restored after each sample, so FRAM output is fully reproducible per task.

One caveat applies to gpuRIR only: its diffuse late-reverberation tail is generated with a process-local pseudo-random state inside the library (not seeded by this script). Runs with an identical GPU layout are sample-identical; runs with a different GPU count may produce slightly different late-tail samples (same scenes and early part, difference in the ~1e-3 range). This is inherent to the gpuRIR library and cannot be controlled from the CLI.

With `--validate`, generated RIRs are checked **before** any manifest is written: actual sample rate and frame count must match the SCP declaration, all samples must be finite, and every static RIR or moving-RIR segment must exceed the configured peak threshold. If any abnormal sample is found, the run exits non-zero and does not publish `rirs.scp` / `rirs_meta.json`; already-written data files remain on disk, so the output directory must be emptied before retrying.

## Generated Files

After running, the output directory contains:

- `rirs.scp` — a four-column `uid fs num_samples path` manifest, directly usable as `data.train.rir_scp`; paths are relative to the manifest directory.
- `rirs_meta.json` — generator identity, normalized parameters, shared RT60 and ratio weights, requested/valid counts, effective formats, and an output inventory. Per-file SHA-256 fingerprints are skipped by default (`sha256: null`); pass `--inventory` to compute them (slow on very large datasets) and `--no-inventory` to state the default explicitly. `compute_sha256` records which behavior was used.
- `.wav`, `.flac` (static, explicit), or `.pkl` RIR files under `{method}/fs{fs}/{static|moving}/{near|far}/shard_{nnnn}/` (default output is PKL for both static and moving).
- With multiple `--sample-rate-hz` values, `rirs.scp` merges every sample rate, and each rate additionally gets its own `rirs_fs{fs}.scp` (same directory, same relative paths).
- Optional per-file `.json` metadata when `--write-metadata-json` is set.

## Common CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | Generator profile YAML beside the RIR code |
| `--output-dir` | optional | Overrides YAML `output_dir`; directory must be new or empty |
| `--method` | profile | `rirgen`, `fram`, `gpurir`, or `all` |
| `--sample-rate-hz` | profile | Target sample rate(s); values must be unique |
| `--field` | `both` | `near`, `far`, or `both` |
| `--total-rirs` | profile | Total RIR count; `both` uses `--near-far-ratio` |
| `--static-moving-ratio` | `1:0` | gpuRIR static:moving weights; static-only by default |
| `--near-far-ratio` | `5:5` | Near:far weights when deriving counts from the total |
| `--near-rirs` / `--far-rirs` | profile | Direct field counts; omitted fields become zero when one is explicit |
| `--near-rir-length-samples` / `--far-rir-length-samples` | auto | 256-aligned RIR lengths |
| `--rt60-range-sec` | profile | Shared near/far RT60 range, seconds |
| `--near-distance-range-m` / `--far-distance-range-m` | profile | Source-microphone distance ranges, meters |
| `--room-length-range-m` / `--room-width-range-m` / `--room-height-range-m` | profile | Room dimension ranges, meters |
| `--workers` | `16` | Number of CPU workers for FRAM/RIRGenerator |
| `--shards` | `10` | Number of shards per `(method, fs, mode, field)` combination |
| `--gpus` | auto-detect (all GPUs) | Tokens for `gpurir`: when `CUDA_VISIBLE_DEVICES` is unset the default auto-detects every GPU (8 on this machine) and bare integers are physical GPU IDs; once CVD is set, bare integers become visible slots. `slot:N` explicitly selects a visible slot, `id:N` explicitly selects a physical GPU; UUID and MIG tokens are matched literally |
| `--format` | `auto` | `auto` writes every RIR as PKL and is recommended for production; FLOAT WAV preserves values outside `[-1,1]`, while FLAC accepts only static RIRs within `[-1,1]`; moving work requires PKL |
| `--write-metadata-json` | off | Write per-file metadata JSON |
| `--validate` | off | Strict publication gate: checks sample rate, frame count, finite samples, and per-segment peaks before writing manifests; failures prevent `rirs.scp` / `rirs_meta.json` publication |
| `--inventory` / `--no-inventory` | off (default) | Compute per-file SHA-256 fingerprints in `rirs_meta.json`; skipped by default (`--no-inventory` states the default explicitly) |
| `--motion-enabled` | off | Enable moving-source RIR generation; only gpuRIR supports it |
| `--n-segments` / `--n-segments-range` | `[3,6]` | Fixed count or per-scene inclusive count range |
| `--segment-duration-sec` | `0.2` | Duration of each moving segment, seconds |
| `--speed-range-mps` | profile | Source speed range, m/s |
| `--trajectory-type` | profile | `linear`, `waypoint`, or `both` |

## Training Config Usage

Point `data.train.rir_scp` in your YAML config to the generated `rirs.scp`:

```yaml
data:
  mode: dynamic
  train:
    rir_scp: /data/rir/sim_rirs/rirs.scp
    prob_reverberation: 0.5
```

### Consuming moving-source RIRs

Consumption is **automatic**. When the training-side `_load_audio_impl` sees `moving: true`
in a pkl payload it keeps the `[T_seg, n]` 2D structure, and `_mix_acoustic` then calls
`add_moving_reverberation`: the speech is chunked by `segment_sec`, each chunk is convolved
with its own segment RIR, and the full convolution tails are overlap-added on the absolute
time axis (each segment keeps its full `n_h - 1` tail, so there is no inter-segment click and
no lost late reverberation). `estimate_early_rir` extracts the early part row by row per
segment, so the `clean_target` definition holds automatically. The segment duration comes from
the pkl payload's own `segment_sec` when present, and otherwise falls back to
`SimulationConfigs.segment_sec` (default `0.5` s). Static and moving RIRs may be mixed in the
same `rir_scp`. Example:

```yaml
data:
  mode: dynamic
  train:
    rir_scp: data/shared/rirs/16k_moving/train_rirs.scp
    simulation_cfg:
      segment_sec: 0.5
```

## Testing

Run the tests with pytest:

```bash
python -m pytest tests/test_generate_simulated_rirs.py -v
```

---

# Wind Noise Generation

`generators/noise/generate_wind_noise.py` batch-generates wind noise with the single-channel algorithm of Mirabilii & Habets (IWAENC 2022), producing a 4-column `wind_noise_sources.scp` that the training framework can load directly.

## Files

- `generators/noise/generate_wind_noise.py` — CLI entry point (parallel workers, sharding, 4-column SCP).
- `generators/noise/_wind_noise_backend.py` — vendored `WindNoiseGenerator` backend (MIT License, copyright header preserved in the file). It uses the equivalent in-repo `lsf2poly` implementation and has no demo-only `matplotlib`/`sounddevice` dependencies, so **no extra pip packages are required**.

## Usage Examples

```bash
# 16 kHz, 200 files, 15 s each (URGENT2026 train recipe)
python -m tools.prepare_data_tools.generators.noise.generate_wind_noise \
    --output-dir /data/noise/wind_noise_16k \
    --fs 16000 --num_data 200 --duration 15 --workers 16

# Multiple sample rates
python -m tools.prepare_data_tools.generators.noise.generate_wind_noise \
    --output-dir /data/noise/wind_noise_multi \
    --fs 8000 16000 24000 48000 --num_data 200
```

## Common CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | required | Output directory (refuses to write into an existing non-empty directory) |
| `--fs` | `16000` | Target sample rate(s), multi-value |
| `--num_data` | `200` | Number of files per sample rate |
| `--duration` | `15.0` | Duration per file (seconds) |
| `--gustiness_range` | `3,10` | Gustiness range (1 = constant wind, higher = gustier) |
| `--seed` | `42` | Random seed (per-file seed = seed + idx, fully reproducible) |
| `--workers` | `16` | Number of parallel worker processes |
| `--shards` | `16` | Number of shards per sample rate |
| `--overwrite` | off | Allow writing into an existing non-empty output directory |

> Reproducibility note: each file's seed is `seed + idx` where `idx` increments globally across sample rates. Two runs are bit-identical **only when the CLI arguments (including the `--fs` list and `--num_data`) are unchanged**; adding/removing a sample rate shifts the seeds of existing files.

## Output Directory Layout

```
out_dir/
├── wind_noise_sources.scp     # 4 columns: uid fs num_samples abs_path
├── wind_noise_meta.json       # per-sample-rate count / gustiness statistics
└── wind_noise/
    └── fs{fs}/
        └── shard_{nnnn}/
            └── wind_fs{fs}_gust{g:.1f}_{idx:08d}.wav
```

## Training Config Usage

Point `data.train.windnoise_scp` in your YAML config to the generated `wind_noise_sources.scp`:

```yaml
data:
  mode: dynamic
  train:
    windnoise_scp: /data/noise/wind_noise_16k/wind_noise_sources.scp
```

## Testing

```bash
python -m pytest tests/test_generate_wind_noise.py -v
```

---

# MC-LibriMix 多通道数据准备

`generators/tse/mc_librimix/` 将已有 Libri2Mix 双源音频转换为多通道 early/reverb WAV、空间 cue 和 WeSep 样本索引。工具只使用本仓库包内模块，不依赖外部 WeSep checkout、`WESEP_ROOT` 或其 shell 工作目录。**本次仅迁入数据生成工具，不扩展 native TSE 的 spatial cue 训练解码。**

来源为 [WeSep `examples/spatial/mc_librimix/local/`](https://github.com/wenet-e2e/wesep/tree/8c2fc2178db3659f3c6cb3638b410c4caa0f7985/examples/spatial/mc_librimix/local)，commit `8c2fc2178db3659f3c6cb3638b410c4caa0f7985`；四个 Python 工具保留对应职责，`simulate_data.sh` / `prepare_data.sh` 改为 Python 编排。迁入文件使用 [Apache-2.0](generators/tse/mc_librimix/LICENSE)，保留来源与修改声明；主仓库 MIT License 不替代此许可。

## 本机数据目录

`data/tse/spk/wesep/` 按用途分为两类目录：

- `sources/`：源语音 `LibriSpeech/`、`VoxCeleb1/`、`VoxCeleb2/`，保留 VoxCeleb 内部的 `flac_audio/`。
- `datasets/`：已准备的 `Libri2Mix/`（保留 `wav16k/min/`）和 `LibriMix-keywords/`（保留内部相对软链接）。

MC-LibriMix 暂未生成，也未创建占位目录；以下 MC 输出与独立索引路径仅供未来显式生成时使用。不保留旧数据目录软链接，原有训练 baseline 配置不随本次目录整理修改。

## 正式入口与参数

在项目根目录运行 `python -m`；从其他目录调用时须保证本项目包可被 Python 导入。下面的 `MC_CONFIG` 必须预先设为调用者提供的真实 YAML 路径；**当前未提供官方 `respeaker_4mic_16k.yaml`，没有可直接代用的 canonical 参数**。示例只处理 `test`，不表示完整 MC-LibriMix 已生成。

```bash
python -m tools.prepare_data_tools.generators.tse.mc_librimix.simulate_data \
    --config "${MC_CONFIG:?请先设置显式场景配置路径}" \
    --librimix-root data/tse/spk/wesep/datasets/Libri2Mix/wav16k/min \
    --output-root data/tse/spk/wesep/datasets/MC-LibriMix \
    --split test

python -m tools.prepare_data_tools.generators.tse.mc_librimix.prepare_data \
    --dataset-root data/tse/spk/wesep/datasets/MC-LibriMix \
    --data data/tse/spk/wesep/datasets/MC-LibriMix-index \
    --mix-condition reverb --target-condition early --splits test
```

`--librimix-root` 直接包含 `<split>/s1` 与 `<split>/s2`，两路同名 WAV 必须是帧数一致、采样率等于 `audio.sample_rate` 的非空单声道音频；文件名为两个不同说话人的 `<speaker>-<chapter>-<utterance>_<speaker>-<chapter>-<utterance>.wav`。已有 Libri2Mix `mix_clean` 不作为输入。

六个模块均提供 `main(argv=None)` 和 `--help`。下表模块名统一加前缀 `tools.prepare_data_tools.generators.tse.mc_librimix.`，用 `python -m` 调用；列出的参数除注明默认值者外均必填。

| 模块 | 参数与用途 |
|---|---|
| `simulate_data` | `--config`、`--librimix-root`、`--output-root`；`--split` 默认 `all`，编排场景、WAV、cue 与生成摘要 |
| `prepare_data` | `--dataset-root`、`--data`；`--mix-condition` 默认 `reverb`、`--target-condition` 默认 `early`，两者均只接受 `early/reverb`；`--splits` 接受一个或多个 split，默认 `train-100 dev test`，不接受 `all` |
| `simulation.create_scenarios` | `--config`、`--librimix-root`、`--output`；`--split` 默认 `all`，输出 CSV，文件名以 `.gz` 结尾时压缩 |
| `simulation.simulate_librimix` | `--config`、`--librimix-root`、`--scenario-manifest`、`--output-root`；`--split` 默认 `all`，按场景生成 WAV |
| `build_spatial_cues` | `--scenario-manifest`、`--output-root`；`--split` 默认 `all`，从 CSV/CSV.gz 导出 NPZ 与 `spatial.json` |
| `scan_librimix` | `--mix-dir`、`--s1-dir`、`--s2-dir`、`--outfile`；校验三路 ID、帧数、采样率和通道数后写 WeSep JSONL |

合法 split 只有 `train-100/dev/test`；仿真的 `all` 仅选择配置 `dataset.splits` 中列出的 split，单独指定的 split 也必须已配置。`prepare_data --data` 必须位于 `--dataset-root` 之外。两个编排入口拒绝非空输出目录和输出符号链接，在同文件系统相邻 staging 中完成校验后发布，失败清理本次 staging；没有覆盖、跳过已有文件或 resume 模式。下层独立 CLI 不等同于整条流水线的事务发布，完整生成优先使用 `simulate_data`。

场景生成需要 NumPy/PyYAML，波形生成需要 NumPy/SciPy/soundfile，实际 RIR 仿真还需要已有的 gpuRIR 与可用 GPU。gpuRIR 仅在执行波形生成时导入，缺失不影响模块 import 和六个 CLI 的 `--help`。

## 显式 YAML 配置

顶层必须精确包含下表八项，各对象也只接受列出的字段；缺失键、未知键、非法区间或无法容纳阵列/声源的几何会报错。所有区间写成 `{min: ..., max: ...}`，边界必须有限且 `min <= max`；几何长度单位为米。

| 顶层项 | 必填字段与约束 |
|---|---|
| `version` | 非空字符串，参与逐场景种子派生；不是工具版本号 |
| `dataset` | `seed` 为非负整数；`splits` 为非空、无重复的 `train-100/dev/test` 列表；`max_sampling_attempts` 为正整数 |
| `room` | `dimensions.min/max` 各为三个正数；`rt60.min/max` 为正数，单位秒；`boundary_margin` 为非负数 |
| `array` | `mic_positions` 为互不重复的局部坐标 `[N,3]`；`height` 为正数区间；`rotation_deg` 为角度区间；阵列与边界余量必须能放入房间 |
| `sources` | `num_speakers` 必须为 `2`；`azimuth_deg` 区间跨度不超过 360°；`horizontal_distance`、`height` 为正数区间；`min_azimuth_difference_deg` 非负且不超过 180° 与方位区间跨度 |
| `audio` | `sample_rate` 为正整数，单位 Hz；必须与实际源音频一致，不自动重采样 |
| `rir` | `early_window_ms` 为正数且至少覆盖一个采样点；`preserve_time_of_flight` 为布尔值，控制 early 截断窗是否计入传播时间；`sound_speed` 为计算该传播时间的正声速值，单位 m/s |
| `outputs` | `early_source`、`early_mixture`、`reverb_source`、`reverb_mixture` 均为布尔值，至少开启一项 |

若需要全部六路 WAV，将四个 `outputs` 开关显式设为 `true`；索引所选 condition 对应的 mixture/source 输出必须存在。场景种子由 `version/seed/split/mixture_id` 经 SHA-256 稳定派生，场景 CSV.gz 使用固定 gzip 时间戳；这不承诺 gpuRIR 波形跨环境逐位相同。测试用小型场景配置不能替代官方 profile，也不能据此宣称全量数据准备完成。

## 输出与消费边界

`simulate_data --output-root` 的布局如下；WAV 目录按 `outputs` 开关创建，通道数由 `array.mic_positions` 决定，采样率由配置决定，长度保持对应源音频帧数。

```text
MC-LibriMix/
├── metadata/
│   ├── scenarios_v1.csv.gz
│   └── generation_summary.json
└── <split>/
    ├── mix_early/<mixture_id>.wav
    ├── mix_reverb/<mixture_id>.wav
    ├── s1_early/<mixture_id>.wav
    ├── s1_reverb/<mixture_id>.wav
    ├── s2_early/<mixture_id>.wav
    ├── s2_reverb/<mixture_id>.wav
    └── cues/
        ├── spatial.json
        └── spatial_npz/<mixture_id>__<speaker_id>.npz
```

WAV 使用 `PCM_16`；同一 mixture 已启用的所有 early/reverb 源和混合共用一个峰值缩放因子，混合为两源相加后再量化。`generation_summary.json` 记录工具版本、上游来源、完整配置及其 SHA-256、seed 和输出计数，**不是共享 GenerationManifest v2 发布合同**。

每条 mixture 有两个 NPZ cue，字段精确为 `azimuth/elevation/distance`，均为有限的 `float32` 标量；角度为弧度，距离为米。可用 `numpy.load(path, allow_pickle=False)` 读取，不使用 `np.save(dict)` 的 pickle 对象。`spatial.json` 以 `<mixture_id>::<speaker_id>` 为键，payload 路径相对于该 JSON 所在目录。

`prepare_data --data` 为每个所选 split 写 `samples.jsonl`、`raw.list`、`cues.yaml`，不复制 WAV 或 NPZ：

- `samples.jsonl` 保留 WeSep 的 `key/spk/mix/src` 记录，音频路径相对于该 JSONL 所在目录；这是样本索引，**不是 native dynamic TSE 的 `uid/fs/length/path/language/dataset_id/speaker_id` speech manifest**。
- `raw.list` 是内容为一行 `samples.jsonl` 的真实 path-list 文件，指向同目录下的样本索引；它本身不是 JSONL 内容，也不是指向 JSONL 的软链接。
- `cues.yaml` 的 spatial policy 为 `fixed/mix_spk_id`，`resource` 相对于 YAML 所在目录；保留上游 `type: npy`、`format: fields`，选择 `azimuth/elevation` 两字段，底层安全载荷实际为 NPZ，额外保存 `distance`。

索引及 cue 资源可用不等于 native spatial cue 训练解码已经接通；本工具不修改 Dataset、collate、模型或训练入口，也不生成 speaker enrollment。迁移资产时须保留索引目录与音频目录之间的相对布局。

---

# SCP Index & Manifest Scripts

This section separates how a file is serialized from which training capabilities its records provide. Do not choose a manifest by extension alone, and do not assume that outputs from different producers are interchangeable.

## Selection Flow: Consumer → Producer → Inspector

1. **Choose the consumer mode first**:
   - Dynamic SE/AEC/TSE modes read split-level dynamic sources such as `speech_scp`, `noise_scp`, `music_scp`, `rir_scp`, and `windnoise_scp` from `data.train` / `data.validation`; dynamic TSE additionally requires `spk2enroll_json`.
   - Offline SE clean/noisy tracks and offline AEC mic/farend/nearend/error tracks consume strict `uid fs length path` SCPs; no external sample-rate or length sidecar participates in runtime.
   - TSE `offline_tse` (wesep/REAL-TSE) consumes `raw.list + samples.jsonl + cues.yaml`; it **does not use the SE/AEC four-column offline SCP contract**, and `raw.list` is a path-list pointing to a sample manifest rather than a dynamic manifest. To migrate an official speaker-level `audio.json`, use `generators/tse/import_wesep_enrollment.py`: it adds the explicit `dataset_id` level and converts the resource into the dynamic-TSE nested `spk2enroll_json`; `mix_spk_id` fixed-trial resources must not be aggregated.
2. **Then choose the producer** that directly matches the Dataset contract. Two files are not interchangeable merely because both use JSON or JSONL.
3. **Finally validate read-only** with the inspector. It reports serialization, counts, sample rates, and capabilities without opening referenced audio.

```bash
python tools/prepare_data_tools/inspect_manifest.py MANIFEST \
    --profile {dynamic-source,dynamic-speech,categorized-noise,strict-noise-publication} [--json]
```

The default profile is `dynamic-source`. Output fields are `serialization_format`, `record_count`, `sample_rates`, `has_metadata`, `length_complete`, `category_complete`, `strict_fields_complete`, and `yaml_hint`. The hint follows the use: `dynamic-speech` emits `speech_scp` and noise profiles emit `noise_scp`. After inspecting an RIR or wind manifest, replace the key with `rir_scp` or `windnoise_scp`; do not copy it literally.

## Serialization and Capability Are Orthogonal

- **Serialization** describes a dynamic-source carrier: strict four-column SCP or JSONL.
- **Capability** describes the training contract: `dynamic-source`, `categorized-noise`, or `strict-noise-publication`. JSONL may carry metadata but **does not mean category balancing is enabled**. Strict schema v6 is a publication-audit contract and **is not required** for category balancing.

| Serialization | Record shape | Supported use |
|---------------|--------------|---------------|
| 4-column SCP | `uid fs length path` | Natural dynamic sampling; fixed `prepare_scp.py` output |
| JSONL | One `uid/fs/length/path` object per line, with optional metadata | Dynamic sampling; category/strict support depends on field completeness |

SCP and JSONL consumers (`core/data/noise_manifest_adapter.py`) resolve relative audio paths against the **manifest directory** (absolute paths and `~` remain unchanged), independent of the training cwd. SCP fields containing spaces must use shell quoting (`prepare_scp.py` does this automatically); every row must contain exactly `uid fs length path`, while legacy three-column rows and JSON arrays fail closed.

Three path rules are worth keeping apart:

- Inside a manifest, relative audio paths anchor to the manifest file's own directory, so moving a whole data directory does not require rewriting the paths inside it.
- In a training config, resource keys (`speech_scp`, `noise_scp`, `rir_scp`, ... — see `core/data/resource_paths.py`) anchor `data/...` values to the repository root and fail closed if the resolved path escapes it; other relative paths keep the caller's original semantics, and absolute paths keep working.
- Logical aliases such as `waveforms:` / `dataset:` in an already verified publication manifest are bound to a generation-root marker. They are not ordinary `../` relative paths and must be kept verbatim.

## Training Input Fields and Startup Cost

| Config input | Current record fields | Recommended production contract |
|--------------|-----------------------|---------------------------------|
| `speech_scp` | `uid/fs/length/path`; speech balancing also needs JSONL `language/dataset_id`, while the speaker controller or TSE strict speech also needs `speaker_id` | Materialize a positive integer `length` offline; use four-column `uid fs length path` or JSONL with complete metadata |
| `noise_scp` | Basic `uid/fs/length/path`; categorized JSONL adds `category_id`, `balance_category`/`category_name`, and `roles`; strict v6 also carries admission/verification fields | Keep basic fields complete; use categorized JSONL from `build_stage_d_noise_manifests.py` when balancing or wind/music pool routing is needed |
| `rir_scp` | `uid/fs/length/path`; path may refer to waveform or RIR pickle data | Use four-column SCP or JSONL; keep moving-RIR fields such as `moving/segment_sec/trajectory` in the pickle payload rather than duplicating them in SCP |
| `windnoise_scp`, TSE `music_scp` | The same `uid/fs/length/path` fields as a general dynamic source | Use strict four-column SCP; categorized `noise_scp` may also route canonical `roles` into these pools and merge them with explicit manifests |
| AEC `real_farend_echo_pair_json` | Canonical object `{"pairs": ["stereo.wav"], "meta": {...}}`; every pair is a non-empty stereo path string (ch0=echo, ch1=farend) | This is a separate pair contract and currently has no unified `uid/fs/length`; root arrays, pair lists/objects, dual-file rows, and SCP aliases are rejected |

`speech_scp` is the critical startup-performance gate. With `strict_load: true` (the default), a dynamic speech manifest must include complete `length` metadata and fails fast at Dataset construction instead of serially calling `soundfile.info()` before DataLoader workers start. Production training must use `prepare_scp.py` to materialize four-column metadata in parallel offline. `speech_length_file` is not part of the SE/AEC offline contract; only explicit `strict_load: false` keeps the training-time `sf.info()` probing fallback for legacy-data migration. Noise, wind, and RIR lengths are for summaries and lightweight metadata, not actual simulation cropping.

| Inspector profile | Required capability | Typical use |
|-------------------|---------------------|-------------|
| `dynamic-source` | Non-empty strict four-column SCP or JSONL dynamic source | Natural dynamic sampling; capability check |
| `dynamic-speech` | Non-empty, every record with positive integer `uid/fs/length/path` | `speech_scp` production publication gate, matching strict Dataset construction semantics |
| `categorized-noise` | Every ordinary-noise record has a non-negative `category_id` and a non-empty, trimmed `balance_category` (with `category_name` accepted for compatibility) | Category/temperature noise sampling |
| `strict-noise-publication` | Every record satisfies the current Noise Filter strict-publication field contract | Noise Filter Stage D publication audit; stricter than training category capability |

## Important Producers and Their Boundaries

| Script | Target scenario | Main output / boundary |
|--------|-----------------|------------------------|
| `scp/prepare_scp.py` | General speech/noise/RIR/wind/music scans | Fixed strict four-column SCP output, and audio `path` values in the normal output are relative to their own manifest directory; basic indexes do not add category/strict fields; `--music` is a first-class TSE `music_scp` resource; with `--sample_rate` a declared/real rate mismatch is counted as `fs_mismatch` in the validation summary by default, and `--strict_sample_rate` turns it into a hard error |
| `scp/prepare_offline.py` | Paired SE clean/noisy offline data | Fixed strict 4-column `spk1.scp` and `wav.scp` under `train/validation`; publication never deletes retired artifacts and fails fast on obsolete sidecars or cross-format artifacts until they are moved or deleted manually |
| `scp/prepare_aec_offline.py` | AEC mic/farend/nearend scan | The former GenerationManifest publication CLI is retired and fails closed; `scan_wavs` remains available |
| `scp/prepare_voicebank.py` | VoiceBank-Demand | Resampled clean/noisy SCPs under train/test (or validation) |
| `scp/prepare_testset_scps.py` | VoiceBank, URGENT, AEC, DNS, and other evaluation indexes | Delegates to shared evaluation logic and writes SCPs under `USE_TRAIN_TEST_DATA_ROOT` (or the repository placeholder when unset) |
| `generators/noise/build_stage_d_noise_manifests.py` | Noise Filter Stage D publication | Strict v6 noise JSONL plus publication sidecars; also has categorized-noise capability |
| `generators/tse/generate_realtse_pseudo_targets.py` | REAL-T-dev teacher pseudo-targets | `--max-trials` remains a non-publishing probe; the default official REAL-T path is fail-closed, while an exact acknowledgement enables only a restricted GenerationManifest v2 publication with TSE full reload and receipt |
| `generators/echo/build_real_farend_echo_pair_manifest.py` | Scan real farend/echo audio | Echo-pair JSON, train/val by default |
| `generators/echo/merge_farend_echo_pair_manifests.py` | Merge echo-pair manifests | Deduplicated and shuffled echo JSON |
| `generators/echo/generate_simulated_farend_echo_pairs.py` | Offline farend + RIR echo simulation | Stereo WAV plus `real_farend_echo_pair_manifest.json` / `real_farend_echo_pair_metadata.json` |
| `generators/aec/prealign_stereo_echo_ref.py` | Offline GCC materialization for stereo AEC pairs | Canonical module: `python -m tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref`; defaults to `ch0=echo/ch1=farend`, preserves relative paths, frame counts, and container headers by default (optional `--output-sample-rate`/`--output-format`/`--output-subtype` resample via soxr VHQ and transcode), and publishes audit JSONL/summary files (not a training manifest) |
| `generators/se/generate_offline_dataset.py` | Materialize an SE offline dataset from a dynamic config | Canonical module: `python -m tools.prepare_data_tools.generators.se.generate_offline_dataset`; schema v2 (`2.8`) publication binds the `dataset` root alias, writes alias-qualified reload SCPs, and commits through an owned sibling staging directory and writer lock |
| `generators/se/generate_dns3_offline_dataset.py` | DNS3/GTCRN paper-recipe offline generation | Canonical module: `python -m tools.prepare_data_tools.generators.se.generate_dns3_offline_dataset`; train/val/test WAVs and strict 4-column SCPs; CLI accepts kebab-case options only |
| `generators/aec/generate_simulated_testset.py` | Seven-scenario AEC dynamic simulation test set | GenerationManifest publication is retired and the CLI fails closed until AEC schema v2 reload support exists |
| `generators/aec/generate_ood_stress_set.py` | AEC OOD stress evaluation | GenerationManifest publication is retired and the CLI fails closed until AEC schema v2 reload support exists |
| `scp/prepare_aec_challenge_offline.py` | AEC Challenge dataset scan | Scan logic is retained; the GenerationManifest publication path also fails closed until AEC schema v2 reload exists |
| `generators/se/generate_scene_rirs.py` | Multi-microphone scene-level offline RIR bundle | Handles `C>1` spatial scenes only, writing each actor's full/direct/early `[C,L]` tracks plus a per-scene `scenes.jsonl`; `C=1` still uses `generate_simulated_rirs` |
| `generators/tse/import_wesep_enrollment.py` | Official wesep speaker cues → dynamic TSE enrollment | Adds only the explicit `dataset_id` level, converting the `spk_id` resources in `cues.yaml` into the nested JSON that `TSEOfflineDataset` consumes; audio content is untouched and `mix_spk_id` fixed-trial resources must not be aggregated |
| `generators/tse/prepare_dynamic_tse_manifests.py` | Strict dynamic_tse resources from explicit single-speaker directory lists | Every input JSONL line must contain exactly `uid`/`path`/`language`/`dataset_id`/`speaker_id`/`enrollments` |
| `generators/ss/prepare_echoset.py` | Strict EchoSet separation manifest | Builds the use_train strict separation JSONL following TIGER EchoSet pairing semantics |

`GenerationManifest` publication and reload are schema v2-only. Every resource and gate input is bound by an explicit root alias and a matching generation-root marker. Schema v1 collections cannot be published or reloaded and must be regenerated. Current schema v2 production reload support covers SE and TSE; AEC GenerationManifest publication CLIs remain fail-closed until their v2 reload migration is implemented.

`generate_offline_dataset` builds each non-dry-run in an owned sibling staging directory and serializes publication with a sibling writer lock. It publishes only a complete set; failures leave the current successful output unchanged. Replacement requires a generator-owned, non-symlink target. Successful reruns replace the complete directory, so obsolete WAV and debug files do not remain. `generation_manifest.json` records `counts.requested/published/failed/dropped` for the published set, and successful samples receive contiguous UIDs starting at `utt_00000000`.

These producers emit JSON, JSONL, SCP, and path-list files with different field and consumer contracts. Renaming extensions or swapping paths does not convert one contract into another.

Example for pre-aligning stereo AEC pairs (placeholder paths only):

```bash
python -m tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref \
    --input-root /path/to/raw_stereo_pairs \
    --output-root /path/to/gcc_prealigned_pairs \
    --workers 16
```

The default layout is `ch0=echo`, `ch1=farend`. Offline GCC only removes intrinsic delay. Constant delay uses one fixed front pad plus cropping to the original frame count. A time-varying pair is never materialized point by point from its trajectory: the tool takes `tau0` from the first reliable window in time order and applies that one fixed shift to the entire farend channel. This aligns the beginning while intentionally preserving the real drift relative to `tau0` for the model or deployment-time GCC/AEC. Low-confidence or zero-applied-delay inputs use a byte copy (when no conversion is requested), while a time-varying estimate without a reliable trajectory point fails closed. With explicit `--output-sample-rate` / `--output-format` (e.g. FLAC, with the extension rewritten to `.flac`) / `--output-subtype`, every file (including low-confidence ones) is decoded, aligned, resampled with soxr `VHQ` (fail-closed when soxr is missing), and re-encoded, and the residual gate runs at the output sample rate.

Before publication, every transformed staged file is decoded again and checked with the same windowed GCC parameters: constant files use `scope=full_pair`, while time-varying files check only the first reliable 1-second `scope=initial_window`. `--max-residual-error-ms` defaults to 5.0. A low-confidence residual, a time-varying full-pair residual for a constant input, or excessive residual error is handled per `--on-residual-failure`: `copy` (default) rewrites the file unaligned (still resampled/re-encoded when conversion is requested), records `policy=copy_residual_gate_failure`, and continues, while `abort` fails the whole run closed and leaves the existing target untouched. Per-file `residual_check` metadata records scope, expected/estimated/error values, and pass status; the summary aggregates checked/passed counts and residual-error max/mean/p50/p95. `gcc_prealign_records.jsonl` and the `gcc_prealign_summary.json` completion marker are published only after all files succeed. After generating a training manifest that points at the output root, set `aec_ref_echo_fine_gcc_align: false` to prevent a second GCC pass: constant intrinsic delay is then removed, a small amount of time-varying intrinsic delay is intentionally retained, and simulated random delay/jitter is still added on top. Do not change production YAML before that manifest exists.

## Full Unsplit Indexing: 14 Prepared Targets

The following recipe uses `scp/prepare_scp.py` in single-directory mode. Each invocation recursively indexes the complete input directory and performs no random train/validation split; names such as `train` and `val` describe source directories that are already separated. In this mode, `-o` is an output **file prefix without `.scp`**, not an output directory; the script appends `.scp`. Each prefix below therefore stays in the same parent directory as its dataset target.

```bash
PREPARE=tools/prepare_data_tools/scp/prepare_scp.py
VB=data/se/voicebank_demand
URGENT=data/shared/urgent26_track1_se_dataset
UNIVERSAL="$URGENT/urgent26_track1_universal_se"
NONBLIND="$URGENT/urgent26_nonblind_with_label"
DNS=data/se/DNS_Challenge_test
AEC=data/aec/AEC_Challenge/datasets

# VoiceBank-Demand: 4 targets
python "$PREPARE" "$VB/clean_trainset_wav" -o "$VB/voicebank_clean_train"
python "$PREPARE" "$VB/noisy_trainset_wav" -o "$VB/voicebank_noisy_train"
python "$PREPARE" "$VB/clean_testset_wav/clean_testset_wav" -o "$VB/voicebank_clean_val"
python "$PREPARE" "$VB/noisy_testset_wav/noisy_testset_wav" -o "$VB/voicebank_noisy_val"

# URGENT universal: 4 targets
python "$PREPARE" "$UNIVERSAL/simulation_train/clean" -o "$UNIVERSAL/urgent26_clean_train"
python "$PREPARE" "$UNIVERSAL/simulation_train/noisy" -o "$UNIVERSAL/urgent26_noisy_train"
python "$PREPARE" "$UNIVERSAL/simulation_validation/clean" -o "$UNIVERSAL/urgent26_clean_val"
python "$PREPARE" "$UNIVERSAL/simulation_validation/noisy" -o "$UNIVERSAL/urgent26_noisy_val"

# URGENT nonblind labels: 2 evaluation-only targets
python "$PREPARE" "$NONBLIND/clean" -o "$NONBLIND/urgent26_clean_nonblind"
python "$PREPARE" "$NONBLIND/noisy" -o "$NONBLIND/urgent26_noisy_nonblind"

# DNS Challenge: 3 noisy blind-set targets
python "$PREPARE" "$DNS/V2_V3_Blindset/noisy_blind_testset_v2_challenge_withSNR_16k" -o "$DNS/v2_blindset"
python "$PREPARE" "$DNS/V2_V3_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k" -o "$DNS/v3_blindset"
python "$PREPARE" "$DNS/V5_Blindset/Track1_Headset/noisy" -o "$DNS/v5_blindset"

# AEC official mixed waveform inventory: 1 asset-index target
python "$PREPARE" "$AEC/blind_test_set_icassp2023" -o "$AEC/blind_test_set_icassp2023"
```

The target inventory is VoiceBank 4 + URGENT universal 4 + URGENT nonblind 2 + DNS 3 + AEC mixed 1 = **14 SCP files**. The AEC mixed file contains **2,400 asset rows** across `*_enrl.wav`, `*_lpb.wav`, and `*_mic.wav`; it is an asset index only, not the paired input contract for official evaluation. Generate the formal per-subset `mic.scp`/`lpb.scp` files with:

```bash
export USE_TRAIN_TEST_DATA_ROOT=/path/to/test-data
python tools/prepare_data_tools/scp/prepare_testset_scps.py \
    --datasets aec_official --aec_subsets all \
    --aec_testset_root data/aec/AEC_Challenge/datasets \
    --output-dir "$USE_TRAIN_TEST_DATA_ROOT"
```

This writes paired indexes under `$USE_TRAIN_TEST_DATA_ROOT/AEC_Challenge_test/blind_test_set_icassp2023/{doubletalk,farend-singletalk,nearend-singletalk}/`.

## Train/Val and the Stage D Publication Set

- **`prepare_scp.py`** defaults to `--val_ratio 0.05 --seed 42` and independently splits every supplied resource (speech/noise/RIR/wind/music/`--extra`) into `train_{prefix}.*` and `val_{prefix}.*`. Therefore, a batch invocation with only `--speech` produces `train_speech_sources.scp` and `val_speech_sources.scp` by default. It writes a single `speech_sources.scp` only with explicit `--val_ratio 0`. `--speech_train/--speech_val` provide explicit speech splits, and `--follow_links` controls traversal of symlinked directories.
- **`prepare_offline.py`** defaults to `val_ratio=0.05, seed=42` and splits the paired UID set into UID-disjoint train/validation sets.
- **`build_real_farend_echo_pair_manifest.py`** defaults to `--val_ratio 0.05 --seed 42`, writing `<stem>_train.json` and `<stem>_val.json`; `--val_ratio 0` keeps one file.
- **`merge_farend_echo_pair_manifests.py`** merges echo-pair manifests with dedup and shuffle. Every entry is resolved against its source manifest directory and written as an **absolute path**, so relative entries are not re-interpreted against the merged manifest location; the output goes through the shared transactional writer (staging + fsync + atomic replace).
- **Validation consumption**: in dynamic mode, `data.validation` with `inherit_simulation: true` may override `speech_scp/noise_scp/music_scp/rir_scp/windnoise_scp` and, for TSE, `spk2enroll_json` with matching validation resources. When TSE sets `inherit_simulation: false`, `data.validation.spk2enroll_json` must be explicit; `real_farend_echo_pair_json` follows its separate echo-JSON contract.

Stage D uses **manifest schema version 6** for the strict record contract and **generator version 6.1** for generator identity; these are distinct versions. `build_stage_d_noise_manifests.py --output-dir ...` defaults to `--val-ratio 0.05 --seed 42` and publishes exactly this **three-file set**:

1. `train_noise_sources.jsonl`
2. `val_noise_sources.jsonl`
3. `noise_publication_metadata.json`

Every train/val JSONL record embeds `category_id`, `category_name`, and `roles`; the generator computes the global category vocabulary internally and returns `category_count`. The training Dataset directly consumes only the two JSONL manifests. Metadata records build provenance, content-integrity fingerprints, and the top-level `manifest_summary`; it is not a training input. `manifest_summary` describes publication size, roles, and the train/val split, not runtime balancing. Audit the realized category/temperature training distribution with `tools/verify/audit_noise_sampling.py`. `--print-stats` additionally prints the current manifest publication summary to stdout.

Single-file mode publishes `<output>.jsonl`, the explicit category vocab requested by `--category-vocab`, and `<output>.metadata.json`; its metadata contains a `manifest_summary` without `split`. It is a separate atomic publication set, not part of the split three-file set.

## Categories, Sampling, and Wind/Music Roles

Stage D generation publishes category/role labels, strict evidence, and statistics; it **does not resample records**. Selection happens at training Dataset runtime:

- natural: naturally selects from ordinary source candidates and does not require category metadata;
- category: chooses categories with equal probability, then chooses a UID uniformly within the selected category;
- temperature: within the first available SR tier, chooses categories with `P(c) ∝ n_c^alpha × difficulty(c)`, then chooses a UID uniformly within the category; the default `alpha` is `0.5`.

Minimal configurations follow. The `~/...` paths are placeholders and **must be replaced with usable resource paths before runtime**. Configure dynamic training sources under `data.train`.

```yaml
# natural
data:
  mode: dynamic
  train:
    noise_scp: "~/manifests/train_noise_sources.scp"
```

```yaml
# equal-probability categories, uniform UIDs within a category
data:
  mode: dynamic
  train:
    noise_scp: "~/manifests/train_noise_sources.jsonl"
    sampling:
      noise:
        balance_categories: true
        balance_mode: category
```

```yaml
# temperature: P(c) ∝ n_c^alpha × difficulty(c)
data:
  mode: dynamic
  train:
    noise_scp: "~/manifests/train_noise_sources.jsonl"
    sampling:
      noise:
        balance_categories: true
        balance_mode: temperature
        alpha: 0.5
```

When `roles` in a categorized JSONL record is a list, it is authoritative for pool splitting. Only a missing or non-list `roles` value falls back to matching `balance_category` (or compatible `category_name`) through `core.data.noise_roles`.

| Source | Runtime behavior |
|--------|------------------|
| Wind | SE, AEC, and TSE split it out of ordinary noise into the wind pool and merge it with an explicit `windnoise_scp` |
| Music | Only TSE splits it into the music pool and merges it with an explicit `music_scp`; SE/AEC retain it as ordinary additive noise |

Canonical `roles` in categorized JSONL route ordinary/wind/music pools directly in the consumer, replacing path-filtered derivative SCPs.

## Current Speech-Balancing Gap

`prepare_scp.py` does not generate the `language`, `dataset_id`, and `speaker_id` metadata required by speech-balancing controllers. In particular, a strict TSE speech manifest still requires an external or dedicated preparation flow. Do not claim that a generic strict four-column SCP or a strict noise-publication JSONL satisfies that speech contract.

## Naming Conventions (de-facto)

Canonical dynamic-source filenames use a fixed resource prefix: `.scp` is strict four-column `uid fs length path`, and `.jsonl` is one JSON object per line; `.json` arrays are no longer source manifests.

| Resource | train | validation | unsplit |
|----------|-------|------------|---------|
| Speech | `train_speech_sources.scp` | `val_speech_sources.scp` | `speech_sources.scp` |
| Noise (plain) | `train_noise_sources.scp` | `val_noise_sources.scp` | `noise_sources.scp` |
| Noise (categorized/strict) | `train_noise_sources.jsonl` | `val_noise_sources.jsonl` | `noise_sources.jsonl` |
| RIR | `train_rirs.scp` | `val_rirs.scp` | `rirs.scp` |
| Wind | `train_wind_noise_sources.scp` | `val_wind_noise_sources.scp` | `wind_noise_sources.scp` |
| Music (TSE) | `train_music_sources.scp` | `val_music_sources.scp` | `music_sources.scp` |
| AEC echo/ref | `real_farend_echo_pair_train.json` | `real_farend_echo_pair_val.json` | `real_farend_echo_pair_manifest.json` (metadata: `real_farend_echo_pair_metadata.json`) |

Sources and semantics of the two dynamic-source serializations:

- **`.scp`**: strict four columns `uid fs length path`; fixed `prepare_scp.py` output.
- **`.jsonl`**: one JSON object per line; basic dynamic records require `uid/fs/length/path` per line. Categorized noise is published by `build_stage_d_noise_manifests.py` (`train/val_noise_sources.jsonl`); the `.jsonl` shape for RIR/wind/music/speech is reserved for dedicated producers or custom flows.

`.list` (one path per line) is not a dynamic-source manifest; use it only with TSE offline/wesep consumers that explicitly support plain lists.

- The batch resource-to-prefix mapping is hardcoded in `prepare_scp.py`'s `BATCH_KEY_TO_PREFIX`: `speech_sources / train_speech_sources / val_speech_sources / noise_sources / rirs / wind_noise_sources / music_sources` (with `train_`/`val_` added when random splitting is enabled). Training YAML references these resources through `data.train.*_scp`; `music_scp` is also a split-level resource under `data.train` / `data.validation`, and `--music` emits the corresponding YAML field.
- Offline outputs follow their producer and Dataset layout: `prepare_offline` uses strict 4-column `spk1.scp/wav.scp` under `train/validation`; `prepare_voicebank` and `generate_dns3_offline_dataset` use strict 4-column `clean.scp/noisy.scp` under `train/test` or `train/val`. Offline training SCP metadata is always embedded in each track.

## Canonical Entry Points

- `python tools/prepare_data_tools/scp/prepare_offline.py`
- `python tools/prepare_data_tools/scp/prepare_testset_scps.py --datasets aec_sim`
- `python -m tools.prepare_data_tools.generators.noise.generate_wind_noise`
- `_common.py` is an internal shared module for scripts under `prepare_data_tools`. Its API is **not stable**; cross-directory reuse should import the corresponding `tools.shared.*` module directly.

## Testing

```bash
python tools/prepare_data_tools/inspect_manifest.py --help
python -m pytest tests/test_prepare_data_manifest_contracts.py -v
python -m pytest tests/test_prepare_scp.py tests/test_prepare_offline_voicebank.py \
    tests/test_farend_echo_manifests.py -v
```
