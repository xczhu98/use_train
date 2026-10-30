# 数据准备工具（prepare_data_tools）

`tools/prepare_data_tools/` 把原始音频目录变成训练 Dataset 能读的索引，并生成仿真资源（RIR、风噪、回声对、离线 clean/noisy 对）。这里不包含训练或推理代码。

- `scp/` —— 索引（SCP/JSONL）生成：`prepare_scp.py`、`prepare_offline.py`、`prepare_voicebank.py`、`prepare_testset_scps.py`、`prepare_aec_offline.py`、`prepare_aec_challenge_offline.py`。
- `generators/` —— 仿真数据生成，按资源类别分为 `aec/`、`se/`、`tse/`、`ss/`、`noise/`、`echo/`。
- `inspect_manifest.py` —— 只读检查器，用来确认某个 manifest 到底是什么格式、具备什么能力。
- `generation_manifest.py`、`offline_publication.py`、`offline_reload_gate.py`、`manifest_contracts.py`、`se_index_publication.py` —— 各 producer 共用的发布/重载合同。
- `_common.py` —— 内部共享模块，**不承诺 API 稳定**（见「正式入口」章节）。

> **注意：** `generators/` 按资源类别分为 `aec/`、`se/`、`tse/`、`ss/`、`noise/`、`echo/`；RIR 生成器位于 `generators/se/`。

## 术语表

- **SCP**：纯文本索引，每行一条记录。本目录统一使用严格四列 `uid fs length path`（`uid` 唯一标识、`fs` 采样率、`length` 帧数、`path` 音频路径）。
- **manifest**：所有索引文件的统称。“序列化格式”（SCP / JSONL / JSON）和“capability”（记录能支撑什么训练能力）是两个独立维度，详见「序列化与 capability 是两个正交维度」。
- **RIR**：房间冲激响应，描述房间如何改变声音；仿真把干净语音与该 RIR 卷积，得到带混响的训练数据。
- **GenerationManifest v2**：生成式离线数据集的发布合同：谁生成的、每个资源的内容 SHA-256、如何重载；schema v1 集合必须重新生成。
- **fail-closed**：发现异常就报错停下，而不是猜一个结果继续跑。本目录的校验基本都是 fail-closed，所以“报错”通常说明校验按预期工作。

## 怎么读这份文档

1. 第一次来：先看「快速开始」。
2. 要生成仿真资源（RIR / 风噪 / 回声）：直接读对应的生成器章节。
3. 不知道该让训练配置消费哪种索引：读「SCP 索引与数据清单脚本」的 consumer → producer → inspector 流程。
4. 要审计发布结果或迁移旧资产：读「重要 producer 及其边界」。

## 快速开始

```bash
# 1. 单目录模式：把一个音频根目录索引成一份严格四列 SCP
#    -o 是不带 .scp 的输出前缀，因此这里写出的文件是 speech_sources.scp
python tools/prepare_data_tools/scp/prepare_scp.py /data/speech \
    -o data/train_sources/speech_sources

#    批处理模式会对每个资源默认按 95:5 划分，写出两份文件
#    （train_speech_sources.scp + val_speech_sources.scp）；--val_ratio 0 关闭划分
python tools/prepare_data_tools/scp/prepare_scp.py --speech /data/speech \
    -o data/train_sources

# 2. 只读检查刚写出的清单（不会打开清单里引用的音频）
python tools/prepare_data_tools/inspect_manifest.py \
    data/train_sources/train_speech_sources.scp --profile dynamic-source

# 3. 用默认 gpuRIR 配方生成仿真 RIR
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --output-dir /data/rir/gpurir_se_16k
```

第 2 步会打印 `serialization_format`、`record_count`、`sample_rates` 以及提示训练配置键的 `yaml_hint`。第 3 步拒绝写入非空目录，产物为 `rirs.scp` + `rirs_meta.json` 与 RIR 文件；训练配置随后把 `data.train.rir_scp` 指向该 `rirs.scp`。

---

# RIR 仿真数据准备

本节只说明 RIR 资源生成。训练期 Layer A–F 语义，以及 Dynamic Dataset 仿真与 AEC 模型输入前端的边界，以 [`docs/guides/simulation/Simulation_Flow_Logic.md`](../../docs/guides/simulation/Simulation_Flow_Logic.md) 为准。

## 文件说明

- `generators/se/generate_simulated_rirs.py`：统一命令行入口。
- `generators/se/rir_backends/`：三种 RIR backend 的实现。

## 支持的 Backend

| Backend | 方法 | 依赖 | 硬件 |
|---------|------|------|------|
| `rirgen` | image-source（`rir_generator`） | `pip install rir_generator` | CPU |
| `fram`   | Fast Random Approximation（FRAM-RIR） | PyTorch + torchaudio | CPU/GPU |
| `gpurir` | GPU 加速 image-source（`gpuRIR`） | `pip install gpuRIR` | GPU |

三种 backend 均为可选依赖。脚本在运行时会自动检测可用 backend，并对缺失依赖给出清晰提示；gpuRIR 还需要至少一个实际可用的 GPU。

## RIR 生成默认值

`generators/se/generate_simulated_rirs.py` 是 RIR 生成的正式入口。输出目录和目录名均可由用户完全指定，例如 `--output-dir /data/rir/gpurir_se_16k` 会把清单和生成数据全部写入该目录；不需要额外的“命名参数”，输出目录本身就是数据集名称。目标目录必须不存在或为空；非空目录会被拒绝，以避免不同清单对应的文件混在一起。

生成器的 canonical profile 与配置说明位于 `generators/se/` 当前代码目录：
`generators/se/rir_gpurir.yaml`、`generators/se/rir_fram.yaml` 和
`generators/se/rir_rirgen.yaml` 分别对应三个 backend；字段均使用
带单位的 `snake_case` 命名，详情见 `generators/se/README.md`。配置文件不属于训练/推理配置，
不放入仓库顶层 `configs/`。

当前默认配置如下：

- 默认推荐使用 `rir_gpurir.yaml`，生产序列化格式为 PKL。
- `rir_gpurir.yaml` 面向普通室内语音；压力测试直接覆盖同一 YAML 的 RT60、房间、距离和移动字段。
- `sample_rates_hz`、`rt60_range_sec`、房间范围、距离范围、移动比例等参数全部写在同目录 YAML 中，并在每个字段旁注明单位和约束。

使用 `--config` 加载 profile；可用 `--output-dir` 覆盖 YAML 中的 `output_dir: null`。`--method`、`--sample-rate-hz` 等 canonical CLI 参数可以覆盖对应配置字段，但不再提供旧版参数别名。`all` 模式运行所有可运行 backend；没有 GPU 时跳过 gpuRIR，继续 CPU backend。

## 输出命名规范

生成的 RIR 文件名与 SCP 中的 `uid` 统一编码关键元信息：

```
{method}_fs{fs}_{field_or_field_moving}_rt{rt60:.2f}_dist{dist:.2f}_room{L:.1f}x{W:.1f}x{H:.1f}_{idx:08d}.{ext}
```

示例：

```
fram_fs16000_near_rt0.50_dist0.32_room4.5x5.2x3.1_00000123.wav
gpurir_fs16000_far_moving_rt0.80_dist1.20_room6.0x5.0x3.0_00000123.pkl
```

`--format auto`（默认）会把静态和移动 RIR 统一写成 `.pkl`，载荷内携带完整元数据（`rt60`、`dist`、`room_dim`、位置，移动 RIR 另含轨迹），**生产数据推荐使用 PKL**。静态 RIR 也可以显式选择 FLOAT WAV 或 24-bit FLAC：FLOAT WAV 与 PKL 均可保留超过 `[-1, 1]` 的 float32 幅值；FLAC 是整数 PCM，仅在全部样本位于 `[-1, 1]` 时写入，超范围会直接报错，避免静默截断。移动 RIR 必须使用 `.pkl`（多段序列无法表示为单声道音频）。移动 pkl 的 `rir` 是 `[T_seg, n_samples]` 的 float32 数组，并包含轨迹等 metadata。

## 输出目录结构

RIR 文件按仿真方法、采样率、静态/移动模式、近/远场和分片组织，目录结构如下：

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
        └── moving/                 # 仅 gpuRIR
            ├── near/
            │   ├── shard_0000/
            │   └── ...
            └── far/
                ├── shard_0000/
                └── ...
```

请求数量为 0 的 mode 或 field 不会创建目录。输出目录必须不存在或为空；脚本会拒绝非空目录，避免不同运行的文件和清单混在一起。`--shards` 控制每个 `(method, fs, mode, field)` 组合的分片数量。`rirs.scp` 始终是四列 `uid fs num_samples path`，其中 `path` 相对于清单文件目录；多采样率时（如 `--sample-rate-hz 16000 32000`）各采样率独立 `fs{fs}/` 目录树，`rirs.scp` 合并全部条目，每个采样率另有独立的 `rirs_fs{fs}.scp`（同目录、同样相对路径）。`rirs_meta.json` 保存共享 RT60、比例、请求/有效数量、实际格式和输出 inventory；每条 RIR 的 SHA-256 指纹**默认跳过**（`sha256: null`），大数据集可传 `--inventory` 显式开启（较慢），`--no-inventory` 显式声明默认行为，`compute_sha256` 字段记录实际选择。增加 `--write-metadata-json` 后，每条 RIR 还会生成一个同名 `.json` metadata 文件。

## 使用示例

### 默认 gpuRIR 配方：100,000 条静态 RIR

```bash
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --output-dir /data/rir/gpurir_se_16k
```

该命令请求近场 50,000 条、远场 50,000 条；默认 `static_moving_ratio=[1,0]` 且 `motion.enabled=false`，因此只生成静态 PKL。`--gpus` 可省略，未设置 `CUDA_VISIBLE_DEVICES` 时自动检测可用 GPU。

### 显式生成静态 FRAM-RIR

```bash
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_fram.yaml \
    --output-dir /data/rir/sim_rirs_fram_8k \
    --field near --total-rirs 10000 --validate
```

显式选择 `fram` 或 `rirgen` 不会创建 moving 目录；需要把所有可用 backend 都作为静态 RIR 运行时使用 `--method all`。

### 自定义比例和共享 RT60

```bash
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --output-dir /data/rir/gpurir_custom --total-rirs 20000 --gpus 0 1
```

### 移动声源 profile

```bash
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --output-dir /data/rir/gpurir_moving --motion-enabled --gpus 0 1 2 3
```

移动 pkl 结构为 `{"rir": [T_seg, n] float32, "fs": int, "moving": true, "n_segments": int, "segment_sec": float, "trajectory": [T_seg, 3], "speed": float, "rt60": float, "dist_min": float, "dist_max": float, ...}`；`dist` 为起点距离，`dist_min`/`dist_max` 是轨迹全程与麦克风的最远/最近距离。训练端通过 `moving: true` 自动识别。

## RT60、吸声系数与 RIR 长度

RT60 是房间属性，不是近场/远场标签的固有属性。近远场差异主要应由声源—麦克风距离、直达声强度以及最终的直达声—混响声比（DRR）体现。因此让两类 field 共用 RT60 范围，比人为给近场和远场绑定不同房间混响先验更合理。

默认 RT60 范围为 `0.1–1.0 s`，近场和远场共用。需要更长混响压力测试时，应在对应 backend YAML 中显式扩大范围。

代码没有使用一个固定的“混响系数”。对每个采样房间和 RT60，`rir_backends/` 先按 Sabine 关系计算平均吸声系数：

```
alpha = 24 * ln(10) * V / (c * S * RT60)
```

并重采样直到 `alpha < 1`。gpuRIR 随后通过 `beta_SabineEstimation`，结合房间尺寸和 RT60 推导墙面反射系数 `beta`，所以 `beta` 也不是全局固定值。按默认房间范围（`L,W=3–10 m`、`H=2.5–4.5 m`、`c=340 m/s`）和 RT60 范围估算，角点 Sabine `alpha` 约为 `0.059–0.962`，默认配置仍落在当前 `alpha < 1` 检查边界内；吸声/反射系数按每个房间动态推导，没有独立 CLI 参数。

未显式指定 `near_rir_length_samples` / `far_rir_length_samples` 时，脚本使用 RT60 上限、对应距离上限和 50 ms 安全余量计算长度，并向上对齐到 256 的倍数：

```
n_samples = ceil(fs * (rt60_hi + dist_hi / c + 0.05) / 256) * 256
```

在默认范围、16 kHz 下，近场自动得到 16,896 samples，远场得到 17,152 samples；32 kHz 下分别为 33,792 和 34,048 samples。显式指定更短长度会发出 warning，并可能截断后期混响尾部。

## 可复现性与严格验证门禁

每条样本的场景种子由基础 `--seed`、backend、field/mode 域和样本序号经 BLAKE2b 派生，**不依赖** GPU 分配、worker 数量或任务调度；相同 CLI 参数在任意 GPU 子集或 worker 数量下重跑，采样的房间、声源/麦克风位置、RT60 目标和移动轨迹完全一致。FRAM backend 的随机状态按任务隔离，单条样本结束后恢复全局 NumPy/Torch 状态，逐任务完全可复现。

一个仅针对 gpuRIR 的注意事项：其晚期漫射混响尾部由库内部的进程内伪随机状态生成（本脚本无法播种）。相同 GPU 布局下重复运行逐样本一致；GPU 数量不同的两次运行，场景几何与早期部分一致，但晚期尾部可能有约 1e-3 量级的细微差异。这是 gpuRIR 库的固有行为，CLI 无法控制。

`--validate` 会在写入任何 manifest **之前**检查生成的 RIR：实际采样率和帧数必须与 SCP 声明一致，样本必须全部有限，且静态 RIR 或移动 RIR 的每个 segment 峰值都不能低于阈值。发现异常样本时进程返回非零状态，不发布 `rirs.scp` / `rirs_meta.json`；已写出的音频文件保留在磁盘上，重试前需要清空输出目录。

## 常用命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 必填 | 同目录 RIR profile YAML；字段全部使用带单位的 `snake_case` |
| `--output-dir` | 可选 | 覆盖 YAML 中的输出目录；目录必须不存在或为空 |
| `--method` | profile | `rirgen`、`fram`、`gpurir` 或 `all` |
| `--sample-rate-hz` | profile | 目标采样率，可多值但不可重复；每个采样率独立 `fs{fs}/` 目录树 |
| `--field` | `both` | `near`、`far` 或 `both` |
| `--total-rirs` | profile | 总 RIR 数；`field=both` 时按 near/far 比例切分 |
| `--near-rirs` / `--far-rirs` | profile | 直接指定 field 数量；未指定时使用 total-rirs |
| `--rt60-range-sec` | profile | 近场/远场共享 RT60 范围，单位 s |
| `--near-distance-range-m` / `--far-distance-range-m` | profile | 源麦距离范围，单位 m |
| `--room-length-range-m` / `--room-width-range-m` / `--room-height-range-m` | profile | 房间尺寸范围，单位 m |
| `--workers` | `16` | FRAM/RIRGenerator 的 CPU 进程数 |
| `--shards` | `10` | 每个 `(method, fs, mode, field)` 组合的分片数 |
| `--gpus` | 自动检测（全部 GPU） | `gpurir` 使用的 GPU token：未设置 `CUDA_VISIBLE_DEVICES` 时默认自动检测全部可用 GPU（本机 8 卡），裸整数表示物理 GPU 编号；设置 CVD 后裸整数表示可见 slot。`slot:N` 显式选择可见 slot，`id:N` 显式选择物理 GPU；UUID 与 MIG token 按字面匹配 |
| `--format` | `auto` | `auto` 默认全部输出 PKL，推荐生产使用；FLOAT WAV 可保留超出 `[-1,1]` 的值，FLAC 仅接受 `[-1,1]` 内的静态 RIR；移动任务只能使用 PKL |
| `--write-metadata-json` | 关闭 | 为每条 RIR 写入 .json 元信息 |
| `--validate` | 关闭 | 严格发布门禁：写入 manifest 前检查采样率、帧数、有限性和逐段峰值；发现异常时返回非零且不发布 `rirs.scp` / `rirs_meta.json` |
| `--inventory` / `--no-inventory` | 关（默认） | 计算并记录 `rirs_meta.json` 中每条 RIR 的 SHA-256 指纹；默认跳过（`--no-inventory` 显式声明默认行为） |
| `--motion-enabled` | 关闭 | 是否生成移动声源；仅 gpurir 支持 |
| `--n-segments` / `--n-segments-range` | `[3,6]` | 固定段数或逐场景采样的段数闭区间 |
| `--segment-duration-sec` | `0.2` | 每个移动段持续时间，单位 s |
| `--speed-range-mps` | profile | 声源移动速度范围，单位 m/s |
| `--trajectory-type` | profile | `linear`、`waypoint` 或 `both` |

## 训练配置接入

将生成的 `rirs.scp` 路径填入 YAML 配置的 `data.train.rir_scp`：

```yaml
data:
  mode: dynamic
  train:
    rir_scp: /data/rir/sim_rirs/rirs.scp
    prob_reverberation: 0.5
```

### 消费移动声源 RIR

移动 RIR 的消费是**自动的**：训练端 `_load_audio_impl` 读到 pkl 的 `moving: true` 时保留
`[T_seg, n]` 二维结构，`_mix_acoustic` 据此调用 `add_moving_reverberation`：语音按
`segment_sec` 分块，每块用对应段 RIR 卷积，各段保留完整 `n_h - 1` 卷积尾音并在绝对时间轴
上 overlap-add，因此不存在段间 click，也不会丢失晚期混响。`estimate_early_rir` 对每段逐行
提取 early，`clean_target` 口径自动成立。分段时长优先取 pkl 自带的 `segment_sec`；字段缺失时
回退到 `SimulationConfigs.segment_sec`（默认 `0.5` 秒）。静态与移动 RIR 可混放在同一 `rir_scp`。示例：

```yaml
data:
  mode: dynamic
  train:
    rir_scp: data/shared/rirs/16k_moving/train_rirs.scp
    simulation_cfg:
      segment_sec: 0.5
```

## 测试

使用 pytest 运行测试：

```bash
python -m pytest tests/test_generate_simulated_rirs.py -v
```

---

# 风噪（Wind Noise）仿真数据生成

`generators/noise/generate_wind_noise.py` 基于 Mirabilii & Habets（IWAENC 2022）的单通道风噪生成算法批量生成风噪数据，输出训练框架可直接加载的 4 列 `wind_noise_sources.scp`。

## 文件说明

- `generators/noise/generate_wind_noise.py`：正式命令行入口（并行、分片、4 列 SCP）。
- `generators/noise/_wind_noise_backend.py`：vendored 的 `WindNoiseGenerator` 后端（MIT License，版权头保留在文件内）。它使用仓库内等价的 `lsf2poly` 实现，不依赖演示用的 `matplotlib`/`sounddevice`，**无需安装任何额外依赖**。

## 使用示例

```bash
# 16kHz、200 条、每条 15 秒（URGENT2026 train 口径）
python -m tools.prepare_data_tools.generators.noise.generate_wind_noise \
    --output-dir /data/noise/wind_noise_16k \
    --fs 16000 --num_data 200 --duration 15 --workers 16

# 多采样率
python -m tools.prepare_data_tools.generators.noise.generate_wind_noise \
    --output-dir /data/noise/wind_noise_multi \
    --fs 8000 16000 24000 48000 --num_data 200
```

## 常用命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | 必填 | 输出目录（已存在且非空时默认拒绝写入） |
| `--fs` | `16000` | 目标采样率列表（可多值） |
| `--num_data` | `200` | 每个采样率生成的条数 |
| `--duration` | `15.0` | 每条时长（秒） |
| `--gustiness_range` | `3,10` | 阵风强度范围（1=恒定风速，越大阵风越强） |
| `--seed` | `42` | 随机种子（每条样本 seed = seed + idx，完全可复现） |
| `--workers` | `16` | 并行进程数 |
| `--shards` | `16` | 每个采样率下的分片数 |
| `--overwrite` | 关闭 | 允许写入已存在的非空输出目录 |

> 可复现性注意：每条样本的种子为 `seed + idx`，其中 `idx` 跨采样率全局递增。**只有 CLI 参数（含 `--fs` 列表与 `--num_data`）完全不变时，两次运行才逐样本一致**；增删采样率会改变既有样本的种子。

## 输出目录结构

```
out_dir/
├── wind_noise_sources.scp     # 4 列: uid fs num_samples path（相对于本清单目录）
├── wind_noise_meta.json       # 每个采样率的条数 / gustiness 统计
└── wind_noise/
    └── fs{fs}/
        └── shard_{nnnn}/
            └── wind_fs{fs}_gust{g:.1f}_{idx:08d}.wav
```

## 训练配置接入

将生成的 `wind_noise_sources.scp` 路径填入 YAML 配置的 `data.train.windnoise_scp`：

```yaml
data:
  mode: dynamic
  train:
    windnoise_scp: /data/noise/wind_noise_16k/wind_noise_sources.scp
```

## 测试

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

# SCP 索引与数据清单脚本

本节区分“文件怎么序列化”和“记录具备什么训练能力”。不要只看扩展名，也不要假设不同 producer 的输出可以互换。

## 选择流程：consumer → producer → inspector

1. **先确定消费模式**：
   - SE/AEC/TSE 的 dynamic 模式从 `data.train` / `data.validation` 读取 split-level dynamic source，例如 `speech_scp`、`noise_scp`、`music_scp`、`rir_scp`、`windnoise_scp`；dynamic TSE 还要求 `spk2enroll_json`。
   - SE offline 的 clean/noisy 与 AEC offline 的 mic/farend/nearend/error 轨消费严格四列 `uid fs length path` SCP；runtime 不再读取外部采样率或长度 sidecar。
   - TSE `offline_tse`（wesep/REAL-TSE）消费 `raw.list + samples.jsonl + cues.yaml`；其中 `raw.list` 是指向样本清单的 path-list，**不使用 SE/AEC 的四列 offline SCP 合同，也不是 dynamic manifest**。若要迁移官方 speaker-level `audio.json`，使用 `generators/tse/import_wesep_enrollment.py` 加入显式 `dataset_id`，转换为 dynamic TSE 的 nested `spk2enroll_json`；`mix_spk_id` fixed-trial 资源不得聚合。
2. **再选择 producer**：优先选与目标 Dataset 布局直接对应的脚本；不要因为两个文件都叫 JSON/JSONL 就互换。
3. **最后只读验证**：用 inspector 检查序列化格式、记录数、采样率和目标 capability；它不会打开记录内音频。

```bash
python tools/prepare_data_tools/inspect_manifest.py MANIFEST \
    --profile {dynamic-source,dynamic-speech,categorized-noise,strict-noise-publication} [--json]
```

默认 profile 是 `dynamic-source`。输出包含 `serialization_format`、`record_count`、`sample_rates`、`has_metadata`、`length_complete`、`category_complete`、`strict_fields_complete` 和 `yaml_hint`。`yaml_hint` 按用途给出对应配置键：`dynamic-speech` 输出 `speech_scp`，noise profiles 输出 `noise_scp`；检查 RIR/wind manifest 时把提示键分别替换为 `rir_scp`、`windnoise_scp`，不能照抄。

## 序列化与 capability 是两个正交维度

- **serialization** 只描述 dynamic source 载体：严格四列 SCP 或 JSONL。
- **capability** 描述训练合同：`dynamic-source`、`categorized-noise`、`strict-noise-publication`。JSONL 可携带 metadata，但**不表示已经按类别均衡**；strict schema v6 是发布审计合同，**不是**启用 category balance 的必要条件。

| 序列化 | 记录形态 | 可用范围 |
|--------|----------|----------|
| 4 列 SCP | `uid fs length path` | natural dynamic；`prepare_scp.py` 固定输出 |
| JSONL | 每行一个含 `uid/fs/length/path` 的 object，可另带 metadata | dynamic；分类/strict 能力取决于字段完整度 |

SCP/JSONL 的消费侧统一约定（`core/data/noise_manifest_adapter.py`）：相对音频路径一律相对 **manifest 文件所在目录**解析（绝对路径与 `~` 开头保持不变），与启动训练时的 cwd 无关。SCP 行内含空格的字段必须 shell quoting（`prepare_scp.py` 输出自动满足）；每行必须恰有 `uid fs length path` 四个字段，旧三列和 JSON array 均 fail-closed。`scp/prepare_scp.py`、`scp/prepare_offline.py` 与 `scp/prepare_voicebank.py` 的普通输出默认采用该 manifest-relative 形式；历史清单中的完整绝对路径仍可直接读取。

## 训练输入字段与启动性能

| 配置输入 | 当前记录字段 | 推荐生产合同 |
|----------|--------------|--------------|
| `speech_scp` | `uid/fs/length/path`；speech balancing 还需要 JSONL 的 `language/dataset_id`，speaker controller 或 TSE strict speech 还需要 `speaker_id` | 在离线准备阶段写入正整数 `length`；使用四列 SCP 或含完整 metadata 的 JSONL |
| `noise_scp` | `uid/fs/length/path`；分类 JSONL 另有 `category_id`、`balance_category`/`category_name`、`roles`；strict v6 再带 admission/verification 字段 | 基础字段必须完整；需要均衡或 wind/music 拆池时使用 `build_stage_d_noise_manifests.py` 的分类 JSONL |
| `rir_scp` | `uid/fs/length/path`；path 可指向波形或 RIR pickle | 使用四列 SCP 或 JSONL；moving RIR 的 `moving/segment_sec/trajectory` 等信息保留在 pickle payload，不复制到 SCP |
| `windnoise_scp`、TSE `music_scp` | 与通用 dynamic source 相同的 `uid/fs/length/path` | 使用严格四列 SCP；也可由分类 `noise_scp` 的 canonical `roles` 自动拆池并与显式清单合并 |
| AEC `real_farend_echo_pair_json` | canonical 对象 `{"pairs": ["stereo.wav"], "meta": {...}}`；每项为非空 stereo 路径字符串（ch0=echo，ch1=farend） | 这是独立 pair 合同，当前不携带统一 `uid/fs/length`；裸 array、list/dict pair、双文件行和 SCP 别名均拒绝 |

`speech_scp` 是当前最重要的性能门禁：`strict_load: true`（默认）下，dynamic speech manifest 必须带完整 `length`，Dataset 构造会直接 fail-fast，**不会**在 DataLoader worker 启动前逐条 `soundfile.info()`。生产训练必须使用 `prepare_scp.py` 在离线阶段并行物化四列 metadata。`speech_length_file` 不是 SE/AEC offline 合同；只有显式 `strict_load: false` 才允许训练期 `sf.info()` 探测（仍建议只在迁移旧数据时使用）。noise/wind/RIR 的 `length` 只用于摘要和轻量 metadata，不控制实际仿真裁切。

| Inspector profile | 必须满足的能力 | 典型用途 |
|-------------------|----------------|----------|
| `dynamic-source` | 非空且可解析为严格四列 SCP 或 JSONL | natural dynamic sampling；能力检查 |
| `dynamic-speech` | 非空，每条记录都有正整数 `uid/fs/length/path` | `speech_scp` 生产发布门禁，与 strict Dataset 构造语义一致 |
| `categorized-noise` | ordinary noise 每条记录都有非负 `category_id`，并有非空、无首尾空白的 `balance_category`（兼容 `category_name`） | category/temperature noise sampling |
| `strict-noise-publication` | 每条记录满足当前 Noise Filter strict publication 字段合同 | Noise Filter Stage D 发布审计；比训练分类能力更严格 |

## 重要 producer 及其边界

| 脚本 | 面向场景 | 主要输出/边界 |
|------|----------|---------------|
| `scp/prepare_scp.py` | speech/noise/RIR/wind/music 的通用扫描 | 固定生成严格四列 SCP，普通输出的音频 `path` 相对于各自 manifest 目录；基础索引不产生 category/strict 字段；`--music` 是 TSE `music_scp` 一等资源；`--sample_rate` 覆盖声明采样率时与真实值不一致默认计入 validation summary 的 `fs_mismatch`，加 `--strict_sample_rate` 改为直接报错 |
| `scp/prepare_offline.py` | SE clean/noisy offline 配对 | 在 `train/validation` 下固定生成严格四列 `spk1.scp`、`wav.scp`，普通输出的音频 `path` 相对于各自 manifest 目录；发布不会自动删除退役 artifact，检测到旧 sidecar 或跨格式 artifact 时要求手工移走或删除后重试 |
| `scp/prepare_aec_offline.py` | AEC mic/farend/nearend 扫描 | 原 GenerationManifest publication CLI 已下线并 fail-closed；保留 `scan_wavs` 严格扫描能力 |
| `scp/prepare_voicebank.py` | VoiceBank-Demand | 重采样后生成 `train/test`（或 validation）clean/noisy SCP，音频 `path` 相对于各自 manifest 目录 |
| `scp/prepare_testset_scps.py` | VoiceBank、URGENT、AEC、DNS 等评测索引 | 委托评估公共逻辑在 `USE_TRAIN_TEST_DATA_ROOT` 下生成 SCP（未设置时使用仓库占位布局） |
| `generators/noise/build_stage_d_noise_manifests.py` | Noise Filter Stage D 发布 | strict v6 noise JSONL 及发布 sidecar；同时具备 categorized-noise 能力 |
| `generators/tse/generate_realtse_pseudo_targets.py` | REAL-T-dev teacher 伪目标 | `--max-trials` 保留为不发布的 probe；官方 REAL-T 默认 fail-closed，只有精确 acknowledgement 才能生成受限的 GenerationManifest v2 publication，并执行 TSE full reload 与 receipt |
| `generators/echo/build_real_farend_echo_pair_manifest.py` | 扫描真实 farend/echo 音频 | echo pair JSON，默认 train/val |
| `generators/echo/merge_farend_echo_pair_manifests.py` | 合并 echo pair manifest | 去重、shuffle 后的 echo JSON |
| `generators/echo/generate_simulated_farend_echo_pairs.py` | farend + RIR 离线仿真回声 | stereo WAV + `real_farend_echo_pair_manifest.json` / `real_farend_echo_pair_metadata.json` |
| `generators/aec/prealign_stereo_echo_ref.py` | stereo AEC pair 的离线 GCC 预对齐物化 | canonical module：`python -m tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref`；固定默认 `ch0=echo/ch1=farend`，默认保持目录、帧数和容器 header（可选 `--output-sample-rate`/`--output-format`/`--output-subtype` 经 soxr VHQ 重采样并转容器），发布审计 JSONL/summary（不是训练 manifest） |
| `generators/se/generate_offline_dataset.py` | 从 dynamic 配置预生成 SE offline 数据 | canonical module：`python -m tools.prepare_data_tools.generators.se.generate_offline_dataset`；当前 schema v2 publication identity 为 `2.8`，绑定 `dataset` root alias、写入带 alias 的 reload SCP，并在同级 owned staging 中完成事务发布 |
| `generators/se/generate_dns3_offline_dataset.py` | DNS3/GTCRN 论文口径离线生成 | canonical module：`python -m tools.prepare_data_tools.generators.se.generate_dns3_offline_dataset`；train/val/test WAV 与严格四列 SCP；CLI 仅接受 kebab-case 参数 |
| `generators/aec/generate_simulated_testset.py` | AEC 七场景动态仿真评测集 | AEC 尚无 schema v2 reload 支持，GenerationManifest publication 已下线，CLI 显式 fail-closed |
| `generators/aec/generate_ood_stress_set.py` | AEC OOD stress 评测 | AEC 尚无 schema v2 reload 支持，GenerationManifest publication 已下线，CLI 显式 fail-closed |
| `scp/prepare_aec_challenge_offline.py` | AEC Challenge 数据集扫描 | 保留扫描逻辑；GenerationManifest 发布路径同样 fail-closed，待 AEC schema v2 reload 就绪 |
| `generators/se/generate_scene_rirs.py` | 多麦克风 scene-level offline RIR bundle | 只负责 `C>1` 的 spatial scene；生成每个 actor 的 full/direct/early `[C,L]` track 与逐 scene `scenes.jsonl`；`C=1` 仍走 `generate_simulated_rirs` |
| `generators/tse/import_wesep_enrollment.py` | 官方 wesep speaker cue → dynamic TSE enrollment | 只加一层显式 `dataset_id`，把 `cues.yaml` 的 `spk_id` 资源转换为 `TSEOfflineDataset` 可消费的 nested JSON；不改音频内容，`mix_spk_id` fixed-trial 资源不得聚合 |
| `generators/tse/prepare_dynamic_tse_manifests.py` | 从显式单说话人目录清单生成 dynamic_tse 严格资源 | 输入 JSONL 每行必须精确包含 `uid`/`path`/`language`/`dataset_id`/`speaker_id`/`enrollments` |
| `generators/ss/prepare_echoset.py` | EchoSet 严格分离清单 | 按 TIGER EchoSet 配对语义生成 use_train strict separation JSONL |

`GenerationManifest` 发布和 reload 现仅支持 schema v2。每个资源与 gate input 都必须使用显式 root alias，并与对应的 generation-root marker 绑定；schema v1 集合既不能发布也不能 reload，必须重新生成。当前 schema v2 production reload 支持 SE 与 TSE；AEC 的 GenerationManifest publication CLI 在完成 v2 reload 迁移前仍然 fail-closed。

`generate_offline_dataset` 每次非 dry-run 都在输出目录同级创建带 ownership marker 的 staging，并用同级 writer lock 串行发布。只有完整集合才会发布；失败时保留现有成功输出。覆盖要求目标属于同一生成器且不是 symlink。成功重跑会整目录替换，不残留过期 WAV 或 debug 文件。`generation_manifest.json` 的 `counts.requested/published/failed/dropped` 与发布集合一致，成功样本的 UID 从 `utt_00000000` 开始连续编号。

这些 producer 的 JSON、JSONL、SCP、path-list 具有不同字段和 consumer 合同，不能仅通过改扩展名或互相替换路径来转换。

stereo AEC pair 的预对齐示例（路径仅为占位）：

```bash
python -m tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref \
    --input-root /path/to/raw_stereo_pairs \
    --output-root /path/to/gcc_prealigned_pairs \
    --workers 16
```

默认通道为 `ch0=echo`、`ch1=farend`。离线 GCC 只清理固有 delay：constant 延迟采用前补零并裁回原帧数；time-varying 不按 trajectory 逐点物化，而是取按时间排序后的第一可靠窗 `tau0`，对整段 farend 仅做一次固定前补零+裁回，段初对齐但保留段中相对 `tau0` 的真实漂移，交给模型或部署实时 GCC/AEC。低置信或实际补齐量为 0 时 byte copy（未请求转换时）；time-varying 无可靠轨迹点时 fail-closed。显式指定 `--output-sample-rate` / `--output-format`（如 FLAC，扩展名随之替换为 `.flac`）/ `--output-subtype` 时，全部文件（含低置信）解码 → 对齐 → soxr `VHQ` 高保真重采样（soxr 缺失时 fail-closed）→ 重编码，残差校验在输出采样率上执行。

实际变换文件在发布前会重读 staged 编码结果：constant 以 `full_pair`、time-varying 以首个可靠 1s `initial_window` 调用同一 windowed GCC 做 residual gate。`--max-residual-error-ms` 默认 5.0；验收 low-confidence、constant 输出变为 time-varying，或 residual 超差时按 `--on-residual-failure` 处理：`copy`（默认）把该文件改写为不对齐的原始音频（请求转换时仍重采样/重编码）、records 标记 `policy=copy_residual_gate_failure` 并继续，`abort` 则整轮 fail-closed，现有 target 不被替换。每条 record 的 `residual_check` 记录 scope、expected/estimated/error 与 passed；summary 汇总 checked/passed 数以及 residual error 的 max/mean/p50/p95。只有全量成功才发布 `gcc_prealign_records.jsonl` 和完成标记 `gcc_prealign_summary.json`。生成指向输出根的训练 manifest 后，必须把 `aec_ref_echo_fine_gcc_align` 设为 `false`，避免二次 GCC；此时 constant 固有 delay 已清理，少量 time-varying 段中固有 delay 有意保留，仿真随机 delay/jitter 仍会叠加。训练 manifest 未生成前不要修改正式 YAML。

## 全量无切分索引：14 个已准备目标

以下命令使用 `scp/prepare_scp.py` 的单目录模式。每条命令递归索引完整输入目录，不做随机 train/validation 划分；`train`、`val` 等名称只表示源目录本身已经分开。该模式下 `-o` 是**不带 `.scp` 的输出文件前缀**，不是输出目录；脚本会自动追加 `.scp`。因此下列前缀均位于对应数据目标的实际父目录中。

```bash
PREPARE=tools/prepare_data_tools/scp/prepare_scp.py
VB=data/se/voicebank_demand
URGENT=data/shared/urgent26_track1_se_dataset
UNIVERSAL="$URGENT/urgent26_track1_universal_se"
NONBLIND="$URGENT/urgent26_nonblind_with_label"
DNS=data/se/DNS_Challenge_test
AEC=data/aec/AEC_Challenge/datasets

# VoiceBank-Demand：4 个目标
python "$PREPARE" "$VB/clean_trainset_wav" -o "$VB/voicebank_clean_train"
python "$PREPARE" "$VB/noisy_trainset_wav" -o "$VB/voicebank_noisy_train"
python "$PREPARE" "$VB/clean_testset_wav/clean_testset_wav" -o "$VB/voicebank_clean_val"
python "$PREPARE" "$VB/noisy_testset_wav/noisy_testset_wav" -o "$VB/voicebank_noisy_val"

# URGENT universal：4 个目标
python "$PREPARE" "$UNIVERSAL/simulation_train/clean" -o "$UNIVERSAL/urgent26_clean_train"
python "$PREPARE" "$UNIVERSAL/simulation_train/noisy" -o "$UNIVERSAL/urgent26_noisy_train"
python "$PREPARE" "$UNIVERSAL/simulation_validation/clean" -o "$UNIVERSAL/urgent26_clean_val"
python "$PREPARE" "$UNIVERSAL/simulation_validation/noisy" -o "$UNIVERSAL/urgent26_noisy_val"

# URGENT nonblind label：2 个仅评估目标
python "$PREPARE" "$NONBLIND/clean" -o "$NONBLIND/urgent26_clean_nonblind"
python "$PREPARE" "$NONBLIND/noisy" -o "$NONBLIND/urgent26_noisy_nonblind"

# DNS Challenge：3 个 noisy blind-set 目标
python "$PREPARE" "$DNS/V2_V3_Blindset/noisy_blind_testset_v2_challenge_withSNR_16k" -o "$DNS/v2_blindset"
python "$PREPARE" "$DNS/V2_V3_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k" -o "$DNS/v3_blindset"
python "$PREPARE" "$DNS/V5_Blindset/Track1_Headset/noisy" -o "$DNS/v5_blindset"

# AEC official 混合波形清单：1 个资产索引目标
python "$PREPARE" "$AEC/blind_test_set_icassp2023" -o "$AEC/blind_test_set_icassp2023"
```

目标清单合计为 VoiceBank 4 + URGENT universal 4 + URGENT nonblind 2 + DNS 3 + AEC mixed 1 = **14 份 SCP**。AEC mixed 文件包含 `*_enrl.wav`、`*_lpb.wav`、`*_mic.wav` 在内的 **2,400 条资产记录**，仅用于资产索引，不是正式评估所需的配对输入合同。正式评估应使用以下命令生成各 subset 的 `mic.scp`/`lpb.scp`：

```bash
export USE_TRAIN_TEST_DATA_ROOT=/path/to/test-data
python tools/prepare_data_tools/scp/prepare_testset_scps.py \
    --datasets aec_official --aec_subsets all \
    --aec_testset_root data/aec/AEC_Challenge/datasets \
    --output-dir "$USE_TRAIN_TEST_DATA_ROOT"
```

配对索引写入 `$USE_TRAIN_TEST_DATA_ROOT/AEC_Challenge_test/blind_test_set_icassp2023/{doubletalk,farend-singletalk,nearend-singletalk}/`。

## train/val 与 Stage D 发布集合

- **`prepare_scp.py`**：默认 `--val_ratio 0.05 --seed 42`，所有给定资源（speech/noise/RIR/wind/music/`--extra`）各自随机划分为 `train_{prefix}.*` 与 `val_{prefix}.*`。因此“只提供 `--speech`”默认得到 `train_speech_sources.scp` 和 `val_speech_sources.scp`；只有显式传 `--val_ratio 0` 才写单个 `speech_sources.scp`。`--speech_train/--speech_val` 可显式划分 speech；`--follow_links` 控制是否跟随符号链接目录。
- **`prepare_offline.py`**：默认 `val_ratio=0.05, seed=42`，在配对后的 UID 集合上划分，train/validation UID 互斥。
- **`build_real_farend_echo_pair_manifest.py`**：默认 `--val_ratio 0.05 --seed 42`，输出 `<stem>_train.json` / `<stem>_val.json`；`--val_ratio 0` 保持单文件。
- **`merge_farend_echo_pair_manifests.py`**：合并多个 echo pair manifest（去重 + shuffle）。每个条目按所属源 manifest 目录解析后写成**绝对路径**，避免相对条目在新 manifest 位置下被重新解释；输出经共享文件事务写入（staging + fsync + 原子 replace）。
- **验证集消费**：dynamic 模式下 `data.validation`（`inherit_simulation: true`）可覆盖 `speech_scp/noise_scp/music_scp/rir_scp/windnoise_scp`，TSE 还可覆盖 `spk2enroll_json` 并指向对应 validation resource。TSE 设置 `inherit_simulation: false` 时，`data.validation.spk2enroll_json` 必须显式提供；`real_farend_echo_pair_json` 遵循独立 echo JSON 合同。

Stage D 的 strict 记录合同使用 **manifest schema version 6**，生成器身份使用 **generator version 6.1**，两者是不同版本，不可混淆。`build_stage_d_noise_manifests.py --output-dir ...` 默认按 `--val-ratio 0.05 --seed 42` 发布准确的**三件套**：

1. `train_noise_sources.jsonl`
2. `val_noise_sources.jsonl`
3. `noise_publication_metadata.json`

每条 train/val JSONL 内嵌 `category_id`、`category_name` 与 `roles`；生成器在内部计算全局 category vocabulary 并返回 `category_count`。训练 Dataset 只直接消费两份 JSONL。metadata 记录构建 provenance、内容完整性指纹及顶层 `manifest_summary`，不是训练输入；`manifest_summary` 只描述发布规模、role 与 train/val split，不代表 runtime 均衡。训练实际 category/temperature 分布使用 `tools/verify/audit_noise_sampling.py` 审计。`--print-stats` 可把当前 manifest 发布摘要额外打印到 stdout。

单文件模式发布 `<output>.jsonl`、`--category-vocab` 指定的显式 category vocab 和 `<output>.metadata.json`；metadata 包含无 split 的 `manifest_summary`。它是独立发布集合，不属于 split 三件套。

## 分类、采样与 wind/music role

Stage D 生成只发布 category/role 标签、strict 证据和统计，**不执行重采样**。真正的选择发生在训练 Dataset 运行时：

- natural：按普通 source 候选自然选择，不要求 category metadata。
- category：先对 category 等权，再在选中类别内对 UID 均匀选择。
- temperature：在首个可用 SR tier 内按 `P(c) ∝ n_c^alpha × difficulty(c)` 选择类别，再在类内均匀选 UID；`alpha` 默认 `0.5`。

最小配置如下。路径使用 `~/...` 占位，**必须在 runtime 前替换为实际可用资源路径**；dynamic 训练源写在 `data.train`。配置中的 `data/...` 路径相对于项目根目录解析；其他相对配置路径仍沿用调用方原有语义，完整绝对路径继续支持。

```yaml
# natural
data:
  mode: dynamic
  train:
    noise_scp: "~/manifests/train_noise_sources.scp"
```

```yaml
# category 等权 + 类内 UID 均匀
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

分类 JSONL 的 `roles` 若为 list，就是角色拆池的权威值；仅当 `roles` 缺失或不是 list 时，才按 `balance_category`（兼容 `category_name`）回退匹配 `core.data.noise_roles`。

| 来源 | 运行时行为 |
|------|------------|
| Wind | SE/AEC/TSE 都从 ordinary noise 拆入 wind pool，并与显式 `windnoise_scp` 合并 |
| Music | 仅 TSE 拆入 music pool，并与显式 `music_scp` 合并；SE/AEC 仍把它作为 ordinary additive noise |

分类 JSONL 的 canonical `roles` 由消费端直接完成 ordinary/wind/music pool 路由，不再依赖按路径过滤的派生 SCP。

## speech balancing 的当前缺口

`prepare_scp.py` 不生成 speech balancing controller 所需的 `language`、`dataset_id`、`speaker_id` metadata。尤其是 TSE strict speech manifest 仍需外部或专用准备流程；不能把通用严格四列 SCP 或 noise strict publication JSONL 宣称为已支持该 speech 合同。

普通 SCP 的路径字段由 producer 写成相对于各自 manifest 文件目录的路径，因此移动整个数据目录后不需要改清单内部音频路径；配置中引用 manifest 的路径仍需满足配置路径解析规则。DNS Challenge 3 等已验证发布清单中的 `waveforms:` / `dataset:` 是受 generation-root marker 绑定的逻辑 alias，不属于普通 `../` 相对路径，必须保持原样。

## 命名约定（事实标准）

Canonical dynamic source 文件名按资源使用固定前缀：`.scp` 为严格四列 `uid fs length path`，`.jsonl` 为逐行 JSON object；`.json` array 不再是 source manifest。

| 资源 | train | validation | 无划分 |
|------|-------|------------|--------|
| Speech | `train_speech_sources.scp` | `val_speech_sources.scp` | `speech_sources.scp` |
| Noise（普通） | `train_noise_sources.scp` | `val_noise_sources.scp` | `noise_sources.scp` |
| Noise（分类/strict） | `train_noise_sources.jsonl` | `val_noise_sources.jsonl` | `noise_sources.jsonl` |
| RIR | `train_rirs.scp` | `val_rirs.scp` | `rirs.scp` |
| Wind | `train_wind_noise_sources.scp` | `val_wind_noise_sources.scp` | `wind_noise_sources.scp` |
| Music（TSE） | `train_music_sources.scp` | `val_music_sources.scp` | `music_sources.scp` |
| AEC echo/ref | `real_farend_echo_pair_train.json` | `real_farend_echo_pair_val.json` | `real_farend_echo_pair_manifest.json`（metadata 为 `real_farend_echo_pair_metadata.json`） |

两种 dynamic source 序列化形态的来源与内容：

- **`.scp`**：严格四列 `uid fs length path`；`prepare_scp.py` 固定输出。
- **`.jsonl`**：每行一个 JSON object；基础 dynamic 记录要求每条含 `uid/fs/length/path`。分类 noise 由 `build_stage_d_noise_manifests.py` 发布（`train/val_noise_sources.jsonl`）；RIR/wind/music/speech 的 `.jsonl` 形态保留给专用 producer 或自定义流程。

`.list`（每行一个 path）不是 dynamic source manifest；它仅供明确支持 plain list 的 TSE offline/wesep consumer 使用。

- 批处理资源 → 文件名映射硬编码在 `prepare_scp.py` 的 `BATCH_KEY_TO_PREFIX`：`speech_sources / train_speech_sources / val_speech_sources / noise_sources / rirs / wind_noise_sources / music_sources`（启用随机划分时再加 `train_`/`val_` 前缀）。训练 YAML 通过 `data.train.*_scp` 引用这些资源，`music_scp` 也属于 `data.train` / `data.validation` 的 split 级 resource；`--music` 会在 YAML 提示中生成对应字段。
- Offline 输出遵循对应 producer 和 Dataset 的布局：`prepare_offline` 在 `train/validation` 下使用严格四列 `spk1.scp/wav.scp`；`prepare_voicebank`/`generate_dns3_offline_dataset` 在 `train/test` 或 `train/val` 下使用严格四列 `clean.scp/noisy.scp`。offline training metadata 始终内嵌于各轨 SCP。

## 正式入口

- `python tools/prepare_data_tools/scp/prepare_offline.py`
- `python tools/prepare_data_tools/scp/prepare_testset_scps.py --datasets aec_sim`
- `python -m tools.prepare_data_tools.generators.noise.generate_wind_noise`
- `_common.py` 是 `prepare_data_tools` 脚本使用的内部共享模块，**不承诺 API 稳定**；跨目录复用请直接 import 对应的 `tools.shared.*` 模块。

## 测试

```bash
python tools/prepare_data_tools/inspect_manifest.py --help
python -m pytest tests/test_prepare_data_manifest_contracts.py -v
python -m pytest tests/test_prepare_scp.py tests/test_prepare_offline_voicebank.py \
    tests/test_farend_echo_manifests.py -v
```
