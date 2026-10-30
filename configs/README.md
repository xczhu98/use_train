# 🗂 训练配置总览 (`configs/`)

本目录存放所有训练 / 推理 YAML 配置，按任务域组织：

任务源码使用 `tasks.<task>` 命名空间（如 `tasks.se.trainer.pase_gan_trainer.PASEGANTrainer`）；配置目录仍为 `configs/<task>/`，仓库内 YAML 均为独立完整文件。

**第一次来这里？** 按下面的顺序读：

1. 在「任务总览」里找到你的任务，进对应 `configs/<task>/` 挑一份配置；
2. 跑 `python train.py -C configs/<task>/<配置文件名>.yaml`，产物写入该配置的 `trainer.exp_path`（见「exp_path 约定」）；
3. 遇到报错或要改仿真/损失/指标，再回到本文件查「命名规范」之后的合同条目；
4. 仿真层语义不在本文件重复维护，以 [`docs/guides/simulation/Simulation_Flow_Logic.md`](../docs/guides/simulation/Simulation_Flow_Logic.md) 为准。

术语速查：

| 术语 | 一句话解释 |
|---|---|
| `data.mode` | 配置的必填路由字段，决定由哪个任务和 Dataset builder 读这份配置；全仓共九种合法取值 |
| dynamic / offline | dynamic 在训练时在线合成带噪（或带回声/混响）输入；offline 直接读磁盘上已物化的配对数据 |
| Layer A–F | dynamic 仿真的六个处理层：A 语音预处理、B 场景分支、C 声学混合、D post-mix 幅度调理、E 信道退化、F 设备/观测退化 |
| `simulation_cfg` | 承载 Layer A–F 参数的配置区；字段归属由 ownership 合同约束 |
| SCP | 空格分列的清单文件，常见四列 `uid fs length path` |
| generator version | 离线数据生成器的版本号，改变生成语义就必须递增，否则旧 manifest 指纹不匹配 |
| STFT / iSTFT | 短时傅里叶变换及其逆变换，模型看频谱、写回波形的两端 |
| DDP | 多卡数据并行训练；PIT（置换不变训练）是分离任务用来对齐多路输出的训练目标 |

### 动态仿真配置分层结构（方案 B）

新的动态仿真配置可以在 `simulation_cfg` 下使用分层结构，按 Layer A-F 的执行顺序
组织字段：`speech`（A）、`scene`（B）、`acoustic`（C，包含 `rir` / `background`）、
`post_mix`（D）、`channel`（E）、`device`（F），最后是跨层的 `spatial` 和任务专属
section。配置入口会将它们一次投影到内部的 canonical `SimulationConfigs`
属性，Dataset、renderer 和五个任务不再各自解释 YAML 层级。

```yaml
simulation_cfg:
  acoustic:
    rir:
      backend: online
      reverberation_prob: 1.0
    background:
      snr_low: -5.0
      snr_high: 15.0
  post_mix:
    normalization:
      mode: off
  channel:
    enabled: false
  device:
    enabled: false
  spatial:
    enabled: true
    geometry:
      profile: linear_2ch_80mm
      reference_mic: 0
    coherence:
      policy: habets_spherical
    background:
      point_source_prob: 0.3
      wind_background_prob: 0.0
      wind_sidechain_prob: 0.0
```

分层配置不能与 flat 键混用；重复映射和未知分层键都会在配置边界 fail-fast。flat
mapping 仅是归一化后的内部 runtime 表示，不是旧 YAML 配置的兼容入口。示例中的
`spatial.background.point_source_prob`、`wind_background_prob`、`wind_sidechain_prob`
分别映射到 canonical 平面键 `prob_point_source_background`、
`prob_wind_noise_background`、`prob_wind_noise_sidechain`；文档和审计均以 canonical
键为准。现役
SE、AEC、TSE、SS、SSL dynamic 配方均已按 Layer A-F 迁移；任务专属 Dataset 参数
分别放在 `se`、`aec`、`tse` section，并在 builder 边界提升到任务构造器。迁移期间不要同时声明
`spatial.enabled` 与 `spatial_enabled`，也不要把 `prob_point_source_background`
重复写入 `spatial.background` 和其他 section。

SSL 配置仅保留 moving dynamic seam 并纳入当前 runtime。`configs/ssl/` 有五份主配置：`fixed_array_ipdnet_dynamic_moving.yaml` 是 native 配方与 base（模型 `FixedArrayIPDNet`，registry 实名，GRU 骨干、仅消费 `reference_ipd`），`fnssl_dynamic.yaml` / `ipdnet_fixed_dynamic.yaml` / `ipdnet_variable_dynamic.yaml` / `ipdnet2_dynamic.yaml` 四份是官方模型合同模板（header 自述），已按各官方结构填好超参，可通过 `_validate_config` 并完成模型构建与合同 smoke（FNSSL/IPDNetFixed/IPDNetVariable 有 CPU smoke；IPDNet2 仅 CPU 构造/合同边界，GPU 前向依赖 CUDA，具备 CUDA 与 `mamba_ssm` 时运行，无 CUDA 时由 marker 跳过）。本机清单由 `tools/prepare_data_tools/generators/ssl/prepare_manifests.py` 生成到 `data/ssl/speech/` 与 `data/ssl/noise/`，Noise92 下载器生成完整 15 类 `data/ssl/noise92/noise92.scp`；LOCATA/RealMAN 下载整理由 `tools/prepare_data_tools/generators/ssl/download_real_datasets.py` 负责，数据本体由 `.gitignore` 隔离。`ssl.forgetting_norm_sample_length` 的通用默认/其他 profile 为 298，`ipdnet2_dynamic.yaml` 官方模板显式设为 249；offline IPDnet 的 `ssl.normalization: global_mean` 对整段 `[C,F,T]` observation 幅度取一个全局均值，direct target 不归一化；实际 mode、aggregation 与 sample length 写入 item 的 `provenance.normalization`，模型内 IPDNet2 同名参数设为 `null` 以避免双重归一化。SSL 的 `sound_speed` 从 `simulation_cfg.gpurir_sound_speed_mps` 贯穿 projection/mapper，并写入 `provenance.sound_speed_mps`；near-field 候选原点是 geometry 的 `reference_channel` 麦克风。由 `tests/test_ssl_plan_contracts.py` 与全配置发现门禁覆盖；这些构建和 smoke 只证明合同与结构接入，不等于官方指标复现。`meta.ssl_profile` 只在已登记 official profile 时提升为 recipe，其他 marker 不会自动转换；真实数据评估使用 `tools/verify/ssl/real_eval.py`，无严格 checkpoint 时直接 fail-fast。

## 任务总览

| 目录 | 任务 | 说明 |
|------|------|------|
| `aec/` | 回声消除（Acoustic Echo Cancellation） | Microsoft AEC Challenge，`dynamic_aec` 仿真 / `offline_aec` canonical 四列 `mic/farend/nearend[/error].scp` |
| `se/` | 语音增强（Speech Enhancement） | GAN / URGENT2026 / 轻量实时 / PASE 系列 / Voicebank / DNS3 / spatial（NBSS 多通道） |
| `tse/` | 目标说话人提取（Target Speaker Extraction） | `dynamic_tse` / `offline_tse` 都产出 canonical dict item，经 collate 形成 validated `TSEBatch`；保留 `TSE_BSRNN_SPK` 及既有 causal streaming 路径，WeSep frame-embedding recipe 已提供 `TSE_Modular` 的 tfmap/context latent 与 block 配置；双 self-estimated 已迁移到 TSETrainer objective；`textual/` 为 DAE-TSE keyword cue 训练配置；`spatial/` 为 MC-LibriMix 4 通道 DOA cue 训练配置 |
| `ss/` | 语音分离（Speech Separation） | `data.mode: dynamic_ss` / `offline_ss`，固定 source count、PIT、TDANet 和 TIGER external-STFT，另有 spatial 多通道配方 |
| `ssl/` | 空间声学定位（Spatial Sound Localization） | `data.mode: dynamic_ssl`，固定阵列 DP-IPD/DOA/VAD 合同；native `FixedArrayIPDNet` 配方 + FNSSL/IPDnet fixed/variable/IPDnet2 官方结构（四份为合同模板，结构已实现，模板不代表官方数据或指标复现）；moving dynamic 仿真 |

## 目录结构

```
configs/
├── README.md                    本文件（总览 + 命名规范 + exp_path 约定）
├── aec/
│   ├── offline_aec_template.yaml offline_aec copy-and-fill 模板（canonical mic/farend/nearend[/error] SCP）
│   └── microsoft_challenge/      AEC 挑战配置（cagcrn/、dbgnet/、deepvqe/、localvqe/ 按模型分目录）
├── se/
│   ├── README.md / README_CN.md SE 任务索引
│   ├── urgent2026/               URGENT2026（BSRNN / DPDFNet2）+ pase/（生成式增强）+ gan/（MetricGAN）子目录
│   ├── dns_challenge3/           DNS Challenge 3（GTCRN）
│   ├── spatial/                  8 kHz SpatialNet / OnlineSpatialNet dynamic 配方、cIRM 配方与 offline spatial mc_sim 样板（含 32 s LF 阶段）
│   └── voicebank_demand/         VoiceBank-DEMAND（十份 recipe：GTCRN / ZipEnhancer-S / FastULCNet / MeanFlowSE / MeanSE / DCCRN / MTFAANet / RT-SEMamba / TFSkiMNet）
├── tse/
│   ├── dynamic/                  dynamic_tse 训练与官方 BSRNN 四变体
│   ├── offline/                  offline_tse / WeSep 参数对齐 baseline
│   ├── textual/                  TSE_Modular + textual（DAE-TSE keyword cue）训练配置
│   ├── spatial/                  TSE_Modular + spatial（DOA 方向 cue，MC-LibriMix 4 通道）训练配置
│   └── inference/                TSE 纯推理配置
├── ss/
│   ├── tdanet/                    TDANet waveform-native separation
│   ├── tiger/                     TIGER external-STFT separation
│   └── spatial/                   OnlineSpatialNet 多通道分离配方（mhsa / ret / mamba）
└── ssl/                           dynamic_ssl 固定阵列配置（moving）：fixed_array_ipdnet_dynamic_moving.yaml native 配方/base，fnssl/ipdnet_fixed/ipdnet_variable/ipdnet2_dynamic.yaml 四份官方模型合同模板，spatial/ 为空间背景实验配方
```

## 共享动态仿真

以下四节（共享动态仿真、Ownership 结构、RIR 资源、物理阵列 profile）属于进阶参考：只有要改仿真资源、字段归属或空间阵列时才需要读；只想跑通一份配置的话可以跳过，直接看「运行入口与 checkpoint」「exp_path 约定」。

SE、AEC、TSE、SS、SSL dynamic 配方均复用 Layer A 画布；TSE/SS/SSL 多源相对窗口使用 `prob_timeline`，并可用 `timeline_mode_weights` 在 `head` / `tail` / `middle` / `crossing` / `containment` 之间配置完整概率分布（默认五者各 `0.2`）。SE/AEC 的 timeline 配置会在 contract 边界 fail-fast，AEC ref/echo delay 不属于该能力。音乐源选择也已共享到 core。各动态配置保留共享参数示例；启用前核对资源与任务含义，TSE 使用七路互斥场景与 `num_interferers`，trace 的 canonical `scene_name` 统一使用带 `prob_` 前缀的名称；SS 使用 `prob_target_only` / `prob_target_noise` / `prob_noise_only` 三路 display name，纯 background 保持固定 K target 合同，旧配方字段直接拒绝。完整口径与迁移见 [多源共享仿真](../docs/guides/simulation/Multi_Source_Simulation.md)。

## Ownership 结构

动态配置按以下边界组织，不能把所有字段都视为 Layer A-F 参数：

| 区域 | 负责内容 | 示例 |
|---|---|---|
| `data.train` / `data.validation` resource | 数据资源路径 | `speech_scp`、`noise_scp`、`music_scp`、`windnoise_scp`、`rir_scp`、AEC echo pair、TSE `spk2enroll_json` |
| `data.train` / `data.validation` lifecycle | 过滤、采样、画布和加载生命周期；runtime 归一为 core `CanvasSpec` | `filter_*`、`speech_length_file`、`sampling`、`max_duration_seconds`、`repeat_*`、`strict_load` |
| `simulation_cfg` Layer A-B | 语音预处理、场景抽样和多源时间关系 | `prob_speed_perturb` / `speed_rate_range`（语速/音高）、pitch/EQ、`prob_*` scene、`prob_timeline`、`timeline_mode_weights` |
| `simulation_cfg` Layer C | RIR、音量、背景、AEC echo coupling、spatial scene target | `rir_backend`、`target_type`（`early` / `dry` / SE spatial `rts`）、`rts_target_t60_sec`、`prob_moving_rir`、`moving_source_speed_range_mps`、`source_volume_*`、`snr_*`、`music_ratio`、AEC SER/ERL |
| `simulation_cfg` spatial | 多通道任务共享的麦克风拓扑、reference、阵列增广与空间背景 | `spatial_enabled`、`array_profile` / `custom_geometry`、`reference_channel`、`spatial_diffuse_coherence_model`、`prob_array_rotation`、`prob_microphone_position_jitter`、`prob_point_source_background`、`prob_wind_noise_background` |
| `simulation_cfg` Layer D-F | post-mix amplitude、channel/device observation degradation | `peak_normalization_mode`、`prob_channel_enabled`、`noisy_post_mix_degradations_enabled` |
| task-local / trainer | enrollment branch、AEC frontend、模型和 loss | `enrollment_augment`、`aec_frontend`、`model`、`loss` |

`speed_rate_range` 是 Layer A 语速扰动；`moving_source_speed_range_mps` 是空间声源运动速度，两者不可互换。`music_scp` / `rir_scp` / TSE `spk2enroll_json` 是 split-level resource；`music_ratio` / `rir_backend` 是 simulation policy。SE offline spatial split 另须在 split-level 给出 `spatial_enabled`、geometry、`reference_channel` 与 `channel_order`，noisy WAV 头必须与 geometry 通道数匹配。Dataset lifecycle 字段不再从 `simulation_cfg` 隐式提升，错误位置会在 strict builder 边界 fail-fast。

## RIR 资源

动态仿真按采样率选择公共 RIR 资源：SE、AEC、SS 使用
`data/shared/rirs/gpurir_static/gpurir/fs{采样率}/{train,val}_rirs.scp`。
16 kHz 清单由 DNS5 RIR 与静态 gpuRIR 合并，32 kHz 清单来自静态 gpuRIR。

下表为 `gpurir_static` 发布清单：

| 采样率 | train | val | 来源 |
|---|---:|---:|---|
| 16 kHz | 152236 | 8012 | DNS5 RIR + 静态 gpuRIR 合并 |
| 32 kHz | 95000 | 5000 | 静态 gpuRIR |

TSE mixture 是明确例外：manifest 使用 `fs16000/tse/{train,val}_rirs.scp`，
只包含 `fs16000/static` 的 100,000 个纯 gpuRIR PKL（train 95,000 / val 5,000，
near+far）。enrollment 独立使用 `fs16000/static/near/{train,val}_rirs.scp`，
分别为 47,551 / 2,449 条。PKL 中的 `room_dim`、`mic_loc`、`source_loc`、
`dist`、`rt60`、`field` 和 `uid` 用于保留几何、距离、声场与资产身份审计；
不是依靠文件名猜测房间分组。在线模式不读取 mixture SCP、不隐式回退到
manifest；所有在线参数均由 core 解析，在 YAML 的 Layer C-0 中以 `gpurir_*`
平面字段配置。

enrollment 近场契约：注册音是近距离录制（物理上 5-20cm），RIR 扰动必须保持
这一事实。`enrollment_augment.rir_backend` 两个取值：`offline`（默认）从
`rir_scp` 指定的近场池抽取（现网 `static/near` 池 dist 实测约 0.2-0.8m，
由池选择保证近场语义，不要混用 mixture 的 near+far 池）；
`online` 在线生成单源近场 RIR，声源距离取
`gpurir_enrollment_distance_range_m`（默认 `[0.05, 0.2]`），房间/RT60/麦克风
高度等几何参数与 mixture 共用同一组 `gpurir_*` 字段；在线模式不接受 `rir_scp`。

`gpurir_*` 完整字段清单（默认值统一由 `core/data/simulation/rir_config.py` 提供；五任务
动态 YAML 只保留会改变实验语义的字段作为显式覆盖入口，稳定的服务默认值不重复展开）：房间
`gpurir_room_dim_range_m`（顺序为 `[length, width, height]`，每一维是一个 `[lower, upper]`
区间），衰减
`gpurir_rt60_range_sec`，声源距离 `gpurir_source_distance_range_m`，声源/麦克风
高度 `gpurir_source_height_range_m` / `gpurir_mic_height_range_m`，声源最小间距
`gpurir_source_min_separation_m`，墙边距 `gpurir_wall_margin_m`，声速
`gpurir_sound_speed_mps`（默认 **343.0 m/s**），漫射过渡 `gpurir_diffuse_transition_db`，在线源预算
`gpurir_max_sources`（≤16，TSE/SS/SSL 多源或 source-track 数受其约束），服务参数
`gpurir_timeout_sec` / `gpurir_queue_size` / `gpurir_cache_size` /
`gpurir_device`。`prob_moving_rir` 是在线 moving RIR 概率：spatial scene（五任务动态空间路径）与
legacy mono（`spatial_enabled=false` 的动态仿真）都按逐样本 Bernoulli 消费，只要求
`rir_backend=online`。mono 路径下 moving 仅在抽中 `prob_reverberation` 后生效，
实际 P(moving) = `prob_reverberation × prob_moving_rir`；spatial scene 强制 reverb。
TSE/SS 的 mono 多说话人 scene 抽中 moving 后只采样一次 RIR bank：全部 speech actor
（以及存在时的 latent SIR reference）共享 room、mono microphone position 和 timebase，
每个 actor 保持独立轨迹；单源 SE/AEC 沿用单 actor 在线 moving 路径。
离线 moving 资产不参与该字段：mono 由 moving PKL 池子组成决定，spatial 由 schema v3
moving scene manifest（`generate_scene_rirs.py --moving` 生成）决定，运行时按 manifest
权威渲染。`segment_sec` 只用于离线 moving RIR 分段卷积，不是在线 moving 的参数。
仅 TSE enrollment 在线近场生成使用的
`gpurir_enrollment_distance_range_m`。

在线房间、麦克风、source positions、RT60 和共同 beta 由 core 一次生成；无需扫描 PKL 去猜 room group。离线 WAV/PKL 只需满足既有 RIR 加载合同，不能把离线独立读取宣称为同房间。完整使用说明见 [Online_gpuRIR.md](../docs/guides/simulation/Online_gpuRIR.md)。

Spatial Dataset owner 统一通过 `core.data.simulation.rir_provider.build_spatial_rir_provider()` 解析当前 split 的 `rir_scp` 或在线 backend；SE/AEC/TSE/SS/SSL 不再自行拼装 provider。`simulation_cfg.rir_backend=offline` 时，`data.train.rir_scp` / `data.validation.rir_scp` 是预生成 RIR 资产；`spatial_enabled=true` 时该 manifest 必须提供与 resolved microphone geometry 一致的 `[K,C,L]` source-microphone RIR（moving scene 为 schema v3 manifest，track 布局 `[P,A,C,L]` 并携带逐 actor 轨迹与 timebase），不能把普通 mono RIR 静默广播为空间 RIR。`simulation_cfg.rir_backend=online` 时不读取 RIR SCP，由在线 provider 生成 RIR。显式 `spatial_rir_provider` 仅保留为 Base Dataset 的 provider seam，也必须通过相同的 normalize、source-order、track 与 provenance 校验。

Formal `dynamic_ssl` 不配置 split-level `channels`；C 由 `simulation_cfg.array_profile` 或 `custom_geometry` 推导，`model.args.input_size` 按具体模型的特征合同校验。loss 默认按模型类型区分：native `FixedArrayIPDNet` 使用 `SSLSpatialMSELoss`，official 模型（`FNSSL`/`IPDNetFixed`/`IPDNetVariable`/`IPDNet2`）使用 `SSLDPIPDPITLoss`；均可显式选择 `SSLDOACompositeLoss`，启用辅助项须由模型声明并输出对应 head。`trainer.val_metrics.enabled` 必须为 `false`：IPD feature/target 不是 waveform，不能用于 PESQ/SI-SDR 等音频指标。SSL 只支持 `strict_load: true`，资源失败直接报错，不生成缺少可信空间监督的 placeholder。train/validation 的场景 seed 来自 `meta.seed`，validation 固定 epoch 0。

`ssl.num_tracks` 可统一声明正整数声源数；`data.{train,validation}.{tracks,max_track}` 和
`model.args.max_track` 同时提供时必须一致，省略统一字段时从现有声明解析。
训练与真实录音评估共用 `tasks.ssl.config.resolve_ssl_num_tracks`；loss `max_tracks`
是容量上限，必须覆盖实际 K，该字段显式 `null` 不合法。四份官方模型模板的
`loss.args: null` 则表示清空父模板 native loss 的参数映射，使用 PIT 默认容量 8、
DOA 权重 0；它不会改变模型或 Dataset 的实际 K，也不会把 `allow_no_stft` 传给 PIT。
Dataset 特征维度从实际阵列 C 推导：多通道实虚 STFT 为 `2C`，reference IPD 为
`2(C-1)`；它与 Fixed 的模型输入参数 `2C`、Variable 的固定参数 4 分别校验。
`ssl.vad_mode` 当前只接受
`direct_path`（默认），由每路 direct-path STFT 的相对能量生成 VAD，并记录
`provenance.vad_supervision`。参数示例与组合损失 head 合同见
[SSL 任务指南](../tasks/ssl/README.md#组合监督与离线分块推理)。

SS 使用固定 K 路源合同，`source_mask` 必须全 True；`source_ids` 可全批省略，提供时每项必须为非空字符串。item 的 `sr`/`length` 为正整数，不接受浮点截断或字符串转换。offline SS 随机裁切通过共享 epoch/seed 状态支持 persistent workers；dynamic validation 可使用共享 `inherit_simulation` 配置。

训练 sampler 的非空门禁在 `drop_last` 后检查各 rank 的真实 batch 数；数据不足时在训练前报错，而非保存零次更新的模型。

`rir_backend=online` 时 split-level `rir_scp` 必须缺失或为 `null`，避免未消费资源静默存在；`rir_backend=offline` 且 `spatial_enabled=true` 时必须提供 split-level `rir_scp`。

`offline_aec` / `offline_tse` 的 `spatial_enabled` 是 split-level reader policy，因为 observation 已经物化，不执行 Dynamic RIR 仿真；禁止把该键重复写入 `simulation_cfg`。这类 offline reader 若没有 geometry metadata，模型构造边界仍要求 native microphone-axis contract，但不会猜测 C 或 reference channel。

### 物理阵列 profile 与通道边界

`array_profile` 只描述任务无关的物理 microphone topology。当前 canonical profile 为
`linear_2ch_80mm`、`linear_4ch_40mm`、`linear_6ch_40mm` 和 `circular_8ch_r80mm`
（8 mic 圆阵、半径 80 mm，`spacing_m` 记录相邻弦长）；名称不携带任务、模型或外部仓库语义。需要非标准阵列时使用 `custom_geometry`，提供 finite 的 `[C,3]` positions、`cartesian_xyz_m` 坐标系和独立的 `reference_channel`；profile 与 custom geometry 不能同时配置。geometry provenance 可记录外部 repository/path/commit，但不能替代解析后的 positions，也不能把 `FN-SSL` / `NBSS` 当作 profile 名称。

空间配置只扩展主 microphone observation 为 `[C,T]`，waveform target/output 保持 `[T]`；AEC `farend` 与 TSE `enrollment` 始终为单通道 `[T]`。`C=1` 继续走 legacy mono 路径。Layer F 是 Dataset 内 CPU-only 的 observation/device degradation，关闭 `noisy_post_mix_degradations_enabled` 时整层跳过；训练侧不得另设 GPU Layer F。

`prob_point_source_background`（默认 0）把空间 C-1 背景按概率从 `diffuse_coherence` 升级为同 room 点源 RIR：噪声 actor 位置按 speech 源相同的分布族采样，moving scene 共享 timebase，解析与注入由 `core/data/simulation/background_resolver.py` 完成。`prob_wind_noise_background`（默认 0）把 wind 资产作为可分解的空间加性背景，复用同一 `BackgroundSpec` diffuse / point-source 链；`prob_wind_noise_sidechain` 则是非加性 compressor，第一阶段只在 SSL/DOA spatial path 开放，并通过 `NonDecomposableObservationContract` 记录不可分解 observation。两项均为多通道空间配置；点源项要求 `spatial_enabled=true` 且 `rir_backend=online`，wind background/sidechain 要求 `spatial_enabled=true`，sidechain 还必须提供 `windnoise_scp` 或可路由的 wind role。SE/AEC/TSE/SS 仍在各自真实训练 smoke 通过前拒绝空间 sidechain；为 0 时零随机数消耗、行为与默认背景完全一致。

### 配置作用域与互斥规则

- `[共享][空间多通道]`：`spatial_enabled`、`array_profile` / `custom_geometry`、`reference_channel`、`spatial_diffuse_coherence_model`、`prob_array_rotation`、`prob_microphone_position_jitter`、`prob_point_source_background`、`prob_wind_noise_background`。这些键由 `SimulationConfigs` 统一校验，不能在任务配置中重复声明不同语义。
- `[共享][空间与单通道均可]`：普通 `prob_*` 场景、RIR、SNR、`noise_mix_prob`、Layer E/F 增广。空间路径会把背景投影为显式 `[C,T]`，不能使用旧的 mono spatial fallback。
- `[AEC 独有]`：`prob_echo_only`、`prob_noise_echo_only`、`prob_target_echo`、`prob_target_noise_echo`、`ser_range`、`far_db_range`、`erl_range` 等 echo/ERL 字段。
- `[TSE 独有]`：`prob_interference_only`、`prob_interference_noise_only`、`prob_target_interference`、`prob_target_interference_noise`、`num_interferers` 以及 enrollment/cue 字段。
- `[SE spatial 独有]`：`target_type: rts` 只允许 SE spatial；其他任务必须使用 `early` 或 `dry`。
- `[SSL 独有]`：SSL 的帧特征、DOA/VAD 和 loss 入口由 `configs/ssl/` 与 `tasks/ssl/` 管理，不重复写入 shared scene 概率字段。
- `[SSL spatial 实验]`：`prob_wind_noise_sidechain` 通过 `NonDecomposableObservationContract` 走严格 `[C,T]` control image；必须提供 split-level `windnoise_scp`，其他任务的 spatial sidechain 暂时拒绝。
- `[SE spatial]`：`prob_colored_noise`（C-3 彩色噪声）已接入 spatial 路径——合成 colored waveform 作为 noise actor，复用与 wind background 相同的 `BackgroundSpec` 渲染链（diffuse / point_source 由每样本 `prob_point_source_background` 门统一决定），在 `scale_noise_image` 唯一标定点按 spatial SNR 标定；mono 路径实现不变，spatial 任务开放清单见下方空间能力矩阵。
- **空间能力矩阵**：spatial 背景能力（彩色噪声 / wind sidechain / rts target）的任务开放清单以 `core/data/simulation/capabilities.py` 的 `SPATIAL_FEATURE_CAPABILITIES` 为单一事实源；capability 门禁（`validate_spatial_background_capability`）、配置期校验（`tests/test_configs.py` 逐配置执行）与本文档消费同一张表。新增 spatial 背景能力须完成接入 checklist（target protection / facts-audit / 测试三件套）后在矩阵行追加任务。
- `configs/ssl/spatial/fixed_array_ipdnet_dynamic_wind.yaml` 提供真实 speech/noise/wind SCP 的空间加性 wind background 配方：在 moving 基线之上叠加嵌套键 `spatial.background.wind_background_prob=0.25`（canonical：`prob_wind_noise_background`），同时保留 `spatial.background.point_source_prob=0.3`（canonical：`prob_point_source_background`；两条独立伯努利门可共现命中）；不启用 `prob_wind_noise_sidechain`，即不隐式打开 non-decomposable sidechain。
- `prob_point_source_background`、`prob_wind_noise_background` 与 `prob_wind_noise_sidechain` 是不同语义，不能用同义键重复表达；旧键 `prob_background_point_source` 已显式拒绝。
- 完整分布键不能通过重复声明或局部追加表达：`timeline_mode_weights`、`num_interferers`、`num_augmentations`、`clip_variant_probs` 以及同一任务的 scene branch 概率必须一次给出完整分布；每份发布配置都要直接写出完整 list。

AEC/TSE 目前没有 spatial 配置样例（原通用的 `spatial/spatial_smoke.yaml` 合同样例已删除）；
SE 与 SS 的 `spatial/` 目录提供可运行的配方，见下一段。Dataset 可以交付 `[C,T]` observation，
但模型必须声明 native microphone-axis contract；mono 模型收到 `C>1` 时直接失败。SS 的 `K` 路
target 和 TSE 的 target/enrollment 仍为 `[T]`；训练前必须提供真实 manifest/provider。

AEC/TSE 不添加伪可运行的 spatial YAML：当前 AEC registry 中模型仍未声明 native microphone-axis
contract，TSE 的 spatial frontend 虽已实现，但动态 spatial cue 尚无本机真实 manifest 闭环。待对应
模型 I/O 和真实数据 smoke 通过后，再分别加入 `configs/aec/spatial/`、`configs/tse/spatial/`；
配置边界会继续拒绝把 mono 模型静默广播成多通道。

配置头中的复杂度统一按独立 profile 口径记录：MACs 使用 THOP 并披露 partial/unverified 覆盖范围，
FLOPs 使用 `torch.utils.flop_counter.FlopCounterMode` 独立测量，二者均以 B=1、1 秒输入和配置采样率
为基准，不从 MACs 乘系数推导。无法构造完整 artifact 链的 PASE/UniPASE 配置继续保留未测量说明，
避免把论文参考值写成当前配置实测值。

SE 与 SS 的 `spatial/` 目录同时提供 NBSS SpatialNet / OnlineSpatialNet 论文对齐配方：
`se/spatial/spatialnet_dynamic.yaml`（离线对照）、
`se/spatial/online_spatialnet_dynamic_{mhsa,ret,mamba}.yaml` 与
`ss/spatial/online_spatialnet_dynamic_{mhsa,ret,mamba}.yaml`。它们按论文口径使用 8 kHz、
`n_fft=256/hop=128/win=256`、6 通道 CHiME-3 拓扑阵列（`simulation_cfg.custom_geometry`，
坐标取自上游 `chime3_array_geometry`）、4 s 片段、在线 gpuRIR 与 `prob_moving_rir: 0.5`；
本地 16 kHz 清单由 Dataset 重采样到 8 kHz。标准 SE 四份配方的训练部分（SCP 清单、Layer A–F 全字段、
loss、dataloader、val_metrics、distributed）遵循空间仿真的统一合同，配置头记录各模型的
8 kHz/4 s、online RIR + moving、splice 关闭、动态空间 validation、论文 scheduler/epochs/clip、
batch=4 等口径；SS 三份配方沿用 `ss/tdanet/tdanet_dynamic_sisdr.yaml` 的 SS 配方。SE spatial
目录中的 ret/mamba 变体均已展开为独立完整配置，只在模型时序模块与优化器上采用各自口径；配置头记录了论文对照的
Params/MACs/FLOPs，复核脚本为 `python tools/verify/verify_nbss_paper_alignment.py`
（默认复核全部 7 份配置并返回非零码表示超差）。

论文 LF 长片段配方有三份标准 32 s 配置：`se/spatial/online_spatialnet_dynamic_{mhsa,ret,mamba}_lf32s.yaml`。
它们按对应 ST 配置口径完整声明，把训练片段改为 32 s、batch 2、epochs 5（Fig. 2 图例口径；正文的 100e
口径需改 `trainer.epochs` 并用独立 exp_path 双轨记录）。另有
`se/spatial/online_spatialnet_dynamic_mhsa_lf10s_36e.yaml`，当前实验画布为 10 s、batch 4（累积步数 1，
全局有效 batch 仍为 32），
改用 8 麦克风、半径 80 mm 的 `circular_8ch_r80mm` 圆阵，单独使用 36 epochs 与配置 stem 对应的实验目录，
并关闭 `model.args.gradient_checkpointing` 以避免逐层激活重计算拖慢训练。分支边界见各配置头：
ret 的逐帧 recurrent 与 mamba 的 CPU 参考 step 在 32 s 下很慢，建议在 GPU 上跑并先试跑。
标准 LF 阶段从头训练，不加载 ST checkpoint：

```bash
python train.py --config configs/se/spatial/online_spatialnet_dynamic_mhsa_lf32s.yaml
```

36 epoch 的 8 麦圆阵配置同样从头训练，不加载 6 麦或其他 checkpoint；当前关闭
`model.args.gradient_checkpointing`，避免长序列逐层重算拖慢训练。

RTF 由 `python tools/model_tools/audit_metrics.py --config <配置> --rtf_duration 4 --rtf_repeats 3`
测量，是**整段/批**口径（waveform→STFT→模型→iSTFT 一次前向），论文未报告 RTF 因此无法对比；
OnlineSpatialNet 的逐帧 `forward_streaming` 是流式部署口径，CPU 上因单帧算子调用密集而明显更慢
（4 s/8 kHz：整段 3,815 次 aten 调用 vs 逐帧累计 989,853 次），mamba 分支在 CPU 走纯 PyTorch
参考 step、不代表 CUDA 内核成本。

两份 48 kHz PostNet 配置（UniPASE Stage 4、GAP-URGENet Phase 2）沿用原有的 16 kHz RIR 资产与 runtime 48 kHz 重采样策略，现已切换到新 16 kHz 清单；当前未提供原生 48 kHz 清单。其余 16/32 kHz 配置严格匹配对应资源。

## 命名规范

配置文件名遵循 `<模型>_<变体>_<关键超参>_<loss 配方>[_<附加项>].yaml` 的模板，各任务域细化如下：

| 任务域 | 模板 | 示例 |
|--------|------|------|
| AEC | `<model>[_<variant>]_<sample_rate>k_<frame>ms_delay<delay>ms_loss_<loss>[_<scenario>]` | `deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml` |
| SE | `<model>[_<variant>]_<causal\|noncausal>_<mode>_<loss>` | `usr_bsrnn_causal_urgent26_offline_hybrid.yaml` |
| SE-GAN | `<model>_<trainer>_dynamic` | `mambattention_metricgan_dynamic.yaml` |
| TSE | `<model>_<cue>[_<variant>]` | `tse_bsrnn_spk.yaml`（通过文件顶部注释块手工切换 cue/causal） |

约定：

- **loss 配方一律用小写缩写**，禁止直接使用 loss 类名（如 `CompositeLoss`）。对照表：

| 缩写 | 含义 |
|------|------|
| `mse_sisnr_l1` | 显式 CompositeLoss：`MSELoss` + `SISNRLoss` + `L1WavLoss` |
| `hybrid_fwsdr` | 显式 CompositeLoss：`GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss` + `FWNegSISDRLoss` |
| `hybrid` | 显式 CompositeLoss：`GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss` |
| `mbrcl` | 显式 CompositeLoss：`SISNRLoss` + `MultiResSTFTLoss` + `MultiBandRelativeContrastiveLoss` |
| `multires_l1_spec` | `MultiResL1SpecLoss` |
| `statece` | 附加 state detector CE 辅助任务（DBGNet 特例） |

- **`hybrid` 的官方证据与命名边界**：GTCRN 固定 commit [`502ebfab`](https://github.com/Xiaobin-Rong/gtcrn/tree/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73) 的 [`loss.py`](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/loss.py) 与 [`infer.py`](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/infer.py) 使用 `512/256/512 + sqrt-Hann` 重建/分析前端；分析窗和合成窗相乘为 Hann，在 50% overlap 下形成常数 overlap-add 包络——这只是官方手工、无包络归一化 WOLA 前端的产物，不是公式本身的要求。同作者 [`SEtrain/loss_factory.py`](https://github.com/Xiaobin-Rong/SEtrain/blob/3ec437d1adef76376775caa9e639d01208426380/loss_factory.py) 则展示了普通 Hann、waveform 输入的近邻 HybridLoss。项目 `hybrid` 两个 component 复用同一 raw prediction/target pair 和同一个 `lengths`，`lengths=None` 时均按完整波形计算；两者均可消费同一顶层 STFT 的 direct spectra，其中 log-ratio 项继承各 YAML 顶层 `stft` 的几何与同窗做 iSTFT，依赖 `torch.istft` 包络归一化对一致谱精确重建，不需要官方 sqrt-Hann 配对。因此 `hybrid` 表示与官方 `HybridLoss` 公式逐项对齐（30/70 压缩谱权重、尺度投影 log-ratio 非 dB）的显式组件拆分，配置名不隐含对官方前端的复制。
- **AEC 采样率与附加项**：采样率必须以 `<sample_rate>k` 置于模型/变体后、帧长前，16 kHz 与 32 kHz 均显式标注；`delayjitter`（启用 `aec_time_delay_jitter`）等场景附加项置于 loss 配方后。
- **AEC delay 分布**：`aec_ref_echo_random_delay_distribution: beta` 的 runtime 语义是 `(1-π)·Beta(a,b) + π·Uniform[min,max]`；省略 `aec_ref_echo_random_delay_uniform_prob` 时 `π` 默认为 `0.25`，不能在注释中简称为纯 Beta。只有配置显式写出该键时才按 YAML 值覆盖。
- **帧长 `<frame>ms` 以顶层 `stft.hop_length` 换算为准**（`hop / sample_rate`），并与配置注释保持一致。

### YAML 注释规范

配置正文的注释属于配置本身，必须能被机械检查。检查器为
`python tools/verify/check_config_comments.py`（默认检查 `configs/` 全部 YAML，非零退出码表示存在违规），
`tests/test_configs.py` 对同一规则设单元门禁。

1. **缩进跟随键**：注释行的缩进等于它描述键的缩进；section 标题与其子键同缩进或顶格。
2. **分节线顶格**：`# ---` / `# ===` 分节线缩进为 `0`，或与其后键同缩进；不得夹在父键与它的子键之间。
3. **一行一键**：每个键最多一行注释（行内 `key: value  # 说明`，或紧邻上一行）；只有多键 section 保留 1–2 行块头，更长的说明移到文件头注释区。
4. **注释块紧贴其键**：注释块与它描述的键之间不得有空行，也不得悬空在文件末尾。
5. **指路写全路径**：描述位于别处的键时写完整路径（如 `data.train.repeat_tile_silence_gap`），不留无键的悬空标题。
6. **键名必须是真键**：注释引用的键名以 YAML 实际键名为准；canonical/内部名只在需要时括注，并同时给出真键。
7. **单位与枚举显式**：时间 s / ms、采样率 Hz、电平 dB / dBFS、概率与权重无量纲；枚举取值列全。
8. **语言**：注释用中文，保留必要英文术语；文件头四标签合同（`Params` / `MACs` / `FLOPs` / `Trainer MOS boundary`）不变。

`max_duration_seconds` / `repeat_to_max_duration` / `repeat_tile_silence_gap[_range]` 属于 split 级 Dataset 参数
（`data.train` / `data.validation`），runtime 拒绝把它们写进 `simulation_cfg`；配置中把它们整理成紧贴
`simulation_cfg:` 之前的带标题块，A-3 段落只留一行指路注释。受保护结构（例如
`configs/tse/dynamic/tse_bsrnn_spk.yaml` 的手工 variant 切换模板）允许用 `# yaml-comment-lint: off` 与
`# yaml-comment-lint: on` 包住，区间内不检查；该标记只用于无法改写成键级注释的接口性注释块，不能用来绕过规范。

### 配置语义与校验合同（进阶参考）

以下条目不是命名规则，而是配置加载与校验时会被 runtime 消费的合同；新用户可以先跳过，遇到报错或需要改 loss、指标、仿真细节时再回来查。

- **loss 上下文注入**：构造 loss 时，runtime 将顶层 `stft` 中 loss 构造器声明的参数按 `setdefault` 注入，并从 `data.sample_rate` 注入其声明的首个采样率别名（`sample_rate` → `sampling_rate` → `sr` → `fs`）；普通 loss 的显式 `loss.args` / component `args` 优先。MBRCL 是严格匹配例外：存在顶层 STFT 或非空 sample-rate 上下文时，显式上下文参数必须与其完全一致，冲突会 fail-fast，而不是覆盖框架上下文。注释应写明实际参数与单位，不能把 loss 未声明的 `win_length` / `window` 误写为已注入。
- **loss builder fail-fast**：普通 loss 顶层仅接受 `type` / `args` / `strict`；`CompositeLoss` 顶层仅接受 `type` / `components` / `strict` / `enable_ss_undesired_energy`，每个 component 仅接受 `type` / `weight` / `args` / `name` / `enable_ss_undesired_energy`。未知结构键会在构造前报错并提示近似拼写；`args` 必须为 mapping 或 `null`，`stft` 上下文未知键及非正整数采样率也会立即拒绝。该校验只约束配置结构与上下文，不改变任何 loss 公式或 reduction。
- **validation metric quota**：`trainer.val_metrics.max_utterances_metrics` 是 SI-SDR/PESQ/STOI/ESTOI 等配置指标的全局 quota，`null` 或省略时回退 `max_utterances`；AEC 的 `max_utterances_erle` 是独立 ERLE 全局 quota，省略时同样回退 `max_utterances`，两者按 rank 精确均分但不共享计数器。ERLE 仅在 `interval` 命中的 validation 运行；优先消费显式 echo stem，没有 stem 时只允许可证明无背景噪声的 `echo_only` residual 近似，不能仅凭 `has_echo=true` 把 noise 计入 echo energy。有效 utterance 逐条先算 dB、再汇总 sum/count；`metrics_workers` 分别限制侵入式指标池与 ERLE 池，ERLE 池还受 CPU 数和本 rank 整轮 quota 限制。建议生产配置显式写出两个 quota，避免依赖回退值造成“共享上限”的误读；AEC/SE 生产配置已显式声明，TSE 仍依赖回退。
- **AEC validation metric 显式合同**：AEC 训练配置在 `val_metrics.enabled: true` 且 `metrics` 非空时，必须把 `ERLE` 显式列在 `metrics` 中；Entry 不再为外部配置补入 `ERLE`，未列出即不计算 ERLE，显式 `metrics: []` 仍表示关闭全部 objective metric。
- **validation metric 空结果**：配置指标按固定顺序在所有 rank 参与聚合。`strict: true` 时，任一配置指标全局 count 为 0 或均值非有限，会在 TensorBoard/history/curve 写入前 fail-closed；`strict: false` 时，只要本轮至少一个配置指标存在有限全局结果，就提交一行 epoch/CSV history，并为无结果指标保留 `NaN` hole 以维持各指标列对齐；TensorBoard 只写有限指标值，绝不写 `NaN` metric scalar。若全部指标为空，则仅告警且不提交 history 行。
- **单位术语**：采样率使用 Hz，waveform 长度使用 samples，时间使用 s/ms；绝对数字电平写 dBFS 并注明 RMS 或 peak，SNR/SER/ERL/SDR 使用 dB，概率与 loss 权重为无量纲。
- 纯推理配置在 `meta` 标记 `inference_only: true`，且命名以 `*_infer.yaml` 结尾（`tests/test_configs.py` 跳过其训练字段校验）。
- 仅评测配置在 `meta` 标记 `evaluation_only: true`；该标记表示只用于评测或数据合同审计，TSE training entry 会直接拒绝，不能作为训练配置或微调入口。
- **配置完整性**：发布配置可使用受控的 `base_config` 文件继承；解析器会递归合并继承链并把参与继承的文件纳入配置 identity，list 字段按整体替换。SSL 四份官方模型模板通过 `base_config` 复用 moving dynamic 基线，spatial wind 配方保持自包含。`data.validation.inherit_simulation` 只表示同一份配置内的 dynamic validation 仿真继承。
- **MetricGAN 学习率调度**：`ExponentialLR(gamma=0.99)` 用于 generator/discriminator 的 epoch 收尾衰减；应设置 `trainer.lr_schedule_per_step: false`，避免每个 optimizer update 都衰减。
- **`data.mode` 是必填路由字段**：合并后的配置必须显式声明九种合法 mode 之一：`dynamic` / `offline`、`dynamic_aec` / `offline_aec`、`dynamic_tse` / `offline_tse`、`dynamic_ss` / `offline_ss`、`dynamic_ssl`；Dataset builder 会立即调用 `normalize_data_mode()`，缺失或非法 `data.mode` 均 fail-fast，绝不回退为 `offline` 或其他默认模式。
- **`offline_tse` 导入合同**：每个 split 必须在 `samples_list` / `samples_jsonl` 中二选一，并显式提供 `manifest_format`（`jsonl` / `path_list`）、`path_base`（`manifest_relative` / `recipe_root`）、`target_selection`、`mix_variant`、`cue_schema` 和 `enabled_cues`；只有启用 cue 时提供 `cues_yaml`。Adapter 可解析 audio/spatial/visual/textual 索引，Dataset 解码 audio 的 waveform/embedding、spatial `npy/fields`（static/track typed cue）、visual `npy/muse_frontend`（预提取帧级特征）与 textual `json/dae_keyword_phoneme`；visual 的 raw `mp4` 需要可选视频后端，构造期显式拒绝。`models/modular_spec.py` 只做组合解析与能力判定，`TSE_Modular` 已在 registry 中，frontend 已接线 speaker/spatial/visual（仅 `representation=features`）/textual，其他 backbone、`listen` 与 raw video 在构造期被拒绝。Dataset 能解码某类 cue 不等于已有对应真实数据集、训练闭环或部署路径。
- **`simulation_cfg` ownership 合同**：训练配置使用单一当前 strict contract，不暴露 `schema_version`；旧 `schema_version` 键按 retired key fail-fast。AEC scenario / SER / farend / ERL 及 `aec_ref_echo_*` 只能用于 AEC；七路目标/干扰场景、`num_interferers` 与 enrollment 键只能用于 TSE；timeline、Music 和 RIR backend 为共享能力，其中 relative timeline 只开放 TSE/SS/SSL。显式 `0` / `false` / `null` / 默认值也属于配置，不能出现在其他任务；unknown `simulation_cfg` 键一律 fail-fast，旧宽容开关 `simulation_cfg_strict` 已退役并在配置边界被拒绝。动态 validation 在 merge 前校验，错误会保留 `data.train...` 的 `train inherited` 或 `data.validation...` 的 `validation-local` 来源。
- **声学角色 policy 不作为 YAML 旁路字段**：目标、干扰、near-end、loudspeaker echo、enrollment 和 digital reference 的角色/RIR 语义由 `core.data.simulation.role_policy` 与 scene facts 记录；不要重新添加已退役的 `spatial_rir_policy` 等 `simulation_cfg` 字段。模型是否能消费 `[C,T]` 仍由 model IO contract 在 entry 边界校验。
- **内部 typed view 与 projection**：外部 YAML 使用当前平面 `SimulationConfigs` 合同；runtime 在 parser 完成后建立 immutable task view。flat projection 仅非破坏地把 Layer A 与任务 Dataset 构造键提升为 kwargs，再将剩余 mapping 交给原 parser。该过程不改变 validation 的 nested dict 递归合并、list 整体替换、deprecated warning 或 strict unknown-key 的触发时点。
- **当前 capability 与 facts 合同**：Layer D 定义为 post-mix amplitude conditioning，固定执行 `D-normalize → D-guard`。公共 `D-normalize` 使用 `peak_normalization_mode` / `peak_normalization_scale_range`，只支持 `off/joint`；PASE/StuPASE/UniPASE/GAP 的 `separate/single` 仅在 SE 专属 `simulation_cfg.pase_preprocess` 中消费，不进入公共 `SimulationConfigs`；`apply_after_channel: true` 时推迟到 `task-post-EF`（Layer E/F 之后）执行。`joint` 会将同一 scale 同步到所有相关 source/target；AEC 的 mic/nearend/echo/noise/farend 使用同一 coupled guard scale。该阶段默认 `off`；scale range 必须满足 `0 < lo <= hi <= 1`。SE/AEC/TSE/SS 的 waveform scene facts 按各任务合同发布，SSL 另发布 spatial renderer 的 feature/DOA/VAD provenance；禁止发布顶层 normalize facts、独立 trace normalize facts、顶层 farend guard facts，也不存在任何独立旧槽、旁路、alias 或 fallback；相关旧 schema 已退出。旧 YAML `adc_hard_clip_prob` / `adc_peak_guard_trigger` 仅在 schema/config 校验中 fail-fast。
- **Layer E/F 职责**：Layer E 保留 channel clipping；`bandwidth_limitation.target_sr` 可固定有效带宽（官方 UniPASE/GAP 用 8000），缺省时随机低采样率。Layer F 是 observation/device clipping 的权威位置，由 Dataset 在 CPU 上执行。`noisy_post_mix_degradations_enabled` 是 Layer F 的唯一总开关；为 `false` 时整层跳过，训练侧不另设 GPU Layer F。AEC 生产配置使用 `fixed_rail`（固定 ±1.0 rail），并在截幅前从 `clip_pre_drive_db_range` 抽取 pre-drive。配置中的 gate、selector 与 path 概率须按其条件域解释；selector 命中率不等于真实 rail-hit、饱和或 waveform 截幅率。它们都不控制 Layer D/E。
- **固定 post-mix 顺序**：SE/AEC/TSE/SS Dynamic Dataset 统一执行 `pre-finite → D-normalize → D-guard → E → F → task-post-EF → final-guard → post-finite`；SSL dynamic 复用共享静态/移动 Layer A/C/D/E/F seam 后再投影 feature/target。SE/AEC 在 post-mix 后执行 scene-wide `TimeTransformPlan`，TSE/SS 在 Layer A 已完成固定画布。`final-guard` 是 E/F 后无随机性的 observation-only 确定性安全网，不修改 clean/source targets，也不是第二个 clipping 增强层。Offline AEC Dataset 根本不运行 Dynamic post-mix；AEC batch 仍在 STFT 前按需运行 `AECDeploymentFrontend`，该前端不属于仿真层。
- **文档维护边界**：上面的固定顺序和所有权摘要用于配置索引；若与 canonical 仿真正文不一致，以 [`Simulation_Flow_Logic.md`](../docs/guides/simulation/Simulation_Flow_Logic.md)、源码和测试为准。
- **Dynamic 训练 RIR / 音量合同**：默认非 `inference_only`、非 `evaluation_only` 且 `data.mode` 的 storage 为 `dynamic` 的现役训练配置，`data.train.simulation_cfg.rir_normalize` 必须为 `true`。PASE/StuPASE/UniPASE/GAP-URGENet 是封闭复现例外，保留 `false` 以维持既有数据语义，禁止机械改值。启用 `source_volume_normalization_enabled` 时，目标电平以 RIR 卷积后的 early target 为参考，抽取的同一 gain 同步作用于 full、early 与最终 target；noise / echo 在该步骤之后混入；offline 与纯推理配置不要求出现该键。

## 运行入口与 checkpoint

训练命令统一使用 `python train.py -C <config>`，九种合法 `data.mode` 来自 `core/data/constants.py`：`dynamic` / `offline` 路由到 SE，`dynamic_aec` / `offline_aec` 路由到 AEC，`dynamic_tse` / `offline_tse` 路由到 TSE，`dynamic_ss` / `offline_ss` 路由到 SS，`dynamic_ssl` 路由到 SSL。offline 变体使用各自的预生成清单且无在线仿真随机性：AEC `offline_aec` 使用 canonical 四列 `mic/farend/nearend[/error].scp`；TSE `offline_tse` 在 `samples_list` / `samples_jsonl` 中二选一，并显式声明 `manifest_format`、`path_base`、`target_selection`、`mix_variant`、`cue_schema`、`enabled_cues` 及按需的 `cues_yaml`；SS `offline_ss` 使用固定 K 的 separation JSONL。

Phase 4C 发布门禁不统一上述业务 schema。`GenerationManifest v2` 是唯一的发布/读取契约。正式 publication 必须以 `strict_load=True` full iterate production adapter 后才写 receipt；TSE `tse-wesep-offline-v1` 还会解码全部 enrollment candidate 并绑定外部 mixture/enrollment 内容身份。AEC simulated 七场景使用 `adapter=none` 完成静态 gate。`--probe` 只用于调试，不能发布。当前 generator 版本为 SE 主 offline `2.11`、AEC simulated `2.18`、AEC OOD `1.2`、TSE pseudo-target `2.0`、DNS3 `2.1`，通用 SE indexer `prepare_offline.py` 为 `2.1`。

通用 SE indexer 使用 `--verified-publication --waveform-root` 生成 schema v2 与 verified receipt；DNS3 `--scp-only` 只复用同 identity 的 owned root。simulation audit 使用独立 envelope：SE/AEC 保留 task-local layers/measurements；TSE 的 capture-off 不保留 waveform stems，仅保留 sealed role-local metadata，并将 stems capability 标记为 unavailable；capture-on 发布 `stem-descriptors-v2` 摘要和 waveform hash，另有 task-local stem capture。

- `--resume` 恢复完整训练状态，并在未指定 `--resume-from` 时自动发现 checkpoint；`--resume-from PATH` 仅在 `resume=True` 时生效，统一 CLI 中该选项隐含启用 resume，可显式选择 `latest_model.tar`、`model_E*_B*.pth` 或 step validation 保存的 `best_step=*.tar` / `best_model.tar`。middle 与 step-best checkpoint 都携带 `batch_cursor`/`epoch_incomplete`，resume 会回到对应 epoch 内已完成的 batch 边界继续训练；`--checkpoint PATH` 仅加载模型权重用于初始化。`--resume-from` 与 `--checkpoint` 语义不同，不应组合使用。
- 省略或设为 `null` 的 `trainer.max_steps` 不限 optimizer step；设置后在 optimizer-step 边界检查，`0` 会在训练开始前直接停止，不执行任何 optimizer step，smoke check 应使用正整数。

## DataLoader 配置

`data.dataloader.num_workers` 与 `persistent_workers` 控制训练 loader。验证 loader 默认沿用这两个值；需要独立限流时可配置 `val_num_workers` / `val_persistent_workers`。train/val 任一侧的 worker 数为 `0` 时，该侧 `persistent_workers` 会强制为 `false`，且不传入 `prefetch_factor`。`pin_memory`、`prefetch_factor` 与 `timeout` 仍由 train/val 共享。

DDP 下每个 rank 都会分别构造 train 和 val loader；两侧都使用多进程时，验证期间的进程峰值应按 `(num_workers + val_num_workers) × world_size` 核算。音频 cache 也按 worker 独立持有，不能只按训练 worker 数估算主机内存。

## exp_path 约定

所有非推理配置必须显式设置 `trainer.type` 和 `trainer.exp_path`；`exp_path` 取配置文件 stem，不再额外编码实验信息：

```
exp/se/<配置文件名去 .yaml>
exp/aec/<配置文件名去 .yaml>
exp/tse/<配置所在子目录>/<配置文件名去 .yaml>
```

TSE 保留配置所在子目录（当前为 `dynamic` / `offline` / `textual` / `inference`），例如
`configs/tse/dynamic/tse_bsrnn_spk.yaml` 对应 `exp/tse/dynamic/tse_bsrnn_spk`；
SE/SS 的 `spatial/` 配方使用 `exp/<task>/<配置文件名去 .yaml>`（如
`configs/se/spatial/online_spatialnet_dynamic_mhsa.yaml` →
`exp/se/online_spatialnet_dynamic_mhsa`）。该规则由
`tests/test_configs.py::test_config_path_contracts` 校验。

例如：

```
exp/se/usr_bsrnn_causal_urgent26_offline_hybrid
exp/aec/deepvqe_small_32k_10ms_delay80ms_loss_hybrid
exp/tse/dynamic/tse_bsrnn_spk
```

- 新增配置时按任务目录使用上述固定前缀，并直接写出完整配置内容。
- `exp/eval/` 的固定目录约定仅适用于 SE/AEC 评估管线，输出独立于训练目录并沿用被评配置名。TSE 评估实际使用显式必填的 `--output_dir`；`exp/eval/tse/<配置名>` 只是推荐路径，不是 runtime 自动选择的固定目录。
- PASE/StuPASE/UniPASE/GAP 阶段之间由本仓库实验产生的 checkpoint，是唯一允许保留 `exp/se/...` 相对路径的资源；其他外部 checkpoint 不得写仓库内路径。例外是 PASE 系列官方发布权重，固定放在 `tasks/se/models/pase_family/<family>/open_model_checkpoint/`，由 `.gitignore` 隔离。

## 数据路径红线

- 真实数据 manifest 支持带引号的本机绝对路径与相对仓库根目录的 `data/...` 路径；尚未配置的资源用 `null` / `~` 表示。外部 checkpoint 使用绝对路径或 `~/use_train_checkpoints/...`。禁止提交真实数据、checkpoint、日志本体（路径字面量可以提交，资源本体由 `.gitignore` 隔离）。
- PASE/StuPASE/UniPASE/GAP 阶段之间由本仓库实验产生的 checkpoint 可使用 `exp/se/...` 相对路径；TSE speaker encoder 权重使用 `tasks/tse/speaker_embedding_models/...`，REAL-T 官方完整模型使用 `tasks/tse/pretrained_models/...`；PASE 系列官方发布权重使用 `tasks/se/models/pase_family/<family>/open_model_checkpoint/...`。以上权重均由 `.gitignore` 隔离。其他外部权重仍必须使用绝对路径或 `~/use_train_checkpoints/...`。
- 未配置清单（`~`）的训练入口见各任务 README 的"数据准备"章节；runtime entry 不豁免 `~`，必需资源必须在训练前替换。
- AEC 的 `configs/aec/offline_aec_template.yaml` 是 **copy-and-fill 模板**：复制为实验配置后，把 train/validation 的 `mic_scp` / `farend_scp` / `nearend_scp`（及可选 `error_scp`）填成真实路径，再按普通 AEC 配置接受完整 entry 校验。它不是测试 hydrate 白名单，不能带着 `~` 直接训练，也不会由 `tests/test_configs.py` 注入虚拟路径。
- 仅 `tests/test_configs.py::TSE_TEMPLATE_CONFIGS` 精确白名单中的 `configs/tse/dynamic/tse_bsrnn_spk.yaml` 是可分发未就绪训练模板；四种官方 cue/causal 组合通过该文件顶部的注释切换块手工选择。不使用 `meta.training_template`，runtime 也不消费模板标记；配置测试只对这一精确路径 hydrate 已声明的资源占位并运行真实 `TSETrainingEntry._validate_config()`，任何其他配置均不得获得该豁免。

## 噪声发布准入与训练消费边界

噪声准入由 Noise Filter Pipeline 负责，训练侧只消费结果。发布/审计合同为 strict JSONL schema v6（admission policy v4、AudioTag evidence v5、noise source parser v10、generator `6.1`、promotion marker v4），Stage D 只有三条执行路径：`skip_d1: true` 时 `C3 → 独立 D2`；启用 D1 且未配置 `d1.post_normalize` 时 `C3 → D1 → 独立 D2`；启用 `d1.post_normalize` 时 D1 内合并 D2，最终子阶段为 D1。C2 的 `audioset_top5` 恰 5 项，每项严格包含 `index/mid/label/prob`；C3 无阈值、无 calibration，top-5 任一 speech MID 即 `reject`。完整口径见 [Simulation_Flow_Logic.md §0.1.1](../docs/guides/simulation/Simulation_Flow_Logic.md)。

训练侧的边界概括为四句：

- source manifest 只接受严格四列 SCP（`uid fs length path`）或逐行 JSON object 的 JSONL；JSON array 与遗留三列 SCP fail-closed，metadata 仅由 JSONL 携带。
- 训练完全信任配置指定清单中的样本，不在训练侧重放准入校验；发布清单的 `length`（`ROUND_HALF_UP(Decimal(str(duration)) × sr)`）原样读取。
- 分类能力取决于 ordinary pool 的 JSONL metadata 完整度：每条需有非负 `category_id` 与非空 `balance_category`（缺失时可从 `category_name` 推导）；启用 `balance_categories` 后 metadata 不完整直接 fail-fast。
- Dataset 只保留投影后的八个字段 `uid/fs/length/path/category_id/balance_category/category_name/roles`，完整 admission/evidence/fingerprints 不进入 Dataset 状态。

### 六种训练噪声策略对比

下表中的“过滤”仅指采样候选排除或 Wind/Music 路由，不是内容准入；六种策略都不会按 `Human` / `Speech`、`vocal` / `foreground_speech` 等标签或发布证据过滤训练样本。

| 策略 | 配置 | 必需信息 | 第一段 | 第二段 | 是否过滤 |
|------|------|----------|--------|--------|----------|
| **Natural UID** | 未启用 `sampling.noise.balance_categories` | `noise_scp` 基础 source 字段；类别 metadata 可选 | 按 UID 均匀选择，SR tier 为 `exact → 首个可用 higher SR → all` | 分类 metadata 完整时排除首类，按 `exact → 所有 higher SR 聚合 → all` 选异类 UID；无异类退化单噪声。metadata 缺失或部分缺失时才回退普通 UID 二次读取 | 不做语义过滤；回退读取允许同 UID/同类 |
| **Category balance** | `balance_categories: true`、`balance_mode: category` | ordinary pool 分类 metadata 完整 | 按 `exact → 所有 higher SR 聚合 → all` 取首个有候选 tier，category 等权、类内 UID 均匀 | 排除第一段 category/UID 后按相同 tier 和 category 等权规则选择；无异类退化单噪声 | 只排除第一段同 category/UID，不按内容标签过滤 |
| **Temperature** | `balance_categories: true`、`balance_mode: temperature`；`alpha` 沿用配置值 | ordinary pool 分类 metadata 完整，且类别有 difficulty | 按 `exact → 所有 higher SR 聚合 → all` 取 tier，以 `P(c) ∝ n_c^alpha × difficulty` 选 category，类内 UID 均匀 | 排除第一段 category/UID 后按过滤后的 tier-local 数量重算权重；无异类退化单噪声 | 有放回采样，无 epoch UID 重复硬上限；只做同 category/UID 排除 |
| **Wind side-chain** | `prob_wind_noise_sidechain > 0`，并配置 `wind_noise_config` | 显式非空 `windnoise_scp`，或 metadata 中可路由的 `wind` role | 抽中时从 wind pool 取样并走 side-chain；Wind 在 SE/AEC/TSE 都从 ordinary pool 移除 | 同时触发 `noise_mix_prob` 时再加入 ordinary noise；未触发时无第二段 | 仅按 role 路由 Wind；内嵌与显式 wind pool 合并扩池 |
| **Spatial wind background** | `prob_wind_noise_background > 0`；必须 `spatial_enabled=true` | 同一 wind pool；空间渲染由 `BackgroundSpec` 选择 diffuse 或 point-source | 抽中时生成严格 `[C,T]` additive background，统一按 spatial image SNR 标定 | `noise_mix_prob` 命中时再加入 ordinary noise；不进入 side-chain compressor | 只在空间 observation 生效；target/direct 与 source/background decomposition 保持不变 |
| **TSE Music override** | TSE `simulation_cfg.music_ratio > 0` | 对应 `data.train` / `data.validation` 下显式非空 `music_scp`，或 metadata 中可路由的 `music` role | 每次 source read 独立按 `music_ratio` 选择 music 或 ordinary family | 再独立执行 override；第一段为 Music 时强制第二段 ordinary，禁止 Music+Music；ordinary 分类完整时仍排除同 category | Music 只在 TSE 从 ordinary pool 拆出；SE/AEC 仍视为 ordinary noise |
| **普通双噪声** | `noise_mix_prob > 0`，混合比例来自 `noise_mix_ratio_range` | 至少一个 ordinary noise；异类第二段依赖完整分类 metadata | 第一段沿用 Natural UID / Category balance / Temperature（TSE 还含 Music override） | 分类完整时必须选不同 category，无异类退化单噪声；仅 Natural 在 metadata 缺失/部分缺失时回退普通 UID 二次读取 | 不做 Human/Speech 等语义过滤 |

补充约定：

- `balance_categories: true` 要求 ordinary pool 每条记录分类完整，metadata 不完整时 fail-fast；但不要求 strict v6、Stage 证据或 `dataset_id`。SE/AEC/TSE/SS dynamic 均支持 noise controller；SSL 使用显式 `noise_scp`，不走该 controller。
- Temperature 默认 `alpha: 0.5`；只有显式启用 `balance_mode: temperature` 时该参数才参与采样。当前会议版 DeepVQE 使用纯四列 SCP 且未配置 `sampling.noise`，因此走 Natural UID 采样，不存在 `alpha` 覆盖或 Temperature 曝光语义。Temperature 模式下 category 内 UID 均匀、有放回，仓库没有 noise UID 的 epoch 重复硬上限。
- selector 派生缓存是双预算 LRU（16 entries / 65,536 UID references），只缓存候选视图；超大 Natural entry 使用零复制 view。它不是 UID hard cap，也不改变 RNG。
- Wind 类别在四个 waveform 任务中都从 ordinary additive pool 路由到 side-chain；TSE 总是拆 Music，SE/AEC/SS 在显式提供 `music_scp` 或 `music_ratio > 0` 时拆 Music。SSL 的 noise pool 由其 spatial supervision 合同独立管理。类别集合以 `core/data/noise_roles.py` 的 `WIND_CATEGORY_NAMES` / `MUSIC_CATEGORY_NAMES` 为准。
- `roles` 值为 list 时即为权威路由，空 list 或不完整 list 也不补判；只有字段缺失或值非 list 时，才按 `balance_category` 推导，缺少该字段时可从 `category_name` 推导。
- 普通无分类 source 开启相应风噪或音乐概率时，必须显式提供对应 side SCP；显式 side SCP 与内嵌 role 始终合并扩池。
- Stage C checkpoint 与 strict v6 发布物绑定五项 artifact fingerprint：voice policy、AudioTag label dict、AudioTag checkpoint、Qwen model、C1 prompt。run 从 Stage B 输出执行 `C1 → C2 → C3`，再按上述三分支执行 Stage D 并导出 strict JSONL v6。

## 模型复杂度与 MOS 报告合同

所有 YAML/YML（包括 `meta.inference_only: true`）都必须在**首个 YAML key 之前**保留人工可读摘要，并显式包含 `Params`、`MACs`、`FLOPs`、`Trainer MOS boundary` 四个标签。人工头合同只强制这四个标签及准确的自然语言边界，不要求每份 YAML 机械重复完整审计字段，也不得把 `[AUDIT]` JSON 嵌入 YAML 头。

YAML 配置头不等同于完整审计记录，也不得批量伪造未经 profiler 复核的数值；未独立测量的项目应明确写 `not measured`。四个标签的自然语言必须准确表达以下边界：

- **Params**：说明是否已测量及其口径；无需在 YAML 头机械展开 `total`、`trainable`、`scope`、`source`。
- **MACs**：说明是否已测量，并在有结果时准确概述输入与覆盖边界；不同输入、sample rate 或 scope 的结果不得直接横比。
- **FLOPs**：与 MACs 独立；没有独立、可靠测量时写 `not measured`，严禁用 MACs 自动乘系数生成 FLOPs。
- **Trainer MOS boundary**：明确 Trainer、模型 profile 和复杂度审计均不计算 MOS；DNSMOS/NISQA/UTMOS/SCOREQ/AECMOS 等只能通过 `evaluate/eval_pipeline_*.py` 离线评估，并随数据集与评测协议报告。

`tools/model_tools/audit_metrics.py` 输出的 `[AUDIT]` JSON 才承载完整的 `total/trainable/scope/source/backend/input/coverage` 等结构化字段，以及独立的 RTF `value/error/duration_seconds`。其中非 GAN scope 只有 `task`，GAN 可报告完整 `task` 或仅 `generator`；命名 profile 输入为 SE `waveform`、AEC 各 1 秒 `mic`/`farend`、TSE 默认 1 秒 `mixture` 与 3 秒 `enrollment`（可用 CLI 调整）。THOP 成功也只能标记 `partial/unverified`，因为 module hooks 可能漏计 functional、complex、custom/unsupported operator；显式按零计数的算子也必须进入 `coverage`。审计命令只读独立完整配置，禁止自动回写 YAML。

`audit_metrics.py` 支持 `--task {auto,se,aec,tse}` 和 `--scope {task,generator}`。`--configs_dir --recursive` 稳定枚举全部 `.yaml` / `.yml`，每个发现文件必须输出且只输出一条 `[AUDIT]` JSON 记录；配置加载或模型构建/Params 失败写 `status=failed`、`stage`、`error`，不得中断或跳过，其余记录继续处理，结尾 `[SUMMARY]` 的 `discovered` 必须等于 `reported`。MACs/RTF profile 失败只标记为未测量并保留 Params。`--no_profile` 只统计 Params，不要求 waveform-native 配置提供 `stft`。无法用合成输入忠实构造的变体（例如需要真实预提取 `spk_embedding` 的 TSE）必须明确报告 Params-only 原因。其输出是审计报告，不是配置来源。

## 配置校验

所有提交的 YAML 必须通过 `tests/test_configs.py`。普通训练配置运行与 `train*.py -C` 相同的完整 entry 校验；`meta.inference_only: true` 配置跳过训练字段校验；`meta.evaluation_only: true` 仅用于评测/数据合同审计，并由 TSE training entry 拒绝。AEC `offline_aec_template.yaml` 仍是人工 copy-and-fill 的普通模板：测试不会 hydrate 其 `~` 数据路径；真正构造 `AECOfflineDataset` 时，未填写 `mic_scp` / `farend_scp` / `nearend_scp` 会 fail-fast。唯一精确白名单 TSE 模板采用封闭 hydrate 例外：测试先证明三个已声明 `null` 必需资源会按完整字段路径 runtime fail-fast，再仅将 train/validation `speech_scp` 与 train `spk2enroll_json` hydrate 为 non-empty、已 trim 的虚拟路径，最后对完整配置运行真实 `TSETrainingEntry._validate_config()`。动态 TSE 的 speech manifest 必须为 strict JSONL，identity 直接来自 manifest。该过程不做文件 I/O，也不豁免其他 model/trainer/loss/validation 错误。

```bash
python -m pytest tests/test_configs.py -v
```
