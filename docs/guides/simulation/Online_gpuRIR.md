# core 在线 gpuRIR 与离线 RIR 切换

RIR（room impulse response，房间冲激响应）决定一段声音在房间里带多少混响、从哪个方向到达麦克风。有两种拿 RIR 的方式：**offline** 从磁盘上提前生成好的 RIR 清单读取；**online** 在训练时用 GPU 现场算（gpuRIR 是执行这一步的库），因此每个样本的房间和声源位置都可以不同。`rir_backend` 就是选择这两者的共享 Layer C-0 配置。

SE、AEC、TSE、SS、SSL 的动态空间路径统一先调用 `prepare_scene_plan()`，固定 room、阵列位姿、actor 位置或轨迹，再生成 full/direct RIR。静态和移动场景分别经 `scene_static` / `scene_moving` IPC；后者返回 `[P,A,C,L]`，由 core 分段卷积生成 waveform。`P` 是轨迹帧数，`A` 是参与声学渲染的 actor 数，`C` 是 microphone 数，`L` 是 RIR 长度；任务监督 source 数独立定义。

`AcousticScenePlan.num_samples` 表示 waveform 长度 `T`；`room.rir_num_samples` 表示 RIR 长度 `L`。两者独立，不能使用 `max(T,L)` 改写训练样本或轨迹时间轴。配置中的 microphone topology 是相对坐标；online sampler 将其整体放置到房间后，plan、provider 和 renderer 使用同一组绝对坐标，不再重新居中或平移。

## 配置

公开 YAML 保持平面字段，按 A–F 排列。常用开关：

```yaml
simulation_cfg:
  # Layer C-0
  rir_backend: offline  # 改为 online 开启在线；offline 从当前 split 的 rir_scp 读取
  prob_reverberation: 1.0  # 空间 source image 的正式配置
  prob_moving_rir: 0.0  # 在线 moving 概率（spatial 与 legacy mono 共用）；离线 moving 由资产组成决定
  early_rir_sec: 0.05
  target_type: early
  rir_normalize: true
  reuse_rir: false
  # 空间配置：profile 与 custom_geometry 二选一
  spatial_enabled: true
  array_profile: linear_4ch_40mm
  reference_channel: 0
  # custom_geometry:
  #   coordinate_system: cartesian_xyz_m
  #   positions_m: [[-0.06, 0.0, 0.0], [-0.02, 0.0, 0.0],
  #                 [0.02, 0.0, 0.0], [0.06, 0.0, 0.0]]
  # 仅 online 使用，采样率来自 data.sample_rate / 当前实际工作 SR
  gpurir_room_dim_range_m: [[3.0, 8.0], [3.0, 8.0], [2.5, 3.5]]  # [length, width, height]
  gpurir_rt60_range_sec: [0.1, 1.0]
  gpurir_source_distance_range_m: [0.2, 3.0]
  gpurir_source_min_separation_m: 0.3
  gpurir_source_height_range_m: [1.1, 1.8]
  gpurir_mic_height_range_m: [1.0, 1.8]
  gpurir_wall_margin_m: 0.1
  gpurir_sound_speed_mps: 343.0
  gpurir_diffuse_transition_db: 15.0
  gpurir_timeout_sec: 60.0
  gpurir_queue_size: 8
  gpurir_cache_size: 8
```

`gpurir_*` 字段只在 `rir_backend: online` 时生效。`gpurir_rt60_range_sec` 是混响时间 RT60 的采样范围（RT60 = 声源停止后声压衰减 60 dB 所需的秒数，越大房间听感越空旷）；`gpurir_room_dim_range_m` 顺序为 `[长, 宽, 高]`。

服务端 waveform cache 默认使用 32 MiB 字节预算，同时受 `gpurir_cache_size` 条目数限制。可在启动训练进程前通过环境变量 `USE_TRAIN_GPURIR_CACHE_BYTES` 调整预算（非负整数，单位为字节；设置为 `0` 可关闭 waveform cache）。该预算应结合 RIR 命中率和主机 RSS 实测调节，不能只增加 DataLoader worker 数量来掩盖 GPU 服务排队。

`array_profile` 只表示物理 microphone topology（麦克风阵列的物理排布，与任务、模型无关）。当前 canonical profiles 为
`linear_2ch_80mm`（2 通道、80 mm spacing）、`linear_4ch_40mm`（4 通道、40 mm
spacing）、`linear_6ch_40mm`（6 通道、40 mm spacing）和 `circular_8ch_r80mm`
（8 通道圆阵、半径 80 mm，spacing 记录相邻弦长）。profile 解析为
`MicrophoneArrayProfile` / `ArrayTopology`，位置以 array center 为原点、沿 linear
axis 给出，包含 positions、spacing 和 coordinate system；reference channel 由任务无关
配置单独校验。同一 profile 可被 SE、AEC、TSE、SS 与 SSL scene loader 使用。需要非标准
拓扑时使用显式 `custom_geometry`，提供同一组物理字段；不能把任务名、模型名或外部
仓库名用作 profile。

FN-SSL、NBSS 等外部仓库只可写入 geometry/model provenance 或 validation recipe，
例如 `external_repository`、`external_path`、`external_commit`；它们不属于 profile
registry，也不替代解析后的 positions。来源键只接受这组 canonical 名称，短键
`repository` / `path` / `commit` 会被直接拒绝。RIR、scene 与 manifest metadata 应保留该来源信息，便于复核而不
改变 array identity。

在线 gpuRIR 只负责 Layer C-0 的 room/source×microphone RIR；五任务九种
`data.mode`、Layer A/B 的 source 与 scene 语义仍由各任务 Dataset 负责。Layer D、E、F
继续按公共合同处理 observation，其中 Layer F 始终是 Dataset 内 CPU-only；target/output、
AEC farend、TSE enrollment 不因在线空间 RIR 变为多通道。

`prob_moving_rir` 由五个动态空间任务与 legacy mono 动态仿真共同消费，要求
`rir_backend=online`。spatial 路径按逐样本 Bernoulli 选择 static/moving scene；
mono 路径在 `_mix_acoustic_rir` 抽中 reverb 后按同一字段再抽一次，因此 mono 的
实际 P(moving) = `prob_reverberation × prob_moving_rir`。各 actor 的轨迹共享
timebase，但独立采样端点；AEC loudspeaker 默认固定，near-end 可以移动。
`[P,A,3]` 是这些独立轨迹按 actor order 排列的无损表示，并不表示同步平移。
`segment_sec` 不属于这个 scene-level 在线轨迹合同；mono moving 的分段长度由
trajectory timebase 的 hop 决定。`gpurir_room_dim_range_m`
顺序为 `[length, width, height]`。`gpurir_max_sources` 默认为 16，`gpurir_device`
默认为 `null` 并跟随 `LOCAL_RANK`。在线模式与 `reuse_rir` 互斥；服务 cache
绑定完整 room、array、actor、timebase 及请求种类，不会将上一房间套给下一场景。

manifest 模式保持各任务的 SCP 路径。SE/AEC/SS 使用对应采样率的公共
`fs{采样率}/{train,val}_rirs.scp`；TSE mixture 使用
`fs16000/tse/{train,val}_rirs.scp` 的 95,000 / 5,000 个纯静态 gpuRIR PKL，
enrollment 使用 `static/near/{train,val}_rirs.scp`。在线模式不读取 mixture
SCP，也不要求该文件存在；enrollment 的独立清单仍按其自己的增强配置读取。
在线失败直接报错，绝不回退 manifest 或静音。上述普通 mono RIR SCP 不替代 spatial manifest：动态空间 offline 路径要求 typed JSON，包含 `sample_rate`、`scene_uid`、`room`、`geometry.positions_m`、`actor_order`、`actor_positions_m`、generator version 和 full/direct/early 内容摘要。sampler 只选择已声明的 actor 子集并保持明确顺序，不重新随机 room 或位置，也不通过填零坐标补缺失元数据。单次声明不足时应补充真实资产元数据或重新生成数据。

离线 spatial moving scene 使用 schema v3 manifest：`metadata.moving: true`，并额外声明
`actor_trajectories`（逐 actor `[P,1,3]`，共享一个 `trajectory_timebase`）与
`trajectory_sha256` 轨迹内容哈希；track 文件为 `[P,A,C,L]` bank（scene bundle 内
按 actor 切片）。load 时校验轨迹形状、房间边界、首帧与 `actor_positions_m` 一致及哈希；
provider 读取时逐 actor 复核 track 哈希与请求轨迹，渲染经共享 moving renderer
分段卷积。scene 的 static/moving 由 manifest 权威决定，运行时不再抽样；index 中
static/moving scene 的混合比例即训练比例。生成侧使用
`tools/prepare_data_tools/generators/se/generate_scene_rirs.py --moving`
（generator version `se.scene-rir-v3.0`）。请求 waveform 长度必须覆盖最后一个轨迹
帧起点，否则 sampler fail-fast。

## 一次场景生成

一次在线请求（一个场景）依次做下面五件事，产出 CPU 侧 float32 RIR；下文的 `K` 是本次场景布置的声源位置数。第 1 步的 Sabine 是房间吸声公式，用来检查抽到的房间尺寸/RT60/吸声系数组合是否物理自洽。

1. CPU 按稳定 seed 采样一个 room 和一个 RT60，拒绝不满足 Sabine 吸声范围的组合。
2. 从 `array_profile` 或 `custom_geometry` 解析 `ArrayTopology`，以 array center 和 coordinate system
   放置整组 microphone positions；再按距离、墙距、高度及 source-source 最小间距布置 K 个位置，有限次失败后报错，不减少 K。
3. sampler 在 CPU 侧产出不可变 plan，provider 和 service 只消费它。静态和 moving scene
   都只有在 `track_kinds` 明确包含 `direct` 时才请求 full/direct 双轨；SE/TSE/SS 的
   full+early 请求只生成 full，early 在共享 renderer 中从 full 派生。需要 direct 的
   AEC、SSL 或显式 direct 请求仍生成双轨。GPU 子进程为请求的轨道调用 `simulateRIR`；moving
   轨迹会把多个帧的 source positions 按内存预算合批调用，再恢复为 `[P,A,C,L]`，
   不再为每个轨迹帧单独启动一次 backend。
   非 spatial legacy 路径的 full-only 请求使用 `moving_full` 轨道，不再额外生成未消费的 direct；仍得到 `[K,L]`，不改变 `C=1` 合同。
4. 返回 CPU float32 RIR 及 room、source、microphone topology、profile/custom identity、beta
   metadata；full/early 卷积继续使用共享 C-0 算子。
5. provider 对已请求的 full/direct、全部 actor/channel/frame 使用唯一共同 peak gain，保留轨道间相对幅度和到达时刻；online early 由显式 `early_rir_sec` 窗口派生，offline early 则校验并读取独立资产。后续 source-volume 与 SIR 为显式训练电平控制。

TSE target-absent 场景会渲染 latent target 作为 SIR/SNR 标定参考，再从 observation 移除，不把它变成监督目标。enrollment 是独立单通道近场增强，不属于 microphone scene 的 actor。`noise_scp` 走 observation-side image 路径，不向 RIR provider 申请噪声 actor。AEC 的 near-end/loudspeaker 在同一 plan 内分别渲染，far-end reference 仍是 mono；当前 echo 输入来自录制 pair，因此这是录制 echo 的空间增强，不能等同于从干净 far-end 合成的完整扬声器播放链。

## 进程与资源所有权

GPU 计算被隔离在一个独立子进程里：训练进程和 DataLoader worker 只持有轻量客户端，不会各自占用显存。

`OnlineRIRProvider` 在 Dataset owner 构造时取得客户端。core 启动非 daemon 的 spawn 服务；只有该子进程设置自己的可见 GPU 并导入 gpuRIR。普通 DataLoader worker 不持有 CUDA 对象，只使用可 pickle 的客户端地址和请求参数。

服务内部维护一组常驻生成 worker（默认 2 个，可用 `USE_TRAIN_GPURIR_WORKERS` 覆盖，取值 1–8）。worker 在服务启动时一次性完成解释器、gpuRIR 导入与 CUDA 初始化，并用一个最小 plan 预热；此后每个请求经 duplex Pipe 派发到空闲 worker。旧实现为每个 cache miss 新建一个 spawn 进程，实测单次请求约 2.2 s，其中约 99% 是进程与 CUDA 启动成本，gpuRIR 本体只有约 20 ms；常驻池把稳态降到约 20–30 ms。超时、管道损坏或 CUDA OOM/非法访问等 backend 致命错误时只回收并重建该 worker（看门狗语义保留，失控的 gpuRIR 调用仍可被强制终止），普通预算/几何校验异常继续复用健康 worker。

服务端按池容量并发处理 socket 请求：每个 handler 线程阻塞在一个空闲 worker 上，单个慢请求不再阻塞其余 DataLoader worker。cache 为线程安全的有界 LRU；同一 canonical scene 的并发 miss 使用 single-flight，只由一个请求生成，其余请求等待同一结果，避免重复占用 GPU。

请求采用私有目录中的本机 socket，JSON 描述加 float32 bytes，校验 request ID、group ID、shape、长度、有限值和 SHA-256。协议不从 socket 读取 pickle。请求有超时，监听 backlog 有上限，LRU 同时受 entry 数及 `USE_TRAIN_GPURIR_CACHE_BYTES` 字节预算限制（默认 32 MiB）。

scene IPC 的生成前校验还包括房间内绝对坐标、actor order、trajectory frame count 和 waveform timebase。服务返回请求轨道的 track kind、scene identity、原始内容 hash；moving 逐 `[A,C,L]` 帧传输并复核整组摘要。provider 再统一归一化，不允许 service 分别缩放 full/direct 轨道。

### 生产可观测性

受管理 RIR service 对每个响应累计并发布以下指标。`RIRClient.metrics_snapshot()` 返回最近一次响应中的 service 累计快照；每条已请求轨道的 `facts` 同时包含 `request_metrics` 和 `service_metrics`。指标中的 bytes 是 JSON request payload 与 float32 waveform payload，不包含 socket framing。

```text
rir_request_count
rir_cache_hit_count
rir_cache_hit_rate
rir_generation_count
rir_generation_latency_ms
rir_generation_latency_ms_total
rir_queue_wait_ms
rir_queue_wait_ms_total
rir_bytes_transferred
rir_bytes_in_total
rir_bytes_out_total
rir_failure_count
rir_timeout_count
rir_service_gpu_id
rir_service_pid
rir_service_started_at
```

单次 request 事实还包含：

```text
request_kind
track_kind
cache_hit
queue_wait_ms
generation_latency_ms
bytes_in
bytes_out
gpu_id
```

低于 IPC frame 上限的 legacy 测试请求如果无法容纳可选 metrics header，service 仍累计内部指标并优先保证 waveform framing；正式生产 frame budget 默认足够发布完整快照。

服务按 owner PID、device 与执行参数复用；训练入口 finally 释放本进程服务，standalone 使用者可显式调用 `close_online_rir_services()`，atexit 提供兜底。常规训练应在完整 epoch 结束后更新 epoch，避免把尚在预取的样本跨 epoch 改写。

## 随机性与审计

TSE/SS 与在线 SE/AEC 的几何/选择种子绑定 global seed、split、epoch、index 和用途；worker/GPU 调度顺序不参与几何种子。validation 的场景 epoch 固定为 0。GPU 漫射尾部的逐位确定性依赖 backend，不能把固定几何等同于跨进程/版本 bit-exact 波形；同一服务内相同计划的 cache 命中复用同一原始数组。

`online_rir_scene` 包含 group UID、room_dim、mic_loc、source_loc、dist、rt60、beta、Tdiff、image counts、原始/归一化 RIR 摘要和 normalization scale，并记录 `ArrayTopology` 的 profile/custom identity、`positions_m`、spacing、coordinate system 与 reference channel。RIR、scene、manifest 的 `geometry_provenance` 可选记录 validation recipe 及 `external_repository`/`external_path`/`external_commit`（短键 `repository`/`path`/`commit` 会被拒绝）；这些字段只用于追溯，不改变物理 profile。TSE/SS scene facts 还记录七路组成、SIR/SNR 测量点、D/E/F 与最终分解关系。

## 验证与性能

`gpuRIR-online-v4` 引入完整 scene 请求、独立 actor moving 轨迹以及明确的 `L/T` 分离。
`tests/test_scene_provider_end_to_end.py` 的 unit 选择覆盖真实 Unix socket、spawn worker、provider 和 renderer，CUDA 选择则不替换 gpuRIR，验证 static/moving 到达时延及固定 loudspeaker。CPU 协议通过与 CUDA 通过分别记录。下面两档中，`-m unit` 只跑 CPU 协议，`-m 'integration and gpu'` 会启动真实 GPU 服务，需要可用的 gpuRIR CUDA 扩展：

```bash
python -m pytest -m unit tests/test_scene_level_rir_ipc.py tests/test_scene_provider_end_to_end.py -q
python -m pytest -m 'integration and gpu' tests/test_scene_provider_end_to_end.py -q
```

SSL 移动配置使用 `configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml`：split 的
`trajectory_points: 50`、`max_displacement_m: 0.25` 描述轨迹，
`simulation_cfg.prob_moving_rir: 1.0` 控制移动概率。概率为零时仍使用 exact
static full/direct。轨迹点数不超过 128，整组双轨 payload 不超过 256 MiB；
moving 按单个 `[K,C,L]` frame 分包，每包不超过 64 MiB，逐帧及整组均校验 hash。
缓存按请求类型、完整 plan 和 trajectory/timebase 区分，禁止静态/移动混用。

`core/data/simulation/gpurir_backend.py` 将 moving 轨迹按输出预算合批送入
`simulateRIR`；若 gpuRIR 内部 allocation 超预算，会自动减半批大小并重试，最终仍受
原有 64 MiB 单轨输出上限约束。`core/data/simulation/moving_scene.py` 按发射时刻分段卷积后 overlap-add，
跨段尾部保留，并在一个 source segment 内复用 source FFT 到全部 microphone channel；未请求 direct 时不分配
direct waveform，与本地 gpuRIR `simulateTrajectory` 语义一致。SSL 的 STFT/IPD
在投影层计算，DOA 使用 STFT 窗中心对应段的 source position；direct waveform
不经过 observation D/E/F。该标签策略是显式分段近似，未宣称连续运动的 Doppler 模型。

`gpuRIR-online-v3` 修正 direct 请求：`nb_img=(1,1,1)`、`Tdiff=Tmax`。
`Tdiff=0` 会启用全程漫射，不能用于表示直达声。
沙箱内 CUDA/NVML 失败必须先在获准的沙箱外复核，不能误报主机没有 GPU。

移动轨迹相关的 GPU 验证：

```bash
python -m pytest -m 'integration and gpu' tests/test_gpurir_moving_end_to_end.py tests/test_moving_rir_ipc.py -q
```

```bash
python -m pytest -m unit tests/test_online_rir_provider.py tests/test_tse_seven_scenes.py -q
python -m pytest -m 'integration and gpu' tests/test_online_rir_provider.py -q
```

第一条不需要 GPU；第二条显式运行真实服务、K=1/2/3、Dataset 与 spawn workers。需要安装可用的 gpuRIR CUDA 扩展；基础 import 与 manifest 模式不依赖它。以上命令都以「全部通过」为预期结果，失败说明对应的在线路径或 IPC 合同被破坏。

在线生成会与训练竞争 GPU，吞吐需结合实际 K、RT60、房间、worker 数量测量。不要把小规模 smoke test 等同于长时间 DDP 训练压测或效果提升。可以选择独立 GPU，或调低最大 RT60/源数与预取深度；不能用静默减少人数或切换 backend 掩盖性能问题。

单次 gpuRIR 调用的内部 allocation 预算为 `GPURIR_MAX_WAVEFORM_BYTES`（256 MiB）。主导项是
ISM 分组缓冲 `K*C*ism_samples*ceil(prod(nb_img)/512)`，小房间加高 RT60 会让 `nb_img` 急剧增长，
因此该预算不能按「RIR 波形长度」直觉估算。超出预算时 `generate_rir_group_budgeted` 按 source
二分重试并沿 source 轴拼回，重建 `source_loc`/`source_ids`/`actor_order`/`rir_sha256`；单个
source 仍超预算时 fail-closed，不静默减少声源。分块只在整块原本会报错时触发，所以不改变任何
此前成功 plan 的输出；同一 plan 在同一预算下分块方式固定，结果可复现。gpuRIR 本身是
shape-dependent 的，不同分块深度之间不保证逐 bit 相同。

调参顺序建议：先用 `USE_TRAIN_GPURIR_WORKERS` 覆盖 worker 数；只有在确认 RIR 服务不再是瓶颈
后，再增加 DataLoader worker 数量，否则请求仍会在池前排成队列。
