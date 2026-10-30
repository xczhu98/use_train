# 多源共享仿真与时间布局

本文说明一次样本里有**多个语音源**时（TSE 的目标说话人加干扰人、SS 的固定 K 路源、SSL 的固定 source tracks），仿真怎么安排它们的时间位置、电平关系和监督目标。SE/AEC/TSE/SS 共用语音预处理、背景源选择和 waveform 声学算子；SSL 共用语音准备与 spatial scene seam，但 noise pool 由 SSL 合同显式管理。TSE/SS/SSL 的多语音源场景复用 shared scene seam；TSE 只负责目标身份、目标缺席和 enrollment，SS 只负责固定 K 源及输出集合，SSL 负责 spatial observation 与 DOA/VAD 投影。

术语：

- **多源（multi-source）**：一次样本里有多路语音，需要额外决定「谁在什么时候说话」。
- **timeline / 相对窗口**：每路语音在画布上的有效时间范围；`prob_timeline` / `overlap_ratio_range` 控制其他语音相对第一路画布的位置。
- **SIR（signal-to-interference ratio）**：目标语音与干扰语音的能量比（dB），越大目标越突出。
- **enrollment（注册音）**：TSE 用来指定「要提取谁」的一小段该说话人录音。
- **latent target（隐式目标）**：只用于电平标定、不进入 mixture 也不作为监督目标的临时目标轨道。
- **fixed K**：SS 每次固定输出 K 路源，K 由 `ss.num_sources` 指定，不支持可变路数。
- **PIT（permutation invariant training）**：分离任务不知道哪一路输出对应哪个源，按所有配对方式算出最小 loss 来训练。

## 能力与支持边界

| 项目 | 当前状态 |
|---|---|
| 五任务共享语音 Layer A、背景源与 DSP | 已接入；场景编排仍由任务 Adapter 决定 |
| TSE/SS/SSL 逐源画布与多源时间布局 | 已实现；TSE 只保留七路场景入口，旧实现已删除 |
| TSE 0..N、身份去重、目标缺席、独立 enrollment | 已实现；模板还需填写 speech/enrollment 数据资源 |
| SS 固定 K、可选身份、worker epoch 与批合同 | 已实现；不支持 variable-K |
| 单人 gap 与 TSE/SS/SSL 多人窗口 | 已实现；窗口占比不等于 VAD 重叠率，不是完整多轮对话规划器 |
| 同房间不同位置的几何一致性 | online gpuRIR 与 typed offline scene manifest 都绑定同 room/array/actor positions；legacy C=1 mono RIR 池不提供此 scene 合同 |
| 原始录音级 enrollment 隔离 | 已按 `(dataset_id, recording_id)` 排除同录音的不同路径；启用身份的 dataset 必须覆盖全部 speech/enrollment 路径，缺失或冲突直接报错。Libri2Mix producer v2 显式生成录音身份；无身份资源仅保证路径隔离 |
| 多源 trace | 提供 scene/source metadata 与最终轨道 |
| SE spatial offline | 已支持 mono clean + channel-first `[C,T]` noisy；split 显式声明 geometry、reference channel 与 channel order |
| MC simulation evaluation set | 有独立生成器与 SE 评估管线入口，维护生成协议、manifest 与数值合同；真实 WSJ0 评估需实际语料与对应模型 |

## SE spatial offline 与 mc_sim 评估

SE offline spatial 数据由配对 clean/noisy reader 加载：clean 必须是 mono，noisy 可为 channel-first `[C,T]`。split 必须显式提供 `spatial_enabled: true`、`array_profile` 或 `custom_geometry`、`reference_channel` 和 `channel_order`；noisy 文件通道数须与 geometry 一致，reader 不重排通道，也不把 mono noisy 扩展成阵列信号。collate 仅在 Dataset 声明 spatial 时允许多通道 noisy。模型输出与 clean 监督仍为 mono/reference-channel 语义。

`tools/prepare_data_tools/generators/se/generate_mc_sim_testset.py` 生成 WSJ0-mix 风格的静态双源多麦克风评估集：分别从 target/interference 四列 SCP 抽取 mono 片段，经共享 spatial renderer 使用在线 RIR 渲染，按指定 full-image SIR 混合，输出 `mixture.scp`/`noisy.scp`、mono `clean.scp`、多通道 `target_image.scp` 和 `mc_sim_manifest.json`。两份输入 SCP 的 UID 必须不重叠；此校验不验证 speaker-disjoint，身份隔离仍由数据准备者保证。生成器不会覆盖已有输出目录。

`evaluate/eval_pipeline_se.py` 可通过 `--datasets mc_sim` 评估，或通过 `--datasets all --include_mc_sim` 纳入全套评估；缺少已生成 manifest 时，all/inference 流程可用 `--mc_sim_target_scp` 与 `--mc_sim_interference_scp` 触发生成。评估会校验 manifest、SCP 与音频 payload 的摘要和 inventory，并对同一套 mixture/target image 计算 oracle MVDR 基线。该数据集只是 WSJ0-mix 风格的生成协议；本机 WSJ0 语料未准备时，不应称为真实 WSJ0 评估。

## 时间布局只有一个单人入口

单人 timeline 已由共享 Layer A 实现，不再增加另一组随机 gap 或把完成的语音重新切成多个 turn：

| 参数 | 控制内容 |
|---|---|
| `speech_splice_gap_range` | 同一路中不同 utterance 的间隔；沿用现有 log-uniform gap 抽样 |
| `repeat_tile_silence_gap` / `repeat_tile_silence_gap_range` | 显式重复语音时的间隔 |
| `silence_pad_prob` / `silence_pad_position` | 单句在固定画布中的一次放置；命中时不 splice、不重复 |
| `silence_pad_mode` / `silence_pad_*_range` | 沿用现有静音分布合同；fixed canvas 的总留白由画布长度减语音长度决定 |

SE/AEC 的单路近端和 TSE/SS/SSL 的每路 source 都走同一个 `_prepare_clean_speech_result()`。同一 identity 内可以 splice；匿名 SS source 只复用自己，不能假设其他文件来自同一个人。

多人关系另外用 `prob_timeline` / `overlap_ratio_range` 控制：保持第一路画布，将其他语音的有效首尾范围放到相对窗口。它不插入第二轮随机 gap、不重复短语音；零窗口产生零样点。ratio 表示窗口相对画布的长度，真实活动重叠率另行计算，不能冒充 VAD overlap（VAD = voice activity detection，语音活动检测）。当前活动审计采用逐样点相对峰值 -40 dB 阈值，不是语义 VAD。

该关系规划仅在有多个语音 source 时生效。AEC 回声/reference 必须保留已有配对、SER/ERL 和 delay 合同，不能把它们当作两个互不相关的说话人随意移动。`farend` 始终保持 mono reference；已有 echo 若是 `[C,T]` microphone-side image 就直接加入 observation，不再卷积 RIR；mono echo source `[T]` 只有在需要空间差异时才经独立 loudspeaker RIR 映射。不要求从 clean farend 重新合成完整 playback echo chain。

## 模块职责与轨道

| 模块 | 职责 |
|---|---|
| `core/data/source_pool.py` | 可选 canonical identity、不同身份采样、明确的不足策略 |
| `core/data/speech_scene.py` | 相对窗口计划、无随机 apply、活动统计、seed/epoch/index 子流 |
| `core/data/background_sources.py` | 四个 waveform 任务共用 ordinary/music 选择与读取；双源禁止 Music+Music；SSL 由 spatial supervision 合同读取显式 noise pool |
| `core/data/simulation/background.py` | 共用 C-1/C-2/C-3，输入语音 image 与 SNR reference 分离 |
| `core/data/simulation/scene_renderer.py` | 每个 source 的 RIR/target、post-RIR SIR、场景 D–F |
| `tasks/tse/data/scene_dataset.py` | 目标身份、缺席场景、可选目标集合、注册音 |
| `tasks/ss/data/dynamic_dataset.py` | 固定 K source，共享 worker/trace 生命周期，SSBatch 数据 |

### 多源 neutral role 与任务投影

| neutral role | TSE | SS |
|---|---|---|
| `speech_source` | 注册目标 + 0..N 个干扰身份 | 固定 K 个 source |
| `designated_target_source` | enrollment 指定身份 | 无唯一目标，K 路均为监督 source |
| `source_full_rir/N` | target/interference 的 full-RIR track | source 的 full-RIR track |
| `target/N` | 当前查询目标或 alternate target 的 early/dry 监督 | K 路 early/dry PIT 监督 |
| `background` | additive noise/music/wind；不含 interference | additive noise/music/wind；不含其他 source |
| `observation` | mixture | mixture |
| `reference` | enrollment | 无 |

多源 bundle 同时保留 `source_full_rir/N`（实际进入线性 mixture 的 full-RIR 语音）、
`target/N`（early 或 dry 监督）、`background` 和 `mixture`。这些是 rendered track
名称，不是对所有任务物理角色的重命名。D-normalize 与 D-guard 对相关轨道使用保持
分解的共同/参考通道增益；E/F 及 final guard 只处理 task-neutral `observation`，在 TSE/SS
中投影为 `mixture`。trace 同时记录 D 后 target invariant、observation stages 后 invariant
和最终 decomposition 是否保持。

Spatial TSE/SS 只在 task capability 允许的 background 分支上运行；当前不具备一致 spatial image/audit 语义的 C-2 side-chain wind 与 C-3 colored-noise 组合在 config/view projection 阶段 fail-fast，不会在生成 observation 后静默丢弃。

`target_type=early` 时，不能用 `sum(targets)` 验证麦克风分解。线性阶段应检查 `mixture = sum(source_full_rir/N) + background`；sidechain 风噪或 observation 非线性退化后，该等式可能不再成立。

TSE 的主监督目标始终是注册目标说话人的 early/dry waveform，或目标缺席场景的
zero target。部分 loss 还会按需使用 complement reference：共享 trainer 对
`FWNegSISDRLoss` 或启用 `enable_ss_undesired_energy` 的 Composite component 构造
`undesired_ref = mixture - target`、`undesired_enhanced = mixture - target_estimate`。
它表示“相对当前监督 target 的剩余内容”，可能同时包含 interference、background、
target 的 late-RIR 差异和 observation degradation；不能命名成纯“非目标说话人
residual”。从 `source_full_rir/N` 与 `background` 单独推导的 interference-only
参考目前仍只用于物理分解审计，不是 TSE Dataset 的第二个监督 target。

## Layer B：统一场景命名

Layer B 决定这次样本里有哪些角色（目标语音、干扰人、背景噪声、回声）。四个 waveform 任务的 Layer B 直接使用 `prob_*` canonical scene ID；SSL 不抽 waveform scene branch，而是固定 source tracks 并把 scene 几何投影为 feature-domain target。配置字段、运行时
`branch`、`SceneSpec.scene_name`、trace、日志和统计使用同一个字符串，不再有
“旧 branch → display name”的映射层。`background` 表示 additive noise / music /
wind / colored noise；AEC 的 `echo` 是独立声学角色，不并入 background。

| 任务 | Layer B canonical 场景名 |
|---|---|---|
| SE | `prob_target_only` / `prob_noise_only` / `prob_target_noise` |
| AEC | `prob_target_only` / `prob_noise_only` / `prob_echo_only` / `prob_noise_echo_only` / `prob_target_noise` / `prob_target_echo` / `prob_target_noise_echo` |
| TSE | `prob_target_only` / `prob_interference_only` / `prob_noise_only` / `prob_interference_noise_only` / `prob_target_noise` / `prob_target_interference` / `prob_target_interference_noise` |
| SS | `prob_target_only` / `prob_target_noise` / `prob_noise_only` |

这里不建议把四个 waveform 任务的概率配置字段硬改成同一组名称：SE/AEC/TSE/SS 的 target、
speaker identity 和 echo semantics 不同。统一的是 canonical ID 的表达方式、trace
字段和统计口径；允许哪些 ID 仍由 task contract 校验。SS 的 `prob_noise_only` 保持
固定 K 输出合同：K 路 target 全零，`source_mask` 仍为 K 路有效，mixture 只包含 background。

## Layer B：TSE 七路互斥场景

T 是 enrollment 指定身份，I 是其他身份，N 是背景噪声/音乐。一次抽样决定 T/I/N 存在性，七项概率总和必须为 1；不再使用隐式余量或目标缺席二次 gate。

| 场景 | 概率字段 | target |
|---|---|---|
| T | `prob_target_only` | 目标 early/dry clean |
| I | `prob_interference_only` | 零 |
| N | `prob_noise_only` | 零 |
| I+N | `prob_interference_noise_only` | 零 |
| T+N | `prob_target_noise` | 目标 early/dry clean |
| T+I | `prob_target_interference` | 目标 early/dry clean |
| T+I+N | `prob_target_interference_noise` | 目标 early/dry clean |

`num_interferers: {1: 0.95, 2: 0.05}` 仅在含 I 的场景抽样；key 必须为正整数，概率和为 1。不同 utterance 的同一 identity 不能算两个人，身份不足报错。`targets_per_mixture` 是监督查询数上限，不是实际说话人数。零目标场景仍提供缺席目标的 enrollment；三个无 T 的场景需要非零 `AbsentTargetLoss` 或 `L1WavLoss`，仅 SI-SNR（scale-invariant signal-to-noise ratio，尺度不变信噪比，语音增强常用的波形相似度指标，越大越好）不提供零目标抑制梯度。

Layer B 控制 C 阶段源组成：N=false 时不生成背景；N=true 时必须产生有效背景。D/E/F 仍可退化 observation，不会把噪声加入 clean target。目标缺席时仍读取注册身份对应的 clean speech：纯 N 将其作为隐式 SNR reference；I-only / I+N 还将其经过 C-0 得到 latent target full-RIR track，作为 `sir_range` 的 numerator。latent target 不进入 mixture、`source_full_rir/N` 或监督 target，最终 target 仍为全零。I+N 的背景 SNR 继续以实际干扰语音和为 reference。

## Layer C：离线 RIR 与在线 gpuRIR

`rir_backend` 是唯一来源开关：

- `offline`：C=1 legacy mono 模式可以从普通 `rir_scp` 逐源读取 RIR（moving 由池内 moving PKL 组成决定）；C>1 spatial 模式必须从 typed scene manifest 读取同一 scene 的 room、array、actor order 和 full/direct/early tracks，不允许把独立 mono RIR 池拼成阵列 scene；moving spatial scene 由 schema v3 manifest 表达（`moving: true` + 逐 actor 轨迹 + timebase + 轨迹哈希），运行时按 manifest 权威渲染，比例由 scene index 组成决定。
- `online`：core 在一个房间、一组墙面反射系数、一个麦克风下布置 K 个不同位置，一次 GPU 调用生成全部响应；spatial 与 legacy mono 都支持 moving，mono moving 由 `prob_moving_rir` 在 reverb 命中后控制。参数、进程服务、缓存和几何事实见 [Online_gpuRIR.md](Online_gpuRIR.md)。

SE/AEC/SS 按实际采样率与 split 使用公共清单：16 kHz 为 DNS5 RIR 与静态
gpuRIR 合并池（152,236 / 8,012），32 kHz 为静态 gpuRIR 池（95,000 /
5,000）。TSE mixture 单独使用
`data/shared/rirs/gpurir_static/gpurir/fs16000/tse/{train,val}_rirs.scp`，
只包含 `fs16000/static` 的 100,000 个纯 PKL（95,000 / 5,000，near+far）；
enrollment 保持 `static/near/{train,val}_rirs.scp`（47,551 / 2,449）。
PKL 内的 `room_dim`、`mic_loc`、`source_loc`、`dist`、`rt60`、`field`、
`uid` 用于审计单个离线响应的几何与来源，但独立抽样仍不等于同房间 group。

同组 online RIR 使用共同 full peak 归一化，保留相对传播增益和时延；full/early 不分别抹平电平。各源后续 C-0b 音量和 SIR 属显式训练电平控制，应与 RIR 传播增益分开解释。

TSE 的 `sir_range` 在所有含 I 场景生效：T+I 使用实际目标 full-RIR track；I-only / I+N 使用不进入 mixture 的 latent target full-RIR track，按注册目标与干扰总和的能量比缩放干扰人。SS 按每个后续 source 相对第一路标定。`snr_reference` 为 `primary` 或 `all_speech`；纯 N 使用注册目标 clean speech 作为隐式 SNR reference，不再使用绝对背景电平字段。SE/AEC 的 early-reference SNR 合同不因 backend 切换而改变。

## 任务能力与声学角色合同

`core.data.simulation.role_policy` 统一登记声学源角色、RIR policy 和背景 policy。它约束语义，不替任务模型虚构多通道能力；`render_speech_scene` 只接受显式 `AcousticActor`，并在入口校验角色、身份顺序、RIR policy 与 mono/spatial mode；这些字段写入 `acoustic_actors` facts，既有 `AcousticScenePlan` schema 保持不变。
进入 speech renderer 的 actor 必须显式使用 `mono_source` 或 `spatial_source`；`none` 只用于不进入 mixture scene 的 enrollment / digital reference。

| 任务 | 单通道观测 | 多通道观测 | 目标/干扰/回声类声源 | 背景默认策略 | 当前模型边界 |
|---|---|---|---|---|---|
| SE | 支持；语音源逐路 C-0，背景可直接 additive | 支持；同一 room/array scene 的 source×mic image | 每个进入场景的语音源各有自己的 `mono_source` 或 `spatial_source` | `diffuse_coherence`（空间） | SpatialNet 类模型消费 `[C,F,T]`；mono 模型仍只消费参考通道 |
| AEC | 支持；near-end 与 acoustic echo 各自走对应 C-0 | 仿真层可生成共享 room/array 的 near-end 与 loudspeaker echo scene | `nearend`、`loudspeaker_echo` 各自绑定 RIR；digital far-end reference 为 `none` | mono 为 additive，空间背景需显式成像 | 当前注册 AEC 模型/entry 仍是单通道模型合同；空间数据不能自动宣称可训练 |
| TSE | 支持；mixture 语音源逐路 C-0 | 支持；mixture 语音源共享同一 spatial scene | target/interference 共享 mixture scene；`enrollment` 使用独立 `nearfield_independent` | 空间默认 diffuse | BSRNN-SPK 的 streaming 路径仍按模型合同限制；Modular 的未实现组合继续 fail-fast |
| SS | 支持；每个 source 逐路 C-0 | 支持；所有 source 共用同一 room/array scene | 每一路 source 都是 `source_speech` actor，不能把独立 mono RIR 拼成阵列 scene | mono additive，空间默认 diffuse | SpatialNet/OnlineSpatialNet 消费多通道；TDANet/TIGER 保持各自单通道或 external-STFT 合同 |
| SSL | 不提供 mono 模式 | 仅支持动态空间观测 | source image 与空间特征来自同一 scene | diffuse coherence | 只消费空间特征，不接受 waveform mono fallback |

这张表区分了“仿真能生成什么”和“模型合同能消费什么”。目标语音、干扰人、near-end、loudspeaker echo 都是物理声源，进入同一 mixture scene 时应各自有对应 RIR；它们不必共享同一条 RIR，但应共享该场景的 room/array 几何。注册音频是 cue，不属于 mixture scene，默认保持独立近场 RIR。数字 AEC reference 是电信号，不能再套声学 RIR。

单通道 additive noise 不需要额外 RIR；多通道噪声必须有空间成像。当前默认 `diffuse_coherence` 是合理且稳定的基线，但它不是点噪声的替代品；需要可定位噪声时，应显式使用 `point_source_rir`，通过 `PointSourceRIRProvider` 提供同场景 source×microphone 静态 RIR，缺 provider 或几何不一致时直接 fail-fast。moving 点噪声使用 `MovingPointSourceRIRProvider`，其 RIR 轨道必须与共享 `TrajectoryTimebase` 逐帧对齐，并复用同一 moving scene 的 room/array 几何；不能把静态 RIR 伪装成 moving，也不能跨 scene 拼接轨道。

背景策略统一通过 `build_background_spec` 选择：没有显式 provider 时，空间观测使用 `diffuse_coherence`，单通道观测使用 `additive_mono`。生产路径由 `core.data.simulation.background_resolver.resolve_background_spec` 按 `prob_point_source_background` 门驱动：门打开时在同一 room 内按 speech 源相同的分布族采样一个噪声 actor（moving scene 共享 timebase 规划轨迹），构造派生 noise scene plan，经 `from_scene_plan` 解析出点源 provider 后注入；门关闭（默认 0）、样本无噪声或单通道观测时退化为默认策略且零随机数消耗，保持 seeded replay 字节一致。该门要求 `spatial_enabled=true` 且 `rir_backend=online`，配置边界显式拒绝非法组合。手动把已按当前 scene 解析出的 `PointSourceRIRProvider` 或 `MovingPointSourceRIRProvider` 注入 `build_background_spec` 或 `render_speech_scene(background_spec=...)` 的入口保留给测试与实验调试；它不从 YAML 猜测噪声位置，缺少 provider、room/array、轨迹或 timebase 身份时直接拒绝。

公共 speech renderer、SE/TSE/SS 的 mono speech ingress 以及 SSL 的 observation post-mix ingress 已使用 `AcousticActor`；AEC 在其专用 C-aec 组合中使用同一角色合同记录 near-end、loudspeaker echo 与 digital reference。各任务的 spatial scene plan 仍由 `AcousticScenePlan.actors` 负责房间/阵列几何，任务 adapter 继续负责 SIR、PIT、direct-path 和 AEC ERL 等专用输出编排。任务 entry 仍负责模型能力检查，拒绝“仿真已空间化、模型仍为 mono”的配置。

## 独立 enrollment

`enrollment_augment` 是唯一可选增强配置。动态注册音始终执行长度/RMS 准备，enabled=false 只关闭额外 speed/pitch/EQ/RIR/noise；offline 仍按自身 cap/cue 合同处理。旧 `*_enroll` / `enroll_*` 仿真字段与 `from_dynamic` 解析已删除。

注册音默认使用独立的近端静态 RIR 清单；mixture 在线时不自动把 enrollment 放进同一个房间。`rir_static_only` 是该独立 provider 的静态过滤选项。enrollment 读取失败、候选缺失或运行期 mutation 均是不可恢复错误，不能用整样本静音掩盖。

新场景排除 mixture/splice 已用的音频路径；speech JSONL 提供 `recording_id` 时，进一步按
`(dataset_id, recording_id)` 排除这些录音的所有文件/切片。启用该字段的 dataset 要求完整
speech/enrollment 身份覆盖，候选耗尽即失败。Libri2Mix producer v2 已生成并校验该字段，
其他语料需由其资源生产器声明真实录音关系；未声明时仅保证路径隔离。

## 删除的字段

下面这些字段已经删除，写在配置里会直接报错，也没有兼容映射：`prob_interference`、`interfere_num_distribution`、`interference_count_distribution`、`interference_shortage_policy`、`prob_target_absent`、`interfere_timeline_prob`、`interfere_overlap_ratio_range`、`source_rir_mode`、`scene_snr_reference` 及旧 enrollment gate。旧 timeline 两字段也已替换为 `prob_timeline` / `overlap_ratio_range`。即使 strict=false，也不能使用这些旧字段；没有兼容映射、旧路径回退或“减少人数后继续训练”。

## SS 配置与随机性

`configs/ss/tdanet/tdanet_dynamic_sisdr.yaml` 提供完整 A–F 配置，使用本机 URGENT clean SCP。没有 speaker metadata 时只能保证不同 source；严格不同人应使用含 `dataset_id/speaker_id` 的单人 JSONL。分类 label 和 enrollment 都不是 SS 必需字段。SS summary 的 `scenario_probs` 发布 `prob_target_only` / `prob_target_noise` / `prob_noise_only` display name，不复用 SE/TSE 的配置字段。

`ss.num_sources` 同时约束 Dataset、模型与 PIT。Layer B 使用 `prob_target_only` / `prob_target_noise` / `prob_noise_only` 三路互斥场景；三者总和必须为 1，背景场景要求 noise/music provider 存在；纯 background 时 K 路 target 全零。当前仍是固定 K，`source_mask` 不能含无效槽；默认 exhaustive PIT 的 `max_sources=4`。扩大 K 或支持可变 K 不能只改采样器。

SS dynamic 已接入共享 `sampling.speech`（language/dataset/speaker）和 `sampling.noise` controller，并使用 Layer A 的 `filter_by_duration`、`filter_min_duration`、`filter_max_duration`。speaker sampling 需要 source JSONL 提供 canonical identity；匿名 SCP 只能做 source 级别的 K 路抽样。validation 不接受有效 sampling 配置，sampling 只写在 `data.train`。

SS 每条样本保留 shared sampler 指定的 anchor UID，再从其他身份中抽 K−1 路；
speech quota 约束 anchor 曝光，不约束其作为其他样本伴随源的次数。身份池过滤使用
集合索引，避免对大规模 manifest 做平方级成员扫描。

`prob_noise_only` 显式传递固定画布长度，产出 K 路零 target，并记录实际
`SceneSpec.source_count=0`；audit 的 `scene.source_count=K` 表示固定监督槽位。
默认 `PairwiseNegSISDRLoss` 对静音参考使用有效长度内的预测均方能量，保留 DC
抑制梯度；有语音参考继续使用 SI-SDR。不能用零参考的固定 SI-SDR 下限冒充有效监督。

SS 的 source-scene audit 与 TSE 共用 `source-scene-stems-v2` envelope：scene 必须包含
`mixture`、`background`，并按 `scene.source_count=K` 提供连续且一一配对的
`source_full_rir/0..K-1` 与 `target/0..K-1`。所有轨道必须是非空一维 float32，
并绑定 scene 的 sample rate、长度与 SHA-256；SS 不允许使用 TSE 专用的
`target/absent`。TSE 的 `noise_only` 是例外：只有 `mixture`、`background` 和
`target/absent`，不要求任何 `source_full_rir/N` 或 `target/N`。

SS train/validation 的资源与 simulation_cfg 分别配置，不隐式继承。默认 strict_load=true；
显式 false 时只允许可恢复资源错误转为固定 K placeholder，并在 trace 标记失败。
schema、身份不足和在线后端错误仍抛出。split 中显式的 `segment_seconds` 与 `max_duration_seconds` 若同时出现必须一致。最终 sampler length 是画布长度。

新 TSE/SS/SSL 的 sample、source、acoustic、post-mix、enrollment 子流绑定 seed/split/epoch/index，不依赖 worker 分配；SS 使用共享 epoch tensor，因此 persistent workers 能观察新 epoch。validation 的场景 epoch 固定为 0；固定 seed/split/epoch/index 可以重现 CPU 几何与选择；GPU F 在训练侧执行，其事实不属于 Dataset trace。

## 事实读取

新 TSE/SS/SSL 的场景事实位于 `trace.metadata_dict()["scene"]`，其中包含 `source_plans`、`sir`、`activity`、`timeline`、`layer_d_facts`、`layer_e`、`layer_f` 与 `decomposable_final`；TSE 另有目标存在性和 requested/actual 人数。SSL 的 spatial renderer provenance 同样发布 `timeline` 与 RNG stream facts。capture 开启时通过 `trace.get_source("scene").stems` 读取最终轨道。它与 SE/AEC 的任务 audit 字段位置不同，不应把同一取值路径硬编码到五任务。

## 验证与发布

当前 SE generator version 为 2.11，TSE pseudo-target generator 为 2.0，AEC 为 2.18；RIR generator 为 1.12。新版本绑定
显式场景概率与 canonical 场景目录，旧 manifest 按既有 publication gate 重建。

七路审计导出使用 contract 1.4 的 `source-scene-stems-v2`，以
`source_full_rir/N` 明确表示实际进入线性 mixture 的 full-RIR source track；不再接受
`source_image/N`。主要测试位于 `tests/test_tse_seven_scenes.py`、
`tests/test_online_rir_provider.py` 与 `tests/test_multisource_scene_simulation.py`。
