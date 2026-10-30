# simulation — 仿真流程索引

本目录说明五个任务的「训练数据是怎么造出来的」：训练时现场组合出受损音频的动态仿真，以及直接读取已有音频的离线数据。

先认识几个反复出现的词：

- **仿真（simulation）**：不预先收集「带噪样本」，而是每次取样本时把干净语音、噪声和房间响应现场组合成受损音频，所以样本几乎不重复。
- **RIR（room impulse response，房间冲激响应）**：一段声音在这个房间里从声源传到麦克风的完整响应。把它和干净语音卷积，就得到带该房间混响的录音。
- **actor**：参与发声的对象，例如说话人或播放回音的扬声器；每个 actor 有独立的位置和轨迹。
- **SCP**：本项目通用的一行一条音频清单（严格四列 `uid fs length path`），告诉 Dataset 有哪些素材可用。
- **IPC**：进程间通信。在线生成 RIR 时计算跑在独立 GPU 子进程里，Dataset 通过本机 socket 请求结果。

动态空间路径统一使用 scene sampler、scene-level provider 和公共 waveform renderer；online static/moving 已接入分段 IPC。移动场景使用固定阵列、各 actor 独立的线性轨迹，共享 room 和 timebase，不代表移动阵列或完整追踪系统。

Dynamic waveform Dataset 的共享基础也按 seam 分层：`ResourceStore` 负责资源 I/O，
`SpeechPreparation` 负责 Layer A，`ScenarioSampler` 负责通用 Layer B 分支抽样，
`AcousticRenderer` 负责 Layer C bundle 投影；任务 Dataset 继续保留任务语义、监督投影和 batch 合同。

## 任务接入边界

| 边界 | 调用方责任 | 共享实现责任 |
|---|---|---|
| task policy / request | 声源身份、target/cue/PIT/DOA、需要的背景与 role | `SceneRenderContext` 绑定同一资源 owner；`SpatialSceneExecutor` 统一 RIR 执行，scene renderer 统一 actor、geometry、placement、mono/spatial 渲染。 |
| Layer A–F 配置 | 使用已校验任务 view 的所需切片 | `simulation_views.py` 只投影严格解析结果；正常 speech preparation 消费 `shared.layer_a`，不复制校验和默认值。 |
| observation post-mix | 声明 observation/protected/derived role 与专属策略 | `PostMixEngine` 固定 D/E/F 顺序；空间 Layer F 使用 `apply_post_augmentations_channels`，保留 shared/per-channel 参数与 RNG 合同。 |
| 资源与生命周期 | 任务只持有所需显式接口 | SSL 通过 `SSLSpeechResourceAdapter` 访问资源、sampler 元数据、strict-load 和 epoch/worker 生命周期；不以宽泛 `__getattr__` 透传整个 Dataset。 |
| batch / trainer | 五任务 collate 直接返回已校验的任务 Batch，实现 `validate()` / `pin_memory()` / `.to()` / `batch_size` | `BatchContract` 搬运后维持对象合同和主样本数；未知 dataclass 不做字段反射，GAN dict 单独按 `clean` 主轴处理。 |
| 样本提交 | 任务返回 supplied trace 与样本，不自行 seal | Dataset 统一 seal/publish；`sample_state.py` 在失败时恢复上次完成的快照，包括嵌套标量与数组 shape/dtype。 |

新增任务优先消费这些边界；保留任务专属物理语义，禁止重复实现背景、随机数映射或声学渲染。
分层职责与任务投影见 [仿真流程说明](Simulation_Flow_Logic.md)。

`fixed_array_ipdnet_dynamic_moving.yaml` 是 native `FixedArrayIPDNet` 的 moving 配方；对应的 `configs/ssl/spatial/fixed_array_ipdnet_dynamic_wind.yaml` 叠加空间加性 wind 背景。正式 dynamic 的 split 提供 speech/noise 与 `simulation_cfg`；online 不依赖 RIR SCP，offline RIR 必须提供 typed scene manifest。`prob_moving_rir` 在五任务的 online 空间路径中控制移动概率；SSL 的 split `trajectory_points` 和 `max_displacement_m` 控制分段数与位移上限。

## RIR 语义角色与 backend 矩阵

**Seam** 是任务与 core 之间的固定接口：任务只提交请求，怎么渲染由 core 负责。RIR 分三条轨道——full（含房间全部混响，用来合成模型输入）、early（只保留早期反射，作为监督目标）、direct（只有直达声，SSL 用它算方向）。

当前空间仿真的公共 Seam 是 `core.data.simulation.spatial_scene.SpatialSceneRenderer`：
调用方提交 `SpatialSceneRequest`，由 core 隐藏 provider、full/direct/early 轨道、
moving trajectory、IPC、background 和 waveform renderer。`core.data.simulation.observation_post_mix`
提供 observation-only 的 D/E/F Module。所有 spatial Dataset owner（SE/AEC/TSE/SS/SSL）
先通过 `core.data.simulation.rir_provider.build_spatial_rir_provider()` 解析 split-level
资源；factory 统一 `source_uids`、`track_kind`、normalize、geometry 与 hash/provenance，显式
provider 必须实现 `sample_scene_tracks` 或 `sample_moving_scene_tracks`；正式 renderer 不接受旧 callable/source-count 自动回退。SSL 已通过任务 Adapter 使用这两个 Seam；SS
和 TSE 仍保留各自的 SIR、enrollment、PIT 和 task-specific post-mix 编排。

下表按**声学角色**说明 RIR 怎么用（同一套 core 实现，不同任务扮演不同角色）。常见缩写：SER（signal-to-echo ratio，近端语音与回声的比值）、ERL（echo return loss，远端参考信号到回声的衰减）、SIR（signal-to-interference ratio，目标语音与干扰语音的能量比）、PIT（permutation invariant training，分离任务不知道哪一路输出对应哪个源，用排列不变的方式两两配对算 loss）；**注册音（enrollment）** 是 TSE 用来指定「要提取谁」的一小段该说话人录音。

| 角色 | 任务 | 语义 | 来源 |
|---|---|---|---|
| speech-source | SE/AEC/TSE/SS | full 卷积前缀截断；early 窗以 direct path 锚定（监督口径）；moving 分段 overlap-add；噪声不加 RIR | `rir_backend=offline|online`，四个 waveform 任务共享 core 卷积实现；SSL 通过 spatial adapter 消费同一 scene seam |
| echo-path | AEC | 回声源来自录制 farend/echo pair；空间模式以独立 loudspeaker RIR 做空间增强，再保持 SER/ERL，不能称为从干净 far-end 合成的完整播放链 | 配对清单与同场景 loudspeaker actor |
| enrollment-near | TSE | 注册音近场契约（物理上 5-20cm）；full-RIR 语义，无 early 概念 | offline 近场池或 `online` 近场生成（`gpurir_enrollment_distance_range_m`，默认 `[0.05, 0.2]`） |

backend 差异：动态空间 `offline` 从 split-level `rir_scp` 的 typed scene manifest 读取真实 room、geometry、actor positions 和轨道身份，不随机替换缺失的几何；`online` 先采样完整 scene，再生成同房间不同 actor 的 RIR。两种 backend 都校验 sample rate、source order、track kind、geometry 和内容摘要。旧 mono RIR 池不被伪装成 typed spatial scene；缺少身份的数据应补真实元数据或重新生成。

## 已物化 offline observation 与模型几何

「已物化（materialized）」指音频文件、清单和几何元数据都已提前生成好，训练时只读盘、不再现场仿真。下文用 `[C,T]` 表示 C 个麦克风通道、每通道 T 个样点（`C=1` 就是常说的 mono 单通道），`[T]` 表示单通道波形。

`offline_aec` / `offline_tse` 的空间 reader 在 split 配置中声明 geometry，不把它混入动态 `simulation_cfg`：

```yaml
data:
  train:
    spatial_enabled: true
    array_profile: linear_4ch_40mm
    reference_channel: 1
    channel_order: [mic_0, mic_1, mic_2, mic_3]
    # AEC 的 mic_scp / farend_scp / nearend_scp，或 TSE 的 samples_jsonl / cues_yaml
```

`channel_order[i]` 是文件第 i 个通道的物理 ID，必须与 geometry 的第 i 个位置对应；只声明顺序，不触发重排。reader 校验 split 的音频 header，并在实际读盘时再次校验 C。静音 placeholder 也保持 `[C,T]` observation，其余 target、far-end、enrollment 保持 mono。波形本身无法证明物理 microphone ID，排列声明应来自数据准备过程。

SE/SS/AEC/TSE training entry 在模型构造前，从 Dataset 的已解析 geometry 向显式接受该参数的模型注入 `ref_channel` / `reference_channel` 和 `n_channel`（前者指定多通道 observation 中当作参考、并作为 mono 输出的那一路，后者是麦克风通道数）。用户显式填写的值必须一致；注入值写回 resolved training config，供 checkpoint 重建。train/validation 必须保持同一 C/reference 输入合同，不要求同一房间。模型没有原生 microphone axis 时仍拒绝 `C>1`。

## 噪声读取与能量标定

SNR（signal-to-noise ratio，信噪比）决定背景比语音低多少 dB；本节说明噪声文件以什么通道形态进入仿真，以及用哪个能量去标定（下文的 image 指噪声在 observation 侧展开成各麦克风通道后的形式）。公开资源入口仍然只有 `noise_scp`。空间任务保留 noise 文件全部通道及同步裁剪窗口，不复用会取首通道的 mono speech/RIR 解码结果。ordinary noise（走普通加性噪声路径的噪声；wind、colored 等成分另有专用增强）的执行顺序为：读取 `[T]` / `[1,T]` / `[C,T]` → image adapter → 一次全 image SNR gain → observation。目标 C=1 时原样返回 `[1,T]`，不消耗空间随机数（RNG）；已匹配阵列的 `[C,T]` 不再次空间化或卷积 RIR。

SNR 使用配置选定的 primary/all-speech image 全通道能量。reference-channel SNR 可单独作为诊断值，不能与全 image 标定值混写。wind、colored 等已有独立增强保留各自的电平策略；加入这些成分后，最终背景总 SNR 不必等于 ordinary 成分的 nominal SNR，审计分别记录。

## 当前入口

- [Task_Mode_Channel_Comparison.md](Task_Mode_Channel_Comparison.md)：五任务 × dynamic/offline × 单通道/多通道的横向对比速查（谁是 `[C,T]`、监督目标形态、trace 位置），细节仍回到下面两份正文。

- [Online_gpuRIR.md](Online_gpuRIR.md)：core 在线同房间 RIR、开关、进程生命周期与验证。

- SS dynamic 的 Layer B、sampling 和 source-scene audit 也遵循 [多源共享仿真](Multi_Source_Simulation.md) 的固定 K 合同。

- [Simulation_Flow_Logic.md](Simulation_Flow_Logic.md)：仿真语义的主文档（五任务总览）、命名空间边界、task-neutral role、任务物理投影和 Dataset 输出合同；Dynamic post-mix 的阶段顺序、Layer D 定义、AEC 模型输入前端和 `TimeTransformPlan` 的权威描述也都在该文（0.1、0.5、2 节），本目录其他文档只补充各自独有的细节。其中 AEC 的 nearend target 经过 Layer D、仅绕过 E/F；导出的 echo audit stem 同步经过 `D-normalize` / `D-guard`，同样不进入 E/F；E/F 只修改 task-neutral observation，在 AEC 中投影为 mic。
- [Multi_Source_Simulation.md](Multi_Source_Simulation.md)：TSE/SS/SSL 多源管线、post-RIR SIR、timeline、近端 enrollment、固定 K 与当前实现边界。
- 五任务分层图：[PNG](../../diagrams/se_aec_tse_simulation_layers_current.png) · [DOT source](../../diagrams/se_aec_tse_simulation_layers_current.dot)。图覆盖 SE/AEC/TSE/SS/SSL 的 mode 分流、共享 Layer A–F、offline reader 与 collate/runtime 边界。
- [sim_pipeline_layers_current.png](sim_pipeline_layers_current.png)（[DOT source](sim_pipeline_layers_current.dot)）：五任务动态 Layer A–F 参考图，明确 Layer D 是 post-mix amplitude conditioning，内部固定 `D-normalize → D-guard`；PNG 可用该 Graphviz source 重现。
- AEC 分支图 [run_branch_order.png](../aec/assets/run_branch_order.png)（[DOT source](../aec/assets/run_branch_order.dot)）描述七个 Layer B 场景分支汇合后的共同顺序。

两份正文、主文档内五任务 Mermaid、代码路径与配置契约共同提供当前导航；图片不单独定义接口，代码和测试是最终 SSOT。更具体的 AEC 入口见 [`../aec/README.md`](../aec/README.md)。
