# 🧭 SSL — Spatial Sound Localization

<p align="center">
  <strong>IPD/DOA/VAD 空间声源定位任务包</strong><br>
  复用共享静态/移动空间仿真 seam；项目原生 baseline 与官方 FN-SSL/IPDnet/IPDnet2 复现路径按显式合同分开。
</p>

---

## 📑 目录

- [任务定位与边界](#-任务定位与边界)
- [目录结构](#-目录结构)
- [数据与仿真](#-数据与仿真)
- [训练](#-训练)
- [模型](#-模型)
- [相关文档](#-相关文档)

---

## 🎯 任务定位与边界

SSL（Spatial Sound Localization，空间声源定位）回答的是「声音从哪个方向来」，不改变音频本身：输入固定麦克风阵列录到的多通道音频，输出每个声源在每一帧、每个频点上的方向。
SSL 是五任务仿真体系中的唯一 feature-domain 任务，公开模式只有 `dynamic_ssl`，不提供 waveform offline 或根文件推理入口。

先统一几个会反复出现的词：

- **IPD**（Inter-channel Phase Difference，通道间相位差）：同一个声音到达两个麦克风有时间差，在频谱上表现为相位差，是定位的主要线索；本任务把它展开成逐个非参考麦克风的 `cos/sin` 实值特征。
- **DOA**（Direction of Arrival，到达方向）：声源相对阵列的方向角。
- **VAD**（Voice Activity Detection，语音活动检测）：逐帧判断有没有人说话，用来区分「有源」与「静音」。
- **DP-IPD target**：由直达路径几何生成的每个 pair 的复数相位差，形状为 `[T,F,K,P,2]`；`P` 是 pair 数，不得沿 pair 轴求平均。
- **target**：监督信号，由仿真里声源到阵列的直达路径几何算出；DOA/VAD 与 DP-IPD 的时间轴必须经过显式对齐，不是音频波形。
- **seam**：框架里给多个任务共用的中间实现。SSL 直接复用共享的静态/移动空间仿真，不重写声学渲染。
- **fail-fast**：发现配置不合法就立刻报错退出，不带着错误继续往下跑。

边界：

- 唯一合法模式是 `data.mode: dynamic_ssl`。`data.mode` 是配置里声明数据来源的键，`train.py` 据此路由到 `SSLTrainingEntry`。
- 覆盖范围是 `dynamic_ssl` 的场景、几何、DP-IPD/DOA/VAD 合同，以及原生 baseline 和四个官方结构的训练 smoke。`FNSSL`、`IPDNetFixed`、`IPDNetVariable`、`IPDNet2` 已完成结构接入；有限官方 `simulation_profile`、本地数据整理/下载脚本和真实录音评估入口也已接入。前三者已有 CPU 合同/one-epoch smoke 证据，IPDNet2 依赖 `mamba_ssm` 且 CUDA-only，GPU smoke 是条件路径：具备 CUDA 与 `mamba_ssm` 时运行，无 CUDA 时由 marker 跳过。LOCATA dev/eval 与 RealMAN val/test 资产已整理到 `data/ssl/`，但当前没有严格 checkpoint，因此不宣称真实数据或官方指标复现。
- 没有公开文件推理入口，也没有 ONNX 发布路径；把 SSL 配置交给根 `inference.py` 会在模型构造前直接报错。当前对外能力是训练与合同校验。
- 模型特征由显式 `ModelSpec.feature_contract` 决定：`native_ipd_baseline` 使用 `reference_ipd`，官方模型使用 `multichannel_stft_real_imag`（多通道 STFT 实部/虚部）。`ModelSpec` 还冻结 target、pair layout、geometry/field mode、时间压缩、causal、mapping 和 `target_layout`；公共输出统一为 `trailing_ri`，官方内部允许 `freq_concat_ri` 但必须由 adapter 还原，不能根据模型名称或 tensor rank 猜测。`IPDNetFixed`/`IPDNetVariable` 的 `causal=true` 是 online 变体（narrow-band LSTM 单向），`causal=false` 是 offline 变体（narrow-band LSTM 双向）；两者的 `CausCnnBlock` 仍始终因果。

## 📁 目录结构

```
tasks/ssl/
├── data/
│   ├── dynamic_dataset.py    # SSLDynamicMixingDataset：动态仿真的取数入口
│   ├── collate.py / batch.py # ssl_collate_fn 与 SSLBatch 合同校验
│   ├── projection.py         # 把渲染好的空间 scene 投影成 IPD/DOA/VAD 与 target
│   └── spatial_adapter.py    # 对接共享空间场景与 observation post-mix 的适配层
├── models/
│   ├── ipdnet.py              # native baseline 与 FNSSL 官方结构
│   ├── ipdnet_official.py     # IPDNetFixed / IPDNetVariable 官方结构
│   └── ipdnet2.py             # IPDNet2 官方 OnlineSpatialNet + Mamba 结构
├── losses.py                 # SSLSpatialMSELoss / SSLDPIPDPITLoss / SSLDOACompositeLoss
├── model_io.py               # ModelSpec、模型 I/O 与 runtime 字段合同
├── runtime.py                # 训练/校验期具名 runtime 调用封装
└── trainer/                  # SSLTrainingEntry + SSLTrainer
```

## 📦 数据与仿真

- 批合同：Dataset 每个样本产出 canonical dict item，`ssl_collate_fn` 将其拼成 `SSLBatch` 并执行 `validate()`。item 必须显式声明 `input_representation`、`target_representation`、`pair_index`、`pair_mask`、`geometry`、`time_stride` 和时间基准；不能通过 tensor rank 猜测模型。`input_lengths`/`output_lengths` 是精确整数，旧的 `length` item 字段直接拒绝；`sample_rate` 是不小于 1000 的精确整数。补齐帧在所有帧域字段中必须为零。维度含义是 `B` 批大小、`T_in/T_out` 输入/输出帧数、`F` 频点数、`K` 声源数、`C` 麦克风数、`P` pair 数：
  - 多通道 STFT 实部/虚部输入 `[B,T_in,F,2C]`，仅由官方模型的 `feature_contract: multichannel_stft_real_imag` 消费；native baseline 使用 `reference_ipd` 输入 `[B,T_in,F,2(C-1)]`；
  - `target_dpipd` `[B,T_out,F,K,P,2]`：每个 pair 的 real/imag DP-IPD，`P=C-1` 仅适用于 `reference_to_all`，all-pairs 时 `P=C(C-1)/2`；
  - `pair_index` `[B,P,2]`、`pair_mask` `[B,P]`、`geometry` `[B,C,3]`；`pair_index` 是事实来源，不得由 batch 通道数临时重建；
  - `doa` `[B,T_out,K,2]`、`vad` `[B,T_out,K]∈[0,1]`、`input_lengths`/`output_lengths` `[B]`、`frame_origin` `[B]`；
  - near-field target 额外携带 `source_distances` `[B,T_out,K]`，它属于 scene metadata/target 或显式 range mapping，不是 IPDnet2 的默认观测输入；距离以 `geometry[reference_channel]` 的参考麦克风为原点，near-field mapper 也以同一参考麦克风构造候选位置；
  - `provenance`：逐样本的来源摘要，含 backend、sha256、`phase_sign`、`normalization{mode,aggregation,forgetting_norm_sample_length}`、`sound_speed_mps` 和 `geometry.reference_channel`，用于事后核对数据出处、DOA mapping 相位约定、归一化时间常数、声速及 near-field 参考原点；通用 forgetting 默认/其他 profile 为 298，IPDNet2 官方模板为 249，`normalization: global_mean` 的 aggregation 为 `global_c_f_t_mean`，`normalization: none` 时 sample length 记录为 `null`。
- `time_stride` 的 projection 只聚合完整 stride 窗口；尾部不足一个窗口的输入帧丢弃，输出长度为 `T_in // time_stride`，与 FNSSL/IPDnet 的模型池化保持一致；`T_in < time_stride` 直接 fail-fast。
- 麦克风几何只来自 `simulation_cfg`（`array_profile` / `reference_channel` 等），不来自 Dataset builder 键。
- 复用共享的静态/移动 Layer A/C/D/E/F seam 后投影 IPD/DOA/VAD；静态还是移动由 `prob_moving_rir` 控制（仓库两份配置均取 `1.0`，即全程移动场景）。在线 RIR 与 geometry provenance 见 [`docs/guides/simulation/Online_gpuRIR.md`](../../docs/guides/simulation/Online_gpuRIR.md)。
- `simulation_cfg.prob_point_source_background` 会在 `ObservationPostMixRenderer` 调用 `render_speech_scene` 前解析为显式 `background_spec`。概率命中时，噪声 actor 在 speech scene 的同一 room、array 和 moving timebase 内生成点源 RIR；未命中时继续使用 `diffuse_coherence`。该噪声只加入 observation，`target`/`direct` 监督轨迹保持不变，SNR 仍由共享 `scale_noise_image` 标定。
- `simulation_cfg.prob_wind_noise_sidechain` 在 SSL/DOA spatial path 中也复用该 `background_spec`：wind asset 先生成严格 `[C,T]` diffuse/point-source control image，再进入 `wind_noise()` 的非加性 compressor；post-mix facts 通过 `NonDecomposableObservationContract` 标记 `decomposable=false`。该实验分支必须提供 split-level `windnoise_scp`，SE/SS/AEC/TSE spatial path 当前仍拒绝该键。
- scene geometry 通过 `render_speech_scene(observation_geometry=...)` 显式传递，SSL 不再临时修改 `simulation_cfg.array_profile/custom_geometry`；最终样本 provenance 同时保留 `noise_image_provenance`、背景 SNR、background audit、policy 和 RNG stream 映射。
- 多 source 的相对活动窗口由共享 `simulation_cfg.scene.timeline`（或其 canonical 平面键）控制，和 TSE/SS/SSL 使用同一 `SharedSimulationView.temporal` 与 timeline provenance；SE/AEC 配置该能力会在 contract 边界拒绝。
- 已登记 recipe 的样本 provenance 还记录 `configured_tracks`、`noise_source`、`official_source_count` 和 `official_data_source_match`；FN-SSL/IPDnet 官方单源配方在当前 2-source PIT 框架扩展或使用 HQ noise 时会显式标记 `simulation_not_official=true`。
- 本机 SSL 配置使用 `data/ssl/speech/{train,validation}.scp` 与 `data/ssl/noise/{train,validation}.scp`；清单由 `tools/prepare_data_tools/generators/ssl/prepare_manifests.py` 从现有 LibriSpeech/HQ noise 资产生成，音频本体不复制。`tools/prepare_data_tools/generators/ssl/download_data.py` 另提供 HTTPS 校验、MAT→16 kHz WAV 转换和 `data/ssl/noise92/noise92.scp` 生成；当前已下载 Noise92 的 15 个资源，但该清单仍不能单独代表完整官方训练集。

数据准备命令：

```bash
PYTHONPATH=. python -m tools.prepare_data_tools.generators.ssl.prepare_manifests \
  --test-split test-clean --output-dir data/ssl --overwrite
PYTHONPATH=. python -m tools.prepare_data_tools.generators.ssl.download_data \
  --download-noise92 babble white leopard machinegun --convert-wav \
  --output-dir data/ssl/noise92
```

## 🚀 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml
```

本仓库提供两份 SSL 训练配置：`configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml`（上方命令所示，共享 diffuse 背景 + `point_source_prob` 点源背景）与 `configs/ssl/spatial/fixed_array_ipdnet_dynamic_wind.yaml`（真实 urgent26 speech/noise/wind 资源，叠加可分解空间加性 wind 背景 `wind_background_prob=0.25`，同时保留 `point_source_prob=0.3`，两条独立伯努利门可共现）；另有 `fnssl_dynamic.yaml`、`ipdnet_fixed_dynamic.yaml`、`ipdnet_variable_dynamic.yaml`、`ipdnet2_dynamic.yaml` 四份官方模型的合同模板。checkpoint 与日志写到配置里的 `trainer.exp_path`（moving 配置默认 `exp/ssl/fixed_array_ipdnet_dynamic_moving/`，checkpoint 在其 `checkpoints/` 子目录下）。moving 配置引用的 `data/ssl` 清单由上面的脚本生成；wind 配置仍按其 urgent26 资源合同执行。

入口约束（不满足即 fail-fast）：

- `model_spec` 必须是顶层 mapping，并显式声明 `feature_contract`、`target_contract`、`pair_layout`、`geometry_mode`、`field_mode`、`reduction`、`causal`、`mapping` 和 `target_layout`；`target_layout` 只允许 `trailing_ri`/`freq_concat_ri`，实际模型公共边界固定为 `trailing_ri`；
- `reference_ipd` 只允许 `native_ipd_baseline`/`FixedArrayIPDNet`，官方模型必须声明 `multichannel_stft_real_imag`；
- `loss.type` 默认按模型选择 `SSLDPIPDPITLoss`（官方 DP-IPD）或 `SSLSpatialMSELoss`（native baseline）；两类模型都可显式选择 `SSLDOACompositeLoss`，loss 必须与 `target_contract` 对齐；
- `ssl.num_tracks` 可统一声明 train/validation Dataset 与模型的声源数；已有 `data.{train,validation}.{tracks,max_track}` 或 `model.args.max_track` 同时出现时必须相等。省略该字段时沿用现有声明；loss 的 `max_tracks` 是 PIT 容量上限，必须覆盖实际 K；
- `ssl.vad_mode` 当前只支持 `direct_path`（默认）：由每源 direct-path STFT 能量生成 VAD，`provenance.vad_supervision` 记录模式、来源与相对能量阈值；
- `strict_load` 必须为 `true`，不支持 placeholder supervision（用占位数据凑数训练）；
- `data.{train,validation}.seed` 必须等于 `meta.seed`；
- 拒绝 waveform `trainer.val_metrics`（需显式 `enabled: false`），因为本任务的验证目标不是波形域指标；
- 时长键使用 `segment_seconds`，不是 `max_duration`；
- `source_distances` 仅在 near-field target 或显式 DOA×range mapping 中允许出现；不能把 ground-truth distance 传给 IPDnet2 作为默认推理输入。
- `normalization: global_mean` 是 offline IPDnet 的 observation 归一化：对整段 STFT `[C,F,T]` 的幅度取一个全局均值（`global_c_f_t_mean`），只作用于 observation，direct target 不归一化；`normalization: forgetting_laplace` 必须使用正整数 `forgetting_norm_sample_length`，通用默认/其他 profile 为 298，IPDNet2 模板在 `ssl` section 显式设为 249，模型内同名参数设为 `null`，避免数据侧和模型侧重复归一化。实际 mode、aggregation 和 sample length 统一写入 `provenance.normalization`。
- `projection`、`SSLBatch` 和训练配置对外只接受 canonical `pair_layout: reference_to_all`；`IPDNetVariable` 在模型内部使用 pair-batched pair sharing，但对外仍还原为 reference-to-all 的 pair 轴。`all_pairs`/`pair_batched` 作为配置值仍在 `SSLTrainingEntry` 边界 fail-fast。

### 组合监督与离线分块推理

以下字段已由配置入口实际消费，可用于现有训练配置：

```yaml
ssl:
  vad_mode: direct_path  # 从每路 direct-path 谱生成活动标签
  num_tracks: 2          # 数据与模型使用同一个声源数
loss:
  type: SSLDOACompositeLoss
  args: null            # 清空父配置的旧 loss 参数，默认只启用 DP-IPD PIT
```

`args: null` 会清空继承的旧损失参数，使用组合损失默认值；自定义辅助权重时，
从已经清空旧参数的基配置继续扩展。四份官方模型模板同样清空了父配置中仅适用于
native loss 的 `allow_no_stft`，PIT 默认容量为 8，实际声源数仍由 `ssl.num_tracks`
及模型/Dataset 的一致性校验决定。

`SSLDOACompositeLoss` 的可选权重为 `doa_classification_weight`、
`doa_regression_weight`、`vad_weight`、`range_weight`，默认均为 0；
DOA 分类与回归互斥，分类须提供 `doa_class_bins [Q,2]`（polar/azimuth 弧度），
range 仅支持 near-field target。所有项复用同一次 DP-IPD PIT 配对。
当前注册模型仍只产出 DP-IPD；启用辅助项时，模型必须在 `IO_CONTRACT`
声明并实际输出对应 head，缺失时配置/模型边界直接报错。两角回归的
`doa_vector [B,T,K,2]` 格式保持不变。具体辅助输出合同如下：

| 输出 | 形状与语义 | 损失 |
|---|---|---|
| `doa_logits` | `[B,T,K,Q]`，类别对应 `doa_class_bins` | 交叉熵 |
| `doa_vector` | `[B,T,K,2]`，polar/azimuth 弧度 | 转为球面单位向量后的 MSE |
| `vad_logits` | `[B,T,K]`，活动概率的 logits | 支持 soft VAD target 的 BCE |
| `range` | `[B,T,K]`，以参考麦克风为原点的距离，单位米 | log-range 域的 Smooth L1 |

DOA/range 项仅监督有效帧内 `vad > 0.5` 的活动源；VAD 项覆盖有效帧内全部槽位。
无有效 pair 的样本和 padding 帧不参与辅助损失。所有有效输出帧的 range 槽位必须
严格为正，padding 可为零。权重必须有限、非负且至少一项为正；`dpipd_weight`
默认 1。DP-IPD PIT 在 `K <= 2` 时枚举排列，`K >= 3` 时使用 Hungarian。
实现与完整校验见 [losses.py](losses.py)。

非因果 `IPDNetFixed` 可在 `model.eval()` 后调用
`model(features, offline_inference=True)`；`n_seg` 必须为 12 的正整数倍。
输入按整段 observation 完成归一化，再右侧补零、分块、拼接并裁到原始
`T//12` 帧。真实录音评估 CLI `tools/verify/ssl/real_eval.py` 的
`--offline-inference` 开关会选择该路径，要求 `causal=false` 和
`ssl.normalization=global_mean`；默认整段推理。`model.args.offline_chunk_batch_size`
是单次前向最多处理的分块总数（跨输入 batch），必须为正整数，默认 1；可按显存
容量提高该值。分块顺序、整段归一化与 `T//12` 裁剪语义保持不变。
评估结果同时记录开关、`n_seg` 和实际启用的 `offline_chunk_batch_size`
（未启用分块时为 `null`）。直接调用模型时应使用 `torch.inference_mode()` 或
`torch.no_grad()`；这时中间激活由小批上限约束，整段输入与输出仍占用随时长增长的
内存。评估 CLI 已使用 `torch.inference_mode()`。
训练态、其他模型或 online 变体启用该开关均报错。

## 🧠 模型

| 模型 | 说明 | 参考来源 |
|------|------|----------|
| **native_ipd_baseline / FixedArrayIPDNet** | 项目原生 smoke 模型，消费 `reference_ipd` `[B,T,F,2(C-1)]`；用于验证 Dataset、collate、PIT 和 trainer，不代表官方 FN-SSL | [项目 Registry](models/__init__.py) |
| **FNSSL** | 已接入官方 full/narrow LSTM、skip connection 和 `average_pool_12`；消费 `multichannel_stft_real_imag` 并输出 pairwise DP-IPD。`fnssl_official` recipe 已接线，模板显式使用 4.79 s 片段，CPU one-epoch smoke 已覆盖；真实 DOA 指标仍需 LOCATA/严格 checkpoint | [ipdnet.py](models/ipdnet.py) |
| **IPDNetFixed / IPDNetVariable** | 已接入官方 FN block + causal convolution（该卷积在 online/offline 两变体中都保持因果）；`causal=true` 是 online、narrow-band LSTM 单向，`causal=false` 是 offline、narrow-band LSTM 双向，offline IPDnet 的 feature normalization 语义为 `global_mean`。`ipdnet_official` recipe、CPU 前向/one-epoch smoke 与真实录音评估入口已接入；未见阵列泛化与真实指标仍需对应资产 | [ipdnet_official.py](models/ipdnet_official.py) |
| **IPDNet2** | 已接入官方 OnlineSpatialNet + Mamba 结构，使用 near-field target/mapping 合同；官方模板数据侧 `forgetting_norm_sample_length=249`，模型内同名参数为 `null` 以避免双重归一化；依赖 `mamba_ssm` 且 CUDA-only，GPU 前向只在具备 CUDA 与该依赖的条件环境运行，无 CUDA 时由 marker 跳过。官方 RealMAN/指标复现仍未完成，不能退回 GRU | [ipdnet2.py](models/ipdnet2.py) |

### ModelSpec 示例

native baseline 的顶层配置必须显式写出：

```yaml
model_spec:
  model_type: native_ipd_baseline
  feature_contract: reference_ipd
  target_contract: dpipd_pair
  pair_layout: reference_to_all
  geometry_mode: fixed_array
  field_mode: far_field
  reduction: model_defined
  causal: false
  mapping: far_field_grid
  target_layout: trailing_ri
```

官方模型将 `feature_contract` 设为 `multichannel_stft_real_imag`，并使用其登记的 STFT、时间压缩和 causal 设置。`tasks.ssl.model_io.ModelSpec` 会拒绝未知字段、官方模型使用 `reference_ipd`、近场使用远场 mapping，以及 `model.type` 与 `model_spec.model_type` 不一致的配置。

### 具名 runtime 与模型输出

训练 runtime 保存 `features`、`target_dpipd`、`pair_index`、`pair_mask`、`geometry`、`source_distances`、`input_lengths`、`output_lengths`、`frame_origin`、`time_stride` 和 representation 元数据。模型只接收自己在 `ModelSpec` 中声明的观测字段，geometry、长度、VAD 和 pair mask 由 loss/mapping 显式消费。

DP-IPD 模型的 canonical 输出为：

```python
{
    "dpipd": dpipd,          # [B,T_out,F,K,P,2]
    "vad_logits": vad,      # 可选
    "doa_logits": doa,      # 分类合同可选
    "doa_vector": doa_vec,  # 连续 DOA 合同可选
    "range": source_range,  # near-field range head 可选
}
```

`dpipd` 的最后一维始终是 real/imag（`target_layout: trailing_ri`）。FNSSL 等官方实现可在模型内部使用频率拼接（`freq_concat_ri`），但 forward 返回前必须还原为该 canonical 布局。

第一阶段只启用 DP-IPD regression；模型和 loss 的主输出只接受 canonical `dpipd`（`[B,T_out,F,K,P,2]`），`prediction`/`ipd_prediction` 等旧输出别名在 model I/O、trainer 和 loss 边界直接 fail-fast。DOA 分类、连续角度回归和 range head 必须在模型、loss 和配置中分别声明，不能依赖旧字段猜测。

## 🔗 相关文档

- 仿真层级与 clean target 边界：[`docs/guides/simulation/Simulation_Flow_Logic.md`](../../docs/guides/simulation/Simulation_Flow_Logic.md)
- 配置入口：[`configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml`](../../configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml)、[`configs/ssl/spatial/fixed_array_ipdnet_dynamic_wind.yaml`](../../configs/ssl/spatial/fixed_array_ipdnet_dynamic_wind.yaml)
- 架构说明：[`docs/architecture/README.md`](../../docs/architecture/README.md)
