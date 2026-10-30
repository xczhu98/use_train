# SSL 使用指南

SSL（Spatial Sound Localization，空间声源定位）回答「声音从哪个方向来」，不改变音频本身：输入固定麦克风阵列录到的多通道音频，输出每个声源在每一帧、每个频点上的方向。先统一几个反复出现的词：

- **IPD**（Inter-channel Phase Difference，通道间相位差）：同一声音到达两个麦克风的时间差在频谱上表现为相位差，是定位的主要线索；native baseline 把它展开成逐个非参考麦克风的 `cos/sin` 实值特征。
- **DP-IPD**（Direct-Path IPD）：由直达路径和阵列几何生成的 pairwise 复数相位目标，保留每一个 microphone pair，不沿 pair 轴平均。
- **DOA**（Direction of Arrival，到达方向）：声源相对阵列的方向角。
- **VAD**（Voice Activity Detection，语音活动检测）：逐帧判断有没有人说话，区分「有源」与「静音」。
- **target**：监督信号，由仿真里声源到阵列的直达路径几何算出，给每个时频点标出 K 个声源的方向，不是音频波形。

SSL 暂不拆分更多 guides 子文档。任务边界、数据合同与模型细节以 [`../../../tasks/ssl/README.md`](../../../tasks/ssl/README.md) 为 SSOT，本文是导读与导航。

`multichannel_stft_real_imag` 或 `reference_ipd` 决定模型看到什么；前者是多通道 STFT 实部/虚部 `[B,T,F,2C]`，不是 raw waveform；`dpipd_far_field` 或 `dpipd_near_field` 决定我们要求模型预测什么。

## 边界（先读这个）

- 唯一合法模式是 `data.mode: dynamic_ssl`，`train.py` 据此路由到 `SSLTrainingEntry`。SSL 没有 offline 模式。
- 覆盖范围是 canonical DP-IPD 合同、静态与移动动态仿真、离线 RIR scene manifest 读取、native baseline，以及 FNSSL、IPDNetFixed、IPDNetVariable、IPDNet2 的官方结构接入。四个官方结构、有限 `simulation_profile` recipe、本地数据整理脚本和真实录音评估入口均已接入；FNSSL、IPDNetFixed、IPDNetVariable 已有 CPU 合同/one-epoch smoke 证据；IPDNet2 依赖 `mamba_ssm` 且 CUDA-only，GPU smoke 是条件路径：具备 CUDA 与该依赖时运行，无 CUDA 时由 marker 跳过。LOCATA dev/eval 与 RealMAN val/test 资产已整理到 `data/ssl/`，但当前没有严格 checkpoint，因此不宣称真实数据指标或官方指标复现。
- 没有公开文件推理入口，也没有 ONNX 发布路径：把 SSL 配置交给根 `inference.py` 会在模型构造前直接报错。当前对外能力是训练与合同校验。
- 帧域特征由 Dataset 产出，模型不重复执行框架 STFT；配置显式声明 `meta.allow_no_stft: true`。官方模型使用 `multichannel_stft_real_imag`，native baseline 使用 `reference_ipd`。

## 数据与仿真

- **批合同**：Dataset 每个样本产出 canonical dict item，`ssl_collate_fn` 分别补齐输入/输出帧轴并生成 `SSLBatch`。维度含义是 `B` 批大小、`T_in/T_out` 输入/输出帧数、`F` 频点数、`K` 声源数、`C` 麦克风数、`P` pair 数：多通道 STFT 实部/虚部 `features [B,T_in,F,2C]`；native IPD `features [B,T_in,F,2(C-1)]`；`target_dpipd [B,T_out,F,K,P,2]`；`pair_index [B,P,2]`、`pair_mask [B,P]`、`geometry [B,C,3]`、`doa [B,T_out,K,2]`、`vad [B,T_out,K]`、`input_lengths/output_lengths [B]`、`frame_origin [B]` 和 `time_stride`。`provenance` 含 backend、sha256、`phase_sign`、`normalization{mode,aggregation,forgetting_norm_sample_length}`、`sound_speed_mps` 和 `geometry.reference_channel`；通用 forgetting sample length 为 298，IPDNet2 合同模板显式使用 249，`normalization: global_mean` 的 aggregation 为 `global_c_f_t_mean`，`normalization: none` 时 sample length 记录为 `null`。near-field `source_distances` 以 `reference_channel` 麦克风为原点，mapper 使用同一参考麦克风作为候选原点；旧 `targets/lengths` 只读别名不能用于构造旧合同。
- `time_stride` 只聚合完整窗口，尾部 partial window 丢弃，输出长度为 `T_in // time_stride`；当输入帧不足一个 stride 时直接 fail-fast，保证 projection 与官方池化时间轴一致。
- 已登记 official recipe 的 provenance 会记录实际 `configured_tracks`、噪声清单来源和单源官方配方是否匹配；当前 2-source PIT 扩展或 HQ noise 清单不会被标记成官方单源指标复现。
- **几何来源**：麦克风几何只来自 `simulation_cfg`（`array_profile`/`reference_channel` 等），不来自 Dataset builder 键；near-field 的 reference origin 是解析后 geometry 中 `reference_channel` 的麦克风位置，不使用隐含坐标原点。声速从 `simulation_cfg.gpurir_sound_speed_mps` 解析后贯穿 projection/mapper，并写入 `provenance.sound_speed_mps`。
- **仿真 seam**：复用共享静态/移动 Layer A/C/D/E/F seam 后投影 DP-IPD/DOA/VAD；静态还是移动由 `prob_moving_rir` 控制，历史 moving 配方取 `1.0`。在线 RIR 与 geometry provenance 见 [`../simulation/Online_gpuRIR.md`](../simulation/Online_gpuRIR.md)；分层合同见 [`../simulation/Simulation_Flow_Logic.md`](../simulation/Simulation_Flow_Logic.md)。
- **资源状态**：本机已用 `tools/prepare_data_tools/generators/ssl/prepare_manifests.py` 生成 `data/ssl/speech/{train,validation,test}.scp` 与 `data/ssl/noise/{train,validation}.scp`；音频本体仍位于原始 `data/` 树。`download_data.py` 已下载并转换 Noise92 的 15 个资源，生成 `data/ssl/noise92/noise92.scp`；`download_real_datasets.py` 已将 LOCATA dev/eval 归档解压到 `data/ssl/LOCATA/{dev,eval}`，并将 RealMAN 的 val/test、定位 CSV、元数据和转写整理到 `data/ssl/RealMAN`。这些资源可审计，但仍需严格 checkpoint 才能形成真实指标。

数据整理与安全下载入口：

```bash
PYTHONPATH=. python -m tools.prepare_data_tools.generators.ssl.prepare_manifests \
  --test-split test-clean --output-dir data/ssl --overwrite
PYTHONPATH=. python -m tools.prepare_data_tools.generators.ssl.download_data \
  --download-noise92 babble white leopard machinegun --convert-wav \
  --output-dir data/ssl/noise92
PYTHONPATH=. python -m tools.prepare_data_tools.generators.ssl.download_real_datasets \
  --locata all --output-dir data/ssl
conda run --no-capture-output -n setrain python -m \
  tools.prepare_data_tools.generators.ssl.download_real_datasets \
  --realman eval --output-dir data/ssl
```

真实录音评估入口为 `tools/verify/ssl/real_eval.py`，支持带显式几何/标签的 canonical JSONL 和本地 LOCATA 目录；没有真实资产或严格 checkpoint 时会直接失败，不会用 synthetic smoke 冒充真实结果。

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -C configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml
```

`configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml` 和 `configs/ssl/spatial/fixed_array_ipdnet_dynamic_wind.yaml` 是历史 native/鲁棒性配方；`configs/ssl/fnssl_dynamic.yaml`、`ipdnet_fixed_dynamic.yaml`、`ipdnet_variable_dynamic.yaml`、`ipdnet2_dynamic.yaml` 是官方模型的合同模板。checkpoint 与日志写到配置的 `trainer.exp_path`（默认 `exp/ssl/fixed_array_ipdnet_dynamic_moving/checkpoints/`）。

入口约束（不满足即 fail-fast）：

- native baseline 默认使用 `SSLSpatialMSELoss`，官方 DP-IPD 模型默认使用 `SSLDPIPDPITLoss`；均可显式选择 `SSLDOACompositeLoss`，辅助权重启用时要求模型声明并产出对应 head；loss 必须与 `ModelSpec.target_contract` 对齐；
- `ssl.num_tracks` 统一 train/validation/model 的实际声源数，重复声明必须一致；loss `max_tracks` 为容量上限。`ssl.vad_mode` 当前仅支持 `direct_path`，模式、来源和阈值进入 provenance；
- `strict_load` 必须为 `true`，不支持 placeholder supervision；
- `data.{train,validation}.seed` 必须等于 `meta.seed`；
- 拒绝 waveform `trainer.val_metrics`（需显式 `enabled: false`），验证目标不是波形域指标；
- 时长键使用 `segment_seconds`，不是 `max_duration`。
- `normalization: global_mean` 是 offline IPDnet 的 observation 归一化，对整段 STFT `[C,F,T]` 幅度取一个全局均值（`global_c_f_t_mean`），direct target 保持原始物理幅度/相位；`normalization: forgetting_laplace` 必须显式使用正整数 `forgetting_norm_sample_length`，通用默认/其他 profile 为 298，IPDNet2 官方模板为 249，实际 mode、aggregation 和 sample length 写入每个 item 的 `provenance.normalization`，避免把不同时间常数混为同一 recipe；
- `projection`、`SSLBatch` 和训练配置对外只接受 canonical `pair_layout: reference_to_all`；`IPDNetVariable` 在模型内部使用 pair-batched pair sharing，但对外仍还原为 reference-to-all 的 pair 轴。`all_pairs`/`pair_batched` 作为配置值仍在 `SSLTrainingEntry` 边界 fail-fast。

非因果 `IPDNetFixed` 已提供 eval-only `offline_inference=True` 模型入口；
真实录音评估 CLI 通过 `--offline-inference` 显式启用，`n_seg` 须为 12 的
正整数倍，整段 `global_mean` 归一化发生在分块之前。
`model.args.offline_chunk_batch_size` 限制单次前向处理的分块总数，默认 1，
在无梯度推理时约束中间激活的显存；整段输入与输出仍随录音时长增长。
该配置与 `n_seg` 一同记录在评估结果中。其余模型与 online 变体
拒绝该开关。组合监督参数、输出合同和推理用法见
[任务 README](../../../tasks/ssl/README.md#组合监督与离线分块推理)。

## 模型

| 模型 | 说明 | 参考来源 |
|---|---|---|
| `native_ipd_baseline / FixedArrayIPDNet` | 消费 `reference_ipd`，输出 `[B,T_out,F,K,P,2]`；只用于合同、PIT 和 trainer smoke | [`tasks/ssl/models/ipdnet.py`](../../../tasks/ssl/models/ipdnet.py) |
| `FNSSL` | 已接入官方 full/narrow LSTM、skip connection 和 `average_pool_12`；消费多通道 STFT 实部/虚部并输出 pairwise DP-IPD。`fnssl_official` recipe 已接线，模板显式使用 4.79 s 片段；已有 CPU one-epoch smoke。真实 DOA 指标仍需 LOCATA/严格 checkpoint | [ipdnet.py](../../../tasks/ssl/models/ipdnet.py) |
| `IPDNetFixed / IPDNetVariable` | 已接入官方 FN block + causal convolution（两种变体都保留因果卷积）；`causal=true` 是 online、narrow-band LSTM 单向，`causal=false` 是 offline、narrow-band LSTM 双向，offline IPDnet 的 feature normalization 语义为 `global_mean`。`ipdnet_official` recipe、CPU 前向/one-epoch smoke 和真实录音评估入口均已接入；未见阵列泛化与真实指标仍需对应资产 | [ipdnet_official.py](../../../tasks/ssl/models/ipdnet_official.py) |
| `IPDNet2` | 已接入官方 OnlineSpatialNet + Mamba 结构，使用 near-field target/mapping 合同；官方模板数据侧 `forgetting_norm_sample_length=249`，模型内同名参数为 `null` 以避免双重归一化；依赖 `mamba_ssm` 且 CUDA-only，GPU 前向只在具备 CUDA 与该依赖的条件环境运行，无 CUDA 时由 marker 跳过。官方 RealMAN/指标复现仍未完成，source distance 不作为模型输入 | [ipdnet2.py](../../../tasks/ssl/models/ipdnet2.py) |

模型 I/O 合同由 `tasks/ssl/model_io.py::validate_ssl_model_io` 校验；训练/校验期调用封装在 `tasks/ssl/runtime.py`。DP-IPD 模型和 loss 的 canonical 主输出只有 `dpipd`（`[B,T_out,F,K,P,2]`）；旧 `prediction`/`ipd_prediction` 输出别名在 model I/O、trainer 和 loss 边界直接 fail-fast。

## 相关文档

- 任务 SSOT：[`../../../tasks/ssl/README.md`](../../../tasks/ssl/README.md)。
- 架构入口（SSL 暂无独立架构文档）：[`../../architecture/README.md`](../../architecture/README.md)。
- 配置入口：[`../../../configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml`](../../../configs/ssl/fixed_array_ipdnet_dynamic_moving.yaml)。
- 仿真层级与 clean target 边界：[`../simulation/Simulation_Flow_Logic.md`](../simulation/Simulation_Flow_Logic.md)。
