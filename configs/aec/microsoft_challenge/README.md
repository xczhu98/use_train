# 🎧 Microsoft AEC Challenge 配置 (`configs/aec/microsoft_challenge`)

AEC（回声消除）训练配置，数据管线为 `dynamic_aec`（在线仿真，实现见 `tasks/aec/data/dynamic_dataset.py`）。完整 A–F 语义以 [`docs/guides/simulation/Simulation_Flow_Logic.md`](../../../docs/guides/simulation/Simulation_Flow_Logic.md) 为准；本文件只记录 Microsoft Challenge 配置的 family、延迟、loss 和发布差异。

AEC 要解决的问题是：设备扬声器播出的远端语音被自己的麦克风收回去，需要在麦克风信号里把这段回声去掉、同时保留近端说话人的语音。本目录的十份生产配置覆盖四个模型家族，差异集中在模型结构、STFT 几何、回看延迟和 loss 配方上；先看「配置速查表」按采样率与 loss 挑一份，再用文末「训练入口」的命令启动。

| 术语 | 一句话解释 |
|---|---|
| mic / farend / nearend | 麦克风录音、远端参考信号（扬声器播出的信号）、近端说话人语音；训练目标是还原 nearend |
| echo / SER / ERL | 回声分量；SER 是近端语音与回声的电平比，ERL 是 far-end 与 echo 的电平差（`far_db = echo_db + ERL`）；两者都由 Layer C-aec 在仿真中抽样 |
| 回看延迟（lookback） | 回声永远晚于参考信号，模型需要缓存过去若干帧；`align_delay` 表示含当前帧的候选帧数，最大回看跨度为 `(align_delay - 1) × frame_ms` |
| 双讲（double-talk） | 近端与远端同时说话，也是评估 AEC 是否会损伤近端语音的关键条件 |
| ERLE | 回声抑制量指标，越大越好；AEC validation 的专属指标 |
| DBGNet / DeepVQE / CAGCRN / LocalVQE | 本目录的四个模型家族；DBGNet 另有论文逐项对齐的非生产配置 |
| MBRCL | 一种 loss 配方（SI-SNR + 多分辨率 STFT + 多带相对对比），见 [`configs/README.md`](../../README.md) 的 loss 缩写表 |
| `AECDeploymentFrontend` | 模型输入前端（可选 Streaming GCC-PHAT / LinearAEC），在 STFT 之前运行，属于 Trainer/Inference 而非 Dataset 仿真 |

当前生产矩阵覆盖四个 model family、十个 Dynamic YAML variant，其中 DBGNet 提供 16 kHz hybrid、32 kHz hybrid、32 kHz MBRCL、32 kHz MR-SI-L1-SS（small 与 base 两种容量）五份配置，DeepVQE 提供三份 32 kHz loss variant。此外 `dbgnet/dbgnet_paper_16k_16ms_delay1000ms_loss_recon.yaml` 是按 DBGNet 论文逐项对齐的非生产配置（16 kHz、512/256、ERB 65/64、F-GRU 192、论文 `L_recon`，`mask_domain: mag`，可切 `com`）。Dataset 执行 Layer A 语音预处理 → B 场景分支 → C 声学混合 / C-aec 清单回声 → Layer D post-mix amplitude conditioning（normalize → guard）→ E channel degradation clipping → F observation/device degradation（fixed-rail clipping）→ final deterministic safety；Dataset 仿真统一只有 Layer A–F。仅 AEC 专属的 `AECDeploymentFrontend` 由顶层 `aec_frontend` 配置；Trainer/Inference 均会解析并在 STFT 前调用它，但只在 `delay_alignment.enabled` 或 `linear_aec` 已配置时执行，否则 mic/farend 严格直通且不报 runtime error，Inference 仍逐文件调用。两者不得混称。配置差异主要在模型、STFT、回看延迟与 loss 配方。

**阅读顺序**：只想跑通训练的话，看完「配置速查表」直接到「数据清单」和「训练入口」；「关键约定」及其后的模型小节、离线仿真产物合同、噪声清单合同是配置语义细节（延迟策略、Layer D/E/F 期望值、生成器 schema、审计口径），改仿真时才需要读。

## 配置速查表

| 配置文件 | 模型（变体） | Params (M)（total/trainable） | 1 秒 mic+farend MACs (G) | STFT / 采样率 | 增强范围 / 回看跨度 | Loss / epochs / batch |
|----------|-------------|--------------------------|----------------------|---------------|---------------------|-----------------------|
| `cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml` | CAGCRN（CATA，d=6） | 0.075 / 0.050 | 0.107 | 512/256，16 kHz | 0–80 ms 纯 Beta(2,4) + 极偶发纯 sudden_jump 压力桶 / 80 ms | mse_sisnr_l1 / 1000 / 32 |
| `dbgnet/dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml` | DBGNet small（state head 关闭） | 0.393 / 0.383 | 0.604 | 320/160，16 kHz | 0–80 ms 纯 Beta(2,4) + 极偶发纯 sudden_jump 压力桶 / FTG 时序跟踪，无显式 lag search | `CompositeLoss`：`GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`（无 StateCE）/ 60 / 16 |
| `dbgnet/dbgnet_small_32k_10ms_delay80ms_loss_hybrid.yaml` | DBGNet small（state head 关闭；ERB 81+80） | 0.576 / 0.538 | 1.348 | 640/320，32 kHz | 0–80 ms 纯 Beta(2,4) + 极偶发纯 sudden_jump 压力桶 / FTG 时序跟踪，无显式 lag search | `CompositeLoss`：`GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`（无 StateCE）/ 60 / 16 |
| `dbgnet/dbgnet_small_32k_10ms_delay80ms_loss_mbrcl.yaml` | DBGNet small（state head 关闭；ERB 81+80） | 0.576 / 0.538 | 1.348 | 640/320，32 kHz | 0–80 ms 纯 Beta(2,4) + 极偶发纯 sudden_jump 压力桶 / FTG 时序跟踪，无显式 lag search | `SISNRLoss` + `MultiResSTFTLoss` + `MultiBandRelativeContrastiveLoss` / 60 / 16 |
| `dbgnet/dbgnet_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml` | DBGNet small（state head 关闭；ERB 81+80） | 0.576 / 0.538 | 1.348 | 640/320，32 kHz | 0–80 ms 纯 Beta(2,4) + 极偶发纯 sudden_jump 压力桶 / FTG 时序跟踪，无显式 lag search | `SIL1WavSeTrainLoss` + `MRSIRelCspecL2Loss` + `MRSpecSuppressPenaltyLoss`（SS undesired-energy 开）/ 60 / 16 |
| `dbgnet/dbgnet_base_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml` | DBGNet base（论文 Table I 通道 32/64/96/128 与 16/32/48/64；F-GRU 192；state head 关闭） | 1.556 / 1.518 | 4.840 | 640/320，32 kHz | 0–80 ms 纯 Beta(2,4) + 极偶发纯 sudden_jump 压力桶 / FTG 时序跟踪，无显式 lag search | `SIL1WavSeTrainLoss` + `MRSIRelCspecL2Loss` + `MRSpecSuppressPenaltyLoss`（SS undesired-energy 开）/ 60 / 16 |
| `dbgnet/dbgnet_paper_16k_16ms_delay1000ms_loss_recon.yaml`（论文对齐，非生产变体） | DBGNet（论文 Table I 几何，DBG-Mag，F-GRU 192） | 1.469 / 1.444 | 2.362 | 512/256/512，16 kHz | 论文配方：delay p=0.5、≤1 s、B(1,3)；SER U(−15,15)；噪声 p=0.6、SNR 0–40；近端混响 p=0.4；扬声器失真 p=0.5 / FTG 时序跟踪 | 论文 `L_recon = 0.7·L_mag + 0.3·L_com`（γ=0.3）/ 75 / 16 |
| `deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml` | DeepVQE small + 会议场景 | 0.960 / 0.960 | 1.511 | 640/320，32 kHz | 0–80 ms 纯 Beta(2,4) + 极偶发纯 sudden_jump 压力桶 / 90 ms | `GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss` / 60 / 16 |
| `deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_MBRCL.yaml` | DeepVQE small + 会议场景（MBRCL） | 0.960 / 0.960 | 1.511 | 640/320，32 kHz | 0–80 ms 纯 Beta(2,4) + 极偶发纯 sudden_jump 压力桶 / 90 ms | `SISNRLoss` + `MultiResSTFTLoss` + `MultiBandRelativeContrastiveLoss` / 60 / 16 |
| `deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml` | DeepVQE small + 会议场景（MR-SI-L1-SS） | 0.960 / 0.960 | 1.511 | 640/320，32 kHz | 0–80 ms 纯 Beta(2,4) + 极偶发纯 sudden_jump 压力桶 / 90 ms | `SIL1WavSeTrainLoss` + `MRSIRelCspecL2Loss` + `MRSpecSuppressPenaltyLoss` / 60 / 16 |
| `localvqe/localvqe_16k_20ms_delay80ms_loss_hybrid_fwsdr.yaml` | LocalVQE（S4D） | 0.780 / 0.780 | 1.860 | 512/256，16 kHz | 0–80 ms 纯 Beta(2,4) + 极偶发纯 sudden_jump 压力桶 / 1008 ms | `GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss` + FW-SDR / 60 / 16 |

表中 Params 以 M（百万）、MACs 以 G（十亿）为单位；数值来自当前模型谱域前向的 THOP 统计：`B=1`、输入为各 1 秒 mic+farend，图外 STFT/iSTFT 不计。THOP 对 `stride>1` 的 `ConvTranspose2d` 采用“输出网格”口径，比精确计数（PyTorch `FlopCounterMode`：输入网格 × 输出通道 × kernel）高 22–29%（论文对齐配置 `2.362 G` vs `1.901 G`）；与论文 Table II 的 `2.0 G/s` 比较时应使用精确口径。精确口径对照（GMACs）：论文对齐 `1.901`、16 kHz Small `0.478`、32 kHz Small `1.097`（hybrid 与 MBRCL 相同）。THOP 结果统一视为 `partial/unverified`，可能漏计 functional、complex、custom/unsupported operator，不得解释为完整部署计算量。未执行 profile 的 Params/MACs 均写为 `not measured`。独立 FLOPs 为 `not measured`，因为当前环境没有独立且已校验的 FLOPs 计数器；不使用 `2 × MACs` 代替。AECTrainer 在线验证不计算 MOS，DNSMOS/NISQA/UTMOS/SCOREQ 等通过 `evaluate/` 对 checkpoint 推理产物离线评估。

> **统一命名格式**：`<model>[_<variant>]_<sample_rate>k_<frame>ms_delay<delay>ms_loss_<loss>[_<scenario>].yaml`。采样率固定置于模型/变体和帧长之间；`delay` 通常表示 `aec_ref_echo_delay_range_ms` 的上限，`loss` 使用小写配方缩写（如 `mbrcl`），场景后缀（如 `meeting`）置于末尾。当前十份 Dynamic 生产 YAML 统一使用 `delay80ms` 文件名和 `[0,80] ms` 纯 `Beta(2,4)`；LocalVQE 实际 STFT 帧移为 16 ms，`align_delay=64` 时最大回看跨度为 1008 ms。论文对齐配置使用 `delay1000ms`，对应论文最长 1 s 的 delay augmentation，不属于该生产命名集合。

## 关键约定

- **四个 family 的对齐机制不同**：CAGCRN 的 `CATA` 在含当前帧的 `align_delay` 个候选上做 delay softmax（现役 recipe `d=6`，16 ms 帧移下最大回看 80 ms；`align_delay` 可配置）；DeepVQE 与 LocalVQE 的 `AlignBlock` 分别在含当前帧的 `align_delay=10` / `64` 个候选上估计 soft delay；DBGNet 使用 causal `FTG` GRU 做 far-end 时序跟踪，并依赖统一的 0–80 ms 纯 `Beta(2,4)` delay 增强学习主体时延鲁棒性，没有显式 lag search。现役 `aec_time_delay_jitter` 只是极偶发 route/buffer sudden-jump 压力桶，不是主体 delay，也不建模 drift 或 periodic jitter。外部 Streaming GCC-PHAT / LinearAEC 仅属于 Trainer/Inference 侧的 `AECDeploymentFrontend` 模型输入前端，在 STFT 前执行，不是 Dataset 仿真，也不是上述模型内机制。
- **echo-only 设备 EQ 的 reference tap 边界**：`aec_echo_device_eq` 只给 echo 做播放着色，仅在数字 farend/reference tap 位于该播放着色之前时成立；若设备在着色之后采集 reference，则必须同步着色 reference，不能沿用 echo-only 假设。十份 Dynamic 生产配置的 echo-path device EQ 概率为 `aec_echo_device_eq.prob: 0.25`，它与 Layer F 对完整 mic observation 执行的 post-mix EQ（`prob_post_eq: 0.15`）是两个独立事件，不能互相迁移或合并概率。`bass_boost` 保留当前名称、算法与数值：它保留低频、衰减 cutoff 以上成分，形成相对 bass emphasis，可近似可选用户/厂商 bass、loudness 或 speaker-compensation EQ；它不是小扬声器固有 bass roll-off。
- **STFT 与采样率上下文由构造器注入**：模型从顶层 `stft` 获取其声明并接受的参数，模型内部不重复做 STFT/iSTFT。`build_loss` 同时将顶层 `stft` 的已声明参数和 `data.sample_rate` 注入声明相应构造参数的 loss；普通 loss 的显式 `loss.*.args` 优先。MBRCL 是严格匹配例外：存在顶层 STFT 或非空 sample-rate 上下文时，显式参数必须与其完全一致，否则构建期 fail-fast。以 LocalVQE 的 `FWNegSISDRLoss` 为例，实际获得 `n_fft=512`、`hop_length=256`、`sr=16000`，但不接收未声明的 `win_length` / `window`。
- **单位口径**：采样率为 Hz，waveform 长度为 samples，延迟/帧移为 ms，绝对数字电平为 dBFS（RMS 或 peak 必须注明），SNR/SER/ERL 与 SDR 边界为 dB；概率和 loss 权重均为无量纲。
- **`align_delay` 表示含当前帧在内的候选数**，最大回看跨度为 `(align_delay - 1) × frame_ms`。工程上建议让该跨度覆盖预期的最大回声延迟；当前构造流程不执行这一关系的强制校验。
- **Layer B 七项 `prob_*` 之和须 = 1.0**（`prob_target_only + prob_noise_only + prob_echo_only + prob_noise_echo_only + prob_target_noise + prob_target_echo + prob_target_noise_echo`）；旧场景命名均不再是有效键。
- **ERL 耦合及例外**：far-end 与 echo 电平通常通过 `far_db_range` / `erl_range` 物理耦合（`far_db = echo_db + ERL`）。`pre_ser_overload` 跳过 SER 抽取与 configured ERL/far_db 耦合，echo 电平由 overload boost 决定，farend 保持加载电平；nominal SER/ERL draw 为 N/A，但 farend/echo stem 可用时仍可计算 realized ERL。
- **C-aec path-local clipping 合同**：十份 Dynamic 生产配置仅在含 echo 样本内以条件概率 `aec_echo_clipping.prob: 0.15` 进入 selector，再按 `post_ser_distortion / pre_ser_overload / synced_ref_clip = 0.80 / 0.15 / 0.05` 选择一条互斥路径。`post_ser_distortion` 使用 `tanh / hard = 0.35 / 0.65` 与 echo 峰值相对阈值 `[0.90, 0.99]`；`pre_ser_overload` 使用 `fixed_rail / tanh = 0.80 / 0.20`、boost `[0, 6] dB`、rail `[-3, -0.5] dBFS`，该范围面向会议麦克风基础训练的轻到中度播放过载，9 dB 仅适合专项压力测试；`synced_ref_clip` 使用 `fixed_rail / tanh = 0.95 / 0.05`，echo 与 farend 共享一次抽取的绝对 rail `[-3, -0.5] dBFS`。按当前 Layer B 中含 echo 样本约占 70% 计算，C-aec selector 的全样本边际概率约为 10.5%；三条路径的全样本边际概率分别约为 8.4%、1.575%、0.525%。这是路径选择率，不是真实 rail-hit、饱和或 waveform 改变率，后者还取决于样本幅度与所选 variant。
- **论文对齐配置与 `mask_domain`**：`dbgnet/dbgnet_paper_16k_16ms_delay1000ms_loss_recon.yaml` 按论文 Table I / Eq.10-12 / Sec. III-C 对齐（512/256/512、ERB 65+64=129、mic/far `[32,64,96,128]`/`[16,32,48,64]`、F-GRU 192（默认规则 `mic[-1]+far[-1]`）、T-GRU 128、AdamW `3e-4`、batch 16、75 epochs cosine、`L_recon = 0.7·L_mag + 0.3·L_com`）。F-GRU 宽度按论文 Table I 规则默认为 `mic[-1]+far[-1]`（论文通道 192、双向输出 384；生产 small 通道 96），不再需要显式写 `fgru_hidden`；非官方参考实现 vkeep/DBGNet 用的是 `(mic[-1]+far[-1])//2`（论文通道下 96），与 Table I 不一致。该规则变更使按旧规则训练的三份生产 DBGNet checkpoint 与已发布 ONNX 形状不匹配，必须重训与重导。`model.args.mask_domain` 选择掩码域：`mag`（默认，DBG-Mag：实值 sigmoid 掩码乘复谱、保留麦克风相位）或 `com`（DBG-Com：2 通道 RI 输入 + tanh 复数比值掩码；论文未给该变体的掩码参数化与激活函数，属项目显式选择）。两种掩码域的 cache 名字/顺序/ABI 相同，仅首层 `mic_in`/`far_in` 通道数不同（1 / 2），切换掩码域必须 cold reset。该配置的语料、echo 来源（真实录制 pair 而非 DNS RIR 合成）、近端混响建模与 L_SD 差异在文件头逐条列出，不得据此宣称复现论文 Table II/III 数值。
- **CAGCRN 的 SER / 加性噪声 SNR 是论文 recipe，不随其余九份统一值**：`cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml` 显式使用 `ser_range: [-10, 10]` 与 `snr_low: 0` / `snr_high: 40`（论文的 SER `-10–10 dB`、SNR `0–40 dB` 均匀采样，见 `docs/paper/aec/BUPT/33_CAGCRN.md`），其余九份为 `ser_range: [-20, 20]` 与 `snr_low: -10` / `snr_high: 25`。`tests/test_configs.py` 的 effective validation contract 对 CAGCRN 的 `ser_range` 有显式白名单；因此 CAGCRN 的 C-1 加性噪声 SNR 与 C-aec SER 不能按“十份统一”口径描述。
- **统一 Layer A**：SE/AEC/TSE 生产 Dynamic Dataset 在 splice 前按 `silence_pad_prob` 预判。模式 A 执行标准 splice/crop/tile/zero-pad；模式 B 只加载单 utterance，A-2 后构造固定 `max_duration` 静音画布，短语音只放一次、长语音只连续 crop 一次，最终 invariant 失效即 fail-fast、禁止 tile。fixed canvas 仍执行 bimodal range 的配置抽样与 RNG 消耗，但该抽样不控制最终静音总量。Offline Dataset 不受影响。
- **RIR shared full-peak + post-RIR target volume**：当前直接承载 `simulation_cfg` 的 AEC 动态训练配置统一显式设置 `rir_normalize: true`；static full/early 共用 full peak，moving 全 sequence/early 共用 full-sequence peak。启用 `source_volume_normalization_enabled` 时，在 RIR 卷积后以 early target 标定 dBFS，并把同一 gain 施加到 full、early 与最终 target；noise / echo 随后混入。runtime 也接受 `rir_normalize: false`，此时保留资产 raw gain；两种模式都保留 full/early 相对增益。房间与摆位造成的传播、反射和频响效应主要由 RIR（或真实录制 pair 中已固化的声学路径）承载，不应归因于 `bass_boost`。
- **Layer A-3 审计**：`a3_alignment_mode` / `a3_mode` / `a3_tile_applied` / `a3_fixed_canvas_applied` 记录实际路径；`a3_repeat_to_max_duration_enabled` 仅记录配置开关，`a3_silence_duration_policy=fixed_canvas_remainder` 表示静音量取画布余量。`repeat_to_max_duration_applied` 仅在真实 tile 时为 `true`，fixed canvas 必须为 `false`；未记录的 `a3_crop_start_samples` 保持 `null`。
- **审计口径分层**：`candidate` 是候选 draw，`nominal` 是实际选中/请求的目标，`calibration` 是 Layer C active-power 标定及 aggregate 重测，`realized` / `realized_final` 分别是 A–F post-mix 完成后、最终裁剪/重采样后的实际波形指标；不得混称。colored noise 与普通 additive noise 都按各自 active-region power 标定；Layer C aggregate 会在合并后的 physical noise stem 上重新 active detect。
- **指标适用性**：pure `prob_echo_only` / `prob_noise_echo_only` 可保留 latent SER draw 用于 echo 缩放，但场景 SER 为 N/A；不可分解样本的 final SNR/SER 与 reconstruction residual 也为 N/A。ERL 只依赖 farend/echo stem，不随 mic 可分解性失效。C-aec 操作本身不必然破坏分解：只要对应 echo stem 与实际混入 mic 的 echo 同步更新，样本仍可保持 decomposable；应以实际 task facts / reason code 判定。
- **DBGNet 状态监督命名**：可选的逐帧辅助监督正式命名为 **frame-level state CE engineering mode**；它使用 `StateCrossEntropyLoss`，不等同于论文带 temporal pooling 的 segment-level `L_SD`。当前三份 DBGNet 生产配置均关闭 state head 且不含 StateCE；其中两份 `hybrid` 配置使用仅含 `GTCRNCompressedSpectralLoss` 与 `GTCRNLogRatioLoss` 的两分量 `CompositeLoss`，`mbrcl` 配置则使用 SI-SNR + MR-STFT + MBRCL。历史 `hybrid` YAML 文件名保持不变，仓库不存在 `*_loss_hybrid_statece.yaml` 生产配置，命令示例不得引用该文件名。
- **DBGNet checkpoint 严格拓扑门禁**：当前递归拓扑通过 state-dict key `_recursive_topology_v1` 版本化。checkpoint 必须与当前 topology 完整匹配并 strict 加载；缺少该 key 的旧拓扑 checkpoint 会 fail-closed，不提供 `non-strict` 或 `partial` 兼容路径，也不得通过忽略 missing/unexpected keys 绕过。旧拓扑权重必须按当前递归拓扑重新训练后再用于推理、导出或发布。
- **collate 精确长度合同**：`aec_collate_fn` 接受 7/11/12 元组（7=offline 无 flags，11=标准，12=含帧级 `state` label）；5/6/8/9/10/13 及其他所有长度均显式拒绝。`dbgnet` 带帧级 `state` label 时使用 12 元组；`StateCrossEntropyLoss` 消费 `(B,T,C)` logits 与 `(B,T)` 类索引，`-1` padding 被忽略。
- **Layer D / final safety**：Layer D 定义为 post-mix amplitude conditioning，内部固定为 D-normalize（peak normalization）→ D-guard（deterministic peak guard）。`layer_d_facts` 与 `simulation_by_layer['D']` 是唯一 canonical 聚合：`{'stage_order': ('D-normalize', 'D-guard'), 'normalize': {...}, 'guard': {'mic_related': five_keys, 'farend': five_keys}}`；每组 `five_keys` 均为 `roles` / `threshold` / `peak` / `scale` / `applied`。AEC 的 mic/nearend/echo/noise/farend 由同一次 coupled guard 计算并使用共同 scale；`mic_related` / `farend` 仅是角色视图，不是两个独立 guard。不存在顶层 normalize 或 farend-guard 发布旁路。E/F 后的 final deterministic safety 同样无随机性，但只保护 observation，不修改 stems 或 farend。
- **关闭项必须显式登记参数范围**：十份生产配置与论文对齐配置的 Layer E/F 关闭项除 `prob_*: 0.0` 外仍写出对应参数（`lowpass_*`、`band_reject_*`、`drc_*`、`codecs_types`、`bit_crush_range`、`spectral_holes_*`、`webrtc_*`、`bass_boost_*`、`spectral_leakage_*`、`loudness_max_n_intervals`、`dc_offset_range`），逐值与 runtime 默认值一致；禁止只写概率而依赖默认值——默认值变化会让训练分布静默漂移，且配置不再自证其行为。
- **Layer E/F 职责与期望值**：Layer E 保留 channel clipping；十份 Dynamic 生产配置的固定输入链路均关闭 Layer E。Layer F 是 observation/device clipping 的权威位置：十份配置均以全样本 selector `prob_clip: 0.15` 选择 `fixed_rail`（`clip_variant_probs.fixed_rail: 1.0`），先从 `[0.0, 9.0] dB` 抽取 pre-drive，再在固定 ±1.0 rail 截幅；同时保留全样本 `prob_post_eq: 0.15`，因此 Layer F clipping 与 postEQ 的期望操作数为 `0.30` 项/样本。按 C-aec selector 全样本约 10.5%、Layer F clipping selector 全样本 15% 且两者独立计算，其 selector 并集约为 `23.925%`；该并集仍不是实际截幅率或 rail-hit 率。仅当 Layer E clipping 实际 applied 时跳过 Layer F clipping，Layer F 的其他操作仍执行；C-aec echo clipping 不触发该跳过。由于当前十份配置关闭 Layer E，不能用 C-aec echo clipping 降低 Layer F clipping selector 的实际触发率。

### LocalVQE 16 kHz 配置

`localvqe/localvqe_16k_20ms_delay80ms_loss_hybrid_fwsdr.yaml` 的主体静态 echo delay 使用 0–80 ms 纯 `Beta(2,4)`，并显式设置 `aec_ref_echo_random_delay_uniform_prob: 0.0`。它另启用极偶发纯 `sudden_jump` route/buffer 压力桶：utterance gate `0.03`、事件率 `[0.02,0.10] events/s`、幅度上限及范围 `20 ms / [-20,20] ms`、最短稳定段 `1000 ms`，不混入 drift 或 periodic jitter。canonical `aec_frontend.delay_alignment` 关闭且未配置 `linear_aec`，所以 Trainer/Inference 的 `AECDeploymentFrontend` 处于关闭状态；Dataset 只执行 A–F，不改变 waveform。模型内部 `AlignBlock` 不属于该模型输入前端。

### DBGNet Small 32 kHz 配置

`dbgnet/dbgnet_small_32k_10ms_delay80ms_loss_hybrid.yaml` 保持 16 kHz DBGNet 的 Small 通道、仅含 `GTCRNCompressedSpectralLoss` 与 `GTCRNLogRatioLoss` 的两分量 `CompositeLoss`（无 StateCE）、关闭的 state head、AEC 模型输入前端关闭，以及主体 0–80 ms 纯 `Beta(2,4)` 和含 echo 样本内 `p=0.03` 的极偶发纯 `sudden_jump ±20 ms` route/buffer 压力桶合同；历史 `hybrid` YAML 文件名保持不变。它将 `stft` 固定为 `640/320/640`、Hann、`center: true`，并将 `data.sample_rate`、`model.args.sample_rate` 固定为 `32000`；`erb_low_bins: 81`、`erb_bands: 80` 是本项目质量优先的 32 kHz 工程 baseline，不是论文配方。

- train/val RIR 分别使用 `data/shared/rirs/gpurir_static/gpurir/fs32000/{train,val}_rirs.scp`；ref/echo 使用现有的 `echo_prealign_32k` **32k-target prealigned manifest**。目录名或 manifest 不能作为“已验证原生全频数据”的证据；若切换到未对齐原始语料，仍须将 `aec_ref_echo_fine_gcc_align` 设为 `true`。
- 三份 Small 配置的复杂度已按统一口径测得（关闭 state head、论文规则 F-GRU）：16 kHz 为 `0.393 M` 参数 / `0.604 G` MACs；32 kHz 两份为 `0.576 M` / `1.348 G`（THOP `partial/unverified`，图外 STFT/iSTFT 不计）。FLOPs 与 RTF 仍为 `not measured`，须在实际 checkpoint 与目标设备上 profile；不得沿用原 state-head 配置的旧参数量。配置本身不要求修改 DBGNet、训练、shared ONNX/runtime/export 或生产模型代码。
- **MBRCL 配置的两套 STFT 几何不得混称**：模型顶层及 MBRCL 分析使用单分辨率 `(n_fft,win_length,hop_length)=(640,640,320)` Hann，即 `20 ms FFT / 20 ms window / 10 ms hop`；这与论文 BLSTM/CNN-U-Net 的 `32 ms FFT / 32 ms Hamming window / 16 ms hop` 不一致。`MultiResSTFTLoss` 独立使用 `(1024,640,320)/(2048,1280,640)/(4096,2560,1280)`，对应 `20/40/80 ms` windows 和 `10/20/40 ms` hops；它是论文 MR-STFT 三组 sample tuples 全部乘 2 后的 32 kHz 等物理时长 schedule。每个 resolution 对 enhanced/target 各计算一次 STFT，三个尺度的 `SC + log-magnitude` 作算术平均。这里仅 MR-STFT schedule 保持物理时长，不能据此宣称顶层分析几何或整套 objective 对齐论文。
- **ONNX 部署前置契约**：部署链路必须显式以 32 kHz 接收 mic/farend，复现 `center=true` 的 Hann STFT/iSTFT（含 reflect padding、帧边界与 overlap-add）并将 10 ms lookahead 纳入端到端延迟；同时固定 `640/320/640`、ERB `81/80` 和关闭的 state head。实际导出前必须以本配置构造模型并完成对应 ONNX/运行时等价性验证，禁止由其他采样率的默认值接管。

### 会议版 DeepVQE 32 kHz 配置

`deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml` 面向 VoIP 会议场景，不能与普通 32 kHz DeepVQE 配置作严格效果对比。当前关键语义如下：

- 源语音音量使用 `source_volume_db_range: [-48, -12]` 与 `Beta(2.0,1.5)`，均值约 `-27.4 dBFS`、众数约 `-24 dBFS`；`prob_loudness_perturb: 0.0`，避免只改变 noisy 而不改变 target。
- Layer A 模式 B 是严格的单 utterance 10 秒 fixed silence canvas：短语音只放一次，长语音只连续 crop 一次；clean/noisy/farend/echo 保持同步，最终对齐不得整体重复。`repeat_to_max_duration: true` 与 tile gap 仅控制未命中 silence pad 的模式 A。
- `rir_normalize: true` 使用 shared full-peak：static full/early 共用 full peak，moving 全 sequence/early 共用 full-sequence peak；随后 `source_volume` 以 early target 标定 dBFS，并让 full/early/target 共用 gain，noise/echo 在后混入。
- 风噪保留 URGENT waveform sidechain：`prob_wind_noise_sidechain` 仍有效；普通噪声走 `noise_scp`、分类噪声池或 colored noise，不再使用 `prob_wind_noise_additive`。sidechain compressor 会非线性改变 near-end；样本因此不可分解时，最终 SNR/SER 与 reconstruction residual 为 N/A，ERL 仍按独立 farend/echo stem 审计。
- 随机 echo 延迟显式采用会议硬件工程策略（非论文分布）：

  ```yaml
  aec_ref_echo_random_delay_distribution: beta
  aec_ref_echo_random_delay_beta_a: 2.0
  aec_ref_echo_random_delay_beta_b: 4.0
  aec_ref_echo_random_delay_uniform_prob: 0.0
  aec_ref_echo_delay_range_ms: [0.0, 80.0]
  aec_time_delay_jitter:
    enabled: true
    prob: 0.03
    jitter_type: sudden_jump
    jitter_rate: [0.02, 0.10]
    jitter_corr_length: [100.0, 500.0]
    jitter_max_ms: 20.0
    jump_ms_range: [-20.0, 20.0]
    min_segment_ms: 1000.0
    transition_ms: 0.0
    edge_mode: nearest
  ```

  该策略仅延迟 echo 路径，并以极偶发纯 sudden_jump 作为播放/采集 buffer 或 route 硬跳变压力桶；它不是主体 delay，且不混入漂移或周期 jitter。
- `align_delay: 10` 在 10 ms 帧移下包含当前帧及前 9 帧，最大回看跨度约为 90 ms，覆盖人工 base delay 与 jitter 轨迹的 80 ms 上限。当前 train/val manifest 已严格排除全部 4119 个 `copy_low_confidence` / `copy_residual_gate_failure` 失败项，与预对齐 records 的失败项交集为 0；后续重建 manifest 时必须保持该过滤合同，避免未对齐固有残差与人工 delay/jitter 叠加。
- `stft.center: true` 保留 10 ms lookahead；部署前端必须实现与 `torch.stft(center=True)` 相同的 reflect padding、帧边界和 iSTFT overlap-add，并把 lookahead 纳入端到端延迟预算。ONNX 流式脚本应使用 `--config` 读取本配置，避免采用其他采样率的默认值。
- 每个 epoch 末执行 loss validation；`max_val_batches: 0` 表示不限制 loss batch 覆盖。`save_top_k: 0` 关闭 step-level top-k；DeepVQE meeting YAML 在 `val_metrics.metrics` 末尾显式列出 `ERLE`，每次 validation 都运行 SI-SDR、ESTOI、ERLE（`val_metrics.interval: 1`）。侵入式指标与 ERLE 分别由 `max_utterances_metrics`、`max_utterances_erle` 控制独立全局 quota；十份生产 YAML 的 validation 目标、原始 SCP 条数与 `max_utterances` 全部对齐为 2200，并已显式写出 `max_utterances_metrics` 与 `max_utterances_erle`。经过指标适用性、有效性或失败过滤后，最终 valid metrics 数仍可能更少。两项专用 quota 各自独立计数，省略时各自回退到 `max_utterances`。`metrics_workers: 2` 同时限制每个 rank 的侵入式指标池与 ERLE 池；ERLE 池还受 CPU 数和本 rank 整轮 quota 约束。
- `num_workers`、`val_num_workers`、`persistent_workers` 与音频 cache 都按 rank/process 生效：当前十份生产配置每 rank 的 train DataLoader 使用 4 个常驻 worker，validation DataLoader 独立使用 1 个非持久 worker；`data.dataloader.timeout: 120` 同时作用于 train 与 validation DataLoader，用于隔离动态仿真卡死。cache 不跨进程共享；1–4 卡或内存不足时，优先降低 worker/prefetch 数，并调小 `USE_TRAIN_AUDIO_CACHE_MAXSIZE`、`USE_TRAIN_AUDIO_CACHE_BYTES` 或 `USE_TRAIN_AEC_STEREO_CACHE_BYTES`，再按实测 RSS 与吞吐逐步调回。

#### 动态噪声清单与角色分流

`noise_scp` 只接受严格四列 SCP（`uid fs length path`）或逐行 JSON object JSONL，metadata 仅由 JSONL 携带；完整的发布/读取合同（strict JSONL v6、admission policy v4、AudioTag evidence v5、noise source parser v10、generator `6.1`、promotion marker v4、Stage D 三条分支）见 [Simulation_Flow_Logic.md §0.1.1](../../../docs/guides/simulation/Simulation_Flow_Logic.md) 与 [configs/README.md 噪声发布准入与训练消费边界](../../README.md#噪声发布准入与训练消费边界)。本文件只保留 AEC 侧需要知道的部分：

```text
noise_scp
  ├─ roles/category = wind  ──> AEC/SE/TSE side-chain wind pool
  │                              └─ 从 ordinary noise pool 移出
  ├─ roles/category = music ──> 仅 TSE music pool
  │                              └─ SE/AEC 中仍是 ordinary noise
  └─ 其他                        ──> ordinary noise pool
                                  └─ sampling.noise 的 category/temperature 均衡
```

- `roles` 仅在值为 list 时权威；字段缺失或非 list 时才按 `balance_category` 推导，缺少该字段时可按 `category_name` 推导。内嵌 Wind role 与显式 `windnoise_scp` 合并扩池；AEC 不拆 Music pool。
- 生成器的 split 模式发布 train/val JSONL 与 `noise_publication_metadata.json`；这些 metadata 用于 provenance 与审计，不是训练输入。训练实际 category/temperature 分布用 `tools/verify/audit_noise_sampling.py` 审计，README 不固定数量快照。

这些配置是训练先验，不是已校准的设备自然延迟直方图。正式结论应按 realized delay、ERLE、SI-SDR、双讲、语音起始和噪声角色分桶评估。

### 离线仿真产物合同

`tools/prepare_data_tools/generators/aec/generate_simulated_testset.py` 当前 generator version 为 **2.18**：`2.14` 将 Pink/Brown colored noise 改为基于物理 Hz 频率轴的功率律 PSD；`2.15` 将仿真分层收敛为 Layer A–F，并把 post-F realized 从 `simulation_by_layer.realized` 移到顶层 `realized` 审计节点；后续版本继续沿用当前 A–F 与审计合同。上述升级会改变生成内容或发布 schema，旧 manifest 不可复用；manifest 必须与当前版本严格匹配，版本不匹配时须按 CLI 提示使用 `--force` 重新生成，不得手改 manifest。

样本 `meta.json` 中，`snr_db_primary_candidate` 是普通主噪声候选，`snr_db_wind_candidate` 是 wind sidechain 候选（即使最终未采用 wind 也可存在），`snr_db_selected_nominal` 是实际选中的 primary nominal 混合目标，`snr_db_realized` 是 `realized_final.snr_db` 的最终波形实测副本；A–F post-mix 完成后、最终裁剪前的实测值位于顶层 `realized.snr_db`。`simulation_by_layer.realized` 属于 2.14 及更早的旧 schema，2.15 会拒绝该旧槽。

七场景生成器不覆盖 Layer-B `prob_*`：`main()` 按原配置解析 kwargs，只构造并复用一个 Dataset，再通过生产 canonical 原子 seam `dataset.get_sample_with_trace(index, force_branch=scenario, capture_stems=True)` 强制场景，并从同一个 sealed `SimulationTrace` 取得 sample、stems 与 branch facts，避免可变 sidecar 串样。普通 `__getitem__` 不参与生成，因为它会按训练概率抽样；`_getitem_simulation_verify` 仅保留给最小 fake Dataset 测试兼容，不是生产生成入口。`simulation_by_layer.B.force_branch` 记录显式请求的场景，`simulation_by_layer.B.branch` / 顶层 `branch` 记录实际采用的分支；三者应一致，不能把非 `null` 的 `force_branch` 解释为概率覆盖。

这些导出字段和强制入口不改变 waveform，Layer-B 概率仍保留原训练配置。会议版仍使用 URGENT wind sidechain 原路径；wind waveform、compressor 及 `(compressed_nearend + wind) / 2` 混合语义均未改。

## 数据清单

`dynamic_aec` 训练需以下清单。加性噪声推荐使用 `tools/prepare_data_tools/generators/noise/build_stage_d_noise_manifests.py` 生成的分类 JSONL，可省略独立的 `windnoise_scp`（风噪类别自动从 JSONL 中提取）。

| 清单 | 位置（示例） | 说明 |
|------|-------------|------|
| `speech_scp` | `data/shared/clean/filter/train.scp` | 干净近端语音（四列 SCP `uid fs length path`） |
| `noise_scp` | `data/shared/noise/filter/v2/train_noise_sources.scp` | 加性噪声（纯四列 SCP：`uid fs length path`，无 category metadata）；分类 JSONL（`train_noise_sources.jsonl` / `val_noise_sources.jsonl`）仍在同目录，供 category 均衡采样实验使用 |
| `rir_scp` | 16 kHz 配置：`data/shared/rirs/gpurir_static/gpurir/fs16000/train_rirs.scp`；32 kHz DeepVQE / DBGNet：`data/shared/rirs/gpurir_static/gpurir/fs32000/train_rirs.scp` | 房间脉冲响应必须与配置 `data.sample_rate` 匹配；不要在 16 kHz 与 32 kHz 配置间混用 |
| `windnoise_scp` | *可选* | 显式风噪清单；若省略且 `noise_scp` 的 JSONL metadata 可分类，则自动使用其中的 `Wind` / `Wind noise (microphone)` 类别 |
| `real_farend_echo_pair_json` | 32 kHz DeepVQE / DBGNet：`data/aec/echo/echo_prealign_32k/real_farend_echo_pair_train.json`；16 kHz CAGCRN/DBGNet/LocalVQE：`data/aec/echo/echo_prealign_16k/real_farend_echo_pair_train.json`（val 均为同名 `_val.json`） | 真实 far-end/echo 对必须是 `{"pairs": ["stereo.wav"], "meta": {...}}`，每项固定为立体声路径（ch0=echo, ch1=farend）；双 wav 行和旧 schema 均拒绝。`echo_prealign_32k` 对 32 kHz 配置仅表述为现有 **32k-target prealigned manifest**，不能表述为已验证原生全频数据。预对齐清单不应再次启用在线 GCC：它会重新估计 delay，并可能在 farend 前端补零或按 crop 位置偏移取样，实际改变已物化 pair 的时序；它不是只读质量门。 |

验证集同名 `val_*` 清单；RIR 与 real far-end/echo 均按 16 kHz 或 32 kHz 配置区分（echo 分别用 `echo_prealign_16k` / `echo_prealign_32k` 预对齐清单）。`echo_prealign_32k` 仅是现有 32k-target prealigned manifest，不是已验证原生全频数据的声明。采样率与配置一致，无需读盘重采样。支持本机绝对路径或仓库根下的 `data/...` 相对路径；只提交路径字面量，不提交数据本体。若清单改回未对齐原始语料，必须同步将 `aec_ref_echo_fine_gcc_align` 改为 `true`。

### ICASSP 2023 官方盲测资产

原始目录为 `data/aec/AEC_Challenge/datasets/blind_test_set_icassp2023/`，其中 `doubletalk`、`farend-singletalk`、`nearend-singletalk` 分别包含 **300 / 300 / 200** 组可配对的 `*_mic.wav` 与 `*_lpb.wav`。以下命令通过精确 alias 仅评估 ICASSP 2023 的全部场景，并由流水线生成各场景独立的 `mic.scp` / `lpb.scp`：

```bash
python evaluate/eval_pipeline_aec.py \
  --exp_dir exp/aec/deepvqe_small_32k_10ms_delay80ms_loss_hybrid \
  --datasets aec_icassp2023 \
  --aec_subsets all \
  --aec_testset_root data/aec/AEC_Challenge/datasets \
  --step all
```

`--datasets aec_official` 也受支持，但会选择根目录下所有已支持的官方年份；只评 ICASSP 2023 时应使用精确 alias `aec_icassp2023`。

本机生成的 `data/aec/AEC_Challenge/datasets/blind_test_set_icassp2023.scp` 把同目录下的 mic、lpb、enrollment 等文件混在一个四列清单中，只用于资产盘点和定位；它不是 AEC 成对输入合同，**不能替代**场景级 `mic.scp` 与 `lpb.scp`，也不应直接传给 paired AEC inference。

AECMOS 必须按官方场景口径报告：far-end single-talk（ST-FE）**仅报告 EMOS**，double-talk（DT）报告 **EMOS + DMOS**，near-end single-talk（ST-NE）**仅报告 DMOS**；项目结果字段 `AECMOS_ECHO` 对应 EMOS，`AECMOS_DEG` 对应 DMOS。不要跨场景混合平均，也不要用 DT 的无 oracle 能量比冒充 ERLE。

## 离线 residual capability 与模板边界

`offline_aec` 可从 `error_scp` 读取预生成 LinearAEC 时域 residual，并通过 7 元组中的 `error` / `error_mask` 交给 Trainer；存在有效 residual 时，Trainer 将其 STFT 结果作为 `error_spectra` 候选输入。显式声明 `aec_input_capabilities` 的模型按该 capability 集合裁剪输入，未声明 capability 的模型保留全输入语义。当前四个生产 family（CAGCRN、DBGNet、DeepVQE、LocalVQE）的显式能力均不包含 `error_spectra`，不会消费预生成 residual；UltraDPCAEC 可选消费 `error_spectra`。此外，runtime LinearAEC 与 `error_scp` 同时配置会 fail-fast。

`configs/aec/offline_aec_template.yaml` 是 DeepVQE 的占位示例，`mic_scp`、`farend_scp`、`nearend_scp` 默认为 `null`，必须先填写真实清单才能运行；仅取消注释 `error_scp` 不会把 DeepVQE 变成三输入模型。

## 训练入口

```bash
# AEC 统一入口（配置按模型分目录：cagcrn/、dbgnet/、deepvqe/、localvqe/）
python train.py -C configs/aec/microsoft_challenge/<模型目录>/<配置文件名>.yaml
```

实验输出写入 `trainer.exp_path`（约定 `exp/aec/<配置文件名>`，与配置所在子目录无关，见 `configs/README.md`）。

## 相关文档

- 统一仿真合同：`docs/guides/simulation/Simulation_Flow_Logic.md`
- 数据仿真设计：`docs/guides/aec/AEC_Data_Simulation.md`
- 架构分析：`docs/architecture/AEC_Architecture.md`
- 任务级说明：`tasks/aec/README.md`（含 ONNX 导出）
