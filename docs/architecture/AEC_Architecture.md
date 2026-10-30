# AEC 架构与模型说明

**姊妹文档**：[SE_Architecture.md](SE_Architecture.md)。

---

## SSOT：AEC 相关代码路径（防漂移）

| 主题 | 代码路径 |
|------|----------|
| CAGCRN | `tasks/aec/models/discriminative/cagcrn.py` |
| DeepVQE | `tasks/aec/models/discriminative/deepvqe.py` |
| AEC 损失 | `core/losses`（`CompositeLoss` + 注册子项；`AECTrainer` 内 `_composite_ss_undesired_kwargs` 组装 kwargs） |
| AEC 训练入口 | `train.py -C <config>`（按 `data.mode: dynamic_aec` 路由）；AEC 专用逻辑在 `tasks/aec/trainer/aec_trainer.py` |
| 动态 AEC 数据 | `tasks/aec/data/dynamic_dataset.py`、`tasks/aec/data/dynamic_aec_build.py`；仿真复用 `core.data.simulation` |
| 离线 AEC 数据 | `tasks/aec/data/offline_dataset.py` 保有 mic/farend/nearend/error 四列 SCP、7 元组与轨对齐 Adapter；`core/data/offline_multitrack.py` 仅共享 filtered index、metadata cache、局部 RNG、加载生命周期与音频 primitives，不统一 manifest/collate/batch |
| mic↔farend 评测配对 | `evaluate/aec_pairing.py`（`inference.py`） |
| validation 指标边界 | 共享 objective metrics 消费 `(nearend, estimate)`；`AECTrainer` 专用 ERLE 消费 `(mic, nearend, estimate, farend[, echo])` |

---

## 一、AEC 模型单模型分析

### 1. CAGCRN

**模型结构分解（公式化）**

- **CAGCRN = ERB(65,64) + CATA + TFGRU×3 + Decoder + Mask + TFAG**；输入输出 **(B, F, T, 2)**，F=n_fft//2+1（如 257）。
- **ERB**：与 GTCRN 同族，低频 65、高频 64 ERB 带，固定滤波器。
- **CATA**：Cross-Attention Temporal Alignment；`d=11` 仅为模型构造默认值，实际候选窗口由 YAML `align_delay` 注入。当前生产 CAGCRN 使用 `align_delay=6`，即当前帧加前 5 帧；16 ms hop 下窗口为 80 ms。cross-attention（softmax over 延迟候选）实现软对齐；双输入 mic/ref 对齐后融合。
- **TFGRU**：Time-Frequency GRU；频率方向 `intra_gru` 为 grouped bidirectional GRU，时间方向 `inter_gru` 为 grouped unidirectional GRU，二者后接各自的 Linear 与残差归一化。
- **Decoder**：DGConvBlock 等，与官方 CAGCRN.py 一致。
- **TFAG**：Time-Frequency Adaptive Gating（论文全称），时频门控。

**创新点结构张量维度变换**

| 创新模块/步骤 | 输入维度 | 输出维度 |
|---------------|----------|----------|
| ERB(65,64) | mic/ref (B, F, T, 2) | (B, 129, T, 2) 各支路 |
| CATA（Cross-Attention） | mic feat, ref feat | `d=11` 仅构造默认；实际窗口由 YAML `align_delay` 注入，当前生产为 `d=6`/80 ms，cross-attention soft alignment 后融合 |
| TFGRU×3 | 融合特征 | intra_fc(24→24) + 频率双向、时间单向的分组 GRU |
| Decoder + Mask | TFGRU 输出 | 复数比掩码 |
| TFAG | Mask 输出 | 时频门控 → (B, F, T, 2) |

**论文主要创新点**

1. **Cross-Attention (CATA)**：显式对 mic/ref 做时延对齐与融合，提升双讲与延迟鲁棒性。
2. **TFGRU**：时频分组 GRU，兼顾时域与频域依赖。
3. **ERB + 复数比掩码**：与 SE 侧 ERB 设计一致，适合 AEC 频谱建模。

**论文原文要点**（*CAGCRN*，**结构创新**）：① **CATA（Cross-Attention Temporal Alignment）**：用 cross-attention 计算 mic 与 ref 的相似度并做时延对齐，替代传统固定 delay 或 LAEC；② **TFAG（Time-Frequency Adaptive Gating）**：时频门控动态调节 AEC 与 NS 特征贡献，多尺度时频特征选择与抑制；③ **TFGRU**：频率双向、时间单向的分组 GRU，兼顾时域与频域依赖；④ **ERB(65,64) + 复数比掩码** 与 SE 侧一致；单模型联合 AEC+NS，约 0.07M 参数。

**Loss**

- 训练配置为 **`CompositeLoss`**（与 SE 相同 YAML，子项见 `core/losses`），经 `AECTrainer` 映射为近端波形/谱监督。
- 配置示例：`configs/aec/microsoft_challenge/cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml`。

**参数量/算力**

- 本实现与 AECNS-CAGCRN-main 结构对齐；若有 audit 或官方数据可填具体 M/G。

**可复用模块**

| 模块 | 可复用场景 |
|------|------------|
| ERB | 与 SE 共用，AEC 双通道频谱压缩。 |
| CATA | 任意双流（mic/ref）时延对齐与融合。 |
| TFGRU | 时频分组 RNN，可作其它 AEC/SE 骨干。 |
| TFAG | 时频门控，可作后处理或与 CATA 配合。 |

---

### 2. DeepVQE

**模型结构分解（公式化）**

- **DeepVQE = 双分支 Encoder(mic 5 块 / far 2 块) + AlignBlock + Bottleneck(GRU) + Decoder( Skip + ResBlock + SubPixel) + CCM**；这些主要模块由论文结构派生，当前实现不是作者代码的逐项严格复现。
- **Feature extraction**：模型内 `_FE` 只执行功率律压缩复谱（c=0.3）和 layout 变换；STFT/iSTFT 与 window 由 Framework / Trainer 的顶层 `stft` 配置管理。论文使用 square-root Hann，当前 32 kHz meeting 配置使用普通 `hann`，两者不可混称。
- **Encoder**：kernel 4×3、stride 1×2、因果 padding；base 5 块 mic / 2 块 far，small 4 块 mic / 2 块 far。
- **ResBlock**：Y = X + ELU(BN(Conv(X)))，kernel 4×3。
- **AlignBlock**：Q/K 点卷、K 沿时间 unfold、conv 5×3、softmax 加权求和，实现时延对齐。
- **Bottleneck**：flatten → GRU → linear → reshape。
- **Decoder**：skip 1×1、ResBlock、SubPixelConv 4×3、BN、ELU；CCM 为复数域 (m+1)×(2n+1) 因果 conv（Eq.2–3）。
- **论文 base 7.5M / small 0.59M；当前实际 32 kHz meeting 配置为 small，配置头记录 960,455 参数、约 1.511G MACs/s（1 秒谱域前向，不含 STFT/iSTFT）**。

**训练输出 clamp 审计说明（防止 code review 误报）**

- 日志中的 `model output sanitized` 不统计 DeepVQE 复谱，也不是 batch/样本数；它统计 `nearend_estimate_spectra` 经共享 iSTFT 后，进入 loss 前满足 `|waveform| > 1` 的**波形采样点数**。实现链路为 `tasks/aec/trainer/aec_trainer.py::_decode_aec_model_output()` → `core/trainer/metrics_trainer.py::sanitize_model_output()`，计数按整个 epoch 累加并跨所有 DDP rank 求和。
- DeepVQE 的末端是无幅度约束的 27 通道 CCM（3×3 复数时频邻域），当前 CCM 系数头采用普通随机初始化而非近恒等初始化；同时 train-mode `BatchNorm2d` 将中间特征保持在 O(1) 尺度。因此随机初始化阶段可能形成数倍但仍为 finite 的等效复滤波增益，使少量 iSTFT 采样点越过 `[-1, 1]`。这不同于 NaN/Inf、AMP overflow、STFT/iSTFT 不可逆或梯度爆炸。
- 机制上，随机 CCM 输出尺度与 train-mode BN 可能在初始化阶段产生有限但越界的 iSTFT 采样点；这与 NaN/Inf、AMP overflow 或 STFT/iSTFT 不可逆不是同一类故障。本文不固化未纳入版本化实验报告的单次复现数值或历史 run 计数，避免把机器、seed、代码版本相关观察误写成架构合同。
- **Review 判据**：先同时查看 `non-finite`、clamp 占有效采样点比例、触发 waveform 数、pre-clamp peak、validation 与 loss/指标趋势。`non-finite = 0` 且 clamp 仅为短暂启动瞬态时，不应仅凭 epoch 累计 raw count 判为 bug，也不应据此移除统一 sanitize；若 `non-finite > 0`、clamp 比例持续较高或上升、validation 同步大量触发，或 loss/指标恶化，才按 AMP、学习率、数据幅度或架构数值故障继续排查。

**创新点结构张量维度变换**

| 创新模块/步骤 | 输入维度 | 输出维度 |
|---------------|----------|----------|
| 双分支 Encoder | mic (B,F,T,2), far (B,F,T,2) | mic 5 块 / far 2 块，kernel 4×3 stride 1×2 |
| AlignBlock | mic feat, far feat | Q/K 点卷 + K 沿时间 unfold → soft-align |
| Bottleneck | Align 输出 | flatten → GRU → linear → reshape |
| Decoder | Bottleneck 输出 | skip + ResBlock + SubPixel 上采样 |
| CCM（Complex Convolving Mask） | Decoder 输出 | (m+1)×(2n+1) 因果卷积 → (B, F, T, 2) |

**论文主要创新点**

1. **联合 AEC+NS+去混响**：单模型多任务，适合实时语音质量增强。
2. **AlignBlock**：显式 Q/K 与时延卷积对齐，提升双讲与延迟鲁棒性。
3. **双分支 Encoder**：mic 深、far 浅，匹配 AEC 中近端主导、远端辅助的角色。

**论文原文要点**（*DeepVQE*，**结构创新**）：① **双分支 Encoder**：mic 分支 5 块（base）/4 块（small）、far 分支 2 块，深度不对称匹配近端主导、远端辅助；② **AlignBlock**：Q/K 点卷 + K 沿时间 unfold，soft-align 近端与远端特征，替代硬对齐；③ **Bottleneck**：flatten → GRU → linear → reshape，时域建模；④ **Decoder**：skip、ResBlock、SubPixel 上采样；⑤ **CCM（Complex Convolving Mask）**：复数因果卷积，逐 T-F bin 可学习混合邻域；单模型联合 AEC+NS+去混响。

**Loss**

- 当前仓库 YAML 与项目 API 示例均显式使用 **`CompositeLoss`**：`GTCRNCompressedSpectralLoss(weight=1.0)` + `GTCRNLogRatioLoss(weight=1.0)`。前者按顶层 STFT 几何计算压缩复谱 RI/幅度误差：幅度固定定义为 `sqrt(real²+imag²+1e-12)`，再用于复谱与幅度压缩，并覆盖 full batch；后者是波形域的 GTCRN `-log10(ratio+eps)`，由 Composite pure 路由仅在 non-pure 且 clean-energy 有效的样本上归约；不能把 log-ratio 项写成标准 SI-SNR。谱进谱出模型下两者均可消费同一 framework STFT 的 direct spectra：谱项直接归约，log-ratio 项按有效帧分组 iSTFT 重建波形后归约。
- **GTCRN 公式来源与 AEC 适配边界**：固定 commit [`502ebfab`](https://github.com/Xiaobin-Rong/gtcrn/tree/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73) 的官方 [`loss.py`](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/loss.py) 用 `512/256/512 + sqrt-Hann` 把 prediction/target spectra 分别 iSTFT 后计算 log-ratio，[`infer.py`](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/infer.py) 在 STFT/iSTFT 两端使用同一 sqrt-Hann；该配对只是官方手工、无包络归一化 WOLA 前端的产物：两个窗的乘积为 Hann，50% overlap 下相邻 Hann overlap-add 包络为常数，并非公式本身的要求。同作者 [`SEtrain/loss_factory.py`](https://github.com/Xiaobin-Rong/SEtrain/blob/3ec437d1adef76376775caa9e639d01208426380/loss_factory.py) 的 waveform `HybridLoss` 使用普通 Hann STFT，并直接在同一 `y_pred/y_true` raw pair 上计算投影项。项目在 AEC 中也把同一 raw `(nearend_estimate, nearend)` pair 与同一个 `lengths` 交给两个 component；`lengths=None` 时两项均消费完整波形。两个 component 均可消费同一 framework STFT 的 direct spectra：谱项直接归约，log-ratio 项继承顶层 `stft` 几何与同窗做 iSTFT，依赖 `torch.istft` 包络归一化对一致谱精确重建，变长样本按有效帧分组并施加有效长度掩码。这是与官方 `HybridLoss` 公式逐项对齐的组件复用，但 GTCRN 官方任务是 SE，AEC 侧应用未经官方验证。
- 当前实际配置：`configs/aec/microsoft_challenge/deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml`（文件名中的 `hybrid` 是历史命名；配置内容为显式 Composite；32 kHz，`n_fft=640`、`hop_length=320`、`win_length=640`、普通 Hann、`align_delay=10`）；10 ms 帧移下使用当前帧加前 9 帧，内部历史跨度约 90 ms，实际 `aec_ref_echo_delay_range_ms` 为 `[0.0,80.0]`。该顶层普通 Hann 同时作为 direct 路径 iSTFT 的合成窗，与框架 analysis 同窗，overlap-add 包络由 `torch.istft` 归一化处理。

**参数量/算力**

- 论文 base 7.5M / small 0.59M；当前 32 kHz meeting 配置头记录 small 为 960,455 参数、约 1.511G MACs/s（1 秒谱域前向，不含 STFT/iSTFT）。当前实现的主要模块由论文 Section 2 派生，但窗口、STFT/iSTFT、工程化 padding、小模型宽度和输入校验以仓库实现为准，不是逐项严格复现。

**可复用模块**

| 模块 | 可复用场景 |
|------|------------|
| 双分支 Encoder | 任意 mic/ref 双流 AEC 或多任务模型。 |
| AlignBlock | 需显式时延对齐的 AEC 或双流融合。 |
| CCM | 复数因果卷积，可用于其它复谱后处理。 |

---

### 3. 其他 AEC 模型指路

以下模型未在本文展开，详见各自专项文档：

- **DBGNet**（状态检测 + AEC）：见 [`tasks/aec/README.md`](../../tasks/aec/README.md) 的 DBGNet 小节。
- **LocalVQE**（外部移植，DCT→STFT 前端）：`tasks/aec/models/discriminative/localvqe.py`；结构对比见 [`docs/paper/aec/analysis/LocalVQE_vs_DeepVQE_Model_Analysis_2026-07-17.md`](../paper/aec/analysis/LocalVQE_vs_DeepVQE_Model_Analysis_2026-07-17.md)。
- **UltraDPCAEC**：已注册（`tasks/aec/models/`）；静态/offline 模型具备时间/频率压缩与 PostNet 接线，流式 wrapper 对频率压缩有原生 cache 路径、对时间压缩仅作逐帧 fallback。`onnx/verify/verify_3way_ultra_dpc.py` 具备 offline PyTorch / streaming PyTorch / ORT 比较能力，但只有 `time_skip=1` 的 static/offline-equivalent 配置可执行完整三路比较；`time_skip>1` 会跳过 offline 两项，只能比较流式 PT/ORT。本轮未绑定 checkpoint 与命令运行，不声明任何配置已通过；暂无生产训练配置。论文精读见 [`docs/paper/aec/Tencent/26_Ultra-DPC-AEC.md`](../paper/aec/Tencent/26_Ultra-DPC-AEC.md)。

## 二、AEC 模型结构对比表

| 模型 | 结构分解（公式化） | 参数量(M) | 算力(G MACs) | 主要 Loss | 配置路径 | Loss 文件路径 | 论文创新点摘要 |
|------|--------------------|-----------|--------------|-----------|----------|----------------|----------------|
| CAGCRN | ERB(65,64) + CATA + TFGRU×3 + Dec + Mask + TFAG | — | — | CompositeLoss | configs/aec/microsoft_challenge/cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml | core/losses | Cross-Attention 对齐、频率双向/时间单向 TFGRU |
| DeepVQE | 双分支 Enc + AlignBlock + Bottleneck(GRU) + Dec + CCM | 论文 base 7.5 / small 0.59；32 kHz meeting small 0.960455 | 32 kHz meeting small 约 1.511 | CompositeLoss（GTCRN 两项显式展开） | configs/aec/microsoft_challenge/deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml | core/losses | 联合 AEC+NS+去混响、AlignBlock、双分支 Enc |
*参数量/算力口径以对应配置头或可复现 audit 命令为准；SE 数据以 `SE_Architecture.md` 与 audit_metrics 为准。*

---

## 三、跨模型可复用模块汇总

该对比表（含「优化优先级建议」）维护在 [SE_Architecture.md](SE_Architecture.md) §四「跨模型可复用模块汇总」，本文不再重复。


---

## 四、结构对照与选型（AEC 摘录）

- **DeepFilterNet vs DPDFNet**（代码与论文一致）
  - **相同**：ERB 包络 + Deep Filtering 双路径 Encoder，GroupedLinear/SqueezedGRU，ErbDecoder + DfDecoder，LSNR 头训练时辅助。
  - **不同**：① **DPRNN**：DPDFNet 在 ERB 路径与 DF 路径末端各插 `DPRNN`（`dprnn_num_blocks`=2/4/8），DF2/DF3 为 0（`tasks/se/models/discriminative/deepfilternet.py` / `tasks/se/models/modules/deepfilternet_impl.py::Encoder`）；② **融合**：DF2/DPDFNet 用 Concat，DF3 用 Add；③ **Mask**：DF3 分频（低频无 Mask），DF2/DPDFNet 全频。论文：DPDFNet 在 DF2 encoder 中集成 dual-path 模块并使用 over-attenuation loss。
  - **时序边界**：当前 DPDFNet 配置固定 `df_lookahead=2`、`conv_lookahead=2`，因此是 **near-causal**，不是严格因果；16 kHz、`hop_length=160` 时对应约 20 ms 前瞻。cache 只能复用历史状态，不能消除这 2 帧未来上下文。

- **CAGCRN vs DeepVQE**（代码与论文）
  - **CAGCRN**：极轻量联合 AEC+NS；**CATA** + **TFGRU** + **TFAG**，见 `tasks/aec/models/discriminative/cagcrn.py`。
  - **DeepVQE**：双分支 Encoder + **AlignBlock** + GRU Bottleneck + Decoder + **CCM**，单模型 AEC+NS+去混响，参数量较大，见 `tasks/aec/models/discriminative/deepvqe.py`。
  - **异同**：两者均为 mic/ref 双流与显式对齐；CAGCRN 极轻量 CRN，DeepVQE 重表达多任务。

- **SE 内部**：GTCRN 与 FastEnhancer 偏极轻量；BSRNN 覆盖 URGENT 基线与大参数量、以及 USR 轻量多 SR；DeepFilterNet 系列在质量与实时性间可扩展（DPRNN 块数）；FastULCNet 针对长序列与压缩域 CRM。
- **AEC**：CAGCRN（CATA+TFGRU+TFAG，极轻量）与 DeepVQE（AlignBlock+CCM，重表达多任务）为当前仓库内两条主实现路径。

- 将 **ERB**、**DPGRNN/DPRNN**、**BandSplit+MaskDecoder** 抽象为统一接口（如 `tasks/se/modules/`），便于跨模型复用与消融；对 ERB 提供 n_fft/fs 预设与可学习选项。
- **CATA / AlignBlock** 在 AEC 中可做对比实验（延迟鲁棒性、双讲）；CRM 与 LSNR 头可单独评估对客观指标与主观听感的影响。
- **嵌入式/实时**：优先考虑 GTCRN、FastEnhancer-T/B、DeepFilterNet2/DPDFNet-2；BSRNN-Lite 适合多 SR 边缘。
- **高质量/离线**：BSRNN (URGENT)、DPDFNet-4/8、DeepVQE-base；AEC 双讲与延迟敏感场景优先 CAGCRN/DeepVQE（显式对齐）。

## 当前真实链路验证记录（2026-09-12）

当前 `tasks/aec` 与共享 `core` 链路已使用历史 DeepVQE 32 kHz checkpoint 完成以下验证：

- 真实 mic/farend WAV 离线推理，输出采样率、长度和有限值检查通过；
- 公共 `run_manifest.json`、`inf.scp`、provenance 和 output identity 发布通过；
- `skip-existing` 在 artifact/input/output identity 稳定时返回 `skipped`，模型不重复前向；
- streaming AEC reset/flush smoke 通过；
- 1000 帧、两次 reset 的离线 PyTorch、streaming PyTorch、streaming ONNX Runtime 三路对齐通过；
- runtime metadata、cache shape、输入输出名称和动态轴检查通过；
- 当前 canonical 数据配置完成一 epoch CPU 训练 smoke，覆盖 dataset、collate、STFT、DeepVQE forward、CompositeLoss、optimizer、validation 和 checkpoint 保存。

历史实验 `resolved_config.yaml` 不能直接作为当前训练配置使用。重建训练数据前必须使用当前资源路径、split-level dataset lifecycle 字段和 canonical `prob_target_*` 场景概率键；旧场景键不会自动迁移。
