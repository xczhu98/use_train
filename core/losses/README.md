# `core/losses`：组合 Loss、MBRCL 与 TAPLoss 说明

**这是什么**：`core/losses/` 是训练目标（loss）的实现目录。loss 决定「模型往哪个方向优化」——
它把模型输出与参考音频的差距折算成一个可反向传播的标量，选错公式或配错权重，模型就会朝错误方向学。
本文是该目录的维护说明，面向要改这些 loss、或要核对某个 loss 到底怎么算的人。

本文覆盖下列 Loss：

- `SIL1WavSeTrainLoss`
- `MRSIRelCspecL2Loss`
- `MRSpecSuppressPenaltyLoss`
- `MRSIRelCspecL2SuppressPenaltyLoss`
- `MultiBandRelativeContrastiveLoss`（MBRCL）
- `TAPLoss`（官方兼容别名 `AcousticLoss`）
- `ComplexIdealRatioMaskLoss`（cIRM-MSE）
- 以及负责组合它们的 `CompositeLoss`

实现以 [`time_domain.py`](time_domain.py)、[`se_train_mr.py`](se_train_mr.py)、[`hybrid.py`](hybrid.py)（GTCRN 压缩谱项与 log-ratio 项）、[`mbrcl.py`](mbrcl.py)、[`tap.py`](tap.py)、[`complex_mask.py`](complex_mask.py) 和 [`composite.py`](composite.py) 为准；构建入口是 [`build.py`](build.py)。完整的 YAML 结构约束和多分辨率 schedule 说明见 [`Loss_Configuration_Guide.md`](../../docs/guides/loss/Loss_Configuration_Guide.md)；面向使用者的逐 Loss 公式、论文对照与归属边界由 [`docs/guides/loss/`](../../docs/guides/loss/README.md) 维护，本文只保留与实现直接相关的契约细节。

## 术语速查

| 术语 | 含义 | 为什么要关心 |
|---|---|---|
| Loss / 训练目标 | 把「模型输出与参考音频的差距」折算成一个标量，反向传播靠它决定优化方向 | 换 loss 等于换模型的学习目标 |
| 波形域 | 直接在时域采样点上算误差（如 `SIL1WavSeTrainLoss`） | 直观、对相位敏感 |
| 谱域 / STFT | 先做短时傅里叶变换（STFT）在时频图上算误差 | 多数音质类目标在谱域更稳定 |
| `lengths` | `[B]` 整数张量，表示每条样本的有效采样点数 | 变长 batch 中 padding 区不应参与 loss |
| reduction | 把逐样本、逐元素误差聚合成标量的方式（均值、masked mean 等） | 换 reduction 等于换各样本的权重分配 |
| padding 不变性 | 同一条音频无论单独跑还是与别的样本拼成 batch，loss 应完全一致 | 否则结果会随 batch 组成漂移 |
| SS（desired/undesired 能量混合） | 除目标语音支路外，再用「非目标」（噪声/回声/其它说话人）支路监督，并按能量比例混合 | 增强抑制能力；比例由能量算出，不是可学习参数 |
| pure 样本 | 没有目标语音的样本（`is_pure_noise` / `is_pure_echo` / `is_pure_echo_noise`） | 这类样本不能套「接近 clean 语音」的目标 |
| capability 标志 | 类属性，声明该 Loss 需要 noisy 参考、采样率、direct spectra 等输入 | 决定 Trainer 要不要给它额外接线 |
| `CompositeLoss` | 把多个 Loss 按权重相加的组合器；`build_loss` 特判处理，不注册进 registry | 绝大多数训练配置的顶层 Loss |

> **归属声明**：`SIL1WavSeTrainLoss`、MR 复谱/抑制项以及它们在 `CompositeLoss` 中的 desired/undesired 能量混合，是本项目对 `se_train` 风格目标的工程化实现与组合，属于项目自研或近邻实现，不是某篇论文的逐字复现。MBRCL 对照 Shen 等人的论文公式实现，但论文没有公开源码；项目的 builder 上下文、动态 SR cache、变长 batch，以及将论文公式一致地扩展到 `target=0` 的 pure 样本属于工程适配。论文没有单独验证 pure 场景，不能把该工程扩展写成论文实验结论。`TAPLoss` 则分为论文、官方实现与项目适配三层来源。本文不会把论文指标写成项目实测结果。

## 静音目标监督

`AbsentTargetLoss` 仅在 target 均方能量不超过 `1e-10` 时惩罚输出均方能量，支持 lengths，padding 不参与计算。与 `SISNRLoss` 组合用于 TSE 的 noise-only / target-absent 样本；全 present batch 返回连接计算图的零。启用 I-only、N-only 或 I+N 的 TSE 配置必须包含非零 `AbsentTargetLoss` 或 `L1WavLoss`。

## 1. 统一接口与调用位置

### 1.1 标准波形接口

标准二元 Loss 与 `CompositeLoss` 子项的基础调用为：

```python
loss = loss_fn(enhanced, clean, lengths=None)
```

MBRCL 在该基础上显式要求 noisy negative anchor，并允许 forward 传入实际采样率：

```python
loss = mbrcl(
    enhanced,
    clean,
    lengths=None,
    noisy=noisy,
    sample_rate=batch_sample_rate,
)
```

其精确 API 为 `forward(enhanced, clean, lengths=None, *, noisy=None, sample_rate=None, **kwargs)`；`noisy` 缺失会 fail-fast，forward 的 `sample_rate` 会覆盖构造期 SR 以选择当前 hard-band matrix。

其中：

- `enhanced`：模型输出波形，形状 `[B, T]`；
- `clean`：目标波形，形状必须与 `enhanced` 相同；pure 样本可为全零 target；
- `noisy`：MBRCL 必需的原始输入波形，形状必须与 `enhanced` 一致；
- 三路波形必须位于同一 device 且均为浮点 Tensor，但可使用不同浮点 dtype；MBRCL 会统一转为 FP32 后执行 STFT、频带投影与 reduction，返回 FP32 标量；
- `lengths`：可选的 `[B]` 整数张量，表示每条样本的有效 sample 数；
- 返回值：零维标量 Tensor，可直接用于反向传播。

训练入口通过 [`build_loss`](build.py) 从 `LOSS_REGISTRY` 构造已注册 Loss。`MultiBandRelativeContrastiveLoss` 是精确 registry/API 名称；registry 名称与 LossContract inventory 必须保持一致。`CompositeLoss` 由 `build_loss` 特判，不单独注册。

仓库现行 GTCRN 配方显式使用 `CompositeLoss`，其中 `GTCRNCompressedSpectralLoss(weight=1.0)` 与 `GTCRNLogRatioLoss(weight=1.0)` 构成一对 component，并保留同一 Composite 中其他 component 及其权重。前者固定令 `|P|=sqrt(Re(P)²+Im(P)²+1e-12)`、`|T|=sqrt(Re(T)²+Im(T)²+1e-12)`，再以 `P_c=P/|P|^(1-c)`、`T_c=T/|T|^(1-c)` 计算 `lamda_ri * (MSE(Re(P_c),Re(T_c)) + MSE(Im(P_c),Im(T_c))) + lamda_mag * MSE(|P|^c,|T|^c)`；这里的 `1e-12` 是固定公式常数，不由可配置 `eps` 或 clamp 代替。STFT 几何由顶层 `stft` 注入；该分量不依赖非零语音参考，因此 pure/`target=0` 仍在 full batch 使用同一公式。后者直接在有效波形区令 `s_target=(<clean,enhanced>/(||clean||²+eps))*clean`、`e_noise=enhanced-s_target`，并按官方精确边界归约 `-log10(||s_target||²/(||e_noise||²+eps)+eps)`；官方公式不先对波形去均值。`CompositeLoss` 先按 pure metadata 将它路由到 non-pure 子集，类内 clean-energy guard 再过滤近静音参考。

GTCRN 两个 component 都声明 `SUPPORTS_DIRECT_SPECTRA`：谱进谱出模型提供 `enhanced_spectra` 时，Trainer 以同一个 framework STFT 和真实 `lengths` 生成 trim-consistent `target_spectra`，再把两路谱直接交给两个 component。`GTCRNCompressedSpectralLoss` 直接在两路谱上归约，避免 `iSTFT→STFT` 将模型预测投影到一致谱子空间；`GTCRNLogRatioLoss` 按各样本有效 frame count 分组裁剪谱，以继承顶层 `stft` 的几何与同窗执行 `torch.istft` 重建波形（提供 `lengths` 时施加 `t < lengths` 有效区掩码），再在重建波形对上计算同一投影 log-ratio。两路谱必须同时提供并匹配 Loss/framework 的 `n_fft`、`hop_length`、`win_length`、`window`、`center`、`normalized`；缺少 direct-spectrum 上下文时两者仍使用 waveform adapter。`CompositeLoss` 只把 `enhanced_spectra`/`target_spectra` 路由给声明 `SUPPORTS_DIRECT_SPECTRA` 的组件。当前 direct-spectrum 路径没有 undesired prediction/reference spectra，因此活动 direct component 启用 SS 时 fail-fast；pure metadata 不会把压缩谱项从 full batch 排除。

`GTCRNLogRatioLoss` 不能简写成 `0.1 * SISNRLoss`：按 `eps=1e-8`，官方公式在 `||s_target||²=0` 时给出最大惩罚 `8`，但不对 ratio 设置上界，因此高 ratio 端没有固定的 `-8` 下界；项目 `SISNRLoss` 则将 SI-SNR 封顶到 `80 dB`，乘 `0.1` 后最低为 `-8`，且低 ratio 端没有 GTCRN 公式的 `+eps` floor。

**GTCRN 官方前端证据与项目边界**：固定上游 commit [`502ebfab`](https://github.com/Xiaobin-Rong/gtcrn/tree/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73) 的 [`loss.py`](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/loss.py) 直接接收 `pred_stft/true_stft`，谱项后用 `512/256/512 + sqrt-Hann` 分别 iSTFT，再在所得波形对上计算尺度投影 log-ratio；同 commit 的 [`infer.py`](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/infer.py) 则在 STFT 与 iSTFT 两端都使用同一 `sqrt-Hann`。该配对只是官方手工、无包络归一化 WOLA 前端的产物：分析窗与合成窗逐点相乘得到普通 Hann，50% overlap 下相邻 Hann 的 overlap-add 包络为常数，从而在无归一化条件下避免窗函数引入周期性幅度起伏；它不是公式本身的要求。同作者的 [`SEtrain/loss_factory.py`](https://github.com/Xiaobin-Rong/SEtrain/blob/3ec437d1adef76376775caa9e639d01208426380/loss_factory.py) 又提供了 waveform `HybridLoss` 证据：谱项用普通 Hann STFT，而投影项直接使用同一 `y_pred/y_true` raw waveform pair，说明作者代码族本身就存在“公式保持、前端几何随承载框架变化”的实现。

本项目据此做的是与官方 `HybridLoss` **公式逐项对齐的组件拆分**：`GTCRNCompressedSpectralLoss` 对应官方 30/70 加权的压缩 RI/magnitude 项，`GTCRNLogRatioLoss` 对应尺度投影 log-ratio 项（`-log10(ratio+eps)`，非 dB）；`CompositeLoss` 将同一 raw `(enhanced, clean)` pair 和同一个 `lengths` 同时交给两个 component，`lengths=None` 时两者都消费完整波形，不做项目侧 trim。两个 component 都可额外接收由同一顶层 STFT 产生的 direct spectra，这不改变 raw pair 的共同调用语义。direct 路径的 iSTFT 完整继承模型顶层 `stft` 的有效 `n_fft` / `hop_length` / `win_length` / `window` / `center` / `normalized`，analysis/synthesis 使用同一普通窗（hann/hamming），NOLA 与几何合法性由 `resolve_stft_config` 在构造期校验，并依赖 `torch.istft` 自带的窗包络归一化对一致谱精确重建——官方 sqrt-Hann 两端配对只是其手工无归一化 WOLA 前端的产物，本项目不需要复制。

模型顶层 `stft` 是生产分析几何的单一事实来源，完整有效参数为 `n_fft` / `hop_length` / `win_length` / `window` / `center` / `normalized`。框架要求 `n_fft`、`hop_length`，省略 `win_length`、`window`、`center`、`normalized` 时分别解析为 `n_fft`、`hann`、`true`、`false`，再向显式接受这些参数的 Loss 注入有效值。普通 STFT-aware Loss 保留显式 `args` 的 `setdefault` 覆盖语义；MBRCL 声明严格匹配，只要存在顶层上下文，其显式 STFT 参数就必须与框架有效值一致（window 名大小写不敏感），否则构建期 fail-fast。固定预训练几何的类可声明 `USE_FRAMEWORK_STFT_DEFAULTS = False` 完全退出注入；**`TAPLoss`/`AcousticLoss` 即采用该退出契约**，因为其 estimator 输入固定为 16 kHz、`512/160/512`、矩形窗与 `center=True`。

训练入口把原始 `data.sample_rate` 传给 `build_loss`：字段缺失或显式为 `null` 时，sample-rate 构建上下文均为 `None`，不会在 builder 调用处补成 `16000`；非空值才会按 `sample_rate`、`sampling_rate`、`sr`、`fs` 的顺序向 Loss 显式声明的第一个构造参数注入。显式 `args` 中的任一采样率别名优先，只有 `**kwargs` 不构成注入契约。因此 `PaseMultiScaleMelLoss.sampling_rate` 默认继承非空的 `data.sample_rate`，显式 `args.sampling_rate` 可覆盖；没有构建上下文时保留类默认值 `16000`。MBRCL 的显式构造 SR 必须与非空上下文一致。构造期 SR 未知时，每次 forward 必须得到有效 runtime batch SR。标准 SE/AEC/TSE Trainer 以及基础 `GANTrainer`/`PASEGANTrainer` reconstruction 路径共用 waveform capability interface，对声明 `REQUIRES_SAMPLE_RATE` 的活动 Loss 传入 batch 实际 SR，runtime 值覆盖构造期值。Trainer 自身解析 `data.sample_rate` 时，仅在 `sample_rate` 键缺失时保留历史 `16000` fallback；键存在且显式为 `null` 时仍为 `None`。单个 batch 内必须同 SR，不同 SR 的连续 batch 可由 MBRCL 按实际 SR 动态建立并复用线程安全的 canonical CPU FP32 per-SR hard-band cache。`TAPLoss` 只接受 `16000`，错误采样率会在构造期 fail-fast。`ScalarPassthroughLoss` 独立注册于无 `librosa`/`numba` 重依赖的轻量模块，PASE 可选依赖不可用时，dual-input scalar-loss 配置仍可正常构造。

`build_loss(..., strict=True)` 还会包裹严格校验：输入形状、`lengths` 的整数类型和值域 `[1, T]`，以及标量输出的有限性。结构/类型/值域违规分别抛 `ValueError`/`TypeError`，非有限数值抛 `StrictFiniteLossError`；这些门禁不依赖 `assert`，因此在 `python -O` / `PYTHONOPTIMIZE=1` 下同样生效。直接实例化类时，以各类自身代码路径为准，不应把 builder 的外围校验误认为每个类都自动执行。

### 1.2 MBRCL capability 与 pure 样本语义

MBRCL 声明 `REQUIRES_NOISY_REFERENCE`、`REQUIRES_SAMPLE_RATE`，并拒绝 SS undesired-energy。标准 SE/AEC/TSE Trainer 通过共享 interface，只对活动的单 Loss 或 Composite 非零分量注入三路退化输入与实际 batch SR；映射分别为 SE `noisy→clean→enhanced`、AEC `mic→nearend→nearend_estimate`、TSE `mixture→target→target_estimate`。TSE multi-target 使用 forward 后按 target row 展开的 mixture，零权重 MBRCL 不触发额外上下文。基础 `GANTrainer`/`PASEGANTrainer` reconstruction 路径同样接入；`MetricGANCompositeTrainer`、adapter distillation 等自行拥有专用 objective、不会调用标准 `self.loss_fn` 的 Trainer 不适用。AEC/TSE 复用属于公式可计算的工程扩展，论文没有验证其任务效果。

MBRCL 对 full batch 归约，不根据 `is_pure_noise`、`is_pure_echo`、`is_pure_echo_noise` 裁剪样本。合法 pure-noise 样本（`clean=0`、`noisy` 非零）中，`L_band` 对非零 enhanced residual 通常提供相对输入抑制梯度，`L_env` 约束残余包络的时间变化；后者单独不保证残余能量趋零。但若 `enhanced` 恰为零，因功率谱原点导数为零，即使 `noisy` 非零，`L_band` 对 `enhanced` 的梯度也为零，该输出是 stationary point。因此，“对 full batch 计算”只表示公式与归约覆盖 pure 样本，不保证每个 pure 输出点都有非零抑制梯度。

三路全零是还带有固定 loss floor 的特殊边界：当 `enhanced=clean=noisy=0` 时，论文 softplus 公式产生非零常数，一般式为 `band_weight * sum_b softplus(margin_b)`；当前统一 margin 时为 `band_weight * B * softplus(margin)`，按默认 `band_weight=0.5`、`B=16`、`margin=0.06` 约为 `5.7887769`。此时 `L_env=0`，loss 对 `enhanced` 的梯度严格为零，因此不得把该日志常数误判为模型退化。上述 `target=0` 行为是与论文公式一致的项目工程扩展；论文没有单独验证 pure 场景，不应将其表述为论文实验证明。

构造期与 runtime hard-band matrix 均不进入 `state_dict`，不改变模型 checkpoint。pickle/deepcopy 会排除可重建的 runtime cache 与线程锁，反序列化后重建锁并按需重新生成 per-SR cache；这些对象也不参与部署前向。

### 1.3 `lengths` 与 padding 不变性

传入 `lengths` 后，以下 waveform/STFT Loss 只应让有效区参与 reduction；TAPLoss 的变长语义单独见第 7 节：

- `SIL1WavSeTrainLoss` 在波形绝对误差和参考能量上做 sample mask；
- 两个独立 MR Loss 在频谱帧上做 mask，并在有效边界使用 reflect 延展，使结果等价于先 trim 到有效长度再做 `center=True` 的 STFT；
- `MRSIRelCspecL2SuppressPenaltyLoss` 分别调用两个 MR 子项，沿用同一口径；
- `SpectralConvergenceLoss`、`LogMagnitudeSTFTLoss`、`MultiResL1SpecLoss`、`PaseMultiScaleMelLoss` 与 `GAPHybridLoss` 复用 `utils.py` 的 length-aware centered-STFT 输入准备和有效帧 mask；PASE 的每个 mel 尺度均只归约有效帧，GAP 的 signed SDR 只统计有效 samples；其中 `MultiResL1SpecLoss` 的方差、尺度投影及时域 L1 也只统计有效 samples；`SpectralConvergenceLoss` 对逐样本原始 target 谱范数 `<=1e-8` 的零/近零参考复用同一有效帧上的 log-magnitude L1，不再以谱 floor 或 `1e-8` 作为 SC 分母；非零参考仍保持标准 `||Y-X||_F/||Y||_F` 公式，完美重建点返回有限零梯度；
- `GTCRNCompressedSpectralLoss` 的 waveform adapter 与 Trainer 构造 direct `target_spectra` 的路径均按真实 `lengths` 准备 STFT 输入，direct prediction/target 谱则按同一有效帧 mask 归约；`GTCRNLogRatioLoss` 的 direct 路径按各样本有效 frame count 分组裁剪谱后 iSTFT，并对重建波形施加 `t < lengths` 有效区掩码，只在有效覆盖区计算投影 log-ratio，聚合为 valid 样本均值；
- MBRCL 对三路波形使用同一有效 sample mask，并按其顶层有效 `center` 几何构造 frame mask；当所有 `lengths == T` 时与 `lengths=None` 保持 full-valid parity，包括短于分析窗、甚至仅一个有效 sample 的极短输入。

`prepare_centered_stft_inputs` 先清零有效长度外的原始 canvas，再无条件追加 `n_fft // 2` 个右侧槽位，并从每条样本的真实有效边界写入 reflect 延展；它不依赖调用方原本提供了多少右 padding。有效长度 `L <= n_fft // 2` 时使用无端点周期反射，`L=1` 时复制唯一 sample，因此极短输入也满足 `torch.stft(center=True, pad_mode="reflect")` 的约束。由此，trim、轻量右 padding 与完整 batch padding 应得到相同 loss 和有效区梯度，原始 padding 区梯度为零。直接调用底层 `prepare_stft_waveform_at_lengths` 的 waveform adapter 在 `lengths=None` 且输入长度大于 `n_fft // 2` 时保持原波形不变；只有不超过该阈值的短输入才自动进入同一延展路径。下式仅描述前述固定 `center=True` 的谱域 Loss；MBRCL 由 `create_frame_mask(..., center=self.center)` 使用自己的顶层有效 `center`：

\[
T_{\text{frame}}(L)=\max\left(1,1+\left\lfloor\frac{L+2\lfloor n_{fft}/2\rfloor-n_{fft}}{hop}\right\rfloor\right).
\]

MR 共享路径使用 Hann 窗、`onesided=True`、`center=True` 和 FP32；MBRCL 则使用顶层有效 Hann/Hamming 与 `center` 几何，但同样在 STFT 前统一为 FP32。相关实现见 [`utils.py`](utils.py) 与 [`stft_cache.py`](stft_cache.py)。

## 2. `SIL1WavSeTrainLoss`：se_train 风格 SI-L1

源码：[`time_domain.py:17`](time_domain.py#L17)。它不是普通 MAE，而是用 clean 的平均绝对幅度进行归一化的逐样本 L1：

\[
 d_b=\operatorname{mean}_{t\in V_b}|\hat{s}_{b,t}-s_{b,t}|,
 \qquad
 r_b=\operatorname{mean}_{t\in V_b}|s_{b,t}|,
\]

\[
 L_b=
 \begin{cases}
 10\,d_b/(r_b+10^{-8}), & r_b>10^{-12},\\
 10\,d_b/(0.1+10^{-8}), & r_b\le 10^{-12},
 \end{cases}
 \qquad
 L_{\mathrm{SI\text{-}L1}}=\frac1B\sum_b L_b.
\]

`V_b` 是有效 sample 集合；没有 `lengths` 时就是整条 `[T]` 波形。这里的“scale-invariant”是幅度比值意义上的归一化，源码没有先求一个可学习或最优的波形缩放因子。

### 2.1 方向偏好与保护

- **优化方向**：最小化；增强波形越接近 clean，loss 越低。
- **对过估计/欠估计是否对称**：波形 L1 本身对误差绝对值对称，不区分增强幅度高于还是低于目标。
- **低能量保护**：当 clean 几乎静音时，不用接近零的 `r_b` 做除数，而使用固定参考幅度 `0.1`。这避免静音或极低电平样本制造异常大的梯度；它不会把该样本从 `SIL1WavSeTrainLoss` 中删除。
- **reduction**：先逐样本归一化，再对 batch 求均值；有效长度只影响各样本内部的 sample 均值。

## 3. 多分辨率 schedule

`MRSIRelCspecL2Loss`、`MRSpecSuppressPenaltyLoss` 以及组合类使用 `_resolve_stft_schedule` 解析分辨率。两种模式互斥：

### 3.1 Legacy power 模式

默认 `fft_power_min=6`、`fft_power_max=11`，因此：

```text
fft_sizes = [64, 128, 256, 512, 1024, 2048]
hop_sizes = [32, 64, 128, 256, 512, 1024]
```

显式给出 `fft_power_min`/`fft_power_max` 时，仍是绝对 sample 数，不会根据采样率或顶层 STFT 自动缩放。要求两个 power 是整数并满足 `1 <= min <= max`。

### 3.2 Frame-relative 模式

给出 `stft_scale_factors` 后，必须同时得到正整数的 `win_length` 和 `hop_length`（通常由顶层 `stft` 注入）：

\[
 n_{fft,k}=win\_length\times f_k,
 \qquad
 hop_k=hop\_length\times f_k.
\]

例如 `factors=[0.125, 0.25, 0.5, 1, 2, 4]`、`win_length=320`、`hop_length=160` 时，得到：

```text
fft_sizes = [40, 80, 160, 320, 640, 1280]
hop_sizes = [20, 40, 80, 160, 320, 640]
```

每个乘积必须是精确正整数；不会静默 round。不能同时配置 `stft_scale_factors` 与任一 legacy power 参数，factor 序列还必须非空、有限、正数且严格递增。

## 4. `MRSIRelCspecL2Loss`：多分辨率 SI-relative 复谱平方误差

源码：[`se_train_mr.py:168`](se_train_mr.py#L168)。每个分辨率 `k` 对 `enhanced` 与 `clean` 做 STFT，并以实数布局 `[B, F, T_k, 2]` 表示复谱。记：

\[
 D_{b,f,t}^{(k)}
 =\left\|E_{b,f,t}^{(k)}-C_{b,f,t}^{(k)}\right\|_2^2,
 \qquad
 P_{b,f,t}^{(k)}=\left\|C_{b,f,t}^{(k)}\right\|_2^2.
\]

这里的 `D` 是 real/imag 两个分量平方和；当前代码**没有额外做 power-law compression**，不要将该类误写成压缩谱损失。

有效频谱元素 mask 记为 `M`，`N_k=sum(M)`。归一化参考量 `q_b` 按如下源码分支计算：

- 无 `lengths` 且 `align_se_train=True`：`q_b = mean(P_b)`；仅当 `q_b < 1e-20` 时**替换**为 `0.5`，因此 `1e-20 <= q_b < 0.5` 的非静音值会原样保留；
- 无 `lengths` 且 `align_se_train=False`：`M` 是全 1，先得到同一个全网格均值，再执行 `q_b = max(q_b, 0.5)`；
- 提供 `lengths`：只在有效帧上计算 `q_b = sum(MP)/sum(M)`，随后同样 clamp 到最低 `0.5`；此时 `align_se_train` 不改变结果。

默认启用的频率权重为：

\[
 w_f=2\cos\left(\operatorname{linspace}(0,\pi/2,F)_f-\pi/16\right)+0.5.
\]

若 `use_frequency_weight=False`，令 `w_f=1`。单个分辨率的两项为：

\[
 L_{\mathrm{SI},k}
 =\frac{1}{N_k}\sum_{b,f,t}M_{b,f,t}
   w_f\frac{D_{b,f,t}}{q_b+10^{-12}},
\]

\[
 L_{\mathrm{rel},k}
 =\frac{1}{N_k}\sum_{b,f,t}M_{b,f,t}
   w_f\frac{D_{b,f,t}}{P_{b,f,t}+0.05}.
\]

最终输出是所有分辨率的平均：

\[
 L_{\mathrm{MR\text{-}SI\text{-}Rel}}
 =\frac1K\sum_{k=1}^{K}
  \left(L_{\mathrm{SI},k}+\alpha_{\mathrm{rel\_L2}}L_{\mathrm{rel},k}\right).
\]

### 4.1 方向偏好与低能量保护

- **优化方向**：最小化复谱误差；同时压低 real/imag 误差。
- **方向对称性**：`D` 是平方误差，因此对 `enhanced` 高于或低于 clean 的误差对称；该类本身不偏好“宁可少保留”或“宁可多保留”。
- **频率偏好**：默认 `w_f` 是固定的启发式频率权重，不是由当前样本 SNR 计算的 SNR 权重。
- **低能量保护**：`q_b + 1e-12` 防止 SI 归一化除零；`P + 0.05` 防止 clean 低能量时 relative 项无限放大。`0.5` 是 `align_se_train=False` 或变长分支的下界；无长度的对齐分支只替换 `<1e-20` 的近零值，不是普遍 clamp。
- **参数**：`alpha_rel_L2` 控制 relative 项；`use_frequency_weight` 控制 `w_f`；`align_se_train` 只影响 `MRSIRelCspecL2Loss` 在 `lengths is None` 时的 `q_b` 低能量处理。传入 `lengths` 后它不生效；在 `MRSpecSuppressPenaltyLoss` 中该参数仅为构造兼容位，当前 forward 两个长度分支都不读取它，因此非默认值（`false`）会在构造期显式拒绝，`MRSIRelCspecL2SuppressPenaltyLoss` 不会把它转发给 penalty 子项。

### 4.2 `stft_cache` 使用

单独调用时可传：

```python
loss = mr_loss(enhanced, clean, lengths=None, stft_cache=stft_cache)
```

`stft_cache` 必须包含与当前 `K` 相同顺序的 `(enhanced_stft, clean_stft)` 对，几何必须匹配 `fft_sizes`/`hop_sizes`。传入 `lengths` 时，为保证 padding 不变性，当前实现会忽略缓存并在 trim-consistent 波形上重算。

## 5. `MRSpecSuppressPenaltyLoss`：多分辨率过估计惩罚

源码：[`se_train_mr.py:299`](se_train_mr.py#L299)。在每个分辨率上计算复谱幅度：

\[
 |E_{b,f,t}|=\sqrt{\operatorname{real}(E)^2+\operatorname{imag}(E)^2},
 \qquad
 |C_{b,f,t}|=\sqrt{\operatorname{real}(C)^2+\operatorname{imag}(C)^2}.
\]

惩罚只针对增强幅度超过目标的部分：

\[
 P_{b,f,t}=\max\left(0,|E_{b,f,t}|-|C_{b,f,t}|\right).
\]

无 `lengths` 时每个分辨率做完整 `[B,F,T_k]` 均值；有 `lengths` 时只对有效帧做 masked mean。最终：

\[
 L_{\mathrm{suppress}}=\frac1K\sum_{k=1}^{K}\operatorname{mean}(P^{(k)}).
\]

### 5.1 方向偏好

- **优化方向**：最小化；重点压制 `enhanced` 在任一频点/帧上的幅度过估计。
- **非对称性**：这是明确的单向惩罚。
  - `|E| > |C|`：产生正惩罚和梯度，推动增强谱幅度下降；
  - `|E| <= |C|`：该项为零，不因“欠估计”额外处罚。
- 因此它不能单独保证语音保真或避免过衰减，通常应与 `SIL1WavSeTrainLoss`、`MRSIRelCspecL2Loss` 等双向误差项组合。
- `align_se_train` 目前保留为兼容构造参数，但当前 `forward` 的 reduction 由是否提供 `lengths` 决定；有长度时总是有效帧 masked mean。该参数在 `forward` 中完全不读取，因此 `false` 在构造期 fail-fast。

该类同样接受匹配的 `stft_cache`；传入 `lengths` 时仍会为 padding 不变性重算 STFT。

## 6. `MRSIRelCspecL2SuppressPenaltyLoss`：MR 误差 + 过估计惩罚

源码：[`se_train_mr.py:387`](se_train_mr.py#L387)。它是一个直接组合类，数学定义为：

\[
 L_{\mathrm{combined}}
 =L_{\mathrm{MR\text{-}SI\text{-}Rel}}
 +\alpha_{\mathrm{penalty}}L_{\mathrm{suppress}}.
\]

其中第一项使用 `MRSIRelCspecL2Loss`，第二项使用 `MRSpecSuppressPenaltyLoss`；两者采用同一套 `fft_sizes`/`hop_sizes`。`alpha_penalty` 只缩放抑制项，不会重新归一化两项或自动归一化权重。

- `lengths is not None`：两条路径各自在 trim-consistent 波形上计算，不使用外部共享缓存；
- `lengths is None`：组合类内部先计算每个分辨率的一对 STFT，再将该缓存交给 MR 误差项和抑制项复用；
- 构造器没有公开 `stft_cache` 参数，外部需要共享缓存时应使用两个独立组件放入 `CompositeLoss`，而不是把缓存传给该组合类。

它的方向偏好是两部分的叠加：复谱平方误差负责双向接近 clean，suppress 项额外惩罚增强谱幅度超过 clean 的情况。由于 suppress 项非对称，增大 `alpha_penalty` 会更强地抑制过估计，但也可能使模型偏向保守输出；实际取值必须结合验证集观察，不能将其理解为论文固定权重。

## 6.5 `MultiBandRelativeContrastiveLoss`：noisy-anchor 感知辅助项

源码：[`mbrcl.py`](mbrcl.py)；论文精读与复现边界：[`03_MBRCL.md`](../../docs/paper/loss/Concordia/03_MBRCL.md)。该类接收 noisy/clean/enhanced 三路波形，先以模型顶层有效 STFT 几何计算功率谱，再按实际 SR 建立互不重叠的 Mel hard bands。每个频带比较 enhanced-clean 与 noisy-clean 的跨帧平均 log-energy 距离：

\[
 L_{\mathrm{band}}=\sum_b \operatorname{softplus}
 (d_{\mathrm{enh},b}-d_{\mathrm{noisy},b}+m),
\]

并以 clean/enhanced 频带幅度包络的一阶时间差分构成 `L_env`。类返回 `band_weight * L_band + envelope_weight * L_env`，默认分别为 `0.5` 与 `0.2`；SI-SNR、MR-STFT 不在类内部，须由 `CompositeLoss` 另行组合。

该实现与论文公式 (2)–(10) 的主要代数逐项对应：带内功率和对应 `E_X`，两个跨帧平均 log-energy 距离对应 `d_enh` / `d_noisy`，band reduction 对应默认 `w_b=1`，包络只使用 clean/enhanced。项目用单一 `eps` 同时实现论文默认均为 `1e-8` 的 log-energy `ε` 和 envelope `ε_e`；默认值对齐，但接口不支持两者独立配置。作者未公开代码，因此 Mel 边界取整、batch reduction、padding/`center` 等实现细节不能宣称逐字复现。

论文总目标中的 `L_SI` 与 `L_spec` 仅分别引用 negative SI-SDR [7] 和 MRSTFT [24]，不是 MBRCL 论文新定义的公开公式。项目 `SISNRLoss` 含静音 target gate 与 80 dB 上限；项目 `MultiResSTFTLoss` 对 raw target 谱范数大于 `1e-8` 的样本逐尺度计算 SC + log-magnitude 后取算术平均；target 谱严格为零时 SC 无定义、近零时分母病态，因此阈值分支令 SC 返回 graph-connected zero，仅保留一次 floored log-magnitude residual suppression。该 pure 路径不是 [24] 定义或验证的场景，但不再以其他分量替代 SC。论文 16 kHz 实验的 MR-STFT sample tuples 为 `(n_fft, win_length, hop_length)=(512,320,160)/(1024,640,320)/(2048,1280,640)`，对应 window/hop 为 `20/10 ms`、`40/20 ms`、`80/40 ms`；这套 loss schedule 不等于论文 BLSTM/CNN-U-Net 模型输入采用的 `32 ms Hamming window / 16 ms hop`。当前 DBGNet 32 kHz 生产配置把 MR-STFT 三项 sample counts 全部乘 2，使用 `(1024,640,320)/(2048,1280,640)/(4096,2560,1280)`，只是在 MR-STFT 内保持 FFT/window/hop 的物理时长；其顶层及 MBRCL 分析使用 `(640,640,320)` Hann，即 `20 ms FFT / 20 ms window / 10 ms hop`，并未对齐论文的 `32 ms Hamming / 16 ms` 模型输入几何。故完整 Composite 只能表述为“outer 权重及 MR-STFT 等物理时长 schedule 对齐、基础项工程近邻”，不能表述为整套 objective 的严格数值复现。

当 `clean=0`、`noisy` 非零时，两个公式仍然成立：`d_enh` 与 `d_noisy` 都以静音为参考，因此 `L_band` 对非零 enhanced residual 通常提供相对 noisy/input 的频带抑制梯度；clean 包络差分为零，因此 `L_env` 约束 residual envelope 的时间变化。`L_env` 不约束常量残余的绝对能量，单独不能保证输出趋零。若 `enhanced` 恰为零，功率谱原点导数为零，`L_band` 对 `enhanced` 也没有梯度，该点是 stationary point。组合时 SI-SNR 只对 non-pure 子集计算，MBRCL 和 MR-STFT 对 full batch 计算。

当 `enhanced=clean=noisy=0` 时，`d_enh=d_noisy=0`，论文 softplus 公式还产生非零常数 floor：一般式为 `band_weight * sum_b softplus(margin_b)`，统一 margin 时为 `band_weight * B * softplus(margin)`；默认值约为 `5.7887769`。该点对 `enhanced` 的梯度严格为零，不能把日志中的常数 floor 误判为模型退化。上述零 target 路径是公式一致的工程扩展，论文未单独验证 pure。

分析上下文必须满足以下维护契约：

- 顶层 `stft` 的有效 `n_fft/hop_length/win_length/window/center` 是单一事实来源；存在该上下文时，MBRCL 显式构造值只能完全匹配，冲突 fail-fast。
- 构造期 `sample_rate` 通常来自 `data.sample_rate`；上下文未知时可保留 `None`，但 forward 必须获得 runtime batch SR。当前标准 SE/AEC/TSE 与基础 GAN/PASEGAN reconstruction-loss 路径均通过 waveform capability interface 传入实际 batch SR；三任务退化输入映射见第 1.2 节，单 batch 不混 SR，跨 batch 可不同 SR。AEC/TSE 接入仅表示工程公式可计算，不代表论文已验证对应任务效果。
- 非构造 SR 的 hard-band matrix 以线程安全方式按 SR 惰性建立，canonical 存储为 CPU FP32；运行时才搬到 power tensor 的 device/dtype。构造期/runtime matrix 与 lock 均不进入 `state_dict`；pickle/deepcopy 也不保留 runtime cache/lock，加载后按需重建。
- 同设备的三路浮点波形允许 dtype 不同，合并后统一执行一次 FP32 STFT；`lengths` 与顶层有效 `center` 共同决定 frame mask。极短 full-valid、单帧 envelope 和含 pure 样本的 full-batch 路径均保持有限。
- window 仅支持 Hann/Hamming；这不表示生产统一使用 Hamming。论文的 16 kHz、512 FFT、32 ms Hamming、16 ms hop 只是复现实验覆盖，论文未说明 `center`。

## 6.6 `ComplexIdealRatioMaskLoss`：cIRM-MSE

源码：[`complex_mask.py`](complex_mask.py)。该 loss 先对 noisy reference waveform 计算混合谱 `X`，再分别用模型的 `enhanced_spectra` 和 `target_spectra` 构造压缩复数理想比率掩码，计算二者实部、虚部的 MSE。对混合谱 `X` 与目标谱 `S`，掩码为

\[
M(S,X)=K\tanh\left(\frac{c}{2}\,\operatorname{clip}_{\min=-100}
\left(\frac{S X^*}{|X|^2+\epsilon}\right)\right),
\]

其中裁剪作用于复比值的实部与虚部，默认 `K=10`、`c=0.1`。`ComplexIdealRatioMaskLoss` 比较 `M(enhanced_spectra, X)` 与 `M(target_spectra, X)`；loss 本身不对 waveform output 做 STFT，要求两路 direct spectra 为相同 `[B,F,T]` 复谱（或末轴为 real/imag 的 `[B,F,T,2]`），并使用一致的 STFT 几何。`X` 由 noisy reference waveform 按 loss 的 STFT 参数计算；多通道 noisy `[B,C,T]` 取配置的零基 `reference_channel`，mono `[B,T]` 直接使用。`lengths` 会屏蔽 padding 并按有效帧归约。该组件声明需要 noisy reference 和 direct spectra，不支持 SS undesired-energy 分支。

它用于 SE 谱进谱出模型的 cIRM MSE 监督，不代表接入 NBC/NBC2 模型。SE offline spatial 数据允许 mono clean 与 `[C,T]` noisy 配对；相应 split 必须显式声明 `spatial_enabled`、`array_profile` 或 `custom_geometry`、`reference_channel`、`channel_order`，并依据 geometry 核验音频实际通道数。`channel_order` 是磁盘排列声明，不会触发重排。cIRM 选择的 noisy 参考通道须与训练配对及模型参考通道语义一致。

### mc_sim 评估数据生成边界

[`generate_mc_sim_testset`](../../tools/prepare_data_tools/generators/se/generate_mc_sim_testset.py) 从 target 与 interference 两份标准四列 SCP 读取音频，要求两份 SCP 的 utterance ID 不交叠，并生成 `mixture.scp`、`noisy.scp`、`clean.scp`、`target_image.scp` 和 manifest。UID 不交叠只保证 utterance ID 不重复，不能证明两侧 speaker-disjoint。生成器支持注入 provider 的确定性 unit fixture；fixture 验证的是代码路径，不是使用真实 WSJ0 语料完成的评估。本地是否存在真实 WSJ0 数据不能由该生成器或 fixture 保证。

## 7. `TAPLoss`：冻结声学参数估计器辅助目标

源码：[`tap.py`](tap.py)；论文精读与实验边界：[`01_TAPLoss.md`](../../docs/paper/loss/CMU/01_TAPLoss.md)。阅读和复现时必须区分三层来源：

1. **论文层**：TAPLoss 将不可微 OpenSmile 声学描述子蒸馏为可微的逐帧 25 维参数估计器，并以其作为增强模型的辅助监督；论文主公式对应能量加权 MAE。
2. **官方实现层**：固定 16 kHz 波形、`n_fft=512`、`hop_length=160`、长度 512 的矩形窗，输入为每个频点的 real/imag 交替 514 维布局，并额外提供四种 `loss_type`。
3. **项目适配层**：保留官方 estimator 与数值定义，同时改为项目统一的 `(enhanced, clean, lengths=None)` 参数顺序，增加 FP32、变长 batch、registry、冻结状态保持、安全 checkpoint 加载和 `CompositeLoss` 语义保护。官方类名 `AcousticLoss` 只是同一实现的 registry/导入别名。

### 7.1 固定输入几何与声学估计器

TAP estimator 是 3 层双向 LSTM（输入 514、每方向 hidden size 256），再接 `512→256→25` 的两层 Linear 和中间 ReLU。对输入波形的 STFT 记为 `S∈C^{B×257×T_f}`，每帧特征按频率顺序展开为：

\[
 x_t=[\Re S_{0,t},\Im S_{0,t},\Re S_{1,t},\Im S_{1,t},\ldots,\Re S_{256,t},\Im S_{256,t}]\in\mathbb{R}^{514}.
\]

几何是 estimator 训练分布的一部分，不是模型 STFT 的可调超参数：

- `sample_rate` 必须是 `16000`；
- `n_fft` 必须是 `512`；
- `hop_length` 必须是 `160`（10 ms）；
- `win_length` 必须是 `512`（32 ms）；
- window 是显式注册的全 1 矩形窗，`center=True`、one-sided STFT；
- `build_loss` 不从顶层 `stft` 块向 `TAPLoss`/`AcousticLoss` 注入 `n_fft`、`hop_length` 或 `win_length`。

因此，即使增强模型使用 320 点 Hann STFT，也不能让 TAP 跟随它。改变任一 TAP 几何会同时改变 514 维特征的时频统计或帧率，不能解释为同一官方 estimator 的普通配置变体。

### 7.2 四种 mode 与能量定义

令冻结 estimator 的输出差为

\[
 \Delta_{b,t,p}=A(\hat{s})_{b,t,p}-A(s)_{b,t,p},\qquad p\in\{1,\ldots,25\}.
\]

项目按官方源码计算增强帧能量：

\[
 E_{b,t}=\frac{2}{512}\sum_{f=0}^{256}|\hat{S}_{b,f,t}|^2,
 \qquad q_{b,t}=\sigma(E_{b,t}).
\]

这是官方 `2/n_fft × one-sided power sum` 的数值定义。由于 DC 和 Nyquist 端点也被乘以 2，它不是对严格 one-sided Parseval 端点逐项修正后的精确总能量；不要把它改写成模型顶层 STFT 的能量。`E≥0`，所以 `q∈[0.5,1)`：高能帧相对零能帧的最大权重倍率趋近 2，但 sigmoid 会逐渐饱和。

四种 `loss_type` 均以最小化为方向：

\[
 L_{\mathrm{l1}}=\operatorname{mean}|\Delta|,
 \qquad
 L_{\mathrm{l2}}=\operatorname{mean}\Delta^2,
\]

\[
 L_{\mathrm{weighted\text{-}l1}}=\operatorname{mean}(q|\Delta|),
 \qquad
 L_{\mathrm{weighted\text{-}l2}}=\operatorname{mean}(q\Delta^2).
\]

对应配置名依次是 `l1`、`l2`、`frame_energy_weighted_l1`、`frame_energy_weighted_l2`。weighted L2 的官方代码先乘 `sqrt(q)` 再平方，代数上就是 `qΔ²`。论文式主配方对应 `frame_energy_weighted_l1`；其余三种是官方实现提供的扩展，不应全部写成论文唯一方法。

未传 `lengths` 时 reduction 是完整 `[B,T_f,25]` mean；传入 `lengths` 时，每条波形先独立 trim 后过双向 LSTM，再在**全 batch 所有有效帧和 25 个参数上做元素加权 mean**。因此长样本按有效帧数贡献更多元素，这不是“先逐样本 mean、再对 batch mean”。STFT、特征、能量和 loss reduction 均以 FP32 计算；padding 帧不参与 loss，也不应产生梯度。

### 7.3 checkpoint 与调用接口

```python
from core.losses import TAPLoss

loss_fn = TAPLoss(
    loss_type="frame_energy_weighted_l1",
    acoustic_model_path="/absolute/path/to/TAP_estimator_model_v0.tensor_only.pt",
    sample_rate=16000,
)
loss = loss_fn(enhanced, clean, lengths=lengths)
```

`acoustic_model_path` 必须指向调用方已取得并信任、且已按下述流程转换的 checkpoint。项目不自动下载或随源码分发第三方权重；运行时加载使用 `safe_torch_load(..., weights_only=True)` 的安全路径，接受 `{"model_state_dict": state_dict}` 或直接 state dict，并以 `strict=True` 校验 estimator 全部参数。加载后参数 `requires_grad=False` 且 estimator 固定为 eval，但**不会**切断到 `enhanced` 的计算图。`mode="train"/"eval"` 只是官方调用兼容标记；项目不会采信官方 README 中“eval 自动不算梯度”的表述，需要无梯度评估时必须显式使用 `torch.no_grad()`。

官方实现固定在 commit `d1fa133d80161bb837f2874d0b18eed89d5bf554`，原始 `TAP_estimator_model_v0.pt` 的 SHA256 必须为 `b6455bba8f282207f8dba8c38ef4c0bc410f76ab8f732d1d6f2de85ee05c4b72`。该文件携带旧版 NumPy pickle 表示，在 PyTorch 2.4+、NumPy 2.x 环境中不能由项目共享 `safe_torch_load` 直接安全读取；SHA 只能确认身份，不能使不兼容的 pickle global 自动变得可加载。禁止改用 unsafe fallback。

应在训练前执行一次离线转换：

```bash
python tools/model_tools/convert_taploss_official_checkpoint.py \
    --input /trusted/path/TAP_estimator_model_v0.pt \
    --output /trusted/path/TAP_estimator_model_v0.tensor_only.pt
```

`--input` 指向官方源文件，`--output` 必须是尚不存在的文件路径。转换器只接受上述 source SHA，并绑定同一份输入 bytes 完成哈希校验与反序列化，避免校验后换文件。原始文件的 pickle globals 先做静态精确审计；实际读取仅在短生命周期子进程中使用 `weights_only=True`，临时加入旧 `numpy.core.multiarray._reconstruct/scalar` 名称、`numpy.ndarray`、`numpy.dtype`、`_codecs.encode` 及运行时 float32/float64 dtype class，父进程 allowlist 不受污染。转换器只提取 `model_state_dict`，要求 28 项均为 `str -> CPU Tensor`，并与 `_TAPAcousticEstimator` 的 key、shape、dtype 严格一致；输出经 sibling 临时文件原子发布，异常时清理，发布后再由当前 `safe_torch_load` 重新加载并 strict 验证。全流程 fail-closed，没有 unsafe fallback。

转换结果约 19.5 MB，是不含 optimizer、NumPy 对象、样例或 metadata 的 **tensor-only direct state dict**。训练 YAML 的 `acoustic_model_path` 必须填写这个转换后文件，而不是约 67 MB 的官方原始文件。原始和转换权重均为本地第三方 artifact，不得提交到仓库。

### 7.4 Composite、pure 样本与项目自研 SS 的边界

TAP estimator 只在 clean speech 上训练，TAPLoss 因而是 **speech-reference-dependent** 组件：

- `is_pure_noise`、`is_pure_echo`、`is_pure_echo_noise` 为真的样本没有有效 clean speech reference，必须从 TAPLoss 子集排除；全 pure batch 返回与 `enhanced` 计算图相连的零值。
- TAPLoss/AcousticLoss 的有效 `enable_ss_undesired_energy` 必须是 `false`。undesired/noise/echo residual 不是 clean speech，把同一个 clean-speech estimator 套到 residual 支路不具有论文或官方定义，`CompositeLoss` 会在构造期 fail-fast。
- 这不否定项目自研 SS：`SIL1WavSeTrainLoss`、MRSI 等能明确比较 desired 与 undesired reference 的组件仍可启用 SS；只是 TAP component 必须显式覆盖为 `false`。

下面是完整 Composite 结构。把 checkpoint 路径替换为已校验的本地文件后可直接交给现有配置加载器；即使顶层 SS 为 `true`，TAP component 的 override 也不可省略：

```yaml
loss:
  type: CompositeLoss
  strict: true
  enable_ss_undesired_energy: true
  components:
    - type: SIL1WavSeTrainLoss
      name: SI-L1_wav
      weight: 1.0
      enable_ss_undesired_energy: true
    - type: TAPLoss
      name: TAP-frame-energy-L1
      weight: 0.1
      enable_ss_undesired_energy: false
      args:
        loss_type: frame_energy_weighted_l1
        acoustic_model_path: /absolute/path/to/TAP_estimator_model_v0.tensor_only.pt
        sample_rate: 16000
```

TAP 的权重 `0.1` 只是配置示例，不是论文推荐常数；论文在 Demucs/FullSubNet 上的权重尺度不同，项目实验必须单独调参与报告。特别是从 noisy 输入从头训练时，官方 README 提醒早期 estimator 预测不准，辅助权重不宜过大。

## 8. `CompositeLoss`：desired/undesired 两分支组合

源码：[`composite.py`](composite.py)。普通组合的定义是：

\[
 L_{\mathrm{total}}=\sum_{i=1}^{N}w_iL_i(\hat{s},s),
\]

其中 `w_i` 是 YAML 中 component 的 `weight`，**不会按权重和归一化**。构造器接受任意可转为 `float` 的有限权重，包括负数和零，但拒绝 `NaN`/`Inf`，并要求至少一项非零；源码没有把权重限制到 `[0, 1]` 或非负区间。负权重会反转对应优化方向，通常不符合 loss 语义，应仅在调用方明确承担该风险时使用。权重总和可以不等于 `1.0`；整体放大权重会直接放大 total loss 和梯度。

标准配置结构：

```yaml
loss:
  type: CompositeLoss
  strict: false
  # 顶层开关；也可只在目标 component 上开启
  enable_ss_undesired_energy: false
  components:
    - type: SIL1WavSeTrainLoss
      weight: 1.0
      name: SI-L1_wav
    - type: MRSIRelCspecL2Loss
      weight: 1.0
      name: MR-SI-Rel
      args:
        fft_power_min: 6
        fft_power_max: 11
        alpha_rel_L2: 1.0
        use_frequency_weight: true
        align_se_train: true
    - type: MRSpecSuppressPenaltyLoss
      weight: 1.0
      name: MR-Suppress
      args:
        fft_power_min: 6
        fft_power_max: 11
        align_se_train: true
```

项目当前使用这套「SI-L1 + MR 复谱 + MR 抑制」三件套的 AEC 配置是 [`dbgnet_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml`](../../configs/aec/microsoft_challenge/dbgnet/dbgnet_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml)，其中三项权重都为 `1.0` 并开启 SS；同目录下的 `dbgnet_base_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml`、`deepvqe_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml` 使用同一组合。该组合是项目自研训练配方，不是 se_train、CAGCRN、DeepVQE 或任何论文的损失复现；归属边界见 [`docs/guides/loss/README.md`](../../docs/guides/loss/README.md)，CAGCRN 论文的对照说明见 [`docs/paper/aec/BUPT/33_CAGCRN.md`](../../docs/paper/aec/BUPT/33_CAGCRN.md)。

### 8.1 desired/undesired 能量混合（SS）

当某个 component 的有效 `enable_ss_undesired_energy` 为 `true` 且其权重非零时，该 component 按样本执行两路监督。目标支路为：

```text
(enhanced, clean)
```

undesired 支路为：

```text
(undesired_enhanced, undesired_ref)
```

逐样本目标能量权重为：

\[
 a_b=\frac{E_{\mathrm{clean},b}}
 {E_{\mathrm{clean},b}+E_{\mathrm{undesired},b}+10^{-8}},
\]

\[
 E_{\mathrm{clean},b}=\sum_{t\in V_b}s_{b,t}^2,
 \qquad
 E_{\mathrm{undesired},b}=\sum_{t\in V_b}u_{b,t}^2.
\]

未提供 `lengths` 时，能量在整条波形上求和。每个开启 SS 的 component 先分别计算两路 Loss，再混合：

\[
 L_{i,b}^{\mathrm{SS}}
 =a_bL_i(\hat{s}_b,s_b)
 +(1-a_b)L_i(\hat{u}_b,u_b),
\]

\[
 L_i^{\mathrm{SS}}=\frac1B\sum_bL_{i,b}^{\mathrm{SS}}.
\]

这里 `a_b` 不是可学习权重，也不是 batch 级单一常数；clean 能量越大，越偏向 desired 监督，undesired 参考能量越大，越保留 undesired 支路的训练信号。它是能量比例，不是 dB SNR，也没有额外的平方根或 RMS 归一化。

### 8.2 trainer 注入的两条波形

SE/AEC trainer 通常按如下线性分解注入：

- SE：`undesired_ref = noisy - clean`，`undesired_enhanced = noisy - enhanced`；
- AEC：`undesired_ref = mic - nearend`，`undesired_enhanced = mic - nearend_estimate`。
- TSE：`undesired_ref = mixture - target`，`undesired_enhanced = mixture - target_estimate`；该 complement 可能同时包含 interference、background、full/early target 差异和 observation degradation，不等价于纯非目标说话人。
- SS：当前 PIT 路径直接监督 K 路 sources，不自动构造单一 undesired residual；未来 per-source loss 若需要，应显式定义相对哪一路 source 的 complement。

`FWNegSISDRLoss` 通过 `REQUIRES_UNDESIRED_REFERENCE` 声明输入需求，单 loss 和
Composite 分量均按需接线。Composite 裁剪 pure 子集时同步裁剪 reference；若 FW-SDR
启用 desired/undesired 两路混合，undesired 支路的互补参考应使用 clean，不能继续
拿 undesired 本身当该支路的噪声参考。`AbsentTargetLoss` 在 Composite 中通过
`ddp_global_mean_mass` 按全局 absent 样本数归约，空 rank 仍按相同序列参与 collective。

如果手工调用 `CompositeLoss`，可以显式传入这两个 keyword argument。开启 SS 时若两者缺失，会直接抛出 `ValueError`；如果没有 `undesired_ref` 但传入了 `noisy`，代码会推导 `undesired_ref = noisy - clean`，但仍需要 `undesired_enhanced`。两条 undesired 波形必须分别与 `enhanced`/`clean` 形状一致。

推荐只对确实需要两路能量分解的 component 开启：

```yaml
loss:
  type: CompositeLoss
  components:
    - type: SIL1WavSeTrainLoss
      weight: 1.0
      enable_ss_undesired_energy: true
    - type: MRSIRelCspecL2Loss
      weight: 1.0
      enable_ss_undesired_energy: true
      args:
        stft_scale_factors: [0.125, 0.25, 0.5, 1, 2, 4]
```

顶层开关会作为默认值继承到省略 component 开关的分项；component 显式值优先。配置键属于 component 外层，不能放到 `args` 中。完整结构校验是 fail-fast 的，未知键不会静默忽略。`TAPLoss`/`AcousticLoss` 是明确禁用 SS 的例外：若继承或显式得到 `true`，构造直接失败；在顶层 SS 配置中必须写 component override `false`。除这类显式声明不支持 SS 的组件外，组合器不会按类名自动排除 `StateCrossEntropyLoss` 等其他组件；是否给它们开启 SS 必须由配置作者按其输入语义决定。

### 8.3 pure 分支与无语音参考

`CompositeLoss` 接受 `[B]` 的 `is_pure_noise`、`is_pure_echo`、`is_pure_echo_noise` metadata。只要三者中任一参数被传入，三掩码按位或后；组件若显式声明 `REQUIRES_SPEECH_REFERENCE`，以该 capability 为准，否则回退到类名兼容集合（当前只保留尚未声明该 capability 的旧类名）：

- `TAPLoss`（`AcousticLoss` 是同一类对象的别名，实例类名仍为 `TAPLoss`）

`PESQLoss`、`STOILoss`、`SISNRLoss`、`FWNegSISDRLoss` 均已显式声明 `REQUIRES_SPEECH_REFERENCE`，不再进入类名兼容集合。

被门控组件仅在非 pure 子集计算。该分支优先于 SS：组件即便配置了 SS，只要调用传入 pure metadata，也只计算 non-pure 的 desired 路径；TAP 在构造期已拒绝 SS。pure mask 与额外 clean-energy guard 在各 rank 本地形成有效样本 mask；DDP 下每个静态 speech-reference 分量仍在所有 rank 固定执行一次有效计数 `all_reduce`，并将局部 component mean 缩放为全局有效样本均值。即使某 rank 或全部 rank 的有效计数为零，也不改变 collective 序列，零有效样本返回 graph-connected zero。`MultiBandRelativeContrastiveLoss` 不声明 speech-reference capability，也不在兼容集合中，因此与 `MultiResSTFTLoss` 一样对 full local batch 计算，不触发这项 speech-reference 计数 collective；`SIL1WavSeTrainLoss`、`MRSIRelCspecL2Loss`、`MRSpecSuppressPenaltyLoss`、`StateCrossEntropyLoss` 同样不会因为 pure metadata 自动跳过。Composite 末尾在需要时注入与 `enhanced` 相连的零值，保证安全 backward。

### 8.4 MR-STFT 共享缓存

构造器按 YAML 顺序寻找**第一对**满足以下条件的组件：

1. 暴露相同顺序的 `fft_sizes` 与 `hop_sizes`（旧组件也可暴露等价的 power 范围）；
2. `forward` 显式接受 `stft_cache`；
3. 使用相同的精确 schedule。

仅这一对会在 `lengths is None` 且两者都未开启 SS 时，通过 [`compute_mr_stft_shared_cache`](stft_cache.py) 每个分辨率只计算一次 `(enhanced, clean)` STFT；后续即使还有相同 schedule 的组件，也不会加入该 cache pair。共享协议只适用于 Hann、`center=True`、`onesided=True`、FP32、`[B,F,T,2]` 的 canonical 布局。signature 比较 `fft_sizes`、`hop_sizes` 与顺序，不比较 `align_se_train`；后者不是 cache 兼容条件。

传入 `lengths` 时组合器主动不构建共享缓存，因为缓存若在原始 padded 波形上计算会污染边界帧；各消费方必须在 trim-consistent 波形上重算。若配对中的任一组件开启 SS，组合器不会为该 pair 构建缓存，两者都走各自计算；开启 SS 的组件再逐样本分别计算 desired/undesired 两路。TAPLoss 使用固定矩形窗与 `[B,T,514]` 实虚交替布局，不属于 MR Hann cache 协议，也不会消费该缓存。

## 9. Loss 选择与方向速查

| 目标 | 推荐项 | 它会做什么 | 它不会做什么 |
|---|---|---|---|
| 波形整体接近 clean | `SIL1WavSeTrainLoss` | 按 clean 平均绝对幅度归一化 L1 | 不区分过估计和欠估计 |
| 多尺度复谱保真 | `MRSIRelCspecL2Loss` | 复谱 real/imag 双向平方误差、relative 稳定项 | 不计算显式 SNR dB；不主动抑制过估计 |
| 抑制增强谱过冲 | `MRSpecSuppressPenaltyLoss` | 仅惩罚 `|enhanced STFT| > |clean STFT|` | 不惩罚欠估计，不能单独防止过衰减 |
| 同时保真并抑制过冲 | `MRSIRelCspecL2SuppressPenaltyLoss` | MR 双向误差 + `alpha_penalty` 倍单向惩罚 | 不会自动寻找或归一化最佳权重 |
| 逐帧声学参数接近 clean speech | `TAPLoss` | 通过冻结 estimator 约束 25 维时序声学轨迹；可强调高能帧 | 不替代主波形/频谱目标；不适用于 pure 或 undesired residual |
| 多目标/两路监督 | `CompositeLoss` + SS | 按有效长度的 clean/undesired 能量逐样本混合 | SS 权重不是可学习 SNR，也不是论文固定公式 |

重要区分：MR Loss 中没有以 `SNR(dB)` 为输入的显式 SNR 权重。`CompositeLoss` 的 desired/undesired 比例 `a_b`、MRSI 的固定频率启发式权重 `w_f`、TAPLoss 的增强帧 sigmoid 能量权重 `q_t` 是三种不同机制。若实验需要数据驱动的低 SNR 频带权重，应使用另行实现的 `FWNegSISDRLoss`，不要把它的语义归给 MRSI 或 TAPLoss。

## 10. 项目出处与论文边界

- `SIL1WavSeTrainLoss` 的源码注释将其标为 `se_train Mixed_SI_L1_Wav_*` 风格；当前仓库没有为该内部 recipe 提供一篇可核验的独立论文引用。这里仅记录实现来源标签，不扩展为论文结论。
- `MRSIRelCspecL2Loss`、`MRSpecSuppressPenaltyLoss` 与组合类的代码/注释明确称为 `se_train` 风格或等价组合，但仓库没有在这些类旁边绑定可核验的原始论文条目。本文只按当前实现给出公式。
- `TAPLoss` 的论文来源是 Zeng 等人的 ICASSP 2023 工作（arXiv:2302.08088），官方实现边界固定在仓库 commit `d1fa133d80161bb837f2874d0b18eed89d5bf554`；项目新增的安全加载、变长 reduction、pure gate、SS 拒绝和 builder 上下文校验不属于论文原始贡献。
- `CompositeLoss` 的 component 权重、SS desired/undesired 能量混合、pure metadata 门控、padding 不变性处理和 MR cache 协议是本项目的训练基础设施逻辑，不应写成某篇论文的原始方法。
- AEC 配置 [`33_CAGCRN.md`](../../docs/paper/aec/BUPT/33_CAGCRN.md) 已明确记录：项目当前 `SIL1WavSeTrainLoss + MRSIRelCspecL2Loss + MRSpecSuppressPenaltyLoss` 配方，与该论文报道的损失组合不同。因此不能用 CAGCRN 论文作为这套组合的出处，也不能把项目训练指标回填为论文指标。

论文引用请以 `docs/paper/` 中实际存在且与具体实现对应的 canonical 文档为准；若没有明确对应关系，应保留“项目自研/近邻实现，未绑定论文复现”的表述。

## 11. 相关验证

当前实现已有针对 MBRCL 公式、不同浮点 dtype 的内部 FP32、极短 full-valid parity、严格 STFT/SR matching、runtime batch SR、线程安全及 pickle-safe hard-band cache、标准 SE/AEC/TSE 与基础 GAN/PASEGAN 注入、TSE multi-target mixture 展开、零 target 的 `L_band`/`L_env` 定义、MBRCL/MR-STFT full-batch 与 SI-SNR rank-local pure gate + DDP 全局有效样本均值的分工，以及 TAP 数值契约、组合、SS、MR schedule、缓存和 padding 不变性的 unit 测试，主要位于：

- [`tests/test_mbrcl.py`](../../tests/test_mbrcl.py)
- [`tests/test_taploss.py`](../../tests/test_taploss.py)
- [`tests/test_composite_loss.py`](../../tests/test_composite_loss.py)
- [`tests/test_composite_loss_mr_cache.py`](../../tests/test_composite_loss_mr_cache.py)
- [`tests/test_loss_padding_invariance.py`](../../tests/test_loss_padding_invariance.py)
- [`tests/test_loss_build.py`](../../tests/test_loss_build.py)

建议在修改这些 Loss 后至少运行：

```bash
python -c "import core.losses; from core.utils.registry import LOSS_REGISTRY; print(len(LOSS_REGISTRY.keys()))"
python -m pytest -m unit tests/test_mbrcl.py tests/test_taploss.py tests/test_composite_loss.py tests/test_composite_loss_mr_cache.py tests/test_loss_padding_invariance.py tests/test_loss_build.py -q
```

这些本地 unit 验证只覆盖项目实现契约，不代表论文 Table I–III 已复现。

官方 checkpoint 的真实转换、普通安全加载、严格构造以及 forward/backward 由独立 external 流程覆盖。默认不联网；可指定已下载的官方源文件，或显式允许从固定 commit 下载：

```bash
TAPLOSS_OFFICIAL_CHECKPOINT=/trusted/path/TAP_estimator_model_v0.pt \
python -m pytest -m external tests/test_taploss_official_checkpoint.py -v

TAPLOSS_DOWNLOAD_OFFICIAL_CHECKPOINT=1 \
python -m pytest -m external tests/test_taploss_official_checkpoint.py -v
```

该测试必须先校验固定 SHA，再转换到 pytest 临时目录，并从转换产物走普通 `safe_torch_load` 与 TAPLoss strict 加载；默认 unit 测试不得联网。测试结束不得把约 67 MB 的官方源文件、约 19.5 MB 的转换权重或其他临时产物留在仓库；训练 checkpoint、日志和评估结果同样不得写入源码目录。
