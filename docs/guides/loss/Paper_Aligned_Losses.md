# 论文对照 Loss 指南

> **这一页回答“这个 Loss 出自哪篇论文，和论文差在哪”。** 每节一个实现：先给注册名与签名，再给源码里**实际运行**的公式，最后点明与论文或官方实现的差异。公式对得上**不等于**本项目复现了论文指标；本文只描述代码契约，不断言效果。

两个反复出现的说法：**canonical 精读**指 `docs/paper/loss/` 下与某个实现严格一一对应的论文解读页，只为那一个实现背书；**用户自研**指用户明确声明的方法或配方，不得归因于任何论文。

本文集中说明 `core.losses` 中与论文或公开作者实现有明确对应关系的训练损失。覆盖范围为：

- `GTCRNCompressedSpectralLoss`
- `GTCRNLogRatioLoss`
- `MultiBandRelativeContrastiveLoss`
- `TAPLoss` / `AcousticLoss`
- `FWNegSISDRLoss`
- `PLCPAAsymmetricLoss`
- `GAPHybridLoss`

`docs/paper/loss` 下三篇 local canonical 与对象保持严格一对一：[`01_TAPLoss.md`](../../paper/loss/CMU/01_TAPLoss.md) 覆盖同一实现的 `TAPLoss` / `AcousticLoss`，[`02_Loud-loss.md`](../../paper/loss/IMU-SpeechLab/02_Loud-loss.md) 只覆盖尚未实现的 Loud-loss，[`03_MBRCL.md`](../../paper/loss/Concordia/03_MBRCL.md) 只覆盖 `MultiBandRelativeContrastiveLoss`；三者不得为其他 Loss 背书。

自研边界只在此声明一次：`SIL1WavSeTrainLoss`、`MRSIRelCspecL2Loss`、`MRSpecSuppressPenaltyLoss`，以及“由这三项组成且三项均设置 component-level `enable_ss_undesired_energy=true` 的一份完整三项组合配方”均为用户自研，不得归因于 `se_train`、任何论文、CAGCRN、DeepVQE 或 MBRCL；后文不再重复。

本文以本地运行事实为主体：保留公式、接口、优化意义、mask/gate/reduction、梯度和 Trainer 边界；每节至多保留一句必要参考或关键本地差异。公式对应不代表本项目复现了完整训练 recipe（配方，指含数据、超参与训练策略的整套训练方案），也不代表能复现论文指标。

## 1. 阅读口径与公共约定

### 1.1 Runtime 边界

完整的注册名清单（含 `TAPLoss` / `AcousticLoss` 别名与不注册的 `CompositeLoss`）以 [Loss 文档中心](README.md#完整-registry-inventory) 为准。

本文各卡片中的“默认参数”是 Python 类默认值，不等于论文推荐值，也不覆盖 `build_loss` 从顶层 STFT 或 sample-rate 上下文注入后的有效值。Trainer 是否传入直谱、`lengths`、pure metadata、`noisy` 或 `undesired_ref`，决定对应分支能否按本文公式运行。

### 1.2 记号与 shape

- 波形：增强结果 `x`、干净参考 `s`、退化输入或不期望参考 `n`，shape 通常为 `[B, T]`。
- 有效长度：`lengths = (ℓ₁,…,ℓ_B)`，shape `[B]`。传入时仅 `t < ℓ_b` 的样点有效。
- 单边复谱：`P,S ∈ ℂ^{B×F×N}`；框架也可用 `[B,F,N,2]` 存储实部和虚部。
- 有效帧 mask：`M_{b,n}∈{0,1}`。本文写

  $$
  D_M=F\sum_{b,n}M_{b,n},
  $$

  源码中的安全分母是 `max(D_M,1)`。若没有 `lengths`，相应 Loss 通常直接对完整 Tensor 做 `mean`。

除非单独说明，返回值均为零维 Tensor，目标方向是最小化。

记号里的 `softplus(x)=ln(1+e^x)` 是平滑版 ReLU（恒为正、处处有梯度），第 3 节 MBRCL 的 band 项用它把“比 noisy 锚点改善多少”变成一个恒正的代价。

## 2. GTCRN 公式组件

GTCRN 的联合 waveform/spectrogram 目标在本地拆为两个 registry component，通常由 `CompositeLoss` 显式组合。

### 2.1 `GTCRNCompressedSpectralLoss`

**注册名与签名**

```python
GTCRNCompressedSpectralLoss(
    n_fft=512,
    hop_length=256,
    win_length=512,
    compress_factor=0.3,
    lamda_ri=30.0,
    lamda_mag=70.0,
    center=True,
    normalized=False,
    window="hann",
)

forward(
    enhanced,                  # [B, T]
    clean,                     # [B, T]
    lengths=None,              # [B] 或 None
    *,
    enhanced_spectra=None,     # complex [B,F,N] 或 real [B,F,N,2]
    target_spectra=None,       # 同上
) -> scalar
```

`enhanced_spectra` 与 `target_spectra` 必须同时提供或同时省略。无论是否提供直谱，源码都会先检查 `enhanced/clean` waveform canvas：两者 shape、device、dtype 必须一致且数值有限；`center=False` 时 canvas 总长 `T` 以及每个 `lengths[b]` 都必须 `>= n_fft`。直谱路径还要求频点数为 `n_fft // 2 + 1`、两路 shape 相同，并与波形位于同一 device；波形路径则按同一 STFT 几何从 `enhanced/clean` 计算复谱。

**当前源码公式**

令 `c=compress_factor`。源码没有可配置的 magnitude `eps`，而是固定在平方和的平方根内加入 `10⁻¹²`：

$$
|P|_\epsilon=\sqrt{\Re(P)^2+\Im(P)^2+10^{-12}},\qquad
|S|_\epsilon=\sqrt{\Re(S)^2+\Im(S)^2+10^{-12}},
$$

$$
P_c=\frac{P}{|P|_\epsilon^{1-c}},\qquad
S_c=\frac{S}{|S|_\epsilon^{1-c}}.
$$

有 `lengths` 时，三个分量都在同一个有效帧 mask 上归约：

$$
L_{RI}=\frac{\sum M(\Re P_c-\Re S_c)^2}{\max(D_M,1)}
      +\frac{\sum M(\Im P_c-\Im S_c)^2}{\max(D_M,1)},
$$

$$
L_{mag}=\frac{\sum M(|P|_\epsilon^c-|S|_\epsilon^c)^2}
                    {\max(D_M,1)},
$$

$$
L_{GTCRN\text{-}spec}=\lambda_{RI}L_{RI}+\lambda_{mag}L_{mag}.
$$

默认 `λ_RI=30`、`λ_mag=70`。无 `lengths` 时，每个实部、虚部或 magnitude Tensor 分别做全元素 mean。计算在 FP32/complex64 中进行。

**mask、gate、reduction、梯度与 Trainer 边界**

- 波形 adapter 在有 `lengths` 时先构造 trim-consistent STFT 输入。直谱虽跳过 STFT，仍先检查 waveform canvas；`center=False` 时总长 `T` 与每个 `lengths[b]` 均必须 `>= n_fft`。
- 直谱路径通过 `compute_spec_frame_lengths` 生成帧 mask。padding 谱帧不进入分子，也不产生预测梯度；分母为所有 batch 有效帧乘频点数，不是逐样本 mean 后再平均。
- 本类不声明 `REQUIRES_SPEECH_REFERENCE`，所以 `CompositeLoss` 不会因 pure metadata 排除它。`clean=0` 的行仍按同一压缩谱公式参与 full-batch reduction。
- Trainer 传直谱时，梯度直接回到 `enhanced_spectra`，waveform canvas 只参与契约校验；不传直谱时，梯度经内部 STFT 回到 `enhanced`。直谱模式不执行 iSTFT，避免 `iSTFT → STFT` 将不一致谱投影到一致谱子空间；`GTCRNLogRatioLoss` 的直谱路径则必须先 iSTFT 为波形。

**优化偏好与限制**

压缩 magnitude 降低大能量频点对梯度的支配，RI 项同时约束幅度与相位方向；`c<1` 会相对抬高低能量区域。它仍是逐 TF 点的二次误差，不直接表达时域尺度投影、主观质量或 noisy-relative 改善。`λ_RI/λ_mag` 是绝对加权，不会自动归一化。

**必要边界**：[`GTCRN` 精读](../../paper/se/Nanjing/22_GTCRN.md) 只提供方法背景；[`core/losses/hybrid.py`](../../../core/losses/hybrid.py) 的本组件以普通 Hann/Hamming、waveform/direct-spectra 双入口和变长 full-batch reduction 运行，不等同于完整 GTCRN recipe。

**本地公式回归**

- [`tests/test_gtcrn_compressed_spectral_loss.py`](../../../tests/test_gtcrn_compressed_spectral_loss.py)：短输入、padding/梯度不变性、完美重建和权重校验。
- [`tests/test_gtcrn_direct_spectrum_loss.py`](../../../tests/test_gtcrn_direct_spectrum_loss.py)：本地公式回归、直谱路由、有效帧 mask、窗与 iSTFT 边界。
- [`tests/test_loss_build.py`](../../../tests/test_loss_build.py)：顶层 STFT 注入与冲突 fail-fast。

### 2.2 `GTCRNLogRatioLoss`

**注册名与签名**

```python
GTCRNLogRatioLoss(
    eps=1e-8,
    n_fft=512,
    hop_length=256,
    win_length=512,
    center=True,
    normalized=False,
    window="hann",
)

forward(
    enhanced, clean, lengths=None, *,
    enhanced_spectra=None,
    target_spectra=None,
) -> scalar
```

两组件共有的直谱契约是：`enhanced_spectra` 与 `target_spectra` 必须成对提供；每路必须是 complex `[B,F,N]` 或 real `[B,F,N,2]`，batch size 与 waveform 一致，频点数为 `n_fft // 2 + 1`，数值有限；两路规范化后的 shape 必须相同，并与 waveform 位于同一 device。该类也会执行公共 waveform/`lengths` 校验，但与 `GTCRNCompressedSpectralLoss` 不同：direct 路径在 `center=False` 时**不要求** waveform canvas 总长 `T` 或每个 `lengths[b]` 达到 `n_fft`；若提供 `lengths`，只要求直谱包含覆盖这些有效长度所需的全部帧。该类声明 `REQUIRES_SPEECH_REFERENCE=True`。

**当前源码公式**

对每条有效波形，源码不做去均值。令预测为 `x`、参考为 `s`：

$$
E_s=\sum_t s_t^2,
\qquad
s_{target}=\frac{\sum_t s_tx_t}{E_s+\epsilon}s,
\qquad
e=x-s_{target},
$$

$$
r=\frac{\sum_t s_{target,t}^2}{\sum_t e_t^2+\epsilon},
\qquad
L_b=-\log_{10}(r+\epsilon),
\qquad \epsilon=10^{-8}\ \text{(默认)}.
$$

有效参考 gate 为

$$
g_b=\mathbf 1[E_s>\epsilon].
$$

最终只对有效样本求均值：

$$
L_{ratio}=\frac{\sum_b g_bL_b}{\max(\sum_b g_b,1)}.
$$

这不是 dB 形式：前面没有系数 `10`；也不是项目 `SISNRLoss` 的 `0.1` 倍。源码只在残差能量分母和 `log10` 输入加 `eps`，不对 ratio 做上下界 clamp。

**mask、gate、reduction、梯度与 Trainer 边界**

- 波形路径有 `lengths` 时先把无效尾部置零，再做投影；类内以有效区 `clean` 能量执行 `E_s > eps` gate。
- `CompositeLoss` 还会先按 `is_pure_noise`、`is_pure_echo`、`is_pure_echo_noise` 把本组件路由到 non-pure 子集。全 pure 或所有参考都未通过能量 gate 时，返回与预测计算图相连的零。
- 直谱路径按每条样本的有效 frame count 分组裁剪谱，以本类继承的 STFT 参数调用 `torch.istft(length=None)`；梯度经 iSTFT 回到 `enhanced_spectra`，waveform 路径则直接回到 `enhanced`。提供 `lengths` 时，重建后再施加 sample mask；最后仍是有效样本 mean，不按帧数加权。
- `center=True`、偶数 FFT 且只有一帧时，`length=None` 的重建长度为零；该组被跳过并贡献保图零，padding 区和被 gate 行不产生预测梯度。

**优化偏好与限制**

尺度投影使目标对整体增益较不敏感，推动预测中可投影到 clean 的能量相对残差能量增大。它不做 zero-mean，且 `eps` 会在极低能量区定义实际边界；pure target 没有有效训练方向，必须依赖 gate。高 ratio 端没有固定下界，只有数值 `eps` 和上游有限输入约束。

**必要边界**：[`core/losses/hybrid.py`](../../../core/losses/hybrid.py) 的本组件保留不去均值、无 `10·log10` 的投影顺序，但增加有效参考/pure gate、变长谱分组和普通 Hann/Hamming iSTFT；两个 GTCRN component 只有共享 raw pair、`lengths` 与顶层 STFT 上下文时才构成 GTCRN-style 组合。

**本地公式回归**

- [`tests/test_gtcrn_direct_spectrum_loss.py`](../../../tests/test_gtcrn_direct_spectrum_loss.py)：本地公式回归、有效样本 mean、普通窗重建、单帧零输出、NOLA 和扩展几何。
- [`tests/test_gtcrn_trainer_direct_ratio_wiring.py`](../../../tests/test_gtcrn_trainer_direct_ratio_wiring.py)：Trainer 直谱接线。
- [`tests/test_loss_audit_fixes.py`](../../../tests/test_loss_audit_fixes.py)：构造参数与数值边界回归。

## 3. `MultiBandRelativeContrastiveLoss`（MBRCL）

必要边界：[`MBRCL` canonical](../../paper/loss/Concordia/03_MBRCL.md) 只对应本实现；[`core/losses/mbrcl.py`](../../../core/losses/mbrcl.py) 用一个 `eps` 控制 log-energy/envelope，并明确 Mel 离散、动态 SR、变长与 pure full-batch 行为，AEC/TSE 可运行不代表论文验证过这些任务。

**注册名与签名**

```python
MultiBandRelativeContrastiveLoss(
    sample_rate=None,
    n_fft=None,
    hop_length=None,
    win_length=None,
    window=None,
    center=None,
    num_bands=16,
    margin=0.06,
    eps=1e-8,
    band_weight=0.5,
    envelope_weight=0.2,
)

forward(
    enhanced, clean, lengths=None, *,
    noisy=None,
    sample_rate=None,
    is_pure_noise=None,
    is_pure_echo=None,
    is_pure_echo_noise=None,
    **kwargs,
) -> scalar FP32
```

三路 `enhanced/clean/noisy` 都必须是同 shape `[B,T]`、同 device 的浮点 Tensor；`noisy` 必填。构造时 `n_fft`、`hop_length` 必填，`win_length/window/center` 省略时分别解析为 `n_fft/Hann/True`。仅支持 Hann 或 Hamming。构造期 `sample_rate` 可未知，但 forward 必须能从构造值或 runtime 参数得到正整数 SR。

**当前源码公式**

项目按实际 SR 计算单边 FFT bin 频率，在 Mel 轴上均匀取 `num_bands+1` 条边界，再用 bucketize 将每个 FFT bin 唯一分配给一个 band。记 hard-band 矩阵 `H_{k,f}∈{0,1}`；空 band 会立即报错。它是**互不重叠**的 hard partition：

$$
E_z(k,n)=\sum_f H_{k,f}|Z_{f,n}|^2,
\qquad z\in\{x,s,y\},
$$

其中 `y` 是 noisy anchor。有变长 mask 时，对每个样本、每个 band：

$$
V_b=\max\left(\sum_nM_{b,n},1\right),
$$

$$
d_x(b,k)=\frac{1}{V_b}\sum_nM_{b,n}
\left|\log(E_x(k,n)+\epsilon)-\log(E_s(k,n)+\epsilon)\right|,
$$

$$
d_y(b,k)=\frac{1}{V_b}\sum_nM_{b,n}
\left|\log(E_y(k,n)+\epsilon)-\log(E_s(k,n)+\epsilon)\right|.
$$

relative band 项对 band **求和而非求均值**：

$$
L_{band,b}=\sum_{k=1}^{K}\operatorname{softplus}
\left(d_x(b,k)-d_y(b,k)+m\right).
$$

包络定义为

$$
a_z(k,n)=\sqrt{E_z(k,n)+\epsilon},
\qquad
\Delta a_z(k,n)=a_z(k,n)-a_z(k,n-1).
$$

相邻两帧都有效时 `Q_{b,n}=M_{b,n}M_{b,n-1}`。源码的分母只计有效 frame pair 数，不再除 band 数：

$$
L_{env,b}=
\frac{\sum_{k,n\ge2}Q_{b,n}
|\Delta a_x(k,n)-\Delta a_s(k,n)|}
{\max(\sum_{n\ge2}Q_{b,n},1)}.
$$

没有有效 frame pair 时，`L_env,b` 是连接 `enhanced` 图的零。最终归约是先得到逐样本组合，再做 batch mean：

$$
L_{MBRCL}=\frac1B\sum_b
\left(\lambda_{band}L_{band,b}+\lambda_{env}L_{env,b}\right),
$$

默认 `ε=10⁻⁸`、`m=0.06`、`λ_band=0.5`、`λ_env=0.2`。

**mask、gate、reduction、梯度与 Trainer 边界**

- 三路先统一转 FP32；传入 `lengths` 时先清零无效 samples，`center=True` 时在每条有效边界 reflect 延展，再生成 frame mask；padding frame/pair 不进入分子，也不向 `enhanced` 传播梯度。
- `sample_rate` runtime 值优先于构造值。不同 SR 的连续 batch 使用按 SR 缓存的 canonical CPU FP32 hard-band matrix；单个 batch 不支持混合 SR。
- Trainer 必须传入 noisy reference 和 sample rate；梯度只需回到 `enhanced`，`noisy/clean` 是锚点。该类 `SUPPORTS_SS_UNDESIRED_ENERGY=False`。
- 本类**不**声明 speech-reference-dependent。pure metadata 在类内被丢弃，MBRCL 对 full batch 计算，包括 `clean=0` 的 target-absent 行。
- 三路全零时，默认 loss floor 为

  $$
  0.5\times16\times\operatorname{softplus}(0.06)\approx5.7887769,
  $$

  但功率谱在 `enhanced=0` 的导数为零，所以此点对增强波形没有梯度。非零 pure residual 通常仍获得相对 noisy anchor 的抑制方向。

**优化偏好与限制**

`L_band` 不是普通 clean reconstruction：它要求增强结果相对 clean 的 log-energy 距离优于原 noisy anchor，并至少留出 margin。`L_env` 补偿跨帧平均距离可能忽略的时间调制和平滑问题。hard bands 不建模跨带相互作用或完整响度非线性；精确零输出是功率表示的 stationary point；MBRCL 本身也不含论文联合目标中的 SI-SDR/MR-STFT 基础项。

**本地公式回归**

- [`tests/test_mbrcl.py`](../../../tests/test_mbrcl.py)：本地公式回归、band sum、包络归约、动态 SR、空 band、pure 边界、FP32 与 padding 不变性。
- [`tests/test_loss_documentation_contract.py`](../../../tests/test_loss_documentation_contract.py)：生产配置的算法权重和 STFT/SR 上下文。
- [`tests/test_composite_loss.py`](../../../tests/test_composite_loss.py)：组合与 pure 路由边界。

## 4. `TAPLoss` / `AcousticLoss`

必要边界：[`TAPLoss` canonical](../../paper/loss/CMU/01_TAPLoss.md) 同时覆盖 `TAPLoss` / `AcousticLoss` 这两个名称下的同一实现；[`core/losses/tap.py`](../../../core/losses/tap.py) 采用 `2/512` 能量、四种 mode、严格冻结与变长/pure/SS 保护，默认 `l1` 不等于 canonical 主式 `frame_energy_weighted_l1`。

**注册名、别名与签名**

```python
TAPLoss(
    loss_type="l1",
    acoustic_model_path=None,  # 必填
    device="cpu",
    sample_rate=16000,
    n_fft=512,
    hop_length=160,
    win_length=None,           # None -> 512
)

forward(enhanced, clean, lengths=None, *, mode="train") -> scalar FP32
```

`AcousticLoss = TAPLoss`，两个 registry 名称构造的是同一个类；它们不是两个不同公式。项目参数顺序是 `(enhanced, clean, lengths)`，与官方 `AcousticLoss(clean_waveform, enhan_waveform, mode)` 相反。

**固定 estimator 与输入 shape**

TAP estimator 是 3 层双向 LSTM：`514 → 2×256 hidden → Linear(512,256) → ReLU → Linear(256,25)`。其输入不是模型顶层谱，而是项目内部固定计算的：

- `sample_rate=16000`
- `n_fft=512`
- `hop_length=160`
- `win_length=512`
- 全 1 矩形窗、`center=True`、单边复谱
- 每个频点实虚交替布局 `[real₀, imag₀, real₁, imag₁, …]`，shape `[B,N,514]`

偏离这些固定值会在构造期报错，且 builder 不向它注入模型顶层 STFT。

**当前源码公式**

令冻结 estimator 输出 `A_x,A_s∈ℝ^{B×N×25}`，差值 `Δ=A_x-A_s`。canonical 记录的论文式 (6) 使用增强帧单边功率均值：

$$
\omega_n^{paper}=\frac1{257}\sum_{f=0}^{256}|X_{f,n}|^2.
$$

本地为保持 estimator checkpoint 的数值口径实际使用：

$$
\omega_n^{impl}=\frac2{512}\sum_{f=0}^{256}|X_{f,n}|^2.
$$

它对 DC/Nyquist 也乘 2，不是严格 one-sided Parseval 端点修正。canonical 主公式对应本地 `frame_energy_weighted_l1`：

$$
L_{TAP,paper}=\operatorname{mean}_{n,p}
\left[\sigma(\omega_n)|\Delta_{n,p}|\right].
$$

本地提供四种 `loss_type`：

$$
L_{l1}=\operatorname{mean}|\Delta|,
\qquad
L_{l2}=\operatorname{mean}\Delta^2,
$$

$$
L_{weighted\text{-}l1}=\operatorname{mean}
\left[\sigma(\omega)|\Delta|\right],
$$

$$
L_{weighted\text{-}l2}=\operatorname{mean}
\left[(\sqrt{\sigma(\omega)}\Delta)^2\right]
=\operatorname{mean}\left[\sigma(\omega)\Delta^2\right].
$$

由于 `ω≥0`，权重位于 `[0.5,1)`。这里没有额外 `eps` 或 clamp；唯一 gate 是输入长度、pure metadata 和有效帧归约。

**mask、gate、reduction、梯度与 Trainer 边界**

- 没有 `lengths` 时对完整 `[B,N,25]` 做 mean。
- 有 `lengths` 时每条波形先独立裁到 `ℓ_b` 后再过 STFT 和双向 LSTM，避免 padding 改变 backward hidden state。之后 pad estimator 输出，仅对所有样本的有效帧和 25 维参数做一个全局 mean：

  $$
  D=25\sum_bN_b,\qquad
  L=\frac{\sum_{b,n,p}M_{b,n}\,q_{b,n,p}}
          {\max(D,1)}.
  $$

  因而帧数更多的样本自然占更大权重，不是逐样本 mean 后平均。
- 每条有效波形长度必须严格大于 `n_fft/2=256`，以满足 centered reflect padding。
- estimator checkpoint 通过 `safe_torch_load` 读取，state dict 必须严格匹配；参数显式冻结并始终保持 eval，但 enhanced 路径不使用 `no_grad`，梯度仍能回到增强模型。
- `mode` 只接受 `"train"/"eval"`，不改变 autograd。`.eval()` 也不等于关闭梯度。
- TAP 是 speech-reference-dependent。`CompositeLoss` 会排除 pure-noise/pure-echo/pure-echo-noise 行；全 pure 返回保图零。`SUPPORTS_SS_UNDESIRED_ENERGY=False`，不能把 clean-speech estimator 用到 noise/echo residual 支路。

**优化偏好与限制**

TAPLoss 约束冻结代理网络所表达的 25 维逐帧声学轨迹，适合作为主波形/谱 Loss 的辅助项。能量权重只提供小于 2 倍的相对强调，并会在高能区饱和。目标质量受 estimator 的训练域、OpenSmile 标签 schema 和标准化统计限制；它不是直接的主观听感度量，也不适用于没有 clean speech reference 的 residual。

**本地公式回归**

- [`tests/test_taploss.py`](../../../tests/test_taploss.py)：双注册名、参数顺序、四种本地公式回归、固定 STFT、变长归约、pure 与 SS。
- [`tests/test_taploss_checkpoint_conversion.py`](../../../tests/test_taploss_checkpoint_conversion.py)：checkpoint 的 fail-closed 转换契约。

## 5. `FWNegSISDRLoss`

必要边界：Monir 等的 [frequency-weighted loss 论文](https://arxiv.org/abs/2506.18714) 只提供方法背景；[`core/losses/fw_sdr.py`](../../../core/losses/fw_sdr.py) 使用正值 softplus 权重、低端近似切线、真实 `undesired_ref`、clean gate 与变长 mask，运行时只按下述本地公式解释。

**注册名与签名**

```python
FWNegSISDRLoss(
    n_fft=320,
    hop_length=160,
    sr=16000,
    n_bins=18,
    scale="mel",
    weights_type="log_snr_speech_norm",
    gamma=0.2,
    eps=1e-8,
    sdr_min_db=-10.0,
    sdr_max_db=35.0,
)

forward(
    enhanced, clean, lengths=None, *,
    undesired_ref=None,
    **kwargs,
) -> scalar
```

`undesired_ref` 必须先通过 Tensor、shape、device 和浮点 dtype 校验；缺失时不允许从模型输出推导替代参考。`n_bins` 当前只接受 18；`scale` 为 `mel` 或 `critical`。

**投影与频带 SDR**

当前类固定调用 `zero_mean=True` 的内部计算。传入 `lengths` 时，`enhanced/clean/undesired_ref` 都只在有效 samples 上减均值并再次乘 sample mask。令去均值后的信号仍记为 `x,s,n`：

$$
s_{proj}=\frac{\langle x,s\rangle}{\|s\|_2^2+\epsilon}s,
\qquad e=x-s_{proj}.
$$

`Φ(·)` 将波形变为 `[B,18,N]` 的非负频带 magnitude：

- `scale="mel"`：Hann、`n_fft` 窗、`center=True` 的单边 STFT magnitude，再经 torchaudio HTK MelScale；
- `scale="critical"`：同类 centered Hann STFT，丢弃 Nyquist，再乘 18 个固定中心频率/带宽的高斯 critical-band filters。源码将这组 18 带参数表标为 16 kHz；当前构造器对非 16 kHz 不拒绝，但该用法尚未验证。

有 `lengths` 时，STFT 前在有效边界 reflect 延展，频带功率统计和最终加权和都乘有效 frame mask。

**频率权重**

设目标语音与不期望参考的频带 magnitude 分别为 `S_{k,n}=Φ(s)`、`N_{k,n}=Φ(n)`。数据驱动模式先计算有效帧平均功率：

$$
P_s(k)=\operatorname{mean}_{M}S_{k,n}^2,
\qquad
P_n(k)=\operatorname{mean}_{M}N_{k,n}^2,
\qquad
\operatorname{SNR}_k=\frac{P_s(k)}{P_n(k)+\epsilon}.
$$

五种权重是：

$$
W^{logsnr}_{k,n}=\operatorname{softplus}
\left[-\log_{10}(\operatorname{SNR}_k+\epsilon)\right]
|S_{k,n}|^\gamma,
$$

$$
W^{softmax}_{k,n}=\operatorname{softmax}_k
\left(\frac1{\operatorname{SNR}_k+\epsilon}\right)|S_{k,n}|^\gamma,
$$

$$
W^{speech}_{k,n}=|S_{k,n}|^\gamma,
\qquad
W^{diff}_{k,n}=\frac1{|S_{k,n}-N_{k,n}|+\epsilon},
$$

以及 `critical` 18 带的固定 `perceptual` 权重表。`perceptual` 模式仍可与构造的 `scale` 组合，但固定表长度为 18；其他模式的权重来自当前样本。padding 帧上的权重最终被置零。

令 `A=Φ(s_proj)`、`E=Φ(e)`：

$$
R=\frac{\sum_{k,n}W_{k,n}A_{k,n}^2}
        {\sum_{k,n}W_{k,n}E_{k,n}^2+\epsilon},
\qquad
d(R)=10\log_{10}(R+\epsilon).
$$

低端不是硬 clamp。实现按未加 `eps` 的理想曲线 `d₀(R)=10log₁₀R` 在 `R=r₀` 处构造近似切线：

$$
r_0=10^{d_{min}/10},\qquad
c=d_0'(r_0)=\frac{10}{\ln(10)r_0},
$$

$$
\widetilde d(R)=
\begin{cases}
 10\log_{10}(R+\epsilon),&R\ge r_0,\\
 d_{min}+c(R-r_0),&R<r_0,
\end{cases}
\qquad
\bar d(R)=\min(\widetilde d(R),d_{max}).
$$

实际对数支路是 `d(R)=10log₁₀(R+eps)`，并非 `d₀(R)`：在拐点右侧的值为 `d_min+10log₁₀(1+eps/r₀)`，斜率为 `10/[ln(10)(r₀+eps)]`，而左侧切线的值和斜率分别是 `d_min` 与 `10/[ln(10)r₀]`。因此这里只是按理想曲线拼接的近似切线；值差为 `O(eps/r₀)`，斜率差为相对 `O(eps/r₀)`（固定默认 `r₀` 时均为 `eps` 量级），不能称为严格 `C¹`。默认 `eps=10⁻⁸`、`r₀=0.1` 时偏差很小但不为零。

最终：

$$
L_{FW}=-\operatorname{mean}_{valid\ samples}\bar d(R),
$$

默认 `d_min=-10 dB`、`d_max=35 dB`。

**mask、gate、reduction、梯度与 Trainer 边界**

有 `lengths` 时 sample/frame mask 共同排除 padding。类内按每条样本有效区的 `mean(clean²)>eps` gate；`CompositeLoss` 还先排除 pure metadata 行。混合 batch 只对有效行做样本 mean；全无效时返回 `(enhanced * 0).sum()` 保留计算图。Trainer 必须传入真实 `undesired_ref`，不能从预测或 clean 推导替代项。

**优化偏好与限制**

默认权重同时偏向低 SNR 频带和仍有目标语音能量的 TF 区域，目的是避免模型丢掉低能量但音素关键的频段。上界 hard clamp 让容易样本饱和；低端切线只消除 hard clamp 对 ratio `R` 的零斜率：仅当上游 `dR/dx != 0` 时才向增强波形提供有界梯度，投影严格为零等驻点仍可能是零梯度。该目标依赖真实 `undesired_ref`，不能只凭 clean/enhanced 二元组完整定义。

**本地公式回归**

- [`tests/test_loss_audit_fixes.py`](../../../tests/test_loss_audit_fixes.py)：低端切线、可配置上下界和高端 clamp。
- [`tests/test_loss_padding_invariance.py`](../../../tests/test_loss_padding_invariance.py)：Mel/critical 两路的长度与 padding 语义。
- [`tests/test_loss_graph_preserving_zeros.py`](../../../tests/test_loss_graph_preserving_zeros.py)：静音 gate 的保图零。
- [`tests/test_composite_loss_kwargs.py`](../../../tests/test_composite_loss_kwargs.py)：`undesired_ref` 接线和零权重行为。

## 6. `PLCPAAsymmetricLoss`

必要边界：Eskimez 等的 [`PLCPA-ASYM`](https://arxiv.org/abs/2110.09625) 只提供方法背景；[`core/losses/plcpa_asymmetric.py`](../../../core/losses/plcpa_asymmetric.py) 默认使用 `320/160/320` 普通 Hann、`eps=1e-8` 的 `abs().clamp`、不去均值、`scale_asym=1`，且三项均按有效 TF 点 mean。

**注册名与签名**

```python
PLCPAAsymmetricLoss(
    n_fft=320,
    hop_length=160,
    win_length=320,
    compress_factor=0.3,
    scale_mag=1.0,
    scale_phase=1.0,
    scale_asym=1.0,
    window="hann",
    eps=1e-8,
    center=False,
)

forward(enhanced, clean, lengths=None, **kwargs) -> scalar
```

波形必须为同 shape `[B,T]`。只支持 Hann/Hamming；`p=compress_factor∈(0,1]`，三个 scale 非负。

**当前源码公式**

对 STFT 结果 `X,S`，先执行 magnitude 下界 clamp：

$$
A_x=\max(|X|,\epsilon),\qquad A_s=\max(|S|,\epsilon),
$$

$$
X_c=\frac{X}{A_x^{1-p}},\qquad
S_c=\frac{S}{A_s^{1-p}}.
$$

三个逐点误差为：

$$
Q_{mag}=(A_s^p-A_x^p)^2,
\qquad
Q_{phase}=|S_c-X_c|^2,
$$

$$
Q_{asym}=\operatorname{ReLU}(A_s^p-A_x^p)^2.
$$

有有效帧 mask 时，每项均使用同一个 `D_M`：

$$
L_j=\frac{\sum M Q_j}{\max(D_M,1)},
\qquad j\in\{mag,phase,asym\}.
$$

无 `lengths` 时三个分量分别做全元素 mean。最终：

$$
L_{PLCPA\text{-}ASYM}
=\lambda_{mag}L_{mag}+\lambda_{phase}L_{phase}
+\lambda_{asym}L_{asym}.
$$

`scale_asym=0` 时源码完全跳过 asym 分支。默认三个 scale 都是 1。

**mask、gate、reduction、梯度与 Trainer 边界**

- 计算显式转 FP32。传入 `lengths` 时先把无效 waveform tail 清零。
- `center=True` 时使用共享的 length-aware centered-STFT 输入准备和帧 mask，可处理极短输入并保持 trim 等价。
- `center=False` 时原始总长以及每个有效长度都必须 `>= n_fft`；帧数由 `compute_spec_frame_lengths` 计算，padding 帧不参与三个 mean，也不向 `enhanced` 传播梯度。
- Trainer 只需提供 waveform 与可选 `lengths`；梯度经 STFT 回到 `enhanced`。本类没有 speech-reference gate、pure gate、ratio clamp 或 SS 专用语义，`clean=0` 仍进入公式。

**优化偏好与限制**

magnitude 项双向逼近压缩幅度，phase-aware 项约束压缩复谱；asym 项只在 `A_s^p>A_x^p` 时生效，即额外惩罚增强结果比目标更弱的 TF 点，偏向减少过度抑制。该方向不会对幅度过冲施加同样的额外惩罚，过大的 `scale_asym` 可能推动保守不足的降噪。

**本地公式回归**

- [`tests/test_plcpa_asymmetric_centered_stft.py`](../../../tests/test_plcpa_asymmetric_centered_stft.py)：三项本地公式回归、centered/non-centered 与 padding 梯度。
- [`tests/test_loss_audit_fixes.py`](../../../tests/test_loss_audit_fixes.py)：复数 phase 计算和有限梯度 smoke。
- [`tests/test_core_losses_audit_regressions.py`](../../../tests/test_core_losses_audit_regressions.py)：non-centered 有效长度与 tail mask。

## 7. `GAPHybridLoss`

必要边界：[`GAP-URGENet` 精读](../../paper/se/Nanjing/GAP_URGENet.md) 只确认 Predictor 使用 STFT-domain loss；[`core/losses/pase_family.py`](../../../core/losses/pase_family.py) 将 SR 固定在构造期，并增加统一波形接口、有效 frame/sample mask 与短输入处理，`30/70/1` 和 signed 公式不反写成论文正文公式。

**注册名与签名**

```python
GAPHybridLoss(
    sample_rate=16000,
    fft_len=0.032,
    hop_len=0.016,
    win_len=0.032,
    compress_factor=0.3,
    eps=1e-12,
    lamda_ri=30,
    lamda_mag=70,
    lamda_sdr=1,
)

forward(y_pred, y_true, lengths=None, **kwargs) -> scalar
```

预期输入为 `[B,T]`，也兼容单通道 `[B,1,T]` 并 squeeze channel。有效 STFT samples 由秒数截断转换：

$$
n_{fft}=\operatorname{int}(fft\_len\cdot fs),\quad
n_{hop}=\operatorname{int}(hop\_len\cdot fs),\quad
n_{win}=\operatorname{int}(win\_len\cdot fs).
$$

默认 16 kHz 时为 `512/256/512`，使用 Hann、`center=True`。

**当前源码公式**

令 `P=STFT(y_pred)`、`S=STFT(y_true)`，`c=compress_factor`。这里与 GTCRN 固定 `sqrt(square sum + 1e-12)` 不同，GAP 使用 magnitude hard clamp：

$$
A_p=\max(|P|,\epsilon),\qquad A_s=\max(|S|,\epsilon),
$$

$$
P_c=\frac{P}{A_p^{1-c}},\qquad S_c=\frac{S}{A_s^{1-c}}.
$$

谱项在有效帧和全部频点上归约：

$$
L_{RI}=\frac{\sum M(\Re P_c-\Re S_c)^2}{\max(D_M,1)}
      +\frac{\sum M(\Im P_c-\Im S_c)^2}{\max(D_M,1)},
$$

$$
L_{mag}=\frac{\sum M(A_p^c-A_s^c)^2}{\max(D_M,1)}.
$$

时域项按样本先求 norm，再做 batch mean：

$$
L_{signed}=\operatorname{mean}_b\left[
-2\log_{10}\left(
\frac{\|y_{pred,b}\odot m_b\|_2}
{\max(\|(y_{pred,b}-y_{true,b})\odot m_b\|_2,\epsilon)}
+\epsilon\right)\right].
$$

最终：

$$
L_{GAP}=\lambda_{RI}L_{RI}+\lambda_{mag}L_{mag}
+\lambda_{sdr}L_{signed}.
$$

默认 `ε=10⁻¹²`、`λ_RI/λ_mag/λ_sdr=30/70/1`。

**mask、gate、reduction、梯度与 Trainer 边界**

- `prepare_centered_stft_inputs` 对有 `lengths` 的两路波形构造 trim-consistent centered-STFT 输入，并返回 sample mask；padding frame/sample 不进入分子，也不向 `y_pred` 传播梯度。
- 谱分母是整个 batch 的有效帧数乘频点数；时域 signed 项是逐样本值的普通 batch mean。二者的样本加权口径不同，梯度共同回到 waveform `y_pred`。
- Trainer 只需传 `y_pred/y_true/lengths`；本类没有 clean-energy/pure gate、zero-mean、尺度投影或 SDR 上下界 clamp。magnitude 与误差 norm 的 clamp 在阈值以下形成各自的局部梯度边界。
- 所谓 signed SDR 不是 SI-SDR：它没有最优缩放、没有去均值，系数是 `-2` 而不是 `-10`。

**优化偏好与限制**

RI/magnitude 两项提供压缩谱保真；signed 项奖励预测 norm 相对误差 norm 增大。由于分子是 `||prediction||` 而不是 clean projection，整体增益和输出能量会直接影响该项。它不应被解释为标准 SI-SNR、SI-SDR 或普通负 SDR。

**本地公式回归**

- [`tests/test_gap_hybrid_loss.py:81-131`](../../../tests/test_gap_hybrid_loss.py#L81-L131)：`GAPHybridLoss` 本地公式回归、短输入、length mask 和梯度不变性。
- [`tests/test_gap_urgenet_predictor_smoke.py`](../../../tests/test_gap_urgenet_predictor_smoke.py)：tiny Predictor Trainer 接线。

## 8. Loud-loss 状态卡

- **状态**：当前 `core.losses` 未注册、未实现 `LoudLoss`。
- **边界**：[`Loud-loss` canonical](../../paper/loss/IMU-SpeechLab/02_Loud-loss.md) 描述的是待实现方法；其核心是 50% overlap Mel 子带上的 log-power MSE 与 40-phon equal-loudness 加权。
- **禁止等价替代**：现有 log-spectral、Mel、MBRCL、frequency-weighted SDR、GTCRN compressed spectral 或 PLCPA 项都只是近邻，不能标记为 Loud-loss。
- **进入主篇条件**：只有新增 registry 实现，并补齐输入输出、`eps`/离散 bin、mask、reduction、梯度与 Trainer 接线的本地公式回归后，才按已实现 Loss 展开。

## 9. Trainer 使用检查

1. 组合前确认 component 所需输入：GTCRN 可接 waveform/direct spectra；TAP 需要可信 estimator checkpoint；FW 必须有真实 `undesired_ref`；MBRCL 必须有原始 `noisy` anchor 与 runtime SR。
2. 始终核对有效 STFT、sample rate、`lengths`、pure metadata、component-level SS 开关及内外两层权重；`CompositeLoss` 不自动归一化 component 权重。
3. 全 pure、静音参考、短帧和 padding 分支必须保留与预测相连的零或本文定义的有效梯度；不能只检查标量有限。
4. 本地公式回归证明的是源码契约与 Trainer 接线，不替代完整训练实验；自研归属以本文开头的唯一声明为准。
