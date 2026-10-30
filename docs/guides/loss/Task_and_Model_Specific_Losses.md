# 任务与模型特定 Loss 实现参考

> **这一页收哪些 Loss**：通用波形/频谱目标之外，只服务特定任务或特定模型的那些。它们要么 `forward` 签名特殊（不吃标准 `(enhanced, clean, lengths)`），要么必须由专用 Trainer 驱动，要么带任务专属的 mask / gate / 归一化语义。YAML 里出现 `MRSIRelCspecL2Loss`、`PaseMultiScaleMelLoss`、`StateCrossEntropyLoss`、`DeepFilterNet*`、`MPSEnetCompositeLoss` 之类名字时，来这里查真实输入输出与运行边界。

先认识几个重复出现的说法：

- **`lengths`**：一批波形长度不齐时，用 `[B]` 整数 Tensor 标出每条的真实样本数；padding 不应进入 loss 和梯度。
- **masked mean（掩码平均）**：只在有效样点 / 有效帧上求平均，分母是广播后的有效元素数，不是固定的 batch 大小。
- **图保留零（graph-connected zero）**：数值为 0、但仍接在某个预测张量计算图上的标量；能反传但梯度为零，用来避免多卡训练里整段分支被跳过。
- **stft_cache**：把多分辨率 STFT 结果在同一次前向里复用给多个子项，只省计算量、不改公式。
- **CE（cross entropy，交叉熵）**：分类常用目标；本页的 state / speaker CE 是**辅助项**，不直接优化增强波形。
- **LSNR（local SNR，局部信噪比）**：逐帧估的“目标语音能量 / 噪声能量”比，DeepFilterNet 用它控制滤波器强度与 alpha 开关。
- **iSTFT consistency**：约束“模型预测谱”与“该谱重建波形后重新分析得到的谱”一致，检查频谱—波形往返是否自洽。

本文记录 `core.losses` 中不适合只用一句话概括的任务专用、模型专用及非标准签名 loss。重点是当前运行公式、函数原理、优化意义、输入输出，以及 mask、gate、reduction、梯度和 Trainer 调用边界；简要参考只用于划清实现归属，不替代源码契约。

## 1. 范围、注册事实与记号

完整的注册名清单（含 `TAPLoss` / `AcousticLoss` 别名与不注册的 `CompositeLoss`）以 [Loss 文档中心](README.md#完整-registry-inventory) 为准。

本文覆盖以下注册名，均无额外 registry 别名：

- `MRSIRelCspecL2Loss`、`MRSpecSuppressPenaltyLoss`、`MRSIRelCspecL2SuppressPenaltyLoss`；
- `PaseMultiScaleMelLoss`、`GAPPostNetAwareLoss`、`MSELoss`、`ScalarPassthroughLoss`；
- `StateCrossEntropyLoss`、`SpeakerCrossEntropyLoss`；
- `MPSEnetCompositeLoss`；
- `DeepFilterNetMultiResSpecLoss`、`DeepFilterNetOverAttenuationLoss`、`DeepFilterNetLSNRLoss`、`DeepFilterNetAlphaLoss`、`DeepFilterNetFullLossWithLSNR`、`DeepFilterNetLSNRComponent`、`DfAlphaLossComponent`；
- `ComplexIdealRatioMaskLoss`。

通用记号如下：

- `B`：batch size；`T`：波形样本数；`F`：单边频率 bin 数；`N`：谱帧数；`C`：类别数或波形通道数，按上下文区分。
- `enhanced`、`clean` 通常是 `[B,T]` 波形；`lengths[b]` 是第 `b` 条波形的有效样本数。
- `M[b,n]∈{0,1}` 是有效帧 mask；广播到频率、通道或实虚维后，分母按**广播后的有效元素数**计算。
- `clamp_min(x,a)=max(x,a)`；所有“图保留零”均表示数值为 0、仍连接指定预测张量的 autograd 图。

配置结构、outer component 权重和 `strict` wrapper 见 [Loss 配置指南](Loss_Configuration_Guide.md)。本文末尾只保留必要的实现验证索引与 `docs/paper/loss` canonical 边界。

---

## 2. 用户自研多分辨率复谱与抑制项

本节归属严格如下：`SIL1WavSeTrainLoss`、`MRSIRelCspecL2Loss`、`MRSpecSuppressPenaltyLoss` 均为**用户自研**；把这三项组成一份 `CompositeLoss` 配方，并给三项都设置 component-level `enable_ss_undesired_energy=true`，所得**完整三项组合配方也为用户自研**。它们及该配方均不得归因于 `se_train`、任何论文、CAGCRN、DeepVQE 或 MBRCL；`se_train` / `align_se_train` 只是历史兼容标识。本文只展开两项 MR 公式，波形项公式见 [波形与频谱 Loss](Waveform_and_Spectral_Losses.md#21-sil1wavsetrainloss)。

三项 MR 实现均位于 [`core/losses/se_train_mr.py`](../../../core/losses/se_train_mr.py)，内部 STFT 强制使用 FP32、单边谱、`center=True`、长度等于 `n_fft` 的 Hann 窗，并返回 `[B,F,N,2]` 实虚张量。

### 2.1 共同 STFT schedule 与有效帧语义

两种 schedule 互斥：

1. **Legacy power 模式**：未给 `stft_scale_factors` 时，
   $$
   q\in\{q_{\min},\ldots,q_{\max}\},\quad
   n_q=2^q,\quad h_q=n_q/2.
   $$
   默认 `fft_power_min=6`、`fft_power_max=11`，即 `n=[64,128,256,512,1024,2048]`、`h=[32,64,128,256,512,1024]`。此模式忽略 builder 注入的 `win_length`、`hop_length`。
2. **Frame-relative 模式**：给定严格递增的有限正数 `stft_scale_factors=(r_k)` 后，必须同时有正整数 `win_length=w` 与 `hop_length=h`，并要求
   $$
   n_k=w r_k\in\mathbb N_{>0},\qquad h_k=h r_k\in\mathbb N_{>0}.
   $$
   不做取整。此时禁止再给 `fft_power_min` 或 `fft_power_max`。

当 `lengths` 存在时，源码先把有效样本外置零，在每条样本自己的有效右边界做 reflect 延展，再重算 STFT；外部 `stft_cache` 不会用于这一分支。对第 `k` 个分辨率，有效帧数与 `torch.stft(center=True)` 对齐：

$$
N_{b,k}=\max\left(1,\ 1+\left\lfloor
\frac{L_b+2\lfloor n_k/2\rfloor-n_k}{h_k}
\right\rfloor\right),
$$

并令 $M_{bfn}=\mathbf{1}\{n<N_{b,k}\}$。因此 batch padding 不进入 loss 或梯度。`lengths=None` 时使用完整 STFT 网格，$M=1$。

### 2.2 `MRSIRelCspecL2Loss`

**注册名与签名**

```python
MRSIRelCspecL2Loss(
    fft_power_min=None,
    fft_power_max=None,
    alpha_rel_L2=1.0,
    use_frequency_weight=True,
    align_se_train=True,
    *,
    stft_scale_factors=None,
    win_length=None,
    hop_length=None,
)

forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None,
        stft_cache: list[(enh_stft, clean_stft)] | None = None) -> scalar
```

**精确公式**

在分辨率 `k` 上，令增强与目标的实虚 STFT 为 `Ŝ_k`、`S_k`：

$$
D_{bfn}^{(k)}=\sum_{r\in\{\mathrm{re},\mathrm{im}\}}
\left(\widehat S_{bfnr}^{(k)}-S_{bfnr}^{(k)}\right)^2,
\qquad
P_{bfn}^{(k)}=\sum_r\left(S_{bfnr}^{(k)}\right)^2.
$$

若 `use_frequency_weight=True`，`F_k=n_k/2+1` 个频率权重按源码的等间隔向量生成：

$$
W_f^{(k)}=2\cos\left(
\operatorname{linspace}(0,\pi/2,F_k)_f-\pi/16
\right)+0.5;
$$

否则 `W_f=1`。令 `Z_k=max(1,Σ_{bfn}M_{bfn}^{(k)})`。每条样本的 clean-power 归一化量 `A_b^(k)` 有两个分支：

- `align_se_train=True` 且 `lengths=None`：先取全频率、全帧均值 `a_b=mean_{f,n}P_bfn`，仅当 `a_b<1e-20` 时替换为 `0.5`；`[1e-20,0.5)` 内的值**不会**被 clamp。
- 其余情况，包括任何提供 `lengths` 的调用：
  $$
  A_b^{(k)}=\max\left(
  0.5,\frac{\sum_{fn}M_{bfn}^{(k)}P_{bfn}^{(k)}}
  {\max(1,\sum_{fn}M_{bfn}^{(k)})}
  \right).
  $$

单尺度项及最终结果为

$$
\begin{aligned}
L_{\mathrm{SI},k}
&=\frac1{Z_k}\sum_{bfn}
\frac{M_{bfn}^{(k)}W_f^{(k)}D_{bfn}^{(k)}}{A_b^{(k)}+10^{-12}},\\
L_{\mathrm{rel},k}
&=\frac1{Z_k}\sum_{bfn}
\frac{M_{bfn}^{(k)}W_f^{(k)}D_{bfn}^{(k)}}{P_{bfn}^{(k)}+0.05},\\
L_{\mathrm{MR}}
&=\frac1K\sum_{k=1}^{K}
\left(L_{\mathrm{SI},k}+\alpha_{\mathrm{rel}}L_{\mathrm{rel},k}\right).
\end{aligned}
$$

这里 `10^-12` 与 `0.05` 是源码固定值；`alpha_rel_L2` 默认 1。分母 `Z_k` 是整个 batch 的有效 `B×F×N` 元素数，所以不同长度样本按有效谱元素数共同归约，不是先逐样本平均再做 batch mean。

**优化偏好、限制与归属**

- SI 分支按样本 clean 平均功率归一化，降低绝对响度对误差尺度的控制；relative 分支用逐 bin clean 功率加 `0.05`，更强调 clean 低能区域的复谱残差。
- 可选频率权重并非 Mel 或 equal-loudness 权重；它由上述 cosine 公式固定生成。
- 这是用户自研的未压缩复谱实虚平方误差，不应与 MBRCL、普通 MR-STFT 或 log-magnitude loss 混称；`se_train` 只出现在历史文件名/兼容标识中，不构成来源。
- `align_se_train` 只是历史兼容参数：它只影响 `lengths=None` 的 `A_b` 低能量规则；一旦传 `lengths`，两种取值结果相同。

### 2.3 `MRSpecSuppressPenaltyLoss`

**注册名与签名**

```python
MRSpecSuppressPenaltyLoss(
    fft_power_min=None,
    fft_power_max=None,
    align_se_train=True,
    *,
    stft_scale_factors=None,
    win_length=None,
    hop_length=None,
)

forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None,
        stft_cache: list[(enh_stft, clean_stft)] | None = None) -> scalar
```

在分辨率 `k` 上先对实虚向量取欧氏范数：

$$
\widehat A_{bfn}^{(k)}=\sqrt{\widehat S_{\mathrm{re}}^2+\widehat S_{\mathrm{im}}^2},
\qquad
A_{bfn}^{(k)}=\sqrt{S_{\mathrm{re}}^2+S_{\mathrm{im}}^2},
$$

再只惩罚增强幅度超过 clean 的部分：

$$
R_{bfn}^{(k)}=\operatorname{ReLU}
\left(\widehat A_{bfn}^{(k)}-A_{bfn}^{(k)}\right),
\qquad
L_{\mathrm{sup}}=\frac1K\sum_k
\frac{\sum_{bfn}M_{bfn}^{(k)}R_{bfn}^{(k)}}
{\max(1,\sum_{bfn}M_{bfn}^{(k)})}.
$$

`lengths=None` 时 `M=1`，即每尺度直接 `penalty.mean()`。`align_se_train` 只是历史构造兼容位，当前 `forward` 完全不读取它，因此非默认值 `false` 在构造期 fail-fast；这个名字不表示该 loss 来自 `se_train`。

该项是用户自研，偏好“增强幅度不高于 clean”，用于抑制过增强或新增谱能量；它不惩罚 `|Ŝ|<|S|`，因此不能单独承担语音保真目标。类内也没有外部配方常见的 `alpha_MR` 或 `alpha_penalty`：在 `CompositeLoss` 中应由 component `weight` 明确表达完整 outer 系数。

### 2.4 `MRSIRelCspecL2SuppressPenaltyLoss`

**注册名与签名**

```python
MRSIRelCspecL2SuppressPenaltyLoss(
    fft_power_min=None,
    fft_power_max=None,
    alpha_rel_L2=1.0,
    use_frequency_weight=True,
    alpha_penalty=1.0,
    align_se_train=True,
    *,
    stft_scale_factors=None,
    win_length=None,
    hop_length=None,
)

forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar
```

组合公式严格为

$$
L=L_{\mathrm{MR}}+\alpha_{\mathrm{penalty}}L_{\mathrm{sup}},
$$

其中两个子项就是前两节的用户自研实现，`alpha_penalty` 默认 1。该类是由两项自研 MR loss 派生的 convenience class，只组合 `L_MR + α_penalty·L_sup`；它**不是**那一份完整生产 recipe。完整 recipe 是 `SIL1WavSeTrainLoss + MRSIRelCspecL2Loss + MRSpecSuppressPenaltyLoss` 三个独立 component 的一份三项组合，且三项均设置 component-level `enable_ss_undesired_energy=true`。`lengths=None` 时 convenience class 为每个分辨率只计算一次 STFT pair 并在两个子项间复用；提供 `lengths` 时，为保持每条样本右边界的 trim 等价语义，两个子项各自在内部重算。缓存策略只改变计算量，不改变公式。

完整生产配方见 [`deepvqe_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml`](../../../configs/aec/microsoft_challenge/deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml#L619-L644)，SS 的逐样本能量混合、AEC undesired reference 注入、pure gate、direct spectra 冲突与 DDP 归约边界见 [Loss 配置指南 §7.3](Loss_Configuration_Guide.md#73-ss-undesired-energy)；本文不重复整段 YAML。该生产配方及其 component-level SS 组合语义是用户自研，不得归因于 `se_train`、CAGCRN、DeepVQE、MBRCL 或其他论文。组合容器自身的限制与两个子项一致：`align_se_train` 在有 `lengths` 时不生效；类内已有 `alpha_penalty`，若再置于 `CompositeLoss`，outer `weight` 会乘整个组合结果。

---

## 3. PASE/GAP 波形重建与 trainer 兼容项

### 3.1 `PaseMultiScaleMelLoss`

**注册名、签名与默认值**

```python
PaseMultiScaleMelLoss(
    sampling_rate=16000,
    n_mels=[5,10,20,40,80,160,320],
    window_lengths=[32,64,128,256,512,1024,2048],
    clamp_eps=1e-5,
    mag_weight=0.0,
    log_weight=1.0,
    pow=1.0,
    weight=1.0,
    match_stride=False,
    mel_fmin=[0.0] * K,
    mel_fmax=[None] * K,
    window_type="hann",
)

forward(enhanced: [B,T] | [B,C,T], clean: [B,T] | [B,C,T],
        lengths: [B] | None = None) -> scalar
```

二维输入会补成 `[B,1,T]`。每个波形通道独立做 STFT；`lengths` 按通道 `repeat_interleave(C)`。四条 schedule `n_mels/window_lengths/mel_fmin/mel_fmax` 必须非空且等长；每个 `n_mels`、窗长必须是精确正整数。源码对 `mel_fmin` 只检查有限且非负，不单独要求其不超过 Nyquist；只有非 `None` 的 `mel_fmax` 才校验有限、严格大于同尺度 `mel_fmin` 且不超过 `sampling_rate/2`。直接构造 `PaseMultiScaleMelLoss` 时，`sampling_rate` 本身没有独立的类型、有限性或正值校验。

对尺度 `k`，窗长 `w_k`、hop `h_k=floor(w_k/4)`，窗由 `scipy.signal.get_window(window_type,w_k)` 生成并转为 float tensor。构造期只显式检查 `w_k>0`；实际 forward 中，`w_k>=4` **只保证** `h_k>=1`，并不保证后续 padding 对任意短输入都可执行。`match_stride` 不额外要求 `w_k` 能被 4 整除，源码断言只核对构造出的 `h_k==w_k//4`。使用 `torch.stft(n_fft=w_k, hop_length=h_k, center=True)`，令线性幅度谱为 `A_k=|STFT(x)|`。Librosa Mel 矩阵 `H_k∈R^{m_k×F_k}` 由当前采样率、`w_k`、`m_k`、`fmin_k/fmax_k` 生成：

$$
X_k[b,c,m,n]=\sum_f A_k[b,c,f,n]H_k[m,f].
$$

`match_stride=False` 是默认常规分支：波形经 `prepare_centered_stft_inputs` 准备后直接执行上述 centered STFT，并按同一 centered 网格调用 `create_frame_mask`。

`match_stride=True` 则在已准备波形上额外执行

$$
\mathrm{right\_pad}=\lceil T/h_k\rceil h_k-T,
\qquad p=\left\lfloor\frac{w_k-h_k}{2}\right\rfloor,
$$

再用 reflect 模式填充 `(p, p+right_pad)`。PyTorch reflect padding 要求左、右 pad **分别严格小于已准备波形的长度**；因此即使 `w_k>=4`、hop 已合法，短输入仍可能在这里触发 `RuntimeError`。随后源码仍以 `center=True` 做 STFT，并裁掉 `stft[..., 2:-2]`。裁剪后仍沿用普通 centered `create_frame_mask`，只把帧数参数设为裁剪后的 `nt`，没有针对 post-crop 时间网格重新推导 mask。因此该分支不能承诺 `lengths` 总能精确屏蔽全部无效帧，mixed-length 短样本与逐条 trim 计算也可能不一致；现有测试未覆盖 `match_stride=True`。

Mel log 表示不是自然对数直接求和，而是

$$
\ell(X_k)=\frac{\log\left(\operatorname{clamp}_{\min}(X_k,\epsilon)^{p}\right)}{\log 10},
\quad \epsilon=\texttt{clamp\_eps},\ p=\texttt{pow}.
$$

令有效帧 mask 为 `M_k[b,c,1,n]`。masked L1 的精确分母为

$$
\operatorname{ML1}(U,V;M_k)=
\frac{\sum_{bcmn}|U_{bcmn}-V_{bcmn}|M_{bcn}}
{\max\left(1,\ m_k\sum_{bcn}M_{bcn}\right)}.
$$

最终结果为**尺度求和而非尺度平均**：

$$
L=w\sum_{k=1}^{K}\left[
\lambda_{\log}\operatorname{ML1}(\ell(\widehat X_k),\ell(X_k);M_k)
+\lambda_{\mathrm{mag}}\operatorname{ML1}(\widehat X_k,X_k;M_k)
\right],
$$

其中 `w=weight`、`λ_log=log_weight`、`λ_mag=mag_weight`。默认只计算 log-Mel L1；生产 PASE 配方常显式将 `weight` 设为 15，但这不是类默认值。

**优化偏好、实现与 Trainer 边界**

- 多尺度短窗强调时间细节，长窗强调频率结构；Mel 投影和 log 压缩更偏向感知频率与相对幅度差。
- 当前实现沿用多尺度 Mel 重建思路，但运行语义以本节公式为准：本地增加了 `lengths` mask、按广播后有效元素归约、短输入 reflect 延展和 schedule 校验；raw-Mel 分支、默认七尺度和权重也不应套用其他实现的默认值。
- `match_stride=False` 时 batch padding 不参与 loss 或梯度；`match_stride=True` 的裁帧后 mask 与短输入限制只能按上文处理，不能承诺逐条 trim parity。
- `core.losses.pase_family` 在模块 import 期直接导入 `librosa.filters.mel`、`scipy.signal` 与 `torchaudio.functional.resample`，不是按分支延迟导入。PASE GAN 的 `feature_loss` / `discriminator_loss` / `generator_loss` 由 `PASEGANTrainer` 直接调用，不属于本 loss。
- [`tests/test_pase_multiscale_mel_loss.py`](../../../tests/test_pase_multiscale_mel_loss.py) 覆盖 `match_stride=False` 下的短输入 padding parity 和常规数值公式，不覆盖 `match_stride=True` 或所有参数失败边界；配方说明见 [PASE family 学习指南](../se/pase_family_learning_guide.md)。

### 3.2 `GAPPostNetAwareLoss`

**注册名与签名**

```python
GAPPostNetAwareLoss(
    sampling_rate=48000,
    pesq_weight=0.0,
    utmos_weight=0.0,
    allow_missing_aware=False,
    utmos_model_dir=None,
    **mel_kwargs,
)

forward(enhanced: [B,T] | [B,C,T], clean: [B,T] | [B,C,T],
        lengths: [B] | None = None) -> scalar
```

上述 `[B,C,T]` 只对两个代理均未构造的 Mel-only 路径普遍成立；进入 aware 路径后，输入仅支持 `[B,T]` 或 `[B,1,T]`。

构造时先建立一个 `PaseMultiScaleMelLoss(sampling_rate=sampling_rate, **mel_kwargs)`。总目标为

$$
L=L_{\mathrm{multi\text{-}scale\ mel}}
+\frac1B\sum_{b=1}^{B}\left[
\lambda_P L_P(\widehat s_b,s_b)+
\lambda_U L_U(\widehat s_b)
\right],
$$

但 forward 实际只对非 `None` 的代理对象求和；缺失 PESQ 的 disabled 对象仍非 `None`，但其 forward 返回图保留零。只要 `pesq_loss` 或 `utmos_loss` 至少一个非 `None`，forward 就进入 aware 路径：每条样本先按 `lengths[b]` 裁剪，并且只在 `sampling_rate != 16000` 时调用已在模块 import 期导入的 `torchaudio.functional.resample`，把增强和 clean 都可微重采样到 16 kHz。aware 路径只接受 `[B,T]` 或单通道 `[B,1,T]`；多通道张量即使 Mel 主项可处理，也会在该路径 fail-fast。

- `L_P` 是 [`PESQLoss`](../../../core/losses/perceptual_metric.py) 的输出：构造参数固定为 `sample_rate=16000, mode="wb"`。该项目包装器把底层 differentiable PESQ loss 原值作为最小化目标。
- `L_U=-mean(UTMOS22Strong(enhanced.float(),16000))`。UTMOS 参数冻结、始终为 eval；梯度仍穿过冻结网络回到 enhanced waveform。
- `pesq_weight`、`utmos_weight` 默认都为 0；构造函数只拒绝 `<0`，没有检查有限性，因此 `NaN` 与 `+Inf` 可通过该检查。配置必须自行保证两个权重均为有限非负数。合法的默认 `0/0` 路径与纯 `PaseMultiScaleMelLoss` 的数值和梯度完全一致，不构造两个代理。
- `allow_missing_aware` 的捕获边界不是任意 aware 失败。PESQ 的 `PESQLoss` 只捕获 `pytorch_pesq.PesqLoss` 导入或实例化时的 `ImportError`：为 `False` 时，它把该 `ImportError` 包装成以原异常为 cause 的 `RuntimeError`，不是重新抛出原异常；为 `True` 时保留一个 disabled `PESQLoss` 对象，其 forward 返回连接 enhanced 的图保留零。其他 PESQ 构造异常和任意 forward 异常仍上抛。UTMOS 侧包围整个 `_UTMOSAwareLoss` 构造并捕获 `ImportError`、`FileNotFoundError`、`RuntimeError`：为 `True` 时，源码/权重缺失、vendored import 失败及以 `RuntimeError` 表现的加载失败都会把该分支置为 `None`；为 `False` 时这些异常按原类型重新抛出。其他异常和任意 forward 异常仍上抛。这项策略只能表示对应 aware 监督缺失后训练仍可继续。
- 必要边界：metric-aware 论文可解释 PESQ/UTMOS 代理的动机，但本地目标严格是上述 `PaseMultiScaleMelLoss + PESQ proxy + UTMOS proxy`。类默认及当前 Phase-2 配置均为 `pesq_weight=0.0, utmos_weight=0.0`，所以默认只有 Mel 主项，不能描述为已启用 `0.5/0.5` aware 配方或完整论文目标。

源码见 [`core/losses/pase_family.py`](../../../core/losses/pase_family.py)。[`tests/test_gap_urgenet_postnet_aware_loss.py`](../../../tests/test_gap_urgenet_postnet_aware_loss.py) 覆盖零权重纯 Mel 数值/梯度 parity 与 monkeypatch 代理组合公式；它不覆盖真实依赖缺失、上述异常捕获边界或非 16 kHz resample 分支。PESQ / UTMOS 两个代理的 import、依赖缺失、逐样本裁剪与 reduction 边界见 [第三方 Wrapper 的 `GAPPostNetAwareLoss` 小节](Third_Party_Wrappers.md)，本文不重复。

### 3.3 `MSELoss`

**注册名与签名**

```python
MSELoss(size_average=None, reduce=None, reduction="mean")
forward(input, target, lengths=None, **kwargs) -> Tensor
```

该类直接继承 `torch.nn.MSELoss`，只在 `forward` 中接收框架兼容参数后调用 `super().forward(input,target)`。对默认 `reduction="mean"`：

$$
L=\frac1{\operatorname{numel}(x)}\sum_i(x_i-y_i)^2.
$$

`reduction="sum"` 返回平方误差和，`reduction="none"` 返回逐元素张量，语义完全沿用 PyTorch。

**关键边界：`lengths` 和所有额外 `kwargs` 都被忽略。** 因而变长 batch 的 padding 仍参与 MSE reduction；它不具备本文其他 waveform loss 的 padding invariance。标准训练若要求标量，不能配 `reduction="none"`，否则 strict wrapper 或后续 backward 契约会失败。它适合形状已经对齐、padding 已由上游处理的 feature distillation 或简单张量回归，不应因为签名里出现 `lengths` 就宣称支持变长掩码。

### 3.4 `ScalarPassthroughLoss`

**注册名与签名**

```python
ScalarPassthroughLoss()
forward(enhanced, clean, lengths=None, **kwargs) -> enhanced
```

实现只有

$$
\operatorname{forward}(\texttt{enhanced},\ldots)=\texttt{enhanced}.
$$

它把 `enhanced` **原样返回**：同一 Tensor 对象、shape、dtype、device、数值与 autograd 图均不变。它忽略 `clean`、`lengths` 和额外参数，不做 clone、detach、求和、平均或标量化。名称中的 `Scalar` **不表示本类会把输入变成标量；它不是可独立最小化的标量 objective**。若标准 waveform trainer 传入 `[B,T]`，返回值仍是 `[B,T]`，启用 `strict` 时会因“loss 必须为标量”立即失败。

该类只用于满足统一配置要求：例如 `DualInputLossTrainer`、MeanFlow/MeanSE 或若干 distillation wrapper 在模型 `forward()` 内部已经计算真实标量 loss，配置中的本对象不会被 trainer 当作训练目标调用。只有先确认 trainer-owned/model-owned objective 路径，才能使用这一注册名。

---

## 4. 分类辅助项：ignore 语义与 DDP mean mass

以下两个 CE loss 都只适合作为 `CompositeLoss` 的辅助 component。它们各自实现 `ddp_global_mean_mass(...)`，返回本 rank 的有效标签数。设 rank `r` 的本地 mean 为 `L_r`、有效标签数为 `m_r`、world size 为 `R`，`CompositeLoss` 在各 rank 前向中重标定为

$$
\widetilde L_r=L_r\frac{R m_r}{\sum_jm_j}.
$$

DDP 随后对各 rank 梯度取平均，得到按全局有效标签总数归约的梯度，而不是“各 rank local mean 再等权平均”。即使某个 rank 全是 ignore，它也必须进入相同的 mass collective；对应 loss 返回与辅助 head 计算图连接的精确零，避免 rank-asymmetric all-ignore 导致 DDP hook 或 collective 序列失配。component `weight=0` 时不调用 CE forward 或 mass callback，但 `CompositeLoss` 仍以零质量执行固定 collective。

两类目标均调用 `torch.nn.functional.cross_entropy`。本地额外定义字段/shape 校验、`ignore_index=-1`、all-ignore 图保留零、有效质量回调、DDP 全局 mean 与 Trainer fail-fast 接线；这些工程边界不能从通用 CE 或 label smoothing 语义中推断。

### 4.1 `StateCrossEntropyLoss`

**注册名、签名与 shape**

```python
StateCrossEntropyLoss(ignore_index=-1, label_smoothing=0.0)

forward(
    enhanced,                         # ignored
    clean,                            # ignored
    lengths=None,                    # ignored
    state_probs: [B,N,C] | None=None, # 必须是浮点 logits，不是 probability
    state: [B,N] | None=None,         # 整数类别索引
    **kwargs,
) -> scalar
```

`state_probs` 这一历史字段名容易误导：源码把它直接作为 logits 交给 `F.cross_entropy`，调用方不得先做 softmax。`state_probs` 与 `state` 必须同时存在、位于同一设备并具有完全相同的 batch/time 维；`C>0`。非 ignore 标签必须满足 `0≤state<C`。`label_smoothing=η` 必须有限且 `0≤η<1`。

令有效位置集合 `V={(b,n)|state_bn≠ignore_index}`，`p_bnc=softmax(z_bn)_c`。当前无 class weight，故 `V` 非空时的精确目标可写为

$$
L_{\mathrm{state}}=\frac1{|V|}\sum_{(b,n)\in V}
\left[(1-\eta)(-\log p_{bn,y_{bn}})
+\eta\left(-\frac1C\sum_{c=1}^{C}\log p_{bnc}\right)\right].
$$

源码通过 `F.cross_entropy(state_probs.transpose(1,2), state, ignore_index=..., label_smoothing=η, reduction="mean")` 实现。若 `V` 为空，则返回 `state_probs[valid].sum()`：空选择既不读取 logits 中的 NaN/Inf，又返回同 dtype/device、可反传的精确零。`ddp_global_mean_mass` 返回 `|V|` 的 float32 标量。

该 loss 是项目的 **frame-level state CE engineering mode**，可支持 DBGNet 的 3 类 FST/DT/NST、项目 8 类场景或任意正类别数；它不同于原论文带时间 pooling 的 segment-level `L_SD`。启用非零 component 时，AEC trainer 要求模型给出 `state_probs` 且数据给出逐帧 `state`；旧 batch 无标签、head 缺失或时间轴错位都会 fail-fast。它只优化分类 head 与共享特征，不使用 `enhanced/clean/lengths`，不能单独作为增强主目标。

源码：[`core/losses/state_ce.py`](../../../core/losses/state_ce.py)。测试：[`tests/test_aec_state_ce_failfast.py`](../../../tests/test_aec_state_ce_failfast.py)、[`tests/test_shared_training_correctness.py`](../../../tests/test_shared_training_correctness.py)、[`tests/test_core_losses_audit_regressions.py`](../../../tests/test_core_losses_audit_regressions.py)。

### 4.2 `SpeakerCrossEntropyLoss`

**注册名、签名与 shape**

```python
SpeakerCrossEntropyLoss(ignore_index=-1, label_smoothing=0.0)

forward(
    enhanced,                       # ignored
    clean,                          # ignored
    lengths=None,                  # ignored
    spk_logits: [B,C] | None=None, # 浮点 logits
    spk_labels: [B] | None=None,   # 整数类别索引
    **kwargs,
) -> scalar
```

`spk_logits`、`spk_labels` 都是必需输入，必须位于同一设备，`C>0`；非 ignore 标签必须在 `[0,C)`。默认 `ignore_index=-1` 表示未知 speaker。令 `V={b|spk_labels_b≠ignore_index}`，则

$$
L_{\mathrm{spk}}=\frac1{|V|}\sum_{b\in V}
\left[(1-\eta)(-\log p_{b,y_b})
+\eta\left(-\frac1C\sum_{c=1}^{C}\log p_{bc}\right)\right].
$$

实现先显式选出 `logits[V]` 与 `labels[V]`，再调用 `F.cross_entropy(..., label_smoothing=η, reduction="mean")`。全 ignore 时返回 `spk_logits[valid].sum()`，即不读取 NaN/Inf logits 的图保留精确零；DDP mass 为本 rank 的有效 utterance 数 `|V|`。

这是 TSE 句级 multi-task 辅助项，不是 speaker embedding 距离本身。非零 component 会驱动 entry 启用分类 head 并要求 `spk_labels`；缺 `spk_labels`、缺 `spk_logits`、标签空间越界都会 fail-fast（TSE 旧 6 元组兼容已移除，`TSEBatch` 恒含 `spk_labels`）。只有 speaker CE 而没有增强主 loss 的 auxiliary-only 配置会在 TSE entry 被拒绝；`weight=0` 视为未启用，并不会据此打开 multi-task head。

源码：[`core/losses/speaker_ce.py`](../../../core/losses/speaker_ce.py)。测试：[`tests/test_tse_trainer_enhancements.py`](../../../tests/test_tse_trainer_enhancements.py)、[`tests/test_tse_entry.py`](../../../tests/test_tse_entry.py)、[`tests/test_core_losses_audit_regressions.py`](../../../tests/test_core_losses_audit_regressions.py)。

---

## 5. `MPSEnetCompositeLoss`：MP-SENet reconstruction（由 MetricGAN trainer 调用）

该类实现 MP-SENet（一个语音增强模型）风格的五项 reconstruction（重建目标）：压缩幅度、anti-wrapping 相位、压缩复谱、波形和 STFT consistency。它由本地 `MetricGANCompositeTrainer` 调用；MetricGAN 指用指标/判别网络参与训练的范式，GAN 指生成对抗网络。这类 metric loss 与 discriminator（判别器）objective 由 Trainer 管理、不在本类中，必须与这五项分开描述。

**注册名、签名与默认值**

```python
MPSEnetCompositeLoss(
    weight_mag=0.9,
    weight_pha=0.3,
    weight_com=0.1,
    weight_time=0.2,
    weight_stft=0.1,
    stft_n_fft=400,
    stft_hop_length=100,
    stft_win_length=400,
    compress_factor=0.3,
    window="hann",  # 只有精确字符串 "hamming" 选择 Hamming，其余值走 Hann
)

forward(forward_dict: dict[str, Tensor]) -> scalar
SIGNATURE = "forward_dict"
```

它虽在 `LOSS_REGISTRY` 中，却不是标准 `(enhanced,clean,lengths)` loss，也不是 builder 特判的通用 `CompositeLoss`。所需字典字段为：

| 字段 | shape | 语义 |
|---|---|---|
| `clean_mag`, `enhanced_mag` | `[B,F,N]` | 已做 `compress_factor=c` 幂压缩的幅度 |
| `clean_pha`, `enhanced_pha` | `[B,F,N]` | 弧度相位 |
| `clean`, `enhanced` | `[B,T]` | 目标与重建波形 |
| `lengths` | `[B]`，可选 | 有效波形样本数 |
| `loss_enhanced_spectra` | `[B,F,N,2]`，可选 | generator 原始 real/imag 谱；存在时优先于 `enhanced_mag/pha` |

当 `loss_enhanced_spectra` 存在时，源码先 `view_as_complex`，再重算

$$
\widehat M=|\widehat S_{\mathrm{raw}}|^c,
\qquad \widehat\Phi=\angle\widehat S_{\mathrm{raw}},
$$

以免 sanitize 后供判别器使用的谱覆盖 generator 原始 reconstruction-loss 路径。`clean_spec`、`enhanced_spec` 不是该 loss 自身的必需读取字段。

### 5.1 mask 与归约原语

`lengths` 生成两类边界：waveform sample mask `Q[b,t]` 排除 time loss 中的无效样点；按 `stft_n_fft` 的 centered-STFT 帧数生成的 frame mask `M[b,n]` 排除 mag/phase/complex/consistency 中的无效谱帧。所有 masked mean 使用同一个原语：

$$
\operatorname{MM}(X;M)=
\frac{\sum X\,\operatorname{expand\_as}(M,X)}
{\max\left(1,\sum\operatorname{expand\_as}(M,X)\right)}.
$$

因此频率、实虚等广播维都会进入分母；空差分轴返回可反传零。这里的 frame mask 只负责排除调用者谱时间轴上的无效帧：mag/phase/complex 项直接消费 `forward_dict` 中的调用者谱（或优先使用 `loss_enhanced_spectra` 推得的增强谱），不会根据 `lengths` 从 waveform 重算这些谱。只有调用者提供的有效谱帧本身与逐条裁剪 waveform 后的谱等价（trim-equivalent）时，这三项及依赖增强谱的 consistency 项才具备完整 padding parity；mask 不能修复已被 batch 右 padding 污染的有效边界帧。

### 5.2 五项精确公式

令 `A(x)=|x-round(x/(2π))·2π|` 为 anti-wrapping 距离。

1. **压缩幅度 MSE**

   $$
   L_{\mathrm{mag}}=\operatorname{MM}
   \left((\widehat M-M)^2;\ M_{bn}\right).
   $$

2. **相位项**：IP 对原相位差；GD 对频率轴 `diff(dim=1)`；IAF 对时间轴 `diff(dim=2)`，且时间 mask 同步去掉第一帧：

   $$
   \begin{aligned}
   L_{\mathrm{IP}}&=\operatorname{MM}(A(\Phi-\widehat\Phi);M_{bn}),\\
   L_{\mathrm{GD}}&=\operatorname{MM}
   \left(A(\Delta_f\Phi-\Delta_f\widehat\Phi);M_{bn}\right),\\
   L_{\mathrm{IAF}}&=\operatorname{MM}
   \left(A(\Delta_t\Phi-\Delta_t\widehat\Phi);M_{b,n\ge1}\right),\\
   L_{\mathrm{pha}}&=L_{\mathrm{IP}}+L_{\mathrm{GD}}+L_{\mathrm{IAF}}.
   \end{aligned}
   $$

3. **压缩复谱项**：

   $$
   C(M,\Phi)=[M\cos\Phi,\ M\sin\Phi],\qquad
   L_{\mathrm{com}}=2\operatorname{MM}
   \left((C(\widehat M,\widehat\Phi)-C(M,\Phi))^2;M_{bn}\right).
   $$

   mask 会广播到最后的 real/imag 维，乘 2 后等价于对每个 TF bin 的两维平方误差求和再按 TF bin 平均。

4. **波形 L1**：

   $$
   L_{\mathrm{time}}=
   \frac{\sum_{bt}Q_{bt}|\widehat s_{bt}-s_{bt}|}
   {\max(1,\sum_{bt}Q_{bt})}.
   $$

5. **STFT consistency**：只重新分析 `enhanced` 波形。若提供 `lengths`，源码先清零 waveform 无效样点、预留右侧余量，并从每条样本的有效右边界做 reflect 延展；若未提供 `lengths`，极短波形也会走同一准备逻辑。随后 STFT 使用构造几何、所选 Hann/Hamming 窗、`center=True`、`pad_mode="reflect"`、`normalized=False`，并裁到调用者增强复谱 `C(Ŝ)` 的帧数。令重新分析复谱为 `R`，源码固定 `ε=1e-10`：

   $$
   M_R=(\Re(R)^2+\Im(R)^2+10^{-10})^{c/2},
   \qquad
   \Phi_R=\operatorname{atan2}(\Im(R)+10^{-10},\Re(R)+10^{-10}),
   $$

   $$
   L_{\mathrm{stft}}=2\operatorname{MM}
   \left((C(\widehat M,\widehat\Phi)-C(M_R,\Phi_R))^2;M_{bn}\right).
   $$

总目标为

$$
L=w_{\mathrm{mag}}L_{\mathrm{mag}}
+w_{\mathrm{pha}}L_{\mathrm{pha}}
+w_{\mathrm{com}}L_{\mathrm{com}}
+w_{\mathrm{time}}L_{\mathrm{time}}
+w_{\mathrm{stft}}L_{\mathrm{stft}}.
$$

幅度、相位和复谱项直接约束 generator 谱；time 项约束 iSTFT 波形；consistency 项约束 generator 谱与该波形重新分析后的谱一致。本地 `MPSEnetCompositeLoss` 只封装上述五项 reconstruction；PESQ injection、真实/生成分数与 discriminator objective 由 `MetricGANCompositeTrainer` 及其父类管理，不是该 loss 的第六项。

### 5.3 MetricGAN trainer 调用边界

直接通过 `build_loss` 构造该类时，`loss.args.weight_*` 会按普通构造参数生效。但生产 `MetricGANCompositeTrainer` **不会使用 entry 构造的 `self.loss_fn` 计算 reconstruction loss**；trainer 会自行重新实例化 `MPSEnetCompositeLoss`，并按以下边界取值：

- 五个权重只来自 `trainer.loss_mag_weight`、`trainer.loss_pha_weight`、`trainer.loss_com_weight`、`trainer.loss_time_weight`、`trainer.loss_stft_weight`，默认 `0.9/0.3/0.1/0.2/0.1`；`loss.args.weight_*` 在这条 trainer 路径不生效。
- STFT 的 `n_fft/hop_length/win_length` 来自顶层 `stft`，默认 `400/100/400`。
- trainer 只从 `loss.args` 消费 `compress_factor` 与 `window`（另保留历史顶层 fallback），默认 `0.3` 与顶层 STFT window/Hann。
- `loss.type` 应写 `MPSEnetCompositeLoss`；写其他类型只会触发 warning，并不会替换 trainer-owned reconstruction objective。

源码：[`core/losses/model_specific/metricgan.py`](../../../core/losses/model_specific/metricgan.py)；trainer 边界：[`tasks/se/trainer/metricgan_trainer.py`](../../../tasks/se/trainer/metricgan_trainer.py)。[`tests/test_metricgan_composite_loss_lengths.py`](../../../tests/test_metricgan_composite_loss_lengths.py) 的 padding parity case 复用预先从 trim waveform 计算的谱字段，因而证明的是 time/consistency 的有效右边界处理，以及在调用者谱已 trim-equivalent 前提下的总 loss/有效 waveform 梯度 parity；独立 phase case 证明 frame mask 排除无效谱帧。它不证明任意 batch-padded 调用者谱会被 `lengths` 修复或重算。[`tests/test_loss_audit_fixes.py`](../../../tests/test_loss_audit_fixes.py) 只覆盖 full-length mask 与 `lengths=None` 的归约 parity；raw generator 谱优先级和生产配置边界另由前一测试及 [`tests/test_loss_build.py`](../../../tests/test_loss_build.py) 覆盖。

---

## 6. DeepFilterNet/DPDFNet 专用项

本节各项借用 DeepFilterNet/DPDFNet 的 MR、local-SNR、alpha 与 over-attenuation 目标动机。DeepFilterNet 与 DPDFNet 是同一模型系列的不同变体（模型侧按 YAML 参数在 DeepFilterNet2/3 与 DPDFNet2/4/8 之间切换，见 [`tasks/se/models/__init__.py`](../../../tasks/se/models/__init__.py)）；本节只借用它们辅助目标的动机，实际运行公式由本仓库 [`deepfilternet.py`](../../../core/losses/model_specific/deepfilternet.py) 定义。凡窗口、hop、幅度距离、target、mask、权重或归约与论文/上游不同之处，均按项目变体理解。

注册名均与类名相同，没有 `DPDFNetOverAttenuationLoss` 或 `DfAlphaLoss` registry 别名；当前实际名称是 `DeepFilterNetOverAttenuationLoss` 与 `DeepFilterNetAlphaLoss`。

### 6.1 `DeepFilterNetMultiResSpecLoss`

**签名与默认值**

```python
DeepFilterNetMultiResSpecLoss(
    fft_sizes=(480, 960, 1920),
    hop_factor=0.5,
    sample_rate=48000,
    max_freq_hz=5000.0,
    reduction="mean",  # "mean" | "sum"
)

forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None,
        **kwargs) -> scalar
```

`fft_sizes` 必须是非空的精确正整数序列；`hop_factor` 必须有限且大于 0；`sample_rate` 是正整数；`max_freq_hz` 为 `None` 或有限正数；`reduction` 只接受 `mean/sum`。

对尺度 `k`，源码不是使用全长窗，而是

$$
w_k=\min(n_k,\max(1,\lfloor n_k/2\rfloor)),\qquad
h_k=\max(1,\lfloor n_k\rho\rfloor),
$$

其中 `ρ=hop_factor`。STFT 使用长度 `w_k` 的 Hann 窗、`n_fft=n_k`、`center=True`、`onesided=True`，输入强制 FP32，幅度为

$$
A_k(x)=\operatorname{clamp}_{\min}(|\operatorname{STFT}_k(x)|,10^{-8}).
$$

若 `max_freq_hz=f_max` 非空且小于 Nyquist，仅保留

$$
f\in\{0,\ldots,\min(\lfloor f_{\max}n_k/f_s\rfloor,F_k-1)\};
$$

频率恰好等于上界的 bin 被包含。令保留后的频率数为 `F'_k`，则

$$
D_{bfn}^{(k)}=|A_k(\widehat s)_{bfn}-A_k(s)_{bfn}|,
\qquad
Z_k=\max\left(1,F'_k\sum_{bn}M_{bn}^{(k)}\right).
$$

单尺度 reduction 为

$$
L_k=\begin{cases}
\sum D_kM_k/Z_k,&\text{若提供 lengths，或 reduction="mean"};\\
\sum D_k,&\text{若 lengths=None 且 reduction="sum"}.
\end{cases}
$$

最终始终对尺度取平均：

$$
L_{\mathrm{DFN\text{-}MR}}=\frac1K\sum_kL_k.
$$

因此 `reduction="sum"` 只决定每尺度是否按元素平均；它不会取消最外层 `/K`。一旦传 `lengths`，即使配置 `sum` 也会改为有效元素 mean。该项偏好多时间/频率尺度上的绝对幅度保真；默认 5 kHz 截断会忽略更高频率误差。

**实现边界**：这是 DeepFilterNet MR 目标的项目变体：半长 Hann 窗、默认 half-hop、`normalized=False`、未压缩幅度 L1、尺度平均、默认 5 kHz 截断，并加入变长 mask、短输入扩展和参数校验；不能按其他 MR spectrogram loss 的 full-window、quarter-hop、幅度 MSE 或压缩复谱公式解释。

### 6.2 `DeepFilterNetOverAttenuationLoss`

**签名与默认值**

```python
DeepFilterNetOverAttenuationLoss(
    n_fft=960,
    hop_length=None,  # falsy 时 n_fft // 2
    win_length=None,  # falsy 时 n_fft，再截到 <= n_fft
    eps=1e-8,
)

forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None,
        **kwargs) -> scalar
```

有效参数解析为 `hop_length = provided or n_fft//2`，`win_length=min(provided or n_fft,n_fft)`，窗为 Hann。波形先强制 FP32，再做 `center=True`、单边 STFT；两侧幅度各自先 clamp 到 `1e-8`：

$$
\widehat A=\max(|\operatorname{STFT}(\widehat s)|,10^{-8}),
\qquad A=\max(|\operatorname{STFT}(s)|,10^{-8}).
$$

精确 per-bin 项是

$$
R_{bfn}=\frac{\operatorname{where}(A-\widehat A>0,
A-\widehat A,(A-\widehat A)\cdot0)}{A+\epsilon},
$$

而不是对称谱距离。最终结果为

$$
L_{\mathrm{OA}}=\frac{\sum_{bfn}M_{bn}R_{bfn}}
{\max(1,F\sum_{bn}M_{bn})}.
$$

它只惩罚 `|Ŝ|<|S|` 的过衰减，与前文 `MRSpecSuppressPenaltyLoss` 只惩罚 `|Ŝ|>|S|` 恰好方向相反。`eps` 默认 `1e-8` 加在已经 clamp 的 clean 幅度之后；幅度 clamp 本身也是固定 `1e-8`，不随 `eps` 改变。完美重建得到 loss 0 和零梯度。该项应作为抗过抑制辅助项与保真/抑噪主项组合，单独使用会偏好不低于 clean 的幅度，却不惩罚过高幅度。

**实现边界**：该类只借用 over-attenuation 动机；实际是单分辨率、未压缩幅度的 clean-relative ReLU/L1 比值并带 `lengths` mask，不是多分辨率压缩幅度/复谱 OA 公式的实现。

谱边界、上界 bin、极短输入、padding 与梯度由 [`tests/test_deepfilternet_loss_boundaries.py`](../../../tests/test_deepfilternet_loss_boundaries.py) 和 [`tests/test_core_losses_audit_regressions.py`](../../../tests/test_core_losses_audit_regressions.py) 覆盖。

### 6.3 共同局部 SNR target：`compute_frame_snr`

`DeepFilterNetLSNRLoss` 与 alpha adapter 使用项目的时域短帧 target。对 `[B,T]` 的 `clean=s`、`noise=n`，用 `Tensor.unfold(1,frame_length,hop_length)` 得到无 padding、无 center 的帧。第 `j` 帧：

$$
P^s_{bj}=\max\left(\frac1Q\sum_{q=1}^{Q}s_{bjq}^2,10^{-8}\right),
\qquad
P^n_{bj}=\max\left(\frac1Q\sum_{q=1}^{Q}n_{bjq}^2,10^{-8}\right),
$$

$$
y_{bj}=10\log_{10}\frac{P^s_{bj}}{P^n_{bj}}.
$$

`compute_frame_snr` 的 `eps` 默认且调用时固定为 `1e-8`。这是项目的**时域** `unfold` target，不含 target-range 映射或谱域平滑；不能与谱域 `LocalSnrTarget` 数值等同。`SEBatch` 不单独携带 noise 时，wrapper/adapter 按 `noise=noisy-clean` 推导噪声。

### 6.4 `DeepFilterNetLSNRLoss`

**非标准签名与默认值**

```python
DeepFilterNetLSNRLoss(
    frame_length=960,
    hop_length=480,
    weight=0.0005,
)

forward(
    predicted_lsnr: [B,N] | [B,N,1],
    clean: [B,T],
    noise: [B,T],
    lengths: [B] | None = None,
) -> scalar
```

它不是标准 `(enhanced,clean,lengths)` 波形 loss：第一参数是 LSNR head 输出，第三个位置参数是 noise。`[B,N,1]` 会 `squeeze(-1)`。若 `T<frame_length`，返回 `(predicted_lsnr*0).sum()`；否则计算上节 target，再令

$$
N'=\min(N_{\mathrm{pred}},N_{\mathrm{target}}),
$$

并将预测和 target 都裁到前 `N'` 帧。`N'=0` 也返回图保留零。

没有 `lengths` 时：

$$
L_{\mathrm{LSNR}}=\lambda\frac1{BN'}
\sum_{b,j<N'}(\widehat y_{bj}-y_{bj})^2,
\qquad \lambda=\texttt{weight}.
$$

提供 `lengths` 时，第 `b` 条有效 unfold 帧数为

$$
n_b=\min\left(N',\max\left(0,
\left\lfloor\frac{L_b-\texttt{frame\_length}}
{\texttt{hop\_length}}\right\rfloor+1\right)\right),
$$

$$
L_{\mathrm{LSNR}}=\lambda
\frac{\sum_{b,j<N'}\mathbf{1}\{j<n_b\}(\widehat y_{bj}-y_{bj})^2}
{\sum_bn_b}.
$$

任一样本 `n_b≤0` 会发 `RuntimeWarning`；全 batch 有效帧总数为 0 时返回图保留零。默认 `0.0005` 是很小的辅助权重。该项使模型内部 local-SNR head 拟合工程 target，主要为 mask/DF 控制提供辅助表征，不应作为增强波形的唯一目标。

**实现边界**：该类采用 `weight·MSE` 的 local-SNR 辅助目标，但 target 是项目时域公式，并增加帧轴裁剪、`lengths` mask、短输入 warning 与图保留零。默认 `weight=0.0005`；target 和这些运行边界均以本节为准。

### 6.5 `DeepFilterNetAlphaLoss`

**非标准签名与默认值**

```python
DeepFilterNetAlphaLoss(
    factor=0.05,
    lsnr_thresh=-7.5,
    lsnr_min=-10.0,
    frame_length=320,
    hop_length=160,
)

forward(
    pred_alpha: [B,N] | [B,N,1],
    target_lsnr: [B,N_target],
    lengths: [B] | None = None,
) -> scalar
```

`[B,N,1]` 先 squeeze；两条时间轴裁到 `N'=min(N,N_target)`，空轴返回图保留零。源码映射函数为

$$
g(y;a,b)=1-\operatorname{clamp}
\left(\frac{y-b}{a-b},0,1\right),
$$

其中调用者应保证 `a≠b`；构造函数当前没有额外参数校验。关断权重和开启权重分别为

$$
w_{\mathrm{off}}=g(y;\theta,m),
\qquad
w_{\mathrm{on}}=g(y;\theta+2.5,0),
$$

`θ=lsnr_thresh`、`m=lsnr_min`。默认值下：`w_off` 在 `y≤-10 dB` 为 1，在 `y≥-7.5 dB` 为 0；`w_on` 在 `y≤-5 dB` 为 0，在 `y≥0 dB` 为 1。第二次调用把 `lsnr_thresh=-5`、`lsnr_min=0` 传给同一个映射函数，虽然参数名暗示的顺序与数值顺序相反，但按源码公式计算，权重确实随 SNR 从 -5 dB 到 0 dB 由 0 升到 1。实际 gate 为：

$$
\ell_{bj}=(\widehat\alpha_{bj}w_{\mathrm{off},bj})^2
+0.1\left|(1-\widehat\alpha_{bj})w_{\mathrm{on},bj}\right|.
$$

无 `lengths` 时 `L_alpha=factor·mean(ℓ)`；提供时使用与 LSNR loss 相同的 `n_b` 公式和 mask：

$$
L_{\alpha}=\texttt{factor}\cdot
\frac{\sum_{b,j<N'}\mathbf{1}\{j<n_b\}\ell_{bj}}{\sum_bn_b}.
$$

全无有效帧时返回连接 `pred_alpha` 的零。源码不 clamp `pred_alpha` 到 `[0,1]`，该范围应由模型输出层保证。默认 `factor=0.05`；该项直接偏置 DF alpha 的开关行为，必须与模型 alpha 定义及 SNR target 对齐。

**实现边界**：两段 LSNR 映射表达 DF alpha 的关断/开启控制；当前类默认 `factor=0.05`，并增加时间轴裁剪、`lengths` mask 与图保留零。帧数截断、mask、短输入和梯度连接由 [`tests/test_deepfilternet_alpha_alignment.py`](../../../tests/test_deepfilternet_alpha_alignment.py) 覆盖。

### 6.6 `DeepFilterNetFullLossWithLSNR`

**签名与默认值**

```python
DeepFilterNetFullLossWithLSNR(
    enhancement_loss_type="SISNRLoss",
    enhancement_loss_args=None,
    stft_cfg=None,
    lsnr_weight=0.0005,
    lsnr_frame_length=960,
    lsnr_hop_length=480,
    use_alpha_loss=False,
    alpha_weight=0.05,
    sample_rate=None,
)

forward(
    enhanced: [B,T], clean: [B,T], lengths: [B] | None = None,
    predicted_lsnr=None, predicted_alpha=None, noise=None,
    **kwargs,  # 可含 noisy
) -> scalar
```

构造期用 `build_loss` 创建主增强项；`enhancement_loss_args` 中显式采样率优先，`stft_cfg` 和 `sample_rate` 继续传入嵌套 builder。内部 LSNR loss 使用给定 frame/hop/weight；启用 alpha 时，alpha loss 的 `factor=alpha_weight`，但 `lsnr_thresh/lsnr_min` 沿用 `-7.5/-10.0`。

令 `L_enh` 为主项。若 `noise is None` 且 `kwargs["noisy"]` 存在，先取 `noise=noisy-clean`。当前 forward 的精确 gate 是：

$$
L=\begin{cases}
L_{\mathrm{enh}}+L_{\mathrm{LSNR}}
+\mathbf{1}\{\texttt{use\_alpha\_loss}\land\widehat\alpha\text{存在}\}L_\alpha,
&\widehat y\text{存在且 noise 可得};\\
L_{\mathrm{enh}},&\text{否则}.
\end{cases}
$$

alpha 项嵌套在 LSNR gate 内；即使 `predicted_alpha` 存在，只要 `predicted_lsnr` 或 noise 缺失，它也不会执行。`clean.shape[1]<lsnr_frame_length` 时 LSNR 返回连接预测 head 的零；启用且提供 alpha 时，full loss 另加 `(predicted_alpha*0).sum()` 保持 alpha head 的图连接。

**Trainer 运行 gate 与静默边界**：该类是项目的主增强项 + 时域 LSNR/alpha 组合 wrapper，不是其他 full-loss wrapper 的逐项复刻。标准 SE 训练中，`trainer.use_lsnr_loss=true` 是请求模型 head 并向 loss 传入 `predicted_lsnr` / `predicted_alpha` 的**必要 gate**；实际请求 LSNR 还要求模型支持 `return_lsnr` 且 loss 接口接受 `predicted_lsnr`，实际请求 alpha 还要求当前 loss 需要 alpha 且模型支持 `return_alpha`。因此：

- `trainer.use_lsnr_loss` 未启用时，训练不请求或传入两个 head；本 full wrapper 不报错，静默只返回 `L_enh`。
- gate 已启用但模型或 loss 不满足上述能力条件时，训练同样可能不传 head；本 wrapper 仍按自身 forward gate 静默退化为 enhancement 主项。`use_alpha_loss=True` 但模型不支持 `return_alpha` 时，`strict_checks=true` 会在启动期报错，否则只告警且 alpha 项不执行。
- 标准 SE validation 固定以 `return_lsnr=false, return_alpha=false` 调用模型，也不向 loss 注入预测 head，所以本 wrapper 的 validation objective 固定只有 enhancement 主项。
- 下一节两个 adapter 不采用这种静默策略：对应预测 head 缺失即 fail-fast。它们若作为非零 component 放入标准 SE validation，会因 validation 固定不请求 head 而失败；只有 train 与 validation 都由自定义路径注入相应 head 时才适用。

嵌套 STFT/sample-rate 透传由 [`tests/test_loss_build.py`](../../../tests/test_loss_build.py) 覆盖。

### 6.7 `DeepFilterNetLSNRComponent`

**标准 Composite adapter 签名**

```python
DeepFilterNetLSNRComponent(frame_length=320, hop_length=160)

forward(enhanced: [B,T], clean: [B,T], lengths=None,
        predicted_lsnr=None, noise=None, **kwargs) -> scalar
```

内部调用 `DeepFilterNetLSNRLoss(..., weight=1.0)`，所以 component 强度只由 `CompositeLoss.components[].weight` 控制，避免双重权重。标准 SE train 要把 head 传到这里，必须设置 `trainer.use_lsnr_loss=true`，且模型支持 `return_lsnr`；否则该非零 component 会因缺头 fail-fast。其输入契约是：

1. `predicted_lsnr` 缺失立即 `ValueError`；
2. `noise` 缺失时尝试 `kwargs["noisy"]-clean`；两者都无则立即 `ValueError`；
3. 输入齐全后返回上文 LSNR MSE，`lengths` 语义不变。

### 6.8 `DfAlphaLossComponent`

**标准 Composite adapter 签名**

```python
DfAlphaLossComponent(
    frame_length=320,
    hop_length=160,
    factor=0.05,
    lsnr_thresh=-7.5,
    lsnr_min=-10.0,
)

forward(enhanced: [B,T], clean: [B,T], lengths=None,
        predicted_alpha=None, noise=None, **kwargs) -> scalar
```

它先按与 LSNR adapter 相同的规则取得 noise，再用 `compute_frame_snr(clean,noise,frame_length,hop_length)` 生成 target，调用 `DeepFilterNetAlphaLoss`。标准 SE train 中 alpha 请求嵌套在 LSNR gate 内：必须设置 `trainer.use_lsnr_loss=true`，且模型同时支持 `return_lsnr`、`return_alpha`；否则该非零 component 会因缺头 fail-fast。精确输入边界为：

1. `predicted_alpha` 缺失立即 `ValueError`；
2. `noise` 缺失时尝试 `noisy-clean`，仍缺失立即 `ValueError`；
3. 输入齐全但 `clean.shape[1]<frame_length` 时返回连接 alpha head 的精确零，不报错；
4. 其余情况按 6.5 节公式计算。

**DFN auxiliary fail-fast 结论**：两个 adapter 把 LSNR/alpha 接入统一 component 接口；对应预测 head 或 noise/noisy reference 缺失时立即报错，短于一帧或有效帧总数为零则返回连接 head 的有定义零。`DeepFilterNetFullLossWithLSNR` 相反，会静默省略缺输入的辅助项。标准 SE validation 固定不请求或注入预测 head，因此非零 adapter component 必然 fail-fast；它们仅适用于 train/validation 都注入相应 head 的自定义路径。缺输入契约由 [`tests/test_core_losses_audit_regressions.py`](../../../tests/test_core_losses_audit_regressions.py) 覆盖，短输入与梯度连接由 [`tests/test_deepfilternet_alpha_alignment.py`](../../../tests/test_deepfilternet_alpha_alignment.py) 覆盖。

### 6.9 `ComplexIdealRatioMaskLoss`

该 SE loss 以模型直接输出谱、clean target 谱和 noisy reference-channel STFT 构造压缩复理想比率掩码监督。它只接入 cIRM loss，不代表仓库接入 NBC/NBC2 模型或完整训练配方。

```python
ComplexIdealRatioMaskLoss(
    n_fft=512, hop_length=256, win_length=512,
    window="hann", center=True, normalized=False,
    reference_channel=0, compression_k=10.0,
    compression_c=0.1, eps=1.1920928955078125e-7,
)
```

Trainer 必须提供 `enhanced_spectra` 与 `target_spectra`（complex `[B,F,N]` 或实虚轴 `[B,F,N,2]`），以及 noisy waveform `[B,T]` 或 `[B,C,T]`；多通道输入按 `reference_channel` 选取 STFT。令 `Y` 为该 noisy reference 的 STFT、`Z` 为输入复谱，则

$$
M(Z,Y)=\frac{ZY^*}{|Y|^2+\epsilon},\qquad
C(M)=k\tanh\left(\frac{c}{2}\operatorname{clamp}(M,-100)\right),\qquad
L=\operatorname{mean}\left[(C(M(\widehat S,Y))-C(M(S,Y)))^2\right].
$$

`clamp` 分别作用于掩码的实部和虚部。提供 `lengths` 时先清零 noisy 的无效样点，再按有效 frame mask 对每个样本的 TF 实虚元素求均值并取 batch mean；未提供时对全谱直接取 mean。`enhanced` waveform 不参与目标计算，梯度通过 `enhanced_spectra` 回传。框架注入的 STFT 几何必须与 direct spectra 一致；该 loss 需要 noisy reference 与 direct spectra，且不支持 `enable_ss_undesired_energy`。源码：[`core/losses/complex_mask.py`](../../../core/losses/complex_mask.py)；测试：[`tests/test_complex_mask_loss.py`](../../../tests/test_complex_mask_loss.py)、[`tests/test_se_offline_spatial.py`](../../../tests/test_se_offline_spatial.py)。

## 7. 必要边界与验证索引

### 7.1 Canonical 一对一边界

[`docs/paper/loss/README.md`](../../paper/loss/README.md) 中三篇 canonical 保持严格一对一：TAPLoss canonical 只对应同一实现的 `TAPLoss` / `AcousticLoss`；Loud-loss canonical 只对应当前未实现的 Loud-loss；MBRCL canonical 只对应 `MultiBandRelativeContrastiveLoss`。三篇均不支持本页其他 Loss，也不支持用户自研的 `SIL1WavSeTrainLoss`、两项 MR loss 或完整三项 SS 组合配方。

### 7.2 主要测试覆盖

| 契约 | 测试 |
|---|---|
| MR schedule、缓存、变长 padding/梯度 | [`test_loss_build.py`](../../../tests/test_loss_build.py)、[`test_loss_padding_invariance.py`](../../../tests/test_loss_padding_invariance.py)、[`test_composite_loss_mr_cache.py`](../../../tests/test_composite_loss_mr_cache.py) |
| PASE `match_stride=False` 下当前七尺度短输入 padding parity、常规无 padding 公式（不覆盖 `match_stride=True` 或全部参数失败边界） | [`test_pase_multiscale_mel_loss.py`](../../../tests/test_pase_multiscale_mel_loss.py) |
| GAP 零权重纯 Mel parity、monkeypatch 代理组合（不覆盖真实依赖缺失、异常包装与 resample） | [`test_gap_urgenet_postnet_aware_loss.py`](../../../tests/test_gap_urgenet_postnet_aware_loss.py) |
| State CE 输入 fail-fast、all-ignore | [`test_aec_state_ce_failfast.py`](../../../tests/test_aec_state_ce_failfast.py)、[`test_shared_training_correctness.py`](../../../tests/test_shared_training_correctness.py) |
| Speaker CE ignore、输入、trainer 接线和 rank-asymmetric DDP | [`test_tse_trainer_enhancements.py`](../../../tests/test_tse_trainer_enhancements.py)、[`test_tse_entry.py`](../../../tests/test_tse_entry.py) |
| CE 全局 mean mass | [`test_core_losses_audit_regressions.py`](../../../tests/test_core_losses_audit_regressions.py) |
| MetricGAN trim-equivalent 谱前提下的 padding parity、frame mask、full-length mask parity、raw 谱与生产权重（不证明任意 padded 谱会被重算） | [`test_metricgan_composite_loss_lengths.py`](../../../tests/test_metricgan_composite_loss_lengths.py)、[`test_loss_audit_fixes.py`](../../../tests/test_loss_audit_fixes.py) |
| DFN 频率边界、公式、短输入、mask 与图保留零 | [`test_deepfilternet_loss_boundaries.py`](../../../tests/test_deepfilternet_loss_boundaries.py)、[`test_deepfilternet_alpha_alignment.py`](../../../tests/test_deepfilternet_alpha_alignment.py)、[`test_core_losses_audit_regressions.py`](../../../tests/test_core_losses_audit_regressions.py) |

## 8. Speech Separation / PIT

`SSTrainer` 将 `[B,K,T]` 的 estimated/reference sources、`lengths:[B]`、`source_mask:[B,K]` 传给 `PITLoss`。这不同于单路 Composite 的 `enable_ss_undesired_energy` 开关。

- `PairwiseL1Loss` 产生 `[B,K_est,K_ref]` cost，每项是在有效 lengths 内的平均绝对误差；padding 不参与。
- `PairwiseNegSISDRLoss` 对有效样点分别去均值，投影到参考源方向，计算负 SI-SDR；`eps=1e-8` 保证能量分母和对数有下界。零参考的有限 cost 不等于有 source-presence 抑制梯度。
- `PITLoss` 要求 estimated/reference 同为 `[B,K,T]` 且形状一致，枚举 source 排列，对每个排列先取 K 路 cost 的均值，再选最小值，最后在 batch 上取均值。`compute()` 还返回 assignment 与重排后的 estimates；`forward()` 只返回标量 loss。
- 当前仅支持 `assignment=exhaustive`、`reduction=mean`，默认 `max_sources=4`；提供的 source_mask 必须全 true。可变 K、空槽监督或其他 assignment 算法均未接入。
- 离散排列选择本身不求导；梯度通过被选中的 pairwise cost 传回估计源。

实现见 [`pairwise.py`](../../../core/losses/pairwise.py)、[`pit.py`](../../../core/losses/pit.py)、[`permutation.py`](../../../core/losses/permutation.py)。动态源、early/dry targets 与 RIR 口径见 [多源仿真](../simulation/Multi_Source_Simulation.md)。
