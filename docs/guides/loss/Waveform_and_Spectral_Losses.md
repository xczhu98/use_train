# 波形与频谱 Loss 实现手册

> **这一页讲最基础的一类 Loss：直接比较波形或频谱。** 刚接触语音增强时，“损失”可以理解成“增强结果离目标差多少”的分数；这页的 Loss 要么在时域逐点比较波形，要么把波形变换到时频域后再比较。每小节给出注册名、签名、源码里实际运行的公式、它偏好什么、以及在哪些输入上会失去梯度。

读之前值得知道：

- **STFT / 频谱**：short-time Fourier transform，把波形切成短帧、逐帧做傅里叶变换得到的时频表示；比较频谱通常比逐点比较波形更贴近人耳对频率结构的敏感度。
- **SI-SNR / SI-SDR（尺度不变信噪比 / 信噪失真比）**：先把预测投影到目标方向上，再比较目标能量与残差能量的比例；对整体音量差不敏感，衡量的是波形结构像不像，而不是幅度准不准。本页 `SISNRLoss` 用的是带 `eps` 的稳定化近似投影。
- **clamp / floor**：把数值强行压到某个下界（例如 `max(x, 1e-8)`），用来避免除零和取对数出问题；副作用是低于该下界的差异不再产生梯度。
- **mask / gate**：在有效长度之外的 padding、或明确不合格的样本上跳过计算的机制。
- **reduction（归约）**：把逐样本、逐帧、逐频点的误差合成一个标量的方式（求和 / 全局平均 / 按有效元素平均）。不同 Loss 的权重口径不同，不能互相套用。
- **驻点**：梯度为零的输入点；出现正的 loss 不代表那里能把模型推走。

本文说明基础波形与 STFT/频谱 Loss，并补充 TSE 静音目标使用的 `AbsentTargetLoss`。公式中的常数、`clamp`、有效长度 mask、静音 gate 和 reduction 均以本仓实现为准；“偏好”描述优化目标施加的梯度倾向，不代表论文结果或本项目实测效果。

## 1. 共同契约与记号

### 1.1 注册与构建事实

基础章节中的类以类名注册；新增 `AbsentTargetLoss` 同样由 `core.losses.time_domain` 注册。`TAPLoss` / `AcousticLoss` 是同一实现的两个注册名；`CompositeLoss` 由 `build_loss` 特判构造。全仓库注册总数会随模型/损失扩展变化，应从当前 `LOSS_REGISTRY` 查询，不在本手册固定计数。注册名清单见 [Loss 文档中心](README.md#完整-registry-inventory)；YAML 结构、builder 注入与 Composite 组合的完整契约见 [Loss 配置指南](Loss_Configuration_Guide.md)，本页只描述各 Loss 自己的公式与运行边界。

单一 Loss 的配置形状为：

```yaml
loss:
  type: SISNRLoss
  args: {}
```

标准调用为：

```python
loss = loss_fn(enhanced, clean, lengths=None)
```

- `enhanced = \hat{s} \in \mathbb{R}^{B\times T}`：增强波形；
- `clean = s \in \mathbb{R}^{B\times T}`：目标波形；
- `lengths = (\ell_1,\ldots,\ell_B)`：可选 `[B]` 整数 Tensor；
- 返回值：零维标量 Tensor。

`build_loss({"type": ..., "strict": True}, ...)`（对应 YAML 的 `loss.strict: true`）构造的外围 wrapper 会检查两路波形同为 `[B,T]`、shape 相同、`lengths` 为 `[B]` 整数且 `1\le\ell_b\le T`，并要求输出是有限标量；`strict` 是 loss 配置字段，不是 `build_loss` 的关键字参数。直接实例化本文各类不会自动获得全部外围检查；以下均按合法输入契约描述。

### 1.2 有效样本与 centered STFT

若提供 `lengths`，记有效样本集合

$$
V_b=\{t\mid 0\le t<\ell_b\},\qquad n_b=|V_b|.
$$

时域 Loss 直接对 `V_b` 做 mask。本文 6 个谱 Loss 都从波形重新计算 STFT，不直接消费模型预测谱；它们使用 Hann 窗、`return_complex=True`、`center=True`，并在送入 `torch.stft` 时将两路波形转为 FP32。`freq_domain.py` 中的 4 类还在对应 STFT 前向上显式禁用 CUDA autocast；`spectral_distance.py` 中的 `SpectralConvergenceLoss` 与 `LogMagnitudeSTFTLoss` 只显式执行 `.float()`，没有“禁用 CUDA autocast”的契约，不能把前 4 类的 AMP 边界推广给这 2 类。

谱 Loss 通过 [`prepare_centered_stft_inputs`](../../../core/losses/utils.py#L27) 先清零有效边界外样本，再在每条样本自己的右边界做反射延展。对有效长度足以满足 PyTorch reflect padding 约束的输入，这可视作逐条 trim 后执行原生 `torch.stft(center=True, pad_mode="reflect")`；极短输入则是 helper 自定义扩展：按无端点的周期式反射继续映射，`length=1` 时复制唯一样本，不能称为与原生 `torch.stft` reflect 直接等价。单边实信号 STFT 的频率 bin 数为 `F_r=\lfloor N_r/2\rfloor+1`，有效帧数为

$$
K_r(\ell_b)=\max\!\left(1,
1+\left\lfloor
\frac{\ell_b+2\lfloor N_r/2\rfloor-N_r}{H_r}
\right\rfloor
\right),
$$

其中 `N_r=n_fft`、`H_r=hop_length`；帧 mask `M_{b,r,k}=1[k<K_r(\ell_b)]` 沿频率维广播。该路径支持短于 `n_fft//2`、乃至只有一个有效样本的输入。另需注意 periodic Hann 窗的端点为零：直接实例化这些 Loss，或在 `loss.args` 显式覆盖几何时，若 `hop_length>=win_length`，帧间可能存在不受任何非零窗系数覆盖的样点；只落在这些样点上的非零 residual 可得到零 loss 和零梯度。安全配置应满足 `hop_length<win_length<=n_fft`，或让几何经过顶层 [`resolve_stft_config`](../../../core/stft.py#L284-L298) 的等价校验；后者还执行更完整的 STFT/iSTFT NOLA 检查。相关边界由 [`test_freq_domain_centered_stft_losses.py`](../../../tests/test_freq_domain_centered_stft_losses.py)、[`test_freq_domain_loss_boundaries.py`](../../../tests/test_freq_domain_loss_boundaries.py) 和 [`test_multi_res_stft_loss.py`](../../../tests/test_multi_res_stft_loss.py) 覆盖。

### 1.3 Trainer、gate 与 Composite 运行边界

训练入口通过 [`build_loss`](../../../core/losses/build.py#L421) 构造单项或组合 Loss；SE Trainer 在训练和验证中都以 `loss_fn(enhanced, clean, lengths, **kwargs)` 调用，并按 Loss capability 补充 pure metadata、undesired 波形或 direct-spectrum 参数。本文这些类都是 waveform-input Loss：Trainer 先得到/重建 `enhanced` 波形，它们再按各自公式计算时域量或重新执行 STFT；标量 loss 随后进入 AMP/梯度累积/反向传播，类内 mask、gate、reduction 和零梯度边界不会由 Trainer 自动改写。

- `SISNRLoss` 显式声明 `REQUIRES_SPEECH_REFERENCE`（`PESQLoss`/`STOILoss`/`FWNegSISDRLoss` 同样声明；类名兼容集合只是旧实现的 fallback）：传入 `is_pure_noise`、`is_pure_echo` 或 `is_pure_echo_noise` 时，只对 non-pure 子集计算；类内还会再执行 centered-clean energy gate。
- `SIL1Loss` 有自己的近静音 gate，但不属于 Composite 的 speech-reference pure gate 集合。
- 其余本文 Loss 不因 pure metadata 自动跳过，默认对 full batch 计算。它们各自的 zero-target 行为必须按下文区分。
- `CompositeLoss.enable_ss_undesired_energy` 是组合器额外施加的两路监督语义，不是本文单个 Loss 类的公式。启用时，普通分量按有效长度内能量比例混合 desired 与 undesired 两次调用；组合器再按静态 component weight 求和，并处理 speech-reference 子集的 DDP 全局 mean 质量校正。完整路由见 [`composite.py`](../../../core/losses/composite.py#L377)。

自研边界必须按对象理解：`SIL1WavSeTrainLoss`、`MRSIRelCspecL2Loss`、`MRSpecSuppressPenaltyLoss` 三个实现，以及**一份由这三项组成、且三项均设置 component-level `enable_ss_undesired_energy: true` 的完整三项组合配方**，均为用户自研；不得归因于 `se_train`、外部论文、CAGCRN、DeepVQE 或 MBRCL。此声明中的 SIL1 只指 `SIL1WavSeTrainLoss`，不扩大到 `SIL1Loss` 或 `MultiResFreqLoss`。

[`docs/paper/loss`](../../paper/loss/README.md) 的三篇 canonical 保持严格一对一：TAPLoss canonical 只对应 `TAPLoss` / `AcousticLoss`，Loud-loss canonical 只对应 Loud-loss，MBRCL canonical 只对应 `MultiBandRelativeContrastiveLoss`；它们都不为本文其他 Loss、SS undesired-energy 机制或上述用户自研三项组合配方背书。

## 2. 波形 Loss

### 2.1 `SIL1WavSeTrainLoss`

**注册名与签名**

- registry：`SIL1WavSeTrainLoss`；无别名。
- 构造：`SIL1WavSeTrainLoss()`，无可配置参数。
- 前向：`forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar`。

**精确公式与归约**

令

$$
d_b=\frac{1}{n_b}\sum_{t\in V_b}|s_{b,t}-\hat{s}_{b,t}|,
\qquad
r_b=\frac{1}{n_b}\sum_{t\in V_b}|s_{b,t}|.
$$

实现中的 `n_b` 最低 clamp 到 `1.0`。逐样本损失为

$$
L_b=
\begin{cases}
\displaystyle \frac{10d_b}{r_b+10^{-8}}, & r_b>10^{-12},\\[6pt]
\displaystyle \frac{10d_b}{0.1+10^{-8}}, & r_b\le 10^{-12},
\end{cases}
\qquad
L=\frac1B\sum_{b=1}^{B}L_b.
$$

先对每条样本的有效点求 mean，再做 batch mean；不同有效长度的样本在最终 batch 中权重相同。`lengths=None` 时 `V_b` 是完整 `T`。

**原理、意义与偏好**

它用目标平均绝对幅度归一化波形 L1，使不同整体电平的非静音样本具有更接近的相对误差尺度。这里的“SI”是幅度比归一化，不会像投影式 SI-SNR 那样先求缩放；本项目 `SISNRLoss` 的系数带 `eps`，具体是稳定化投影近似而非严格最优缩放。L1 对正负误差、过估计和欠估计对称，偏好逐样点波形接近，同时比 L2 对少数大误差更不敏感。

**边界与限制**

- `r_b\le10^{-12}` 时样本不会被删除，而改用固定分母 `0.1+10^{-8}`；因此 zero target 仍会惩罚非零 residual。
- `10^{-8}` 使普通分支只近似严格尺度不变；`10^{-12}` gate 与固定 `0.1` fallback 进一步打破极低能区的尺度不变性，这是有意的稳定保护。
- 不去均值、不做相位或频带建模，也不区分语音与 undesired residual。
- 它不受 Composite pure metadata 自动门控；若作为 SS 分量启用，会由组合器分别对 desired/undesired 两路调用并做外层能量混合。

**实现边界**

`SIL1WavSeTrainLoss` 实现本身是用户自研；完整配方只指第 1.3 节定义的那一份三项组合，不能把单项扩写成另一份配方，也不得归因于 `se_train`、论文、CAGCRN、DeepVQE 或 MBRCL；公式以当前 [`time_domain.py`](../../../core/losses/time_domain.py#L16) 为定义。

### 2.2 `SIL1Loss`

**注册名、签名与默认值**

- registry：`SIL1Loss`；无别名。
- 构造：`SIL1Loss()`；静音阈值是类常量 `EPS=10^{-4}`，不是构造参数。
- 前向：`forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar`。

**精确公式、gate 与归约**

`d_b`、`r_b` 与上一节相同。实现先计算

$$
q_b=\frac{10d_b}{\max(r_b,10^{-4})},
\qquad
g_b=\mathbf 1[r_b>10^{-4}],
$$

再返回

$$
L=\frac{\sum_b g_bq_b}{\max(\sum_b g_b,1)}.
$$

这是“仅对有效非静音样本做 mean”，不是先用零填充再除以固定 batch size。全 batch 都无效时，分子通过 `q_b * 0` 保持计算图连接，返回零损失与零梯度。

**原理、意义与偏好**

普通有效样本上，它与 `SIL1WavSeTrainLoss` 一样以目标平均绝对幅度归一化 L1，但采用“跳过近静音 target”而非“固定分母继续抑制”的策略。它适合只希望波形相对误差由有声参考决定、不希望静音 target 主导该分量的场景。

**边界与限制**

- gate 是严格 `r_b>10^{-4}`；恰等于阈值也被排除。
- 对 `r_b\le10^{-4}` 的样本，无论 enhanced residual 多大，该分量都不给抑制梯度；若任务含 target-absent/pure 样本，必须另配有定义的 residual suppression 目标。
- 它没有声明 Composite speech-reference capability，因此 pure metadata 本身不裁剪它；实际跳过仅由类内 `r_b` gate 决定。
- `10^{-4}` floor 使阈值附近并非严格尺度不变；不去均值、不做最优尺度投影。

**实现边界**

`SIL1Loss` 以当前 [`time_domain.py`](../../../core/losses/time_domain.py#L41) 为定义；`se_train` 仅是源码历史标签，且用户自研声明只覆盖 `SIL1WavSeTrainLoss`，不覆盖本类。

### 2.3 `SISNRLoss`

**注册名、签名与默认值**

- registry：`SISNRLoss`；无别名。
- 构造：`SISNRLoss()`；固定 `EPS=10^{-8}`，SI-SNR 上限固定为 `80 dB`。
- 前向：`forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar`。

**精确公式、clamp、gate 与归约**

先只在有效样本上去均值。记

$$
\bar{s}_b=\frac1{n_b}\sum_{t\in V_b}s_{b,t},\quad
\bar{\hat{s}}_b=\frac1{n_b}\sum_{t\in V_b}\hat{s}_{b,t},
$$

$$
s'_b=(s_b-\bar{s}_b)\odot m_b,\qquad
\hat{s}'_b=(\hat{s}_b-\bar{\hat{s}}_b)\odot m_b,
$$

其中 `m_b` 是 sample mask；`lengths=None` 时在完整 `T` 上去均值。然后

$$
E_b=\|s'_b\|_2^2,
\qquad
g_b=\mathbf 1[E_b>10^{-8}],
$$

$$
s_{\mathrm{target},b}
=\frac{\langle s'_b,\hat{s}'_b\rangle}{E_b+10^{-8}}s'_b,
\qquad
e_b=\hat{s}'_b-s_{\mathrm{target},b},
$$

$$
\mathrm{SI\!\mbox{-}SNR}_b
=\min\!\left(
10\log_{10}
\frac{\max(\|s_{\mathrm{target},b}\|_2^2,10^{-8})}
     {\max(\|e_b\|_2^2,10^{-8})},
80
\right).
$$

最终返回负值供最小化：

$$
L=-\frac{\sum_b g_b\,\mathrm{SI\!\mbox{-}SNR}_b}
          {\max(\sum_b g_b,1)}.
$$

**原理、意义与偏好**

实现用带 `10^{-8}` 分母稳定项的投影系数，将 enhanced 中近似与零均值 clean 共线的部分视为目标，其余视为误差；最小化负 SI-SNR 等价于提高该目标项与残差的能量比。由于系数分母是 `E_b+10^{-8}` 而不是 `E_b`，它是稳定化投影近似，不是严格最优正交投影。它对整体增益差较宽容，偏好 waveform 的方向/结构一致，而不是绝对幅值完全一致。

**边界与限制**

- `E_b\le10^{-8}` 的 centered target 被排除；全无效 batch 返回 graph-connected zero。Composite pure metadata 还会在类调用前排除 pure 样本。
- 投影系数、分子与分母都带固定稳定化处理：系数分母加 `10^{-8}`，两路能量各自 clamp 到 `10^{-8}`；高端再 clamp 到 `80 dB`，低端没有对称下限。
- 对非静音 clean，若 enhanced 精确全零，投影能量和误差能量都被 floor 到 `10^{-8}`，该样本 SI-SNR 为 `0 dB`，loss 约为 `-0.0`；由于投影、误差与能量 floor 在该原点的一阶导均为零，enhanced 梯度为零。这是驻点，不是显式 residual suppression 目标。有效区内任意常量 enhanced 去均值后也精确为零，因此落在同一驻点。
- 未经最终上限 clamp 的 SI-SNR 一旦超过 `80 dB`，`clamp(max=80)` 会形成 loss 为 `-80` 的零梯度平台；此时继续改善波形也不再由该分量提供梯度。
- 去均值意味着常量 DC 不计入目标结构。固定 eps 与 80 dB ceiling 使有限精度实现不具备数学上的完全尺度不变性，也使投影不再严格正交。
- 该类名为 `SISNRLoss`；不要把它与不去均值、不同 eps 位置且不乘 `10` 的 `GTCRNLogRatioLoss` 混同。

**必要参考与实现边界**

投影式 SI-SNR 可参考 Le Roux 等人 [ICASSP 2019](https://doi.org/10.1109/ICASSP.2019.8683855)，但本地 eps、gate、有效长度归约、`80 dB` ceiling 及梯度平台均以当前 [`time_domain.py`](../../../core/losses/time_domain.py#L69) 为定义。

### 2.4 `L1WavLoss`

**注册名与签名**

- registry：`L1WavLoss`；无别名。
- 构造：`L1WavLoss()`，无可配置参数。
- 前向：`forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar`。

**精确公式与归约**

提供 `lengths` 时：

$$
L=\frac{\sum_b\sum_{t\in V_b}|\hat{s}_{b,t}-s_{b,t}|}
        {\max(\sum_b n_b,1)}.
$$

不提供时：

$$
L=\frac1{BT}\sum_{b,t}|\hat{s}_{b,t}-s_{b,t}|.
$$

与两个 SI-L1 不同，这里对整个 batch 的所有有效 sample 做一次全局 mean；有效长度较长的样本贡献更多元素权重。

**原理、意义与偏好**

它直接优化时域 MAE，保留绝对电平语义；对过估计与欠估计对称，梯度幅值不像 L2 那样随误差无限增大。适合简单波形保真项，也常作为组合目标中的绝对幅值锚。

**边界与限制**

- zero target 上会直接惩罚 enhanced residual 的平均绝对幅度。
- 不做尺度归一化：相同相对误差在高电平样本上产生更大绝对损失。
- 不去均值，不单独建模频率、相位或感知权重。
- 不受 Composite pure metadata 自动门控。

**实现边界**

这是标准 MAE 的项目封装，不对应特定论文；数值行为以当前 [`time_domain.py`](../../../core/losses/time_domain.py#L111) 为定义。

### 2.5 `AbsentTargetLoss`

该 Loss 为 TSE 的 noise-only / target-absent 样本提供输出抑制梯度。有效区域内 target 均方能量不超过 `1e-10` 时，计算该样本 enhanced 的均方能量，再按 absent 样本数取均值；lengths 外 padding 不参与。全 present batch 返回连接计算图的零，保持 DDP 各 rank 的一致训练流程。

`SISNRLoss` 会对零能量 target 返回零贡献，因此不能独自训练目标缺席场景。现役动态 TSE 模板使用：

```yaml
loss:
  type: CompositeLoss
  components:
    - type: SISNRLoss
      weight: 1.0
    - type: AbsentTargetLoss
      weight: 1.0
```

动态 TSE 只要启用任意一路无目标场景（`data.train.simulation_cfg` 中 `prob_interference_only` / `prob_noise_only` / `prob_interference_noise_only` 大于 0），TSE entry 就要求 `AbsentTargetLoss` 或 `L1WavLoss` 至少有一项权重非零。它不属于 SS 的 PIT source-presence 扩展；固定 K SS 仍遵守其独立输出合同。实现见 [`time_domain.py`](../../../core/losses/time_domain.py)，对应数据语义见 [多源仿真](../simulation/Multi_Source_Simulation.md)。

## 3. 多分辨率频谱 Loss

### 3.1 `MultiResSTFTLoss`

**注册名、签名与默认值**

- registry：`MultiResSTFTLoss`；无别名。
- 构造：

  ```python
  MultiResSTFTLoss(
      fft_sizes=(2048, 1024, 512),
      hop_sizes=(240, 120, 50),
      win_lengths=(1200, 600, 240),
  )
  ```

- 前向：`forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar`。
- 三个 schedule 按相同 index 配成 `(n_fft, hop_length, win_length)`；类参数名不是单分辨率 `n_fft/hop_length/win_length`，因此 builder 不会用顶层 STFT 覆盖该独立多尺度 schedule。

**精确公式、clamp、gate 与归约**

对第 `r` 个分辨率，记 raw 幅度谱

$$
X^0_r=|\operatorname{STFT}_r(\hat{s})|,
\qquad
Y^0_r=|\operatorname{STFT}_r(s)|,
$$

数值谱与 raw target gate 为

$$
X_r=\max(X^0_r,10^{-8}),\qquad
Y_r=\max(Y^0_r,10^{-8}),
$$

$$
g_{b,r}=\mathbf 1\!\left[
\|Y^0_{b,r}\odot M_{b,r}\|_F>10^{-8}
\right].
$$

每条样本的有效频谱元素数为

$$
D_{b,r}=\max\!\left(F_r\sum_k M_{b,r,k},1\right),
\qquad F_r=\left\lfloor N_r/2\right\rfloor+1.
$$

log-magnitude 项与 spectral convergence 项为

$$
L^{\log}_{b,r}
=\frac{\sum_{f,k}M_{b,r,k}
|\log X_{b,r,f,k}-\log Y_{b,r,f,k}|}{D_{b,r}},
$$

$$
A_{b,r}=\|(Y_{b,r}-X_{b,r})\odot M_{b,r}\|_F,
$$

$$
\widetilde L^{\mathrm{SC}}_{b,r}=
\begin{cases}
\displaystyle
\frac{A_{b,r}}{\|Y_{b,r}\odot M_{b,r}\|_F}, & g_{b,r}=1,\\[7pt]
A_{b,r}\cdot0, & g_{b,r}=0.
\end{cases}
$$

最终先在每个分辨率内做 batch mean，再对 `R` 个分辨率做算术平均：

$$
L=\frac1R\sum_{r=1}^{R}
\frac1B\sum_{b=1}^{B}
\left(\widetilde L^{\mathrm{SC}}_{b,r}+L^{\log}_{b,r}\right).
$$

**原理、意义与偏好**

SC 用 target 谱范数归一化整体幅度形状误差；log-magnitude L1 压缩动态范围，使低能频点相对线性谱误差获得更大关注。多个窗长同时约束长时低频结构与短时瞬态。该类只比较幅度，不直接惩罚复谱相位差。

**驻点、zero-target 行为与限制**

- raw target 谱范数 `<=10^{-8}` 时，**仅关闭本分辨率的 SC 分支**，并用 `A*0` 保持图连接；原本就属于 MR-STFT 的 **一次** log-magnitude 项仍保留。因此 zero target + 非零 enhanced 只有在至少一个相关谱幅度高于 `10^{-8}` floor 时才通常有 residual suppression 梯度；全部相关幅度落在 floor 内时形成无抑制梯度的谱死区。
- 该谱死区不只影响 zero target。对非零 target，若 enhanced 精确为零或其估计幅度都落在 `10^{-8}` floor，loss 可因 target 与 floored estimate 不同而为正，但 `abs(STFT(enhanced))` 在零点以及后续 clamp 的 floor 区域对 enhanced 不提供一阶梯度，SC 与 log 两个分支都可为零梯度。
- target 与 enhanced 都为零时，clamped log 谱相同，SC 与 log 项均为 graph-connected zero。
- 这一行为与独立 `SpectralConvergenceLoss` 不同：后者在 zero/near-zero target 时，把整个 standalone SC 输出替换成另一套 log-magnitude fallback；详见第 4.1 节；也不同于含未 clamp 复谱误差项的 `MultiResFreqLoss` 和 `MagnitudePlusComplexL1Loss`，后两者不因幅度 floor 而整体失去该更新路径。
- 每项 schedule 必须是 exact positive integer；三个序列必须非空且等长；每个 `win_length<=n_fft`。构造器虽允许 `hop_length>=win_length`，但这会触发第 1.2 节的 Hann 覆盖盲点，不能因这里只做 STFT 就视为安全几何。
- `10^{-8}` 谱 floor 会改变极低能区并产生上述死区；SC 和 log 项无额外可调权重，分辨率也等权平均。

**必要参考与实现边界**

`SC + log-magnitude` 的多分辨率思路可参考 [Parallel WaveGAN](https://doi.org/10.1109/ICASSP40776.2020.9053795)，但本地 gate、floor、mask、极短输入处理、默认几何与 reduction 均以当前 [`freq_domain.py`](../../../core/losses/freq_domain.py#L21) 为定义。

### 3.2 `MultiResL1SpecLoss`

**注册名、签名与默认值**

- registry：`MultiResL1SpecLoss`；无别名。
- 构造：

  ```python
  MultiResL1SpecLoss(
      window_sz=(256, 512, 768, 1024),
      hop_sz=None,                 # 逐项 window_sz // 2
      eps=1e-6,
      time_domain_weight=0.5,
      normalize_variance=True,
      gama=1.0,
  )
  ```

  参数名按源码拼写为 `gama`，不是 `gamma`。

- 前向：`forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar`。
- 每个尺度固定 `n_fft=win_length=window_sz[r]`、Hann、`center=True`。

**方差归一化与尺度投影**

输入先在 `V_b` 外清零。若 `normalize_variance=True`，有显式 `lengths` 时对有效样本计算 population variance：

$$
\mu_b(x)=\frac1{n_b}\sum_{t\in V_b}x_{b,t},
\qquad
v_b(x)=\frac1{n_b}\sum_{t\in V_b}(x_{b,t}-\mu_b(x))^2+\varepsilon.
$$

`lengths=None` 时使用 `torch.var(..., unbiased=False)+eps`，即完整波形 population variance。归一化分母不是无条件 `\sqrt v`，而是

$$
d_b(x)=
\begin{cases}
\sqrt{v_b(x)}, & v_b(x)>\varepsilon,\\
1, & v_b(x)\le\varepsilon.
\end{cases}
$$

因此精确常量波形的方差为 `eps` 时不被除以 `sqrt(eps)`。令归一化后的 estimate/target 为 `e_b,c_b`；关闭方差归一化时直接取 masked 波形。随后用带 `eps` 的分母求 estimate-to-reference 缩放：

$$
\alpha_b=\frac{\sum_t e_{b,t}c_{b,t}}
                 {\sum_t e_{b,t}^2+\varepsilon},
\qquad
\tilde e_b=\alpha_be_b.
$$

该系数是稳定化/正则化的 LS 投影近似，不是无 `eps` 时的严格最优正交投影；`alpha_b` 没有非负约束。

**精确频谱与总归约**

时域项是 sum 而非 mean：

$$
L^{\mathrm{time}}_b=
\sum_{t\in V_b}|\tilde e_{b,t}-c_{b,t}|.
$$

对第 `r` 个尺度，令

$$
A_{b,r}=|\operatorname{STFT}_r(\tilde e_b)|,
\qquad C_{b,r}=|\operatorname{STFT}_r(c_b)|.
$$

若 `gama == 1.0`，直接使用 `A,C`；否则精确替换为

$$
A_{b,r}\leftarrow(A_{b,r}+\varepsilon)^{\texttt{gama}},
\qquad
C_{b,r}\leftarrow(C_{b,r}+\varepsilon)^{\texttt{gama}}.
$$

每尺度谱项也是有效 bin 的 sum：

$$
L^{\mathrm{spec}}_{b,r}
=\sum_{f,k}M_{b,r,k}|A_{b,r,f,k}-C_{b,r,f,k}|,
\qquad
L^{\mathrm{spec}}_b=\frac1R\sum_rL^{\mathrm{spec}}_{b,r}.
$$

最终：

$$
L=\frac1B\sum_b
\left[
\lambda L^{\mathrm{time}}_b+(1-\lambda)L^{\mathrm{spec}}_b
\right],
\qquad \lambda=\texttt{time\_domain\_weight}.
$$

**原理、意义与偏好**

方差归一化与 estimate-to-reference 的正则化 LS scaling 降低整体电平差对目标的影响；时域 L1 锚定波形，多窗幅度 L1 同时约束不同时间尺度。`gama<1` 时压缩幅度动态范围，增加低能谱相对权重；默认 `gama=1` 不压缩。

**边界与限制**

- `window_sz`/`hop_sz` 必须非空、等长且均为 exact positive integer。`eps`/`gama` 只拒绝类型不属于 `int`/`float` 或值 `<=0` 的输入；没有 finite 检查，因此 `NaN`/`Inf` 会通过构造。`time_domain_weight` 完全没有 finite 或 `[0,1]` 检查；越界会令一项系数为负，非有限值会传播，不应依赖这些未约束行为。
- 时域与谱域都使用 sum，故 loss 尺度随有效样本数、频率 bin 数和帧数变化；只有分辨率数与 batch 做 mean。不同长度样本在最终 batch 仍等权，但单条较长样本的 raw loss 通常更大。
- clean 为零时，投影分子为零，`alpha=0`，于是任意非零 estimate 都被缩为零；该 loss 返回零，**不提供 zero-target residual suppression**。
- 对非零 clean，若 estimate 精确全零，则 `alpha=0`，时域与谱域误差可为正；但 `\tilde e=\alpha e` 在 `e=0` 原点的一阶导为零，因此 enhanced 梯度为零。这同样是驻点，不能把正 loss 解读为该点存在有效更新方向。
- `eps` 使尺度不变性和 LS 投影都只是数值近似；该类不比较复谱相位。

**必要参考与实现边界**

该类借鉴 ESPnet2 同名目标，但本地默认值、方差分支、`gama`、mask、短输入处理和固定 sum→batch mean 均不同，数值行为以当前 [`freq_domain.py`](../../../core/losses/freq_domain.py#L147) 为定义。

### 3.3 `MultiResFreqLoss`

**注册名、签名与默认值**

- registry：`MultiResFreqLoss`；无别名。
- 构造：`MultiResFreqLoss(fft_sizes=(64, 128, 256, 512, 1024, 2048))`。
- 每尺度固定 `hop_length=n_fft//2`、`win_length=n_fft`、Hann、`center=True`。
- 前向：`forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar`。

**精确公式与全局归约**

第 `r` 个尺度记复谱 `E_r=STFT_r(enhanced)`、`C_r=STFT_r(clean)`，floored 幅度为

$$
A_r=\max(|E_r|,10^{-8}),\qquad
R_r=\max(|C_r|,10^{-8}).
$$

令该尺度 `N_r=n_fft`，则单边频率 bin 数与整个 batch 的有效复谱 bin 数分别为

$$
F_r=\left\lfloor N_r/2\right\rfloor+1,
\qquad
D_r=\max\!\left(F_r\sum_{b,k}M_{b,r,k},1\right).
$$

复谱项对 real/imag 两个分量做 MSE：

$$
L^{\mathrm{complex}}_r
=\frac{1}{2D_r}
\sum_{b,f,k}M_{b,r,k}
\left[(\Re E_r-\Re C_r)^2+(\Im E_r-\Im C_r)^2\right].
$$

相对幅度项为

$$
L^{\mathrm{rel}}_r
=\frac1{D_r}\sum_{b,f,k}M_{b,r,k}
\left(
\frac{A_{b,r,f,k}-R_{b,r,f,k}}
     {A_{b,r,f,k}+R_{b,r,f,k}+10^{-7}}
\right)^2.
$$

最终等权平均：

$$
L=\frac1R\sum_r
\left(L^{\mathrm{complex}}_r+L^{\mathrm{rel}}_r\right).
$$

这里每尺度先把 batch、频率和有效帧合并做全局 mean；有效帧更多的样本具有更大的元素权重，不是逐样本 mean。

**原理、意义与偏好**

复谱 MSE 同时约束幅度与相位；交换 estimate/target 不改变该项，因此它对两路复谱是对称距离。相对幅度项用两路幅度和归一化，使不同绝对能量的 bin 更可比；其代数式交换 `A_r,R_r` 也不变，但训练时 target 固定、只对 estimate 求梯度，分母同样依赖 estimate，所以对固定 target 的过估计与欠估计优化偏好并不对称，不能仅因外层平方就称其对称。多分辨率提供不同时间—频率折中。

**边界与限制**

- 幅度先 clamp 到 `10^{-8}`，相对项分母再加 `10^{-7}`；复谱 MSE 本身不 clamp real/imag。相对幅度低于 floor 的区域不可分辨，会形成该分支的谱死区。
- clean 与 enhanced 全零时两项均为零；在固定 `hop_length=n_fft//2<win_length` 的非零 Hann 覆盖下，clean 为零、enhanced 非零时，复谱 MSE 会抑制 residual，相对幅度项只在 enhanced 幅度高于 `10^{-8}` floor 的区域提供梯度。复谱 MSE 未 clamp，故非零 clean + 零 enhanced 仍有更新梯度。
- 相对项是有界化趋势，不是 spectral convergence，也不是 SI-SNR；复谱项保留绝对尺度，因此整体不具尺度不变性。
- `fft_sizes` 必须是非空 exact positive integer 序列，但当前仅检查 `>0`。因此 `n_fft=1` 可通过构造，却因固定 half-hop 得到 `hop_length=0` 而在 forward 的 STFT 处失败；实际可执行的每项 `n_fft` 必须 `>=2`。分辨率等权，不能分别配置两项权重。

**实现边界**

`MultiResFreqLoss` 不在用户自研声明内，也不归因于 MBRCL；其组合、默认尺度、clamp、mask、reduction 和短输入行为以当前 [`freq_domain.py`](../../../core/losses/freq_domain.py#L269) 为定义。

## 4. 单分辨率频谱 Loss

### 4.1 `SpectralConvergenceLoss`

**注册名、签名与默认值**

- registry：`SpectralConvergenceLoss`；无别名。
- 构造：`SpectralConvergenceLoss(n_fft=1024, hop_length=256, win_length=1024)`。
- 固定 Hann、`center=True`；builder 可从顶层 STFT 注入这三个同名几何参数，但不能注入 `window`、`center` 或 `normalized`，因为构造函数不接受它们。
- 前向：`forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar`。

**普通 target 的精确公式**

记 raw 幅度谱 `X^0=|STFT(enhanced)|`、`Y^0=|STFT(clean)|`，以及

$$
X=\max(X^0,10^{-7}),\qquad
Y=\max(Y^0,10^{-7}),
$$

$$
g_b=\mathbf1[\|Y^0_b\odot M_b\|_F>10^{-8}].
$$

当 `g_b=1`：

$$
L_b^{\mathrm{SC}}
=\frac{\|(Y_b-X_b)\odot M_b\|_F}
       {\|Y_b\odot M_b\|_F}.
$$

分子和分母都使用先 clamp 到 `10^{-7}` 的谱；`10^{-8}` 只用于 raw target gate，不作为普通分支的分母加数或 clamp。

**zero/near-zero target fallback**

当 `g_b=0`，整个逐样本输出替换为固定 eps 的有效帧 log-magnitude L1：

$$
L_b^{\mathrm{fallback}}
=\frac{1}{D_b}\sum_{f,k}M_{b,k}
\left|
\log(X_{b,f,k}+10^{-8})-
\log(Y_{b,f,k}+10^{-8})
\right|,
$$

$$
D_b=\max\!\left(F\sum_kM_{b,k},1\right).
$$

最终：

$$
L=\frac1B\sum_b
\begin{cases}
L_b^{\mathrm{SC}}, & g_b=1,\\
L_b^{\mathrm{fallback}}, & g_b=0.
\end{cases}
$$

**与 `MultiResSTFTLoss` 的 zero-target 差异**

两者不能写成同一种行为：

1. `MultiResSTFTLoss` 的每尺度目标本来就是 `SC + log-mag`；zero target 时只令 **SC 分支为 graph-connected zero**，原有 log-mag 保留一次。
2. 独立 `SpectralConvergenceLoss` 的普通目标只有 SC；zero/near-zero target 时将 **整个 standalone 输出改为 log-mag fallback**，不是返回零 SC。
3. 两套 log 公式的 floor 也不同：MR-STFT 先将幅度 clamp 到 `10^{-8}` 后直接取 log；独立类先 clamp 到 `10^{-7}`，再在 log 内额外加固定 `10^{-8}`。

因此它们在 zero target + 非零 enhanced 且 residual 谱至少有幅度越过各自 floor 时提供抑制信号，但数值定义、分支角色和谱死区阈值不同；若相关 enhanced 谱幅度全部不超过 floor，clamp 会令对应 log 路径对 residual 无梯度。

**原理、意义与限制**

SC 以 target 谱 Frobenius 范数归一化整体幅度误差，偏好全局谱形收敛；它不直接比较复相位。raw target 太小时标准 SC 无定义或病态，fallback 改用 log 幅度抑制 residual，但固定 `10^{-7}` 谱 floor 带来上述死区。该死区同样存在于普通非零 target 分支：enhanced 精确为零或估计幅度落在 floor 时，SC 可为正而 enhanced 梯度为零。完美重建时 `vector_norm` 给出有限零梯度。不同于此纯幅度路径，`MultiResFreqLoss` 与 `MagnitudePlusComplexL1Loss` 含未 clamp 的复谱误差项，不会仅因估计幅度落入 floor 而整体失去梯度。该类构造函数不验证 STFT 几何；`win_length>n_fft` 等组合会在窗口或 STFT 调用处报错，`hop_length>=win_length` 还可能静默产生第 1.2 节的 Hann 覆盖盲点。

**必要参考与实现边界**

SC 基础形式可参考 [Arik、Jun 与 Diamos](https://doi.org/10.1109/LSP.2018.2880284)，但本地 raw-target gate、整项 log fallback、两层 floor、mask、reduction 和短输入行为以当前 [`spectral_distance.py`](../../../core/losses/spectral_distance.py#L64) 为定义。

### 4.2 `LogMagnitudeSTFTLoss`

**注册名、签名与默认值**

- registry：`LogMagnitudeSTFTLoss`；无别名。
- 构造：`LogMagnitudeSTFTLoss(n_fft=1024, hop_length=256, win_length=1024, eps=1e-8)`。
- 固定 Hann、`center=True`；builder 可注入三个同名几何参数，不能注入窗类型或 `center`。
- 前向：`forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar`。

**精确公式与归约**

先固定执行

$$
X=\max(|STFT(\hat{s})|,10^{-7}),
\qquad
Y=\max(|STFT(s)|,10^{-7}).
$$

然后对每条样本计算

$$
L_b=\frac{1}{D_b}\sum_{f,k}M_{b,k}
\left|\log(X_{b,f,k}+\varepsilon)-
          \log(Y_{b,f,k}+\varepsilon)\right|,
$$

$$
D_b=\max\!\left(F\sum_kM_{b,k},1\right),
\qquad
L=\frac1B\sum_bL_b.
$$

即先对每条样本的有效频率—帧元素做 mean，再做 batch mean。

**原理、意义与偏好**

log 压缩减弱高能 bin 对线性谱距离的支配，使相对谱包络与低能细节获得更多权重；L1 对 log 域正负偏差对称。它只比较幅度，不约束复相位，也不按 Mel/equal-loudness 频带加权。

**边界与限制**

- 有两层稳定项：幅度固定 clamp 到 `10^{-7}`，随后 log 参数再加可配置 `eps`；不要把公式简写成只加一个 eps。
- 默认 zero target + 非零 enhanced 在至少一个相关谱幅度高于 `10^{-7}` floor 时会产生有限正损失并通常推动 residual 幅度下降；若全部相关幅度都落在 floor 内，则该分支对 residual 无梯度。两路全零时为零。
- 谱死区不只限于 zero target：对非零 target，enhanced 精确为零或估计幅度落在 `10^{-7}` floor 时，log-magnitude loss 可为正而 enhanced 梯度为零。`MultiResFreqLoss` 与 `MagnitudePlusComplexL1Loss` 含未 clamp 复谱误差项，不能把这个纯幅度驻点结论推广给它们。
- `eps` 当前无构造期正值/有限性校验；负值可能导致 log 非有限，应保持正数。
- 固定 floor 使低于 `10^{-7}` 的谱幅度不可分辨并形成谱死区；该目标不具整体尺度不变性。直接构造或显式 `loss.args` 还需避免 `hop_length>=win_length` 的 Hann 覆盖盲点。

**必要参考与实现边界**

Log-magnitude L1 可参考 [Arik、Jun 与 Diamos](https://doi.org/10.1109/LSP.2018.2880284)，但本地两层稳定项、mask、reduction 与短输入行为以当前 [`spectral_distance.py`](../../../core/losses/spectral_distance.py#L126) 为定义；本类不是采用 log-power、Mel 子带 MSE 和 equal-loudness 加权的 Loud-loss。

### 4.3 `MagnitudePlusComplexL1Loss`

**注册名、签名与默认值**

- registry：`MagnitudePlusComplexL1Loss`；无别名。
- 构造：`MagnitudePlusComplexL1Loss(n_fft=512, hop_length=256, win_length=512)`。
- 固定 Hann、`center=True`；builder 可从顶层 STFT 注入三个同名几何参数，但不会注入 window/center/normalized。
- 前向：`forward(enhanced: [B,T], clean: [B,T], lengths: [B] | None = None) -> scalar`。

**精确公式与全局归约**

记 `S=STFT(clean)`、`\hat S=STFT(enhanced)`，逐 bin 误差为

$$
e_{b,f,k}
=\big||S_{b,f,k}|-|\hat S_{b,f,k}|\big|
 +|S_{b,f,k}-\hat S_{b,f,k}|.
$$

第二项 `|S-\hat S|` 是复差的模，不是 `|\Delta\Re|+|\Delta\Im|`。令整个 batch 的有效 bin 数

$$
N=\max\!\left(F\sum_{b,k}M_{b,k},1\right),
$$

则

$$
L=\frac1N\sum_{b,f,k}M_{b,k}e_{b,f,k}.
$$

它把 batch、频率与有效帧合并做一次全局 mean；不是逐样本 mean。

**原理、意义与偏好**

幅度 L1 明确约束谱能量包络，复谱 L1 同时对幅度和相位不一致敏感；二者等权相加。相较只看幅度的损失，它更偏好可重建的复谱结构；相较 real/imag 独立 L1，复差模对复平面方向具有旋转对称性。

**边界与限制**

- 没有谱 clamp 或 eps；复谱模在精确零点采用 PyTorch 的有限次梯度语义。
- 在 Hann 非零覆盖到的样点上，zero target + 非零 enhanced 时，两项都会惩罚 residual；两路全零时为零。复谱项未 clamp，因此非零 target + 零 enhanced 不属于纯幅度 floor 导致的正 loss 零梯度驻点。
- 使用未压缩波形 STFT，绝对幅度大的 bin 贡献更大，不具尺度不变性。
- 类本身不验证 `n_fft/hop_length/win_length`；非法值在窗口或 STFT 处失败，`hop_length>=win_length` 还可能静默产生第 1.2 节的 Hann 覆盖盲点。
- 当前标准训练链先由框架从模型谱重建 waveform，再由该类重新 STFT；它不是 direct-spectrum loss 接口，可能把不一致预测谱先投影到可重建波形。

**必要参考与实现边界**

代数结构可参考 [Fast-ULCNet 精读](../../paper/se/Bang-Olufsen/09_Fast_ULCNet.md)，但本仓是标准 Hann 的 waveform adapter，并非论文的 WOLA/几何配方；数值行为以当前 [`freq_domain.py`](../../../core/losses/freq_domain.py#L324) 为定义。

## 5. 选择时必须保留的差异

| Loss | 核心归一化与 reduction | zero target + 非零 enhanced | 主要偏好 | 关键限制 |
|---|---|---|---|---|
| `SIL1WavSeTrainLoss` | 逐样本 MAE / target mean-abs，再 batch mean | 固定分母 `0.1+1e-8`，继续抑制 | 相对波形 L1 | 不是最优投影式 SI |
| `SIL1Loss` | 逐样本相对 L1，仅对 `r>1e-4` 样本 mean | 样本被 gate，零梯度 | 有声参考相对 L1 | 不负责 target-absent 抑制 |
| `SISNRLoss` | 零均值稳定化投影比，valid 样本 mean | target energy gate 跳过 | 波形方向/结构、增益宽容 | 有效区常量 enhanced 是零梯度驻点；超过 80 dB 是零梯度平台 |
| `L1WavLoss` | 全 batch 有效 sample mean | 直接抑制 | 绝对波形 MAE | 高电平样本绝对误差更大 |
| `MultiResSTFTLoss` | 每样本 `SC+log`，batch mean，再尺度 mean | SC=图零；log 仅在越过谱 floor 时抑制 | 多尺度幅度形状与低能细节 | 非零 target + 零/floored estimate 也可正 loss 零梯度；几何须防 Hann 盲点 |
| `MultiResL1SpecLoss` | 时域/谱域都按样本 sum，尺度 mean、batch mean | 正则化 LS scaling 把 estimate 置零，整体可为零 | 方差/尺度归一化的波形+多窗幅度 | 不提供 zero-target residual suppression；非零 clean + 零 enhanced 可为正 loss 但无梯度 |
| `MultiResFreqLoss` | 每尺度对 batch×有效 bin 全局 mean，再尺度 mean | 复谱 MSE 抑制；相对项有谱 floor 死区 | 对称复谱 MSE + fixed-target 非对称相对幅度 | 未 clamp 复谱项保留梯度；`n_fft` 实际须 `>=2` |
| `MagnitudePlusComplexL1Loss` | batch×有效 bin 全局 mean | 非零 Hann 覆盖处由幅度+复差抑制 | 单尺度幅度与相位一致 | 未 clamp 复谱项保留梯度；几何须防 Hann 盲点 |
| `SpectralConvergenceLoss` | 每样本 SC 或 fallback，再 batch mean | standalone SC 改为 log fallback，仅在越过 floor 时抑制 | target-normalized 全局谱形 | 非零 target + 零/floored estimate 也可正 loss 零梯度；几何须防 Hann 盲点 |
| `LogMagnitudeSTFTLoss` | 每样本有效 bin mean，再 batch mean | 越过 `1e-7` floor 时 log 幅度抑制 | 相对谱包络、低能细节 | 非零 target + 零/floored estimate 也可正 loss 零梯度；几何须防 Hann 盲点 |
