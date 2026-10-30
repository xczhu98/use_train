# 第三方 Loss Wrapper 与感知代理边界

> **这一页讲包在别人代码外面的那层壳。** `core.losses` 里有几个 Loss 不自己实现算法，而是整理输入后交给第三方库（PESQ、STOI、auraloss、UTMOS）计算。这里记录壳层的真实行为：怎么 import、依赖缺失会怎样、`lengths` 怎么用、返回值是什么形状、梯度能否回到模型；第三方库内部的公式不在本仓库推断。

几个反复出现的名词：

- **wrapper / delegate（包装层 / 被委托对象）**：wrapper 是本仓库的薄壳，delegate 是真正算数的第三方对象。
- **PESQ / STOI**：需要干净参考的语音质量 / 可懂度客观度量；`PESQLoss`、`STOILoss` 把它们做成可微代理当作训练目标。
- **UTMOS**：不需要参考的主观听感 MOS 预测模型；本仓库使用 vendored 的 `UTMOS22Strong` 加指定 checkpoint，不是官方 ensemble。
- **auraloss**：第三方频谱/时域 loss 库；本仓库只把 schedule 与权重透传给它的 delegate。
- **`allow_missing`**：可选依赖缺失时是否允许继续跑。为真时该分量变成图连通零值（能反传但梯度为零），**不是**等价算法或近似降级。

本文只记录 `core.losses` 中四个第三方 wrapper，以及 `GAPPostNetAwareLoss` 的 PESQ/UTMOS-aware 分支在**当前源码中的运行契约**：实际 import/delegate、依赖状态、输入变换、返回 shape、mask/reduction、梯度和缺依赖行为。delegate 内部没有出现在本仓库的计算不在此推测。

## 统一运行边界

完整的注册名清单（含 `TAPLoss` / `AcousticLoss` 别名与不注册的 `CompositeLoss`）以 [Loss 文档中心](README.md#完整-registry-inventory) 为准。

本文覆盖 `PESQLoss`、`STOILoss`、`AuralossMultiResSTFTLoss`、`AuralossSDRLoss` 和 `GAPPostNetAwareLoss`。标准调用为 `forward(enhanced, clean, lengths=None)`：

- PESQ/STOI 直接路径使用同形状 `[B, T]`。
- 两类 Auraloss wrapper 接受 `[B, T]`（转为 `[B, 1, T]`）或 `[B, C, T]`。
- 配置 `strict: true` 后，外层 [`_StrictLossWrapper`](../../../core/losses/build.py#L27-L226) 只接受同形状 `[B, T]`，要求整型 `lengths: [B]` 且 `1 <= lengths[i] <= T`，并要求输出是有限零维 Tensor。`build_loss` 本身没有 `strict` keyword，`strict` 应写在 loss 配置中。

[`requirements.txt`](../../../requirements.txt#L64-L73) 的实际状态如下；注释中的下界不是 lock，源码也没有版本检查：

| 分发包 | wrapper 的实际 import/API | requirements / 当前环境 / 锁定状态 |
|---|---|---|
| `pytorch-pesq` | `pytorch_pesq.PesqLoss(sample_rate, mode)` | 注释项 / 当前不可 import / 未锁版本 |
| `torch-stoi` | `torch_stoi.NegSTOILoss(sample_rate=..., use_vad=...)` | 注释项 / 当前不可 import / 未锁版本 |
| `auraloss` | `auraloss.freq.MultiResolutionSTFTLoss`、`auraloss.time.*Loss` | 注释项 / 当前不可 import / 未锁版本 |
| `pystoi` | wrapper 不 import | 默认评估依赖 / 当前可 import / `>=0.3` 非精确锁定 |

`pytorch_pesq` 也不能由 import/API 不同的 `torch_pesq` 或评估包 `pesq` 直接替代。定义 `GAPPostNetAwareLoss` 的 [`pase_family.py`](../../../core/losses/pase_family.py) 在模块 import 期还需要 requirements 已启用的 `librosa`、`scipy` 和 `torchaudio`。

想知道本机实际缺哪些依赖，逐个 import 即可：

```bash
python -c "import pytorch_pesq" ; python -c "import torch_stoi" ; python -c "import auraloss"
```

预期现象：装了哪个包就无输出、退出码 0；没装的会打印 `ModuleNotFoundError` 并以非零退出码结束。缺依赖时按默认 `allow_missing=False` 构造 wrapper 会在构造期抛 `RuntimeError`，这是正式训练更安全的选择。

### 输入 shape 与本地 padding mask

PESQ/STOI wrapper 直接调用时沿二维 `[B, T]` 路径构造

$$
M_{b,t}=\mathbf 1[t<\ell_b],\qquad
\widetilde x_{b,t}=x_{b,t}M_{b,t},\qquad
\widetilde y_{b,t}=y_{b,t}M_{b,t}.
$$

Auraloss wrapper 直接调用时支持 `[B, T]` 与 `[B, C, T]`：前者先增加 channel 维成为 `[B, 1, T]`，后者保持不变；`lengths` mask 形状为 `[B, 1, T]`，依靠广播覆盖所有 channel。strict wrapper 下则无论 delegate 能否接受三维输入，四类 wrapper 都只接受 `[B, T]`。

前四个 wrapper 的 `lengths` 处理都只是把 padding sample 置零，不按样本裁剪、pack，也不把有效长度传入第三方 delegate；delegate 仍看到统一的末尾零填充长度 `T`。因此，这个 mask 能阻断 padding sample 上的直接梯度，却不保证与“每条样本裁剪后分别计算”数值等价：第三方内部的分帧、时长归一化或全局 reduction 仍可能感知零尾长度。`GAPPostNetAwareLoss` 的 aware 分支是例外，它会逐样本裁剪后再委托，详见后文。

### `allow_missing` 不是等价降级

四个 wrapper 缺依赖且显式设置 `allow_missing=True` 时，返回

$$
L_{missing}=\operatorname{sum}\bigl(\operatorname{float}(enhanced)\times 0.0\bigr).
$$

该值是连接到 `enhanced` 计算图的 FP32 标量零，因此 `backward()` 可执行，且 `enhanced` 获得零梯度；[`test_missing_perceptual_metric_returns_graph_connected_zero`](../../../tests/test_core_losses_audit_regressions.py#L71-L80) 明确覆盖了 PESQ/STOI 两类。Auraloss 两类使用相同源码表达式，但当前没有对应的缺依赖回归测试。

这只是容许运行继续的保图零值，**不是等价算法、近似替代或性能降级模式**：该分量不再提供任何优化信号。作为唯一 loss 时，训练可能能反传却没有该目标的有效梯度；位于 `CompositeLoss` 中时，则等价于从总目标中移除这一项。默认 `allow_missing=False` 会在构造期抛出 `RuntimeError`，这是正式训练更安全的行为。`allow_missing=True` 只捕获 import 与 delegate 构造所在 `try` 块抛出的 `ImportError`；其他参数错误、shape 错误、delegate 计算错误和数值错误不会被吞掉。捕获缺依赖时会记一条 ERROR，首次 fallback forward 还会每实例再记一条 ERROR。

## `PESQLoss`

### 注册、签名与默认参数

- 注册名：`PESQLoss`；无别名。
- 构造签名：`PESQLoss(sample_rate=16000, mode="wb", allow_missing=False)`。
- 前向签名：`forward(enhanced, clean, lengths=None)`。
- 预期 shape：`enhanced`、`clean` 为同形状 `[B, T]`；`lengths` 可选，为 `[B]`。
- 预期幅值：源码注明输入应归一化到 `[-1, 1]`，但 wrapper **不会** clamp、峰值归一化或响度归一化。
- 采样率意图：`mode="wb"` 对应 16 kHz，`mode="nb"` 对应 8 kHz；wrapper 本地不校验二者组合，最终约束由 `pytorch_pesq` delegate 决定。

### 委托与本地公式

构造时实际执行：

```python
from pytorch_pesq import PesqLoss
pesq_fn = PesqLoss(sample_rate, mode)
```

可用时，wrapper 只做可选的 sample mask，然后原样委托：

$$
L_{PESQ}=D_{pytorch\_pesq}(\widetilde{enhanced},\widetilde{clean}).
$$

`D_pytorch_pesq` 表示外部 `PesqLoss`。本项目没有在 wrapper 层添加 `eps`、`clamp`、额外负号或 `.mean()`；delegate 的内部 PESQ 近似、分段、数值稳定项、符号和 reduction 均属于第三方实现边界，本文不伪造其公式。项目按该 delegate 的 loss 方向直接最小化返回值。

### 原理、优化偏好与限制

该 wrapper 用可微 PESQ 代理把增强输出朝感知质量代理方向优化，适合作为重建目标的辅助项或后期微调项，而不应被表述为标准 ITU PESQ 评估程序本身。它偏好 delegate 所建模的质量因素，不保证同步改善 SI-SDR、频谱距离、主观偏好或域外条件。

主要限制：

- 依赖 `pytorch-pesq`，不是评估用 `pesq` 包。
- wrapper 不负责幅值合法化；超出预期范围的输入仍直接进入 delegate。
- `lengths` 只执行零尾 mask，不逐样本裁剪，变长 batch 的结果可能受 delegate 时长处理影响。
- wrapper 不校验 device、dtype、shape、采样率与 mode；只有把 `strict: true` 写入传给 `build_loss` 的 loss 配置或 YAML，才由 strict 层保证输入 shape。
- 在 `CompositeLoss` 中，`PESQLoss` 通过类名兼容规则被视为 speech-reference-dependent：传入 pure metadata 时只计算 non-pure 子集；直接调用类本身没有这层 gate。
- wrapper 原样返回 delegate 结果；非标量输出是否被拒绝取决于外层构建模式，不能归因于 wrapper 自身。

实现位置：[`core/losses/perceptual_metric.py`](../../../core/losses/perceptual_metric.py#L17-L77)。训练 wrapper 使用 `pytorch_pesq.PesqLoss`；评估路径使用另一个 `pesq` binding，二者不可互换。

## `STOILoss`

### 注册、签名与默认参数

- 注册名：`STOILoss`；无别名。
- 构造签名：`STOILoss(sample_rate=16000, use_vad=False, allow_missing=False)`。
- 前向签名：`forward(enhanced, clean, lengths=None)`。
- 预期 shape：`enhanced`、`clean` 为同形状 `[B, T]`；`lengths` 可选，为 `[B]`。
- `use_vad` 原样传给 `torch_stoi.NegSTOILoss`；wrapper 不实现自己的 VAD、静音 gate 或阈值。

### 委托与本地公式

构造时实际执行：

```python
from torch_stoi import NegSTOILoss
stoi_fn = NegSTOILoss(sample_rate=sample_rate, use_vad=use_vad)
```

可用时，本地只施加同一 sample mask，然后原样返回 delegate 输出：

$$
\boldsymbol L_{STOI}=D_{torch\_stoi}(\widetilde{enhanced},\widetilde{clean})\in\mathbb R^B.
$$

`torch_stoi.NegSTOILoss` 当前真实 API 对 `[B, T]` 只约去时间维，返回逐样本 `[B]`；项目 wrapper 不再取负、不添加 `eps`/`clamp`，也**没有**对该结果执行 `.mean()`。STOI 的分帧、频带变换、相关性计算、VAD 规则和数值稳定细节仍属于第三方边界，不能由本项目这几行 wrapper 反推。

### 原理、优化偏好与限制

该 wrapper 偏向可懂度代理，适合与时域或频谱保真目标组合。最小化方向来自外部 `NegSTOILoss` 的返回契约。可懂度代理改善不等价于自然度、音质、说话人相似度或主观 MOS 全面改善。

当前存在明确的 reduction 契约缺口：项目标准 loss 契约要求 scalar，但 `STOILoss` 正常可用路径返回 `[B]`。配置 `strict: true` 的单 loss 会由 `_StrictLossWrapper` 拒绝该非零维输出；strict `CompositeLoss` 也会拒绝 `[B]`。`strict: false` 的 `CompositeLoss` 则可能经广播返回 `[B]`，违反标准标量契约。当前没有任何一层自动取 mean。

其他限制：

- 实际 import 是 `torch_stoi`；仅安装 requirements 中评估用的 `pystoi` 不足以构造本 wrapper。
- `use_vad=False` 是项目默认值；改为 `True` 后的具体 VAD 行为完全由所装 `torch-stoi` 版本决定。
- `lengths` 只把尾部置零，不把有效长度交给 delegate，不能假设与逐样本裁剪等价。
- wrapper 不做幅值归一化、采样率转换或本地 shape 校验。
- 在 `CompositeLoss` 中，它和 `PESQLoss` 一样通过类名兼容规则进入 speech-reference pure gate；该 gate 不会补上缺失的 mean；直接调用时无该 gate。
- `allow_missing=True` 的零值没有 STOI 优化意义，只保证图连通。

实现位置：[`core/losses/perceptual_metric.py`](../../../core/losses/perceptual_metric.py#L80-L140)。训练 wrapper 使用 `torch_stoi.NegSTOILoss`；评估路径使用 `pystoi.stoi`，二者不可互换。仓库未锁定 `torch-stoi` 版本。

## `AuralossMultiResSTFTLoss`

### 注册、签名与默认参数

- 注册名：`AuralossMultiResSTFTLoss`；无别名。
- 构造签名：

  ```python
  AuralossMultiResSTFTLoss(
      fft_sizes=[1024, 2048, 512],
      hop_sizes=[120, 240, 50],
      win_lengths=[600, 1200, 240],
      w_sc=1.0,
      w_log_mag=1.0,
      w_lin_mag=0.0,
      allow_missing=False,
  )
  ```

- 前向签名：`forward(enhanced, clean, lengths=None)`。
- direct 输入 shape：支持 `[B, T]`（本地转换为 `[B, 1, T]`）或 `[B, C, T]`（原样委托）；`enhanced` 与 `clean` 应同形状。strict wrapper 只接受 `[B, T]`。
- `lengths` 为 `[B]`，mask 为 `[B, 1, T]`，通过广播应用到 `[B, C, T]` 的所有 channel。

### 委托与本地公式

构造时，所有 schedule 和权重原样传给：

```python
from auraloss.freq import MultiResolutionSTFTLoss
loss_fn = MultiResolutionSTFTLoss(
    fft_sizes=fft_sizes,
    hop_sizes=hop_sizes,
    win_lengths=win_lengths,
    w_sc=w_sc,
    w_log_mag=w_log_mag,
    w_lin_mag=w_lin_mag,
)
```

记 `U(x)` 为二维输入增加 channel 维、三维输入保持不变，`M_{b,1,t}=1[t<lengths_b]`，则 wrapper 的完整本地关系只有：

$$
L_{AuralossMRSTFT}
=D_{auraloss.freq}(U(enhanced)\odot M,\ U(clean)\odot M).
$$

未提供 `lengths` 时不乘 mask。wrapper 自身不计算 spectral convergence、log-magnitude 或 linear-magnitude 公式，也不添加本地 `eps`、clamp、窗函数默认、尺度归一化或最终 reduction；这些全部由所安装的 `auraloss.freq.MultiResolutionSTFTLoss` 决定。`w_sc`、`w_log_mag`、`w_lin_mag` 只是透传参数，不能据此伪造 delegate 的组合细节。

### 原理、优化偏好与限制

该项从多个 STFT 尺度约束时频结构，通常偏好跨短窗与长窗同时保持瞬态和谐波细节。它是第三方频谱重建目标，不等价于本项目 [`MultiResSTFTLoss`](../../../core/losses/freq_domain.py)，两者即使 schedule 相同，也不能假定 `eps`、窗、padding、单尺度定义或 reduction 数值一致。

主要限制：

- wrapper 对三个 schedule 不做本地校验，只原样传给 auraloss。第三方构造器仅做部分检查（例如要求三份列表等长），并不会系统拒绝空列表或所有几何非法值；部分问题会延迟到 delegate 内部构造或 `forward` 才报错。
- 二维输入只增加单声道维；三维输入的 channel 语义原样交给 delegate。其他 rank 的 Tensor 不会在本地被明确拒绝。
- `lengths` 只做 waveform 零尾 mask；STFT 边界帧和 delegate 的归约仍可能包含零尾影响。
- 该类未声明 `REQUIRES_SPEECH_REFERENCE`，也不在 `CompositeLoss` 的类名兼容集合内，因此 pure metadata 不会自动排除静音 reference 样本；它按 full batch 调用 delegate。
- `allow_missing=True` 时整个频谱分量变成保图零值，不是切换到项目内置 MR-STFT。

实现位置：[`core/losses/perceptual_lib.py`](../../../core/losses/perceptual_lib.py#L14-L95)。运行时只认 `auraloss.freq.MultiResolutionSTFTLoss`；仓库未锁定 `auraloss` 版本，因此最终公式与 reduction 以实际安装版本为准。

## `AuralossSDRLoss`

### 注册、签名与默认参数

- 注册名：`AuralossSDRLoss`；无别名。
- 构造签名：`AuralossSDRLoss(sdr_type="sisdr", zero_mean=True, allow_missing=False)`。
- 前向签名：`forward(enhanced, clean, lengths=None)`。
- direct 输入 shape：支持 `[B, T]`（本地转换为 `[B, 1, T]`）或 `[B, C, T]`（原样委托）；`lengths` mask 以 `[B, 1, T]` 广播到所有 channel。strict wrapper 只接受 `[B, T]`。
- 合法 `sdr_type`：`"snr"`、`"sisdr"`、`"sdsdr"`。构造流程先 import `auraloss`，再校验该值：依赖可用时非法值抛 `ValueError`；依赖缺失且 `allow_missing=True` 时会先进入 fallback，拼写错误不会被校验。
- capability：显式声明 `REQUIRES_SPEECH_REFERENCE = True`。

### 委托选择与本地公式

构造映射严格为：

| `sdr_type` | delegate | 本地构造调用 |
|---|---|---|
| `snr` | `auraloss.time.SNRLoss` | `SNRLoss(zero_mean=zero_mean)` |
| `sisdr` | `auraloss.time.SISDRLoss` | `SISDRLoss(zero_mean=zero_mean)` |
| `sdsdr` | `auraloss.time.SDSDRLoss` | `SDSDRLoss()` |

wrapper 只在 `snr` 与 `sisdr` 分支传入 `zero_mean`；`sdsdr` 固定调用无参 `SDSDRLoss()`，因此 wrapper 配置的 `zero_mean` 对该分支无效，实际默认行为由所安装的 auraloss 决定。前向只进行 channel 维适配与可选 sample mask：

$$
L_{AuralossSDR}=D_{sdr\_type}(U(enhanced)\odot M,\ U(clean)\odot M).
$$

wrapper 不在本地实现 SNR、SI-SDR 或 SD-SDR 投影公式，不添加 `eps`/clamp/额外负号，也不做 `.mean()`；具体定义、符号、稳定项、channel/batch reduction 都属于对应 auraloss 版本。项目按 delegate 返回的 loss 方向直接最小化。

### 原理、优化偏好与限制

三种 delegate 面向不同的波形失真比目标，`sisdr` 默认分支偏向尺度不变的参考对齐。只有 `snr`/`sisdr` 的去均值由 wrapper 参数控制；`sdsdr` 使用 delegate 默认值。该项适合与频谱或感知项组合，不能仅凭名称推断与项目 `SISNRLoss` 数值一致。

主要限制：

- `sdr_type` gate 位于 `auraloss` import 之后；依赖可用时非法值抛 `ValueError`，依赖缺失且 `allow_missing=True` 时不会校验该参数。
- 变长输入只做零尾 mask，不逐样本裁剪；全局能量比和 reduction 可能受统一长度影响。
- 在 `CompositeLoss` 中，`REQUIRES_SPEECH_REFERENCE=True` 使 Trainer 只把 non-pure 子集交给该组件；直接调用时没有 pure gate。
- `allow_missing=True` 不会回退到项目 `SISNRLoss` 或其他 SDR 实现。

实现位置：[`core/losses/perceptual_lib.py`](../../../core/losses/perceptual_lib.py#L98-L170)。运行时只认所列 `auraloss.time` delegate；仓库未锁定 `auraloss` 版本。

## `GAPPostNetAwareLoss` 的 PESQ/UTMOS-aware 边界

### 注册、签名与默认参数

- 注册名：`GAPPostNetAwareLoss`；无别名。
- 构造签名：

  ```python
  GAPPostNetAwareLoss(
      sampling_rate=48000,
      pesq_weight=0.0,
      utmos_weight=0.0,
      allow_missing_aware=False,
      utmos_model_dir=None,
      **mel_kwargs,
  )
  ```

- 前向签名：`forward(enhanced, clean, lengths=None)`。
- 默认两项 aware 权重均为 `0.0`；此时不构造 PESQ 或 UTMOS 代理，返回纯 `PaseMultiScaleMelLoss`。
- Mel 基础项的默认值由 `PaseMultiScaleMelLoss` 提供：`n_mels=[5,10,20,40,80,160,320]`、`window_lengths=[32,64,128,256,512,1024,2048]`、`clamp_eps=1e-5`、`mag_weight=0.0`、`log_weight=1.0`、`pow=1.0`、`weight=1.0`、`match_stride=False`、`mel_fmin` 全为 `0.0`、`mel_fmax` 全为 `None`、`window_type="hann"`。
- 构造器只拒绝 `< 0` 的 `pesq_weight`、`utmos_weight`，不检查有限性；`NaN` 与 `+Inf` 可通过。配置必须自行确保两项权重有限且非负。
- I/O 契约取决于代理是否实际构造：PESQ 与 UTMOS 两个代理都未构造时，仅执行 Mel 基础项，支持 `[B, T]` 或 `[B, C, T]`；只要至少一个代理对象存在并进入 aware 路径，就仅接受 `[B, T]` 或 `[B, 1, T]`，多通道 `[B, C, T]` 且 `C != 1` 会被 `_mono` 拒绝。

### 总目标与逐样本边界

记 `L_mel` 为本地 [`PaseMultiScaleMelLoss`](../../../core/losses/pase_family.py#L32-L215)，batch size 为 `B`，有效长度为 `ell_b`；若没有 `lengths`，则 `ell_b=T`。`R_{f_s -> 16000}` 表示仅在 `sampling_rate != 16000` 时调用 `torchaudio.functional.resample`，否则保持原波形。aware 总目标严格为：

$$
L_{GAP}=L_{mel}
+\frac{1}{B}\sum_{b=1}^{B}\left[
 w_p\,D_{PESQ}\!\left(R(enhanced_{b,:\ell_b}),R(clean_{b,:\ell_b})\right)
+w_u\,D_{UTMOS}\!\left(R(enhanced_{b,:\ell_b})\right)
\right],
$$

其中只有实际构造成功且启用的项才进入和式，`w_p=pesq_weight`、`w_u=utmos_weight`。实现按样本切成 `[1, ell_b]` 后重采样，再逐项累加，最后除以原 batch size `B`。与前四个通用 wrapper 不同，这里 aware delegate 不会看到 batch padding。代码没有为该 aware 聚合再增加 `eps`、clamp 或其他 gate。

`L_mel` 的本地定义不是第三方 delegate。对每个尺度 `r`，令 `X_r`、`Y_r` 为 Mel magnitude，`F_r` 为 Mel bin 数，`M_r` 为有效 STFT frame mask，源码中的 masked L1 为

$$
\operatorname{ML1}(X,Y;M)
=\frac{\sum |X-Y|M}{\max\left((\sum M)F,1\right)}.
$$

log-Mel 变换严格为

$$
\log_{10,src}(X)=\frac{\ln\left(\operatorname{clamp}_{min}(X,\epsilon)^{p}\right)}{\ln(10)},
$$

默认 `epsilon=1e-5`、`p=1.0`。基础项为

$$
L_{mel}=w\sum_r\left[
 w_{log}\operatorname{ML1}(\log_{10,src}(X_r),\log_{10,src}(Y_r);M_r)
+w_{mag}\operatorname{ML1}(X_r,Y_r;M_r)
\right],
$$

默认 `w_log=1.0`、`w_mag=0.0`、外层 `w=1.0`。这些公式只描述本地 Mel 基础项，不能拿来解释 PESQ 或 UTMOS delegate 内部。

### PESQ-aware 分支

当 `pesq_weight > 0` 时，类构造本文前述：

```python
PESQLoss(sample_rate=16000, mode="wb", allow_missing=allow_missing_aware)
```

因此边界是固定的 16 kHz wideband `pytorch_pesq.PesqLoss` 代理。外层先逐样本裁剪和必要重采样，内部 `PESQLoss` 不再收到 `lengths`；这里没有额外的 GAP 专用 PESQ 公式。

若 `pytorch-pesq` 缺失且 `allow_missing_aware=True`，`PESQLoss` 对象仍存在，但每条样本返回保图零值；PESQ-aware 项因此没有优化信号。它不是 UTMOS 替代，也不是近似 PESQ 降级。

### UTMOS-aware 分支

这里使用 **vendored SpeechMOS 的 `UTMOS22Strong` + 指定 checkpoint**，不是 UTMOS 官方 ensemble。`utmos_weight > 0` 时构造私有、未注册的 `_UTMOSAwareLoss`：

1. 从 `utmos_model_dir` 或默认 `third_party_scripts/UTMOS` 查找 `SpeechMOS/speechmos` 与 `utmos22_strong_step7459_v1.pt`。
2. import `speechmos.utmos22.strong.model.UTMOS22Strong`。
3. 通过项目 [`safe_torch_load`](../../../core/utils/safe_load.py) 加载 checkpoint，再以 strict `load_state_dict` 装载。
4. estimator 始终为 eval 且参数 `requires_grad=False`；模型参数不更新，但 enhanced waveform 到 MOS 输出的输入梯度保留。

$$
D_{UTMOS}(enhanced)=-\operatorname{mean}\left(
UTMOS22Strong(enhanced.float(),16000)
\right).
$$

`clean` 被忽略，输入固定按 16 kHz 调用。源码未锁定 vendored SpeechMOS 版本或 checkpoint 内容身份，因此运行语义最终由本机 vendored 源码和该文件决定。

`_build_utmos_loss()` 只捕获 `ImportError`、`FileNotFoundError` 和 `RuntimeError`：默认 `allow_missing_aware=False` 时直接报错；设为 `True` 时返回 `None` 并移除整个 UTMOS 项，不返回 UTMOS 保图零 delegate。其他异常仍 fail-fast，总计算图由 Mel 基础项和其他可用项连接。

### 配置、gate 与 Trainer 边界

当前 [`gap_urgenet_phase2_postnet_gan.yaml`](../../../configs/se/urgent2026/pase/gap_urgenet/gap_urgenet_phase2_postnet_gan.yaml#L150-L164) 的 `pesq_weight` / `utmos_weight` 是 **`0.0/0.0`**，因此实际只运行 Mel 基础项，不构造两个 aware delegate。

源码见 [`core/losses/pase_family.py`](../../../core/losses/pase_family.py#L218-L317)。运行时还需注意：

- `GAPPostNetAwareLoss` 未声明 `REQUIRES_SPEECH_REFERENCE`，类名也不在 `CompositeLoss` 的兼容 gate 集合中；Trainer/Composite 不会按 pure metadata 自动筛掉内部 PESQ 或 Mel 分支。
- UTMOS 不使用 clean，但 PESQ 与 Mel 使用 clean，因此整个组合不是无参考 loss。
- `sampling_rate` 控制 Mel 基础项及 aware 前重采样；aware delegate 固定 16 kHz。
- aware 启用后逐样本裁剪，梯度经裁剪、可微重采样和可用 delegate 回到 enhanced；batch reduction 为逐样本代理之和除以原始 `B`。

## 使用边界摘要

| wrapper | 优化意义 | 关键运行边界 |
|---|---|---|
| `PESQLoss` | 可微感知质量代理 | 非标准 PESQ evaluator；delegate 返回 shape/reduction 原样透传 |
| `STOILoss` | 可懂度代理 | 正常返回 `[B]`，没有 `.mean()`；strict 路径拒绝 |
| `AuralossMultiResSTFTLoss` | 多尺度频谱重建 | 不等同项目内置 `MultiResSTFTLoss`；零尾仍参与 delegate 长度 |
| `AuralossSDRLoss` | SNR/SI-SDR/SD-SDR 代理 | pure gate 仅由 Composite/Trainer 执行；delegate reduction 原样透传 |
| `GAPPostNetAwareLoss` | Mel + 可选 PESQ/UTMOS 代理 | 当前配置 `0.0/0.0`；aware 启用后仅支持 mono 并逐样本裁剪 |

配置入口与 `CompositeLoss` 的 gate/reduction/Trainer 路由见 [Loss 配置指南](Loss_Configuration_Guide.md)。
