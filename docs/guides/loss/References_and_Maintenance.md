# Loss 实现映射与维护规范

> **这一页给维护者看。** 它不解释公式，只固定三件事：每个 Loss 的运行时身份（谁和谁是同一个实现、谁不在 registry 里）、必须保持的归属红线（哪些是用户自研、每篇 canonical 只能对应哪个实现）、以及改完代码或文档后要做的防漂移动作。只想配置或理解某个 Loss 时，从 [Loss 文档中心](README.md) 和对应专题页进入即可。

先解释两个本文反复使用的说法：**canonical 精读**指 `docs/paper/loss/` 下与某个实现严格一一对应的论文解读页，它只为那一个实现背书；**用户自研**指用户明确声明的方法、机制或精确配方，数学相似、类名、目录名和 Git 历史都不能改写其归属。

本文是 `core.losses` 的紧凑维护入口，不再承担文献检索账本。每个 Loss 的公式、函数原理、优化意义、输入输出、mask/gate/reduction、梯度边界与 Trainer 调用方式由实现手册维护；本页只固定运行时身份、归属红线、家族映射和防漂移动作。

> 单元测试只能证明当前实现契约，不能证明训练收敛、论文复现或效果指标。

## 1. 运行时基线

完整的注册名清单（含 `TAPLoss` / `AcousticLoss` 别名与不注册的 `CompositeLoss`）以 [Loss 文档中心](README.md#完整-registry-inventory) 为准。

[`CompositeLoss`](../../../core/losses/composite.py#L377) 由 [`build_loss`](../../../core/losses/build.py) 特判构造，不注册、也不计入 registry。`MPSEnetCompositeLoss` 则是已注册的模型专用实现，调用契约为 `forward(forward_dict)`，不能按标准 `(enhanced, clean, lengths)` 使用。

`build_loss(cfg, stft_cfg=None, *, sample_rate=None)` 的维护基线：

- 普通 Loss、Composite 顶层和 `components[]` 各有独立的允许键集合，未知结构键必须 fail-fast。
- STFT 与 sample rate 只注入构造函数显式接收的参数；matching-context Loss 的显式值必须与框架上下文一致。
- `strict: true` 负责输入结构和有限标量输出检查，不改变组件公式。
- 详细 YAML、capability 与 Trainer 路由只在 [Loss 配置指南](Loss_Configuration_Guide.md) 维护。

## 2. 必要来源标签与严格边界

来源只在有助于防止错误归因时保留，使用以下简短标签：

- **数学基础**：L1/L2、cross entropy、SI 投影、STFT 距离等通用数学族。
- **外部方法或 delegate**：实现与论文方法、作者实现或第三方运行时有关；只说明对应关系，不代表本地逐值复现。
- **项目工程**：本仓库的 builder、接口适配、mask、gate、reduction、padding、cache、日志和 DDP 运行语义。
- **用户自研**：用户明确声明的方法、机制或精确配方；数学相似、类名、目录名、注释和 Git 历史都不能改写其归属。
- **未实现**：论文或设计存在，但当前 registry 和项目实现均不存在。

确有必要保留外部参考时，只记录会影响公式解释或运行边界的最短结论。

### 2.1 三篇 canonical 一对一

`docs/paper/loss` 的三篇 canonical 必须严格一对一：

- [`01_TAPLoss.md`](../../paper/loss/CMU/01_TAPLoss.md) 只支持 `TAPLoss` / `AcousticLoss`；
- [`02_Loud-loss.md`](../../paper/loss/IMU-SpeechLab/02_Loud-loss.md) 只支持当前尚未实现的 Loud-loss；
- [`03_MBRCL.md`](../../paper/loss/Concordia/03_MBRCL.md) 只支持 `MultiBandRelativeContrastiveLoss`。

三篇均不得为用户自研项、SS 机制、GTCRN、PASE、MP-SENet、DeepFilterNet 或其他 Loss 背书。

### 2.2 用户自研归属红线

以下四项归属必须保持无歧义：

1. `SIL1WavSeTrainLoss`；
2. `MRSIRelCspecL2Loss`；
3. `MRSpecSuppressPenaltyLoss`；
4. **一份完整三项组合配方**：由上述三项组成，且三项都显式设置 component-level `enable_ss_undesired_energy: true`。

它们均为**用户自研**，不得归因于 `se_train`、论文、CAGCRN、DeepVQE 或 MBRCL。配置路径中的 `deepvqe`、参数名中的 `align_se_train` 以及历史注释都不是来源证据。

`CompositeLoss` 的构建、组件路由、cache、日志和 DDP 基础设施属于**项目工程**；desired/undesired 能量混合这一目标机制，以及上述精确三项组合配方属于**用户自研**。二者必须分层描述，不能因为机制由 `CompositeLoss` 执行就把其方法归属改成项目通用基础设施。

<a id="3-实现来源偏差总账"></a>
## 3. 紧凑实现与家族映射

每个 registry 名称均在下表恰好归入一个实现手册；完整逐名 inventory 仍以 [README](README.md#完整-registry-inventory) 为准。

| 家族 | Registry 名称 | 必须保留的运行边界 | 详细手册 |
|---|---|---|---|
| 波形 | `SIL1WavSeTrainLoss`、`SIL1Loss`、`SISNRLoss`、`L1WavLoss` | `SIL1WavSeTrainLoss` 对零/近零参考使用固定分母 fallback，**不是 silent gate**；`SIL1Loss` 与 `SISNRLoss` 才各有自己的有效参考 gate | [波形与频谱 Loss](Waveform_and_Spectral_Losses.md#2-波形-loss) |
| 通用频谱 | `MultiResSTFTLoss`、`MultiResL1SpecLoss`、`MultiResFreqLoss`、`MagnitudePlusComplexL1Loss`、`SpectralConvergenceLoss`、`LogMagnitudeSTFTLoss` | 各实现的 STFT 几何、center、mask 分母、zero-target 行为和尺度求和/平均不能互相套用 | [波形与频谱 Loss](Waveform_and_Spectral_Losses.md#3-多分辨率频谱-loss) |
| 论文对照 | `GTCRNCompressedSpectralLoss`、`GTCRNLogRatioLoss`、`MultiBandRelativeContrastiveLoss`、`TAPLoss` / `AcousticLoss`、`FWNegSISDRLoss`、`PLCPAAsymmetricLoss`、`GAPHybridLoss` | direct spectra、noisy/undesired reference、固定 estimator、pure gate 和不支持 SS 的组件必须按各自 capability 路由 | [论文对照 Loss](Paper_Aligned_Losses.md) |
| 用户自研 MR | `MRSIRelCspecL2Loss`、`MRSpecSuppressPenaltyLoss`、`MRSIRelCspecL2SuppressPenaltyLoss` | 前两项为用户自研；第三项只是两项谱 Loss 的项目派生容器，不等于完整三项 SS 配方 | [任务与模型特定 Loss](Task_and_Model_Specific_Losses.md#2-用户自研多分辨率复谱与抑制项) |
| PASE / GAP | `PaseMultiScaleMelLoss`、`GAPPostNetAwareLoss` | 多尺度 Mel 的尺度 reduction 与 aware 代理的逐样本 trim、可选依赖和零权重 gate 分开处理 | [任务与模型特定 Loss](Task_and_Model_Specific_Losses.md#3-pasegap-波形重建与-trainer-兼容项) |
| 基础适配与 CE | `MSELoss`、`ScalarPassthroughLoss`、`StateCrossEntropyLoss`、`SpeakerCrossEntropyLoss` | `MSELoss` 的 `lengths` 当前不参与 reduction；`ScalarPassthroughLoss` **原样返回 `enhanced`**，不保证它是标量；CE 的 ignore mask 和 DDP mass 不可省略 | [任务与模型特定 Loss](Task_and_Model_Specific_Losses.md) |
| MP-SENet | `MPSEnetCompositeLoss` | 只接收 `forward_dict`；五项 reconstruction 与 MetricGAN 的 metric/discriminator 路径分离 | [任务与模型特定 Loss](Task_and_Model_Specific_Losses.md) |
| DeepFilterNet | `DeepFilterNetMultiResSpecLoss`、`DeepFilterNetOverAttenuationLoss`、`DeepFilterNetLSNRLoss`、`DeepFilterNetAlphaLoss`、`DeepFilterNetFullLossWithLSNR`、`DeepFilterNetLSNRComponent`、`DfAlphaLossComponent` | full wrapper 由 `build_loss` 构造**任意 enhancement 主项**，再按输入 gate 可选加入 LSNR/alpha；两个 adapter 从显式 forward 参数（由 Composite kwargs 路由）读取 `predicted_lsnr` / `predicted_alpha` 与 `noise` / `noisy`，**不读取 `forward_dict`** | [任务与模型特定 Loss](Task_and_Model_Specific_Losses.md#6-deepfilternetdpdfnet-专用项) |
| 第三方 wrapper | `PESQLoss`、`STOILoss`、`AuralossMultiResSTFTLoss`、`AuralossSDRLoss` | delegate 输出、padding mask、标量性和 `allow_missing` 的保图零必须按实际 wrapper 契约理解 | [第三方 Wrapper](Third_Party_Wrappers.md) |

Loud-loss 当前未实现，不属于 registry；PASE GAN helper 由 Trainer 直接调用，也不属于 registry。

### 3.1 用户自研完整三项配方

当前完整配方承载于 [`deepvqe_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml`](../../../configs/aec/microsoft_challenge/deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml#L619-L644)，但目录名不改变用户自研归属：

| Component | `weight` | component-level SS | 关键参数 |
|---|---:|---:|---|
| `SIL1WavSeTrainLoss` | `1.0` | `true` | 默认参数 |
| `MRSIRelCspecL2Loss` | `1.0` | `true` | `fft_power_min/max: 6/11`、`alpha_rel_L2: 1.0`、frequency weight 与 `align_se_train` 开启 |
| `MRSpecSuppressPenaltyLoss` | `1.0` | `true` | `fft_power_min/max: 6/11`、`align_se_train` 开启 |

`6..11` 对应 FFT/window `[64,128,256,512,1024,2048]`，hop 为窗长一半。AEC Trainer 注入 `undesired_ref = mic - nearend` 与 `undesired_enhanced = mic - nearend_estimate`。三项公式、逐样本 SS 调用、mask 与 cache 禁用条件分别见 [实现手册](Task_and_Model_Specific_Losses.md#2-用户自研多分辨率复谱与抑制项) 和 [配置指南](Loss_Configuration_Guide.md#73-ss-undesired-energy)。

<a id="412-compositeloss"></a>
### 3.2 非 registry 的 `CompositeLoss`

线性组合为

$$
L_{\mathrm{total}}=\sum_i w_iL_i,
$$

其中有限权重不归一化，总和以 FP32 累加。pure/valid gate、component kwargs 路由、MR cache、分项日志和 DDP global-mean mass 是项目基础设施。

用户自研的 SS 机制只对 effective `enable_ss_undesired_energy=true` 的 component 生效。对样本 $b$，令有效 sample mask 为 $M_{b,t}$：

$$
E_b^{\mathrm{des}}=\sum_t M_{b,t}s_{b,t}^{2},\qquad
E_b^{\mathrm{und}}=\sum_t M_{b,t}u_{b,t}^{2},
$$

$$
a_b=\frac{E_b^{\mathrm{des}}}{E_b^{\mathrm{des}}+E_b^{\mathrm{und}}+10^{-8}},
$$

$$
L_{i,b}^{\mathrm{SS}}
=a_bL_i(\widehat{s}_b,s_b)
+(1-a_b)L_i(\widehat{u}_b,u_b),\qquad
L_i^{\mathrm{SS}}=\frac{1}{B}\sum_bL_{i,b}^{\mathrm{SS}}.
$$

这里 $u$ 是 `undesired_ref`，$\widehat{u}$ 是 `undesired_enhanced`。`undesired_ref` 可由 `noisy-clean` 推导；`undesired_enhanced` 必须提供。只要活动 direct-spectrum component 同时启用 SS 且本次实际传入 direct spectra，就因没有 undesired spectra 而 fail-fast。开启 SS 的 MR component 不使用 shared cache。

## 4. 关键运行边界

完整公式卡片不在本页重复：波形与通用频谱见 [Waveform_and_Spectral_Losses.md](Waveform_and_Spectral_Losses.md)，论文对照项见 [Paper_Aligned_Losses.md](Paper_Aligned_Losses.md)，任务与模型专用项见 [Task_and_Model_Specific_Losses.md](Task_and_Model_Specific_Losses.md)。维护时仍必须保证每个注册实现的专题卡片同时写清公式、原理、优化意义、输入输出、mask/gate/reduction、梯度边界和 Trainer 路由。

### 4.1 输入、mask 与 reduction

- 标准 waveform 路径通常接收 `enhanced/clean: [B,T]` 与可选 `lengths: [B]`；非标准签名和模型字典输入必须在专题卡片中显式标出。
- centered STFT 路径应按每条有效右边界准备输入并计算 frame mask；non-centered 路径必须单列短输入和零有效帧行为。
- 不能用“masked mean”省略分母：需明确是 sample、utterance、frame、TF bin、channel 还是 batch 全局质量，以及多分辨率最外层是求和还是平均。
- graph-connected zero 只保证 autograd/DDP 图连通，表示该分支梯度为零，不表示仍有优化信号。

### 4.2 gate、梯度与 Trainer

- pure metadata gate、组件内部 valid gate、SS 能量权重和零权重 component 是不同层次；文档必须分别说明触发条件与被保留的 batch 质量。
- `SIL1WavSeTrainLoss` 的零/近零参考分支仍计算 residual，并把分母替换成 `0.1+10^{-8}`；不得写成丢弃样本的 silent gate。
- direct-spectrum capability 需要成对谱输入、布局/频点/帧数校验和明确的 waveform fallback；其 SS 冲突按本次实际调用路径判断。
- speech-reference 或 CE 等局部 mean 在 DDP 下必须以 reduction mass 重标定；所有 rank 保持同一 collective 序列，空 rank 以零质量参与。
- Trainer 必须只向活动 capability 注入 `noisy`、runtime sample rate、direct spectra、pure metadata、state/speaker label 或 DFN 预测头；不能假定 `**kwargs` 自动完成语义路由。

## 5. 维护约定

以下规则适用于每次 loss 行为或文档变更。

- 重新导入 `core.losses` 核对 registry 数量与名称；同步更新 [README inventory](README.md#完整-registry-inventory)。别名按 registry 名称计数、按实现类合并，`CompositeLoss` 不计入 registry。可直接复用的核对命令见 [配置指南第 1 节](Loss_Configuration_Guide.md#1-当前构建边界)。
- 在唯一对应的实现手册更新构造参数、forward 输入输出、完整公式、优化意义、限制，以及每个 `eps`、clamp、mask、gate、reduction 和梯度边界；本页只更新家族映射与跨组件边界。
- 核对 `lengths=None` 与显式 `lengths`、centered/non-centered STFT、短输入、全静音/全 ignore、零帧、perfect reconstruction、NaN/Inf、AMP 与 graph-connected zero。
- 核对 capability 和 Trainer 注入：waveform/direct spectra、noisy/undesired reference、runtime sample rate、pure metadata、state/speaker labels、DFN 显式预测 kwargs；缺必需输入应按组件契约 fail-fast。
- 核对 Composite 的 component 权重、pure/valid gate、用户自研 SS 能量混合、MR cache、日志键、DDP reduction mass 与所有 rank collective 顺序。
- 保持 `SIL1WavSeTrainLoss`、`MRSIRelCspecL2Loss`、`MRSpecSuppressPenaltyLoss` 和完整三项 component-level SS 配方的用户自研归属；不得被名称、目录或相似外部方法覆盖。
- 保持三篇 canonical 严格一对一；Loud-loss 在真正注册并实现前始终标为未实现。
- 运行与行为改动对应的公式 oracle、padding、路由、DDP 和空边界测试；测试结论只写实现契约，不写成论文复现或效果结论。
- 检查本目录相对链接、Markdown 标题层级与公式渲染；尤其确认 SS 公式中的 mask、上下标和 desired/undesired 两路无歧义。
