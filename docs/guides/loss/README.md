<h1 align="center">Loss 文档中心</h1>

本页是 Loss 文档入口，也是 core 与任务专属 loss 的 registry inventory。注册集合随任务模块导入而扩展，应从当前 `LOSS_REGISTRY` 查询；`TAPLoss` 与 `AcousticLoss` 指向同一实现类；`CompositeLoss` 由 `build_loss(...)` 特判组装，**不注册**。

## 先读这段

**Loss（损失函数）** 是训练时衡量“模型输出离目标有多远”的数值；Trainer 把它的值反传成梯度，模型据此更新。本目录记录本项目所有 Loss 的运行事实：公式、输入输出、mask/gate、归约、梯度，以及 YAML 怎么配。

几个先要认识的名词：

- **registry（注册表）**：`core/losses` 里每个 Loss 类在 import 时用 `@LOSS_REGISTRY.register("名字")` 登记；YAML 的 `loss.type` 必须写这个精确注册名，写错在启动期就报错，不会静默回退。
- **builder**：`build_loss(...)` 是统一构造入口，负责查注册表、注入 STFT 与采样率上下文、按 `strict` 加一层输入/输出校验。`strict` 是 `loss` 配置里的字段，不是 `build_loss` 的关键字参数。
- **`CompositeLoss`**：把多个 Loss 按权重相加的容器（权重不自动归一化）；它不是普通注册项，而由 builder 特判组装。
- **capability（能力声明）**：Loss 用类属性声明自己需要哪些额外输入，例如 `noisy` 退化波形、采样率、模型原生谱。Trainer 只向声明过的 Loss 传这些参数，所以在 YAML 里自己发明一个键不会生效。
- **pure gate**：样本被数据侧标成纯噪声 / 纯回声 / 纯回声加噪声时，依赖干净语音参考的 Loss 会被路由到其余样本上计算。
- **SS（`enable_ss_undesired_energy`）**：项目历史键名，指按样本能量比例把“目标分支”和“残留分支”两路监督混合起来，与语音分离任务 `ss` 无关；该机制是**用户自研**，归属见下文。

任务缩写（各页会反复出现）：

- `se`（speech enhancement，语音增强）：输入带噪语音，输出干净语音。
- `aec`（acoustic echo cancellation，回声消除）：输入麦克风信号与远端参考，输出近端语音。
- `tse`（target speaker extraction，目标说话人提取）：输入混合语音与目标说话人的注册音，只输出该说话人的语音。
- `ss`（speech separation，语音分离）：输入混合语音，输出固定 K 路分离语音；输出顺序事先未知，靠 PIT 监督。
- `ssl`（spatial sound localization，空间声源定位）：输入固定麦克风阵列录音，输出声源方向，不改变音频本身。

最小上手路径：照 [Loss YAML / Trainer 配置契约](Loss_Configuration_Guide.md) 第 2 节写 `type` / `args` / `strict`，再从第 9 节挑一个示例复制进 YAML。只验证配置能否构建时执行：

```bash
python -m pytest -m unit tests/test_loss_build.py -q
```

预期现象：用例全部通过；写法有误（未知 `type`、未知结构键、`args` 里塞了目标类不接受的参数）会在构建期直接抛错，不会拖到训练中途。

## 文档导航

| 目标 | 文档 | 必查内容 |
|---|---|---|
| 编写或审查 YAML | [Loss YAML / Trainer 配置契约](Loss_Configuration_Guide.md) | builder 注入、`CompositeLoss`、strict、SS、MR cache 与 Trainer 路由 |
| 查波形和通用频谱 Loss | [波形与频谱 Loss 实现手册](Waveform_and_Spectral_Losses.md) | 公式、函数原理、优化意义、输入输出、padding / mask、reduction 与梯度 |
| 查论文对照实现 | [论文对照 Loss 指南](Paper_Aligned_Losses.md) | 公式对应关系、实现差异、gate、梯度与 Trainer 边界 |
| 查任务或模型专用 Loss | [任务与模型特定 Loss 实现参考](Task_and_Model_Specific_Losses.md) | 特殊 `forward`、辅助 Tensor、mask / gate、归一化与 Trainer 路由 |
| 查可选第三方 Loss | [第三方 Loss Wrapper 与感知代理边界](Third_Party_Wrappers.md) | 输入输出、依赖降级、可微性、reduction 与运行限制 |
| 查来源边界与维护规则 | [来源边界与防漂移维护](References_and_Maintenance.md) | 紧凑来源边界与防漂移维护入口 |

每个 `core/losses` Loss 的完整说明由对应专题页维护，必须覆盖：**公式、函数原理、优化意义、输入输出、mask / gate、reduction、梯度，以及 Trainer 实际运行边界**。本页不重复这些实现卡片，以免形成第二份事实清单。

## 权威口径

1. [`core/losses/`](../../../core/losses/)、[`build.py`](../../../core/losses/build.py)、[`registry.py`](../../../core/utils/registry.py) 与相关测试决定实际注册、构建、参数注入和输出契约。
2. [配置指南](Loss_Configuration_Guide.md) 说明 YAML、builder、capability 与 Trainer 如何把 Loss 接入运行时。
3. 各实现专题页解释公式和行为；文档与运行时冲突时，以源码和测试为准并修正文档。
4. [`docs/paper/loss/`](../../paper/loss/README.md) 只界定论文归属，不覆盖本项目的运行时事实。

关键运行边界：Trainer 按 capability（Loss 声明自己需要哪些额外输入）注入 `noisy` 退化波形、sample rate、direct spectra（模型直接给出的预测谱，不经波形往返）、pure metadata 和任务辅助 Tensor；`CompositeLoss` 负责组件加权、pure / valid gate、reduction mass（每条样本/帧在全局平均里占的权重）、DDP 重标定（多卡训练下用全局质量重新缩放各卡局部均值）、MR cache（多分辨率 STFT 的共享缓存）与通用 SS 路由。标准 strict 路径要求有限零维 Tensor（即标量）；非标准签名、非标量输出和专用 Trainer 路径以专题页及配置指南为准。`ScalarPassthroughLoss` 被调用时**原样透传** `enhanced`，不是“标量适配器”，也不保证输出为标量。

## 自研与论文边界

以下内容均为**用户自研**：

- `SIL1WavSeTrainLoss`
- `MRSIRelCspecL2Loss`
- `MRSpecSuppressPenaltyLoss`
- 一份由上述三项组成、且三项均设置 component-level `enable_ss_undesired_energy=true` 的完整三项组合配方

不得将这三个 Loss 或该组合配方（包括三项 component-level SS 设置）归因于 se_train、任何论文、CAGCRN、DeepVQE 或 MBRCL；`CompositeLoss` 的通用容器能力也不改变其自研归属。

现有三篇 canonical 仅允许严格一对一引用：

- [`01_TAPLoss.md`](../../paper/loss/CMU/01_TAPLoss.md) → 仅支持 `TAPLoss` / `AcousticLoss` 的 TAP 边界。
- [`02_Loud-loss.md`](../../paper/loss/IMU-SpeechLab/02_Loud-loss.md) → 仅支持 Loud-loss；当前 registry 未实现该 Loss。
- [`03_MBRCL.md`](../../paper/loss/Concordia/03_MBRCL.md) → 仅支持 `MultiBandRelativeContrastiveLoss`。

## 优化倾向口径

下列标签描述的是**正权重最小化时，Loss 公式直接施加的优化方向**，而不是模型用途或最终指标保证：

读法：希望输出波形更贴近目标就看“保真”，希望回声/噪声残留被压掉就看“抑制”；两者都只说明公式往哪个方向推，不代表评估指标（PESQ、SI-SDR 等）一定变好。

- **保真**：直接拉近 `enhanced` 与 `clean` / target，或保护目标语音的波形、频谱和可辨结构。
- **抑制**：显式惩罚 undesired / residual energy、额外能量，或相对 target 的过高估计。
- **兼具**：同一目标含独立的保真项和抑制项；**条件性**表示该方向只在特定输入、gate、开关或系数下成立。
- **防过抑制**：惩罚目标语音被压低或输出过弱，方向与“进一步抑制”相反，属于目标保护。
- **代理 / 辅助 / 控制**：感知质量、可懂度、分类、辅助回归、gate 控制或 loss 透传，不能仅按保真 / 抑制解释。

普通重建误差在 `clean=0` 时也可能压低输出，但表中只有存在明确零目标处理时才标出“条件性残留抑制”；它不等同于独立的 undesired reference 或单侧过高估计惩罚。`CompositeLoss` 的负 component weight 会反转方向，不在本表常规口径内。

## 完整 Registry Inventory

以下清单按主要专题页归组，每个注册名只出现一次；分组用于导航，不改变运行时注册关系。

### SSL 任务专属

这三项由 `tasks.ssl.losses` 导入时注册，SSL 训练入口负责导入；独立调用
`build_loss` 时需要先导入该任务模块。它们消费帧域空间标签，不能替代 waveform loss。

| 注册名 | 合同 | 详情 |
|---|---|---|
| `SSLSpatialMSELoss` | native baseline 的空间监督 | [SSL 任务指南](../../../tasks/ssl/README.md) |
| `SSLDPIPDPITLoss` | canonical DP-IPD 逐帧 PIT，可选同配对的 DOA 辅助项 | [SSL 损失实现](../../../tasks/ssl/losses.py) |
| `SSLDOACompositeLoss` | 复用同一 PIT 指派组合 DP-IPD、DOA 分类或回归、VAD 与 near-field range | [组合监督参数](../../../tasks/ssl/README.md#组合监督与离线分块推理) |

### 波形与通用频谱：11 个

| 注册名 | 定位 | 优化倾向 | 详情 |
|---|---|---|---|
| `SIL1WavSeTrainLoss` | **用户自研**波形训练目标；也是完整三项组合配方的一个 component | **兼具（条件性）**：相对波形保真；零参考时以固定分母惩罚残留 | [实现手册](Waveform_and_Spectral_Losses.md) |
| `SIL1Loss` | 尺度不变 L1 | **保真**：只训练非静音参考，不负责 target-absent 抑制 | [实现手册](Waveform_and_Spectral_Losses.md) |
| `SISNRLoss` | SI-SNR 目标 | **保真**：clean 投影结构对齐；零参考样本被 gate | [实现手册](Waveform_and_Spectral_Losses.md) |
| `AbsentTargetLoss` | 零目标输出能量约束 | **条件性抑制**：仅 absent 样本参与，padding 不参与 | [实现手册](Waveform_and_Spectral_Losses.md#25-absenttargetloss) |
| `L1WavLoss` | 波形 L1 | **保真；条件性抑制**：零参考时等价于残留幅度惩罚 | [实现手册](Waveform_and_Spectral_Losses.md) |
| `MultiResSTFTLoss` | 多分辨率 STFT | **保真；条件性抑制**：零参考时由 log 幅度项压制可分辨残留 | [实现手册](Waveform_and_Spectral_Losses.md) |
| `MultiResL1SpecLoss` | 多分辨率频谱 L1 | **保真**：投影后的波形与多窗幅度重建；零参考时无抑制梯度 | [实现手册](Waveform_and_Spectral_Losses.md) |
| `MultiResFreqLoss` | 多分辨率频域目标 | **保真；条件性抑制**：零参考时复谱误差压制残留能量 | [实现手册](Waveform_and_Spectral_Losses.md) |
| `MagnitudePlusComplexL1Loss` | 幅度与复数谱 L1 | **保真；条件性抑制**：零参考时幅度与复谱项均压制残留 | [实现手册](Waveform_and_Spectral_Losses.md) |
| `SpectralConvergenceLoss` | 频谱收敛 | **保真；条件性抑制**：零参考时回退到 log 谱残留约束 | [实现手册](Waveform_and_Spectral_Losses.md) |
| `LogMagnitudeSTFTLoss` | 对数幅度 STFT | **保真；条件性抑制**：零参考时压制谱 floor 以上的残留 | [实现手册](Waveform_and_Spectral_Losses.md) |

### 论文对照：8 个注册名、7 个独立实现

| 注册名 | 定位 | 优化倾向 | 详情 |
|---|---|---|---|
| `GTCRNCompressedSpectralLoss` | GTCRN 压缩频谱目标 | **保真**：压缩 RI 与幅度重建 | [论文对照](Paper_Aligned_Losses.md) |
| `GTCRNLogRatioLoss` | GTCRN 对数比例目标 | **保真**：clean 投影相对残差优化，无显式 undesired 项 | [论文对照](Paper_Aligned_Losses.md) |
| `MultiBandRelativeContrastiveLoss` | 多频带相对对比目标 | **兼具（相对改善）**：clean / 包络对齐，并要求优于 noisy anchor | [论文对照](Paper_Aligned_Losses.md) |
| `TAPLoss` | TAP / Acoustic 实现的主注册名 | **保真代理**：对齐 clean 的声学参数，不是残留抑制项 | [论文对照](Paper_Aligned_Losses.md) |
| `AcousticLoss` | `TAPLoss` 同一实现类的别名 | **保真代理**：与 `TAPLoss` 相同 | [论文对照](Paper_Aligned_Losses.md) |
| `FWNegSISDRLoss` | 频率加权负 SI-SDR | **保真 / 目标保护**：undesired-aware 频带加权，不直接最小化 undesired 能量 | [论文对照](Paper_Aligned_Losses.md) |
| `PLCPAAsymmetricLoss` | PLCPA 非对称目标 | **保真 + 防过抑制**：额外惩罚输出幅度低于 clean | [论文对照](Paper_Aligned_Losses.md) |
| `GAPHybridLoss` | GAP-URGENet Predictor 混合目标 | **保真 + 防过弱**：谱重建并偏向避免输出能量塌缩 | [论文对照](Paper_Aligned_Losses.md#7-gaphybridloss) |

### 第三方 Wrapper：4 个

| 注册名 | 定位 | 优化倾向 | 详情 |
|---|---|---|---|
| `PESQLoss` | PESQ 感知代理 | **感知代理**：提高参考型 PESQ 代理，不属于直接保真 / 抑制二分 | [第三方边界](Third_Party_Wrappers.md) |
| `STOILoss` | STOI 感知代理 | **可懂度代理**：提高参考型 STOI 代理，不属于直接保真 / 抑制二分 | [第三方边界](Third_Party_Wrappers.md) |
| `AuralossMultiResSTFTLoss` | auraloss 多分辨率 STFT wrapper | **保真**：多尺度频谱重建 | [第三方边界](Third_Party_Wrappers.md) |
| `AuralossSDRLoss` | auraloss SDR wrapper | **保真**：参考信号 SDR 对齐 | [第三方边界](Third_Party_Wrappers.md) |

缺少可选依赖且启用 `allow_missing=True` 时，第三方 Wrapper 返回“图连通零值”——一个数值为 0、仍接在模型计算图上的标量：能反传，但该 component 的梯度为零，不再提供表中所述优化信号。

### 任务与模型特定：17 个

| 注册名 | 定位 | 优化倾向 | 详情 |
|---|---|---|---|
| `MRSIRelCspecL2Loss` | **用户自研**多分辨率 SI 相对复谱目标；也是完整三项组合配方的一个 component | **保真**：多尺度 clean-referenced 复谱重建 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `MRSIRelCspecL2SuppressPenaltyLoss` | 项目组合容器；不等同于用户自研的完整三项组合配方 | **兼具（条件性）**：复谱保真；`alpha_penalty > 0` 时增加过高幅度抑制 | [实现参考](Task_and_Model_Specific_Losses.md)、[配置契约](Loss_Configuration_Guide.md#73-ss-undesired-energy) |
| `MRSpecSuppressPenaltyLoss` | **用户自研**多分辨率频谱抑制惩罚；也是完整三项组合配方的一个 component | **抑制**：只惩罚 `|enhanced STFT| > |clean STFT|` 的额外谱能量 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `PaseMultiScaleMelLoss` | PASE 多尺度 Mel 目标 | **保真**：多尺度 Mel / log-Mel 重建 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `GAPPostNetAwareLoss` | GAP PostNet-aware 目标 | **条件性**：Mel 保真 + 可选 PESQ / UTMOS 感知代理；默认仅 Mel 保真 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `MSELoss` | 通用均方误差 | **条件性 target-matching**：target 为 clean 时属于保真 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `ScalarPassthroughLoss` | 原样透传 `enhanced` | **透传**：不是独立 objective，不属于保真 / 抑制二分 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `StateCrossEntropyLoss` | AEC state 分类辅助目标 | **分类辅助**：不直接优化 enhanced 的保真或残留能量 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `SpeakerCrossEntropyLoss` | speaker 分类辅助目标 | **分类辅助**：不直接优化 enhanced 的保真或残留能量 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `MPSEnetCompositeLoss` | MP-SENet 字典式复合目标 | **保真 + consistency**：重建 clean，并约束增强谱 / 波形自一致 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `DeepFilterNetAlphaLoss` | DeepFilterNet alpha 目标 | **条件性控制**：按 LSNR 控制 DF gate，不直接度量保真或残留 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `DeepFilterNetFullLossWithLSNR` | DeepFilterNet 完整目标与 LSNR | **配方依赖**：可替换的增强主项 + LSNR 辅助 + 可选 alpha 控制 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `DeepFilterNetLSNRComponent` | DeepFilterNet LSNR 组件 | **辅助回归**：回归 LSNR target，不直接优化增强输出 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `DeepFilterNetLSNRLoss` | DeepFilterNet LSNR 目标 | **辅助回归**：回归 LSNR target，不直接优化增强输出 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `DeepFilterNetMultiResSpecLoss` | DeepFilterNet 多分辨率频谱目标 | **保真**：多分辨率 clean 幅度谱重建 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `DeepFilterNetOverAttenuationLoss` | DeepFilterNet 过衰减惩罚 | **防过抑制**：只惩罚增强幅度低于 clean，不负责残留抑制 | [实现参考](Task_and_Model_Specific_Losses.md) |
| `DfAlphaLossComponent` | DeepFilterNet alpha 组件 | **条件性控制**：`DeepFilterNetAlphaLoss` 的 component adapter | [实现参考](Task_and_Model_Specific_Losses.md) |

### Speech Separation / PIT：3 个

PIT（permutation invariant training，排列不变训练）指源分离里“第 k 路输出对应哪个真实说话人”事先未知，训练时枚举所有配对、取代价最小的那一种作为监督；下面的注册名就服务于这套配对代价。

| 注册名 | 定位 | 优化倾向 | 详情 |
|---|---|---|---|
| `PairwiseL1Loss` | 所有估计/参考 source 配对的 L1 cost | 波形保真；零参考时约束输出幅度 | [SS/PIT 合同](Task_and_Model_Specific_Losses.md#8-speech-separation--pit) |
| `PairwiseNegSISDRLoss` | 配对负 SI-SDR cost | 尺度不变保真；不提供 source-presence 目标 | [SS/PIT 合同](Task_and_Model_Specific_Losses.md#8-speech-separation--pit) |
| `PITLoss` | 固定 K 的 utterance-level 最优排列 | 按最小配对代价监督源集合 | [SS/PIT 合同](Task_and_Model_Specific_Losses.md#8-speech-separation--pit) |

该 `ss` 任务合同与 Composite 的历史 `enable_ss_undesired_energy` 开关分别定义，不能混用二者的 Tensor 接口。
