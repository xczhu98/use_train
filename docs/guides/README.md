# guides 使用指南

本目录保存「怎么跑」的文档：训练、数据仿真、推理、评估、部署和学习指南。论文精读不在这里，统一位于 [`paper/`](../paper/)。

内容按五个任务加框架级主题组织。五个缩写分别是：

| 缩写 | 全称 | 解决的问题 |
|---|---|---|
| SE | Speech Enhancement，语音增强 | 一段录音混进了噪声、混响或设备失真，保留说话人、去掉干扰 |
| AEC | Acoustic Echo Cancellation，回声消除 | 从麦克风信号里去掉设备自己播放出去、又经房间传回来的回声 |
| TSE | Target Speaker Extraction，目标说话人提取 | 多人混音中只保留指定说话人的声音 |
| SS | Speech Separation，语音分离 | 把一段多人混音拆成固定数量的语音源 |
| SSL | Spatial Sound Localization，空间定位 | 用固定麦克风阵列估计声源方向 |

读之前先认识两个词：

- **`data.mode`**：配置里声明「这份数据从哪来」。`dynamic*` 是训练时现场合成受损音频，每次取到的样本都不一样；`offline*` 是直接读取事先配好的音频文件。五个任务共九种模式，完整清单见 [simulation](simulation/README.md)。
- **Layer A–F**：动态仿真把「一段干净语音变成训练样本」拆成 A 到 F 六个阶段（读语音、抽场景、加混响和噪声、加信道与设备退化等）。想知道每一层做什么，读 [Simulation_Flow_Logic.md](simulation/Simulation_Flow_Logic.md)。

## 任务指南

每个任务有三层入口：guides 层的导读（本目录）、任务 README 的可复制命令、architecture 层的结构 SSOT。

| 任务 | guides 导读 | 任务 README（命令 SSOT） | 架构文档 |
|---|---|---|---|
| SE | [se/](se/README.md) | [tasks/se/README.md](../../tasks/se/README.md) | [SE Architecture](../architecture/SE_Architecture.md) |
| AEC | [aec/](aec/README.md) | [tasks/aec/README.md](../../tasks/aec/README.md) | [AEC Architecture](../architecture/AEC_Architecture.md) |
| TSE | [tse/](tse/README.md) | [tasks/tse/README.md](../../tasks/tse/README.md) | [TSE Architecture](../architecture/TSE_Architecture.md) |
| SS | [ss/](ss/README.md) | [tasks/ss/README.md](../../tasks/ss/README.md) | [SS Architecture](../architecture/SS_Architecture.md) |
| SSL | [ssl/](ssl/README.md) | [tasks/ssl/README.md](../../tasks/ssl/README.md) | 暂无独立架构文档（见 ssl/ 导读的说明） |

## 框架级指南

跨任务的主题在这里：

| 主题 | 入口 | 覆盖 |
|---|---|---|
| 训练 | [training/](training/README.md) | `train.py` 路由链、TrainingEntry 合同、trainer 能力地图、optimizer/scheduler、checkpoint 格式 |
| 运行时参考 | [reference/Runtime_Environment_Variables.md](reference/Runtime_Environment_Variables.md) | 环境变量全表（DDP/NCCL、AMP、数据路径、诊断开关）与 DDP 内存预算 |
| 推理 | [inference/](inference/README.md) | 根 `inference.py`、合同五件套、artifact 身份与 `skip-existing`、流式 seam |
| 评估 | [evaluation/](evaluation/README.md) | `evaluate/` 一键流水线、`core/metrics/` 分层、第三方指标、评估资产纪律 |
| 训练目标 | [loss/](loss/README.md) | core 与任务专属 Loss registry inventory、YAML 配置契约、逐实现专题 |
| 数据仿真 | [simulation/](simulation/README.md) | 五任务九种 `data.mode`、RIR 语义角色与 backend、Layer A–F 合同、任务×模式×通道对比速查 |
| 部署 | [onnx/README.md](../../onnx/README.md) | ONNX 导出、三路精度验证、发布门禁（独立模块，guides 不复制） |
| 配置 | [configs/README.md](../../configs/README.md) | 配置归属、mode 路由、命名与 `base_config` 继承语义 |
| 数据准入 | [Audio Filter Pipeline](../../tools/audio_filter_pipeline/README.md)、[Noise Filter Pipeline](../../tools/noise_filter_pipeline/README.md) | 语音质量筛选、噪声准入与训练源路由 |
| 预训练权重 | [reference/Pretrained_Weights.md](reference/Pretrained_Weights.md) | 未入库权重的目标路径、官方下载入口与验证命令 |
| 第三方资产 | [reference/Third_Party_Assets.md](reference/Third_Party_Assets.md) | 评估、VAD、Noise Filter、REAL-TSE 外部资产与克隆入口 |

## 任务 × 主题矩阵

按「我在做哪个任务、要做哪件事」直接定位；空格表示该组合暂无专门入口，以括号内的 SSOT 为准。

| 任务 | 训练 | 数据 / 仿真 | 推理 | 评估 | 部署 |
|---|---|---|---|---|---|
| SE | [se/](se/README.md)、[training/](training/README.md) | [simulation/](simulation/README.md) | [inference/](inference/README.md)、根 `inference.py` | [evaluation/](evaluation/README.md)、`eval_pipeline_se` | [onnx/](../../onnx/README.md)、[VoiceBank Quickstart](se/VoiceBank_Deployment_Quickstart.md) |
| AEC | [aec/](aec/README.md)、[training/](training/README.md) | [simulation/](simulation/README.md)、[AEC Data Simulation](aec/AEC_Data_Simulation.md) | [inference/](inference/README.md)、根 `inference.py` | [evaluation/](evaluation/README.md)、`eval_pipeline_aec` | [onnx/](../../onnx/README.md) |
| TSE | [tse/](tse/README.md)、[training/](training/README.md) | [Multi_Source_Simulation](simulation/Multi_Source_Simulation.md) | `TSEExtractor`（[tasks/tse 推理节](../../tasks/tse/README.md)） | `eval_pipeline_tse`（[tasks/tse 评测节](../../tasks/tse/README.md)） | TorchScript/ONNX 路径（[tasks/tse 推理节](../../tasks/tse/README.md)） |
| SS | [ss/](ss/README.md)、[training/](training/README.md) | [Multi_Source_Simulation](simulation/Multi_Source_Simulation.md) | `SSExtractor`（[ss/](ss/README.md)） | 暂无一键流水线（以 `core/metrics/separation` 与 `SSExtractor` 为入口） | 暂无 |
| SSL | [ssl/](ssl/README.md) | [simulation/](simulation/README.md)、[Online_gpuRIR](simulation/Online_gpuRIR.md) | 暂无（无公开文件推理入口） | 暂无（entry 拒绝 waveform val_metrics，验证目标非波形域） | 暂无（无 ONNX 发布路径） |

## 按目的阅读

- **第一次跑通一个任务**：从上面任务指南表进任务 README，里面有可复制的数据准备、训练与推理命令；SE 从 [VoiceBank Quickstart](se/VoiceBank_Deployment_Quickstart.md) 开始，AEC 从 [AEC Training Primer](aec/AEC_Training_Primer.md) 开始。
- **改数据或仿真行为**：先做跨任务对比定位——[任务×模式×通道对比速查](simulation/Task_Mode_Channel_Comparison.md) 回答「这个模式下哪条信号是 `[C,T]`、监督长什么样」；再读 [Simulation_Flow_Logic.md](simulation/Simulation_Flow_Logic.md) 的 Layer A–F 合同，按任务读 [Multi_Source_Simulation](simulation/Multi_Source_Simulation.md) 或 [AEC Data Simulation](aec/AEC_Data_Simulation.md)；数据准入问题进两个 pipeline 文档。
- **配训练目标（loss）**：读 [loss/](loss/README.md) 的配置契约，再按族查对应专题页。
- **写训练配置**：先读 [configs/README.md](../../configs/README.md) 的归属与继承语义，再读 [training/](training/README.md) 的路由与 entry 合同。
- **推理与产物复用**：读 [inference/](inference/README.md)，重点是 artifact 身份与 `skip-existing` 失效语义。
- **评估与指标口径**：读 [evaluation/](evaluation/README.md)，比赛/榜单指标以对应流水线文档与任务 README 的口径节为准。
- **部署上线**：读 [onnx/README.md](../../onnx/README.md)，发布必须过对应验证 gate，禁止 `--no-verify`。

## 维护纪律与防漂移

本目录的文档是导航与解释层，不是第二份事实清单。改代码或改文档时遵守：

1. **事实 SSOT 顺序**：代码与测试 > 任务 README > guides 本文。本文与运行时冲突时，以源码和测试为准并修正文档，不反向改代码迁就文档。
2. **枚举事实必须同步**：任务数（5）、`data.mode` 清单（9 种）、loss 注册数、registry 模型清单等枚举值，改代码时同步改对应指南；本文出现的枚举值以 `core/data/constants.py`、`tasks/registry.py`、`core/losses/` 为准。
3. **不复制事实清单**：各专题页（loss、simulation）维护各自的事实清单，本页只做索引与一句话定位；新增文档后在「任务指南」「框架级指南」两表和 [`docs/README.md`](../README.md) 回填入口。
4. **链接只用仓库相对路径**；新增图片与图源码放对应 `assets/`，生成图与非原文证据按 `paper/` 的分层口径标注。
5. **论文与榜单指标不得写成项目复现**；本机未验证的路径写「未验证」，不写「通过」。

论文精读与工程指南分开维护；TSE 和 Survey 的论文入口分别见 [`paper/tse/`](../paper/tse/) 和 [`paper/survey/`](../paper/survey/)；多通道增强（原 MCSE）论文已并入 [`paper/se/`](../paper/se/)。模型结构见 [`architecture/`](../architecture/)。需要运行任务时，优先从根 README、`configs/README.md` 和对应任务 README 进入。
