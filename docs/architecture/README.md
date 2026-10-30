# architecture 架构文档

本目录回答「模型长什么样、接口怎么约定」，内容分两类：模型结构、接口与张量语义（连同对应的源码位置），以及跨模型的机制分析。这里不放操作手册，也不放论文正文——训练、部署和学习流程见 [`guides/`](../guides/)，论文精读见 [`paper/`](../paper/)。

本目录的所有描述都以代码为准（SSOT，Single Source of Truth，唯一事实来源）：文中写出的源码位置就是该接口的权威实现，描述与代码不一致时以代码为准。

五个任务的源码各占一个目录：`tasks/se/`（Speech Enhancement，语音增强）、`tasks/aec/`（Acoustic Echo Cancellation，回声消除）、`tasks/tse/`（Target Speaker Extraction，目标说话人提取）、`tasks/ss/`（Speech Separation，语音分离）、`tasks/ssl/`（Spatial Sound Localization，空间声源定位）。Python 里按 `tasks.<task>` 导入；旧的顶层任务包（如 `se/`、`aec/`）已删除，不提供兼容入口。跨任务复用的实现放在 `core/`，配置放在 `configs/<task>/`。

`tasks/descriptors.py` 是任务清单：每个任务一条记录，写明训练入口、模型注册函数和它负责的 `data.mode`。`tasks/registry.py` 按 `core/data/constants.py` 里的公开 mode 常量路由到对应任务；`train.py` 只加载被选中任务的入口（惰性加载，不会把五个任务的模型全部导入），`core.models.register_task_models()` 读的是同一份清单。任务级 `supports_streaming` / `supports_external_stft` 只说明该任务包里有没有流式推理、是否消费框架的 STFT；某个具体模型是否可用，仍以 `ModelIOContract`（模型自己声明的输入输出字段）、`ResolvedExecutionPlan`（启动时把该声明与 `data.mode`、采样率、Dataset/Loss 能力、顶层 STFT 合并成的执行计划）和部署验证结果为准。

`core/inference/` 负责统一的文件推理生命周期：`InferenceSpec`、`FileInferenceItem`、`PreparedOutput` 是任务与公共 runner 之间的数据契约（推理什么、输入怎么准备、产出什么），再加 artifact/input identity（产物与输入的内容身份，用于指纹比对）、reuse（指纹一致就复用上次结果）、原子发布（产物要么完整可见、要么完全不出现）、`run_manifest.json` 和 `inf.scp`。任务侧 Adapter 只负责模型输入输出语义，manifest、输出锁和 `--skip-existing` 由公共层处理；`evaluate/` 负责评测编排，`evaluate/multi_gpu_inference.py` 负责分片 worker 生命周期，TSE 专属 sidecar（每个结果附带的描述文件）由 `tasks/tse/inference/sidecar.py` 从公共 `InferenceReport` 派生，`onnx/` 保持独立的模型 artifact/runtime 发布门禁。

跨仓库引用要写全：外部脚本和配置里的 dotted class path（点号分隔的类路径，例如 `tasks.<task>.*`）必须以 `tasks.<task>` 开头；`configs/<task>/`、`data/<task>/`、`exp/<task>/` 目录和 YAML 里的任务键各自含义不变。TSE 自带的 speaker checkpoint 放在 `tasks/tse/` 内。推理指纹（对配置、checkpoint、输入清单等算出的内容摘要，用来判断能否复用上次的推理产物）覆盖任务源码和 registry 边界，所以任务源码变化后旧 manifest 会失配并重新生成，不能用 `--skip-existing` 绕过校验。

## 主文档

- [SE Architecture](SE_Architecture.md)：SE 模型接口、结构对比与可复用模块。
- [AEC Architecture](AEC_Architecture.md)：AEC 模型接口，以及 CAGCRN、DeepVQE 等结构说明。
- [TSE Architecture](TSE_Architecture.md)：TSE 的模型路由、训练挂载和模型接口。
- [SS Architecture](SS_Architecture.md)：SS 的模型路由、固定源数量（fixed source count）、PIT（permutation invariant training，排列不变训练：分离结果与目标的对应关系未知，训练时对所有配对取最优）和 TDANet/TIGER/SpatialNet 前端边界。
- [任务化仓库目录结构](Task_Namespace_Tree.md)：当前工作区的实际目录布局，包括仓库边界、任务公共入口、五个任务包结构、共享层职责和边界检查命令。
- SSL：暂无独立架构文档；任务包说明见 [`tasks/ssl/README.md`](../../tasks/ssl/README.md)，SSL 数据合同见「任务合同矩阵」（`AGENTS.md`）与 [`guides/simulation/Simulation_Flow_Logic.md`](../guides/simulation/Simulation_Flow_Logic.md)。

## 当前真实验证状态

读下面的结论时先分清三类：只跑过代码的（代码验证）、用真实数据或 checkpoint 跑过的（真实资源验证）、依赖本机 GPU 或分布式环境的（环境能力验证）。

- AEC DeepVQE：用历史 32 kHz checkpoint 完成真实 mic/farend 推理和 streaming smoke（流式推理冒烟测试），并用 1000 帧、中途两次 reset 的输入做三路对齐——离线 PyTorch、流式 PyTorch、ONNX Runtime 流式三条路径输出一致——同时校验 runtime metadata；另用当前 canonical 配置副本跑通一 epoch CPU 训练链路。
- 历史实验模型：不在当前支持矩阵里的 checkpoint 不能靠别名映射或兼容 loader 混进 registry；必须先走正式的模型迁移、注册、部署和测试。
- 所有结论都要标注适用范围：socket/Gloo（分布式训练的通信后端）、外部数据、GPU 或第三方资源缺失时，对应路径只能写成「未验证」，不能写成业务链路通过。

## 跨任务契约图

- [SE/AEC/TSE 模型接口与结构总览 PNG](../diagrams/se_aec_tse_model_contracts.png) · [DOT source](../diagrams/se_aec_tse_model_contracts.dot)：用于比较三任务的 registry（模型名到模型类的注册表）、Framework/Trainer 所有权（谁持有时频变换、谁只做模型前向）及模型输入输出边界；PNG 可由该 Graphviz source 重现。

该图只提供跨任务契约层视图；各主文档及其中保留的任务专属结构图、机制图继续作为模型细节层入口，不由总览图替代。

训练启动时，现代模型通过 `ModelIOContract` 声明输入/输出字段、domain（每个字段是波形、频谱、embedding 还是标签等）、frontend owner（STFT 归框架还是模型）与 forward kind（估计、loss-returning、GAN 生成器/判别器等形态）；`ResolvedExecutionPlan` 将该声明与 `data.mode`、采样率、Dataset/Loss capability 和顶层 STFT 合并为一份不可变执行计划。SE、AEC、TSE、SS Trainer 的模型路由都消费该计划。任务热路径不再提供旧顶层任务包、旧 inference contract 或隐式配置迁移。

共享 validation 使用固定阶段协议：`prepare → model/output preflight → loss → task hooks → metric pairs`。每个阶段都无条件执行跨 rank 异常汇聚（多进程训练时先对齐各进程的异常状态再一起退出，避免某个进程提前失败、其他进程卡在通信里）；可能包含 loss collective（跨进程归约）的路径，只有在 model/output preflight 阶段所有 rank 都成功后才进入。

## 机制分析

- [Xiaobin-Rong 模型系列分析](analysis/Rong_Model_Family_Analysis.md)：GTCRN、UL-UNAS、PASE、StuPASE、UniPASE 与 GAP-URGENet 的机制演进。
- [LocalVQE 与 DeepVQE 对比](../paper/aec/analysis/LocalVQE_vs_DeepVQE_Model_Analysis_2026-07-17.md)：AEC 论文与实现对照分析。

架构事实以当前代码为准；接口或张量语义变化时，应同步更新对应主文档。
