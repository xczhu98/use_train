# diagrams 图表索引

根目录保存跨模块或任务级流程图；SE、PASE 与相关模型图统一位于 [`se_models/`](se_models/)。同一张图通常有两个文件：**PNG 用来阅读，`.dot` 是可重绘的 Graphviz 源文件**（Graphviz 是画流程图的工具，DOT 是它的文本描述格式），要改图就改 `.dot` 再重新渲染 PNG。下文只列 PNG、不列 DOT 的图表示它是历史概念图，没有可维护的拓扑源。五任务的仿真与空间特征边界以 [`Simulation_Flow_Logic.md`](../guides/simulation/Simulation_Flow_Logic.md) 为准，图表不另定义 SSL 或 waveform task contract（任务与模型之间的输入输出约定）。canonical 分层图现在覆盖 SE/AEC/TSE/SS/SSL；SSL 的 fixed-array（固定麦克风阵列）profile、scene manifest、IPD/DOA/VAD（空间特征与语音活动标记）与 provenance（数据来源记录）合同仍以仿真正文和对应测试为准。

## 跨模块与任务级流程图

- README 框架总览：[PNG](readme_architecture.png) · [Mermaid 结构与支持边界](readme_architecture.md)；图像模型生成视觉图，架构关系维护在 Mermaid 中。
- SE/AEC/TSE/SS/SSL 当前仿真分层：[PNG](se_aec_tse_simulation_layers_current.png) · [DOT](se_aec_tse_simulation_layers_current.dot)
- SE/AEC/TSE/SS/SSL 公共 post-mix 仿真流程（仅维护 Graphviz 源）：[DOT](simulation_pipeline_flow.dot)
- SE/AEC/TSE 模型接口与结构总览：[PNG](se_aec_tse_model_contracts.png) · [DOT](se_aec_tse_model_contracts.dot)
- Loss YAML 到 Trainer / Composite 运行时工作流：[PNG](loss_composite_runtime.png) · [DOT](loss_composite_runtime.dot)
- AEC 代表模型与 canonical dict ABI（统一用 dict 传频谱：输入 `mic_spectra`/`farend_spectra`，输出 `nearend_estimate_spectra`）：[PNG](aec_model_family_overview.png) · [DOT](aec_model_family_overview.dot)
- TSE speaker cue 变体：[PNG](tse_speaker_cue_variants.png) · [DOT](tse_speaker_cue_variants.dot)
- TSE 训练与推理：[PNG](tse_training_inference_flow.png)

## SE 与 PASE 模型图

下面按模型列出结构图；实现都在 `tasks/se/models/`（PASE 系列在 `tasks/se/models/pase_family/`），标注「仅 PNG」的是历史概念图。

- [BSRNN](se_models/bsrnn_overview.png)、[BS-RoFormer](se_models/bsroformer_overview.png)、[Conformer](se_models/conformer_overview.png)、[TFGridNet](se_models/tfgridnet_overview.png)、[MossFormer2](se_models/mossformer2_overview.png)
- [DeepFilterNet](se_models/deepfilternet_overview.png)、[DeepFilterNet 双滤波机制](se_models/deepfilternet_dual_filter_mechanism.png)
- [FastEnhancer](se_models/fastenhancer_overview.png)、[FastEnhancer RNNFormer](se_models/fastenhancer_rnnformer_mechanism.png)、[FastULCNet](se_models/fastulcnet_overview.png)
- GTCRN 概览：[PNG](se_models/gtcrn_overview.png) · [DOT](se_models/gtcrn_overview.dot)；因果数据流：[PNG](se_models/gtcrn_causal_dataflow.png) · [DOT](se_models/gtcrn_causal_dataflow.dot)
- UL-UNAS 概览：[PNG](se_models/ulunas_overview.png) · [DOT](se_models/ulunas_overview.dot)；NAS 数据流：[PNG](se_models/ulunas_nas_search_dataflow.png) · [DOT](se_models/ulunas_nas_search_dataflow.dot)；[APReLU/cTFA 历史概念图（仅 PNG）](se_models/ulunas_aprelu_ctfa_mechanism.png)
- [PASE 系列演进（历史概念图，仅 PNG）](se_models/pase_family_series_evolution.png)；训练流程：[PNG](se_models/pase_family_training.png) · [DOT](se_models/pase_family_training.dot)；推理流程：[PNG](se_models/pase_family_inference.png) · [DOT](se_models/pase_family_inference.dot)；UniPASE 架构：[PNG](se_models/unipase_architecture.png) · [DOT](se_models/unipase_architecture.dot)
- [GAP-URGENet 概览](se_models/gap_urgenet_overview.png)、[GAP-URGENet 局部机制](se_models/gap_urgenet_mechanism.png)
