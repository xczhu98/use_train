# SE 使用与学习指南

**SE**（Speech Enhancement，语音增强）解决的问题是：一段录音里混进了噪声、混响或设备
失真，希望尽量保留说话人的声音、去掉这些干扰。本页是 SE 文档的入口——想先把模型跑起来
看「操作指南」，想理解模型为什么这样设计看「学习指南」，最后一节说明 SE 与其他四个任务
共享的数据仿真边界。

## 操作指南

- [VoiceBank Deployment Quickstart](VoiceBank_Deployment_Quickstart.md)：以 VoiceBank-DEMAND
  数据集为例，走通 GTCRN 的数据准备 → 训练 → 评估 → 批量推理 → ONNX 导出
  （ONNX 是一种跨平台模型文件格式，导出后可以脱离 PyTorch 部署）。第一次上手建议从这份开始。
- [URGENT2026 Guide](URGENT2026_Guide.md)：URGENT2026 比赛场景下 BSRNN 的推理、多版本批量对比与官方口径指标评估。
- [SE vs AEC Training Framework](SE_vs_AEC_Training_Framework.md)：SE 与
  **AEC**（Acoustic Echo Cancellation，回声消除）在训练入口、数据、前向和损失上的分叉对照。

## 学习指南

- [PASE 系列学习指南](pase_family_learning_guide.md)：PASE、StuPASE、UniPASE 与 GAP-URGENet
  四个生成式模型的原理、训练、复现边界和上手流程。
- [PASE 论文精读](../../paper/se/Nanjing/PASE.md)：PASE 原始方法、训练目标与复现证据。
- [Xiaobin-Rong 模型系列分析](../../architecture/analysis/Rong_Model_Family_Analysis.md)：端侧线与生成式大模型线的机制演进。

## 数据模式与仿真边界

`data.mode` 是配置里声明「数据从哪来、怎么准备」的字段：`dynamic` 表示训练时现场合成
带噪语音，`offline` 表示直接读取已经配好对的 clean/noisy 文件。全仓库共九种 mode，分属五个任务：

| data.mode | 任务 |
|---|---|
| `dynamic` / `offline` | SE |
| `dynamic_aec` / `offline_aec` | AEC |
| `dynamic_tse` / `offline_tse` | TSE |
| `dynamic_ss` / `offline_ss` | SS |
| `dynamic_ssl` | SSL |

SE 只用 `dynamic` / `offline`，其余七种属于别的任务，不能拿来训练 SE。

「Layer A–F」是动态仿真管线的分层命名：A 准备干净主语音，B/C 抽场景角色与混响、组装观测信号，D/E/F 逐层加上后置退化。SE 的 Dynamic 数据按固定 post-mix 顺序执行，完整
stage 列表、`TimeTransformPlan`（统一 target 采样率与裁切/拼接）以及各任务的共享约定以
[仿真流程逻辑](../simulation/Simulation_Flow_Logic.md) 为准；SE 侧只需要记住三点：

- SE 的 `task-post-EF` 当前是 waveform no-op（不改变波形），仿真统一只有 Layer A–F。
- Layer D 是 post-mix 幅度调节层，内部固定 `D-normalize → D-guard`：前者按配置执行 peak normalization，后者固定执行 deterministic peak guard。Layer D 的公开 facts 只通过 `simulation_by_layer['D']` / `layer_d_facts` 聚合对象发布。
- `noisy_post_mix_degradations_enabled` 是 Layer F 总开关：关闭时整层跳过，开启时由 Dataset 执行 observation/device degradation。该开关不控制 Layer D/E。

共享时间布局、Music 路由与 TSE/SS 的差异见 [多源共享仿真](../simulation/Multi_Source_Simulation.md)。SE 仍由自身 Adapter 编排单路主语音。

SE canonical 论文精读统一归档在 [`paper/se/`](../../paper/se/README.md)，PASE 系列位于 [`paper/se/Nanjing/`](../../paper/se/Nanjing/)。模型接口与结构见 [`architecture/SE_Architecture.md`](../../architecture/SE_Architecture.md)，仿真层级、SE offline spatial 数据合同及 mc_sim 评估集生成/评估入口见 [`../simulation/Simulation_Flow_Logic.md`](../simulation/Simulation_Flow_Logic.md) 和 [`../simulation/Multi_Source_Simulation.md`](../simulation/Multi_Source_Simulation.md)；cIRM loss 的公式与输入边界见 [`../loss/Task_and_Model_Specific_Losses.md`](../loss/Task_and_Model_Specific_Losses.md)。
