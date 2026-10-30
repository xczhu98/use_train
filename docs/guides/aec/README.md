# AEC 使用指南

AEC（Acoustic Echo Cancellation，声学回声消除）要解决的是：设备自己播放出去的远端语音会被扬声器放出来、经房间传回本机麦克风，形成回声；AEC 模型要做的是从麦克风信号（`mic`）里去掉这段回声，同时保住本机说话人的语音（`nearend`，近端目标）。

四路基本信号：`farend` 远端参考（对方传过来的、即将被播放的信号）、`echo` 回声（`farend` 经播放-房间-录制这条链路后的结果）、`mic` 麦克风混合（近端语音 + 回声 + 噪声）、`nearend` 希望模型输出的近端目标。仿真构数据时按 `mic = nearend + echo + noise` 组装（Layer E/F 的非线性退化之后不再严格成立，见 [`AEC_Data_Simulation.md`](AEC_Data_Simulation.md)）。

本目录收录 AEC 的训练入门、数据仿真与延迟对齐三份指南。训练入口统一是根目录 `train.py`，配置在 `configs/aec/`；数据准备、训练命令与推理命令见任务入口 [`../../../tasks/aec/README.md`](../../../tasks/aec/README.md)。

## 主指南

- [AEC Training Primer](AEC_Training_Primer.md)：把教科书里的自适应 AEC 框图逐环节对应到本仓库代码（`train.py` 路由、`AECTrainer`、`core/losses`），并对比 AEC 与 SE 两条训练流水线。适合第一次读 AEC 代码。
- [AEC Data Simulation](AEC_Data_Simulation.md)：训练数据怎么造——`farend`/`echo` 的物理约束、SER/ERL 电平耦合，以及回声截幅（`aec_echo_clipping`）、播放端设备 EQ（`aec_echo_device_eq`）两个增强的设计。
- [AEC Delay Alignment](AEC_Delay_Alignment.md)：延迟从哪来、怎么处理——磁盘资产的 GCC 预对齐、随机 echo 延迟与样本内 jitter 三层机制，以及 GCC-PHAT 的算法对比和 C++ 后端。

其余缩写：SER（Signal-to-Echo Ratio，信号回声比，越小表示回声越响）、ERL（Echo Return Loss，回声回损，度量扬声器到麦克风的衰减）、ERLE（Echo Return Loss Enhancement，回声抑制量，是评估指标）、STFT（短时傅里叶变换，把波形切成帧再看频谱）在各文档首次出现处另有说明。

## 关键事实（先过一遍，避免误读后面的细节）

- **数据模式**：AEC 只使用 `data.mode: dynamic_aec`（在线仿真造数据）或 `offline_aec`（直接读预混音频）。Dynamic Dataset 只执行 Layer A–F 仿真；`offline_aec` 只读取、重采样并同步裁补预混轨道，不运行 post-mix pipeline。
- **模型输入前端 vs 仿真**：两种模式都在 collate 之后、STFT 之前调用同一个仅 AEC 的前端 `AECDeploymentFrontend`，顺序固定为「可选 Streaming GCC-PHAT → 可选 LinearAEC → STFT」；`inference.py` 在整文件 STFT 前调用同一前端。这个前端属于模型输入预处理，不是数据集仿真，也不会被编号成额外仿真层。
- **Layer D 与 Layer F**：Layer D 是 post-mix 幅度调理，内部固定 `D-normalize`（按配置做 peak normalization）→ `D-guard`（确定性 peak guard），唯一公开事实是 `simulation_by_layer['D']` / `layer_d_facts` 聚合对象。Layer F 是观察/设备退化的权威位置，总开关是 `noisy_post_mix_degradations_enabled`：关闭时整层跳过，开启时由 Dataset 在 CPU 上执行。
- **动态模式 post-mix 顺序**：固定为 `pre-finite → D-normalize → D-guard → E → F → task-post-EF → final-guard → post-finite`，之后才应用 scene-wide `TimeTransformPlan`。
- **模板配置**：`configs/aec/offline_aec_template.yaml` 是 copy-and-fill 模板（与 TSE 的测试 hydrate 白名单无关）。复制后必须填 train/validation 的 `mic_scp` / `farend_scp` / `nearend_scp`（以及可选 `error_scp`）；未填写时 `AECOfflineDataset` 构建报错。

模型接口与结构 SSOT 见 [`architecture/AEC_Architecture.md`](../../architecture/AEC_Architecture.md)。AEC 论文精读及 LocalVQE/DeepVQE 专题分析见 [`paper/aec/`](../../paper/aec/README.md)。A–F 仿真合同及 AEC 模型输入前端边界以 [`guides/simulation/Simulation_Flow_Logic.md`](../simulation/Simulation_Flow_Logic.md) 为准，本文只保留 AEC 的差异。
