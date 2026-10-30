# 任务 × 模式 × 通道对比速查

本文把「五个任务在动态/离线、单通道/多通道两个维度上的数据流差异」压成几张对照表，用于快速回答「这个任务在这个模式下哪条信号是 `[C,T]`、哪条保持 mono、监督目标长什么样」。它是导航层：每个结论的权威描述链接到对应 SSOT 文档，本文不复制细节，也不替代它们。

权威口径：

- 仿真层级与逐任务链路摘要（0.0.6 节）：[Simulation_Flow_Logic.md](Simulation_Flow_Logic.md)。
- TSE/SS/SSL 多源管线、声学角色合同与事实读取：[Multi_Source_Simulation.md](Multi_Source_Simulation.md)。
- 离线 RIR scene manifest 与在线 gpuRIR：[Online_gpuRIR.md](Online_gpuRIR.md)。

下文用 `[T]` 表示单通道波形、`[C,T]` 表示 C 通道 observation、`K` 表示 SS 源数或 SSL 声源数、`R` 表示 TSE 批内目标行数（`R=sum(num_targets)`）。

## 1. 任务 × `data.mode` 矩阵

| 任务 | dynamic | offline | 备注 |
|---|---|---|---|
| SE | `dynamic` | `offline` | offline 为配对 clean/noisy reader |
| AEC | `dynamic_aec` | `offline_aec` | offline 是 cap-only reader，不进 C-aec |
| TSE | `dynamic_tse` | `offline_tse` | offline 读 mixture/target + 独立 enrollment |
| SS | `dynamic_ss` | `offline_ss` | offline 读 mixture + 固定 K 路 mono source |
| SSL | `dynamic_ssl` | **无** | SSL 只有动态仿真模式 |

九种 `data.mode` 的常量定义在 `core/data/constants.py`；offline 路径都不重跑 Layer A–F，由各任务 reader 自行定义读盘、重采样与裁补语义（[Simulation_Flow_Logic.md 1.2 节](Simulation_Flow_Logic.md)）。

## 2. 单通道 vs 多通道（spatial opt-in）

「spatial opt-in」指 split/配置显式声明空间观测后，只有主 observation 扩展为多通道；监督、参考和 cue 默认保持 mono。仿真能生成什么与模型合同能消费什么是两件事，后者以各任务 registry 模型为准。

| 任务 | 多通道时谁是 `[C,T]` | 保持 mono 的信号 | 模型合同边界 |
|---|---|---|---|
| SE | noisy（dynamic 渲染 / offline 直接读取） | clean（spatial 时取 `reference_channel` 的 mono 监督） | SpatialNet 类消费 `[C,F,T]`；mono 模型只消费参考通道 |
| AEC | mic（dynamic 空间模式 / offline 预配对） | nearend、farend、error；echo 若已是 mic 侧 `[C,T]` image 直接入 observation，mono echo source 只在需要空间差异时经独立 loudspeaker RIR 映射 | 当前注册 AEC 模型/entry 仍是单通道模型合同；空间数据不能自动宣称可训练 |
| TSE | mixture（target/interference 共享同一 spatial scene） | 每路 target `[T]`、speaker waveform cue `[T]`（enrollment 用独立近场 RIR，不进 mixture scene） | BSRNN-SPK streaming 路径按模型合同限制；Modular 未实现组合 fail-fast |
| SS | mixture（所有 source 共用同一 room/array scene） | K 路 target 仍投影为 `[K,T]`；offline spatial 需提供 supplied `[K,C,L]` RIR 重建 observation | SpatialNet / OnlineSpatialNet 消费多通道；TDANet / TIGER 保持单通道 / external-STFT 合同 |
| SSL | **不提供 mono 模式**：观测永远是阵列 | 输出不是波形，是 IPD features `[B,T,F,2(C-1)]` | 只消费空间特征，不接受 waveform mono fallback |

三条通用红线：通道顺序只声明不重排（offline split 用 `channel_order` 显式给出物理 mic ID）；farend 是数字参考（电信号），不再套声学 RIR；背景空间默认 `diffuse_coherence`，点噪声源策略在未接入 provider 时 fail-fast，不静默退化。声学角色与 RIR policy 的合同表见 [Multi_Source_Simulation.md「任务能力与声学角色合同」](Multi_Source_Simulation.md)。

## 3. dynamic vs offline 数据流对比

| 对比项 | dynamic | offline |
|---|---|---|
| 数据从哪来 | SCP/JSONL source pool + 现场仿真（Layer A–F；SSL 复用 A/C/D/E/F seam） | 各任务 reader 读取已物化音频与清单 |
| 场景/几何 | 每次抽样：Layer B 场景、room/array、actor 位置与轨迹 | 不做场景抽样；spatial 几何由 split 显式声明（`spatial_enabled` / `array_profile` / `reference_channel` / `channel_order`）或 reader 的 supplied-RIR 合同提供 |
| RIR | `rir_backend=offline`（typed scene manifest）或 `online`（gpuRIR 子进程 IPC，同房间不同 actor） | SS offline spatial 需 supplied source×mic RIR bank；其余任务不重卷 RIR |
| 时长语义 | 固定画布（Layer A-3 完成；TSE/SS 在 Layer A 完成采样率与画布，之后不再随机裁切） | SE 短样本原长返回由 collate padding；AEC cap-only 取公共有效长度；TSE/SS 按 cap 同段裁切；tile 语义各任务不同（见 1.2 节） |
| 部署前端 | AEC：Dataset 不执行；batch 由 trainer 在 STFT 前跑 canonical `AECDeploymentFrontend` | 同左——offline AEC batch 仍进同一前端（Dataset 本身不跑） |
| 批输出 | SE 7 元组 / AEC 11 或 12 元组 / TSE `TSEBatch` / SS `SSBatch` / SSL `SSLBatch` | 同左（批合同不随 mode 改变） |

## 4. 监督目标形态对照

| 任务 | 目标形态 | 空间扩展时的监督 | 绕过 E/F 的部分 |
|---|---|---|---|
| SE | clean 波形 `[T]` | 仍 mono：取 `reference_channel`；`target_type` 支持 `early` / `dry` / `rts`（rts 仅 SE spatial path 消费） | 同所有任务：E/F 只改 observation，不改 clean |
| AEC | nearend 波形 `[T]` | nearend 始终 mono | nearend 目标经 Layer D、绕过 E/F；导出的 echo audit stem 同步经 `D-normalize`/`D-guard`、不进 E/F |
| TSE | target 波形，按 R 行 `[R,T]` | 每路 target 保持 `[T]`（mixture 的空间化不传染监督） | 同左 |
| SS | K 路 source 波形 `[K,T]`；纯背景场景 K 路全零 target | 仍 `[K,T]` mono | 同左 |
| SSL | 空间谱 target `[B,T,F,K,2]`（real/imag），由 direct-path 几何投影 | 无波形监督；VAD/DOA 同为投影产物 | observation 经共享 E/F，监督由投影层另行计算 |

## 5. 事实（trace / audit）读取位置

| 任务 | 事实位置 |
|---|---|
| SE / AEC | 任务 audit 字段（`simulation_by_layer` / `layer_d_facts` 等聚合对象） |
| TSE / SS | `trace.metadata_dict()["scene"]`（`source_plans`、`sir`、`activity`、`layer_d_facts`、`layer_e`、`layer_f`、`decomposable_final`；capture 开启时经 `trace.get_source("scene").stems` 读最终轨道） |

两个位置不通用，不要把同一取值路径硬编码到五个任务。细节见 [Multi_Source_Simulation.md「事实读取」](Multi_Source_Simulation.md)。

## 使用建议

- 改某个任务的仿真行为前，先查第 1、3 节确认该 mode 下执行权在 Dataset 仿真还是 reader。
- 做多通道改造前，先查第 2 节右列的模型合同，再查声学角色合同表，区分「仿真能生成」与「模型能消费」。
- 判断某个增强是否只影响模型输入，查 [Simulation_Flow_Logic.md 第 2 节](Simulation_Flow_Logic.md) 的逐步 clean 是否改动表。
