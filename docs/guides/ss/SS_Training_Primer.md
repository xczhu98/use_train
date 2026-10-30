# SS Training Primer

本文把 SS 的训练链路串一遍：模型 I/O 的三种形态怎么选、执行计划在启动期校验什么、PIT 如何接到 loss、动态仿真的固定 K 合同长什么样。所有命令与清单示例以 [`../../../tasks/ss/README.md`](../../../tasks/ss/README.md) 为准，本文只做导读与定位。

## 1. 选模型：三种 I/O 形态

| 形态 | 模型 | 输入 → 输出 | 说明 |
|---|---|---|---|
| waveform-native | `TDANet` | `[B,T] → [B,K,T]` | 模型直接吃波形，不挂框架 STFT；EchoSet baseline 用与官方 checkpoint 兼容的 16 kHz / 2 ms 拓扑 |
| external-STFT | `TIGER` | `mixture_spectra` → `[B,K,F,T,2]` | 消费框架算好的复数谱，**必须传 `lengths`**；parity 固定 `640/160/640` 几何 |
| 多通道谱进谱出 | `SpatialNet`（离线）/ `OnlineSpatialNet`（因果） | `[B,C,F,T,2] → est_sources_spectra [B,K,F,T,2]` | 参考公开 NBSS 项目（`Audio-WestlakeU/NBSS`）设计；`configs/ss/spatial/` 下有三份 8 kHz NBSS 论文对齐配方（`mhsa`/`ret`/`mamba`） |

三个形态的共同出口是 `SSModelRuntime.separate()`：trainer、推理与 verifier 都通过它拿 source waveform，不要绕过它直接调模型 forward。

## 2. 执行计划与启动期校验

`ResolvedExecutionPlan`（`core/models/execution_plan.py`）在启动期统一校验 Dataset、Model、Loss、采样率、source count 与 STFT 所有权；需要几何或执行计划校验的模型按 `core/models/io_contract.py` 声明 `IO_CONTRACT`。语义：训练和验证不再各自从 YAML 重新推断模型输入域，配置之间的矛盾（例如 waveform-native 模型配了 STFT 注入）在启动期 fail-fast，而不是训练中途报错。

## 3. PIT 与 loss

`PITLoss` 由统一 `loss:` 配置构造（SS 的 loss 与其他任务同一入口），内部包装 `PairwiseL1Loss`（所有估计/参考配对的 L1 cost）或 `PairwiseNegSISDRLoss`（配对负 SI-SDR cost），枚举 K! 种配对取最优监督。三个注册名的输入输出与 zero-reference 语义见 [`../loss/README.md`](../loss/README.md) 的 SS/PIT 合同节；注意该合同与 `CompositeLoss` 的历史键 `enable_ss_undesired_energy`（项目自研机制，与 SS 任务无关）分别定义，不能混用 Tensor 接口。

Trainer 的 loss 直接消费模型原始输出（只检查是否有限），不做幅度 clamp；只有写音频时才在发布边界裁剪。

## 4. 动态仿真的固定 K 合同

`dynamic_ss` 与 SE 共享 Layer A–F 骨架，但 Layer B 是 SS 特有的：Dataset 先从 source pool 采样互不相同的目标源（带 `(dataset_id, speaker_id)` 身份；`speaker_id` 可省略，只保证源互不相同；身份或数量不足时启动失败），固定 K 在 Layer A 就完成输出采样率与固定画布，之后只做多源相对窗口，不在 F 后再次随机裁切。mono online moving scene 的全部 speech source 共用一次采样的 room、microphone position 与 timebase，每路保留独立轨迹。细节与 worker 随机性见 [`../simulation/Multi_Source_Simulation.md`](../simulation/Multi_Source_Simulation.md)。

## 5. 训练、parity 与推理入口

- 训练：`python train.py -C configs/ss/tdanet/tdanet_echoset_offline.yaml`（TDANet + EchoSet offline）、`configs/ss/tiger/tiger_echoset_external_stft.yaml`（TIGER）；dynamic 配方示例 `configs/ss/tdanet/tdanet_dynamic_sisdr.yaml`。
- Parity：官方权重不进仓库，用环境变量注入后显式运行 `python -m pytest -m external tests/test_ss_official_checkpoint_parity.py -q`；TDANet 以 checkpoint 内嵌 `model_args` 为实际模型合同。
- 推理：无根入口，走 `SSExtractor`（`tasks/ss/inference/extractor.py`）；分离指标用共享 `core/metrics/separation.py`。

## 权威口径

- 边界与命令：[`../../../tasks/ss/README.md`](../../../tasks/ss/README.md)。
- 批合同与 loss 合同：[`../../architecture/SS_Architecture.md`](../../architecture/SS_Architecture.md)、[`../loss/README.md`](../loss/README.md)。
- 多源仿真：[`../simulation/Multi_Source_Simulation.md`](../simulation/Multi_Source_Simulation.md)。
