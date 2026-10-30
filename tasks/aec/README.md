# 🔇 AEC — Acoustic Echo Cancellation

<p align="center">
  <strong>声学回声消除（AEC）的训练、模型与推理模块</strong><br>
  与 <code>tasks/se/</code> 共用基础设施，支持 DeepVQE、CAGCRN、LocalVQE、UltraDPCAEC、DBGNet 等模型。
</p>

---

## 📑 目录

- [先读这里：AEC 是什么](#-先读这里aec-是什么)
- [数据准备](#-数据准备)
- [训练](#-训练)
- [推理](#-推理)
- [模型清单](#-模型清单)
- [与 SE 已对齐的部分](#-与-se-已对齐的部分)
- [四个生产 family 的对齐机制](#-四个生产-family-的对齐机制)
- [CAGCRN 与官方对齐](#-cagcrn-与官方对齐)
- [DBGNet](#-dbgnet)
- [工业仿真 (`data.mode: dynamic_aec`)](#aec-dynamic-simulation)
- [离线三元组 (`data.mode: offline_aec`)](#aec-offline)
- [帧级 State Detector](#aec-state-detector)
- [AEC 样本格式](#-aec-样本格式)
- [ONNX 部署](#-onnx-部署)
- [构建 C++ 延迟估计扩展](#-构建-c-延迟估计扩展)
- [BibTeX 引用](#-bibtex-引用)

---

## 🧭 先读这里：AEC 是什么

通话或开会时，对方的声音从扬声器放出来，又被本地麦克风重新录进去，这就是**回声（echo）**。**AEC（Acoustic Echo Cancellation，声学回声消除）**要做的就是：从麦克风信号里去掉那部分回声，只留下本地说话人的近端语音，同时尽量压掉背景噪声。比 SE（语音增强，只降噪）多了一路“远端参考信号”，因此数据与模型输入都由两路音频组成。

后文反复出现的词先在这里说清：

| 术语 | 含义 |
|---|---|
| `mic` | 麦克风录到的信号 = 近端语音 + 回声 + 噪声 |
| `farend`（也叫 reference / lpb） | 远端参考信号，即送给扬声器播放的那一路音频 |
| `echo` | farend 经扬声器播放、再被麦克风拾取后形成的回声 |
| `nearend` | 本地说话人的干净语音，是训练目标（target） |
| 双讲（double-talk, DT） | 远端与近端同时说话，评估 AEC 时的重点场景之一 |
| ERLE | Echo Return Loss Enhancement，回声抑制量，衡量回声被压低了多少 dB |
| SCP | 纯文本音频清单，每行是 `uid 采样率 时长 路径`；本项目的数据集都靠清单索引音频 |
| STFT / iSTFT | 短时傅里叶变换及其逆变换：把波形转成时频谱交给模型处理，再变回波形 |
| Layer A–F | 共享波形仿真的六个分层（A 语音预处理、B 场景分支、C 声学混合、D post-mix 幅度条件、E 信道退化、F 设备/后处理退化） |
| ONNX | 跨框架的模型交换格式，用来把训练好的模型交付给 ONNX Runtime 等推理引擎 |
| DDP | PyTorch 多卡数据并行训练（`torchrun --nproc_per_node=N`） |
| checkpoint | 训练保存的权重文件（本项目为 `best_model.tar` / `latest_model.tar`） |

最短路径（每一步的完整参数见对应小节）：

1. **准备数据**：`dynamic_aec` 需要 `speech_scp`、`noise_scp`、`rir_scp`（采样率必须与配置一致，16 kHz 与 32 kHz 清单不能混用）以及 `real_farend_echo_pair_json`（farend/echo 立体声对，ch0=echo、ch1=farend）。本机可用的示例路径见 [`configs/aec/microsoft_challenge/README.md`](../../configs/aec/microsoft_challenge/README.md) 的「数据清单」表。
2. **训练**：`CUDA_VISIBLE_DEVICES=0 python train.py -C configs/aec/microsoft_challenge/cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml`，日志与 checkpoint 写到配置里的 `exp_path`。
3. **推理**：`python inference.py -C <config> --checkpoint <ckpt> --input_mic_dir <mic_dir> --input_far_dir <far_dir> --output_dir <out>`，每个输入产出一个增强后的 wav。
4. **部署**：正式发布 ONNX 必须走模型专属的 `onnx/verify/verify_3way_*.py` 三路验证（见「ONNX 部署」）；`onnx/export.py --output` 只用于调试，不是发布件。

---

## 📦 数据准备

`dynamic_aec` 训练使用的清单与 SE 的 `dynamic` 基本相同（`speech_scp` 干净近端语音、`noise_scp` 噪声、`rir_scp` 房间脉冲响应，采样率必须与配置的 `data.sample_rate` 一致），另外多一份 `real_farend_echo_pair_json`（farend/echo 立体声对）。四类清单的字段含义与本机可用的示例路径见 [`configs/aec/microsoft_challenge/README.md`](../../configs/aec/microsoft_challenge/README.md) 的「数据清单」表。

`real_farend_echo_pair_json` 由下面这条命令从已录好的立体声文件生成（ch0=echo，ch1=farend），输出 canonical 的 `{"pairs": ["stereo.wav"], "meta": {...}}` JSON：

```bash
python -m tools.prepare_data_tools.generators.echo.build_real_farend_echo_pair_manifest \
    /path/to/echo_corpus_10s \
    --out data/echo_manifest.json
```

开启 `--val_ratio` 划分时，该工具默认输出 `real_farend_echo_pair_train.json` / `real_farend_echo_pair_val.json`（细节见 `--help`）。

正式 AEC 评估需要成对的官方清单（mic 与 lpb 两路都要有），用下面这条命令按子集生成：

```bash
export USE_TRAIN_TEST_DATA_ROOT=/path/to/test-data
python tools/prepare_data_tools/scp/prepare_testset_scps.py \
    --datasets aec_official --aec_subsets all \
    --aec_testset_root data/aec/AEC_Challenge/datasets \
    --output-dir "$USE_TRAIN_TEST_DATA_ROOT"
```

该命令生成 `doubletalk`、`farend-singletalk`、`nearend-singletalk` 三个子集，配对数分别为 **300**、**300**、**200**。`data/aec/AEC_Challenge/datasets/blind_test_set_icassp2023.scp` 是一份 **2,400** 条 mixed inventory（800 mic + 800 lpb + 800 enrl），可用于 inventory 检查，但不能替代正式的双输入 AEC 评估。

`noise_scp` 的准入与发布由 Noise Filter Pipeline 负责，合同见根 README 与 [`tools/noise_filter_pipeline/README.md`](../../tools/noise_filter_pipeline/README.md)；离线三元组的数据准备工具见「离线三元组 (`data.mode: offline_aec`)」。

---

## 🚀 训练

训练入口是仓库根目录的 `train.py`，用 `-C` 指定配置文件；配置里的 `data.mode` 决定走 `dynamic_aec`（在线仿真造数据）还是 `offline_aec`（读预混三元组）。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/aec/microsoft_challenge/cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml

# DBGNet Small，32 kHz
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/aec/microsoft_challenge/dbgnet/dbgnet_small_32k_10ms_delay80ms_loss_hybrid.yaml

# 会议 VoIP DeepVQE，32 kHz
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/aec/microsoft_challenge/deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml
```

AEC 配置必须显式声明 `data.mode: dynamic_aec` 或 `offline_aec`，并定义顶层 `stft` 块。两种模式都会在 collate 后、STFT 前解析仅 AEC 使用的 `AECDeploymentFrontend`（模型输入预处理，不属于 Dataset 仿真）：只有 `aec_frontend.delay_alignment.enabled` 为 true 或配置了 `aec_frontend.linear_aec` 时才实际执行，否则 mic/farend 严格直通且不报 runtime error，`inference.py` 仍会逐文件调用它。runtime LinearAEC 的配置路径为 `aec_frontend.linear_aec`；非线性/设备 GPU 增强会使其 residual 失效，因此应使用其 `gpu_batch` PFDKF 后端，或关闭 GPU 增强。

会议 DeepVQE 32 kHz 配置要点：

- 使用预对齐 manifest，不再进行第二次在线 Dataset GCC 对齐；仅在 echo 路径施加额外 echo/reference 错位，使用纯 `Beta(2,4)`，范围 `[0.0,80.0] ms`。
- `align_delay: 10` 包含当前帧及前 9 帧，对应约 90 ms 的内部历史跨度，另有 3% 条件门控的纯 `sudden_jump` 抖动（`±20 ms`、segment 最短 1000 ms）。
- `AECDeploymentFrontend` 解析为 no-op；`stft.center=true` 要求部署前端实现一致的 10 ms lookahead。该 delay 分布是会议硬件工程先验，不是论文分布或实测的本地麦克风阵列延迟分布。
- `max_val_batches: 0` 表示 loss validation 不限制 batch 覆盖。
- 32 kHz DeepVQE 与 DBGNet 配置使用 `rirs/32k`；16 kHz CAGCRN、DBGNet 与 LocalVQE 配置使用 `rirs/16k` 及匹配的 16 kHz echo 清单。
- DBGNet Small 32 kHz baseline 的 STFT（`640/320/640`）、ERB（`81/80`）、两分量 `CompositeLoss` 与参数量/MACs（`0.576 M` / `1.348 G`，关闭 state head 后按论文规则复测）见下文「模型清单」；它的 echo delay、`sudden_jump` 抖动与 no-op deployment frontend 与会议 DeepVQE 相同（历史 `hybrid` YAML 文件名保持不变）。

详细仿真语义与噪声角色分流，以及十份生产配置的逐项参数对照（`配置速查表` / `关键约定`）见 [`configs/aec/microsoft_challenge/README.md`](../../configs/aec/microsoft_challenge/README.md)。

验证阶段的配置细节（都在 `val_metrics` 下）：`max_utterances_metrics` 与 `max_utterances_erle` 分别是通用侵入式指标和 ERLE 的独立全局 quota。`max_utterances_metrics` 省略或为 `null` 时回退 `max_utterances`；`max_utterances_erle` 省略时回退。每个全局 quota 都精确分配到各 rank，余数优先分给低 rank。ERLE 仅在 `val_metrics.interval` 命中的轮次运行，使用 validation epoch 级 spawn `ProcessPool`，并在 epoch 结束或异常时关闭。DBGNet state 指标始终先跨 rank 汇总 support：全局无有效 state label 时只记录 `state_support_count=0` 及逐类 support，不伪造 accuracy/F1=0；有 support 时同时记录历史 all-class `state_macro_f1` 与仅对标签中实际出现类别求均值的 `state_macro_f1_supported`。

---

## 🔍 推理

AEC 是双输入任务，所以每条推理命令都要同时给出 mic 和 farend 两路输入。根 `inference.py` 通过任务 descriptor 驱动的 adapter factory 复用 `tasks/aec/inference` 的模型 I/O；公共 runner 负责 manifest、output lock、`skip-existing` 与原子输出，产物目录下会写出增强后的 wav 与 `run_manifest.json`。

```bash
# 双输入: mic + farend, 目录模式
python inference.py -C /path/to/config.yaml --checkpoint /path/to/best_model.tar \
    --input_mic_dir data/mic --input_far_dir data/farend --output_dir results/aec

# 双输入: mic + farend, SCP 模式
python inference.py -C /path/to/config.yaml --checkpoint /path/to/best_model.tar \
    --input_mic_scp data/mic.scp --input_far_scp data/far.scp \
    --scp_root_dir /path/to/audio --output_dir results/aec
```

- `--sr`：AEC 只能重复声明配置中的 `data.sample_rate`；与配置不一致会立即报错（SE 才允许覆盖）。
- `--aec-review-multichannel`：AEC 审听模式。实际消费的 reference 为零时写 `[mic, enhanced]`，否则写 `[mic, reference, enhanced]`；其多通道文件不能当作普通单通道增强输出使用。
- `--aec_flat_dir`、`--chunk_seconds`、`--cpu-long-audio-seconds`、`--aec_stream_align` / `--aec_no_stream_align` 等入口选项见 `python inference.py --help`。
- 每个输入文件都会经过 `AECDeploymentFrontend`（enabled 时），语义见下文「工业仿真 (`data.mode: dynamic_aec`)」。

---

## 🧠 模型清单

| 模型 | 说明 | 参考来源 |
|------|------|----------|
| **DeepVQE** | Microsoft AEC Challenge 基线；声明 `aec_input_capabilities = {"mic_spectra", "farend_spectra"}`，只消费所需 dict 谱输入 | [deepvqe (非官方)](https://github.com/Xiaobin-Rong/deepvqe) |
| **CAGCRN** | 轻量联合 AEC+NS（谱进谱出） | [论文 PDF](https://www.isca-archive.org/interspeech_2025/wang25d_interspeech.pdf) |
| **UltraDPCAEC** | Ultra Dual-Path Compression AEC（mic/ref/error 三输入，TrainMel 时间/频率压缩 + PostNet；流式已核验，未压缩约 110K 参数） | [tencent-ailab/UltraDualPathCompression](https://github.com/tencent-ailab/UltraDualPathCompression) |
| **DBGNet** | AEC + 可选帧级 State Detector (FST/DT/NST)；五份配置：16 kHz hybrid、32 kHz hybrid、32 kHz MBRCL、32 kHz MR-SI-L1-SS（small 与 base 两种容量，前两份用 `GTCRNCompressedSpectralLoss` + `GTCRNLogRatioLoss`；MBRCL 用 SI-SNR + MR-STFT + MBRCL；MR-SI-L1-SS 用 SI-L1 + MR-SI-Rel + MR-Suppress 并开启 SS undesired-energy），均 `use_state_detector: false`、不含 StateCE（16 kHz ERB 41/40；32 kHz ERB 81/80 quality-first engineering baseline）；历史 `hybrid` YAML 文件名保持不变；关闭 state head 后按论文规则复测：16 kHz small `0.393 M` 参数 / `0.604 G` MACs，32 kHz small `0.576 M` / `1.348 G`，32 kHz base（论文通道）`1.556 M` / `4.840 G`（THOP `partial/unverified`，B=1、1 秒 mic+farend，图外 STFT/iSTFT 不计；FLOPs/RTF 未测量） | [论文 DOI](https://doi.org/10.1109/TASLP.2024.3519876)、[作者 Demo/项目主页](https://aec-project.github.io)、[vkeep/DBGNet（非官方实现）](https://github.com/vkeep/DBGNet/tree/main)、[AEC 配置](../../configs/aec/microsoft_challenge/README.md) |
| **LocalVQE** | streaming-friendly 的因果 DeepVQE 基础（S4D bottleneck + CausalGroupNorm）；已有 cache wrapper 和 generic ONNX debug export，尚无模型专属 `verify_3way` 发布门禁（`configs/aec/microsoft_challenge/localvqe/localvqe_16k_20ms_delay80ms_loss_hybrid_fwsdr.yaml`） | [LocalVQE](https://github.com/localai-org/LocalVQE) |

DeepVQE 将 `aec_input_capabilities` 声明为包含 `mic_spectra` 与 `farend_spectra` 的 `frozenset`；未声明 capabilities 的旧模型继续使用全输入 legacy 行为。其 `forward` 接收谱形状为 `[B, F, T, 2]` 的 dict，其中 `F = n_fft // 2 + 1`，输出 `nearend_estimate_spectra`。它不消费 target 或辅助谱；`use_state_detector` / `state_nclass` 是 DBGNet 专属字段，显式传入即报错（包括 `false` / `0`），不做静默容忍。构造时未知 kwargs 会 fail-fast，`variant`、`n_fft`、`hop_length`、`align_delay` 与 `compression` 均执行范围校验。

---

## 🔄 与 SE 已对齐的部分

AEC 与 SE 共用同一套框架能力，AEC 侧不重复实现：

- ✅ Python >= 3.10 版本检查
- ✅ 顶层 `stft` 强制校验
- ✅ `GroupedBatchSampler`（按 SR + 长度分桶）
- ✅ 模型注册（`DeepVQE`、`CAGCRN`、`UltraDPCAEC`、`DBGNet`、`LocalVQE`）；`LocalVQE` 同时存在于 `tasks.aec.models` lazy map 与模块内 `@MODEL_REGISTRY.register("LocalVQE")`
- ✅ 统一损失（`CompositeLoss`）
- ✅ 离线 AEC 文件推理与模型特定的流式/ONNX 能力（具体边界见各模型文档）

---

## 🧭 四个生产 family 的对齐机制

Microsoft Challenge 目录当前覆盖 CAGCRN、DBGNet、DeepVQE、LocalVQE 四个 family；它们都接收 `mic_spectra` / `farend_spectra`，但处理回声延迟的机制并不等价：

| Family | 模型内机制 | 当前生产配置的时间范围 |
|--------|------------|------------------------|
| CAGCRN | 当前生产 recipe 以 `align_delay=6` 的 CATA 候选窗口在 delay/window 轴做 softmax，对 far-end 特征软对齐 | 16 ms 帧移下包含当前帧与前 5 帧，最大回看跨度 80 ms |
| DeepVQE | `AlignBlock` 在含当前帧的 `align_delay` 个候选上估计 soft delay | 会议配置 `align_delay=10`、10 ms 帧移，最大回看跨度 90 ms |
| LocalVQE | DeepVQE 风格 `AlignBlock` 后接 S4D bottleneck；`align_delay` 同样包含当前帧 | `align_delay=64`、16 ms 帧移，最大回看跨度 1008 ms |
| DBGNet | far-end encoder 中的 causal `FTG`（GRU）做时序跟踪；没有显式 delay 候选窗口或 lag search | 依靠 0–80 ms 纯 `Beta(2,4)` delay 与 jitter 增强学习鲁棒性；当前 `AECDeploymentFrontend` 模型输入前端关闭 |

`align_delay` 的含义是“含当前帧在内一共看多少个候选帧”，所以最大回看跨度约等于 `(align_delay − 1) × 帧移`：越大覆盖的回声延迟范围越广，计算量也越高。

这些机制不能互相替代为同一个“已对齐”标签。外部 Streaming GCC-PHAT（用互相关估计 mic 与 farend 之间的时延）与 LinearAEC（线性自适应回声消除器）属于仅 AEC 使用的 `AECDeploymentFrontend` 模型输入前端，在 Trainer/Inference 的 STFT 前运行；是否启用由 `aec_frontend` 单独决定。

---

## 🤖 CAGCRN 与官方对齐

- 参考实现：`AECNS-CAGCRN-main`
- ERB(65, 64)、CATA、TFGRU×3、Decoder、Mask、TFAG
- 输入 `(B, F=257, T, 2)`，n_fft=512

---

## 🆕 DBGNet

- **模型文件**: `tasks/aec/models/discriminative/dbgnet.py`
- **实现性质**: 本仓库实现及 [vkeep/DBGNet（unofficial implementation）](https://github.com/vkeep/DBGNet/tree/main) 均为 unofficial implementation；作者公开资源是 [paper (DOI)](https://doi.org/10.1109/TASLP.2024.3519876) 与 [Demo/项目主页](https://aec-project.github.io)，不要将其表述为作者官方代码
- **定位**: AEC + 可选帧级 State Detector，支持 3-class (FST/DT/NST) 和 8-class scenario 状态检测。掩码域由 `model.args.mask_domain` 选择：`mag`（默认，DBG-Mag：实值 sigmoid 掩码乘复谱、保留麦克风相位）或 `com`（DBG-Com：2 通道 RI 输入 + tanh 复数比值掩码、复数乘法；论文未给该变体的掩码参数化与激活函数，属项目选择）。两种掩码域的 cache 名字/顺序/ABI 相同，仅首层 `mic_in`/`far_in` 通道数不同（1/2），切换必须 cold reset
- **配置**: 共六份且都在 `configs/aec/microsoft_challenge/dbgnet/`：五份生产配置 `dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml`、`dbgnet_small_32k_10ms_delay80ms_loss_hybrid.yaml`、`dbgnet_small_32k_10ms_delay80ms_loss_mbrcl.yaml`、`dbgnet_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml`、`dbgnet_base_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml`，以及论文对齐的非生产配置 `dbgnet_paper_16k_16ms_delay1000ms_loss_recon.yaml`（16 kHz、`512/256/512`、ERB `65/64`、论文通道、F-GRU 192（论文规则默认推导）、论文 `L_recon`、`mask_domain: mag`；对齐范围与未对齐项见该文件头）。各配置的 loss 配方、容量与 MACs 见上文「模型清单」，完整对照见 [`configs/aec/microsoft_challenge/README.md`](../../configs/aec/microsoft_challenge/README.md)
- **ONNX**: 支持流式 debug 导出（`python onnx/export.py -C <config> --checkpoint <ckpt> --output runs/onnx_debug/dbgnet.onnx --smoke`，该产物不是正式发布件）；正式发布必须用专属 `verify_3way_dbgnet.py --save-onnx` 并通过 widened gate，命令与门禁见「ONNX 部署」
- **State Detector**: 模型可选内嵌 `AECStateDetectorHead` 并配合 `StateCrossEntropyLoss` 训练；当前 hybrid baseline 未启用，详见「帧级 State Detector」
- **输出瓶颈诊断**: `tools/model_tools/dbgnet_output_bottleneck.py` 是诊断实验 harness，不是训练或发布入口。它对照 oracle real-mask、oracle CRM 与 reference sensitivity 条件；必须显式传入 checkpoint 和 resolved config，并输出 JSON report：

  ```bash
  python tools/model_tools/dbgnet_output_bottleneck.py \
      --checkpoint /path/to/best_model.tar \
      --config /path/to/resolved_config.yaml \
      --data-dir /path/to/canonical_aec_testset \
      --num-samples 4 \
      --out report.json
  ```

---

<a id="aec-dynamic-simulation"></a>
## 🔬 工业仿真 (`data.mode: dynamic_aec`)

本节只保留 AEC 特有的配置和数据边界。统一 Layer A–F、post-mix 顺序、公开 audit facts 以及 `AECDeploymentFrontend` 的所有权以 [`docs/guides/simulation/Simulation_Flow_Logic.md`](../../docs/guides/simulation/Simulation_Flow_Logic.md) 为准；AEC 的仿真物理约束（SER/ERL 电平耦合、回声延迟、回声截幅与播放设备 EQ 的设计）见 [`docs/guides/aec/AEC_Data_Simulation.md`](../../docs/guides/aec/AEC_Data_Simulation.md)；十份生产配置的 Layer E/F 期望值、C-aec 截幅三路径参数、回声延迟与抖动设置见 [`configs/aec/microsoft_challenge/README.md`](../../configs/aec/microsoft_challenge/README.md) 的「关键约定」。

- Dynamic Dataset 执行 Layer A–F、C-aec 与 final deterministic safety，`AECDeploymentFrontend` 不属于 Dataset 仿真；分层含义与固定 post-mix 顺序以上面链接的仿真合同为准。AEC 的一个建模特点是 Layer E/F 只修改组装后的 `mic`（含 C-aec 回声），nearend target 经过 Layer D 后绕过 E/F。
- `task: joint`（默认，AEC+NS，同时去回声和降噪）或 `task: aec`（仅去回声）。

训练、validation 和 `inference.py` 都解析 canonical `AECDeploymentFrontend`，顺序为**可选 Streaming GCC-PHAT → 可选 LinearAEC（`speex` / `pfdkf` / `pfdkf_torch`）**，但只在 `aec_frontend.delay_alignment.enabled` 为 `true` 或配置了 `aec_frontend.linear_aec` 时执行。两者均未启用时，mic/farend 严格直通且不报 runtime error；`inference.py` 仍对每个文件调用该前端。`aec_frontend.linear_aec.bulk_delay` 当前仅接受 `0`；需要延迟补偿时配置前置 Streaming GCC-PHAT。

validation 指标分两条路径：共享 objective metrics 只消费 `(nearend, nearend_estimate)`；AEC-specific ERLE 单独消费 `(mic, nearend, nearend_estimate, farend[, echo])`，不得把 ERLE 混入共享二元组指标调用。

当前四个模型家族对应十份生产配置（DBGNet 有五份）均关闭 `aec_frontend.delay_alignment`，且均未配置 LinearAEC。CAGCRN、DeepVQE、LocalVQE 分别使用 CATA 或 AlignBlock；DBGNet 使用 causal FTG 时序跟踪并配合 delay/jitter 增强，不提供显式 lag search，也不叠加 Streaming GCC-PHAT。具体窗口见上方 family 对照表。

### 动态噪声清单合同

- `noise_scp` 只接受严格四列 SCP（`uid fs length path`）或逐行 JSON object JSONL；JSON array 与遗留三列 SCP 直接拒绝，分类 metadata 只能通过 JSONL 传递。发布/审计合同当前为 strict JSONL schema v6（admission policy v4、AudioTag evidence v5、noise source parser v10、generator `6.1`、promotion marker v4）；Noise Filter Pipeline 的准入规则（C2/C3）、`roles` 与 Wind/Music 分流、发布摘要与审计工具见 [`configs/aec/microsoft_challenge/README.md`](../../configs/aec/microsoft_challenge/README.md) 的「动态噪声清单与角色分流」，噪声准入与发布流程见 [`tools/noise_filter_pipeline/README.md`](../../tools/noise_filter_pipeline/README.md)。
- **echo source/image 合同**：`farend` 是 mono reference；已有 echo 可以是 mono `[T]`，也可以是 microphone-side image `[C,T]`。已有 `[C,T]` echo 直接加入 spatial mic observation，不重复卷积；mono echo 只有在需要空间差异时才经独立 loudspeaker RIR 映射。当前不要求从 clean farend 重新实现完整 playback echo synthesizer。展开说明见 [`docs/guides/aec/AEC_Data_Simulation.md`](../../docs/guides/aec/AEC_Data_Simulation.md)。

---

<a id="aec-offline"></a>
## 📦 离线三元组 (`data.mode: offline_aec`)

`offline_aec` 直接读磁盘上已经配好的 `mic` / `farend` / `nearend` 三元组（可选预生成的 `error`），并按 uid 对齐：Dataset 只做读取、重采样和同步裁补，不跑仿真、不做 Streaming GCC 或 LinearAEC，也不保证已经做过部署侧对齐。因为没有在线仿真随机性，它适合真实采集数据或外部管线预渲染的数据。

- `AECOfflineDataset` 输出 7 元组 `(nearend, mic, farend, fs, length, error, error_mask)`，可直接交给 `aec_collate_fn`；配置 `error_scp` 时要求四轨 uid 完全一致，并将 `error_mask` 置为真。显式 `spatial_enabled` 时允许磁盘 `mic` 为 `[C,T]`，nearend/farend/error 仍为 mono。
- collate 之后仍由 `AECTrainer` 按顶层 `aec_frontend` 在 STFT 前执行与 dynamic 相同的可选前端；runtime LinearAEC 与 `error_scp` 同时存在时 fail-fast。
- 数据准备脚本为 `tools/prepare_data_tools/scp/prepare_aec_offline.py`；`error_scp` 的数据侧 capability 语义、`offline_aec_template.yaml` 模板（默认路径为 `null`，不能原样直接训练）以及“只取消注释 `error_scp` 不会让模型变成三输入”的边界见 [`configs/aec/microsoft_challenge/README.md`](../../configs/aec/microsoft_challenge/README.md) 的「离线 residual capability 与模板边界」。

---

<a id="aec-state-detector"></a>
## 🏷️ 帧级 State Detector

`AECStateDetectorHead` 支持 3-class（FST 远端单讲 / DT 双讲 / NST 近端单讲）或 8-class scenario 帧级标签，当前仅由 DBGNet 显式集成；DeepVQE 明确拒绝启用该 head。当前 DBGNet 16/32 kHz hybrid baseline 均关闭该 head，未启用 StateCE。`StateCrossEntropyLoss` 消费 `(B,T,C)` logits 和 `(B,T)` 类索引，并忽略 `-1` padding。

---

## 📦 AEC 样本格式

Dataset 每个样本用标量 `length`，`aec_collate_fn`（把不等长样本补齐成 batch 张量的函数）输出 `AECBatch` 时用 `[B]` 的 `lengths`。Batch 按名包含以下字段，`state` 可省略：

```
(nearend, mic, farend, fs, lengths, error, error_mask,
 is_pure_noise, is_pure_echo, is_pure_echo_noise, has_echo[, state])
```

字段含义：`is_pure_*` 标记纯噪声/纯回声/纯回声噪样本，`has_echo` 表示该样本含磁盘 echo 成分（Trainer 据此精确挑选 ERLE 样本），`state` 是可选的帧级状态标签。

`aec_collate_fn` 精确接受 Dataset 的 7/11/12 元组样本：7 元组无分支标志，11 元组是当前基础格式，12 元组追加帧级 `state`；同一 batch 一旦含 `state`，所有样本都必须是 12 元组。`fs` / `length` 必须是精确正整数标量，旗标只接受标量 bool 或有限数值 `0/1`。collate 返回已执行 `validate()` 的 `AECBatch`，提供 `pin_memory()`、`.to()` 和 `batch_size`，覆盖 error、flags 和可选 state。5/6/8/9/10/13 元组及其他未知长度均显式拒绝，不会静默截断未知扩展。空间 observation 下 batch 的 `mic` 为 `[B,C,T]`，其余波形轨道仍为 `[B,T]`；Trainer 只接受 `AECBatch`，tuple/list 批显式拒绝。

已物化 `offline_aec` 使用 split-level `spatial_enabled`、`array_profile` / `custom_geometry`、`reference_channel`、`channel_order` 声明 microphone 数据合同。reader 校验所有 mic 文件的 C，实际读取及 placeholder 都保持同一 `[C,T]` 布局；不重排文件通道。模型 reference 参数从 `resolved_microphone_array` 在构造前注入，显式冲突拒绝。详见 [空间 reader 与噪声合同](../../docs/guides/simulation/README.md)。

---

## 🚢 ONNX 部署

ONNX 是跨框架的模型交换格式，用来把训练好的模型交给推理引擎（如 ONNX Runtime）部署。本项目区分两类产物：generic `onnx/export.py --output` 只生成 debug artifact，用于快速确认模型能否导出；正式发布必须由模型专属 verifier 完成三路检查（同一段音频分别走离线 PyTorch、流式 PyTorch、流式 ONNX Runtime，比较三者数值是否对齐）后写入 `onnx/onnx_model/`：

```bash
# DeepVQE 正式发布；当前会议版 YAML 必须与实际训练所得 checkpoint 匹配
python onnx/verify/verify_3way_deepvqe.py \
    -C configs/aec/microsoft_challenge/deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml \
    --checkpoint "$HOME/use_train_checkpoints/aec/deepvqe/meeting_32k/best_model.tar" \
    --save-onnx onnx/onnx_model/deepvqe_small_streaming.onnx

# CAGCRN 正式发布
python onnx/verify/verify_3way_cagcrn.py \
    -C configs/aec/microsoft_challenge/cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml \
    --checkpoint "$HOME/use_train_checkpoints/aec/cagcrn/best_model.tar" \
    --save-onnx onnx/onnx_model/cagcrn_stream.onnx

# DBGNet 正式发布；widened gate 至少 1000 帧、2 次中途 reset
python onnx/verify/verify_3way_dbgnet.py \
    -C configs/aec/microsoft_challenge/dbgnet/dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml \
    --checkpoint "$HOME/use_train_checkpoints/aec/dbgnet/best_model.tar" \
    --frames 1200 --resets 2 \
    --save-onnx onnx/onnx_model/dbgnet_stream.onnx

# UltraDPCAEC 正式发布；压缩参数必须与训练 checkpoint 一致
python onnx/verify/verify_3way_ultra_dpc.py \
    --checkpoint "$HOME/use_train_checkpoints/aec/ultra_dpc/best_model.tar" \
    --freq-ratio 2 --time-skip 1 \
    --save-onnx onnx/onnx_model/ultra_dpc_stream_fr2.onnx
```

发布统一要求 checkpoint、float32、不可放宽的预注册阈值和 A/B/C 三路全部 `PASS`；空比较、非有限指标或 `SKIP` 都会阻断。DeepVQE、CAGCRN、DBGNet 还要求 config。DBGNet 发布 gate 默认使用 `--frames 1200 --resets 2`，且不接受低于 1000/2 的缩窄验证。UltraDPCAEC 的 `time_skip>1` 会跳过 A/C，只能手工验证，不能正式发布；完整说明见 `onnx/README.md`。

---

## ⚡ 构建 C++ 延迟估计扩展

`tasks/aec/inference/csrc/` 提供 C++ 加速的延迟估计器，属于可选优化：不编译也能跑，只是延迟估计会走纯 Python 实现。实际加速比取决于算法、帧长、编译器和目标 CPU，应以目标平台 benchmark 为准。从仓库根目录构建：

```bash
python tasks/aec/inference/csrc/setup.py build_ext --inplace
```

> 构建产物固定写入 `tasks/aec/inference/_native/`；运行时只从该目录加载与当前平台/Python ABI 匹配且不早于 C++ 源码的扩展。未通过定位或新旧校验时自动回退到纯 Python 实现（功能等价但更慢）。

---

## 📚 BibTeX 引用

模型来源见上方「模型清单」；本节集中维护论文作者、会议与引用信息。感谢 [Microsoft AEC Challenge](https://github.com/microsoft/AEC-Challenge) 提供评测数据集与基线参考。

```bibtex
@inproceedings{indenbom2023deepvqe,
  title={DeepVQE: Real Time Deep Voice Quality Enhancement for Joint Acoustic Echo Cancellation, Noise Suppression and Dereverberation},
  author={Indenbom, Evgenii and Ristea, Nicolae-C{\u{a}}t{\u{a}}lin and Saabas, Ando and P{\"a}rnamaa, Tanel and Gu{\v{z}}vin, Jegor and Cutler, Ross},
  booktitle={Proc. INTERSPEECH},
  pages={3819--3823},
  year={2023},
  doi={10.21437/Interspeech.2023-2176},
  url={https://arxiv.org/abs/2306.03177}
}

@inproceedings{wang2025cagcrn,
  title={CAGCRN: Real-Time Speech Enhancement with a Lightweight Model for Joint Acoustic Echo Cancellation and Noise Suppression},
  author={Wang, Yuyang and Liu, Yonghui and Liu, Jianbing and Niu, Kai and He, Zhiqiang},
  booktitle={Proc. INTERSPEECH},
  pages={768--771},
  year={2025},
  doi={10.21437/Interspeech.2025-608}
}

@article{han2025dbgnet,
  title={Dual-Branch Guidance Encoder for Robust Acoustic Echo Suppression},
  author={Han, Hyewon and Peng, Xiulian and Kim, Doyeon and Lu, Yan and Kang, Hong-Goo},
  journal={IEEE/ACM Trans. Audio, Speech, Language Process.},
  volume={33},
  pages={627--639},
  year={2025},
  doi={10.1109/TASLP.2024.3519876}
}

@inproceedings{chen2023ultradpc,
  title={Ultra Dual-Path Compression For Joint Echo Cancellation And Noise Suppression},
  author={Chen, Hangting and Yu, Jianwei and Luo, Yi and Gu, Rongzhi and Li, Weihua and Lu, Zhuocheng and Weng, Chao},
  booktitle={Proc. INTERSPEECH},
  pages={2523--2527},
  year={2023},
  doi={10.21437/Interspeech.2023-2302}
}
```

---

<p align="center">
  <sub>最后更新: 2026-10-08</sub>
</p>
