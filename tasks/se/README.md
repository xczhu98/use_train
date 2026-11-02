# 🎧 SE — Speech Enhancement

<p align="center">
  <strong>语音增强训练、模型与推理模块</strong><br>
  与 <code>tasks/aec/</code> / <code>tasks/tse/</code> 共用 core 基础设施；覆盖判别式、GAN 生成式、Flow 生成式与预训练生成式（PASE 系列）四种训练范式。
</p>

---

## 📑 目录

- [这是什么，怎么跑起来](#-这是什么怎么跑起来)
- [任务定位与边界](#-任务定位与边界)
- [目录结构](#-目录结构)
- [数据准备](#-数据准备)
- [训练](#-训练)
- [推理](#-推理)
- [评估](#-评估)
- [模型清单](#-模型清单)
- [BibTeX 引用](#-bibtex-引用)

---

## 🧭 这是什么，怎么跑起来

**语音增强（SE, Speech Enhancement）**要解决的问题：一段录音里混进了背景噪声、混响等干扰，目标是把里面的人声恢复干净。本模块负责 SE 的数据构建、训练、推理与评估，是框架五个任务之一（其余四个是 AEC 回声消除、TSE 目标说话人提取、SS 语音分离、SSL 空间声学定位）。

后文反复出现的词先在这里说清：

| 术语 | 含义 |
|---|---|
| `data.mode` | 数据模式开关；`dynamic` 在线仿真，`offline` 读 clean/noisy 配对。两种模式都可显式启用多通道 noisy；offline spatial 要在 split 声明 geometry、`reference_channel` 和 `channel_order` |
| SCP | 纯文本音频清单，每行是 `uid 采样率 时长 路径`；本项目的数据集都靠清单索引音频 |
| `speech_scp` / `noise_scp` / `rir_scp` | 干净语音 / 噪声 / 房间脉冲响应（RIR，卷积到语音上得到房间混响）三类清单 |
| STFT / iSTFT | 短时傅里叶变换及其逆变换：把波形转成时频谱交给模型处理，再变回波形 |
| Layer A–F | 共享波形仿真的六个分层（A 语音预处理、B 场景分支、C 声学混合、D post-mix 幅度条件、E 信道退化、F 设备/后处理退化）；`dynamic` 模式下由 Dataset 依次执行 |
| checkpoint | 训练保存的权重文件（`best_model.tar` / `latest_model.tar`） |
| DDP | PyTorch 多卡数据并行训练，用 `torchrun --nproc_per_node=N` 启动 |

最短路径（每一步的完整参数见对应小节）：

1. **准备数据**：用 `tools/prepare_data_tools/scp/prepare_scp.py` 扫描音频目录并生成 SCP 清单。
2. **训练**：`CUDA_VISIBLE_DEVICES=0 python train.py -C configs/se/dns_challenge3/gtcrn_causal_offline.yaml`，日志与 checkpoint 写到配置里的 `exp_path`。
3. **推理**：`python inference.py -C <config> --checkpoint <ckpt> --input_dir <noisy_dir> --output_dir <out>`，每个输入文件产出一个增强后的 wav。
4. **评估**：`python evaluate/eval_pipeline_se.py --exp_dir <exp_dir> --datasets voicebank --step all`，需要同时给出对应数据集的清单参数。

---

## 🎯 任务定位与边界

- 入口：训练统一从 `train.py` 进入，按配置里的 `data.mode` 路由；SE 支持 `dynamic` 在线仿真和 `offline` 预混合配对。offline spatial 的 clean 保持 mono，noisy 保持 `[C,T]`；split 必须声明 `spatial_enabled`、`array_profile`/`custom_geometry`、`reference_channel` 与 `channel_order`。文件推理走根 `inference.py`，模型 I/O 适配在 `tasks/se/inference/adapter.py`，manifest、输出锁与产物校验由公共 runner 负责。
- 批合同：Dataset 样本为 4 元组，或追加 `is_pure_noise` / `is_pure_echo` / `is_pure_echo_noise` 的 7 元组；`core.data.collate.collate_fn` 补齐后返回已校验的 `SEBatch`。该类型定义于 `core.data.speech_enhancement_batch`，任务侧 `tasks.se.data.batch` 再导出；提供 `validate()`、`pin_memory()`、`.to()` 和 `batch_size`，支持 DataLoader 锁页及异步设备搬运。5 元组旧样本和 Trainer 的 tuple/list 批均显式拒绝。
- 模型 I/O 基线：判别式模型继承 `tasks/se/models/base_model.py::BaseSEModel`，签名为 `forward(noisy_spec, lengths=None, *, fs=None)`，即输入含噪频谱、输出增强频谱；dual-input、loss-returning 与 GAN 形态是显式例外，由对应 Trainer 驱动。`MossFormer2_SE_48K` 内部自带 iSTFT→Kaldi fbank 前端；`CoFi-Lite` 是 waveform-native 判别式例外，按上游方案在模型内部完成 sqrt-Hann STFT/iSTFT（`frontend_owner="model"`，YAML 设 `meta.allow_no_stft: true`）。
- STFT/iSTFT 由 Trainer 或顶层模块统一管理，模型内部不重复引入；`MossFormer2_SE_48K` 与 `CoFi-Lite` 的模型自有前端是登记例外。
- 仿真合同（Layer A–F、clean target 边界）以 [`docs/guides/simulation/Simulation_Flow_Logic.md`](../../docs/guides/simulation/Simulation_Flow_Logic.md) 为准；本文件只保留 SE 的运行入口与数据边界。

---

## 多通道仿真与离线数据

`SpatialNet_SE` / `OnlineSpatialNet_SE` 在框架 STFT 后消费 `[B,C,F,T,2]` noisy spectra，并只输出 reference-channel enhanced spectrum。Dynamic spatial 数据仍走 `SpatialSceneRenderer`、typed scene plan 与在线 RIR provider；mono noise 的空间化使用 `spatial_diffuse_coherence_model: habets_spherical`（spherical sinc coherence，生成资产仍标为 approximate）。Canonical CHiME-3 阵列为 `chime3_6ch_rect_100x190mm`。`prob_array_rotation` / `array_rotation_z_rad_range` 控制阵列平面 yaw，mic jitter 默认关闭；实际变换进入 room plan identity。

速度字段有两种不同物理含义：`prob_speed_perturb` / `speed_rate_range` 是 Layer A 的语速和音高扰动；`prob_moving_rir` 与 `moving_source_speed_range_mps` 分别控制是否移动与房间内声源速度。二者不可互换。`moving_source_speed_range_mps` 采样带自适应缩速：位移放不进房间时重试按 0.9^n 缩速并把端点投影回房间内，实际速度可能低于配置下限（长画布配置仍应按位移口径等比缩速，见 configs/se/spatial/online_spatialnet_dynamic_*_lf32s*.yaml）。`target_type: rts` 仅支持 SE spatial path，按实际房间 RT60 对 full RIR 做 reverberation-time shortening 后生成监督目标；`rts_target_t60_sec` 与 `rts_time_after_max_sec` 固定其窗口口径。

`offline` spatial split 保持 clean mono 与 noisy channel-first `[C,T]`。`data.train` / `data.validation` 要显式配置 `spatial_enabled: true`、`array_profile` 或 `custom_geometry`、`reference_channel` 和物理通道顺序 `channel_order`；Dataset 会按每个 noisy WAV header 校验通道数，trainer 才会启用 `collate_fn(allow_spatial_noisy=True)`。offline 输入使用配对四列 `clean_scp` / `noisy_scp`，不会重排通道或把 mono 噪声静默广播成阵列信号。

生成 WSJ0-mix 风格双源空间评估集时，先准备目标与干扰语音各自的严格四列 SCP（两份清单应代表不同来源池），再运行：

```bash
python -m tools.prepare_data_tools.generators.se.generate_mc_sim_testset --config configs/se/spatial/spatialnet_dynamic.yaml --target-scp <target_speech.scp> --interference-scp <interference_speech.scp> --output-dir data/se/mc_sim_testset --num-samples 100 --seed 42 --duration-seconds 4 --sir-range-db -5 5
```

生成器由我方 `prepare_scene_plan` / `SpatialSceneRenderer` 生成 room 与 source images，并发布 `mixture.scp`、可用于 offline Dataset 的同内容 `noisy.scp`、reference clean `clean.scp`、oracle 目标空间图像 `target_image.scp` 与 hash manifest。推理及 oracle MVDR 对比入口（首次生成时也传两份源 SCP）：

```bash
python evaluate/eval_pipeline_se.py --exp_dir <exp_dir> --datasets mc_sim --mc_sim_testset_root data/se/mc_sim_testset --mc_sim_target_scp <target_speech.scp> --mc_sim_interference_scp <interference_speech.scp> --step all
```

首次生成时用两项 `--mc_sim_*_scp` 指定目标与干扰语音 SCP；若复用已冻结的 testset，可只传 `--mc_sim_testset_root`。生成器会校验四列 SCP、UID 交集、逐条 payload hash 和 manifest；speaker-disjoint 由数据准备方保证，因为标准 SCP 不携带 speaker identity。本机当前未安装 WSJ0 corpus；没有授权语料时可用小型 fixture 验证工具链，但不可把 fixture 指标报告为 WSJ0 结果。offline 空间训练样板见 `configs/se/spatial/spatialnet_offline_mc_sim.yaml`，train/validation 应指向独立生成集。

`ComplexIdealRatioMaskLoss` 仅吸收 cIRM MSE objective，不引入 NBSS NBC/NBC2 模型。它通过 direct spectra 与 noisy reference 对现有谱输出构造预测 ratio mask；示例配置为 `spatialnet_offline_mc_sim.yaml`，`reference_channel` 必须与模型 / 数据阵列一致。

---

## 📁 目录结构

```
tasks/se/
├── data/                 # dynamic / offline Dataset、collate 复用 core.data.collate
├── inference/            # adapter.py：根 inference.py 的 SE 模型 I/O 适配
├── models/
│   ├── base_model.py     # BaseSEModel 接口基线
│   ├── discriminative/   # GTCRN、BSRNN、DeepFilterNet、FastEnhancer、SpatialNet_SE 等
│   ├── gan/              # 生成器与判别器（MetricGAN 系列）
│   ├── generative/       # complex-STFT Flow 模型（MeanFlowSE / MeanSE）
│   ├── pase_family/      # PASE / StuPASE / UniPASE / GAP_URGENet
│   └── modules/          # 共享网络模块
└── trainer/              # 各训练范式的 Trainer 实现
```

`trainer/` 下按训练范式分开：普通判别式训练用 `SETrainer`，GAN 与 MetricGAN 用 `GANTrainer` / `MetricGANCompositeTrainer`，Flow 模型用 `MeanFlowSETrainer`，PASE 系列用 `PASEGANTrainer` / `UniPASEAdapterGANTrainer`，需要双输入或多输出的用 `DualInputLossTrainer`。

`mambattention_impl` 与 `rwsa_mamba_impl` 的幅度解码器共用
[`models/modules/learnable_sigmoid.py`](models/modules/learnable_sigmoid.py)；
激活函数仍挂载为 `lsigmoid`，checkpoint 中的 `lsigmoid.slope` 名称与形状保持一致。

配置、数据与实验目录分别位于 `configs/se/`、`data/...` 与 `exp/se/`；配置说明见 [`configs/se/README.md`](../../configs/se/README.md)，按场景组织的使用指南见 [`docs/guides/se/README.md`](../../docs/guides/se/README.md)。

---

## 📦 数据准备

音频目录的 SCP 清单由 `tools/prepare_data_tools/scp/prepare_scp.py` 生成：它扫描音频目录（不转码、不复制音频文件），写出四列 SCP，每行是 `uid 采样率 时长 路径`。噪声清单若要带分类 metadata，则改用 Noise Filter Pipeline 输出的 JSONL（见下文「噪声与 RIR」）。

> 💡 **SE 与 AEC 数据差异：** SE 使用 clean/noisy 对（离线）或 speech+noise+rir 清单（动态）。AEC 额外需要 farend/echo 立体声对，见 [`tasks/aec/README.md`](../aec/README.md)。

### VoiceBank-Demand

VoiceBank-Demand 自带配对好的 clean/noisy 音频。下面用 full-unsplit 模式分别扫描官方 train 与 test 目录，不做随机划分（`--val_ratio 0` 即“不切分、只写一份清单”）：

```bash
VOICEBANK_SRC=/path/to/voicebank_demand
VOICEBANK_OUT=data/se/voicebank_demand

python tools/prepare_data_tools/scp/prepare_scp.py \
    "$VOICEBANK_SRC/clean_trainset_wav" -o "$VOICEBANK_OUT/voicebank_clean_train" \
    --val_ratio 0
python tools/prepare_data_tools/scp/prepare_scp.py \
    "$VOICEBANK_SRC/noisy_trainset_wav" -o "$VOICEBANK_OUT/voicebank_noisy_train" \
    --val_ratio 0
python tools/prepare_data_tools/scp/prepare_scp.py \
    "$VOICEBANK_SRC/clean_testset_wav/clean_testset_wav" -o "$VOICEBANK_OUT/voicebank_clean_val" \
    --val_ratio 0
python tools/prepare_data_tools/scp/prepare_scp.py \
    "$VOICEBANK_SRC/noisy_testset_wav/noisy_testset_wav" -o "$VOICEBANK_OUT/voicebank_noisy_val" \
    --val_ratio 0
```

跑完后 `$VOICEBANK_OUT` 目录下会得到 `voicebank_clean_train.scp`、`voicebank_noisy_train.scp`、`voicebank_clean_val.scp` 与 `voicebank_noisy_val.scp` 四个清单。注意这里的 `val` 实际采用官方 test 集（test-as-validation）：用它调参会造成最终测试泄漏，严格实验应从官方 train 中另划 validation，并把官方 test 仅用于最终报告。数据处理与部署细节见 [`configs/se/voicebank_demand/README.md`](../../configs/se/voicebank_demand/README.md)。

### URGENT 2026 full-unsplit 离线数据准备

```bash
DATA_ROOT=data/shared/urgent26_track1_se_dataset
UNIVERSAL_ROOT="$DATA_ROOT/urgent26_track1_universal_se"
NONBLIND_ROOT="$DATA_ROOT/urgent26_nonblind_with_label"

python tools/prepare_data_tools/scp/prepare_scp.py "$UNIVERSAL_ROOT/simulation_train/clean" \
    -o "$UNIVERSAL_ROOT/urgent26_clean_train" --val_ratio 0
python tools/prepare_data_tools/scp/prepare_scp.py "$UNIVERSAL_ROOT/simulation_train/noisy" \
    -o "$UNIVERSAL_ROOT/urgent26_noisy_train" --val_ratio 0
python tools/prepare_data_tools/scp/prepare_scp.py "$UNIVERSAL_ROOT/simulation_validation/clean" \
    -o "$UNIVERSAL_ROOT/urgent26_clean_val" --val_ratio 0
python tools/prepare_data_tools/scp/prepare_scp.py "$UNIVERSAL_ROOT/simulation_validation/noisy" \
    -o "$UNIVERSAL_ROOT/urgent26_noisy_val" --val_ratio 0
python tools/prepare_data_tools/scp/prepare_scp.py "$NONBLIND_ROOT/clean" \
    -o "$NONBLIND_ROOT/urgent26_clean_nonblind" --val_ratio 0
python tools/prepare_data_tools/scp/prepare_scp.py "$NONBLIND_ROOT/noisy" \
    -o "$NONBLIND_ROOT/urgent26_noisy_nonblind" --val_ratio 0
```

premixed 清单（clean/noisy 各一份）用于 `offline` 模式：noisy 是模型输入，clean 是监督目标。完整数据包含 **436,240** 对训练样本、**2,200** 对 validation 样本和 **1,023** 对 nonblind 样本。nonblind 仅用于评估，不得用于训练、调参或 Trainer validation；`urgent26_noisy_train.scp` 等 premixed noisy 清单是离线配对输入，不能当作 dynamic mixing 的 `noise_scp`（那份清单要的是纯噪声，不含干净语音）。目录、配置消费路径与动态混合说明见 [`configs/se/urgent2026/README.md`](../../configs/se/urgent2026/README.md)。

### 噪声与 RIR

`dynamic` 模式还需要噪声清单 `noise_scp` 和房间脉冲响应清单 `rir_scp`：训练时 Dataset 随机抽一条语音、一条 RIR（卷积出房间混响）和若干噪声实时合成含噪样本。两者都遵循根 README 描述的 source manifest 合同（严格四列 SCP 或逐行 JSON object JSONL），噪声的准入与发布由 [`tools/noise_filter_pipeline/README.md`](../../tools/noise_filter_pipeline/README.md) 负责。`prepare_scp.py --rir` 的 pickle 采样率契约见根 README。

---

## 🚀 训练

训练入口是仓库根目录的 `train.py`，用 `-C` 指定配置文件，配置里的 `data.mode` 决定走 dynamic 还是 offline：

```bash
# 单卡
CUDA_VISIBLE_DEVICES=0 python train.py \
    -C configs/se/dns_challenge3/gtcrn_causal_offline.yaml

# 多卡 DDP（4 卡数据并行）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
    -C configs/se/urgent2026/dpdfnet2_causal_urgent26_dynamic_hybrid.yaml

# 恢复训练（接着上次的 optimizer / scheduler 状态继续）
CUDA_VISIBLE_DEVICES=0 python train.py -C /path/to/config.yaml --resume

# 微调（只用已有权重初始化模型，不恢复训练状态）
CUDA_VISIBLE_DEVICES=0 python train.py -C /path/to/config.yaml --checkpoint ckpt/pretrained.pth

# 启用 WandB 记录
CUDA_VISIBLE_DEVICES=0 python train.py -C /path/to/config.yaml --wandb
```

运行后日志和 checkpoint 会写到配置的 `exp_path`（约定为 `exp/se/<配置文件名>`）。调试与 `resume` / `max_steps` 语义见根 README 的「训练」章节。SE 侧配置入口：

- 判别式与 GAN：[`configs/se/dns_challenge3/`](../../configs/se/dns_challenge3/)、[`configs/se/voicebank_demand/`](../../configs/se/voicebank_demand/)
- URGENT2026 与 PASE 系列：[`configs/se/urgent2026/`](../../configs/se/urgent2026/)，其中 [PASE](../../configs/se/urgent2026/pase/README.md) / [UniPASE](../../configs/se/urgent2026/pase/unipase/README.md) / [GAP-URGENet](../../configs/se/urgent2026/pase/gap_urgenet/README.md) 有独立说明
- 空间 SE：[`configs/se/spatial/`](../../configs/se/spatial/)

---

## 🔍 推理

根 `inference.py` 逐文件跑增强：输入音频 → 模型 → 输出 wav。它通过 `tasks/se/inference/adapter.py` 复用 SE 模型 I/O；清单、输出锁、`skip-existing` 与原子输出由公共 runner 负责。产物目录下会写出增强后的 wav 和记录本次运行的 `run_manifest.json`。

```bash
# 目录模式：整个目录批量推理
python inference.py -C /path/to/config.yaml --checkpoint /path/to/best_model.tar \
    --input_dir data/test/noisy --output_dir results/enhanced

# SCP 模式：按清单推理
python inference.py -C /path/to/config.yaml --checkpoint /path/to/best_model.tar \
    --input_scp data/test.scp --scp_root_dir /path/to/audio \
    --output_dir results/enhanced

# DNS Challenge V3 盲测（SCP 模式）
python inference.py -C configs/se/dns_challenge3/gtcrn_causal_offline.yaml \
    --checkpoint /path/to/best_model.tar \
    --input_scp data/se/DNS_Challenge_test/v3_blindset.scp \
    --output_dir results/dns3_blind
```

`--sr` 会固定/覆盖 `data.sample_rate` 并自动重采样输入；`--skip-existing` 只复用能通过 `run_manifest.json` 校验的旧产物，缺失或不匹配会重新推理；`--device`、`--allow-partial`、`--no-progress`、`--output_relpath_scp` 等通用选项见根 README。

---

## 📊 评估

评估流程是「先推理、再算指标」：`--step all` 表示两步都做，也可以只跑 `inference` 或 `evaluate`。`--audio_quality` 打开的是不需要参考音频的质量指标（DNSMOS / NISQA / UTMOS / SCOREQ）；如果同时提供 clean 参考，还会计算需要参考的侵入式指标 PESQ / STOI / ESTOI / SI-SDR。

```bash
# SE：VoiceBank + DNS Challenge V3
python evaluate/eval_pipeline_se.py --exp_dir exp/se/example_experiment \
    --datasets voicebank,dns_challenge --step all --audio_quality \
    --voicebank_clean_scp data/se/voicebank_demand/voicebank_clean_val.scp \
    --voicebank_noisy_scp data/se/voicebank_demand/voicebank_noisy_val.scp \
    --dns_challenge_noisy_dir data/se/DNS_Challenge_test/V2_V3_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k
```

`--exp_dir` 指向实验目录，脚本会自己找配置与 checkpoint；`--voicebank_clean_scp` 是参考音频，`--voicebank_noisy_scp` 是待评估的含噪输入。

URGENT nonblind 的 `noisy/0/*.flac` 嵌套布局目前不符合统一 pipeline 的 URGENT directory contract。请先通过 SCP 推理，再独立评估：

```bash
python inference.py -C /path/to/config.yaml --checkpoint /path/to/best_model.tar \
    --input_scp data/shared/urgent26_track1_se_dataset/urgent26_nonblind_with_label/urgent26_noisy_nonblind.scp \
    --output_dir results/urgent_nonblind
python evaluate/evaluate_urgent.py \
    --ref_scp data/shared/urgent26_track1_se_dataset/urgent26_nonblind_with_label/urgent26_clean_nonblind.scp \
    --inf_scp results/urgent_nonblind/inf.scp \
    --output_dir results/urgent_nonblind_eval --audio_quality
```

流水线状态语义（`done` / `partial` / `error`、指标覆盖率与 `--allow-partial-inference`）见根 README 与 [`evaluate/README.md`](../../evaluate/README.md)。

---

## 🧠 模型清单

SE 模型按**训练范式**划分成四类。它们都是 SE 模型的子类（只是训练方式不同），不是与 SE 平行的独立任务；代码分别在 `tasks/se/models/discriminative/`、`tasks/se/models/gan/`、`tasks/se/models/generative/` 与 `tasks/se/models/pase_family/`。

- **判别式**：网络直接预测干净谱或波形，训练目标是最小化失真，是本项目的主力形态。
- **GAN 生成式**：生成器与判别器对抗训练，用于提升听感质量。
- **Flow 生成式**：用 Mean Flow 一类生成式采样从噪声还原干净语音（本仓库为 complex-STFT 实现）。
- **预训练生成式（PASE 系列）**：以大规模预训练模型（如 WavLM）的语音先验做生成式增强，主打低幻觉。

下表是代表性模型列表，不是完整的 Registry 清单；其他已注册模型和可运行 recipe 见 `configs/se/`。

### 判别式模型

| 模型 | 说明 | 参考来源 |
|------|------|----------|
| **GTCRN** | 轻量因果 SE | [gtcrn-main](https://github.com/Xiaobin-Rong/gtcrn) |
| **CoFi-Lite** | waveform-native 超轻量因果 SE；模型内部 sqrt-Hann STFT/iSTFT，双路径包络/细节建模 | [CoFi-Lite](https://github.com/Acceleration123/CoFi-Lite) |
| **BSRNN** | 频带切分 RNN | [URGENT2026 Baseline](https://github.com/urgent-challenge/urgent2026_challenge_track1) |
| **DeepFilterNet** | 深度滤波增强 (DF2/DF3) | [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) |
| **FastEnhancer** | 轻量非因果/流式增强 | [fastenhancer](https://github.com/aask1357/fastenhancer) |
| **Conformer** | Conformer 编码器 | [ESPnet](https://github.com/espnet/espnet) |
| **TFGridNet** | 时频网格网络 | [ESPnet](https://github.com/espnet/espnet) |
| **MossFormer2** / **MossFormer2_SE_48K** | MossFormer2 与 48 kHz ClearerVoice 变体 | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) |
| **FRCRN_SE_16K** | ClearerVoice 16 kHz 变体（上游名 FRCRN） | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) |
| **BSRoFormer** | 频带分片 RoFormer | [BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) |
| **SpatialNet_SE** | 原生多通道谱输入、单通道 reference 谱输出；实现参考公开 NBSS 设计，不宣称上游 checkpoint parity | [项目 Registry](models/__init__.py) |
| **OnlineSpatialNet_SE** | 因果多通道 backend，窄带分支可配置 `attention='mhsa(scope)' \| 'ret(factor,share_qk)' \| 'mamba(d_state,d_conv)'`，使用 tensor-only streaming cache（ABI v2）；mamba 分支依赖可选 `mamba_ssm` 且无 ONNX 路径；本项目实现不等同于完整上游 NBSS 集成 | [项目 Registry](models/__init__.py) |
| **TFSkiMNet** / **RT_SEMamba** / **RT_SEMambaDistill** | 因果时频与状态空间 SE 模型 | [项目 Registry](models/__init__.py) |
| **ZipEnhancer** | Zipformer 增强基线；这是 Registry 中的 `model.type`。官方 checkpoint 的 chunk 推理入口为 `onnx/infer/infer_chunk_zipenhancer.py`，不是独立 Registry 名 | [ModelScope](https://www.modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base) |
| **FastULCNet** | 超轻量因果网络；已集成官方 CRM/Comfi 语义、训练 YAML 与 synthetic 长序列漂移工具。共享标准 Hann 仍不同于官方 `hann75`/WOLA，尚无 checkpoint 或真实 90 秒质量复现。 | [Fast-ULCNet](https://github.com/narrietal/Fast-ULCNet) |
| **ULUNAS** | U-Net Look-ahead NAS 架构 | [ul-unas](https://github.com/Xiaobin-Rong/ul-unas) |

### GAN 生成式模型

| 模型 | 说明 | 参考来源 |
|------|------|----------|
| **MambAttentionGenerator** | Mamba + Attention 生成器 | [MambAttention: Mamba With Multi-Head Attention for Generalizable Single-Channel Speech Enhancement (IEEE/ACM TASLP 2026)](https://github.com/NikolaiKyhne/MambAttention) |
| **RWSAMambaUNetGenerator** | 已固定官方 commit `02635ab7e92a0584fde729f8a55f398c2379da82`，并完成 framework wrapper 与官方结构语义对齐；尚未验证官方 checkpoint 或论文指标。 | [RWSA-MambaUNet (ICASSP 2026)](https://github.com/NikolaiKyhne/RWSAMamba-UNet/tree/02635ab7e92a0584fde729f8a55f398c2379da82) |
| **MossFormerGANGenerator** | MossFormer2 风格 GAN 生成器 | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) |
| **MPSEnetGenerator** | MP-SENet 风格生成器 | [MP-SENet (Interspeech 2023)](https://github.com/yxlu-0102/MP-SENet) |
| **MetricDiscriminator** / **SpectralDiscriminator** / **MetricMagDiscriminator** | MetricGAN 判别器 | [MetricGAN (Fu et al. 2019)](https://arxiv.org/abs/1905.04874) / [MP-SENet](https://github.com/yxlu-0102/MP-SENet) |

### Flow 生成式模型

| 模型 | 说明 | 参考来源 |
|------|------|----------|
| **MeanFlowSE** | 已提供 complex-STFT MeanFlow objective、FD-JVP、task-local trainer/config/tests 与纯 PyTorch NCSN++ 风格项目实现；不兼容官方 checkpoint，且尚未训练复现论文指标。 | [MeanFlowSE](https://github.com/liduojia1/MeanFlowSE/tree/6e0fe649e9916046ca1147343fb4a2fb69f87e19) |
| **MeanSE** | 项目原生 complex-STFT 生成式 SE 变体；与 `MeanFlowSE` 分别注册，尚未训练复现论文指标。 | [项目 Registry](models/__init__.py) |

### 预训练生成式模型（PASE 系列）

| 模型 | 说明 | 参考来源 |
|------|------|----------|
| **PASE** / **PASEDiscriminator** | 基于 WavLM 音素先验的低幻觉生成式语音增强 | [cisco-open/pase](https://github.com/cisco-open/pase) / [arXiv](https://arxiv.org/abs/2511.13300) |
| **StuPASE** / **StuPASEFlowTrainModel** / **StuPASEMelVocoderGenerator** | 基于流匹配的录音室级低幻觉生成式语音增强 + Mel 声码器 | [cisco-open/pase](https://github.com/cisco-open/pase) / [arXiv](https://arxiv.org/abs/2603.09234) |
| **UniPASE** | 基于 WavLM+PLC 的生成式 waveform-native 语音增强（4 阶段离线训练，见 `configs/se/urgent2026/pase/unipase/`） | [UniPASE](https://github.com/Xiaobin-Rong/unipase) |
| **GAP_URGENet** | 面向 URGENT2026 的广义音频先验 URGENet（PASE 系列，见 `configs/se/urgent2026/pase/gap_urgenet/`） | [URGENT2026 Challenge](https://urgent-challenge.github.io/urgent2026/) |

---

## 📚 BibTeX 引用

模型来源见上方「模型清单」；本节集中维护论文作者、会议与引用信息。

```bibtex
@inproceedings{rong2024gtcrn,
  title={GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources},
  author={Rong, Xiaobin and Sun, Tianchi and Zhang, Xu and Hu, Yuxiang and Zhu, Changbao and Lu, Jing},
  booktitle={Proc. IEEE ICASSP},
  pages={971--975},
  year={2024},
  doi={10.1109/ICASSP48485.2024.10448310}
}

@article{ahn2025fastenhancer,
  title={FastEnhancer: Speed-Optimized Streaming Neural Speech Enhancement},
  author={Ahn, Sunghwan and Han, Jinmo and Woo, Beom Jun and Kim, Nam Soo},
  journal={arXiv preprint arXiv:2509.21867},
  year={2025}
}

@inproceedings{schroter2022deepfilternet,
  title={DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio Based on Deep Filtering},
  author={Schr{\"o}ter, Hendrik and Escalante-B., Alberto N. and Rosenkranz, Tobias and Maier, Andreas},
  booktitle={Proc. IEEE ICASSP},
  pages={7407--7411},
  year={2022},
  doi={10.1109/ICASSP43922.2022.9747055}
}

@inproceedings{zhao2024mossformer2,
  title={MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation},
  author={Zhao, Shengkui and Ma, Yukun and Ni, Chongjiu and Zhang, Chong and Wang, Hao and Nguyen, Trung Hieu and Zhou, Kun and Yip, Jia Qi and Ng, Diaruwen and Ma, Bin},
  booktitle={Proc. IEEE ICASSP},
  pages={10356--10360},
  year={2024},
  doi={10.1109/ICASSP48485.2024.10445985}
}

@inproceedings{zhao2022frcrn,
  title={FRCRN: Boosting Feature Representation Using Frequency Recurrence for Monaural Speech Enhancement},
  author={Zhao, Shengkui and Ma, Bin and Watcharasupat, Karn N. and Gan, Woon-Seng},
  booktitle={Proc. IEEE ICASSP},
  pages={9281--9285},
  year={2022},
  doi={10.1109/ICASSP43922.2022.9747578}
}

@inproceedings{yu2023bsrnn,
  title={High Fidelity Speech Enhancement with Band-split RNN},
  author={Yu, Jianwei and Chen, Hangting and Luo, Yi and Gu, Rongzhi and Weng, Chao},
  booktitle={Proc. INTERSPEECH},
  pages={2483--2487},
  year={2023},
  doi={10.21437/Interspeech.2023-1433}
}

@inproceedings{lu2024bsroformer,
  title={Music Source Separation with Band-Split RoPE Transformer},
  author={Lu, Wei-Tsung and Wang, Ju-Chiang and Kong, Qiuqiang and Hung, Yun-Ning},
  booktitle={Proc. IEEE ICASSP},
  pages={481--485},
  year={2024},
  doi={10.1109/ICASSP48485.2024.10446843}
}

@inproceedings{wang2023tfgridnet,
  title={TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural Speaker Separation},
  author={Wang, Zhong-Qiu and Cornell, Samuele and Choi, Shukjae and Lee, Younglo and Kim, Byeong-Yeol and Watanabe, Shinji},
  booktitle={Proc. IEEE ICASSP},
  year={2023},
  doi={10.1109/ICASSP49357.2023.10095281}
}

@inproceedings{gulati2020conformer,
  title={Conformer: Convolution-augmented Transformer for Speech Recognition},
  author={Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Parmar, Niki and Zhang, Yu and Yu, Jiahui and Han, Wei and Wang, Shibo and Zhang, Zhengdong and Wu, Yonghui and Pang, Ruoming},
  booktitle={Proc. INTERSPEECH},
  pages={5036--5040},
  year={2020},
  doi={10.21437/Interspeech.2020-3015}
}

@inproceedings{lu2023mpsenet,
  title={MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra},
  author={Lu, Ye-Xin and Ai, Yang and Ling, Zhen-Hua},
  booktitle={Proc. INTERSPEECH},
  pages={3834--3838},
  year={2023},
  doi={10.21437/Interspeech.2023-1441}
}

@article{wang2025zipenhancer,
  title={ZipEnhancer: Dual-Path Down-Up Sampling-based Zipformer for Monaural Speech Enhancement},
  author={Wang, Haoxu and Tian, Biao},
  journal={arXiv preprint arXiv:2501.05183},
  year={2025}
}

@inproceedings{chen2022fullsubnetplus,
  title={FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms for Speech Enhancement},
  author={Chen, Jun and Wang, Zilin and Tuo, Deyi and Wu, Zhiyong and Kang, Shiyin and Meng, Helen},
  booktitle={Proc. IEEE ICASSP},
  pages={7857--7861},
  year={2022},
  doi={10.1109/ICASSP43922.2022.9747888},
  url={https://github.com/RookieJunChen/FullSubNet-plus}
}

@article{rong2026ulunas,
  title={UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search},
  author={Rong, Xiaobin and Yang, Leyan and Wang, Dahan and Hu, Yuxiang and Zhu, Changbao and Chen, Kai and Lu, Jing},
  journal={IEEE/ACM Trans. Audio, Speech and Language Processing},
  volume={34},
  pages={1085--1096},
  year={2026},
  doi={10.1109/TASLPRO.2026.3661271}
}

@article{arrieta2026fastulcnet,
  title={Fast-ULCNet: A Fast and Ultra Low Complexity Network for Single-Channel Speech Enhancement},
  author={Arrieta Larraza, Nicol{\'a}s and de Koeijer, Niels},
  journal={arXiv preprint arXiv:2601.14925},
  year={2026}
}

@article{rong2026pase,
  title={PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement},
  author={Rong, Xiaobin and Hu, Qinwen and Yesilbursa, Mansur and Wojcicki, Kamil and Lu, Jing},
  journal={Proc. AAAI Conf. Artif. Intell.},
  volume={40},
  number={39},
  pages={32826--32834},
  year={2026}
}

@article{rong2026stupase,
  title={StuPASE: Towards Low-Hallucination Studio-Quality Generative Speech Enhancement},
  author={Rong, Xiaobin and Gao, Jun and Wang, Zheng and Yesilbursa, Mansur and Wojcicki, Kamil and Lu, Jing},
  journal={arXiv preprint arXiv:2603.09234},
  year={2026}
}

@misc{rong2026unipase,
  title={UniPASE: A Generative Model for Universal Speech Enhancement with High Fidelity and Low Hallucinations},
  author={Rong, Xiaobin and Wang, Zheng and Wang, Yushi and Gao, Jun and Lu, Jing},
  year={2026},
  eprint={2604.14606},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2604.14606}
}
```
