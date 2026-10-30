# 📊 evaluate/ — 评估脚本

<p align="center">
  <strong>SE、AEC 与 TSE 模型的全方位评估工具集</strong><br>
  从一键流水线到独立脚本，覆盖侵入式指标、非侵入式音质评估与下游任务。
</p>

---

## 这是什么

模型训练完之后，用这里的脚本回答一个问题：**这个模型到底做得怎么样**。

做法是：把测试集的音频（带噪语音、带回声的麦克风信号、多人混合语音）送进模型，拿到模型处理后的
音频；再用一组客观指标，把它和「干净参考音频」或「人工标注」比较并打分。本目录提供两种入口：

1. **统一流水线**（推荐，覆盖绝大多数场景）：一条命令完成「推理 → 算指标 → 汇总」，SE / AEC / TSE 各有一个入口；
2. **独立脚本**（高级用法）：只做其中一环，或流水线暂未覆盖的场景。

第一次使用请读 [快速开始](#快速开始)；只想评 TSE 请读 [TSE（REAL-TSE）评测](#tse-real-tse-评测与官方工具链对齐)。

## 术语速查

先认识本文反复出现的词，后面读起来会顺畅很多：

| 术语 | 含义 | 为什么要关心 |
|---|---|---|
| SE / AEC / TSE | 语音增强 / 声学回声消除 / 目标说话人提取，本目录可评估的三个任务 | 决定用哪条流水线 |
| SCP | 纯文本清单文件（Kaldi 风格），每行是「音频 ID + 路径」（本仓库的部分清单还带采样率与采样点数字段） | 流水线靠它确定「要评哪些文件」 |
| 有参考指标 | 必须同时提供干净参考音频才能计算，即业内所称「侵入式」指标：PESQ、STOI、ESTOI、SI-SDR | 没有参考音频时这些指标直接不可用 |
| PESQ | 语音质量的客观近似分（ITU-T P.862），越高越好 | 最常用的音质分数 |
| STOI / ESTOI | 可懂度分数（0–1），越高越听得清 | 只关心「听不听得懂」时看它 |
| SI-SDR | 尺度不变信号失真比（dB），越高越好 | 对整体失真最敏感的波形级指标 |
| 无参考 MOS 类 | 不需要干净参考、由模型直接预测音质，即业内所称「非侵入式」指标：DNSMOS、NISQA、UTMOS、SCOREQ | 盲测集没有干净参考，只能靠它们 |
| ERLE / AECMOS | 回声抑制量（dB）/ AEC 专用的无参考回声与损伤评分 | AEC 的核心指标 |
| TER | 目标语音经 ASR 转写后的错误率（中文按字 CER、英文按词 WER），越低越好 | TSE 的可懂度指标 |
| SpkSim | 说话人相似度：输出与注册语音的声纹余弦相似度 | 判断 TSE 是否真的提对了人 |
| Timing-F1 | 目标说话人活动时间段与人工标注的重合度（帧级 F1） | 判断 TSE 提得干不干净、漏不漏 |
| `--exp_dir` | 一次训练实验的输出目录（内含配置与 checkpoint） | 流水线靠它自动找到要评的模型 |

## 📑 目录

- [推荐方式：评估流水线](#-推荐方式评估流水线)
  - [三步跑通](#三步跑通以-se-为例)
  - [快速开始](#快速开始)
  - [TSE（REAL-TSE）评测与官方工具链对齐](#tse-real-tse-评测与官方工具链对齐)
  - [VoiceBank 评估说明](#voicebank-评估说明)
  - [输出目录规范](#输出目录规范)
  - [流水线特性](#流水线特性)
  - [高级选项](#高级选项)
  - [数据集标签](#数据集标签)
  - [评估资产与指标边界](#评估资产与指标边界)
  - [测试清单规模](#测试清单规模)
  - [单独准备 SCP](#单独准备-scp)
- [独立脚本（高级用法）](#-独立脚本高级用法)
- [AEC 仿真测试集](#-aec-仿真测试集)

---

## 🚀 推荐方式：评估流水线

大部分场景请使用统一评估流水线，无需手动逐个调用脚本。

### 三步跑通（以 SE 为例）

1. **准备测试数据**：`export USE_TRAIN_TEST_DATA_ROOT=/path/to/test-data`。
   不设置时，默认路径只指向仓库内被 `.gitignore` 忽略的 `data/test/` 占位布局，实际评估会直接报错。
2. **跑流水线**：`python evaluate/eval_pipeline_se.py --exp_dir "$SE_EXP_DIR" --datasets all --step all --audio_quality`。
3. **看结果**：结果写在 `$USE_TRAIN_TEST_DATA_ROOT/exp/eval/se/<实验名>/`（未设置环境变量时是仓库内 `exp/eval/`）。
   每个测试集子目录下有 `enhanced/`（增强音频，含 `inf.scp` 清单）和 `RESULTS.txt`（该测试集的指标），
   上一级还有 `summary.json` 汇总全部测试集。

| 流水线 | 任务 | 测试集 | 指标 |
|--------|------|--------|------|
| `eval_pipeline_aec.py` | AEC 模型全维度测试 | AEC 官方盲测集 (1 个, icassp2023)、已有 AEC 仿真资产 (7 场景)、VoiceBank、URGENT、DNS Challenge | PESQ、STOI、ESTOI、SI-SDR、DNSMOS、NISQA、UTMOS、SCOREQ、**ERLE**、**AECMOS（需显式 `--aecmos`，默认跳过）** |
| `eval_pipeline_se.py` | SE 模型标准测试 | VoiceBank、URGENT、DNS Challenge、AEC 仿真 (可选) | PESQ、STOI、ESTOI、SI-SDR、DNSMOS、NISQA、UTMOS、SCOREQ |
| `eval_pipeline_tse.py` | TSE 模型 REAL-TSE 评测 | REAL-T dev / EVAL（官方 `*_meta.csv`） | **TER**（zipformer zh/en）、**SpkSim**（wespeaker）、**DNSMOS**（P.835+P808）、**Timing-F1**（FireRedVAD）、SI-SDR、PESQ、STOI |

### TSE（REAL-TSE）评测与官方工具链对齐

`eval_pipeline_tse.py` 按仓库内移植的 `REAL-TSE-Challenge/utils/` 评测流程计算四类指标；只有官方资产、模型与依赖版本齐全且一致时，结果才具备相应官方口径的可比性。汇总器对可识别的 current summary 只读取 active 且 `computed > 0` 的指标 CSV；summary 未提供显式 `summary.metrics` 时，可由 `computed > 0` 推断。仅非空 `run_id` 触发同一 run 的 `result_manifest.json` 绑定。summary 缺失或损坏时 payload 为 `None`，回退 legacy 模式并固定读取官方约定的五个 CSV；summary 存在但没有 `summary.coverage` 时 fail-closed，不读取任何指标 CSV。

TSE 推理 artifact 的 canonical 实现在 `tasks/tse/inference/artifacts.py`。下面这段规则可以用一句话概括：**每条输出都绑定「用了哪份模型、哪份输入」，事后能被独立复核；只要有一环内容对不上，流水线就拒绝复用旧产物**。主 checkpoint 与外部 speaker checkpoint 可以是 symlink，但都会先解析到 `resolve(strict=True)` 的 canonical target，再通过稳定文件描述符（FD）一次捕获不可变 bytes；对应 loader 直接消费这份快照，不会按路径重新打开，并在加载后复核路径仍绑定同一内容身份。mixture、enrollment、cue 资源文件、输出 WAV 与 sidecar 的 leaf 或任一 ancestor symlink 均被拒绝。

模型 IO 合同声明的 cue role 决定该次推理要读哪些 cue：`speaker_waveform`（`enrollment`/`enroll` 是兼容别名）、`speaker_embedding`、`spatial`（static/track npy）、`visual`（features npy 或 raw video）、`textual`（phoneme JSON）；`attributes['cues']` 里的 inline cue JSON 与文件 cue 走同一条绑定路径。声明的 role 缺失时显式报错，未声明的 role 一律拒绝。

每条输出 UID 绑定 dataset、mixture 与 cue identity（全部实际 cue 文件路径 + canonical inline cue JSON 摘要）的 canonical JSON 的 SHA-256 前 32 个十六进制字符；纯 enrollment 入口继续使用 dataset/mixture/enrollment identity，保持既有 REAL-TSE 输出 basename 稳定。manifest 使用 `tse-inference-manifest-v4` 并声明 `uid_scheme`（`...cues-v2-128`），artifact metadata 使用 `tse-inference-artifact-v6`；旧 schema、旧 UID 或旧 sidecar 资产不会复用，也不会自动删除。

```bash
# 单数据集示例：DEV/AISHELL-4。Timing 官方口径还需要 DEV/json 下的 GT
# overlap 资产与可用的 FireRedASR2S；缺失时会记录并降级为非官方近似口径。
# REAL_TSE_ROOT 必须是一个具体 split 的根目录，其下直接包含
# DEV/、mixtures/、enrolment_speakers/；不要传这些 split 的共同父目录。
REAL_TSE_ROOT=/path/to/REAL-TSE/REAL-T-dev
CHECKPOINT=/path/to/checkpoint
# --output_dir 的值就是本轮评估目录，不会自动追加任何前缀；
# exp/eval/tse/run_name 只是推荐组织方式，后续汇总必须使用完全相同的目录。
TSE_OUTPUT_DIR=exp/eval/tse/run_name
TER_ZH_MODEL_DIR=/path/to/zipformer-zh
TER_EN_MODEL_DIR=/path/to/zipformer-en
python -m evaluate.eval_pipeline_tse \
    --meta "$REAL_TSE_ROOT/DEV/AISHELL-4_meta.csv" \
    --audio_root "$REAL_TSE_ROOT" \
    --model "$CHECKPOINT" --output_dir "$TSE_OUTPUT_DIR" \
    --metrics spksim,dnsmos,timing,ter \
    --timing_gt_json_base_dir "$REAL_TSE_ROOT/DEV/json" \
    --fireredasr2s_root third_party_scripts/FireRedASR2S \
    --ter_model_dir_zh "$TER_ZH_MODEL_DIR" \
    --ter_model_dir_en "$TER_EN_MODEL_DIR" \
    --timing_frame_shift 0.01 --timing_collar 0.05 --timing_match_tolerance 0.02

# 可识别 current summary 只消费 active 且 computed > 0 的 CSV；无显式
# summary.metrics 时由 computed > 0 推断，仅非空 run_id 触发同 run manifest 绑定。
# summary 缺失或损坏时 payload=None，回退官方五个 CSV；summary 无 coverage 时
# fail-closed，不读取任何指标 CSV。
python -m evaluate.aggregate_eval_summary --output_dir "$TSE_OUTPUT_DIR"

# 因果/延迟验证（仓库内移植的扰动法；
# TSE/SE/AEC 三任务统一入口：tse x=mix+aux=enroll / se 单输入 /
# aec x=mic+aux=ref 只扰动 mic，ref 自动与 mic 等长。
# SE/AEC 路径按配置中的 data.mode 判定任务：checkpoint 内嵌配置或 --config
# 必须显式声明 data.mode，未声明直接报错，不做默认回退）
CONFIG=/path/to/config.yaml
python -m evaluate.latency_verification --task tse --model "$CHECKPOINT" --config "$CONFIG"
python -m evaluate.latency_verification --task se  --model /path/to/se-checkpoint.tar
python -m evaluate.latency_verification --task aec --model /path/to/aec-checkpoint.tar
```

注意：官方开源工具链不含 leaderboard 的 dense ranking 排名（排名在官方评测服务器侧）；
`score.py` 为 SI-SNR/PESQ/STOI 有参考指标，已由 pipeline 的 `sisdr/pesq/stoi` 覆盖。

### 快速开始

```bash
# /path/to/writable/... 表示本机可写的实验目录
AEC_EXP_DIR=/path/to/writable/exp/aec/run_name
SE_EXP_DIR=/path/to/writable/exp/se/run_name

# AEC 模型 — 常规全自动路径；显式排除已退役的 AEC-sim 离线生成
python evaluate/eval_pipeline_aec.py \
    --exp_dir "$AEC_EXP_DIR" \
    --datasets aec_official,voicebank,urgent,dns_challenge --step all --audio_quality

# SE 模型 — 全自动
python evaluate/eval_pipeline_se.py \
    --exp_dir "$SE_EXP_DIR" \
    --datasets all --step all --audio_quality

# 仅推理（显式排除 aec_sim，避免触发已退役生成器）
python evaluate/eval_pipeline_aec.py \
    --exp_dir "$AEC_EXP_DIR" \
    --datasets aec_official,voicebank,urgent,dns_challenge --step inference

# 仅评估（读取已有增强音频）
python evaluate/eval_pipeline_aec.py \
    --exp_dir "$AEC_EXP_DIR" \
    --datasets aec_official,voicebank,urgent,dns_challenge --step evaluate --audio_quality

# 指定普通数据集（不含 AEC-sim）
python evaluate/eval_pipeline_aec.py --exp_dir "$AEC_EXP_DIR" --datasets aec_official,voicebank
```

### VoiceBank 评估说明

- `--exp_dir` 与 `--config`+`--checkpoint` 二选一；显式指定时 `--config` 与 `--checkpoint`
  **必须成对提供**，只给 `--checkpoint` 会直接报错
  `Provide --exp_dir OR both --config and --checkpoint`。
- 默认只计算带 clean 参考的客观指标（SI-SDR / PESQ / STOI / ESTOI）；MOS 类指标
  （DNSMOS / NISQA / UTMOS / SCOREQ）需显式加 `--audio_quality` 才运行（较慢）。

### 输出目录规范

下面先给目录布局；再解释 `*_current.json` / `*_manifest.json` 这些标记文件为什么存在。**只想跑通流水线的话，看完目录树即可，标记文件的细节可以跳过。**

以下 `${USE_TRAIN_TEST_DATA_ROOT}/exp/eval/` 目录规范仅适用于 SE/AEC 流水线；TSE 精确写入其 `--output_dir` 指定的目录，`exp/eval/tse/` 只是推荐位置。未设置 `USE_TRAIN_TEST_DATA_ROOT` 时，SE/AEC 保留仓库内 `exp/eval/` 兼容布局：

```
${USE_TRAIN_TEST_DATA_ROOT}/exp/eval/
├── aec/<实验名>/
│   ├── blind_test_set_icassp2023_doubletalk/
│   │   ├── enhanced/          # 推理结果 (含 inf.scp)
│   │   └── RESULTS.txt        # 评估结果
│   ├── aec_sim/<scenario>/    # 7 场景各占一个子目录
│   │   ├── enhanced/          # 推理结果 (含 inf.scp)
│   │   └── RESULTS.txt        # 评估结果
│   ├── voicebank/
│   ├── urgent/
│   ├── summary.json             # SE/AEC 兼容汇总
│   ├── result_manifest.json     # 本轮指标文件清单与 SHA-256
│   ├── evaluation_artifact.json # 公共生命周期 envelope
│   └── evaluation_current.json  # 最终原子提交标记（CURRENT）
└── se/<实验名>/
    └── ...
```

每次评估先生成唯一 `run_id`，并由 `begin_artifact_run()` 将 `evaluation_current.json` 原子写为 `in_progress`，立即使目录中的旧 envelope 不再可消费。公共 runner 随后按 prepare → run → summary 持久化 → envelope 发布执行；只有同一 `run_id` 的 summary 与 `evaluation_artifact.json` 都存在且一致，才最后把 CURRENT 原子替换为 `committed`。committed CURRENT 同时记录 summary 和 envelope 的文件名与 SHA-256；读取方必须通过该标记校验状态、哈希及 `run_id`，发布中断时遗留的旧 envelope 或混合 run 会 fail-closed。

三个任务的 `evaluation_artifact.json` 统一记录 `status`、模型/配置 provenance、推理 coverage、指标 coverage 和任务原生 summary，指标字段仍保持 SE/AEC/TSE 各自语义。当本轮产生指标结果时，`result_manifest.json` 绑定同一 `run_id`，逐个记录指标产物的相对路径、大小和 SHA-256；summary 保存该 manifest 自身的大小与 SHA-256 引用，envelope 内嵌同一 summary，因此结果文件、manifest、summary、envelope 和 CURRENT 形成一条可校验链。TSE 继续保留 `tse_eval_summary.json`、`tse_inference_manifest.json`、`tse_eval_results.csv`、官方 CSV 和逐 WAV `.meta.json`；空 meta 或显式 `--max_samples 0` 会发布同 run 的 committed error lifecycle，并以非零状态退出。

`tse-inference-manifest-v4` 的全局 `artifact_fingerprint` 绑定主 checkpoint、配置及 resolved config、可选 speaker checkpoint、speaker 前端身份（checkpoint/fbank/preprocess）、推理实现源码与参数；每个成功条目再分别绑定 mixture/enrollment/cue 文件输入、cue 声明（角色 kind 与 inline cue JSON）、输出 WAV 和 sidecar 的内容身份。sidecar 同时交叉绑定该条目的输入/cue/输出与上述全局 artifact；manifest 写出后会重读并逐项复核整条链，旧版、缺项或任何身份漂移都 fail-closed。

每个增强 WAV 与同名 `.meta.json` sidecar 作为一个集合事务提交：二者先写入 staging 文件并落盘，发布前复核输入身份，再共同替换并校验发布结果；任一步失败都会回滚，避免只发布其中一项。完整流水线及公开 manifest/summary writer 统一使用同一个非阻塞 `output_dir` 独占 writer lock；从整批 preflight 和 lifecycle begin 到 WAV/sidecar、manifest、CSV、summary、envelope 与 CURRENT 全部完成，第二个 writer 都会在写入本轮 artifact 前 fail-fast，生产路径每轮只执行一次 begin。summary 与 envelope 写出后、CURRENT 最终提交前，还会再次复核全局 artifact 和完整 inference manifest；失败时不会提交 `committed` CURRENT。

SE/AEC 的 `--step evaluate` 会携带当前 config/checkpoint provenance 校验已有推理资产；此时缺少 `run_manifest.json` 会直接报错，不会从孤立 wav/`inf.scp` 猜测生产结果。无 config/checkpoint provenance 的独立兼容调用仍按其显式输入契约处理。

SE/AEC checkpoint 按明确 family 加载：普通模型只接受 `model`、`model_state_dict`、`state_dict` 或裸 tensor state dict，GAN 只接受 `model_generator`；两类都统一处理 `module.` 前缀、严格 key 校验与可选 EMA。TSE/第三方 checkpoint 继续走各自兼容加载器，不会被万能 loader 混载。

流水线可从任意 cwd 启动：仓库脚本按仓库根绝对定位，调用方传入的相对路径会在启动 subprocess 前解析为绝对路径，生产代码不会通过 `chdir` 改变进程 cwd。

### 流水线特性

- ⚡ **一键完成** — 多数据集推理 + 评估
- 🎯 **分步控制** — `--step all` (默认) / `inference` / `evaluate`
- 🏷️ **数据集自选** — AEC 的 `--datasets all` 展开为 `aec_official,aec_sim,voicebank,urgent,dns_challenge`；SE 展开为 `voicebank,urgent,dns_challenge`，另加 `--include_aec_sim` 才包含 `aec_sim`
- ⚠️ **AEC-sim 生命周期** — AEC 的 `--step all` / `inference` 在包含 `aec_sim` 时会尝试调用仿真生成器；当前生成器只支持 `--dry-run`，非 dry-run 会 fail-fast。常规运行请显式排除 `aec_sim`；已有仿真资产仅可走 `--step evaluate`。
- 📋 **标准化 SCP** — 流水线会在 `USE_TRAIN_TEST_DATA_ROOT` 指定的 test 根下准备所需 SCP；未设置时才使用仓库内的 `data/test/` 占位布局
- 🔀 **AEC 跨任务** — AEC 模型在 SE 测试集上自动使用 zero-filled farend
- 🙈 **盲测集** — 无 clean reference 时跳过侵入式指标；只有传入 `--audio_quality` 才计算 DNSMOS/NISQA/UTMOS/SCOREQ 等非侵入式指标
- 📊 **ERLE + AECMOS** — 对实现支持的含回声场景计算 ERLE；仅显式传入 `--aecmos` 时计算可选 AECMOS。官方盲测 doubletalk 无 oracle echo，ERLE 为 N/A。

### 高级选项

| 标志 | 作用 |
|------|------|
| `--config` / `-C` | 指定 YAML 配置 (替代 `--exp_dir` 自动检测) |
| `--checkpoint` | 指定 checkpoint 路径 (配合 `--config` 使用) |
| `--device` | 推理设备 (默认: `cuda:0`) |
| `--aecmos` | 显式启用可选 AECMOS 指标（默认不执行） |
| `--downstream` | 启用 URGENT2026 下游指标 (WER/说话人/情感/LID) |
| `--num_sim_samples` | AEC 仿真测试集样本数 (默认: 100) |
| `--seed` | 随机种子 (默认: 42) |
| `--include_aec_sim` | SE 流水线也跑 AEC 仿真测试集 |
| `--skip-existing-inference` | 跳过已有的增强音频文件 (仅 AEC 流水线) |
| `--aec_no_stream_align` | 禁用 AEC 流式对齐 (仅用于消融测试) |
| `--aec_subsets` | AEC 官方测试子集: `all` / `doubletalk` / `farend-singletalk` / `nearend-singletalk` (默认: `all`) |

**路径覆盖选项** (用于自定义测试集位置):

| 标志 | 作用 |
|------|------|
| `--aec_testset_root` | AEC 入口：官方测试集根路径 |
| `--aec_sim_testset_root` | AEC/SE 入口：已有 AEC 仿真测试集根路径；仅读取或准备现有资产，不恢复离线生成发布 |
| `--voicebank_noisy_scp` | AEC/SE 入口：VoiceBank noisy SCP |
| `--voicebank_clean_scp` | AEC/SE 入口：VoiceBank clean SCP |
| `--urgent_noisy_scp` | AEC 入口：URGENT2026 nonblind noisy SCP（适用于嵌套 FLAC 布局） |
| `--urgent_clean_scp` | AEC 入口：URGENT2026 nonblind clean SCP（与 noisy SCP 配对） |
| `--urgent_noisy_dir` | AEC/SE 入口：URGENT2026 noisy 目录 |
| `--urgent_clean_dir` | AEC/SE 入口：URGENT2026 clean 目录 |
| `--dns_challenge_noisy_dir` | AEC/SE 入口：DNS Challenge noisy 目录 |

评估流水线不会再回退到机器相关的绝对路径。设置 `USE_TRAIN_TEST_DATA_ROOT` 后，默认输入源和标准化 SCP 输出都位于该 test 根；未设置时，共享默认值只指向仓库内被 `.gitignore` 忽略的 `data/test/...` 占位布局。实际评估时请设置该环境变量，或逐项传入上表中的 CLI 路径；显式路径不存在时会直接报错，避免继续使用陈旧 SCP。

### 数据集标签

**AEC 流水线 (`eval_pipeline_aec.py`):**
- `aec_official` — 1 个 AEC Challenge 官方盲测集 (`blind_test_set_icassp2023`；默认子集 `all`，含 doubletalk + farend-singletalk + nearend-singletalk)
- `aec_sim` — 已有 AEC 仿真测试集资产 (7 场景；仅 `--step evaluate` 可直接消费)
- `voicebank` / `urgent` / `dns_challenge` — `--datasets all` 时无条件包含 (AEC 统一 joint 模式)

**SE 流水线 (`eval_pipeline_se.py`):**
- `voicebank` / `urgent` / `dns_challenge`
- `aec_sim` — 通过 `--include_aec_sim` 可选开启

### 评估资产与指标边界

- **AEC ICASSP 2023**：原始目录为 `data/aec/AEC_Challenge/datasets/blind_test_set_icassp2023/`；`doubletalk`、`farend-singletalk`、`nearend-singletalk` 分别有 **300 / 300 / 200** 组 `mic/lpb`。推荐 `--datasets aec_official`；需要精确标签时仅使用已修复并验证过的 `aec_icassp2023`。本机生成的混合清单 `data/aec/AEC_Challenge/datasets/blind_test_set_icassp2023.scp` 同时索引 mic、lpb、enrollment 等资产，只适合盘点，不能替代流水线要求的场景级 `mic.scp` + `lpb.scp` 双 SCP。
- **AECMOS 分场景口径**：ST-FE 报 EMOS，DT 报 EMOS + DMOS，ST-NE 报 DMOS；结果字段 `AECMOS_ECHO` = EMOS、`AECMOS_DEG` = DMOS。三类场景不得混成一个均值；DT 没有 oracle echo，ERLE 仍为 N/A。
- **DNS Challenge**：本机 `data/se/DNS_Challenge_test/v2_blindset.scp`、`v3_blindset.scp`、`v5_blindset.scp` 分别索引 V2/V3/V5 noisy-only 资产，均无 clean reference，只能计算 DNSMOS/NISQA/UTMOS/SCOREQ 等非侵入式指标。当前 `dns_challenge` 的默认口径是 V3：默认值取自测试根下的 `<test root>/DNS_Challenge_test/blind_test_set_v3/noisy.scp`。`--dns_challenge_noisy_dir` 名字里虽有 `dir`，实际既可给 noisy **目录**也可给 `.scp` 文件（按扩展名判断）：给目录时只扫描该目录**顶层**的扁平 `*.wav`，不递归子目录；给 `.scp` 时该 SCP 必须带有效的 `<scp>.provenance.json` sidecar。评 V2/V5 时须显式指向对应版本的 SCP 或目录，并分开保存结果，禁止跨版本混算。
- **URGENT 与 VoiceBank**：`urgent` 的指标边界是带 clean label 的 **nonblind** 配对集，可计算侵入式指标，并可用 `--downstream` 加下游指标；它不是 leaderboard blind set。`--urgent_noisy_dir` / `--urgent_clean_dir` 同样既可给目录也可给 `.scp`：给目录时只扫描顶层扁平 `*.wav`，因此不会命中 `noisy/0/*.flac` 等嵌套文件；给 `.scp` 时要求该 SCP 带有效的 `<scp>.provenance.json` sidecar，`prepare_scp.py` 新生成、无 sidecar 的 nonblind SCP 因此接不进来。请用 noisy SCP 做显式 SCP 推理，再用 clean SCP 与 enhanced 输出 SCP 运行 `evaluate/evaluate_urgent.py` 等独立评估。`voicebank` 必须同时提供配对的 noisy/clean SCP，同样属于有参考评估。两者若缺 clean 输入，不应降级后仍宣称 PESQ/STOI/ESTOI/SI-SDR 等有参考结果。

### 测试清单规模

- DNS：`data/se/DNS_Challenge_test/v2_blindset.scp` **700** 条、`v3_blindset.scp` **600** 条、`v5_blindset.scp` **389** 条；experiment report 默认使用 V3。
- AEC：`data/aec/AEC_Challenge/datasets/blind_test_set_icassp2023.scp` 是一份 **2,400** 条 mixed inventory（800 mic + 800 lpb + 800 enrl），仅用于盘点，不能替代正式的双输入 AEC 评估。
- AEC 官方三子集：`doubletalk` / `farend-singletalk` / `nearend-singletalk` 配对数分别为 **300 / 300 / 200**。

### 单独准备 SCP

```bash
# 只准备 AEC 官方集的标准化 SCP，不跑评估；显式包含全部三个场景
python tools/prepare_data_tools/scp/prepare_testset_scps.py \
    --datasets aec_official --aec_subsets all

# 指定 test 根目录与输出目录
export USE_TRAIN_TEST_DATA_ROOT=/path/to/test-data
python tools/prepare_data_tools/scp/prepare_testset_scps.py \
    --datasets aec_official --aec_subsets all \
    --aec_testset_root data/aec/AEC_Challenge/datasets \
    --output-dir "$USE_TRAIN_TEST_DATA_ROOT"
```

`prepare_testset_scps.py` 的 `--aec_subsets` 默认仅为 `doubletalk`；准备 AEC 官方集的 doubletalk、farend-singletalk、nearend-singletalk 三个场景时必须显式传入 `--aec_subsets all`。其他目录布局兼容的数据集可按需加入 `--datasets`；不要为上述 nonblind 布局加入 `urgent`，否则 noisy 目录扫描不会命中 `noisy/0/*.flac`。该 nonblind 集应沿用显式 SCP 推理与独立评估路径。

---

## 🔧 独立脚本（高级用法）

以下脚本作为独立工具保留，用于精细化控制或流水线暂未覆盖的场景。

| 脚本 | 任务 | 使用场景 |
|------|------|----------|
| `evaluate/evaluate_urgent.py` | URGENT2026 综合评估 | 需要对自定义 enhanced SCP 跑完整下游指标 (WER、说话人相似度等) |
| `evaluate/eval_single_mos.py` | 单音频非侵入式 MOS 评估 (DNSMOS/NISQA/UTMOS/SCOREQ) | 免 SCP 快速评估单个 wav, 长音频自动分块 |
| `tools/prepare_data_tools/scp/prepare_testset_scps.py --datasets aec_sim` | 为已有 AEC 仿真资产准备 SCP（7 场景） | 仅消费现有 `--aec_sim_testset_root`；不负责恢复已退役的离线生成发布 |

### 独立脚本示例

```bash
# evaluate_urgent.py — 自定义 SCP 跑完整下游指标
python evaluate/evaluate_urgent.py \
    --ref_scp data/ref.scp --inf_scp data/enh.scp --all \
    --meta_tsv data/meta.tsv --utt2lang data/utt2lang
```

---

## 🔬 AEC 仿真测试集

当前仓库只支持消费已经准备好的 AEC 仿真测试集资产；canonical module `tools.prepare_data_tools.generators.aec.generate_simulated_testset` 的离线发布路径已退役。生成器当前仅支持 `--dry-run`，任何非 dry-run 调用都会 fail-fast 并提示迁移，`--force` 也不能恢复发布能力。不要在包含 `aec_sim` 的 AEC `--step all` / `--step inference` 中运行流水线，因为这些步骤会尝试调用已退役的生成器。

如果已有完整的 `AEC_SIM_ROOT`，请先准备 SCP，再仅用 `--step evaluate` 消费已有资产：

```bash
AEC_SIM_ROOT=/path/to/aec-sim-root
AEC_EXP_DIR=/path/to/writable/exp/aec/run_name

# 仅检查 dry-run 计划；不会发布仿真测试集
python -m tools.prepare_data_tools.generators.aec.generate_simulated_testset \
    -C configs/aec/microsoft_challenge/cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml \
    --output-dir "$AEC_SIM_ROOT" \
    --num-samples 100 --dry-run

# 为已有 AEC 仿真资产准备 SCP
python tools/prepare_data_tools/scp/prepare_testset_scps.py \
    --datasets aec_sim --aec_sim_testset_root "$AEC_SIM_ROOT"

# 仅评估已有 AEC 仿真资产，不触发生成器
python evaluate/eval_pipeline_aec.py \
    --exp_dir "$AEC_EXP_DIR" \
    --datasets aec_sim --step evaluate \
    --aec_sim_testset_root "$AEC_SIM_ROOT"
```

---

<p align="center">
  <sub>最后更新: 2026-09-21</sub>
</p>
