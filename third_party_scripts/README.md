# 📦 第三方脚本与模型 (third_party_scripts)

<p align="center">
  <strong>与 use_train 配套的第三方评估模型、VAD、分类器的下载脚本及使用说明</strong>
</p>

---

## 📑 目录

- [目录概览](#third-party-overview)
- [音频质量评估模型](#audio-quality-models)
  - [DNSMOS](#dnsmos)
  - [NISQA](#nisqa)
  - [UTMOS](#utmos)
  - [SCOREQ](#scoreq)
  - [URGENT-PK](#urgent-pk)
- [AEC 评估模型](#aec-evaluation-models)
  - [AECMOS](#aecmos)
- [VAD 模型](#vad-models)
  - [SileroVAD](#silerovad)
  - [NeMo VAD ONNX](#nemo-vad-onnx)
- [分类与检测模型](#classification-models)
  - [Qwen2.5-Omni](#qwen25-omni)
  - [icefall AudioTag](#icefall-audiotag)
- [REAL-TSE 可选 external runtime](#real-tse-可选-external-runtime)
- [依赖安装汇总](#dependency-installation)
- [项目来源与致谢](#project-sources)

---

<a id="third-party-overview"></a>
## 📁 目录概览

> 统一的下载入口、目标文件名、可选性和上游 commit 锚点见 [`docs/guides/reference/Third_Party_Assets.md`](../docs/guides/reference/Third_Party_Assets.md)。本文件保留各资产的用途、依赖和运行边界说明。

| 分类 | 目录 | 当前仓库定位 |
|------|------|----------|
| 当前直接消费的本地资产 | **DNSMOS/**、**NISQA/**、**UTMOS/**、**SCOREQ/**、**AECMOS/**、**SileroVAD/**、**NeMo-VAD-onnx/**、**URGENT-PK/** | 评估、VAD 或工具代码从这些本地路径加载源码、模型或权重；运行前确认所需文件已存在。 |
| 当前直接消费的本地资产 | **Qwen2.5-Omni-7B/**、**icefall_audiotaq/**、**Apollo/** | Noise Filter Pipeline 当前按 Stage C `C1 → C2 → C3` 执行：Qwen 仅供 C1，AudioTag 供 C2，C3 为纯规则最终准入；旧 C4 产物仅用于 legacy 拒绝/清理，C4 只保留在历史说明中。Apollo 仍仅在启用可选 Stage D1 时消费。准备方式见 [`tools/noise_filter_pipeline/README.md`](../tools/noise_filter_pipeline/README.md)。 |
| 可选 external runtime | **wespeaker/**、**zipformer/**、**FireRedASR2S/** | 仅 REAL-TSE 的 spksim、TER、timing 官方口径按需使用，入口与依赖见下文。 |

---

<a id="audio-quality-models"></a>
## 🎧 音频质量评估模型

### DNSMOS

**用途**: DNSMOS P.835 非侵入式语音质量评估。SE/AEC 的通用 loader 读取 `DNSMOS/sig_bak_ovr.onnx` 并输出 `SIG`、`BAK`、`OVRL`；Audio Filter Pipeline 的 loader 同样以该 P.835 主模型为准，另读 `DNSMOS/model_v8.onnx` 生成辅助 `P808` 分数。

**依赖**:
```bash
# SE/AEC 通用 P.835 loader；非 16 kHz 输入另需 soxr 或 librosa 重采样
pip install "onnxruntime>=1.15" numpy soxr

# Audio Filter 的 P.835 + P808 loader 还会直接导入这些包
pip install librosa pandas soundfile requests tqdm
```

**使用**: 模型目录应包含 `third_party_scripts/DNSMOS/DNSMOS/sig_bak_ovr.onnx`；Audio Filter 还要求同目录存在 `model_v8.onnx`。模型来源见 [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)。

---

### NISQA

**用途**: 语音质量与自然度预测 (NISQA v2)。用于 SE/AEC 评估流水线的 `--audio_quality` 选项。

**运行时依赖**:
```bash
pip install torch numpy pandas tqdm pyyaml librosa scipy matplotlib
```

`scikit-learn` 与 `seaborn` 属于上游 `env.yml` 的完整训练/分析环境，不是当前 NISQA 推理源码的直接 import；需要复现上游完整环境时请按 [`NISQA/README.md`](NISQA/README.md) 创建 Conda 环境。

**源码与权重**: 运行时直接导入本地 `third_party_scripts/NISQA/nisqa/` 源码，并加载 `third_party_scripts/NISQA/weights/nisqa.tar`。权重来源与上游使用方法见 [`NISQA/README.md`](NISQA/README.md)；loader 只识别 `nisqa.tar`，其他文件名须先按此约定重命名。

**仓库管理**: `NISQA/` 采用整目录忽略的本地资产模式，源码与权重需单独准备；主仓库不登记 Git 子模块。

---

### UTMOS

**用途**: UTMOS (UTokyo-SaruLab MOS) 语音自然度预测。用于 SE/AEC 评估流水线。

**依赖**:
```bash
pip install torch torchaudio
```

**源码与权重**: 运行时导入 `third_party_scripts/UTMOS/SpeechMOS/speechmos/`，并加载 `third_party_scripts/UTMOS/utmos22_strong_step7459_v1.pt`；两项均必须存在。SpeechMOS 上游来源见 [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS)。

---

### SCOREQ

**用途**: SCOREQ 语音质量评分。用于 SE/AEC 评估流水线，支持 natural/synthetic 与 nr/ref 组合。

**依赖**:
```bash
pip install "onnxruntime>=1.15" torch torchaudio
```

**使用**: natural NR 默认加载 `third_party_scripts/SCOREQ/adapt_nr_telephone.onnx`；其他模式按 loader 报错中给出的文件名准备。模型发布页为 [Zenodo record 15739280](https://zenodo.org/records/15739280)。

---

### URGENT-PK

**用途**: Audio Filter Pipeline Generation 3 Step3 的成对感知质量排序模型，与 SCOREQ、DNSMOS 共同组成固定的 3/3 MOS 打分与筛选链路。

**资产边界**: 启用时必须显式配置 `urgent_pk_checkpoint`、`urgent_pk_source_dir`、`urgent_pk_encoder` 和 `urgent_pk_backbone`；当前随附配置使用 `third_party_scripts/URGENT-PK/exp/mel_resnet34_urgent24.ckpt`、`third_party_scripts/URGENT-PK`、`mel`、`resnet34`。运行时还会校验 checkpoint digest 与 source commit，不能只提供目录后依赖隐式猜测。

**详细资料**: 上游安装、训练、推理和 checkpoint 说明见 [`URGENT-PK/README.md`](URGENT-PK/README.md)。

---

<a id="aec-evaluation-models"></a>
## 🔇 AEC 评估模型

### AECMOS

**用途**: AECMOS 回声消除质量评估 (echo + degradation)。用于 AEC 评估流水线。

**依赖**:
```bash
pip install onnxruntime numpy librosa torch
```

**使用**: AEC 评估流水线从 `third_party_scripts/AECMOS/` 中的本地 `Run_*.onnx` 文件调用；运行前确认所需文件已就位。

---

<a id="vad-models"></a>
## 🎤 VAD 模型

### SileroVAD

**用途**: Silero VAD 语音活动检测。当前消费者为 `tools/audio_filter_pipeline/` 和 `tools/noise_filter_pipeline/`。

**依赖**:
```bash
pip install torch torchaudio
```

**使用**: Silero 依赖 PyTorch 音频栈；Noise Filter Pipeline 可将其作为 VAD backend，并优先加载本地 `third_party_scripts/SileroVAD/silero_vad.jit`，缺失时才尝试 Torch Hub 缓存、下载或 `torch.hub.load` 兼容回退。Audio Filter Pipeline Generation 3 的主链已固定为 native CPU TEN VAD（`ten_vad_native_cpu`），不再以 Silero 作为主 VAD。

### NeMo VAD ONNX

**用途**: Noise Filter Pipeline 的可选 NeMo MarbleNet ONNX VAD，不依赖 `nemo_toolkit` 执行推理。

**依赖**:
```bash
pip install torch torchaudio "onnxruntime>=1.15"
```

**使用**: 将 `vad_marblenet.onnx` 或 `vad.onnx` 放入 `third_party_scripts/NeMo-VAD-onnx/`，或通过配置显式传入 `nemo_onnx_path`。受校验的下载/导出辅助入口见 [`tools/noise_filter_pipeline/README.md`](../tools/noise_filter_pipeline/README.md)。

---

<a id="classification-models"></a>
## 🏷️ 分类与检测模型

### Qwen2.5-Omni

**用途**: Qwen2.5-Omni 仅用于 `tools/noise_filter_pipeline/` Stage C1 的可懂度与人声/噪声判定。

**依赖与使用**: Qwen2.5 utilities 与模型依赖必须安装在实际运行 `tools/noise_filter_pipeline/main.py` 的同一 Python 环境中；不能仅在独立 external runtime 中安装。完整安装、模型准备和命令以 [`tools/noise_filter_pipeline/README.md`](../tools/noise_filter_pipeline/README.md) 的“环境安装”为准。

### icefall AudioTag

**用途**: 基于 icefall 的音频标签分类器 (AudioTagging Zipformer)。`tools/noise_filter_pipeline/` Stage C2 仅持久化固定 `audioset_top5`：恰 5 项，每项严格包含 `index/mid/label/prob`，evidence v5。Stage C3 无阈值、无 calibration：top-5 任一 speech MID 即 `reject`，否则按 C1 Noise/Babble 分支分别 `accept_noise` / `accept_babble`；babble MID 不参与决策；类别取 top-5 首个 noise MID，无则回退 `/m/096m7z` / `Noise`。accepted 记录的 `balance_category_mid`、`balance_category`、`balance_category_source` 均来自 C3。

发布门禁固定为 strict schema v6、policy/evidence `4/5`、parser v10、generator `6.1`、promotion marker v4。旧 full-527、`audioset_top10`、阈值、calibration 与旧版本证据字段均 fail-closed。

> 💡 当前 Noise Filter Pipeline 的推荐主方案为 Qwen2.5-Omni + icefall AudioTag；YAMNet/BEATs/AST/Step-Audio-2-mini 不属于当前管道。

**模型来源**: [marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12](https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12/tree/main)

**依赖与使用**: AudioTag 可独立以 `subprocess_daemon` external runtime 运行，主环境仅负责调度和 manifest 管理。权重与标签表位于 `third_party_scripts/icefall_audiotaq/exp/pretrained.pt` 与 `third_party_scripts/icefall_audiotaq/data/class_labels_indices.csv`；完整环境准备、模型下载和命令以 [`tools/noise_filter_pipeline/README.md`](../tools/noise_filter_pipeline/README.md) 为准。

---

## REAL-TSE 可选 external runtime

`evaluate/eval_pipeline_tse.py` 仅在请求相应指标时消费以下目录；这些均不属于默认训练依赖，缺失时对应可选指标会跳过或按实现降级。

| 目录 | 指标与入口 | 可选依赖 / 资产约定 |
|------|------------|--------------------|
| **wespeaker/** | `--metrics spksim`（默认 `--spksim_backend wespeakerruntime`） | `pip install wespeakerruntime`；本地 ONNX 约定为 `wespeaker/{en,chs}/model.onnx`。 |
| **zipformer/** | `--metrics ter` 配合 `--ter_model_dir` 或 `--ter_model_dir_zh/en` | `pip install sherpa-onnx meeteval whisper-normalizer opencc-python-reimplemented`；目录须包含 Zipformer 的 encoder/decoder/joiner ONNX 与 `tokens.txt`。 |
| **FireRedASR2S/** | `--metrics timing` 配合 `--timing_gt_json_base_dir` | FireRedASR2S 本地工作副本及其 `pretrained_models/FireRedVAD/VAD` 资产；缺失时实现会使用非官方的兼容回退。 |

REAL-TSE 指标口径、完整参数及模型准备边界见 [`tasks/tse/README.md`](../tasks/tse/README.md) 和 `evaluate/eval_pipeline_tse.py --help`。

---

<a id="dependency-installation"></a>
## 📦 依赖安装汇总

### 评估流水线指标依赖

```bash
# 基础评估 (PESQ, STOI, SI-SDR)
pip install pesq pystoi

# 本地 MOS 模型运行时；源码与模型资产仍须按上文准备
pip install "onnxruntime>=1.15" torch torchaudio numpy soundfile pandas requests tqdm pyyaml \
    librosa scipy matplotlib
```

### Noise Filter Pipeline 依赖

```bash
# 安装该管线声明的基础包与默认 VAD 依赖；Qwen2.5 utilities 和 AudioTag
# external runtime 仍须按 tools/noise_filter_pipeline/README.md 的“环境安装”另行准备。
pip install -r tools/noise_filter_pipeline/requirements.txt

# 可选 GPU VAD 加速
pip install "fireredvad[gpu]"
```

### 基础依赖合并安装（不含可选模型 runtime）

```bash
# 评估 + 过滤管线的公共 Python 依赖；模型源码、权重、Qwen/AudioTag/Apollo
# runtime 及其他按需组件不包含在此命令中。
pip install pesq pystoi "onnxruntime>=1.15" torch torchaudio numpy soundfile pandas requests tqdm \
    pyyaml librosa scipy matplotlib webrtcvad fireredvad funasr modelscope
```

---

<a id="project-sources"></a>
## 项目来源与致谢

以下集中列出评估与筛选组件的项目来源，安装与资产准备见上文。

| 项目 | 来源 |
|---|---|
| **DNSMOS** | [GitHub](https://github.com/microsoft/DNS-Challenge) |
| **NISQA** | [GitHub](https://github.com/gabrielmittag/NISQA) |
| **UTMOS** | [GitHub](https://github.com/sarulab-speech/UTMOS22) |
| **SCOREQ** | [GitHub](https://github.com/tjohnson1118/SCOREQ) |
| **URGENT-PK** | [GitHub](https://github.com/urgent-challenge/URGENT-PK) |
| **AECMOS** | [GitHub](https://github.com/microsoft/AEC-Challenge) |
| **Qwen2.5-Omni** | [GitHub](https://github.com/QwenLM/Qwen2.5-Omni) |
| **icefall AudioTag** | [GitHub](https://github.com/k2-fsa/icefall) |
| **WeSpeaker** | [GitHub](https://github.com/wenet-e2e/wespeaker) |
| **ECAPA-TDNN** | [wespeaker](https://github.com/wenet-e2e/wespeaker) |
| **Zipformer ASR (sherpa-onnx)** | [GitHub](https://github.com/k2-fsa/sherpa-onnx) |
| **FireRedVAD / FireRedASR2S** | [FireRedTeam](https://github.com/FireRedTeam/FireRedVAD) |
| **SileroVAD** | [GitHub](https://github.com/snakers4/silero-vad) |
| **TENVAD** | [GitHub](https://github.com/TEN-framework/ten-vad) |
| **FunASR (FSMN-VAD) / WebRTC VAD** | [GitHub](https://github.com/alibaba-damo-academy/FunASR) |
| **Apollo** | [GitHub](https://github.com/JasonSWFu/Apollo) |

使用相关组件时，请遵循对应项目的 License 与引用要求。

---

<p align="center">
  <sub>最后更新: 2026-10-08</sub>
</p>
