# 第三方资产获取与放置

本文件是 `third_party_scripts/` 下**未纳入 Git** 的第三方源码、模型和权重的统一索引。第三方资产体积大、许可证各自独立，主仓库只提交本文件、`third_party_scripts/README.md` 和少量适配脚本；资产本体按需下载。

用途、依赖和运行边界的详细说明仍以 [`third_party_scripts/README.md`](../../../third_party_scripts/README.md) 为准；本文件只补下载入口、目标路径和版本锚点。路径均相对仓库根目录。

## 评估与指标模型

| 资产 | 本地路径 | 用途 | 官方下载 | 目标文件 |
|---|---|---|---|---|
| DNSMOS | `third_party_scripts/DNSMOS/` | SE/AEC 的 P.835 MOS | [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | `DNSMOS/sig_bak_ovr.onnx`、`DNSMOS/model_v8.onnx` |
| NISQA | `third_party_scripts/NISQA/` | NISQA v2 MOS | [gabrielmittag/NISQA](https://github.com/gabrielmittag/NISQA) | `weights/nisqa.tar`（随上游仓库提供） |
| UTMOS | `third_party_scripts/UTMOS/` | UTMOS22 MOS | [sarulab-speech/UTMOS22](https://github.com/sarulab-speech/UTMOS22)、[权重直链](https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt) | `utmos22_strong_step7459_v1.pt`；代码在 `SpeechMOS/speechmos/` |
| SCOREQ | `third_party_scripts/SCOREQ/` | SCOREQ 质量分 | [Zenodo record 15739280](https://zenodo.org/records/15739280) | `adapt_nr_telephone.onnx` |
| URGENT-PK | `third_party_scripts/URGENT-PK/` | 成对感知质量排序 | [urgent-challenge/URGENT-PK](https://github.com/urgent-challenge/URGENT-PK)、[权重直链](https://huggingface.co/Twinkzzzzz/URGENT-PK/resolve/main/mel_resnet34_urgent24.ckpt) | `exp/mel_resnet34_urgent24.ckpt` |
| AECMOS | `third_party_scripts/AECMOS/` | AECMOS 回声/退化分 | [microsoft/AEC-Challenge](https://github.com/microsoft/AEC-Challenge/tree/master/AECMOS) | `Run_1663829550_Stage_0.onnx`、`Run_1663915512_Stage_0.onnx`、`Run_1668423760_Stage_0.onnx` |

DNSMOS 可直接取单文件：

```bash
mkdir -p third_party_scripts/DNSMOS/DNSMOS
wget -O third_party_scripts/DNSMOS/DNSMOS/sig_bak_ovr.onnx \
    https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
wget -O third_party_scripts/DNSMOS/DNSMOS/model_v8.onnx \
    https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx
```

NISQA 直接用上游仓库即可，`weights/nisqa.tar` 已在仓库中：

```bash
git clone https://github.com/gabrielmittag/NISQA.git third_party_scripts/NISQA
git -C third_party_scripts/NISQA checkout f0cdd29
```

## VAD 与 Noise Filter Pipeline

| 资产 | 本地路径 | 用途 | 官方下载 | 目标文件 |
|---|---|---|---|---|
| SileroVAD | `third_party_scripts/SileroVAD/` | 可选 VAD backend | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | `silero_vad.jit` |
| Qwen2.5-Omni-7B | `third_party_scripts/Qwen2.5-Omni-7B/` | Noise Filter C1 可懂度/人声判定 | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)、[ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B) | 完整模型目录（约 21 GB） |
| icefall AudioTag | `third_party_scripts/icefall_audiotaq/` | Noise Filter C2 AudioSet 标签 | [HF 模型](https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12) | `exp/pretrained.pt`、`data/class_labels_indices.csv` |
| Apollo | `third_party_scripts/Apollo/` | Noise Filter 可选 Stage D1 语音修复 | [JusperLee/Apollo](https://github.com/JusperLee/Apollo)、[HF 权重](https://huggingface.co/JusperLee/Apollo) | `apollo_model_uni.ckpt` |

Qwen 与 AudioTag 的下载示例：

```bash
huggingface-cli download Qwen/Qwen2.5-Omni-7B \
    --local-dir third_party_scripts/Qwen2.5-Omni-7B
huggingface-cli download marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12 \
    --local-dir third_party_scripts/icefall_audiotaq
```

`icefall_audiotaq` 里的适配脚本和 `.gitkeep` 是主仓库文件；从 Hub 下载后只覆盖/补充模型与标签表，不要删除这些脚本。Apollo 的 `apollo_repo` 在上游是独立 checkout；本机若使用符号链接，新机器可直接把仓库克隆到 `third_party_scripts/apollo_repo`。

## REAL-TSE 可选 external runtime

这些资产只在 `evaluate/eval_pipeline_tse.py` 请求对应指标时消费；缺失时对应可选指标会跳过或按实现降级。

| 资产 | 本地路径 | 指标 | 官方下载 | 目标文件 |
|---|---|---|---|---|
| WeSpeaker | `third_party_scripts/wespeaker/` | `spksim` | [WeSpeaker pretrained models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md) | `en/model.onnx`、`chs/model.onnx` |
| Zipformer ASR | `third_party_scripts/zipformer/` | `ter` | [sherpa-onnx ASR models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | `sherpa-onnx-zipformer-gigaspeech-2023-12-12/`、`sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/` 下的 encoder/decoder/joiner/tokens |
| FireRedASR2S | `third_party_scripts/FireRedASR2S/` | `timing` | [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)、[模型合集](https://huggingface.co/collections/FireRedTeam/fireredasr2s)、[ModelScope 合集](https://www.modelscope.cn/collections/xukaituo/FireRedASR2S) | `pretrained_models/FireRedVAD/VAD/` 等 |

`zipformer` 的两个目录可以从 sherpa-onnx release 直接取：

```text
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2
```

FireRedASR2S 的完整安装与模型准备见其 README；本仓库只消费 `pretrained_models/FireRedVAD/VAD`。

## 需要 Git 克隆的上游仓库

| 本地路径 | 上游 | 本机锚定 commit |
|---|---|---|
| `third_party_scripts/NISQA/` | https://github.com/gabrielmittag/NISQA.git | `f0cdd29` |
| `third_party_scripts/URGENT-PK/` | https://github.com/urgent-challenge/URGENT-PK.git | `484269f` |
| `third_party_scripts/icefall/` | https://github.com/k2-fsa/icefall.git | `7a35ca2` |
| `third_party_scripts/FireRedASR2S/` | https://github.com/FireRedTeam/FireRedASR2S.git | `4e7d9aa` |
| `third_party_scripts/wespeaker_official/` | https://github.com/wenet-e2e/wespeaker.git | `dfa7419` |
| `third_party_scripts/apollo_repo/` | https://github.com/JusperLee/Apollo.git | `c46640f` |

```bash
git clone https://github.com/urgent-challenge/URGENT-PK.git third_party_scripts/URGENT-PK
git -C third_party_scripts/URGENT-PK checkout 484269f
```

其余仓库同理。commit 是本机验证过的版本，不是强制版本；上游更新后如果要复现已验证行为，优先回到该 commit。

## 说明

- 以上资产都没有进入主仓库；`third_party_scripts/` 只跟踪 README、`.gitkeep` 和少量适配脚本。
- 只做训练主链时不需要全部安装：评估指标、Noise Filter、REAL-TSE 可选指标各自按需准备。
- 新增第三方资产时，在本文件补一行官方入口与目标文件，并在 `third_party_scripts/README.md` 记录用途、依赖和可选性。
- 下载脚本必须保持 HTTPS 校验；不要为单个资产引入关闭证书校验的下载逻辑。
