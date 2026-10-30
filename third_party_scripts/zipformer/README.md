# third_party_scripts/zipformer — TER 评测用 Zipformer ASR 模型

本目录存放 REAL-TSE 评测 TER 指标（zh=CER / en=WER）使用的
**sherpa-onnx Zipformer transducer** 模型（与 REAL-TSE-Challenge 官方
评测口径一致）。模型文件（`*.onnx`、`test_wavs/` 等）**不进 git**
（`.gitignore` 已忽略），需按下方说明自行下载放置。

## 目录结构与存放约定

```
third_party_scripts/zipformer/
├── README.md                                              # 本文件
├── sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/          # 中文（~290MB）
│   ├── encoder-epoch-20-avg-1.int8.onnx
│   ├── decoder-epoch-20-avg-1.int8.onnx
│   ├── joiner-epoch-20-avg-1.int8.onnx
│   └── tokens.txt
└── sherpa-onnx-zipformer-gigaspeech-2023-12-12/           # 英文（~290MB）
    ├── encoder-epoch-99-avg-1.int8.onnx
    ├── decoder-epoch-99-avg-1.int8.onnx
    ├── joiner-epoch-99-avg-1.int8.onnx
    └── tokens.txt
```

| 子目录 | 语种 | 训练数据 | 来源 |
|--------|------|----------|------|
| `sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/` | 中文 | WenetSpeech + AISHELL-1/2/4 + AliMeeting + KeSpeech + MagicData-RAMC | k2-fsa/sherpa-onnx GitHub Release（tag `asr-models`） |
| `sherpa-onnx-zipformer-gigaspeech-2023-12-12/` | 英文 | GigaSpeech 10000h | 同上 |

## 下载方式

```bash
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
DEST=third_party_scripts/zipformer
for name in sherpa-onnx-zipformer-multi-zh-hans-2023-9-2 \
            sherpa-onnx-zipformer-gigaspeech-2023-12-12; do
  curl -L -C - -o "/tmp/$name.tar.bz2" "$BASE/$name.tar.bz2"   # 支持断点续传
  mkdir -p "$DEST/$name"
  tar -xjf "/tmp/$name.tar.bz2" -C "$DEST/$name" --strip-components=1
done
```

也可参考 REAL-TSE-Challenge 仓库的 `utils/download_zipformer.py`。

## 使用方式

评测流水线经 CLI 注入（见 `evaluate/eval_pipeline_tse.py`）：

```bash
python -m evaluate.eval_pipeline_tse \
    --metrics ter \
    --ter_model_dir_zh third_party_scripts/zipformer/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2 \
    --ter_model_dir_en third_party_scripts/zipformer/sherpa-onnx-zipformer-gigaspeech-2023-12-12 \
    ...
```

推理引擎为可选依赖 `sherpa-onnx`（`pip install sherpa-onnx`），未安装时
评测流水线自动跳过 TER 指标。
