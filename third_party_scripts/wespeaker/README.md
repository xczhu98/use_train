# third_party_scripts/wespeaker — spksim 评测用 WeSpeaker ONNX 模型

本目录存放 REAL-TSE 评测 spksim 指标（与 REAL-TSE-Challenge 官方
`utils/spk_similarity_eval.py` 口径一致）使用的 **wespeakerruntime**
ResNet34-LM ONNX 模型。模型文件（`model.onnx`）**不进 git**（`.gitignore`
已忽略），需按下方说明放置；缺失时评测脚本会按官方 Hub 机制自动下载到
`~/.wespeaker/`（与本目录二选一，本目录优先）。

## 目录结构

```
third_party_scripts/wespeaker/
├── README.md            # 本文件
├── en/
│   └── model.onnx       # voxceleb_resnet34_LM.onnx（英文，~25MB）
└── chs/
    └── model.onnx       # cnceleb_resnet34_LM.onnx（中文，~25MB）
```

## 下载方式

与官方 `wespeakerruntime.hub.Hub` 同源（腾讯 COS）：

```bash
mkdir -p third_party_scripts/wespeaker/{en,chs}
wget -O third_party_scripts/wespeaker/en/model.onnx \
  https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx
wget -O third_party_scripts/wespeaker/chs/model.onnx \
  https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/cnceleb/cnceleb_resnet34_LM.onnx
```

或直接 `python -c "from wespeakerruntime.hub import Hub; Hub.get_model_by_lang('en'); Hub.get_model_by_lang('chs')"`
后从 `~/.wespeaker/{en,chs}/model.onnx` 拷贝过来。

## 使用

`evaluate/eval_pipeline_tse.py` 的 spksim 默认后端（`--spksim_backend wespeakerruntime`），
语种按 meta `language` 列 / `--dataset_lang_overrides` / `--wespeaker_lang` 选择
en 或 chs 模型，输出 enroll↔tse 与 enroll↔mixture 两组余弦均值。
