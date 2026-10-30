# tasks/tse/pretrained_models — TSE 模型预训练 checkpoint（比赛官方）

本目录存放 REAL-TSE-Challenge 官方发布的 `TSE_BSRNN_SPK` 预训练 checkpoint（在 Libri2Mix-100 上训练）。checkpoint 就是「模型权重 + 它的配置」这一份可加载文件；权重文件（`*.pt`）**不进 git**（`.gitignore` 已忽略），需要按下方说明自行下载放置。

## 目录结构与存放约定

```
tasks/tse/pretrained_models/
├── README.md                          # 本文件
├── spk_emb_100/                       # spkemb cue，非因果（离线）
│   ├── avg_model.pt
│   └── config.yaml
├── spk_emb_causal_100/                # spkemb cue，因果（流式）
├── tfmap_context_100/                 # tfmap+context cue，非因果
└── tfmap_context_causal_100/          # tfmap+context cue，因果
```

每个目录含 `avg_model.pt`（模型权重）与 `config.yaml`（wesep 格式的模型配置，仅作对照参考）。四个变体可以粗分两类：

- **非因果（离线）**：前向时能看到整段音频，但不能边收边出，不能用于实时场景。
- **因果（流式）**：只用当前帧和历史帧，可以边收边出；本仓库的 waveform chunk 会话、ONNX 与 TorchScript 路径都基于它。

cue 是什么、四个变体之间的机制差异以 `configs/tse/dynamic/tse_bsrnn_spk.yaml` 顶部的注释切换块为准。

## 下载方式

官方发布于 Google Drive（wesep-real-tse README 的 "Download pretrained model" 一节）：

```bash
pip install gdown
gdown "https://drive.google.com/uc?export=download&id=1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC" -O /tmp/realtse_ckpts.zip
unzip /tmp/realtse_ckpts.zip -d tasks/tse/pretrained_models/
```

解压后 `tasks/tse/pretrained_models/<变体名>/avg_model.pt` 存在即算放好；权重不进版本控制，换机器要重新下载。

## 加载边界（对照实验）

- 主 checkpoint 的 wesep wrapper（`{"models": [state_dict], ...}`）可被 `TSEExtractor` 识别；STFT/iSTFT 本身没有可训练权重，谱进谱出的改造不改变这些键名。四个存档分别与对应模型完整匹配 `972 / 948 / 1040 / 1016` 个键，且主存档已包含 ECAPA speaker encoder，因此 joint 推理构造阶段不会去读独立的 ECAPA 权重。
- 官方目录自带的 wesep 平铺 `config.yaml` 缺少 use_train 所需的顶层 `model.type` 与 `stft` 字段，不能直接当作 `model_dir`（`model_dir` 是推理入口读取配置与权重的目录）。必须使用下表已验证的配置：

  | checkpoint | use_train infer 配置 |
  |---|---|
  | `spk_emb_100` | `tse_bsrnn_spk_realt_baseline_infer.yaml` |
  | `spk_emb_causal_100` | `tse_bsrnn_spk_realt_baseline_causal_infer.yaml` |
  | `tfmap_context_100` | `tse_bsrnn_spk_realt_tfmap_context_infer.yaml` |
  | `tfmap_context_causal_100` | `tse_bsrnn_spk_realt_tfmap_context_causal_infer.yaml` |

  这些配置位于 `configs/tse/inference/`。

- 端到端验证命令会从 REAL-T DEV 取一段固定前缀的真实音频，对四个模型各做两次完整 `TSEExtractor` 前向，检查输出有限、两次逐值相同、长度一致、峰值归一到 0.9：

  ```bash
  python tools/verify/verify_tse_official_ckpts.py \
      --ckpt_root tasks/tse/pretrained_models \
      --realt_root data/tse/REAL-TSE/REAL-T-dev \
      --device cpu
  ```

  每个变体会打印一行 `[E2E] <变体名>: keys=... out=... finite=True deterministic=True peak=0.9 passed=True`，最后一行 `[RESULT] 四 checkpoint 端到端验证通过`；任一 checkpoint 缺失会打印 `[MISS]` 并以非零状态退出。
- 上述限制不改变现有 fail-closed 策略：安全反序列化、0 参数命中、state 不匹配及 ECAPA buffer 的 shape/dtype/value 校验都不得降级或静默绕过。
