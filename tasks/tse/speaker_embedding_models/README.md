# tasks/tse/speaker_embedding_models — 声纹 embedding 预训练模型

TSE 需要知道「要提取哪个人」，判断依据是目标说话人的注册音（enrollment）。这里存放的就是把注册音变成说话人向量（embedding）的**预训练 speaker encoder 权重**：ECAPA-TDNN 和 WeSpeaker ResNet34 都是说话人识别方向的公开模型，本仓库把它们 vendor 进来，只当作特征提取前端使用。权重文件（`*.pt`）**不进 git**（`.gitignore` 已忽略），需要按下方说明自行下载放置。

## 目录结构与存放约定

```
tasks/tse/speaker_embedding_models/
├── README.md                              # 本文件
├── ecapa_tdnn_glob_c512/                  # ECAPA-TDNN，192d
│   └── avg_model.pt
└── resnet34/                              # WeSpeaker ResNet34-TSTP，256d
    └── avg_model.pt
```

## 模型说明

| 子目录 | 模型 | 输入 | 输出 | 来源 |
|--------|------|------|------|------|
| `ecapa_tdnn_glob_c512/` | ECAPA_TDNN_GLOB_c512（C=512，ASTP 全局上下文池化） | 80 维 kaldi-fbank（25ms/10ms 帧，mean-CMN） | 192 维句级 embedding / 512 维帧级特征 | wespeaker VoxCeleb 预训练 |
| `resnet34/` | ResNet34（TSTP 池化） | 80 维 kaldi-fbank（25ms/10ms 帧，mean-CMN） | 256 维句级 embedding / 2560 维帧级特征 | WeSpeaker VoxCeleb 预训练 |

输入口径说明：fbank 是滤波器组频谱特征，本仓库用与 kaldi 一致的算法批量提取；mean-CMN 指减掉整句均值。句级 embedding 是整段注册音的一个向量，帧级特征是逐帧向量，用于需要帧对齐的 cue。

- **官方下载地址**（wespeaker，VoxCeleb v2 训练）：
  - Hugging Face（可用，已验证）：<https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512>
    取其中的 `avg_model.pt`（md5 `7c29240440b2f45f6a67e85c7c8cf022`）放入
    `ecapa_tdnn_glob_c512/` 即可。
  - ModelScope：wespeaker `docs/pretrained.md` 模型表中的 VoxCeleb ECAPA512 条目。
  - 旧直链 `https://wenet.org.cn/wespeaker/voxceleb/voxceleb_ECAPA512.tar.gz`
    已随官网改版失效，仅作历史参考。
  - 仓库：https://github.com/wenet-e2e/wespeaker

  `resnet34/` 的权重同样需要自行下载；本文件目前只给出 ECAPA 的下载地址，详见 WeSpeaker 官方模型表。

## 加载方式

- 训练（joint_training 模式）：在 YAML 里注入权重路径，例如
  ```yaml
  model:
    type: TSE_BSRNN_SPK
    args:
      speaker_model:
        speaker_encoder:
          pretrained: tasks/tse/speaker_embedding_models/ecapa_tdnn_glob_c512/avg_model.pt
  ```
  由 `tasks/tse/models/speaker/encoder.py::SpeakerEncoder` 加载：只适配 `module.` 前缀，并过滤掉明确允许的 loss/projection namespace，随后对键、shape、dtype 与 value 做 fail-closed 校验（任一不匹配就报错，不静默跳过）；`num_batches_tracked` 只有满足安全整数转换合同才允许转换。
- REAL-T 官方 TSE 主 checkpoint 已内含 ECAPA，因此 joint 推理不会读取本目录的独立权重。当前本机 ECAPA artifact 的 29 个 `num_batches_tracked` 是带小数的 `float32` 平均值（如 `619921.625`），独立严格加载会拒绝；训练 strict baseline 需要换成官方原始、键与 dtype 完整兼容的 WeSpeaker 权重，禁止把这些小数截断来绕过校验。
- 评测（spk-sim）：可以复用同一路径，但仍受上述严格 checkpoint 合同约束。

## ResNet34 配置

`SpeakerEncoder` 已注册 `ResNet34`，实现在
`tasks/tse/models/speaker/resnet.py` 内 vendored，不依赖 WeSpeaker Python 包：

```yaml
speaker_encoder:
  model: ResNet34
  pretrained: tasks/tse/speaker_embedding_models/resnet34/avg_model.pt
  freeze: true
  spk_args:
    feat_dim: 80
    embed_dim: 256
    pooling_func: TSTP
```

`SpeakerEncoder` 内部默认 `two_emb_layer=false`；当前 `TSE_BSRNN_SPK` 的严格配置白名单不暴露这个实验开关。用于 `context` cue 时还需显式设置 `features.context.embed_dim: 2560`。后续 ResNet18/50/101/152 与 CAMPPlus 应继续通过同一 `SpeakerEncoder` 注册表接入。

严格 WeSep baseline 可以直接从官方 `spk_emb_100/avg_model.pt` 里挑出内含的 ECAPA 子模块，配置 `pretrained_prefix: spk_ft.encoder.spk_model.`（`pretrained_prefix` 用来指定从完整 checkpoint 的哪一段键取值）。该路径仍要求 prefix 下完整命中全部 speaker keys；prefix 之外的 separator 权重不会进入 `SpeakerEncoder`。
