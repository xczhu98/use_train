# 预训练权重获取与放置

本文件是仓库内**未纳入 Git** 的预训练权重统一索引。权重本体被 `.gitignore` 忽略（`*.pt`、`*.pth`、`*.ckpt`、`*.tar`、`*.onnx` 等），仓库只提交本文件和各资产目录下的 README。换机器或清理工作区后按本文准备；路径均相对仓库根目录。

下载前先确认目标目录已存在。下面所有官方入口都指向上游发布页，链接失效时以对应目录 README 与上游仓库为准。

## TSE 官方 checkpoint

REAL-TSE Challenge 官方发布的 `TSE_BSRNN_SPK` 权重，共四个变体：

| 变体 | 目标路径 | 约大小 |
|---|---|---|
| `spk_emb_100` | `tasks/tse/pretrained_models/spk_emb_100/` | 270 MB |
| `spk_emb_causal_100` | `tasks/tse/pretrained_models/spk_emb_causal_100/` | 241 MB |
| `tfmap_context_100` | `tasks/tse/pretrained_models/tfmap_context_100/` | 295 MB |
| `tfmap_context_causal_100` | `tasks/tse/pretrained_models/tfmap_context_causal_100/` | 266 MB |

下载入口（wesep-real-tse README 的 “Download pretrained model”）：

```bash
pip install gdown
gdown "https://drive.google.com/uc?export=download&id=1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC" -O /tmp/realtse_ckpts.zip
unzip /tmp/realtse_ckpts.zip -d tasks/tse/pretrained_models/
```

解压后每个目录应包含 `avg_model.pt` 与官方 `config.yaml`。官方 `config.yaml` 只作对照，不能直接当 `model_dir`；推理配置使用 `configs/tse/inference/` 下的已验证 YAML。细节见 [`tasks/tse/pretrained_models/README.md`](../../../tasks/tse/pretrained_models/README.md)。

验证：

```bash
python tools/verify/verify_tse_official_ckpts.py \
    --ckpt_root tasks/tse/pretrained_models \
    --realt_root data/tse/REAL-TSE/REAL-T-dev \
    --device cpu
```

## TSE speaker embedding

| 文件 | 目标路径 | 来源 |
|---|---|---|
| ECAPA-TDNN `avg_model.pt` | `tasks/tse/speaker_embedding_models/ecapa_tdnn_glob_c512/avg_model.pt` | [Wespeaker/wespeaker-voxceleb-ecapa-tdnn512](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512) |
| ResNet34 `avg_model.pt` | `tasks/tse/speaker_embedding_models/resnet34/avg_model.pt` | [WeSpeaker pretrained models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md) |

ECAPA 下载：

```bash
huggingface-cli download Wespeaker/wespeaker-voxceleb-ecapa-tdnn512 avg_model.pt \
    --local-dir tasks/tse/speaker_embedding_models/ecapa_tdnn_glob_c512
```

ECAPA 的 `avg_model.pt` md5 为 `7c29240440b2f45f6a67e85c7c8cf022`。ResNet34 权重需从 WeSpeaker 官方模型表下载后放入目标目录；当前仓库只保留目录与 README。加载边界和 `pretrained_prefix` 用法见 [`tasks/tse/speaker_embedding_models/README.md`](../../../tasks/tse/speaker_embedding_models/README.md)。

## PASE 系列

四个模型的权重都放在各目录的 `open_model_checkpoint/` 下；`config.json` 是仓库本地 `model_type` 标记，不要删除或替换成上游文件。

| 模型 | 目标目录 | 需要文件 | 官方下载 |
|---|---|---|---|
| PASE | `tasks/se/models/pase_family/pase/open_model_checkpoint/` | `DeWavLM.pt`、`Vocoder_Dual.pt`、`Vocoder_L24.pt` | [huggingface.co/cisco-ai/pase](https://huggingface.co/cisco-ai/pase) |
| StuPASE | `tasks/se/models/pase_family/stupase/open_model_checkpoint/` | `DeWavLM-R.pt`、`CFM.pt`、`Vocoder_Mel-16k.pt` | [huggingface.co/cisco-ai/stupase](https://huggingface.co/cisco-ai/stupase) |
| UniPASE | `tasks/se/models/pase_family/unipase/open_model_checkpoint/` | `DeWavLM-Omni.pt`、`Adapter.pt`、`Vocoder_DWO-L1.pt`、`Vocoder_WavLM-L24.pt`、`PostNet.pt` | [huggingface.co/Xiaobin-Rong/unipase](https://huggingface.co/Xiaobin-Rong/unipase) |
| GAP-URGENet | `tasks/se/models/pase_family/gap_urgenet/open_model_checkpoint/` | `DeWavLM-Omni.pt`、`Adapter.pt`、`Vocoder.pt`、`Predictor.pt`、`PostNet.pt` | [huggingface.co/Xiaobin-Rong/gap-urgenet](https://huggingface.co/Xiaobin-Rong/gap-urgenet) |

按整仓库下载：

```bash
huggingface-cli download cisco-ai/pase --local-dir tasks/se/models/pase_family/pase/open_model_checkpoint
huggingface-cli download cisco-ai/stupase --local-dir tasks/se/models/pase_family/stupase/open_model_checkpoint
huggingface-cli download Xiaobin-Rong/unipase --local-dir tasks/se/models/pase_family/unipase/open_model_checkpoint
huggingface-cli download Xiaobin-Rong/gap-urgenet --local-dir tasks/se/models/pase_family/gap_urgenet/open_model_checkpoint
```

也可以直接取单文件，URL 形如：

```text
https://huggingface.co/<owner>/<repo>/resolve/main/<filename>
```

例如 `https://huggingface.co/cisco-ai/pase/resolve/main/DeWavLM.pt`。如果 Hub 页面把权重放在子目录，按页面下载后移动到上表的目标文件名。各目录 README 还记录了官方推理语义和权重角色，见 `tasks/se/models/pase_family/*/open_model_checkpoint/README.md`。

验证：

```bash
PYTHONPATH=. python tools/verify/infer_official_ckpts.py \
    --input /path/to/noisy.wav \
    --output_dir /tmp/pretrained_check \
    --models PASE StuPASE UniPASE GAP-URGENet
```

## GTCRN / UL-UNAS

| 模型 | 目标文件 | 官方入口 |
|---|---|---|
| GTCRN | `tasks/se/models/discriminative/gtcrn/open_model_checkpoint/model_trained_on_dns3.tar`、`model_trained_on_vctk.tar` | [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn) |
| UL-UNAS | `tasks/se/models/discriminative/ulunas/open_model_checkpoint/model_trained_on_dns3.tar` | [Xiaobin-Rong/ul-unas](https://github.com/Xiaobin-Rong/ul-unas) |

上游权重通常位于其仓库的 `checkpoints/` 下，按目录 README 的约定复制为上述文件名。`config.json` 是仓库本地标记，不是上游文件。

验证：

```bash
PYTHONPATH=. python tools/verify/infer_official_ckpts.py \
    --input /path/to/noisy.wav \
    --output_dir /tmp/pretrained_check \
    --models GTCRN UL-UNAS
```

## ONNX 验证权重

`onnx/pytorch_model/` 存放 ONNX 导出/三路验证用的 PyTorch 权重，不属于训练 checkpoint：

| 文件 | 用途 | 获取方式 |
|---|---|---|
| `deepvqe.pth` | DeepVQE 导出/验证 | 从 `exp/aec/deepvqe_*` 的 checkpoint 复制或按 `onnx/README.md` 重新导出 |
| `dpgtcrn.pth` | DP-GTCRN 导出/验证 | 从 `exp/se/dpgtcrn_*` 的 checkpoint 复制或重新导出 |
| `cagcrn_joint.tar` | CAGCRN 导出/验证 | 从对应训练实验 checkpoint 复制 |
| `gtcrn_ofical.tar` | GTCRN 官方权重 | 从 [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn) 获取 |
| `zipenhancer_pytorch.bin` | ZipEnhancer 导出/验证 | 从 [ModelScope iic/speech_zipenhancer_ans_multiloss_16k_base](https://www.modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base) 获取 |

`onnx/onnx_model/*.onnx` 是导出产物，不是预训练输入；可以用 `onnx/export.py` 重新生成。正式发布必须经过对应 `onnx/verify/verify_3way_*.py` gate，具体命令和阈值见 [`onnx/README.md`](../../../onnx/README.md)。

## 说明

- 权重文件体积大且受上游许可证约束，禁止直接提交到主仓库。
- 本文件只维护统一入口；每个目录的 README 仍是该资产加载语义、目标路径和验证命令的详细说明。
- 新增未入库权重时，在本文件补一行来源链接，并在对应目录 README 记录目标文件名和验证方式。
