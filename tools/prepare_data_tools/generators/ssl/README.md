# SSL 本地数据清单

`prepare_manifests.py` 将已有 LibriSpeech 语音文件和 HQ 噪声源清单整理成 `dynamic_ssl` 可读的四列 SCP。脚本只生成索引和 provenance，不复制或改写音频。

默认输入为本机 `data/shared/LibriSpeech/` 下的 `train-clean-100` 与 `dev-clean`，噪声输入为 `data/shared/noise/filter/v2/{train_noise_sources,val_noise_sources}.scp`。默认输出位于 `data/ssl/speech/` 和 `data/ssl/noise/`。噪声源按固定种子分别抽取最多 50,000 / 20,000 条；全量扫描输入 SCP 的字段、UID 和音频路径以拒绝格式错误或重复记录，入选音频会验证文件存在、采样率和帧数。设置对应的 `--noise-*-limit 0` 可输出全量噪声。需要完整 LibriSpeech 三 split 时传入 `--test-split test-clean`，脚本会额外生成 `data/ssl/speech/test.scp`；默认不生成 test 清单，避免改变现有训练配置。

```bash
python -m tools.prepare_data_tools.generators.ssl.prepare_manifests
```

若本机只有 Libri2Mix source JSONL，可显式使用备用入口：

```bash
python -m tools.prepare_data_tools.generators.ssl.prepare_manifests \
  --speech-source libri2mix-jsonl \
  --speech-train-jsonl data/tse/spk/wesep/datasets/Libri2Mix/dynamic_tse/train-100/speech.jsonl \
  --speech-validation-jsonl data/tse/spk/wesep/datasets/Libri2Mix/dynamic_tse/dev/speech.jsonl
```

覆盖已有清单必须传 `--overwrite`。所有输出作为一个文件集合事务发布到 `data/ssl/provenance.json` 与四个 split SCP；默认路径相对于各自 SCP 所在目录，便于同一仓库数据树迁移。需要跨目录使用时可传 `--absolute-paths`。

当前本地训练噪声清单来自 HQ noise source pool，脚本不会把它标记成 Noise92，也不会声称这些清单构成官方 FN-SSL 数据 recipe 或真实评测集。官方 Noise92 可用 `download_data.py` 通过 HTTPS 下载 `.mat`、转换成 16 kHz mono WAV，并生成 `noise92.scp`；本机已保留全部 15 个资源，清单可审计但不能单独代表完整官方训练集。WebRTC VAD、片段抽样和 overlap policy 由 `tasks/ssl/data/recipe.py` 在 Dataset 内按登记 profile 执行。

```bash
python -m tools.prepare_data_tools.generators.ssl.download_data \
  --download-noise92 babble white leopard machinegun --convert-wav \
  --output-dir data/ssl/noise92
```

## LOCATA 与 RealMAN

LOCATA 官方 Zenodo 归档约 6.2 GB（dev）和 13.0 GB（eval）。脚本会校验官方
MD5 与本地 SHA256，并将归档解压为 `data/ssl/LOCATA/{dev,eval}`：

```bash
python -m tools.prepare_data_tools.generators.ssl.download_real_datasets \
  --locata all --output-dir data/ssl
```

RealMAN 官方 Hugging Face 仓库约 719 GB；用于真实评估时默认下载约 67 GB 的
`val`、`test`、`dataset_info` 和 `transcriptions.trn`，文件大小与 revision 会写入
`data/ssl/RealMAN/download_manifest.json`。完整训练资源需要显式选择：

```bash
conda run --no-capture-output -n setrain python -m \
  tools.prepare_data_tools.generators.ssl.download_real_datasets \
  --realman eval --output-dir data/ssl
conda run --no-capture-output -n setrain python -m \
  tools.prepare_data_tools.generators.ssl.download_real_datasets \
  --realman full --output-dir data/ssl
```

LOCATA 使用 ODC-By/ODbL 条款，RealMAN 按官方仓库声明使用 CC BY 4.0；数据本体
被 `.gitignore` 隔离，不进入 Git。
