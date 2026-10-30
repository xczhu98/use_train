# SSL 真实录音验证

`real_eval.py` 是 SSL 模型的离线真实录音验证入口。它不注册新的
`data.mode`，也不改训练或根文件推理路径。脚本支持两种输入：

- canonical JSONL：每行必须给出 `audio`、`geometry.positions_m`、
  `labels.times_s`、`labels.azimuth_rad` 和 `labels.vad`；方位单位为弧度。
  例如：`{"key":"rec-01","audio":"wav/rec-01.wav","sample_rate":16000,
  "channel_indices":[2,5],"geometry":{"positions_m":[[0,0,0],[0.08,0,0]],
  "reference_channel":0},"labels":{"times_s":[0,1],"azimuth_rad":[0.2,0.2],"vad":[1,1]}}`。
- upstream FN-SSL 的 LOCATA 目录：根目录下可以有 `eval/`、`dev/`，每个
  `taskN/recording/array/` 目录需要 `audio_array_*.wav`、
  `position_array_*.txt`、`position_source_*.txt`、`required_time.txt`，VAD
  文件可选。当前只登记了 `dicit` 与 `dualch` 的本地几何；其它阵列请先
  转成 JSONL，避免猜测阵列坐标。

示例：

```bash
PYTHONPATH=. python tools/verify/ssl/real_eval.py \
  --locata-root data/ssl/LOCATA --task 1 --array dicit --mic-indices 0,1 \
  --config configs/ssl/fnssl_dynamic.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --output /tmp/ssl-locata-task1.json
```

默认方位域是水平半平面 `[-90°, 90°]`。双麦克风只能在该合同内进行可辨识
的方位比较；域外标注会计入 `excluded_out_of_domain_frames`，不会被镜像折叠。
使用 `--azimuth-domain min_deg,max_deg` 可显式改变域。VAD 门限、输出时间
对齐、STFT、模型 `ModelSpec` 和 DP-IPD mapper 均写入结果 JSON；声速从配置的
`data.train/validation.simulation_cfg.gpurir_sound_speed_mps` 读取，缺失时直接失败，
不使用隐式默认值。

推理阶段只读取录音观测和阵列 geometry。LOCATA 的 source position、VAD 及
JSONL 的 azimuth 仅用于验证指标，`source_distances` 不会传入模型或 mapper。
checkpoint 通过严格的 `evaluate.checkpoint_loading`/`safe_torch_load` 路径
加载；缺 checkpoint、几何或真实录音时直接失败。结果文件应放在 `/tmp` 或
被 `.gitignore` 忽略的 `data/ssl` 下，不能提交数据和结果文件。

LOCATA/RealMAN 数据整理由
`tools/prepare_data_tools/generators/ssl/download_real_datasets.py` 负责。LOCATA
官方 Zenodo 的 dev/eval 归档会校验官方 MD5 和本地 SHA256 后解压；RealMAN 默认
下载 `val`、`test`、`dataset_info` 和转写，完整训练集需显式传 `--realman full`。
RealMAN 的 RAR 内容仍需按官方目录解压并转换为 canonical JSONL 后才能交给本脚本；
下载完成本身不等于指标已经运行。

输出指标包含有效 VAD 帧数、预测 coverage、circular azimuth MAE/median/p90
和给定误差阈值下的 accuracy。没有有效交集时指标为 `null`，不会用零值代替。
脚本不宣称 LOCATA 官方指标复现；需要先记录真实资产、checkpoint、配置和
半平面合同，再解释结果。
