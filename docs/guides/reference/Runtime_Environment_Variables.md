# 运行时环境变量参考

本页集中登记 use_train 代码定义的全部环境变量。这些变量的作用域由代码定义，均不是通用 YAML 设置；不要假设它们全局作用于训练、评估或所有第三方脚本。

| 作用域 | 环境变量 | 默认值 / 行为 |
|-------|----------|---------------|
| 训练 entry | `CUDA_VISIBLE_DEVICES` | 选择启动进程可见的 GPU 设备。 |
| 训练 entry | `TORCH_CPP_LOG_LEVEL` | `core.trainer.entry` 仅在变量未设置时写入 `ERROR`；用户预设值优先。 |
| 训练 entry | `MASTER_PORT` | 未设置时统一 entry 自动选择空闲端口；显式值优先。 |
| DDP 启动 | `RANK` / `LOCAL_RANK` / `WORLD_SIZE` | 由 `torchrun` 注入；`LOCAL_RANK` 选择本地 CUDA 设备，且不是 CLI 参数。 |
| SE/AEC 评估流水线 | `USE_TRAIN_TEST_DATA_ROOT` | 用于默认测试输入、标准化 SCP/test artifact、`exp/eval` 评估输出以及 AEC badcase 输入/输出；各 pipeline 的 CLI 可逐项覆盖输入路径。 |
| 音频缓存 | `USE_TRAIN_AUDIO_CACHE_MAXSIZE` | 每个 worker 进程独立的解码音频缓存条目上限（默认 `2048`）；不跨 rank/process 共享。 |
| 音频缓存 | `USE_TRAIN_AUDIO_CACHE_BYTES` | 每个 worker 进程独立的解码 float32 音频缓存字节预算（默认 `536870912`，512 MiB；`0` 禁用缓存）；不跨 rank/process 共享。 |
| 音频 I/O | `USE_TRAIN_AUDIO_WINDOW_READ_MIN_SECONDS` | 长文件窗口读取阈值（秒，默认 `0`，关闭）。 |
| AEC 音频缓存 | `USE_TRAIN_AEC_STEREO_CACHE_BYTES` | 每个 worker 进程独立的 farend/echo stereo 缓存字节预算（默认 `268435456`，256 MiB）；每个 rank/process 各自持有。 |
| DDP 诊断 | `USE_TRAIN_NCCL_DEBUG` | 任意非空值启用 collective 前流程日志；建议设为 `1`。 |
| 第三方资源定位 | `USE_TRAIN_THIRD_PARTY_DIR` | 覆盖 MOS，以及 Audio/Noise Filter 的 VAD、MOS 和 VAD bootstrap 使用的第三方根目录。 |
| Noise Filter NeMo 下载辅助脚本 | `NEMO_VAD_ONNX_URL` | 仅供 `tools/noise_filter_pipeline/scripts/download_nemo_vad_onnx.py` 指定下载 URL；未设置时不尝试 URL 下载。 |
| Noise Filter FireRedVAD bootstrap | `XDG_CACHE_HOME` | 指定 `use_train/FireRedVAD` 的缓存根目录；默认 `~/.cache`。 |
| Audio/Noise Filter 入口 | `TF_NUM_INTEROP_THREADS` / `TF_NUM_INTRAOP_THREADS` | 两条 pipeline 都会在导入可能加载 TensorFlow 的模块前将未设置值设为 `1`；显式值优先。 |
| Audio Filter Step1 worker | `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` | 每个 worker 都会在创建 VAD runtime 前无条件将三个变量设为 `1`，避免继承的线程设置导致 CPU 超订。 |
| 评估 subprocess | `USE_TRAIN_EVAL_SUBPROCESS_TIMEOUT_S` | 评估 subprocess 统一超时（秒，默认 `43200`，12 小时）；超时后杀子进程并抛 `TimeoutExpired`。 |
| 评估流水线 | `USE_TRAIN_EVAL_CONFIG` / `USE_TRAIN_EVAL_CHECKPOINT` | 成对设置时覆盖流水线使用的 config 与 checkpoint 路径；只设置其中一个会直接报错。 |
| 音频 I/O | `USE_TRAIN_SOX_TIMEOUT_SEC` | SoX 子进程超时（秒，默认 `600`）。 |
| 音频 I/O | `USE_TRAIN_FFMPEG_TIMEOUT_SEC` | ffmpeg 进程超时（默认 `7200`，2 小时）。 |
| 动态仿真 Layer E codec | `USE_TRAIN_CODEC_TIMEOUT_SEC` | DataLoader worker 内隔离 codec 的 ffmpeg 子进程超时（秒，默认 `60`）；超时只跳过该次 codec 增强并保留原波形。 |
| STFT | `USE_TRAIN_STFT_VALIDATE_FINITE` | 设为 `1`/`true` 时对 STFT 输入执行逐元素有限性检查（默认关闭）。 |
| 清单缓存 | `USE_TRAIN_MANIFEST_CACHE_DISABLE` | 设为 `1`/`true` 时禁用 source manifest 缓存。 |
| Noise Filter Qwen | `USE_TRAIN_QWEN_GENERATE_TIMEOUT_SEC` | 单事件生成超时（秒，默认 `0`，不限时）。 |
| SCP 准备 | `PREPARE_SCP_ALLOW_UNSAFE_PICKLE=1` | 仅对可信数据启用完整 pickle 回退。 |

## DDP 下的内存预算

DDP 下，`data.dataloader.num_workers`、`prefetch_factor` 和 `persistent_workers` 都按 rank/process 创建，decoded-audio 与 AEC stereo cache 按 Python 进程独立持有。因此 host 内存预算约按 `ranks × workers` 线性累加。使用 1–4 卡或主机内存不足时，先降低 `num_workers`、`prefetch_factor`，缩小相关 cache 的预算与条目上限；必要时关闭 `persistent_workers`，待内存和吞吐稳定后再逐步调高。
