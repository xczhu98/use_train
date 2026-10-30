# 验证 CLI 索引

`tools/verify/` 存放**不依赖 pytest 的独立验证、对比和质量门禁 CLI**。这些脚本用于把训练/仿真/评估里“看不见”的东西变成可检查的证据：导出一段真实波形、跑一次真实模型、统计一次延迟误差，然后给出通过或不通过。

## 与 pytest 的分工

- **pytest** 验证入口的参数、返回码和发布契约，用小型 fixture，不加载真实数据。
- **本目录 CLI** 验证真实资产：真实音频、真实 checkpoint、真实 GPU 或第三方模型。

因此，pytest 里不应复制这些 CLI 的实现；新增一次性比较或验证脚本时放到 `tools/verify/`（而不是 `tests/`）。

## 使用前须知

- **先看 `--help`**：每个脚本的必填参数不同，本节表格只给用途，不给参数清单。
- **不隐式兜底**：参数缺失、资产校验失败或依赖不可用时脚本直接报错退出，不会静默换成 CPU、伪分或随机权重。
- **退出码**：`0` 表示全部通过；非零表示存在失败项或参数/资产错误。报告类脚本会先写出报告再返回非零，方便定位问题。
- **`_audio_verify_utils.py`** 是这些脚本之间的内部共享模块（RMS 等小工具），不承诺对外 API 稳定。

## 按任务索引

### AEC（回声消除）——`tools/verify/aec/`

| 脚本 | 作用 |
|------|------|
| `aec/compare_pfdkf_erle.py` | NumPy 版与 PyTorch 版 PFDKF（线性回声消除算法）的 ERLE 对比：在真实 farend/echo pair 上跑，输出 ERLE 分布并给出两者一致性结论 |
| `aec/verify_linear_aec_backends.py` | 验证 Speex / PFDKF 两种线性 AEC backend 的原理与精度；任一 backend 失败即非零退出（其它 backend 成功也不会掩盖），失败的 backend 不写 WAV |
| `aec/verify_aec_simulation_audio.py` | 按 `force_branch` 场景导出 AEC 动态仿真波形（mic / nearend / farend 及 Layer D 快照），并写出 A–F 分层实际取值 |
| `aec/deep_check_aec_simulation.py` | AEC 仿真链路深度检查：energy / SNR / SER / 一致性 / 峰值异常；报告写出后 `anomalies` 非空即返回非零 |
| `aec/verify_aec_echo_clipping.py` | 回声截幅统计：各路径触发率、截幅前后 echo peak/RMS、FST/DT/NST 状态标签占比（开启与关闭对照） |
| `aec/diagnose_aec_stereo_data.py` | 扫描 AEC stereo WAV，找出全零或能量极低的通道、被写成 stereo 的单声道文件、双通道静音 |
| `gcc_quality_gate.py` | GCC（广义互相关）延迟估计的质量门报告：delay MAE / false-lock / low-confidence 比例。默认 `--synthetic` 零数据依赖；给 `--pairs-json` 则改跑真实 manifest |
| `streaming_gcc_benchmark.py` | 可复现的流式 GCC-PHAT 配置 benchmark；主排名只使用带 delay oracle 的合成场景，真实 AEC 音频只做 finite / 抖动 / 激活稳定性 veto |

### SE（语音增强）

| 脚本 | 作用 |
|------|------|
| `verify_se_simulation_audio.py` | 按 `force_branch` 场景导出 SE 动态仿真的 `clean` / `noisy` 与可选中间 stem，并写出 A–F 分层实际取值 |
| `verify_unipase_official_weights.py` | 端到端对比本框架 UniPASE 与官方 demo 输出（max_abs_diff / SNR / SI-SNR），确认移植没有改变数值行为 |
| `verify_fast_ulcnet_state_drift.py` | 用合成输入分离验证 Fast-ULCNet 整段前向与 temporal hidden 漂移；默认参数仅供 CPU smoke，不评估语音质量 |
| `infer_official_ckpts.py` | 加载官方开源权重（GTCRN / UL-UNAS / PASE / StuPASE / UniPASE / GAP-URGENet）并对测试音频推理，输入输出路径由参数注入 |

### TSE（目标说话人提取）与 SS（语音分离）

| 脚本 | 作用 |
|------|------|
| `verify_tse_official_ckpts.py` | 对 REAL-T 官方 checkpoint 走完整 `TSEExtractor` 端到端验证：安全加载、完整 state 预校验、确定性前向、输出有限且等长 |
| `export_tse_torchscript.py` | 导出并验证与 TSE ONNX 相同 flat-cache ABI 的 TorchScript artifact |
| `plot_tse_timing_timeline.py` | 按官方 timing 语义绘制 TSE 的 GT / pred 时间线，用于人工核对时序指标 |
| `verify_ss_echoset.py` | 用 SS checkpoint 在 strict EchoSet manifest 上计算 SI-SDR / SI-SDRi |
| `verify_nbss_paper_alignment.py` | 只读复核 SpatialNet / OnlineSpatialNet 的论文口径 Params 与 FLOPs |

### 数据、清单与 MOS

| 脚本 | 作用 |
|------|------|
| `verify_offline_dataset.py` | 只读验证 `GenerationManifest v2` 与生产 `OfflineDataset` 能否完整重载；不传 `--probe` 时执行 full scan 并报告发布资格，`--probe N` 只用于定位问题，永不具发布资格 |
| `audit_noise_sampling.py` | 只读审计动态数据集的噪声 sampling 分布；不调用 `__getitem__`、不解码波形 |
| `benchmark_mos_batching.py` | 用真实本地 SCOREQ / UTMOS 资产比较“逐条打分”与“有界混长 batch”的一致性，并拒绝 fallback 或非有限分数 |

### 发布与文档契约（非音频验证）

| 脚本 | 作用 |
|------|------|
| `simulation_audit_contract.py` | 统一 SE / AEC / TSE 仿真审计 envelope 的定义与校验入口；写出并只接受当前 `contract_version="1.4"`，旧版本一律拒绝读取 |
| `check_config_comments.py` | 只读检查 `configs/` 全部 YAML 是否满足 `configs/README.md`「YAML 注释规范」（分节线缩进、注释块紧贴其键、注释缩进等于键缩进），违规逐行打印并返回非零码；同一规则由 `tests/test_configs.py::test_yaml_comment_contract` 门禁 |
| `paper_publication_contract.py` | 审计论文 Markdown 的链接安全与本地证据缓存存在性（图片与本地 PDF 允许直链、不要求 tracked；只拦绝对路径、`..` 逃逸、非 https 与缺失目标） |

## 常用调用形态

不同脚本的必填参数不同，下面只给典型形态；占位符 `<...>` 需要替换为真实路径。

```bash
# 只读验证一个已发布的 GenerationManifest v2（full scan，报告发布资格）
python tools/verify/verify_offline_dataset.py <generation-root> --root waveforms=/path/to/waveforms

# 审计动态数据集在 train split 上的噪声采样分布（只读、不解码音频）
python tools/verify/audit_noise_sampling.py <merged-config.yaml> --split train --draws 10000

# 检查 configs/ 全部 YAML 的注释规范（只读；也可传入单个文件或目录）
python tools/verify/check_config_comments.py

# GCC 延迟估计质量门（合成模式，无需真实数据，报告写到默认目录）
python tools/verify/gcc_quality_gate.py --synthetic

# 按 force_branch 导出 SE 仿真波形，便于逐场景人工核对
python tools/verify/verify_se_simulation_audio.py -C <config.yaml> -o <output-dir>

# SS checkpoint 在 EchoSet manifest 上的 SI-SDR / SI-SDRi
python tools/verify/verify_ss_echoset.py -C <config.yaml> --checkpoint <ckpt.tar> \
    --manifest <manifest.jsonl> --output <output-dir>
```

产物一般是“报告文件 + 导出波形/JSON”，脚本会在 stdout 打印路径或结论；需要真实 GPU、checkpoint 或第三方模型资产的脚本在资产缺失时会直接报错，不会降级。
