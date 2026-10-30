# 🔧 tools/ — 数据准备与辅助工具集

<p align="center">
  <strong>把原始音频变成框架能读的清单、生成仿真数据、搬运旧权重、导出 ONNX、验证数据质量</strong><br>
  本文所有命令都在 <code>use_train</code> 仓库根目录执行。
</p>

---

## 这个目录解决什么问题

`tools/` 放的是**不参与训练/推理主流程**的辅助工具。主线（`train.py` / `inference.py`）只消费已经准备好的清单和 checkpoint；把这些原料造出来、检查它们对不对，就是本目录的工作。

按目标找入口：

| 我要做的事 | 去哪里 |
|---|---|
| 把音频目录变成 Dataset 能读的索引文件 | [`prepare_data_tools/scp/`](#-数据准备与清单生成)（最常用 `prepare_scp.py`） |
| 生成仿真数据：RIR、风噪、回声对、离线 clean/noisy | [`prepare_data_tools/generators/`](#-数据准备与清单生成) |
| 检查数据与仿真是否符合预期 | [`verify/`](verify/README.md)（完整 CLI 索引） |
| 把旧项目的 checkpoint 换成当前框架的权重 | [`model_tools/convert_checkpoint.py`](#-模型与权重工具) |
| 导出 / 发布 ONNX | [`onnx/`](../onnx/README.md)（本目录只给跳转说明） |
| 筛掉质量不达标的语音或噪声数据 | [`audio_filter_pipeline/`](audio_filter_pipeline/README.md)、[`noise_filter_pipeline/`](noise_filter_pipeline/README.md) |
| 跑一次性数据整理、报告或分析 CLI | [`ops/`](#-运维与分析-cli) |

> 💡 **推理 / 盲测**：使用项目根目录的 `inference.py`；本仓库**没有** `tools/inference/` 子目录。

### 30 秒术语表

- **SCP**：纯文本索引清单，每行一条记录。本项目统一使用严格四列 `uid fs length path`（`uid` 唯一标识、`fs` 采样率、`length` 帧数、`path` 音频路径）；训练 Dataset 只认这种形状。
- **manifest**：更宽泛的“清单”叫法，可能是 SCP、JSONL 或 JSON。扩展名不同通常代表合同不同，不能互相替换。
- **GenerationManifest v2**：离线数据集的发布清单，把“这批数据由谁生成、内容 SHA-256、如何重放”绑定在一起；旧 v1 集合必须重新生成。
- **checkpoint**：训练保存的权重文件（`.tar` / `.pt`）。
- **RIR**：房间冲激响应，描述房间的混响特性；仿真用它把干净语音“放进”一个房间里。
- **ONNX**：跨框架的模型文件格式，用于部署；正式发布必须经过模型专属验证门禁。
- **fail-closed**：发现异常就报错停下，而不是猜一个结果继续跑。本目录大量使用这一策略，所以“报错”通常说明校验正常工作。
- **SHA-256 / digest**：文件内容的指纹，内容一变指纹就变，用来判断旧产物能不能复用。

---

## 📑 本文内容

- [目录治理与迁移规则](ARCHITECTURE.md)
- [目录结构](#-目录结构)
- [数据准备与清单生成](#-数据准备与清单生成)
- [数据准备安全契约](#数据准备安全契约)
- [噪声数据集目录重排](#噪声数据集目录重排)
- [验证 CLI 资产归属](#验证-cli-资产归属)
- [模型与权重工具](#-模型与权重工具)
- [音频数据筛选管道](#audio-filter-pipelines)
- [运维与分析 CLI](#-运维与分析-cli)
- [常用命令](#-常用命令)

---

## 📁 目录结构

| 目录 | 用途 |
|------|------|
| **prepare_data_tools/** | 数据准备与离线生成（SCP、Voicebank、DNS3、AEC ref/echo JSON、清单验证） |
| **prepare_data_tools/scp/** | SCP 生成脚本（`prepare_voicebank.py` / `prepare_offline.py` / `prepare_scp.py` / `prepare_testset_scps.py` 等） |
| **prepare_data_tools/generators/** | 按资源类别分为 `aec/`、`se/`、`tse/`、`ss/`、`noise/`、`echo/` 的离线数据与清单生成 |
| **model_tools/** | 模型 / 权重工具（`convert_checkpoint.py`、`convert_taploss_official_checkpoint.py`、`audit_metrics.py` 等） |
| **noise_filter_pipeline/** | 噪声 / 音乐筛选管道（C1 Qwen 三分类 → C2 AudioTag 固定 top-5 → C3 speech-only veto） |
| **audio_filter_pipeline/** | 语音质量筛选管道（Step4 SCOREQ + URGENT-PK + DNSMOS 严格 3/3 合取） |
| **ops/** | 一次性运维与分析 CLI，按 `data/`、`reporting/`、`evaluation/`、`aec/` 分组 |
| **shared/** | 工具共享模块（`audio_io` / `file_scan` / `pickle_io` / `scp_io` / `scp_provenance`） |
| **verify/** | 仿真 / 数据验证 CLI（`verify_*_simulation_audio.py`、`verify_offline_dataset.py`、`diagnose_aec_stereo_data.py` 等，见 [verify/README.md](verify/README.md)） |
| **environment/** | 运行环境能力探测（canonical 模块：`tools.environment.environment_capabilities`） |
| **visualization/** | 论文方法图检查、栅格化与发布（canonical 模块：`tools.visualization.paper_diagram_rendering`） |

---

## 📋 数据准备与清单生成

| 脚本 | 功能 | 用途 |
|------|------|------|
| `prepare_data_tools/scp/prepare_scp.py` | 通用 SCP/JSONL 索引生成（4 列格式；默认 `val_ratio=0.05` 对全部资源做 train/val 随机划分，`--val_ratio 0` 关闭） | 通用 |
| `prepare_data_tools/scp/prepare_offline.py` | SE clean/noisy 离线数据一键准备（SCP + train/val 划分） | 通用 |
| `prepare_data_tools/scp/prepare_voicebank.py` | Voicebank-Demand 数据准备（递归扫描 + 重采样 + SCP） | VB 实验 |
| `prepare_data_tools/scp/prepare_testset_scps.py` | 一键为所有测试集生成标准 SCP | 统一评估 |
| `prepare_data_tools/scp/prepare_aec_offline.py` | AEC mic/farend/nearend 严格扫描；原 GenerationManifest 发布 CLI 已下线并 fail-closed | AEC |
| `prepare_data_tools/scp/prepare_aec_challenge_offline.py` | AEC Challenge 扫描逻辑；发布路径同样 fail-closed，待 AEC schema v2 reload 就绪 | AEC |
| `prepare_data_tools/generators/echo/build_real_farend_echo_pair_manifest.py` | 递归收集 `.wav` → echo manifest JSON（默认 `val_ratio=0.05` 划分 train/val 两个 JSON） | AEC |
| `prepare_data_tools/generators/echo/merge_farend_echo_pair_manifests.py` | 合并多个 echo manifest（去重 + shuffle） | AEC |
| `prepare_data_tools/generators/echo/generate_simulated_farend_echo_pairs.py` | 离线仿真回声对（stereo WAV + manifest） | AEC |
| `prepare_data_tools/generators/aec/prealign_stereo_echo_ref.py` | stereo AEC pair 的离线 GCC 预对齐物化（canonical module：`python -m tools.prepare_data_tools.generators.aec.prealign_stereo_echo_ref`；输出审计 JSONL，不是训练清单） | AEC |
| `prepare_data_tools/generators/aec/generate_simulated_testset.py` | AEC 7 场景仿真测试集生成（canonical module：`python -m tools.prepare_data_tools.generators.aec.generate_simulated_testset`） | AEC 评估 |
| `prepare_data_tools/generators/aec/generate_ood_stress_set.py` | AEC OOD stress set 生成（canonical module：`python -m tools.prepare_data_tools.generators.aec.generate_ood_stress_set`；9 场景，固定 seed 可回放，JAE-M03） | AEC 评估 |
| `prepare_data_tools/generators/se/generate_offline_dataset.py` | 将 SE dynamic simulation 物化为离线训练对（canonical module：`python -m tools.prepare_data_tools.generators.se.generate_offline_dataset`） | SE |
| `prepare_data_tools/generators/se/generate_dns3_offline_dataset.py` | DNS3 离线数据生成（Clean + RIR + Noise；三 split 同源采样，见脚本 docstring） | DNS3 |
| `prepare_data_tools/generators/se/generate_simulated_rirs.py` | RIR 仿真生成（canonical module：`python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs`；rirgen/fram/gpurir，4 列 `rirs.scp`） | 通用 |
| `prepare_data_tools/generators/se/generate_scene_rirs.py` | 多麦克风 scene-level offline RIR bundle（只负责 `C>1` 的 spatial scene；`C=1` 仍用 `generate_simulated_rirs`） | 通用 |
| `prepare_data_tools/generators/noise/generate_wind_noise.py` | 风噪仿真生成（canonical module：`python -m tools.prepare_data_tools.generators.noise.generate_wind_noise`；IWAENC22，4 列 `wind_noise_sources.scp`） | 通用 |
| `prepare_data_tools/generators/noise/build_stage_d_noise_manifests.py` | Noise Filter Stage D strict JSONL 发布清单生成 | Noise |
| `prepare_data_tools/generators/tse/generate_realtse_pseudo_targets.py` | REAL-TSE teacher 伪目标生成 | TSE |
| `prepare_data_tools/generators/tse/import_wesep_enrollment.py` | 官方 wesep speaker cue 资源 → `dynamic_tse` nested enrollment JSON（只加 `dataset_id` 一层，不改音频） | TSE |
| `prepare_data_tools/generators/tse/prepare_dynamic_tse_manifests.py` | 从显式单说话人目录清单生成 `dynamic_tse` 严格资源 | TSE |
| `prepare_data_tools/generators/ss/prepare_echoset.py` | 按 TIGER EchoSet 配对语义生成 strict separation JSONL | SS |
| `ops/data/reorganize_noise_datasets.py` | 按数据集名称重排并复制噪声数据，合并采样率并移除哈希分片目录 | 数据整理 |

### 仿真 / 数据验证脚本

完整索引与使用形态见 [`verify/README.md`](verify/README.md)；脚本本体位于 `tools/verify/`，AEC 专属脚本收在 `tools/verify/aec/`。

| 脚本 | 功能 | 用途 |
|------|------|------|
| `verify/verify_se_simulation_audio.py` | SE 仿真链路验证（按 `force_branch` 导出波形） | 开发调试 |
| `verify/aec/verify_aec_simulation_audio.py` | AEC 仿真验证（导出波形 + SNR/SER） | 开发调试 |
| `verify/aec/verify_aec_echo_clipping.py` | AEC 回声截幅统计验证 | 开发调试 |
| `verify/aec/diagnose_aec_stereo_data.py` | AEC stereo 数据质量诊断（检测静音通道） | 数据质量 |
| `verify/verify_unipase_official_weights.py` | UniPASE 移植与官方权重端到端一致性校验 | 开发调试 |
| `verify/gcc_quality_gate.py` | GCC 延迟估计质量门报告（delay 误差 / false-lock 统计，JAE-M09） | 数据质量 |
| `verify/aec/deep_check_aec_simulation.py` | AEC 仿真链路深度检查（energy/SNR/SER/峰值异常） | 开发调试 |
| `verify/aec/compare_pfdkf_erle.py` | NumPy / PyTorch PFDKF ERLE 对比 CLI | 开发调试 |
| `verify/aec/verify_linear_aec_backends.py` | Speex / PFDKF 线性 AEC backend 验证 CLI | 开发调试 |
| `verify/verify_offline_dataset.py` | `GenerationManifest v2` 只读验证入口：full scan 并报告发布资格 | 数据质量 |
| `verify/audit_noise_sampling.py` | 动态数据集噪声采样分布审计（只读、不解码音频） | 数据质量 |

### 数据准备安全契约

以下规则由脚本强制，违反时直接报错（fail-closed），不会“尽力而为”地继续：

- `prepare_scp.py --extra name=path[,path...]` 的 `name` 必须是非空安全 basename：拒绝绝对路径、`.` / `..`、目录分隔符、CR/LF/NUL、空路径项、重复名称、内置资源名以及保留的 `train_` / `val_` 前缀；所有派生输出前缀还会校验唯一性，并保证词法包含于 `--output-dir`。
- `prepare_offline.py` 会先分别探测 clean/noisy 的真实采样率与帧数，序列化 train、validation 的严格四列 `uid fs length path` SCP，再把整组 file-set 一次事务发布；普通异常导致发布失败时会恢复全部旧目标，避免新旧 split 混合。
- 显式选择 `--datasets aec_sim` 时，未提供 `--aec_sim_testset_root`、根路径不存在或不是目录、或全部场景合计零个有效 WAV 条目都会返回非零，且不会发布空 SCP。仅默认 `all` 且未设置 `USE_TRAIN_TEST_DATA_ROOT` 时保留告警后跳过 `aec_sim` 的兼容行为。
- `shared/scp_io.py` 在解析记录及序列化 UID/path 时拒绝 CR、LF、NUL，校验整组 entries 后才写文件，防止单条记录注入多行或截断。
- `shared/pickle_io.py` 的 restricted pickle 精确白名单为：`builtins` 中的 `dict/list/tuple/set/frozenset/bytes/str/int/float/bool/complex/bytearray/slice/range/type/NoneType/object/enumerate/map/zip/property`；`collections` 中的 `OrderedDict/deque/defaultdict`；`copyreg._reconstructor`、`_compat_pickle._reconstructor`、`_codecs.encode`；以及 `numpy.dtype`、`numpy.ndarray`、`numpy.core.multiarray.{_reconstruct,scalar}`、`numpy.core.numeric._frombuffer`、`numpy._core.multiarray.{_reconstruct,scalar}`、`numpy._core.numeric._frombuffer`。其它 global 一律拒绝；仅可信数据可设置 `PREPARE_SCP_ALLOW_UNSAFE_PICKLE=1|true|yes`，且调用方必须提供非空 `purpose`，才能显式回退完整 `pickle.load`。
- Phase 4C shared module 包含 `prepare_data_tools/generation_manifest.py`、`offline_reload_gate.py` 与 `offline_publication.py`。`GenerationManifest v2` 是唯一的发布与读取契约，通过 `root_bindings` 显式绑定多个受控 root，当前支持 SE 与 TSE reload adapter；AEC adapter 仍 fail-closed。资源使用 `root_alias` 与 root 内相对 POSIX path 绑定 size/SHA-256，SCP waveform 写为 `waveforms:<relative-path>`，所有 root 必须共享 owner/generation identity marker。
- 当前接入 generator 版本为：SE 主 offline `2.11`（CPU clipping selector 与 Layer D post-mix amplitude conditioning 语义），AEC simulated `2.18`（`2.14` 将 Pink/Brown colored noise 改为基于物理 Hz 频率轴的功率律 PSD；`2.15` 将仿真分层收敛为 Layer A-F，并把 post-F realized 改为顶层审计节点；沿用后续版本的当前 A–F/审计合同）、AEC OOD `1.2`、AEC offline indexer `2.1`、AEC Challenge offline producer `1.1`，TSE pseudo-target `2.0`（GenerationManifest v2、TSE full reload 与 REAL-T evaluation-only 合规门禁），DNS3 `2.1`。旧 D-02/D02 schema 已退出；旧 YAML `adc_hard_clip_prob` / `adc_peak_guard_trigger` 仅在 schema/config 校验中 fail-fast。通用 SE indexer 的 legacy 模式继续输出绝对路径且不写 verified receipt；只有显式 `--verified-publication --waveform-root` 才发布 `GenerationManifest v2`。DNS3 `--scp-only` 只能通过 `ExistingOwnedRoot` 复用同 identity waveform root；旧资产缺少 shared ownership/generation marker 时 fail-closed，禁止原地补写 marker 接管。
- 正式 publication 只接受最终路径上的 full scan：有 production Dataset adapter 时强制 `strict_load=True` full reload，通过后才写 `generation_gate_receipt.json`；`adapter=none`（如 AEC simulated 七场景）不构造 OfflineDataset，但仍须通过 full static/publication gate。`--probe` 只用于调试，永不具发布资格。`python tools/verify/verify_offline_dataset.py <manifest-root> --root waveforms=/path` 是只读的 `GenerationManifest v2` 独立验证入口：不传 `--probe` 时执行 full scan 并报告资格，传入时只扫描前 N 条；该 CLI 本身不发布集合，也不写 receipt。
- `tools/verify/simulation_audit_contract.py` 定义独立 simulation audit envelope：当前写出并只接受 `contract_version="1.4"`（固化 post-mix D/E/F clipping 分层与 deterministic final guard 审计语义），旧版本 envelope 一律拒绝读取；SE/AEC 保留各自 layers/measurements，TSE capture-off 仅发布 sealed role-local metadata trace，stems capability 为 unavailable（reason=`capture_not_requested`），capture-on 发布 `source-scene-stems-v2` / `stem-descriptors-v2` 摘要及实际 waveform bytes 的 SHA-256，不把 waveform 嵌入 JSON；另有 task-local stem capture，禁止伪造 stem capability。

---

## 噪声数据集目录重排

把 `source/<采样率>/<噪声类型>/<数据集名称>/<哈希分片>/<文件>` 复制为
`output/<数据集名称>/<噪声类型>/<文件>`。同一数据集、同一噪声类型的不同采样率文件会合并到一个目录，哈希分片目录不会写入目标；合并后出现同名文件时脚本报错。目标路径的已存在祖先与运行时创建的目标父目录、以及源树中的采样率/噪声类型/数据集目录、子目录和文件均拒绝 symlink；全树预检会绑定源文件的 device/inode/mode/size/mtime/ctime 与 SHA-256，复制前及复制期间身份或内容变化即失败。源目录没有可复制文件时返回非零；全部文件 staging 完成后才集合事务发布。默认执行并行复制；目标内容摘要一致时跳过，不一致时需显式添加 `--overwrite`。

```bash
# 使用默认源目录和目标目录执行复制；产物 = output/<数据集名称>/<噪声类型>/<文件>
python -m tools.ops.data.reorganize_noise_datasets

# 自定义目录并完整预检；dry-run 仍会扫描全部文件，但不写目标
python -m tools.ops.data.reorganize_noise_datasets \
    --source /path/to/source --output /path/to/output --dry-run
```

---

## 验证 CLI 资产归属

非 pytest 的 compare / verify CLI 统一归 `tools/verify/` 维护（完整索引见 [`verify/README.md`](verify/README.md)）。正式入口为：

```bash
python tools/verify/aec/compare_pfdkf_erle.py --help
python tools/verify/aec/verify_linear_aec_backends.py --help
```

`verify_linear_aec_backends.py` 会校验输入、各 backend 输出及 ERLE/RMS 指标：输出必须是一维、非空、长度与输入一致且不含 NaN/Inf，指标也必须为有限值；任一已返回输出校验失败，或所有 backend 都未成功完成，CLI 均返回非零，且失败结果不会写 WAV。任一被请求的 backend 执行失败（即使其它 backend 成功）同样返回非零；只有显式未请求/环境不可用导致的 skip 不影响退出码。

---

## 🔩 模型与权重工具

| 脚本 | 功能 | 用途 |
|------|------|------|
| `model_tools/convert_checkpoint.py` | 旧项目权重迁移（se_train / urgent2026 / use_train_plus / pase_official / unipase_official） | 迁移 |
| `model_tools/extract_model_checkpoint.py` | 从可信训练 checkpoint 提取出可安全 weights-only 加载的 model-only checkpoint | 安全迁移 |
| `model_tools/convert_taploss_official_checkpoint.py` | 把固定 SHA 的官方 TAPLoss checkpoint 转成 direct tensor-only state dict | 安全迁移 |
| `model_tools/convert_rt_semamba_checkpoint.py` | 严格验证并转换官方 RT-SEMamba KD1/KD8 checkpoint | 迁移 |
| `model_tools/convert_mambattention_rwsa_checkpoint.py` | 把 MambAttention/RWSA 官方 generator checkpoint 转成本仓 GAN 格式 | 迁移 |
| `model_tools/export_pase_artifacts.py` | 把 PASE/StuPASE 训练 checkpoint 导出为独立推理 artifact | 部署 |
| `model_tools/download_pase_pretrained.py` | 下载 PASE/StuPASE/UniPASE 官方预训练权重（HuggingFace `cisco-ai/pase`、`cisco-ai/stupase`、`Xiaobin-Rong/unipase`） | 迁移 |
| `model_tools/average_checkpoints.py` | checkpoint 逐键平均（SWA-like，SE/AEC/TSE 通用） | 训练收尾 |
| `model_tools/audit_metrics.py` | 按统一合同审计模型的 Params、MACs 与 RTF | 开发 |
| `model_tools/dbgnet_output_bottleneck.py` | DBGNet 输出表达瓶颈实验 harness（oracle real mask / CRM / reference sensitivity 对照，JAE-M08） | 实验分析 |

### TAPLoss 官方 checkpoint 安全转换

官方 TAPLoss checkpoint 是第三方权重，原始文件使用旧版 NumPy pickle 表示，**不能直接加载**，也不能启用 unsafe fallback；必须先离线转换一次。支持的官方源固定在 commit `d1fa133d80161bb837f2874d0b18eed89d5bf554`；约 67 MB 的 `TAP_estimator_model_v0.pt` 必须具有 SHA-256 `b6455bba8f282207f8dba8c38ef4c0bc410f76ab8f732d1d6f2de85ee05c4b72`。

```bash
python tools/model_tools/convert_taploss_official_checkpoint.py \
    --input /trusted/path/TAP_estimator_model_v0.pt \
    --output /trusted/path/TAP_estimator_model_v0.tensor_only.pt
```

`--input` 指向固定官方源文件，`--output` 必须指向现有目录中的新文件。转换器拒绝 input/output 叶子及所有已存在祖先路径组件中的 symlink。它只读取一次 source bytes，以同一份绑定 bytes 完成 SHA 校验和 pickle globals 精确白名单审计，再将这些 SHA 绑定 bytes 传入短生命周期 worker。只有 worker 会配置兼容 globals，并通过 `safe_torch_load` 加载其私有、非 symlink 的临时普通文件；该文件在成功和异常路径都会清理，父进程的 safe globals 不受污染。整个流程没有 unsafe fallback。

worker 只提取并严格校验 `_TAPAcousticEstimator` 预期的 28 项 CPU Tensor。转换器原子发布约 19.5 MB 的 **direct tensor-only state dict**（产物），再用普通 `safe_torch_load` 重新加载并执行 strict 模型校验。产物不包含 optimizer state、NumPy 对象、样例或 metadata。训练配置的 `acoustic_model_path` 必须指向这个 `.tensor_only.pt` 产物，不得指向官方原始 checkpoint；原始与转换后权重均为本地第三方 artifact，不得提交到仓库。

external 门禁使用真实官方资产，默认不联网。可指定本地文件，或显式允许测试下载：

```bash
TAPLOSS_OFFICIAL_CHECKPOINT=/trusted/path/TAP_estimator_model_v0.pt \
python -m pytest -m external tests/test_taploss_official_checkpoint.py -v

TAPLOSS_DOWNLOAD_OFFICIAL_CHECKPOINT=1 \
python -m pytest -m external tests/test_taploss_official_checkpoint.py -v
```

---

<a id="audio-filter-pipelines"></a>
## 🎛️ 音频数据筛选管道

两条管道都用来在海量原始音频里筛出可用数据，但目标不同：一条筛“不含人声的噪声/音乐”，另一条筛“质量达标的语音”。两条链路都冗长，细节见各自 README。

| 管道 | 用途 | 文档 |
|------|------|------|
| **noise_filter_pipeline** | 噪声/音乐筛选（去语音）；C1 Qwen 三分类 → C2 AudioTag 固定 top-5 → C3 speech-only veto | [README](noise_filter_pipeline/README.md) |
| **audio_filter_pipeline** | 语音质量筛选（MOS）；Step4 SCOREQ + URGENT-PK + DNSMOS 严格 3/3 合取 | [README](audio_filter_pipeline/README.md) |

---

## 🧹 运维与分析 CLI

`tools/ops/` 下是一次性但可复用的数据整理、报告、评估和任务调试 CLI。
所有运维 CLI 均使用 `python -m tools.ops.<domain>...` 作为规范入口。

| 脚本 | 功能 | 用途 |
|------|------|------|
| `ops/data/normalize_dir_names.py` | 递归规整目录名：去除空格/中文/特殊字符，严格 ASCII，支持 dry-run 与重命名映射输出 | 数据整理 |
| `ops/data/reorganize_noise_datasets.py` | 噪声数据集目录重排与复制 | 数据整理 |
| `ops/data/audio_qc_check.py` | 音频批量质检（截幅 / 电平 / 动态范围 / 静音 / 脏数据）并输出处置报告 | 数据质量 |
| `ops/data/merge_short_audio.py` | 把过短的片段按顺序拼接成满足最小时长的新文件（常配合 `audio_qc_check.py` 的报告） | 数据整理 |
| `ops/data/generate_noise_attributes.py` | 生成噪声标签 → 四维声学属性静态映射表 | 数据整理 |
| `ops/evaluation/calibrate_noise_difficulty.py` | 用参考模型实测各类噪声的抑制难度，校准 `DEFAULT_DIFFICULTY` | 数据整理 |
| `ops/reporting/experiment_report.py` | 实验报告自动生成：发现资产、运行评估、解析结果、更新累积 Markdown 报告 | 实验分析 |
| `ops/evaluation/compare_pase_metrics.py` | 对比 PASE/StuPASE 复现指标与 `RESULTS.txt`（本身不算指标） | 实验分析 |
| `ops/aec/debug_aec_state_labels.py` | 可视化单条样本的帧级 AEC state 标签（8 类场景或旧 3 类 FST/DT/NST） | 开发调试 |
| `ops/aec/realt_aec_enhance.py` | 用训练好的 DeepVQE AEC 模型对单声道 mixture 做近端增强推理 | 开发调试 |
| `ops/reporting/experiment_report_identity.py` | 实验报告 checkpoint identity 的纯逻辑合同与迁移分类 | 实验分析 |

---

## 📐 常用命令

### Voicebank-Demand 数据准备

```bash
python tools/prepare_data_tools/scp/prepare_voicebank.py \
    --data_root /path/to/DS_10283_1942 \
    --output-dir data/se/voicebank_demand \
    --target_sr 16000
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | *(必填)* | 数据集根目录 |
| `--output-dir` | `data/se/voicebank_demand` | SCP 输出目录 |
| `--target_sr` | `16000` | 目标采样率（非该采样率的音频会被重采样） |

产物：`data/se/voicebank_demand/train/{clean,noisy}.scp` 与 `test/{clean,noisy}.scp`，均为严格四列格式。

### 通用离线数据准备

**情况一：训练/验证集已分目录**

```bash
python tools/prepare_data_tools/scp/prepare_offline.py \
    --clean_dir     /path/to/simulation_train/clean \
    --noisy_dir     /path/to/simulation_train/noisy \
    --val_clean_dir /path/to/simulation_validation/clean \
    --val_noisy_dir /path/to/simulation_validation/noisy \
    --output-dir data/urgent26
```

**情况二：单数据集自动划分（默认 95:5）**

```bash
python tools/prepare_data_tools/scp/prepare_offline.py \
    --clean_dir data/clean --noisy_dir data/noisy \
    --output-dir   data/offline --val_ratio 0.05 --sample_rate 16000
```

**输出结构：**

```
data/offline/
├── train/
│   ├── spk1.scp  # clean 四列索引：uid fs length path
│   └── wav.scp   # noisy 四列索引：uid fs length path
└── validation/
    ├── spk1.scp
    └── wav.scp
```

### ONNX 导出

> ⚠️ ONNX 导出功能已移至 `onnx/` 目录，详见 [onnx/README.md](../onnx/README.md)。generic `onnx/export.py` 只生成调试/通用 artifact；正式发布必须使用模型专属 `verify_3way_*.py` 门禁，并将 `--save-onnx` 指向仓库内的 `onnx/onnx_model/` release-root。

```bash
# DBGNet config-based generic 调试导出；--smoke 只执行单步 PT/ORT smoke
# 产物 runs/onnx_debug/dbgnet_stream.onnx 仅供调试
python onnx/export.py \
    -C configs/aec/microsoft_challenge/dbgnet/dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml \
    --checkpoint exp/aec/dbgnet/best_model.tar \
    --output runs/onnx_debug/dbgnet_stream.onnx \
    --smoke
```

```bash
# 正式发布示例：GTCRN 模型专属三路门禁（该 verifier 不接收 -C）
# 产物 onnx/onnx_model/gtcrn_streaming.onnx 需通过完整验证后才会发布
python onnx/verify/verify_3way_gtcrn.py \
    --checkpoint "$HOME/use_train_checkpoints/se/gtcrn/best_model.tar" \
    --save-onnx onnx/onnx_model/gtcrn_streaming.onnx
```

`runs/onnx_debug/dbgnet_stream.onnx` 仅用于调试，不能替代 verifier 发布的正式资产；不要使用 generic exporter 的 `--save-onnx`（该选项已被显式拒绝）。专属 verifier 要求真实、结构匹配的 checkpoint；dry-run、随机权重或单步 smoke 都不能发布。

DBGNet 当前递归 topology 由 checkpoint key `_recursive_topology_v1` 严格标识。缺少该 key 的旧拓扑 checkpoint 必须 fail-closed；不支持 `non-strict` / `partial` 加载，也不得忽略 missing/unexpected keys 绕过门禁。旧拓扑权重需按当前递归拓扑重新训练，之后才能用于推理、导出或发布。

### 权重迁移

```bash
python tools/model_tools/convert_checkpoint.py \
    --source se_train --model_type GTCRN \
    --input old_ckpt/best.tar --output ckpt/gtcrn.pth
```

支持来源：`se_train`（GTCRN）、`urgent2026`（BSRNN）、`use_train_plus`（GTCRN/BSRNN）、`pase_official`（PASE/StuPASE）、`unipase_official`（UniPASE）。`--model_type` 的完整可选值见 `python tools/model_tools/convert_checkpoint.py --help`。

---

<p align="center">
  <sub>最后更新: 2026-09-21</sub>
</p>
