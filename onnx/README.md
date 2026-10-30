# 🚢 ONNX Export & Deployment

<p align="center">
  <strong>流式模型 ONNX 导出、精度验证与生产级部署</strong><br>
  覆盖 SE / AEC / TSE 模型的导出、三路精度对齐与发布门禁，支持面见下方「这是什么」。
</p>

---

## 这是什么

训练出来的模型是 PyTorch 的，PyTorch 又重又难塞进手机、嵌入式设备或 C++ 服务里。本目录负责把
**流式（streaming，即一边收音频一边出结果）** 模型导出成 [ONNX](https://onnx.ai/) 格式：
ONNX 是一种跨框架的模型交换格式，导出后可以用轻量的 ONNX Runtime（下文简称 ORT）在没有 PyTorch 的
环境里推理。导出后还要证明「导出前后算出来一样」，否则部署上线就是另一种模型了。

本目录做三件事：

1. **导出**（`export.py`）：把 PyTorch 模型变成 ONNX 文件，附带推理所需的元数据（metadata）；
2. **验证**（`verify/`）：跑三路数值对齐，确认没有导坏；
3. **推理**（`infer/`）：用 ORT 加载导出结果，逐帧跑推理，作为部署示例。

当前导出支持面：`onnx/export.py` direct mode 支持 GTCRN、CAGCRN、DeepVQE、LocalVQE、FastEnhancer、
UL-UNAS、UltraDPCAEC、causal TSE 与 OnlineSpatialNet；config-based 路径支持因果 BSRNN 与 DBGNet。
TSE 使用固定 enrollment condition + temporal cache 的版本化 flat ABI；OnlineSpatialNet 当前仅支持
generic debug 导出。

## 术语速查

| 术语 | 含义 | 为什么要关心 |
|---|---|---|
| ONNX / ORT | 跨框架模型格式 / ONNX Runtime，执行 ONNX 的推理引擎 | 导出产物最终由它们运行 |
| cache | 流式推理的「记忆」：上一帧要留给下一帧的状态（卷积历史、LSTM 状态等） | cache 的名称、顺序、shape 一旦不匹配，输出就是错的 |
| ABI | 这里指模型对外的输入输出接口约定：张量名称、顺序、shape、dtype | 训练侧与部署侧必须用同一套 ABI |
| metadata | 写进 ONNX 文件的额外说明（STFT 参数、cache 配置、发布绑定等） | ORT 脚本启动时校验它，缺失或漂移直接报错 |
| direct mode / config-based mode | 两种导出入口：直接给 `--model` 名字，或给 `-C` 配置文件 | 决定你该用哪种命令 |
| generic debug 导出 | 只用于调试的导出：`--output` 只能写仓库内 `runs/onnx_debug/`，不算正式发布 | 别把 debug 产物当正式资产 |
| 三路验证（A/B/C） | 离线 PyTorch / 流式 PyTorch / 流式 ORT 三者数值对齐 | 只有全 `PASS` 才允许正式发布 |
| gate / fail-closed | 发布门禁：任一条件不满足就拒绝写出，不降级放行 | 决定产物能不能进 `onnx/onnx_model/` |
| release-root | 正式产物的落地目录，本仓库为 `onnx/onnx_model/` | 只有通过门禁的产物才写这里 |

## 📑 目录

- [目录结构](#-目录结构)
- [快速开始](#-快速开始)
- [模块说明](#-模块说明)
- [高级用法](#advanced-usage)
- [精度验证报告](#-精度验证报告)
- [支持的模型](#-支持的模型)
- [部署说明](#-部署说明)
- [已知限制](#known-limitations)
- [依赖](#-依赖)
- [相关文档](#-相关文档)

---

## 📂 目录结构

```
onnx/
├── __init__.py                        # 模块导出
├── export.py                          # 统一 ONNX 导出工具
├── download_official_fastenhancer_onnx.py  # 下载官方 FastEnhancer ONNX (GitHub releases)
├── utils.py                           # 通用工具（public metadata writer、PyPI onnx 隔离）
├── runtime_spec.py                    # 单一 StreamingRuntimeSpec / wrapper-session adapter
├── models/                            # 流式 ONNX 模型实现
│   ├── streaming_gtcrn.py             # StreamingGTCRN
│   ├── streaming_cagcrn.py            # StreamingCAGCRN
│   ├── streaming_deepvqe.py           # StreamingDeepVQE
│   ├── streaming_localvqe.py          # StreamingLocalVQE（仅 generic debug）
│   ├── streaming_online_spatialnet.py # OnlineSpatialNet flat ABI（仅 generic debug）
│   ├── streaming_fastenhancer.py      # StreamingFastEnhancer
│   ├── streaming_ulunas.py            # 官方 StreamingULUNAS
│   ├── streaming_dbgnet.py            # StreamingDBGNet（部署默认不导出 state detector）
│   ├── streaming_ultra_dpc.py         # StreamingUltraDPCAEC
│   ├── streaming_bsrnn.py             # StreamingBSRNN wrapper
│   ├── streaming_tse.py               # TSE fixed-enrollment flat cache ABI
│   └── streaming_wrapper.py           # 通用流式推理包装器
├── infer/                             # ONNX Runtime 推理脚本
│   ├── streaming_ort_runner.py         # 通用 NumPy-state 单帧 runner
│   ├── infer_chunk_zipenhancer.py      # ZipEnhancer PT/ORT 分块离线推理
│   ├── infer_streaming_cagcrn_ort.py
│   ├── infer_streaming_deepvqe_ort.py
│   ├── infer_streaming_fastenhancer_ort.py
│   ├── infer_streaming_gtcrn_ort.py
│   ├── infer_streaming_ulunas_ort.py
│   ├── infer_streaming_tse_ort.py
│   └── infer_streaming_ultra_dpc_ort.py
├── verify/                            # PT vs ONNX 三路验证与发布门禁
│   ├── verify_3way_cagcrn.py
│   ├── verify_3way_dbgnet.py
│   ├── verify_3way_deepvqe.py
│   ├── verify_3way_fastenhancer.py
│   ├── verify_3way_gtcrn.py
│   ├── verify_3way_ulunas.py
│   ├── verify_3way_tse.py
│   ├── verify_3way_ultra_dpc.py
│   ├── verify_common.py                # lifecycle、ThreeWayResult、发布 helper
│   └── manifest.py                     # 验证/发布 manifest 记录
├── test_audio/                        # 测试音频 (noisy/result)
├── pytorch_model/                     # PyTorch 权重 (验证用)
└── onnx_model/                        # 导出的 ONNX 模型（`*.onnx` 由 .gitignore 忽略）
```

---

## 🚀 快速开始

导出有两条路径，先选路径，再照抄命令：

| 你想做什么 | 用哪条路径 | 产物落在哪 |
|---|---|---|
| 只想确认导出跑得通、做结构或数值诊断 | 第 1 节 Generic debug 导出（`onnx/export.py`） | 仓库内 `runs/onnx_debug/`，**不是**发布资产 |
| 要发布一个能上线的 ONNX 文件 | 第 2 节 专属 verifier（`onnx/verify/verify_3way_*.py --save-onnx`） | `onnx/onnx_model/`，且只有三路门禁全 `PASS` 才会写出 |

两条路径都需要 checkpoint；config-based 路径还需要与 checkpoint 匹配的 config。
导出后可用 `onnx/infer/` 下的 ORT 脚本逐帧跑一遍，确认推理链路真的能用。

### 1. Generic debug 导出

`--output` 只能指向仓库内 `runs/onnx_debug/` 下的 `.onnx` 文件；默认不验证。需要单步 PyTorch/ORT smoke 时显式加 `--smoke`（兼容别名 `--verify`）：

```bash
# direct mode
python onnx/export.py \
    --model streaming_gtcrn \
    --checkpoint exp/se/gtcrn_causal_offline/checkpoints/<run-id>/best_model.tar \
    --output runs/onnx_debug/gtcrn_stream.onnx \
    --smoke

# config-based mode（仅 causal BSRNN 或 DBGNet）
python onnx/export.py \
    -C configs/aec/microsoft_challenge/dbgnet/dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml \
    --checkpoint exp/aec/dbgnet/best_model.tar \
    --output runs/onnx_debug/dbgnet_stream.onnx \
    --smoke

# LocalVQE 仅支持 direct generic debug；必须提供匹配的 config
python onnx/export.py \
    --model streaming_localvqe \
    -C configs/aec/microsoft_challenge/localvqe/localvqe_16k_20ms_delay80ms_loss_hybrid_fwsdr.yaml \
    --checkpoint exp/aec/localvqe/best_model.tar \
    --output runs/onnx_debug/localvqe_stream.onnx \
    --smoke
```

`--save-onnx` 在 generic exporter 中仅为兼容旧命令而保留解析，程序会在构造模型或读取配置前 fail-fast；正式发布必须改用仓库中实际存在的模型专属 verifier。对 OnlineSpatialNet，当前没有实现 dedicated verifier，拒绝提示不会指向不存在的 `verify_3way_online_spatialnet.py`；只能使用 `--output` 生成 debug artifact，不能正式发布。OnlineSpatialNet 的 flat cache ABI 为 v2（`mhsa` 增量 K/V + `frame_index`、`ret` recurrent kv/scale、`mamba` conv/ssm）；`attention='mamba(...)'` 分支依赖 mamba_ssm 的 selective scan 内核，导出边界直接拒绝。`--output` 会拒绝 `runs/onnx_debug/` 之外的路径，避免把单步 smoke 误当正式三路验证。当前无专属 verifier 的 BSRNN 与 LocalVQE 也只能由 generic exporter 生成 debug artifact；OnlineSpatialNet 尚无专属 ORT CLI。`onnx/models/streaming_gtcrn.py` 的独立 demo 同样只允许在该 debug 根下创建新文件，并通过 owned 临时文件与安全 debug publisher 发布主图和简化图。

ONNX 只属于已声明的模型/runtime artifact 发布边界，不是 SSL 的任务入口。SSL 的空间标签训练、scene manifest 与 geometry/provenance 合同由 `tasks/ssl/` 负责；根 `inference.py` 和 ONNX exporter 都不会把 SSL scene 或 audio file 推理伪装成可发布的通用模型。AEC 文件推理的 descriptor factory 接收根入口的 `stft_cfg` 并显式映射到任务 Adapter 的 `stft` 合同；这不会改变 ONNX 的模型专属 STFT/runtime metadata。AEC 的 farend 仍必须是 mono，multichannel mic 只能经显式 reference adapter 投影。需要 SSL 运行时扩展时，必须新增任务专属 exporter、metadata 与 verifier，而不是复用 SE/AEC 的流式 ABI。

### 2. 专属 verifier 正式发布

以下命令会临时导出并执行模型专属三路验证，全部通过后才原子发布：

```bash
# GTCRN
python onnx/verify/verify_3way_gtcrn.py \
    --checkpoint exp/se/gtcrn/best_model.tar \
    --save-onnx onnx/onnx_model/gtcrn_stream.onnx

# DeepVQE
python onnx/verify/verify_3way_deepvqe.py \
    -C configs/aec/microsoft_challenge/deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml \
    --checkpoint exp/aec/deepvqe/best_model.tar \
    --save-onnx onnx/onnx_model/deepvqe_stream.onnx

# CAGCRN
python onnx/verify/verify_3way_cagcrn.py \
    -C configs/aec/microsoft_challenge/cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml \
    --checkpoint exp/aec/cagcrn/best_model.tar \
    --save-onnx onnx/onnx_model/cagcrn_stream.onnx

# DBGNet
python onnx/verify/verify_3way_dbgnet.py \
    -C configs/aec/microsoft_challenge/dbgnet/dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml \
    --checkpoint exp/aec/dbgnet/best_model.tar \
    --save-onnx onnx/onnx_model/dbgnet_stream.onnx

# FastEnhancer
python onnx/verify/verify_3way_fastenhancer.py \
    --checkpoint exp/fastenhancer_b/best_model.tar \
    --save-onnx onnx/onnx_model/fastenhancer_b_stream.onnx

# UL-UNAS
python onnx/verify/verify_3way_ulunas.py \
    --checkpoint tasks/se/models/discriminative/ulunas/open_model_checkpoint/model_trained_on_dns3.tar \
    --save-onnx onnx/onnx_model/ulunas_stream.onnx

# Ultra-DPC-AEC；正式发布仅允许完整三路可比的 time_skip=1
python onnx/verify/verify_3way_ultra_dpc.py \
    --checkpoint exp/ultra_dpc_aec/best_model.tar \
    --freq-ratio 2 --time-skip 1 \
    --save-onnx onnx/onnx_model/ultra_dpc_stream_fr2.onnx

# causal TSE：离线 PT / 流式 PT / ORT + waveform WOLA
python onnx/verify/verify_3way_tse.py \
    -C configs/tse/inference/tse_bsrnn_spk_realt_baseline_causal_infer.yaml \
    --checkpoint tasks/tse/pretrained_models/spk_emb_causal_100/avg_model.pt \
    --save-onnx onnx/onnx_model/tse_bsrnn_spk_causal_streaming.onnx
```

正式发布统一要求：提供 checkpoint；使用 `float32`；阈值为有限正数且不得比脚本预注册值更宽；A/B/C 三路状态必须全部为 `PASS`，每路比较数量必须大于 0，所有指标必须有限。`SKIP`、空比较、`NaN` 或 `Inf` 均阻断发布。DeepVQE、CAGCRN、DBGNet 还必须提供与 checkpoint 匹配的 config。ONNX loader 显式区分 single-model 与 GAN generator family，拒绝多个权重 wrapper key 的歧义 checkpoint；存在 EMA 时先加载基础权重再应用 EMA，与根推理和评估入口保持一致。随机权重和低精度模式只可用于手工结构/诊断验证。

> 下面两段是发布门禁的完整实现语义（谁在什么时候校验什么、并发时怎么保证不写坏已有资产）。第一次读只要记住一句：**checkpoint 齐全、三路全 `PASS` 的产物才会被发布，任何异常都不降级放行**。

8 个专属 verifier 均在创建 ORT session 前绑定 verifier-owned snapshot 的 inode、`size + SHA256`，并由同一 `VerifiedOnnxArtifact` 对象执行发布。共享 publisher 以 target 级进程锁和原子 rename/exchange 实现 CAS：并发 target 漂移时保留并发写入；commit 前失败只对本事务 inode 执行一次身份校验 rollback，恢复既有 target 或撤销本事务新文件。目录 `fsync` 后视为已提交，后续清理异常会显式报告，不伪装成未提交。

官方 FastEnhancer 下载脚本不信任运行时远端元数据：每个 release asset 的固定 SHA-256 与 byte size 均内置在脚本中，下载后先校验，再复用上述 `publish_verified_onnx()` 共享发布事务；摘要或大小不匹配时 fail-closed，不覆盖既有正式资产。

UL-UNAS 官方 ABI 为 `mix [B,257,1,2]`、`conv_cache [B,5358]`、`tfa_cache [B,402]`、`inter_cache [B,1056]`；输出为 `enh` 及三个对应的 `_out`。Ultra-DPC-AEC 的 `--freq-ratio` 与 `--time-skip` 必须匹配训练配置；`time_skip>1` 因 A/C 为 `SKIP`，只能手工验证，不能正式发布。

CAGCRN 新 artifact 必须同时包含完整 `streaming_runtime_spec`（`n_fft`、`hop_length`、`win_length`、`sample_rate`、`window`、`center`）、`cagcrn_cache_cfg` 与 `cagcrn_runtime_spec` v2。后者固定要求 `cache_abi_version=2`，并以严格顺序声明 40 个 cache 输入 `cache_0` … `cache_39` 及对应输出 `out_cache_0` … `out_cache_39`；`delay_d` 是由配置控制的 CATA 含当前帧历史候选窗口（当前 recipe 为 6），`algorithmic_delay_frames=0`。session preflight 会在首帧前同时校验 cache input/output 的名称、顺序、shape、ORT tensor type，以及谱 I/O graph shape 与 STFT metadata。缺少该 metadata、旧 `cagcrn_delay_d`、v1 或 41-cache artifact、或缺少完整 STFT metadata 的 artifact 均已淘汰；runtime/infer 不提供 legacy compatibility switch，必须使用专属 verifier 重新导出。

DeepVQE 与 DBGNet 新 artifact 还必须包含各自的 `deepvqe_runtime_spec` / `dbgnet_runtime_spec` 以及带 SHA-256 的 `deployment_config_binding`。绑定内容覆盖 model args、完整图外 STFT、`data.sample_rate` 与 `aec_frontend`；runtime 会在首帧前校验 metadata 版本和字段集合、binding 摘要、谱 I/O 的 frequency bins/rank/type、cache output shape/type，并要求 artifact 与 config 构建出的 wrapper 在 I/O 名称、cache shape、STFT、模型专属 metadata 和 deployment binding 上完全一致。DBGNet 启用 state detector 时还会校验 `state_probs` 的存在性、shape、类别数与 tensor type。缺失或漂移均 fail-closed，即使内部调用方请求 legacy metadata 兼容也不会放行这两类 artifact。

DeepVQE ORT CLI 会在读取音频前完成上述 preflight；纯 ORT 推理可直接采用已验证 artifact 中的参数，若同时提供 `--config`，则 config 必须与 artifact 的 deployment binding 完全一致。DeepVQE 与 DBGNet 的正式发布还要求 widened 口径（至少 1000 帧、2 次同步 reset）。DeepVQE 每段 reset 都让离线/PT/ORT 同时冷启动：既保留跳过该段前 `align_delay` 帧的稳态 A/C 口径，又以独立硬门禁覆盖 cold-start/reset 后的全部帧；任一段造成有效稳态比较为空、冷启动帧未完整覆盖或 A/C 任一路失败都阻断发布。DeepVQE 还必须通过独立的 `External iSTFT synthesis capability` gate；该 gate 只证明绑定参数可合成非空、有限 waveform，不证明流式/离线 waveform 等价、产品 OLA/flush/输出长度、音质或端到端延迟。当前 Hann 窗边界下 `center=false` 不满足 NOLA，会阻断 waveform 正式发布；对绑定 `center=true` 的训练配置，部署必须保持一致，`center=false` 只能作为独立的前端漂移实验，不能冒充匹配部署。DBGNet 还要求 state gate（启用时）与同一 synthesis capability gate 通过。narrow 口径、低精度、缺 metadata 或 wrapper/artifact 合同不一致只可诊断，不能通过 `--save-onnx` 发布。

---

## 📖 模块说明

### `export.py` — debug/general 导出工具

支持 `--model` direct mode 与 `-C/--config` config-based mode。两者都必须通过 `--output` 写入仓库内 `runs/onnx_debug/`；单步 smoke 默认关闭，按需显式使用 `--smoke`：

```bash
python onnx/export.py \
    --model streaming_gtcrn --checkpoint /path/to/checkpoint \
    --output runs/onnx_debug/model.onnx --smoke

python onnx/export.py \
    -C /path/to/config.yaml --checkpoint /path/to/checkpoint \
    --output runs/onnx_debug/model.onnx --smoke
```

导出使用模型实际提供的 cache 初始化接口推导 dummy cache。TSE 的 cache 前缀是固定 enrollment condition（`enroll_spectra` / `context` / `spkemb`，按 cue 出现），其后是 separator LSTM 与可选 USEF history；输出统一为同名 `_out` loopback。cache shape、名称和动态轴全部来自 `StreamingTSEFlat`，导出脚本不硬编码。

TSE runtime metadata v2 还绑定实际加载后的全部参数/buffer 内容 SHA-256，包括
condition encoder。ORT artifact 与提供 enrollment condition 的 PyTorch 模型必须同时
匹配 ABI 和权重身份；同架构但不同权重不能混用。`reset(condition_state)` 从 cold
state 清空全部时序 cache 后再注入 condition，不能继承上一录音或上一说话人的历史。

支持 metadata 的导出分支通过 public `write_onnx_metadata()` / `write_streaming_runtime_metadata()` 一次写入完整 metadata。纯文件 metadata 修改由隔离子进程加载 PyPI `onnx`；`torch.onnx.export` 必须接收活 PyTorch model，无法可靠跨进程序列化，因此仍在当前进程临时切换 PyPI `onnx`。该 fallback 由进程内全局 `RLock` 串行化，支持同线程嵌套，并在最外层退出时完整恢复本地 `onnx.*` 模块和 `sys.path`；它不能隔离其他绕过该锁、并发直接修改 import 状态的代码。

### `runtime_spec.py` 与 `streaming_ort_runner.py`

`StreamingRuntimeSpec` 是 export、infer 和 verifier 共用的 I/O、STFT、cache metadata 与 graph shape 契约。wrapper adapter 从模型权威接口推导 ABI；session adapter 严格解析 artifact metadata，并拒绝名称、顺序、rank、静态 shape 或未授权动态轴漂移。CAGCRN 还在 preflight 阶段校验每个 cache output 的 shape/type 与对应 cache input、metadata 一致，sentinel cache 也必须携带真实 batch 维。`StreamingORTFrameRunner` 按该 spec 初始化和更新 cache，状态从首帧到末帧始终保持 NumPy array；各 CLI 只负责模型专属 STFT、对齐、音频 I/O 和 gate，不再复制 `session.run()` cache 循环。本实现不宣称 CUDA I/O binding。

### `streaming_wrapper.py` — 流式推理包装器

提供三个核心类：

- **`StreamingInferenceWrapper`** — PyTorch 高级推理包装器，自动分块 + OLA 拼接
- **`StreamingAECInferenceWrapper`** — AEC 双输入 (mic + far-end) PyTorch 推理包装器
- **`StreamingONNXWrapper`** — ONNX 导出低级包装器，将 dict cache 转为 flat tensor 接口

---

<a id="advanced-usage"></a>
## 🛠️ 高级用法

### 手工三路结构验证

不带 `--save-onnx` 时，专属 verifier 可使用随机权重或诊断参数检查结构与数值路径；结果不会发布：

```bash
python onnx/verify/verify_3way_ulunas.py --frames 8
python onnx/verify/verify_3way_ultra_dpc.py --frames 80 --freq-ratio 2
python onnx/verify/verify_3way_ultra_dpc.py --frames 80 --freq-ratio 2 --time-skip 2
```

正式发布命令和统一门禁见“快速开始”。以下命令使用模型专属 verifier 做不发布的三路检查；只有加入 `--save-onnx` 且完整门禁通过时才会发布：

```bash
# FastEnhancer：随机权重手工检查，不生成正式资产
python onnx/verify/verify_3way_fastenhancer.py --frames 80
```


---

## 📊 精度验证报告

### FastEnhancer 三路精度对齐

| 对比路径 | MSE | max_abs | 状态 |
|----------|-----|---------|------|
| 离线 PT vs 流式 PT | ~1.3e-10 | ~1.2e-04 | ✅ PASS |
| 流式 PT vs 流式 ORT | max=1.4e-18 | — | ✅ PASS |
| 离线 PT vs 流式 ORT | ~1.3e-10 | ~1.2e-04 | ✅ PASS |

### UL-UNAS 三路精度对齐

当前 verifier 的 `--frames` 默认值为 `8`；官方 DNS3 checkpoint 的离线 PT、流式 PT、流式 ORT 的 MSE 均低于 `1e-4`。下面命令显式使用当前默认值 `8`。旧记录中出现的 `--frames 3` 仅是历史显式参数，不代表当前默认或推荐复现口径：

```bash
python onnx/verify/verify_3way_ulunas.py \
    --checkpoint tasks/se/models/discriminative/ulunas/open_model_checkpoint/model_trained_on_dns3.tar \
    --frames 8
```

### CAGCRN 三路精度对齐

下表保留既有结构/数值对拍记录，不表示本次文档同步重新执行了真实 checkpoint/config 的正式三路导出。

| 对比路径 | MSE | max_abs | 状态 |
|----------|-----|---------|------|
| 离线 PT vs 流式 PT | ~5e-14 | ~2.5e-06 | ✅ PASS |
| 流式 PT vs 流式 ORT | ~8e-14 | — | ✅ PASS |
| 离线 PT vs 流式 ORT | ~4e-14 | ~2.6e-06 | ✅ PASS |

### Ultra-DPC-AEC 三路精度对齐

随机权重、`--frames 20`、fp32 阈值 1e-5 实测：

| 配置 | 离线 PT vs 流式 PT | 流式 PT vs 流式 ORT | 离线 PT vs 流式 ORT | 状态 |
|------|-----|---------|------|------|
| `freq_ratio=1, time_skip=1` | ~6e-13 | ~1e-12 | ~4e-13 | ✅ ALL PASS |
| `freq_ratio=2, time_skip=1` | ~2e-13 | ~3e-13 | ~1e-13 | ✅ ALL PASS |
| `freq_ratio=4, time_skip=1` | ~2e-13 | ~3e-13 | ~1e-13 | ✅ ALL PASS |
| `freq_ratio=2, time_skip=2` | SKIP | ~1e-13 | SKIP | B PASS；A/C SKIP（仅诊断） |

`time_skip>1` 时流式 wrapper 使用逐帧 fallback；verifier 只比较流式 PT 与流式 ORT 的 B 路，A/C 路标记为 `SKIP`。该模式仅用于诊断，不能作为正式发布门禁；正式发布只允许 `time_skip=1`。

---

## 📋 支持的模型

| 模型 | 训练实现 | ONNX 实现 | 理论延迟 | 说明 |
|------|----------|-----------|----------|------|
| GTCRN | `tasks/se/models/discriminative/gtcrn.py` | `streaming_gtcrn.py` | **0 ms** | 严格因果 |
| CAGCRN | `tasks/aec/models/discriminative/cagcrn.py` | `streaming_cagcrn.py` | **模型图 0 帧输出延迟** | CATA/TFAG 仅访问历史；外部 STFT、OLA 与调度延迟另算 |
| DeepVQE | `tasks/aec/models/discriminative/deepvqe.py` | `streaming_deepvqe.py` | 模型图无未来谱帧输出延迟 | `align_delay` 仅控制历史窗口；需匹配训练配置/完整 checkpoint |
| LocalVQE | `tasks/aec/models/discriminative/localvqe.py` | `streaming_localvqe.py` | **0 帧模型 lookahead** | 仅 `n_fft=512`、`arch_version=2/3`、FP32；当前只有 generic debug 导出，无专属 verifier、正式发布或 ORT CLI |
| OnlineSpatialNet | `tasks/se/models/discriminative/online_spatialnet.py` | `streaming_online_spatialnet.py` | 由训练 STFT/历史 cache 决定 | 多通道 observation、mono 输出；cache ABI v2（mhsa 增量 K/V 与 `frame_index` warm-up 计数、ret recurrent kv/scale 状态、mamba conv/ssm 且不可导出）；当前仅 generic debug 导出，无专属 verifier、ORT CLI 或正式 release |
| DBGNet（unofficial implementation） | `tasks/aec/models/discriminative/dbgnet.py` | `streaming_dbgnet.py` | **神经网络未来 lookahead 为 0 帧** | config-based；该结论只覆盖谱图 neural path，不代表端到端 0 ms；正式 flat ONNX ABI 固定 batch=1；当前 hybrid baseline 与部署默认均关闭 state detector，可用 `--export-state-detector` 显式开启 |
| BSRNN | `tasks/se/models/discriminative/bsrnn.py` | `streaming_bsrnn.py` | 取决于训练 STFT | 仅因果模型，config-based wrapper |
| FastEnhancer | `tasks/se/models/discriminative/fastenhancer.py` | `streaming_fastenhancer.py` | 由训练 STFT/部署前端决定 | 模型图本身为逐谱帧流式处理 |
| UL-UNAS | `tasks/se/models/discriminative/ulunas.py` | `streaming_ulunas.py` | **网络 0 ms look-ahead**；前端约 32 ms 窗 | 官方 16 kHz / n_fft=512 / hop=256；严格官方 cache ABI；当前 ORT 示例 `center=True` |
| UltraDPCAEC | `tasks/aec/models/discriminative/ultra_dpc_aec.py` | `streaming_ultra_dpc.py` | 0 ms | 严格因果，三输入；流式支持频率压缩；`time_skip>1` 采用逐帧 fallback，仅可诊断，正式发布只允许 `time_skip=1` |
| ZipEnhancer | `tasks/se/models/discriminative/zipenhancer.py` | `onnx/onnx_model/` | N/A | 非因果离线 (预导出模型，无自动导出路径) |

---

## 📝 部署说明

### 流式模型延迟来源

| 模型 | 延迟 | 来源 |
|------|------|------|
| **GTCRN** | 0 ms | 所有卷积均为因果 padding |

其余模型的延迟口径与来源见上一节「支持的模型」表的「理论延迟」列及各模型说明。

---

<a id="known-limitations"></a>
## ⚠️ 已知限制

下列限制在本文前面的章节都有依据，这里集中列出，方便在挑模型时先看一眼：

- **只能 debug 导出、不能正式发布**：LocalVQE、BSRNN 与 OnlineSpatialNet 没有专属 verifier，只能用 `onnx/export.py` 生成 `runs/onnx_debug/` 下的调试产物。
- **OnlineSpatialNet**：没有专属 ORT CLI；`attention='mamba(...)'` 分支依赖 mamba_ssm 的 selective scan 内核，导出边界直接拒绝。
- **Ultra-DPCAEC 的 `time_skip>1`**：A/C 两路为 `SKIP`，只能手工诊断；正式发布只允许 `time_skip=1`。
- **DBGNet 的 flat ONNX ABI 固定 batch=1**：不能把 Python 侧的 batch>1 计算能力当作已验证的发布能力。
- **`center=false` 不能作为正式发布的前端**：当前 Hann 窗边界下 `center=false` 不满足 NOLA，会阻断 waveform 正式发布；训练配置绑定 `center=true` 时，部署必须保持一致。
- **不宣称 CUDA I/O binding**：`StreamingORTFrameRunner` 的状态自始至终是 NumPy array，本实现不提供 CUDA I/O binding。
- **DeepVQE 的 synthesis capability gate 覆盖范围**：它只证明绑定参数能合成非空、有限的 waveform，不证明流式/离线 waveform 等价、产品 OLA/flush/输出长度、音质或端到端延迟。

---

## 🔧 依赖

导出与验证依赖可选第三方库；未安装时相关功能会明确跳过或报错，不影响基础训练路径。

```bash
pip install "onnx>=1.14" "onnxruntime>=1.15"
```

- `onnx` / `onnxruntime`：已在 `requirements.txt` 声明（`onnx>=1.14`、`onnxruntime>=1.15`），导出与 ORT 验证的基础依赖；
- `onnxsim`：仅在需要 `--simplify` 时安装；缺失时导出脚本只 warning 并保留未简化模型；
- `mamba_ssm`：仅 OnlineSpatialNet 的 `attention='mamba(...)'` 分支需要，而该分支在导出边界被直接拒绝，所以并不构成导出依赖。

---

## 📚 相关文档

- 各任务模型清单与边界：[`tasks/se/README.md`](../tasks/se/README.md)、[`tasks/aec/README.md`](../tasks/aec/README.md)、[`tasks/tse/README.md`](../tasks/tse/README.md)
- 训练与推理总览：[根 README](../README.md)
