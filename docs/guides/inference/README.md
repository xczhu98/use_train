# 推理指南

本文是推理链路的导航层：根入口支持谁、`core/inference` 的合同五件套各自管什么、artifact 身份如何决定产物复用、流式与部署推理从哪进。具体 flag 语义以 `python inference.py --help` 与 [`core/inference/`](../../../core/inference/) 代码为 SSOT，本文不复制完整帮助。

## 根入口与任务覆盖

根目录 `inference.py` 是 SE、AEC 与 TSE 的文件推理入口；SS、SSL 没有根文件推理入口，把对应配置交给 `inference.py` 会在边界直接报错（fail-closed，不构造模型）：

| 任务 | 推理入口 | 说明 |
|---|---|---|
| SE | `python inference.py -C <config> --input_dir/--input_scp ...` | 整文件 STFT 推理；waveform-native 模型（PASE/StuPASE，`meta.allow_no_stft=true`）走长音频分块（`--chunk_seconds`） |
| AEC | 同上 | mic/far 配对输入（`--input_mic_dir` + `--input_far_dir` 或对应 scp；`--aec_flat_dir` 吃 Microsoft 风格 `*_mic.wav` + `*_lpb.wav` 扁平目录） |
| TSE | `python inference.py -C <config> --checkpoint <ckpt> --tse-manifest <jsonl>` | 严格 JSONL 描述 mixture 与模型声明的 canonical cue role；复用 `TSEExtractor`/`TSEInferenceAdapter`，公共 runner 负责 manifest、锁与 reuse，sidecar 由 `tasks/tse/inference/sidecar.py` 发布 |
| SS | `tasks/ss/inference/extractor.py::SSExtractor` | 构造需要 `model, plan, stft_module`；返回 `[K,T]` 波形 |
| SSL | 无 | 当前对外能力只有训练与合同校验 |

常用 flag 导读：`--checkpoint` 指定训练 checkpoint（waveform-native 模型忽略，权重由 artifact 注入）；`--tse-manifest` 指定 TSE 严格 JSONL 清单，`--tse-apply-vad` 开启可选 enrollment VAD；`--expected-effective-config-sha256` 在模型构造前严格校验父进程传入的 merged config 摘要；`--skip-existing` 只复用通过匹配的严格 `run_manifest.json` 校验的产物；`--allow-partial` 允许部分文件失败仍正常退出（默认 fail closed）；`--aec_stream_align` / `--aec_no_stream_align` 强制开关 canonical delay alignment（互斥，默认保持配置值）；`--sr` 固定模型采样率；`--output_relpath_scp` 自定义 UID → 相对输出路径映射。

## core/inference 合同五件套

任务 adapter 只处理任务模型 I/O（实现 `InferenceAdapter` 协议，`infer(FileInferenceItem) -> PreparedOutput`）；manifest、output lock、`skip-existing` 与 sidecar 发布都由公共层负责：

- `contracts.py`：共享数据合同（`InferenceSpec`/`FileInferenceItem`/`PreparedOutput`/`InferenceReport`）。
- `input_inventory.py`：输入 inventory——item 的 `inputs` 与 `attributes` 进入严格 canonical JSON inventory，任一变化使整个 run 失效。
- `artifact_binding.py`：artifact/source identity 绑定——已加载模型必须把 loader 冻结的身份（`ArtifactBinding.from_fingerprint` / `SourceBinding.from_fingerprint`）原样交给 runner，禁止按路径重新绑定。
- `runner.py`：任务无关的最小推理 runner；推理前与发布 manifest 前（含 `allow_partial` 中止路径）复核同一份身份，失配不得降级为可发布结果。
- `streaming.py`：流式 session seam——公共层只约定会话生命周期与输入角色（`input_roles` + `open_session()`），模型 cache、延迟估计器、任务 state 不得泄漏进公共合同。
- `output_writer.py`：原子输出、`run_manifest.json` 与 `inf.scp` 发布。

音频 manifest 以 `MANIFEST_SCHEMA_VERSION` 显式版本化：旧版本产物按复用策略重推或报错，这是 `--skip-existing`「manifest 失配即重推理」的来源。

## 部署与流式推理

- PyTorch 流式推理走 `core/inference/streaming` 的 session seam；AEC 的 canonical 延迟对齐由 `AECDeploymentFrontend` 在 STFT 前处理（属于模型输入前端，不是仿真层）。
- ONNX 导出、三路精度验证与发布门禁是独立模块：[`onnx/README.md`](../../../onnx/README.md)（export / verify / infer / 发布合同）。已发布资产的真值以该模块的验证 gate 为准。
- TorchScript 路径（TSE 等）见任务 README 的推理节。

## 权威口径

- 代码 SSOT：[`core/inference/`](../../../core/inference/)、`inference.py`、`core/utils/inference_artifacts.py`（source boundary，指纹/manifest/skip-existing 校验实现的变化会使旧产物失配）。
- 安全红线：根 `AGENTS.md`「推理 fingerprint 必须绑定实际 loader」。
- 评估侧 artifact 校验：[`../evaluation/`](../evaluation/README.md)。
