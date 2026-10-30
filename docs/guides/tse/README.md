# TSE 使用指南

TSE（Target Speaker Extraction，目标说话人提取）要解决的是：一段录音里混着多个人和背景噪声，只保留其中指定某一个人的声音。输入是待处理的 mixture 和用于指定目标的 cue（线索，例如目标说话人的注册音、声纹向量、视觉特征或文本关键词）；输出是目标说话人在该 mixture 上的干净语音。默认语义是 PSE（Personalized Speech Enhancement，个性化语音增强）的单目标提取，Dataset 也支持同一条 mixture 挂多个目标及各自的 typed cue。

数据合同为 speaker、spatial、visual、textual 四类 cue 保留 typed slots。cue 指「把目标身份以哪种方式送进模型」，例如 enrollment waveform（注册音波形）、speaker embedding（声纹向量）、预提取的 visual features 或 phoneme ids；同一条 mixture 只存一份，`targets` 列表按目标逐行携带 cue。

本目录收录 TSE 的任务指南与训练入门。训练入口统一是根目录 `train.py`（按 `data.mode: dynamic_tse` / `offline_tse` 路由到 `TSETrainingEntry`），配置在 `configs/tse/`；可复制的数据准备、训练与推理命令见任务入口 [`../../../tasks/tse/README.md`](../../../tasks/tse/README.md)。

## 主指南

- [TSE Training Primer](TSE_Training_Primer.md)：两个 registry 模型的定位与选择、四类 cue 的接线状态、训练实现约定（多目标批、SSA、multi-task speaker CE）与评测入口。适合第一次读 TSE 代码。
- [任务 README](../../../tasks/tse/README.md)：数据清单格式、动态均衡采样、仿真管线、四模态完成度表、训练命令与 speaker embedding 权重准备。跑通训练以该文为准。

## 关键事实（先过一遍，避免误读后面的细节）

- **数据模式**：TSE 使用 `data.mode: dynamic_tse`（在线仿真，含多源 T/I/N 场景与 enrollment 采样增强）或 `offline_tse`（读取 WeSep 风格 `samples.jsonl` + `cues.yaml` 的显式清单）。新 TSE 多源管线的场景抽样、SIR、timeline 与近端 enrollment 契约见 [`../simulation/Multi_Source_Simulation.md`](../simulation/Multi_Source_Simulation.md)。
- **批合同**：collate 产出带 `validate()` 的 `TSEBatch`（dict item → `tse_collate_fn`）。`mixture [B,C,T]`、`target [R,T]`、`num_targets [B]` int32 且 `R=sum(num_targets)`；cue 按 R 行 typed 对齐，启用但缺失的 slot 在 item 用 `value=None/present=False`、在 batch 用零占位 + `present=False`；tuple item/batch 一律显式拒绝。trainer 的 `_unpack_batch` 只接受 `TSEBatch`。
- **模型 I/O**：STFT/iSTFT 由 trainer 统一管理，模型侧是 cue-aware dict 多输入、谱进谱出——registry 按配置装配 `mix_spectra` 与 `enroll_spectra` / `enroll_wav` / `spk_embedding`，输出固定为 `target_estimate_spectra`。当前 registry 含 `TSE_BSRNN_SPK` 与 `TSE_Modular` 两个模型身份：前者保留 causal streaming / ONNX / TorchScript 路径；后者为 BSRNN 模块化组合（speaker/spatial/visual/textual frontend 已接线，`listen` 与 raw video 由配置边界拒绝），不支持 streaming，`supports_multi_task=False`（配 `SpeakerCrossEntropyLoss` 会显式报错）。
- **训练入口约定**：`find_unused_parameters` 由 `TSETrainingEntry` 按 cue 配置自动推导（当前合法配置下恒为 `false`）；多目标批训练（`targets_per_mixture > 1`）对 `[B,C,T]` mixture 只算一次 STFT，按 `num_targets` 展开后与 `[R,T]` target 逐行对齐；SSA（Self-estimated Speech Augmentation）与 multi-task speaker CE 的细节见 [Primer](TSE_Training_Primer.md)。
- **推理**：根 `inference.py` 通过 `--tse-manifest` 接受严格 JSONL 文件清单，并用 `TSEExtractor`（fail-closed 加载）与 `TSEInferenceAdapter` 执行；清单中的文件 cue 必须使用模型声明的 canonical role，inline cue 放在 `attributes.cues`。公共 runner 负责 manifest、输出锁和 reuse，TSE sidecar 由任务侧从 `InferenceReport` 发布。需要直接调用模型时仍可使用 `tasks/tse/inference/extractor.py::TSEExtractor`。

## 权威口径

- 模型接口与挂载方式：[`../../architecture/TSE_Architecture.md`](../../architecture/TSE_Architecture.md)。
- TSE 论文精读（REAL-TSE Challenge 专题、LExt/LExTra 等）：[`../../paper/tse/`](../../paper/tse/README.md)。
- 训练目标与 PIT 无关声明：[`../loss/README.md`](../loss/README.md)；SS/PIT 合同属于 SS 任务，不在 TSE 消费链上。
- 配置命名与继承语义：[`../../../configs/README.md`](../../../configs/README.md)。
