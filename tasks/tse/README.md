# 🎙️ TSE — Target Speaker Extraction

<p align="center">
  <strong>声纹降噪 / 目标说话人提取训练、模型与推理模块</strong><br>
  与 <code>tasks/se/</code> / <code>tasks/aec/</code> 共用 core 基础设施，支持 REAL-T baseline 与多 cue Modular。
</p>

---

## 📑 目录

- [任务定位](#-任务定位)
- [目录结构](#-目录结构)
- [数据流与数据准备](#-数据流与数据准备)
- [仿真管线](#-仿真管线)
- [模型 TSE_BSRNN_SPK](#tse-bsrnn-spk)
- [模型清单](#-模型清单)
- [训练](#-训练)
- [推理](#-推理)
- [评测（REAL-TSE）](#-评测real-tse)
- [speaker embedding 预训练权重](#-speaker-embedding-预训练权重)

---

## 🎯 任务定位

TSE 要解决的问题是：一段录音里混着好几个人和背景噪声，而我只要其中某一个人的声音。输入包含待处理的 **mixture** 和用于指定目标的 cue。数据合同为 speaker、spatial、visual、textual 四类 cue 保留 typed slots；当前 registry 包含既有 `TSE_BSRNN_SPK` 和 BSRNN-based `TSE_Modular`，后者的四类 cue frontend 均已接线（visual 仅预提取 features），但不支持 streaming，也没有 multi-task speaker 分类头。

输出是目标说话人在这段 mixture 上的干净语音。默认是 PSE（Personalized Speech Enhancement，个性化语音增强）语义的**单目标提取**；Dataset 统一产出 canonical dict item，同一条 mixture 只存一份，`targets` 列表可包含一个或多个目标及其 typed cues。

这份实现从哪里来、怎么接进框架：

- 模型移植来源：`wesep-real-tse`（REAL-TSE-Challenge baseline，唯一模型 `TSE_BSRNN_SPK`，带 5 种可插拔 speaker cue）。cue 指「把注册音信息送进模型的哪种方式」，见 [模型 TSE_BSRNN_SPK](#tse-bsrnn-spk)。
- 仿真策略来源：`tse_pse_zxc`（干扰人声场景按概率混合 + enrollment 采样增强），落到统一的 Layer A–F 仿真管线上；enrollment 独立于 scene bundle 处理。
- 架构约定：既有 speaker 模型和 `TSE_Modular` 的 BSRNN 路径采用 **cue-aware dict 多输入、谱进谱出**接口，STFT/iSTFT 由 trainer 统一管理；model I/O 边界从 `TSEBatch` 的 typed cue 装配按需字段。四类 cue 都有 typed 合同与已接线的 frontend，但「代码已接线」与「本机已跑通真实数据闭环」是两件事，逐模态的完成度见下表。

### 四模态完成度

「frontend」指模型侧把 cue 转成条件的模块，「Dataset 解码」指 offline/dynamic Dataset 能否物化该 cue，「真实数据闭环」指本机已有数据并在其上完成训练/推理验证。Params 只登记已确认的 textual 配置值 40.470500M（separator 21.402400M + textual frontend 19.068100M）；其余模型的 Params/MACs/FLOPs 未测，不在此处给数。

| cue | frontend | 数据合同与解码 | 本机真实数据 | 真实数据闭环 |
|---|---|---|---|---|
| speaker_waveform / speaker_embedding | `TSE_BSRNN_SPK` 的 5 种内部特征（`listen`/`usef`/`tfmap`/`context`/`spkemb`）可用；`TSE_Modular` 支持 `usef`/`tfmap`/`context`/`spkemb`，`tfmap.type=embedding` 会复用 frame-level speaker feature 同时生成 TFMap/context，支持 latent 或 block 注入；`listen` 尚未接线 | offline 与 dynamic 均解码 waveform/embedding | `data/tse/spk/wesep/datasets/` 下的 LibriMix-keywords 与 Libri2Mix | ✓ CPU 训练 + checkpoint 重载 + 文件推理已记录 |
| spatial | `modules/spatial/` 的 `SpatialFrontend`：IPD/CDF/SDF/ΔSTFT + cyclic/initstate；dynamic 可按 actor_uid 生成方向 cue | offline 解码 WeSep `npy/fields`（static/track，显式 sample anchors，不按点数 resize）；dynamic 生成 per-target static/track cue | `data/tse/spk/wesep/datasets/MC-LibriMix/`（gpuRIR 仿真 4 通道 16 kHz，19900 mixture × {early,reverb} × {mix,s1,s2}；spatial cue 为结构化 NPY（azimuth/elevation/distance，弧度制），已实测经 `TSEOfflineDataset` + collate 解码；`wesep_index/` 含三套 mix/target condition 索引） | ✓ 真实数据一 epoch/一步训练、validation、checkpoint strict 重载与 4 通道 dev 文件推理已通过（CPU smoke，配置 `configs/tse/spatial/tse_modular_spatial_mclibrimix.yaml`） |
| visual | `modules/visual/` 的 MuSE frontend；`representation=features`（预提取帧级特征）可用，raw `mp4` 明确未接 | offline 解码 `npy/muse_frontend`；raw `mp4` 在构造期拒绝 | `data/tse/spk/wesep/datasets/VoxCeleb2Mix/`（按 MuSE 官方 manifest 生成 train 20000 / val 5000，cue 为 `mp4/raw_video`，模型侧消费前需先用 MuSE frontend 预提取特征；`test` 依赖 VoxCeleb2 官方 test 视频，尚未取得） | ✗ 未做闭环 |
| textual | `modules/textual/` 的 KCE frontend（`representation=phoneme_ids`）；KCE backbone 使用官方 DAE-TSE 权重，缺失时构造即 fail-fast，无随机初始化兜底 | offline 解码 WeSep `json/dae_keyword_phoneme`（LibriMix-keywords，`selection=random` + `word_range` 选词） | `data/tse/spk/wesep/datasets/LibriMix-keywords/`（samples.jsonl + cues.yaml + cues/textual.json） | ✓ 真实数据一 epoch/一步训练、validation、完整 checkpoint 重载与 Textual 文件推理已通过 |

本机数据现状：WeSep 相关数据统一在 `data/tse/spk/wesep/`，`datasets/` 放任务就绪数据集，`sources/` 放原始语料。`datasets/Libri2Mix/` 含官方 wav16k/min 音频（train-100 13900 / dev 3000 / test 3000）、三个 split 全套 WeSep offline 清单（`samples.jsonl`/`cues.yaml`/`enroll.json`/`raw.list`，另附 WeSep 原生 `cues/audio.json`、`spk2label.json` 与 BUT SpeakerBeam 固定 enrollment 映射 `cues/fixed_enroll.json`）和 dynamic_tse 清单（`dynamic_tse/{train-100,dev,test}/speech.jsonl` 与 `spk2enroll.json`）。`datasets/LibriMix-keywords/` 具备完整的 `samples.jsonl` + `cues.yaml` + `cues/textual*.json` + KCE/DAE-TSE 资源。`datasets/MC-LibriMix/` 是 gpuRIR 仿真的 4 通道空间数据集（`simulation_config.yaml` 系按上游 README 与其模型配置中的 ReSpeaker 4-mic 几何重建，上游 v0.1 未提交该文件）。`datasets/VoxCeleb2Mix/` 按 MuSE 官方 manifest 生成 train 20000 / val 5000（test 需 VoxCeleb2 官方 test 视频，需注册授权，尚未取得）。`sources/` 下：VoxCeleb1 含官方 `wav/<spk>/<video>/<utt>.wav` 布局（1251 说话人 / 153,516 条）与 online-mixing 索引（`wesep/clean/train-vox1/`），VoxCeleb2 含官方 dev 视频（MuSE 用到的 918 个说话人按 `orig/{train,test}/` 排布，其余 5076 个保留在 `dev/mp4/`），`wespeaker_models/` 含官方 `voxceleb_resnet34` 与 `voxceleb_ECAPA512`；`flac_audio/` 与 `data/tse/spk/ori/` 为同一批文件的硬链接。

---

## 📁 目录结构

```
tasks/tse/
├── __init__.py
├── README.md                          # 📑 本文件
├── cue_contract.py                    # 四类 typed cue 的样本/批级合同
├── ../task_api.py                     # descriptor 驱动的跨任务 data API
├── model_io.py                        # trainer / extractor 共用合同驱动输入装配与输出解码
├── data/
│   ├── __init__.py                    # 唯一 build_dataset 入口（dynamic_tse / offline_tse）
│   ├── interface.py                   # dynamic/offline 共同 Dataset 接口校验
│   ├── dynamic_dataset.py             # TSEDynamicMixingDataset（干扰支路 + 音乐噪声源 + enroll 仿真）
│   ├── scene_dataset.py               # TSE 多源场景 adapter（目标身份 / 缺席场景 / enrollment），复用 core 共享渲染
│   ├── offline_dataset.py             # TSEOfflineDataset（显式 WeSep manifest/cue schema）
│   ├── wesep_adapter.py               # 只读解析 manifest、路径、目标、mix 与 cue 索引
│   ├── spatial_cues.py                # spatial static/track typed cue 构造与帧对齐
│   ├── enrollment_augment.py          # 无状态 enrollment 增强实现
│   ├── batch.py                       # validated TSEBatch
│   └── collate.py                     # canonical dict item → TSEBatch
├── models/
│   ├── __init__.py                    # 两个模型的 lazy map + register_tse_models
│   ├── modular_spec.py                # TSE_Modular 纯配置解析与能力合同
│   ├── discriminative/
│   │   ├── tse_bsrnn_spk.py           # TSE_BSRNN_SPK（dict 谱进谱出）
│   │   └── tse_modular.py             # TSE_Modular（BSRNN speaker/spatial 组合）
│   ├── modules/                       # separator / cue frontend / fusion / norm
│   │   └── frontend_registry.py       # 四类 cue frontend 注册与惰性构造
│   └── speaker/
│       ├── config.py                  # speaker 配置类型与 checkpoint 路径解析
│       ├── ecapa.py                   # vendored ECAPA-TDNN（去 wespeaker 依赖）
│       ├── resnet.py                  # vendored WeSpeaker ResNet34
│       ├── fbank.py                   # 批量矢量化 kaldi-fbank
│       └── encoder.py                 # SpeakerEncoder（句级 + 帧级双输出，freeze 可配）
├── pretrained_models/                 # REAL-TSE 官方 TSE checkpoint（权重不进 git）
│   ├── README.md                      # 下载、变体与加载边界
│   └── {spk_emb_100, spk_emb_causal_100, tfmap_context_100, tfmap_context_causal_100}/
│                                      # 每个变体含 avg_model.pt + wesep config.yaml
├── speaker_embedding_models/          # 声纹 encoder 预训练模型（权重不进 git）
│   ├── README.md                      # 模型来源 / 下载地址 / 加载方式
│   ├── ecapa_tdnn_glob_c512/          # wespeaker VoxCeleb 预训练，embed_dim=192
│   └── resnet34/                      # WeSpeaker ResNet34-TSTP，embed_dim=256
├── inference/
│   ├── extractor.py                   # TSEExtractor（比赛 Extractor 同语义）
│   ├── adapter.py                     # TSEInferenceAdapter（公共 file inference 合同）
│   ├── root_manifest.py               # 根 inference.py 使用的严格 TSE JSONL 清单
│   ├── sidecar.py                     # 按公共 InferenceReport 发布 TSE sidecar
│   ├── artifacts.py                   # TSE sidecar 的 artifact 身份与校验
│   ├── streaming.py                   # waveform chunk STFT/WOLA 会话
│   └── torchscript.py                 # flat-cache TorchScript 发布
└── trainer/
    ├── entry.py                       # TSETrainingEntry
    └── tse_trainer.py                 # TSETrainer（MetricsCapableTrainer 模板）
```

相关框架侧文件：`configs/tse/dynamic/tse_bsrnn_spk.yaml`（示例配置）、`configs/tse/inference/*_infer.yaml`（官方 checkpoint 的推理配置）、`evaluate/eval_pipeline_tse.py`（REAL-TSE 评测流水线）、`tests/test_tse_*.py`（测试套件）。设计与分层说明见 [`docs/architecture/TSE_Architecture.md`](../../docs/architecture/TSE_Architecture.md)。

---

## 📦 数据流与数据准备

TSE 有两种拿数据的方式，由配置里的 `data.mode` 选择（`data.mode` 是声明数据来源的键，`train.py` 据此路由）：

- **在线动态仿真**（`data.mode: dynamic_tse`）：训练时现场合成 mixture——按概率抽取说话人与噪声、加 RIR（房间冲激响应，用来模拟房间混响）、标定 SIR/SNR（干扰说话人、噪声分别相对目标语音的能量比），再产出 canonical dict item。
- **离线 WeSep manifest**（`data.mode: offline_tse`，`TSEOfflineDataset`）：在 `samples_list` / `samples_jsonl` 中二选一，并显式配置 `manifest_format`、`path_base`、`target_selection`、`mix_variant`、`cue_schema` 和 `enabled_cues`；启用 cue 时另提供 `cues_yaml`。`manifest_format` 明确区分 JSONL 内容和 JSONL 路径列表，`path_base` 明确相对载体文件或 `recipe_root` 解析，runtime 不按后缀猜格式，也不搜索候选目录。offline 的 `spk_label` 用 manifest speaker id 建立连续映射，validation 复用 train label 空间。

两种模式只在 `tasks.tse.data.build_dataset` 内选择 Dataset 实现，输出和 collate 合同不分叉；跨任务入口由 `tasks.task_api.load_task_data_api` 根据 `tasks.descriptors.TaskDescriptor.data_module` 路由。TSE 的 `cues.<kind>.frontend` 是配置注册键，当前 canonical 值为 `speaker`、`spatial`、`visual`、`textual`；省略时由 `modular_spec` 填入同名 frontend。模型通过 `tasks/tse/models/modules/frontend_registry.py` 惰性构造 frontend，未知名称或 cue kind 不匹配会在构造期拒绝。

  WeSep Adapter 已能严格解析 audio、spatial、visual、textual 四类 cue 的索引与缺失状态；`TSEOfflineDataset` 已接通 spatial `npy/fields`（按 static/track 物化 typed `TSECue`，并校验字段顺序、弧度、array-local 坐标系、声速和 source-sample timebase）、visual `npy/muse_frontend`（预提取帧级特征，raw `mp4` 在构造期拒绝）与 textual `json/dae_keyword_phoneme`。spatial 已有本机 MC-LibriMix 数据集并完成 smoke 级真实数据闭环；visual 仍无已整理的数据集与真实训练闭环。通用 pseudo-target producer 为 `tools/prepare_data_tools/generators/tse/generate_realtse_pseudo_targets.py`（generator `2.0`）：正式输出会经过 GenerationManifest v2、生产 `TSEOfflineDataset(strict_load=True)` full reload、enrollment 全候选解码、receipt 与 owned publication transaction 几道检查。官方 REAL-T v1.0.0 是 evaluation-only benchmark，不放进 `configs/tse/offline/` 训练配置目录。

两种 Dataset 都返回同一个 canonical item：

```text
{
  "key": str,
  "mixture": [C,T], "sr": int, "length": int,
  "targets": [
    {"target": [T], "speaker_id": str | None, "spk_label": int,
     "target_present": bool, "cues": TSECues},
    ...
  ],
  "geometry": ..., "timebase": ..., "provenance": ...
}
```

`TSECues` 固定包含 speaker waveform、speaker embedding、spatial、visual、textual slots：未启用 slot 为 `None`；已启用但当前目标缺 cue 时必须使用 typed `TSECue(value=None, present=False)`，时序 cue 的 `length=0`。只有 collate 才把缺失行物化为零 tensor，并保留 `present=False`，不能用全零值猜测 cue 是否存在。`spk_label=-1` 表示未知说话人；`target_present=False` 与 cue 缺失是不同语义，前者要求目标监督为零。`DatasetContract` 的 typed spatial 字段与 `TSECueSpec` 同源：static 为 `[target,direction]`，track 为 `[target,direction,cue_point]`，track 另有 `[target]` 的 lengths 与 `[target,cue_point]` 的 sample-anchor timebase；geometry、mixture timebase 和 provenance 是 batch-size 对齐的 host metadata，不是 TensorField。

空间 observation 启用后，item 的 `mixture` 为 channel-first `[C,T]`；每个 target 与 speaker waveform cue 仍为单通道 `[T]`。动态模式的空间 RIR 可从当前 split 的 `rir_scp`（`rir_backend=offline`）读取，或由 `rir_backend=online` 现场生成；scene geometry 会生成 static/track per-target spatial cue，并把 geometry、scene timebase 与空间 provenance 放在 item 的 host metadata 中。该动态数据接线不等于所有模型的 spatial 前向、动态 cue 训练或 streaming 已完成。offline 模式使用 split-level `spatial_enabled`、`array_profile` / `custom_geometry`、`reference_channel` 和 `channel_order` 校验 observation，并从 NPY 资源写入 static/track spatial cue。普通 mono RIR 不会被伪装成 source-microphone RIR。`TSE_Modular` 的 BSRNN 路径实际支持 speaker/spatial 输入；其余 legacy speaker 模型收到 `C>1` 会直接报错，数据边界具备 microphone 轴不等于所有模型都支持空间前向。

批（collate）边界的规则：

- `tse_collate_fn` 只接收 canonical dict item，返回已执行 `validate()` 的 `TSEBatch`；Dataset item、collate 和 trainer 都不接受 tuple，也没有 `from_tuple` 回退。
- `TSEBatch.mixture` 固定为 `[B,C,T]`，`target` 固定为 `[R,T]`，`num_targets` 固定为 `[B]` int32，且 `R=sum(num_targets)`；target、speaker identity、label、presence 与全部 cue 按 sample-major 的 R 行对齐。
- 批内 sample rate、麦克风数和 geometry topology 固定；`lengths` 是 `[B]` int32，`spk_labels` 是 `[R]` int64，`target_present` 是 `[R]` bool。trainer 直接消费 `TSEBatch`，递归搬运 cue tensor。

动态仿真的 stems（仿真中各成分的波形）只在显式调用 `get_sample_with_trace(..., capture_stems=True)` 时捕获，不改变 canonical dict item。simulation audit envelope 的 `schema_version` 为 1、当前 `contract_version` 为 `1.4`：不请求捕获时只保留 sealed role-local metadata，stems capability 为 unavailable，reason 为 `capture_not_requested`；请求捕获时发布 `source-scene-stems-v2` / `stem-descriptors-v2` 摘要与实际 waveform bytes 的 SHA-256，不把 waveform 嵌进 JSON；另有 task-local stem capture。旧的 `source_image/N` 轨道命名不再接受。

可选增强（默认关闭）：

- **timeline**：单人布局复用 Layer A 的 gap / silence canvas；新多源配方用 `prob_timeline` / `overlap_ratio_range` 规划各源的相对窗口，并可用 `simulation_cfg.timeline_mode_weights` 为 `head` / `tail` / `middle` / `crossing` / `containment` 指定完整概率分布，默认五种模式均匀。旧配方已删除，零窗口产生零样点。
- **speaker-axis 多目标（省内存版）**：Dataset kwarg `targets_per_mixture > 1` 时，干扰说话人也成为可提取目标；仍使用同一个 canonical dict，仅在 `targets` 中增加 target dict，mixture 只存一份。`repeat_to_max_duration` 也支持多目标：mixture、主目标、noise 与全部备选目标共享同一 target-SR 与裁切/拼接画布计划；此时 Layer D 的 `D-normalize` 只允许 `off` / `joint`，`separate` / `single` 在构造期直接报错。若运行期 speaker waveform cue 构造失败，Dataset 会抛出不可恢复错误，拒绝发布稀疏的 `alternate_target/N` 或 `enrollment/N` role。
- **simulation capability**：Layer D 的 `peak_normalization_mode` / scale range 在 TSE 已接线；enrollment 不进入 scene bundle，因此不参与 Layer D 缩放。各层顺序与 Layer D/E/F 的 clipping ownership 见下一节；AEC 专属键（scenario / SER / farend / ERL）不得出现在 TSE 配置的 ownership 合同见 [`configs/README.md`](../../configs/README.md) 与 [`docs/guides/simulation/Simulation_Flow_Logic.md`](../../docs/guides/simulation/Simulation_Flow_Logic.md)。

### 数据清单格式

数据清单是「音频文件 + 元信息」的列表文件，TSE 用它们驱动采样与仿真。

身份规则集中在 [`manifest_contract.py`](manifest_contract.py)：Dataset 与资源生成器
共同调用 speaker 身份映射、nested enrollment 解析和 recording 身份覆盖校验；
该模块不加载 Dataset、采样器或训练运行时。

| 资源 | 动态 TSE 合同 |
|------|----------------|
| `speech_scp` | **仅 strict JSONL**；每行必须提供稳定 `uid`、`fs`、`length`、`path`、`language`、`dataset_id`、`speaker_id` |
| `data.train.spk2enroll_json` / `data.validation.spk2enroll_json` | **nested** `{dataset_id: {speaker_id: [path, ...]}}` |
| `music_scp` / `noise_scp` / `rir_scp` / `windnoise_scp` | 遵循 source manifest 合同：严格四列 SCP `uid fs length path` 或逐行 JSON object JSONL；JSON array 与遗留三列 SCP 直接报错。metadata（包括 `noise_scp` 的分类与角色）只由 JSONL 传递 |

同一原始录音存在多个文件或切片时，在 speech JSONL 增加非空、已 trim 的 `recording_id`，
使这些记录共享 `(dataset_id, recording_id)`。Dataset 为每个 mixture 排除其全部已用
speech/splice 路径及对应录音身份，enrollment 候选耗尽时明确报错。一个 dataset 中只要
有记录声明 `recording_id`，该 dataset 的全部 speech 记录和 enrollment 路径都必须在
speech JSONL 中具备身份；缺失或同一路径身份冲突会在构造期拒绝。未声明身份的资源
保持路径隔离，运行时不根据文件名猜测录音关系。

`prepare_librimix_dynamic_source.py` 的 generator v2 从已校验的 Libri2Mix
`s1/s2` 文件名提取原始 LibriSpeech utterance ID，筛除同录音的 enrollment，并去重候选。
`prepare_dynamic_tse_manifests.py` 接受并保留可选 `recording_id` / `generator_version`，
发布前执行相同完整性检查。2026-10-08 本机 `dynamic_tse/train-100` 的 27,800 条和
`dynamic_tse/dev` 的 6,000 条资源已重新生成并校验；dev 涉及 2,250 条原始录音。

噪声清单的准入与字段细节：Noise Filter Pipeline 负责准入，strict JSONL v6 是当前的发布/审计合同（admission policy v4、AudioTag evidence v5、noise source parser v10、generator `6.1`、promotion marker v4），但不是训练 runtime 的唯一输入格式。Dataset 只投影 `uid/fs/length/path/category_id/balance_category/category_name/roles` 并信任清单内容，发布清单按 `ROUND_HALF_UP(Decimal(str(duration)) × sr)` 生成 `length`。C1/C2/C3 的判定规则与证据字段见 [`tools/noise_filter_pipeline/README.md`](../../tools/noise_filter_pipeline/README.md) 和 [`docs/guides/simulation/Simulation_Flow_Logic.md`](../../docs/guides/simulation/Simulation_Flow_Logic.md)；`roles` 的推导规则与 Wind/Music 分池（属于训练路由，不是准入过滤）见下文采样小节。

路径合同分两层：

- **配置里的资源文件路径**：YAML 中实际提供的 `data.train.spk2enroll_json` / `data.validation.spk2enroll_json` 必须是 non-empty、已 trim 的字符串（首尾空白不会被自动去掉）。`data.mode: dynamic_tse` 必须提供非空的 `data.validation` mapping，并显式设置 `mode: dynamic` / `dynamic_tse` 或 `use_dynamic: true`。validation 的 `speech_scp` 只有在键缺失时才继承 train：显式 `null`、空串、纯空白、首尾空白或非字符串都按 `data.validation.speech_scp` 报错；validation 的 `simulation_cfg` 缺失或为 `null` 时按继承语义处理，显式给出 list/string/number 等非 Mapping 值则按 `data.validation.simulation_cfg` 报错。`data.mode: offline_tse` 不使用这套动态 validation 入口规则。
- **enrollment JSON 内部的音频路径**：nested enrollment 里每个 `path` 同样必须是 non-empty、已 trim 的字符串。

以上路径全部经 YAML 注入（允许提交路径字面量，数据本体不入版本控制）。与配置相关的另外几点：

- 唯一的 `configs/tse/dynamic/tse_bsrnn_spk.yaml` 由 `tests/test_configs.py` 按精确路径白名单识别为可分发训练模板；四种官方 cue/causal 组合通过该文件顶部的注释切换块手工选择，而不是通过 `meta.training_template` 或 runtime 模板标记。
- 静态配置发现有 `null` 占位资源是允许的，但 runtime `TSETrainingEntry` 一定会报错；训练前必须把这些路径替换成符合上面合同的实际路径。配置测试只 hydrate 三处已声明占位，其他字段错误不会被豁免。
- `configs/tse/offline/` 只保留许可允许的 WeSep/LibriMix offline recipe；REAL-T v1.0.0 只作 evaluation benchmark，不进入该目录。
- TSE Dataset builder 提升字段的权威全集是 `core.data.dataset_build.TSE_DATASET_KEYS`：`enrollment_duration` / `enrollment_db` / `enroll_policy` / `enrollment_augment` / `targets_per_mixture`。这些键写在 `simulation_cfg` 顶层，构建时提升为 Dataset 构造 kwargs；`spk2enroll_json` 和 `music_scp` 属于 data.train/data.validation 资源，不能写进 `simulation_cfg`。`enrollment_augment` 是 TSE-local 的无状态注册音增强配置，不进入 `SimulationConfigs`。
- validation 默认可以继承 train 的 nested enrollment，也可以用 split-local 资源覆盖；`inherit_simulation: false` 时必须在 validation 自己的 `data.validation.spk2enroll_json` 提供资源。

### 动态均衡采样（可选）

采样控制的是「每个 epoch 里各类数据被抽到多少次」。TSE dynamic 支持 `data.train.sampling.speech` 下三个互相独立的 controller（`language` / `dataset` / `speaker`，子块存在即启用），也支持 `data.train.sampling.noise` 的 category / temperature 均衡；schema 与 SE 相同，见 [`configs/se/README.md`](../../configs/se/README.md) 的采样小节。下面是 TSE 特有的规则（anchor 指某个样本被选中的主说话人语音，其余成分围绕它拼接）：

- **speaker absolute cap**（需要 canonical identity metadata）：

```yaml
sampling:
  speech:
    speaker:
      max_samples_per_epoch: 1000
```

  `max_samples_per_epoch` 是某个 canonical identity 在全局 epoch 的 primary anchor 中的绝对上限（正整数，不是概率、权重或精确 margin）。cap **只计 primary anchor**：interference、alternate target、enrollment 以及它们的音频秒数与梯度贡献都不计入。
- **canonical identity = `(dataset_id, speaker_id)`**：strict JSONL manifest 的 `dataset_id` / `speaker_id` 是主事实来源（与是否启用 sampling 无关）；不同 dataset 里相同的裸 speaker label 算不同 identity。
- **speaker-aware speech splice**：目标与每路干扰语音各自执行 Layer A splice（拼接补长）时，额外片段只从 anchor 的 canonical identity 对应 UID 里按 manifest 顺序抽取；同一个裸 speaker 跨 dataset 不合并。identity 缺失或是 singleton 时只允许复用 anchor。enrollment 走独立的注册音管线，不参与 speech splice。SE/AEC 仍使用 manifest-order 的全语音池。
- **strict nested enrollment**（`spk2enroll_json`）：

```json
{
  "dataset-a": {"speaker-001": ["/path/enroll-a.wav"]},
  "dataset-b": {"speaker-001": ["/path/enroll-b.wav"]}
}
```

  key 会归一为 tuple identity；flat speaker map 会被拒绝；未知 dataset/speaker key 报 coverage 错误；每条 path 必须是非空、已 trim 的字符串。
- **noise category / temperature**：`sampling.noise.balance_categories: true` 下可选 `balance_mode: category`（类别等权）或 `balance_mode: temperature`。Temperature 在候选 SR tier 内按 `P(c) ∝ n_c^alpha × difficulty` 选类，类内 UID 均匀且有放回；不存在 epoch 内 UID 重复的硬上限。代码默认 `alpha: 0.5`，`alpha: 1.0` 只是取消容量平滑，difficulty 仍影响曝光。
- **noise SR tier**：Natural 第一段按 `exact → 首个可用 higher SR → all` 降级；classified Natural 的第二段排除首类后按 `exact → 所有 higher SR 聚合 → all`，Category/Temperature 也用后者。
- **双 ordinary noise**：ordinary pool 的分类 metadata 完整时第二段必须是异类，找不到异类则退化为单噪声；Natural 只在 metadata 缺失或部分缺失时回退到普通 UID 二次读取，balanced metadata 不完整时直接报错。selector 派生缓存是双预算 LRU（16 entries / 65,536 UID references），超大的 Natural entry 使用零复制 view；它不是 UID 上限，也不改变 RNG。
- **Babble / interference**：Babble（如 speech babble/hubbub）属于 Layer C 的 additive noise 成分，经 `noise_scp` 与 SNR 路径采样；含 I 的七路场景则从 `speech_scp` 里选不同 canonical identity 的可分说话人、按 SIR 混入。两者不是同一条支路。
- **Wind / Music 路由**：Wind 在 TSE 从 ordinary pool 路由到 side-chain；Music 只在 TSE 从 ordinary pool 拆到 music pool。内嵌 role 分池会分别与显式 `windnoise_scp` / `music_scp` 合并扩池；Natural 模式下 `music_ratio > 0` / `prob_wind_noise_sidechain > 0` 也必须分别存在内嵌 role 或显式池。Music 按 **per source-read** 应用：单源、enrollment 与双源的第一/第二段分别判定；双源禁止 Music+Music，ordinary+ordinary 在分类完整时必须异类。只有 `music_ratio=1` 且 Music pool 非空时才允许 ordinary 为空，此时双源在 Music first 之后没有 ordinary second，会退化为单源。Wind 命中时，若 `noise_mix_prob` 未命中则由 Wind 替代普通 additive read；若命中双源，则 Wind 与一段 ordinary/Music additive noise 叠加。
- **validation**：train 的 `sampling` 不继承到 validation，构建 validation 时会剥离继承值；validation 自己显式配置有效 sampling 也会报错。label 空间由 entry 显式注入 train 的 `speaker_identity_to_idx`，只出现在 validation 的 identity 其 label 为 `-1`（CE 经 `ignore_index` 跳过）；validation 可继承 train 的 nested enrollment，`inherit_simulation: false` 时必须提供 val-local enrollment。

---

## 🔬 仿真管线

动态仿真的作用是把干净语音一步步变成「训练时模型看到的录音」。框架把这套处理拆成 Layer A–F 六层，五任务共用；本节只讲 TSE 独有的部分：干扰说话人、Music/Wind 路由、多目标 scene bundle 和 enrollment。

层的顺序、Layer D 的 audit schema、Layer E/F 各自的 clipping 归属和 offline 边界以 [`docs/guides/simulation/Simulation_Flow_Logic.md`](../../docs/guides/simulation/Simulation_Flow_Logic.md) 为准；TSE 在混合完成后的固定顺序是 `pre-finite → D-normalize → D-guard → E → F → task-post-EF → final-guard → post-finite`（Layer D 是 post-mix 幅度调理，内部固定 normalize 后 guard；`final-guard` 是独立的确定性兜底，只保护最终 `mixture`）。

当前动态模板使用共享的多源渲染：

```
场景选择（人数 / 目标是否存在）
→ 每个 identity 独立 Layer A（splice gap / silence canvas 只做一次）
→ 共享多人相对窗口
→ 各 source 的 full-RIR image 与 early/dry target
→ post-RIR SIR → 明确 reference 的背景 SNR
→ 共享 D–F 与 observation final guard
→ 独立 speaker waveform cue 配对及一次增强
→ canonical dict item（单目标与多目标同一 schema）
```

Layer B 用七路互斥的 T/I/N 场景（T = target 目标说话人，I = interference 干扰说话人，N = noise 背景噪声），`num_interferers` 只决定含 I 场景里的人数，身份不够就报错。目标缺席由 I-only / N-only / I+N 三路直接表达，并由 `AbsentTargetLoss` 监督模型输出零；不再使用 `prob_target_absent` 这类二次 gate。

`rir_backend=offline` 从当前 split 的 typed scene manifest 读取现成的静态 actor RIR；`rir_backend=online` 先在 core 里固定 room、绝对 microphone geometry 和 actor 位置/轨迹，再用受管理的 spawn 服务生成 full/direct tracks。enrollment 是独立的单通道近场增强，不属于 microphone actor scene。TSE 旧的动态函数、旧 timeline 和旧字段解析均已删除。TSE_BSRNN_SPK 当前是 mono-only；多通道 mixture 需要原生支持空间输入的 TSE 模型。

在 `spatial_enabled=false` 的 online moving scene 中，目标、干扰说话人和可选 latent SIR reference 共用一次采样得到的 room、mono microphone position 与 trajectory timebase；每个 speech actor 使用自己的轨迹。enrollment 仍走独立的单通道近场 RIR 增强，不并入 mixture scene。

字段与七路语义见 [多源共享仿真](../../docs/guides/simulation/Multi_Source_Simulation.md)，在线进程/随机性/性能限制见 [在线 gpuRIR](../../docs/guides/simulation/Online_gpuRIR.md)。

---

<a id="tse-bsrnn-spk"></a>
## 🤖 模型 TSE_BSRNN_SPK

`TSE_BSRNN_SPK` 是 BSRNN（BandSplit-RNN）风格的目标说话人提取模型：先把频谱切成 32 个子带（16 kHz、win512），每个子带用 SubbandNorm 归一化后过 6 组 BSNet，最后由 BandMasker 产生复数掩码乘回原谱（复数掩码同时调整幅度与相位）。它照 AEC 的先例改造成 **cue-aware dict 多输入、谱进谱出**接口——mixture 与 enrollment 是两路输入语义，但具体字段按 cue/joint 模式提供，并不是固定的「两个张量」或所有字段都必填：

```python
forward({"mix_spectra":    [B, F, T,  2],   # 必需：mono 混合谱（view_as_real 格式）
         "enroll_spectra": [B, F, Te, 2],   # 按需：listen/usef/tfmap
         "enroll_wav":     [B, Te_wav],     # 按需：joint context/spkemb
         "enroll_lengths": [B],              # joint waveform cue 的有效长度
         "spk_embedding":  [B, 192]})       # 按需：非 joint spkemb
  -> {"target_estimate_spectra": [B, F, T, 2]}   # multi_task=true 时另含
                                                 # "spk_logits": [B, spks_in_train]
```

形状说明：`B` 批大小、`F` 频点数、`T` / `Te` 分别是 mixture 与 enrollment 的帧数、末维 `2` 表示实部/虚部（`view_as_real` 布局）。

- **内部 STFT/iSTFT 已剥离**：谱参数由顶层 `stft` 块注入。`tasks/tse/model_io.py` 按实例的 cue-resolved I/O contract 构造输入：只有 listen/usef/tfmap 才计算 enrollment STFT，joint context/spkemb 传入 `enroll_wav + enroll_lengths`，非 joint spkemb 只传预提取 embedding。joint SpeakerEncoder 按有效长度分桶，避免 collate 补零污染统计池化或 context attention。四个官方 checkpoint 都已通过真实 REAL-T 波形的完整 `TSEExtractor` 构造、strict state、重复确定性前向和输出门禁；joint 官方推理以主 checkpoint 内含的 ECAPA 为唯一权重来源，不预加载独立 ECAPA。
- **5 种可插拔 speaker cue**（`model.args.features`，全部默认关闭）：`listen`（谱域前缀拼接）、`usef`（跨谱注意力）、`tfmap`（TF 注意力图）、`context`（帧级 embedding cross-attention，仅 joint 模式）、`spkemb`（句级 embedding + SpeakerFuseLayer 融合）。机制详解、causal 差异和比赛官方 4 个 checkpoint（spk_emb_100 / spk_emb_causal_100 / tfmap_context_100 / tfmap_context_causal_100）以 `configs/tse/dynamic/tse_bsrnn_spk.yaml` 顶部的注释切换块为准；四种训练组合统一由该文件表达，当前默认是 `spk_emb_100`。
- **布尔配置类型**：`model.args.joint_training` 与五种 cue 的 `features.*.enabled` 必须写成 YAML 原生 `true` / `false`；字符串、整数和 `null` 不会被当作真假静默解释，而是在入口或模型构造时直接报错。
- **multi-task speaker 分类头**（`model.args.multi_task: true`，默认关闭）：在句级 embedding（`spkemb` 的产物）后接一层 `nn.Linear(embed_dim, spks_in_train)`，forward 额外输出 `spk_logits`，供 `SpeakerCrossEntropyLoss`（说话人分类辅助任务，CE = 交叉熵）使用；需要先启用 `spkemb` cue。`spks_in_train` 可以显式配置，缺省时由 `TSETrainingEntry` 从 train dataset 的 `speaker_identity_to_idx` 大小自动注入（显式值与它冲突时报错）。
- **joint / 非 joint 双模式**（`model.args.joint_training`）：
  - `true`（默认，与比赛 checkpoint 对齐）：模型内含 vendored ECAPA-TDNN，直接从 `enroll_wav` 提取 embedding；
  - `false`：`context` 会被入口拒绝；`listen` / `usef` / `tfmap` 仍可用 `enroll_spectra`，`spkemb` 则由独立的冻结 `SpeakerEncoder` 从当前 batch 的 `enroll` 波形在线生成句级 embedding。这个 encoder 只用于 train/validation 特征提取，不进入 TSE 主 optimizer、DDP（DistributedDataParallel，多卡数据并行训练）或 `torch.compile`；必须在 YAML 中给出真实存在的 `model.args.speaker_model.speaker_encoder.pretrained`，禁止随机初始化。它的 checkpoint 内容身份会写进 TSE 训练 checkpoint，resume 时在首次前向前严格校验，文件内容变化或旧 checkpoint 缺少身份字段都会直接报错。
- 波形张量统一 `[B, T]`（已去掉原实现的 `[B,1,T]` 断言与 `[B,S,T]` 输出）。
- **SpeakerEncoder 插件**：当前内置 `ECAPA_TDNN_GLOB_c512`（192d）与 `ResNet34`（TSTP，256d），两者共用安全的 checkpoint loader。ResNet34 的 context 帧特征是 2560d。
- **冻结 encoder cache**：冻结且处于 `eval()` / no-grad 状态的句级 `SpeakerEncoder` 会按 waveform 内容的 SHA-256 使用有界缓存；训练态、可更新 encoder 和帧级输出不使用该缓存。可用 `embedding_cache_info()` 查看命中统计，必要时调用 `clear_embedding_cache()` 清理。
- **双 self-estimated objective**：同一个 step 先算原 enrollment 的估计，再把第一路输出 `detach()`（切断梯度）后当作 enrollment 算第二路，默认按 `0.4/0.6` 加权；validation 保持单路，且与随机 `trainer.ssa_prob` 互斥。canonical 配置入口是 `trainer.objectives.dual_self_estimated`，WeSep 双路 recipe 使用 `TSE_BSRNN_SPK + objective`。

---

## 🧠 模型清单

| 模型 | 说明 | 参考来源 |
|------|------|----------|
| **TSE_BSRNN_SPK** | REAL-TSE Challenge baseline 移植（5 种可插拔 speaker cue） | [wesep-real-tse](https://github.com/wenet-e2e/wesep) / REAL-TSE Challenge |
| **TSE_Modular** | BSRNN 模块化组合，支持 speaker、spatial、visual（预提取 features）与 textual cue 及其组合；当前为 full-sequence forward，不支持 streaming，也无 multi-task speaker 分类头 | [项目实现](models/discriminative/tse_modular.py) |

registry 当前包含以上两个模型身份。`TSE_BSRNN_SPK` 保留既有 causal streaming/ONNX/TorchScript 路径；`TSE_Modular` 的 frontend 已接线 speaker（`usef`/`tfmap`/`context`/`spkemb`）、spatial、visual（`representation=features`）与 textual，`listen`、raw video 与其他 backbone 仍由配置边界拒绝，`supports_multi_task=False`（无 speaker 分类头，配 `SpeakerCrossEntropyLoss` 会显式报错）。WeSep frame-embedding recipe 使用 `configs/tse/offline/tse_modular_speaker_tfmap_emb_wesep.yaml` 及 block fusion 变体；本机真实数据闭环覆盖 speaker、textual 与 spatial（MC-LibriMix CPU smoke 级，见「推理」节），动态 spatial cue 的端到端训练验收与部署发布仍未完成。

---

## 🚀 训练

最短路径是下面五步（示例配置默认启用 `spkemb` cue + joint training，loss 为 `CompositeLoss`（SI-SNR，尺度不变信噪比），与比赛 baseline 的训练目标一致）：

1. **准备数据清单**：当前模板已指向本机 Libri2Mix dynamic 清单。迁移到其他数据集时，先运行 `prepare_librimix_dynamic_source`（Libri2Mix）或准备同等字段的 source JSONL，再运行 `prepare_dynamic_tse_manifests` 生成 strict speech JSONL 与 nested enrollment（清单格式见 [数据清单格式](#数据清单格式)）。

   ```bash
   python -m tools.prepare_data_tools.generators.tse.prepare_librimix_dynamic_source \
       --dataset-root data/tse/spk/wesep/datasets/Libri2Mix/wav16k/min \
       --split train-100 \
       --output-source-jsonl data/tse/spk/wesep/datasets/Libri2Mix/dynamic_tse/train-100/source_records.jsonl
   python -m tools.prepare_data_tools.generators.tse.prepare_dynamic_tse_manifests \
       --source-jsonl data/tse/spk/wesep/datasets/Libri2Mix/dynamic_tse/train-100/source_records.jsonl \
       --speech-jsonl data/tse/spk/wesep/datasets/Libri2Mix/dynamic_tse/train-100/speech.jsonl \
       --spk2enroll-json data/tse/spk/wesep/datasets/Libri2Mix/dynamic_tse/train-100/spk2enroll.json
   ```
2. **放置 speaker encoder 权重**：按 [speaker embedding 预训练权重](#-speaker-embedding-预训练权重) 下载 ECAPA-TDNN 权重，否则 strict baseline 起不来。
3. **（可选）选 cue 组合**：在 `configs/tse/dynamic/tse_bsrnn_spk.yaml` 顶部注释块里挑一组复制到 `model.args.features`，同时改 `model.args.causal` 与 `trainer.exp_path`，一次只留一组。
4. **启动训练**：

   ```bash
   # 统一入口：detect_task 按 data.mode: dynamic_tse 自动路由到 TSETrainingEntry
   python train.py -C configs/tse/dynamic/tse_bsrnn_spk.yaml
   ```

5. **看产物**：checkpoint 与日志写在配置的 `trainer.exp_path` 下（默认 `exp/tse/dynamic/tse_bsrnn_spk/`，checkpoint 在其 `checkpoints/`，最佳模型是 `best_model.tar`）。

训练相关的实现约定：

- trainer 的 `_unpack_batch` 只接受 collate 已验证的 `TSEBatch`，tuple 会显式报错；设备搬运走 `TSEBatch.to()`，包括全部 typed cue tensor。要新增字段必须先进入 canonical item、`TSEBatch`、`validate()` 和实际消费链路。
- `distributed.find_unused_parameters` 由 `TSETrainingEntry` 按 cue 配置自动推导（当前已注册模型的合法配置下恒为 `false`），显式配置优先。这个结论依赖 multi-task 分类头始终进入 loss graph：Speaker CE 非零时，trainer 遇到缺少 `spk_labels` 的 batch 或缺少 `spk_logits` 的模型输出会立即报错；全 `-1` label 则返回与 logits 保持 autograd/DDP 连接的精确零。
- **多目标批训练**（`simulation_cfg.targets_per_mixture > 1`，经 `TSE_DATASET_KEYS` 提升为 Dataset kwarg）：trainer 对 `[B,C,T]` mixture 只算一次 STFT，按固定 `[B]` `num_targets` 用 `repeat_interleave` 展开，与 `[R,T]` target、speaker cue 和 `spk_labels` 逐行对齐，loss 在 `R=sum(num_targets)` 行上计算；GPU 增强只作用于原始 B 条 mixture（每个目标 stem 保持干净监督）。
- **SSA（Self-estimated Speech Augmentation，自估计语音增强）**（`trainer.ssa_prob`，默认 0.0，即关闭）：以该概率整批先用 `no_grad` 预前向得到自估计目标 est0，再把 est0 当作 enrollment（长度与 mixture 相同）做真正的训练前向；随机性由 `(meta.seed, epoch, batch_idx)` 派生，因此 DDP 各 rank 决策一致、可复现。只在训练路径生效；`joint_training: false` 时 embedding 由冻结的外部 `SpeakerEncoder` 在线生成，SSA 不适用，会 warning 一次后跳过。
- **multi-task speaker CE**：在 loss 的 `CompositeLoss.components` 里加入 `{type: SpeakerCrossEntropyLoss, weight: ...}` 后，`TSETrainingEntry` 会自动推导 `model.args.multi_task: true` 并注入 `spks_in_train`（取 dataset `speaker_identity_to_idx` 的大小）；trainer 把模型输出的 `spk_logits` 与 `batch.spk_labels` 经 kwargs 传给 CE 分量（照 AEC state detector 的接线范例），未知说话人（label=-1）由 `ignore_index` 跳过。
- **WeSep 参数/训练语义对齐 recipe**：`tse_bsrnn_spk_wesep_baseline.yaml` 对齐 3 秒 chunk、150 epochs、Adam `weight_decay=1e-4`、逐 step 的 `ExponentialDecrease` 与 dither，但不保证 processor/随机流 bit-exact；其余三个配置分别覆盖双路目标、BSRNN_Feats 和 block-level fusion。
- WeSep offline 资源必须通过显式 `manifest_format` / `path_base` / `target_selection` / `mix_variant` / `cue_schema` / `enabled_cues` 导入；当前训练 recipe 只启用可解码的 audio cue。若要把 speaker-level `audio.json` 转成 dynamic TSE 的 nested enrollment，可以运行：

  ```bash
  python -m tools.prepare_data_tools.generators.tse.import_wesep_enrollment \
      --cues-yaml data/.../cues.yaml \
      --dataset-id librimix \
      --speech-jsonl data/.../speech.jsonl \
      --output-json data/.../spk2enroll.json
  ```

  `mix_spk_id` 的 fixed-trial resource 不能转成全局 speaker pool，应继续用于 `offline_tse`。

### textual 配置（TSE_Modular + DAE-TSE keyword cue）

`configs/tse/textual/tse_modular_textual_keywords.yaml` 是 textual cue 的训练配置：`data.mode: offline_tse`，`enabled_cues: [textual]`，指向本机已整理的 WeSep v0.1 LibriMix-keywords（train-100 / dev 的 `samples.jsonl` + `cues.yaml`）。textual 资源是 `json/dae_keyword_phoneme`，`scope=speaker`、`policy=fixed`、`key=mix_spk_id`、`selection=random` with `word_range`；目标监督取 `raw.list` 的 `src[spk]`，因此该配置不需要 enrollment 音频索引。

```
python train.py -C configs/tse/textual/tse_modular_textual_keywords.yaml
```

model 为 `TSE_Modular`（`backbone=bsrnn` + `cues.textual.features.dae_kce`）。textual cue 由 `TextualFrontend`（KCE）消费 mixture 波形（16 kHz）与 phoneme id 计算句级条件并按 presence 融合，STFT/iSTFT 仍由 runtime 统一管理。官方 DAE-TSE `epoch_149.pt`、`model.yaml`、`data.yaml` 已下载到 `data/tse/spk/wesep/datasets/LibriMix-keywords/resources/kce/`，配置直接指向该权重。KCE 将官方训练 sidecar 映射到推理子集，并对全部 525 个 backbone 参数键执行 strict load。真实 LibriMix-keywords 一样本/一步 CPU 训练、validation、外层 checkpoint strict reload 和真实 dev textual cue 文件推理均已通过；这是集成 smoke，不代表完整 150 epoch 训练。该配置已确认 Params 为 40.470500M（separator 21.402400M + textual frontend 19.068100M），MACs/FLOPs 未测。

### spatial 配置（TSE_Modular + DOA 方向 cue）

`configs/tse/spatial/tse_modular_spatial_mclibrimix.yaml` 是 spatial cue 的训练配置：`data.mode: offline_tse`，`enabled_cues: [spatial]`，指向本机 MC-LibriMix（gpuRIR 仿真 4 通道 16 kHz，ReSpeaker 4-mic 几何）的 `wesep_index/reverb_early/` 索引（mix 取 reverb、target 取 early）。spatial 资源是 `npy/fields`（azimuth/elevation/distance，弧度制，array-local 坐标系）；目标监督由 `reference_channel=0` 把 4 通道 target 归约为单通道，因此该配置不需要 enrollment 音频索引。

```
python train.py -C configs/tse/spatial/tse_modular_spatial_mclibrimix.yaml
```

model 为 `TSE_Modular`（`backbone=bsrnn` + `cues.spatial.features` = IPD/CDF/SDF/ΔSTFT，`representation=static`）。spatial cue 由 `SpatialFrontend` 消费 runtime STFT 的 4 通道 mixture 谱与句级静态 DOA，谱特征拼入 separator 输入，STFT/iSTFT 仍由 runtime 统一管理。`n_channel`/`ref_channel`/`mic_positions_m`/`speed_of_sound_mps` 不在配置中重复声明，由 entry 从 dataset geometry（split 段的 `custom_geometry` 等键）自动注入，冲突显式报错；推理侧无 dataset，对应推理配置 `configs/tse/inference/tse_modular_spatial_mclibrimix_infer.yaml` 显式固化同一组几何值。真实 MC-LibriMix 一样本/一步 CPU 训练、validation、checkpoint strict reload 和真实 dev 4 通道 spatial cue 文件推理均已通过（smoke 细节见「推理」节）；这是集成 smoke，不代表完整 150 epoch 训练。该配置 Params 为 22.61M（CPU smoke 训练日志实测），MACs/FLOPs 未测。

---

## 🔍 推理

整句推理用 `tasks/tse/inference/extractor.py::TSEExtractor`，语义与比赛 `Extractor` 一致：内部重采样到模型采样率（默认 16k）、可选 silero-VAD 裁掉 enroll 的静音（`apply_vad`，未安装 silero-vad 时自动跳过）、输出峰值归一化到 0.9。加载是 fail-closed 的：主模型在复制任何参数前校验全部 key/shape/dtype，禁止部分加载；checkpoint 含 EMA（指数滑动平均权重）时先验证全部 `shadow_params` 的数量、类型、dtype 与 shape，再整体应用，不会因为依赖缺失或状态损坏就静默退回原始权重。模型配置里显式声明的 `pretrained` 路径必须存在，缺失即报错，不再静默置 None；joint 模式内部 SpeakerEncoder 的该字段由主 checkpoint 提供，构造时会被显式清空。

```python
from tasks.tse.inference import TSEExtractor

# use_train 训练 checkpoint：直接读取 checkpoint 内嵌的 config / resolved_config
ext = TSEExtractor(
    "exp/tse/dynamic/tse_bsrnn_spk/checkpoints/<run-id>/best_model.tar",
    device="cuda:0",
)
wav = ext.extract_speech("mix.wav", "enroll.wav")   # np.float32, 16k, 峰值 0.9

# 或直接从 PCM 推理
wav = ext.extract_speech_from_pcm(pcm_mix, sr_mix, pcm_enroll, sr_enroll)
```

`tasks/tse/inference/adapter.py::TSEInferenceAdapter` 是公共文件推理入口：它按模型 IO 合同声明的 cue role 读取 `FileInferenceItem.inputs` 中的文件，并把 `attributes['cues']` 里的 inline cue JSON 纳入同一条绑定路径，然后交给 extractor 完成模型 I/O。支持的 role 与来源格式：

| role | 文件 | inline JSON |
|---|---|---|
| `speaker_waveform` | 任意 torchaudio 音频 | — |
| `speaker_embedding` | `.npy` 1-D float | `{"values": [...]}` |
| `spatial` | `.npy` structured `{azimuth, elevation[, sample_anchor]}`（static 也接受 `[P,2]`） | static `{"azimuth": a, "elevation": e}`；track 追加 `"sample_anchor": [...]` |
| `visual` | `.npy` structured `{features, timebase}` 或 `.mp4` raw video（需可选 torchvision.io 后端） | `{"features": [[...]], "timebase": [...]}` |
| `textual` | `.json` | `{"phoneme_ids": [...]}` |

spatial cue 的方向角约定（唯一定义见 `tasks/tse/data/spatial_cues.py::azimuth_elevation_from_positions`）：`azimuth = atan2(y, x)` 弧度（数学惯例，array-local 坐标系），`elevation` 从水平面起算、向上为正（弧度）。注意与 SSL 内部约定的区别：SSL（`tasks/ssl/data/projection.py::_doa`、`tasks/ssl/models/ipdnet.py::DPIPD`）使用从 +Z 起算的极角（0..π），两者相差 `π/2 - 极角`，跨任务消费方向角必须显式换算，禁止直接互换。

模型声明的 role 缺失、提供未声明的 role、或 representation 与合同不符都在 adapter 边界显式报错；`enrollment`/`enroll` 旧别名与 mixture+enrollment 双输入入口已移除，所有 cue 必须按 canonical role 提供，extractor 未声明 `declared_cue_roles` 时在 adapter 构造期失败。cue 文件、inline cue JSON 与 speaker 前端身份（checkpoint/fbank/preprocess）由 `tasks/tse/inference/artifacts.py` 绑定进 UID/sidecar/manifest（`tse-inference-artifact-v6`、`tse-inference-manifest-v4`），文件内容变化会让公共 runner 拒绝复用旧输出。streaming 与 ONNX 目前只覆盖 speaker 条件（`STREAMING_SUPPORTED_CUE_ROLES`），spatial/visual/textual 不进入这两条发布路径。

根目录 `inference.py` 也支持 TSE 文件推理。使用 `--tse-manifest` 传入严格 JSONL 清单，每行至少包含 `uid` 与 `mixture`，文件 cue 使用模型声明的 canonical role（例如 `speaker_waveform`），inline cue 放在 `attributes.cues`；相对路径按清单所在目录解析。根入口用同一个 `TSEExtractor` 和 `TSEInferenceAdapter`，公共 runner 统一负责 `run_manifest.json`、`inf.scp`、输出锁和 `--skip-existing`，随后由 `sidecar.py` 发布 TSE metadata：

```bash
python inference.py \
  -C configs/tse/inference/tse_bsrnn_spk_realt_baseline_infer.yaml \
  --checkpoint tasks/tse/pretrained_models/spk_emb_100/avg_model.pt \
  --tse-manifest data/tse/infer.jsonl \
  --output_dir results/tse
```

textual cue 的真实数据闭环已验证：train-100/dev/test 共 19,900 条 mixture 样本及对应 textual cue 可由 `TSEOfflineDataset` 读取；项目训练入口用真实 train mixture/target/cue 完成 1 epoch、1 optimizer step 和 validation，随后 `TSEExtractor` 严格重载保存的训练 checkpoint，并用真实 dev mixture 与 phoneme id 推理，输出为 16,000 样点且全部有限。此次 smoke 使用 1 秒裁剪和单条训练/验证样本，证明链路可运行，不代表完整训练收敛或正式评测。

spatial cue 的真实数据闭环已验证（MC-LibriMix，`wesep_index/reverb_early`）：train-100 13,900 / dev 3,000 / test 3,000 条 4 通道 mixture 及结构化 NPY spatial cue（azimuth/elevation/distance，弧度制）可由 `TSEOfflineDataset` 读取；项目训练入口（`configs/tse/spatial/tse_modular_spatial_mclibrimix.yaml`，geometry 由 entry 从 dataset 自动注入）用真实 train 4 通道 mixture（reverb）/单通道 target（early，reference_channel 归约）/spatial cue 完成 1 epoch、1 optimizer step 和 validation，保存的 model_0001.pth/best_model.tar/latest_model.tar 均通过 strict 重载（0 missing / 0 unexpected）；推理侧（`configs/tse/inference/tse_modular_spatial_mclibrimix_infer.yaml`，geometry 显式固化）用真实 dev 4 通道 mixture 与 spatial NPY 完成文件推理，输出 62,720 样点（16 kHz）且全部有限。此次 smoke 使用 1 秒裁剪和单条训练/验证样本，证明链路可运行，不代表完整训练收敛或正式评测。visual 角色虽已接入 adapter 与 artifact 绑定，但因本机没有已整理的数据集，没有对应的真实文件推理记录。


第一个参数可以给 checkpoint 文件，也可以给一个目录（`model_dir` 模式，此时目录里要有 `avg_model.pt`，配置默认取同目录的 `config.yaml`）。

官方解压目录自带的是 wesep 平铺配置，仍不能直接作为 use_train 的 `model_dir`。四个官方 checkpoint 分别使用 `tse_bsrnn_spk_realt_baseline_infer.yaml`、`tse_bsrnn_spk_realt_baseline_causal_infer.yaml`、`tse_bsrnn_spk_realt_tfmap_context_infer.yaml`、`tse_bsrnn_spk_realt_tfmap_context_causal_infer.yaml`（都在 `configs/tse/inference/`）；可以运行 `python tools/verify/verify_tse_official_ckpts.py --device cpu` 用真实波形复核端到端加载。

causal（因果）模型另外提供 waveform chunk 会话、ONNX/ORT 与 TorchScript flat-cache runtime。ONNX 是跨框架的推理格式，ORT 是它的运行时；TorchScript 是 PyTorch 自带的导出格式：

```bash
# 正式 ONNX 只能通过专属三路 verifier 发布
python onnx/verify/verify_3way_tse.py \
  --config configs/tse/inference/tse_bsrnn_spk_realt_baseline_causal_infer.yaml \
  --checkpoint tasks/tse/pretrained_models/spk_emb_causal_100/avg_model.pt \
  --save-onnx onnx/onnx_model/tse_bsrnn_spk_causal_streaming.onnx

# TorchScript 使用相同的 positional flat-cache ABI，并写 .meta.json sidecar；
# 默认 trace batch size 为 1，可用 --batch-size 固定成其他 batch
python tools/verify/export_tse_torchscript.py \
  --config configs/tse/inference/tse_bsrnn_spk_realt_baseline_causal_infer.yaml \
  --checkpoint tasks/tse/pretrained_models/spk_emb_causal_100/avg_model.pt \
  --output runs/tse_bsrnn_spk_causal_streaming.pt
```

waveform runtime 使用与训练一致的 centered Hann STFT、固定 enrollment condition、逐帧 state 和 WOLA（overlap-add 重建）；要换目标说话人必须显式 reset。

---

## 📊 评测（REAL-TSE）

`evaluate/eval_pipeline_tse.py` 读取比赛的 `*_meta.csv`（必含 `mixture_utterance` 与 `enrolment_speakers_utterance` 两列），逐条跑推理再算指标。`--audio_root` 指向 REAL-T split 根目录，runtime 会自动在官方布局下搜索 `mixtures/` 与 `enrolment_speakers/`，不需要不存在的扁平 `wav/` 目录，也不需要额外 mapping：

```bash
python -m evaluate.eval_pipeline_tse \
    --meta data/tse/REAL-TSE/REAL-T-dev/DEV/AISHELL-4_meta.csv \
    --audio_root data/tse/REAL-TSE/REAL-T-dev \
    --model exp/tse/dynamic/tse_bsrnn_spk/checkpoints/<run-id>/best_model.tar \
    --output_dir exp/eval/tse/tse_bsrnn_spk \
    --metrics spksim,dnsmos,timing,ter \
    --timing_gt_json_base_dir data/tse/REAL-TSE/REAL-T-dev/DEV/json \
    --ter_model_dir_zh third_party_scripts/zipformer/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2 \
    --ter_model_dir_en third_party_scripts/zipformer/sherpa-onnx-zipformer-gigaspeech-2023-12-12
```

`--metrics` 可以单选或组合。输出写在 `--output_dir` 下：逐条明细 `tse_eval_results.csv`、均值汇总 `tse_eval_summary.json`，以及官方汇总脚本能直接读的 `eval_metrics/<目录名>_*.csv`（只导出本次实际算过的指标），可用 `python -m evaluate.aggregate_eval_summary --output_dir <目录>` 生成按 dataset/language/overall 分组的报表。

模型声明了 speaker 之外的 cue role 时，评测从 meta CSV 的可选列读取对应资源：`speaker_embedding_cue`、`spatial_cue`、`visual_cue`、`textual_cue`（`speaker_waveform` 复用 `enrolment_speakers_utterance`），相对路径按 `--cue_root`（缺省回退 `--audio_root` 及其 `cues/` 子目录）解析。缺列或缺真实文件时该行会以明确原因失败，不会静默跳过或使用占位。

| 指标 | 说明 |
|------|------|
| `spksim` | 说话人相似度（spk cosine similarity，余弦相似度）：提取结果与注册音的说话人向量越接近越好。**官方口径**：wespeakerruntime ResNet34-LM ONNX，en=voxceleb / chs=cnceleb；报 enroll↔tse 与 enroll↔mixture baseline 两组余弦的均值。`--spksim_backend ecapa` 可退回 vendored ECAPA-TDNN 后备 |
| `dnsmos` | DNSMOS P.835（SIG/BAK/OVRL 三项无参考主观分）+ **P808**（官方口径：`sig_bak_ovr.onnx` 非个性化 polyfit + `model_v8.onnx`，模型位于 `third_party_scripts/DNSMOS/DNSMOS`） |
| `timing` | TSE timing（时间对齐）：模型输出的语音段与目标说话人真实重叠段是否对得上，报 10ms 帧级 micro precision/recall/F1。**官方口径**：提供 `--timing_gt_json_base_dir`（内含 `<dataset>/overlap_records.json`）后，GT label = 目标说话人 overlap 段（collar=0.05s，即边界附近 0.05s 的容差区），hyp = FireRedVAD 对 tse 输出判定的语音段；FireRedVAD 缺失时降级到 silero-vad 并在日志中注明。没有 GT json 时退回 mixture-VAD 近似口径，非官方 |
| `ter` | TER（Token Error Rate，文本错误率；中文按 CER、英文按 WER），用 sherpa-onnx Zipformer transducer 转写后比对；文本归一化与 meeteval 错误率均对齐官方 `utils/asr_metrics.py` / `calculate_TER.py` |
| `sisdr` / `pesq` / `stoi` | 常规有参考指标（需要有干净参考）：SI-SDR 是尺度不变信噪失真比，PESQ / STOI 是语音质量与可懂度指标。需 meta 含 `target_utterance` 列（或用 `--ref_root` 提供参考目录） |

以上前四项与 REAL-TSE-Challenge 官方实现（`utils/spk_similarity_eval.py`、`utils/dnsmos_eval.py`、`utils/tse_timing_evaluation.py`（含 FireRedVAD/GT label 准备脚本）、`utils/calculate_TER.py`）逐项对齐，对齐依据见 `evaluate/eval_pipeline_tse.py` 的模块 docstring。

评测 artifact 额外绑定了 REAL-T release README、split、全部 meta、显式 mapping、实际使用的 overlap JSON 与 EVAL2 `scene_channel.csv` 的内容身份，任一文件变化都会让旧的推理产物失配。逐样本 timing 可视化用 `tools/verify/plot_tse_timing_timeline.py`。

> **spksim 依赖**：`pip install wespeakerruntime`；模型约定放在 `third_party_scripts/wespeaker/{en,chs}/model.onnx`（不进 git，下载方式见该目录 README.md），缺失时按官方 Hub 机制自动下载到 `~/.wespeaker/`。语种按 meta 的 `language` 列判断（zh/ch→chs），缺失时按 `--dataset_lang_overrides`（默认 `AISHELL-4:chs,AliMeeting:chs`，与官方一致）再退回 `--wespeaker_lang`。

> **timing 官方口径依赖**：FireRedASR2S 仓库（浅克隆到 `third_party_scripts/FireRedASR2S/`，整个目录不进 git）+ VAD 模型 `pretrained_models/FireRedVAD/VAD`（`huggingface-cli download FireRedTeam/FireRedVAD --include "VAD/*"`），另外需要 GT overlap json（官方数据集自带）。

> **TER 为可选依赖（决策 D3）**：未安装 sherpa-onnx 时跳过 TER 并 warning，其余指标照常。可选安装：`pip install sherpa-onnx whisper-normalizer opencc-python-reimplemented meeteval`；另需 Zipformer 模型文件（encoder/decoder/joiner ONNX + tokens.txt），经 `--ter_model_dir`（或按语言的 `--ter_model_dir_zh/en`）注入。本仓库约定模型放在 `third_party_scripts/zipformer/`（模型文件不进 git，下载方式见该目录 README.md）。

常规开发指标（SI-SDR / PESQ / ESTOI 等）走 core val metrics（YAML 的 `trainer.val_metrics` 配置，与 SE/AEC 一致）。

---

## 🔑 speaker embedding 预训练权重

训练（joint 模式）与评测（spksim 的 `--spksim_backend ecapa` 后备）共用同一份 ECAPA-TDNN 权重，放在 `tasks/tse/speaker_embedding_models/` 下。**权重文件不进 git**（`.gitignore` 已忽略），需要自行下载放置：

```
tasks/tse/speaker_embedding_models/
├── ecapa_tdnn_glob_c512/       # ECAPA-TDNN，192d
│   └── avg_model.pt
└── resnet34/                   # WeSpeaker ResNet34-TSTP，256d
    └── avg_model.pt
```

下载地址、校验与加载方式详见 [`tasks/tse/speaker_embedding_models/README.md`](speaker_embedding_models/README.md)。在 YAML 里通过 `model.args.speaker_model.speaker_encoder.pretrained` 注入路径；`freeze: null` 表示默认行为「有预训练就冻结」。

未冻结时（`pretrained: null` 或 `freeze: false`）joint 训练的内置 encoder 会在 `model.train()` 下进入 train 模式：ECAPA 句级 head 的 `BatchNorm1d` 作用在池化后的 2-D `(B, dim)` 上，要求每个通道至少 2 个样本，因此 per-rank `data.dataloader.batch_size` 必须 >= 2（`TSETrainingEntry` 在配置边界 fail-fast，早于 dataloader 构建）。句级 cue 按 enrollment 真实长度分桶编码，batch 内长度互不相同时每个长度都落成单行组：该组临时改用 BatchNorm running statistics（`tasks/tse/models/speaker/encoder.py::sentence_batch_statistics_off`），每行 embedding 仍只依赖自身（保持逐行不变），不会触发 `Expected more than 1 value per channel when training`；单行组不更新 running stats，多行同长度分组仍用 batch statistics 并更新 running stats。冻结 encoder、`joint_training: false` 的预提取 embedding、只启用帧级 cue（`context` / `tfmap`）以及推理/评估（eval 模式用 running stats）都不受此限制。

---

<p align="center">
  <sub>最后更新: 2026-09-23</sub>
</p>
