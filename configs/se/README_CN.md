# 🎧 语音增强配置 (`configs/se`)

<p align="center">
  <strong>SE 模型训练配置集合 — 覆盖 VoiceBank-DEMAND、DNS Challenge 3 和 URGENT2026</strong>
</p>

---

语音增强（Speech Enhancement，SE）从一段带噪录音里去掉噪声，输出同一说话人的干净版本；部分 recipe 还处理混响或通道损伤。每个子目录对应一个数据集或比赛，所有 recipe 的启动方式相同：

```bash
python train.py -C configs/se/<子目录>/<配置文件名>.yaml
```

训练产物写入该配置的 `trainer.exp_path`（约定 `exp/se/<配置文件名去 .yaml>`），因此不同 recipe 不会共用同一个 run 目录。

## 📑 目录

- [术语速查](#-术语速查)
- [配置概览](#-配置概览)
- [子目录文档](#-子目录文档)
- [Validation 仿真继承策略](#inheritance-semantics)
- [动态均衡采样（可选）](#balanced-sampling)
- [快速入口](#-快速入口)

---

## 🔤 术语速查

| 术语 | 一句话解释 |
|---|---|
| recipe / 配置 | 一个 YAML 文件，描述一次完整训练 |
| `data.mode` | 决定由哪个任务和 Dataset builder 读这份配置的路由字段。全仓共九种合法取值，SE 使用 `dynamic` 与 `offline` |
| `offline` | 带噪输入已在磁盘上预混好，需要提供 clean/noisy 两份 SCP |
| `dynamic` | 训练时用干净语音 + 噪声/RIR 清单在线合成带噪输入 |
| A–F 层 | dynamic recipe 依次执行的六个仿真层：语音预处理、场景分支、声学混合、post-mix 幅度调理、信道退化、设备退化。完整合同见 [`Simulation_Flow_Logic.md`](../../docs/guides/simulation/Simulation_Flow_Logic.md) |
| SCP | 以空格分列的清单文件，每行四列 `uid fs length path` |
| 独立 recipe | 仓库内每份 YAML 都直接声明完整的模型、数据、loss、optimizer、scheduler、trainer 与 distributed section |
| STFT | 短时傅里叶变换；把波形切成帧并变成频谱，模型看的通常就是它 |
| SI-SDR / PESQ / STOI / ESTOI | validation 阶段对有干净参考的音频计算的侵入式客观指标，数值越高越好 |
| DNSMOS / NISQA / UTMOS / SCOREQ | MOS 类质量预测器；Trainer 不计算它们，需按 [`evaluate/`](../../evaluate/README.md) 离线运行 |
| DDP | 多卡数据并行训练 |

---

## 📁 配置概览

| 目录 | 说明 |
|------|------|
| **urgent2026/** | 🏆 URGENT2026：BSRNN（非因果/因果）+ DPDFNet2 离线/动态；含 pase/（PASE/StuPASE/UniPASE/GAP-URGENet）与 gan/（MetricGAN 系列）子目录 |
| **dns_challenge3/** | 🌐 DNS Challenge 3（当前提供 `gtcrn_causal_offline.yaml`） |
| **voicebank_demand/** | 🎤 VoiceBank-DEMAND 上的十份 recipe：GTCRN、ZipEnhancer-S、FastULCNet、MeanFlowSE、MeanSE、DCCRN、MTFAANet、RT-SEMamba（KD8 与 KD1 蒸馏）与 TFSkiMNet |
| **spatial/** | 🛰 8 kHz 的 NBSS SpatialNet / OnlineSpatialNet 论文对齐配方：一份离线对照 + 三份在线流式变体（`mhsa`/`ret`/`mamba`）及对应的 32 s LF 阶段配置（MHSA 另有 36 epoch 变体） |

数据准备脚本位于 `tools/prepare_data_tools/`。`model.type` 到实现模块的完整映射以 `tasks/se/models/__init__.py::_MODEL_REGISTRY_MAP` 为准；本目录仅列出随仓库提供的示例 YAML。

> **复现边界**：RWSA-MambaUNet 固定上游 commit `02635ab7e92a0584fde729f8a55f398c2379da82`，但尚未验证官方 checkpoint 或论文指标。FastULCNet 已接入 CRM/Comfi 路径及 `tools/verify/verify_fast_ulcnet_state_drift.py`；共享标准 Hann STFT 不等同于官方 `hann75`/WOLA，尚无 checkpoint 或真实 90 秒质量复现。MeanFlowSE 是带 complex STFT、MeanFlow objective 与 finite-difference JVP 的纯 PyTorch NCSN++ 风格项目实现，不兼容官方 checkpoint，且尚未训练复现论文指标。
>
> **统一口径**：模型参数量/MACs 及未测项以 [`docs/architecture/SE_Architecture.md`](../../docs/architecture/SE_Architecture.md) 和 `tools/model_tools/audit_metrics.py` 为准。THOP 结果只能标为 partial/unverified；独立 FLOPs 未测量时写 `not measured`。DNSMOS/NISQA/UTMOS/SCOREQ 等 MOS 指标不在 Trainer validation 中计算，统一按 [`evaluate/README.md`](../../evaluate/README.md) 使用离线评估流水线。

---

## 📖 子目录文档

| 数据集 | 英文文档 | 中文文档 |
|--------|----------|----------|
| VoiceBank-DEMAND | [README.md](voicebank_demand/README.md) | [README_CN.md](voicebank_demand/README_CN.md) |
| DNS Challenge 3 | [README.md](dns_challenge3/README.md) | [README_CN.md](dns_challenge3/README_CN.md) |
| URGENT2026 | [README.md](urgent2026/README.md) | [README_CN.md](urgent2026/README_CN.md) |
| PASE / StuPASE | [README.md](urgent2026/pase/README.md) | — |
| UniPASE | [README.md](urgent2026/pase/unipase/README.md) | — |
| GAP-URGENet | [README.md](urgent2026/pase/gap_urgenet/README.md) | — |

### 数据 SCP 索引

- **VoiceBank offline**：监督训练/验证使用 `voicebank_{clean,noisy}_{train,val}.scp` 四份清单；详见 [VoiceBank-DEMAND 数据准备](voicebank_demand/README_CN.md#2-数据准备)。
- **URGENT offline / dynamic**：offline 训练/验证使用配对的 `urgent26_{clean,noisy}_{train,val}.scp`；dynamic 训练/验证使用 `urgent26_clean_{train,val}.scp` 作为 `speech_scp`，并使用独立的 noise/RIR 清单。详见 [URGENT2026 数据准备](urgent2026/README_CN.md#-数据准备)。
- **DNS synthetic protocol**：`generate_dns3_offline_dataset` 为论文协议复现物化配对的 train/val/test 波形及严格四列 `clean.scp` / `noisy.scp`；官方 DNS V2/V3/V5 challenge 集仍是 noisy-only 评估输入，不能进入 supervised train 或 Trainer validation。详见 [DNS Challenge 3](dns_challenge3/README_CN.md)。
- **URGENT nonblind**：clean/noisy SCP 仅用于评估；AEC `experiment_report` 默认使用该 nonblind 成对清单，并通过 `--urgent_noisy_scp` / `--urgent_clean_scp` 传给 AEC pipeline，以 zero-farend 方式评估；目录参数仍不适用于嵌套的 nonblind FLAC 布局。详见 [URGENT2026 nonblind 说明](urgent2026/README_CN.md#离线配对-scp)。

---

<a id="inheritance-semantics"></a>
## 独立配置与 validation 仿真继承

仓库内配置不使用文件级继承；剩余的继承只发生在同一份完整配置的 validation 数据构建阶段：

- `data.validation.inherit_simulation` 是**同一份已合并配置内的 dynamic validation 构建策略**，只在 dynamic validation 生效。默认 `true` 时，validation 沿用 `data.train` 的仿真参数，只允许覆盖 `speech_scp`、`noise_scp`、`rir_scp`、`windnoise_scp` 等数据源键；设为 `false` 时，validation 可覆盖仿真参数，未写字段仍从 train 继承，`simulation_cfg` 按 nested dict 递归合并。
- `inherit_simulation` 不加载其他 YAML。offline validation 直接由 `clean_scp` / `noisy_scp` 构造，不使用 dynamic simulation inheritance。
- dynamic validation 会剥离从 train 继承的 balanced-sampling controller；validation 显式配置有效 `sampling` controller 会报告 capability error。

---

<a id="balanced-sampling"></a>
## ⚖️ 动态均衡采样（可选）

SE/AEC dynamic 训练通过 `data.train.sampling` 启用动态均衡采样。speech 与 noise 相互独立；子块存在即启用对应 controller。配置示例：

```yaml
# language-only：只控制语言比例（全局 exact margin）
sampling:
  speech:
    language:
      probs: {zh: 0.5, en: 0.4, other: 0.1}

# dataset-only：全局 dataset 池；alpha/max_ratio 省略时分别取 0.5/0.3
sampling:
  speech:
    dataset:
      alpha: 0.5
      max_ratio: 0.3

# 联合：language 与全局 dataset 两组 exact margin 在
# (language, dataset_id) transportation 上联合精确满足
sampling:
  speech:
    language:
      probs: {zh: 0.5, en: 0.4, other: 0.1}
    dataset:
      alpha: 0.5
      max_ratio: 0.3

# noise 类别均衡（与 speech 独立 opt-in）
sampling:
  noise:
    balance_categories: true
```

- `language.probs` 只接受 `zh`、`en` 和 `other`，正概率之和必须为 1；每个候选 `actual` 都重新执行 Hamilton 取整。
- dataset 是全局 exact margin：同名 `dataset_id` 跨语言、跨采样率合并；权重来自过滤后的有效时长；单 dataset cap 为 `min(dataset_capacity, ceil(max_ratio * actual))`。
- speech strict JSONL 为 eligible 音频提供 `language` 和 `dataset_id`。manifest 在构造期读取，epoch 内 UID 无放回，DDP 不通过 padding 复制 balanced UID。
- `speech.speaker` 只用于 TSE；SE/AEC 配置该子块时报告 capability error。
- `noise.balance_categories` 只影响 ordinary additive noise，并与 additive `reuse_noise` 互斥；wind 和 wind reuse 不变。SE/AEC/TSE 均支持 Category / Temperature controller。
- selector 派生缓存采用双预算 LRU（16 entries / 65,536 UID references）；超大 Natural entry 使用零复制 view。缓存不是 UID hard cap，也不改变 RNG。
- **噪声采样与角色路由的完整合同**（Natural / Category / Temperature 的 SR tier、双噪声异类规则、`roles` 推导、Wind/Music 分流、发布侧 strict JSONL v6 与 admission policy v4 / AudioTag evidence v5 / noise source parser v10 / generator `6.1` / promotion marker v4）统一维护在 [仿真流程逻辑 §0.1.1](../../docs/guides/simulation/Simulation_Flow_Logic.md)，配置侧对照表见 [configs/README.md 六种训练噪声策略对比](../README.md#六种训练噪声策略对比)。配置入口速记：`sampling.noise.balance_categories` 打开 Controller，`balance_mode: category|temperature` 选择类间等权或 Temperature，`alpha` 默认 `0.5` 且仅在 Temperature 下生效；旧键 `prob_wind_noise_additive` 与 `allow_unverified_noise_manifest` 已删除，遗留配置会 fail-fast。
- **TSE 音乐来源**：Music 只在 TSE 从 ordinary pool 路由到 music pool；SE/AEC 仍将 Music 与乐器类视为 ordinary additive noise。`music_scp` 可省略并与 metadata 中的 `music` role 合并；`music_ratio > 0` 但合并后的 music pool 为空时报错。

`tools/prepare_data_tools/generators/noise/build_stage_d_noise_manifests.py` 的 split 模式发布 train/val
JSONL 与 `noise_publication_metadata.json` 三件套；`category_id`、`category_name` 与 `roles`
已内嵌每条 JSONL，不再生成重复的 `noise_categories.json` / `noise_category_roles.json`
sidecar。metadata 用于构建 provenance、内容完整性与顶层 `manifest_summary` 审计，不是训练
输入；摘要不代表 runtime 均衡。训练实际 category/temperature 分布使用
`tools/verify/audit_noise_sampling.py` 审计，README 不再硬编码数量快照。

- AEC speech 只平衡 near-end `speech_scp` anchor；far-end、echo、wind 和 RIR 不受影响。
- validation 构建时主动剥离继承的动态均衡 controller；validation 显式配置有效 controller
  时报告 capability error。

---

## 🔗 快速入口

| 想要... | 去这里 |
|---------|--------|
| 复现 GTCRN 论文结果 | [voicebank_demand/](voicebank_demand/) |
| 跑 DNS Challenge 3 盲测 | [dns_challenge3/](dns_challenge3/) |
| 训练 URGENT2026 baseline | [urgent2026/](urgent2026/) |
| 训练 PASE / StuPASE 生成式增强 | [urgent2026/pase/](urgent2026/pase/) |
| 训练 MetricGAN 对抗增强 | [urgent2026/gan/](urgent2026/gan/) |
| 训练多通道 / moving speaker 的 NBSS 配方 | [spatial/](spatial/) |

---

<p align="center">
  <sub>最后更新：2026-09-21</sub>
</p>
