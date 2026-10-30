# 评估指南

本文是评估链路的导航层：一键评测流水线在哪、指标实现分几层、第三方指标怎么接、评估资产要守什么纪律。各流水线的完整参数以 `--help` 与模块 docstring 为 SSOT，本文不复制。

## 一键评测流水线

`evaluate/` 下按任务各有一条「推理 + 指标」一键流水线，模型侧用 `--exp_dir`（自动找 config + checkpoint）或 `-C` + `--checkpoint` 指定：

| 流水线 | 覆盖 | 说明 |
|---|---|---|
| [`eval_pipeline_se.py`](../../../evaluate/eval_pipeline_se.py) | SE | 多测试集一键推理 + 指标；`--datasets` 可选内置测试集，默认含 DNS 类测试集，`--include_aec_sim` / `--include_mc_sim` 追加 AEC 模拟集与多通道 mc_sim 集 |
| [`eval_pipeline_aec.py`](../../../evaluate/eval_pipeline_aec.py) | AEC | 同构的 AEC 评测流水线 |
| [`eval_pipeline_tse.py`](../../../evaluate/eval_pipeline_tse.py) | TSE / REAL-TSE | 读比赛 `*_meta.csv` 逐条推理，算 `spksim`/`dnsmos`/`timing`/`ter` 四比赛指标与可选有参考指标；用法、官方口径与依赖见 [`tasks/tse/README.md`](../../../tasks/tse/README.md) 的「评测（REAL-TSE）」节 |

公共模块：`common.py`（流水线公共工具）、`eval_io.py`（评估侧低层音频/SCP/推理 artifact 校验）、`aggregate_eval_summary.py`（REAL-TSE 结果按 dataset/language/overall 分组汇总）、`aec_pairing.py`（mic ↔ farend 配对，inference 与评测共用）。

特殊评估入口：

- [`oracle_beamformer.py`](../../../evaluate/oracle_beamformer.py)：基于目标空间协方差的 MVDR oracle 基线，供 SE 多通道路径对照。
- [`latency_verification.py`](../../../evaluate/latency_verification.py)：算法延迟/因果一致性验证（扰动法），TSE / SE / AEC 三任务统一入口。

## 指标实现分层

`core/metrics/` 是指标 SSOT，按类型分层：

- `separation.py`：PIT 感知的分离指标（`si_sdr` / `si_sdr_improvement` / `build_separation_metric_pairs`），SS 与 TSE 的分离质量共用。
- `intrusive.py`：有参考的侵入式指标（SI-SDR、PESQ、STOI、ESTOI），可选依赖缺失时返回 `None` 由上层处理。
- `mos.py` / `mos_models.py`：非侵入式 MOS 包装与批处理（DNSMOS、NISQA、UTMOS、SCOREQ），模型文件约定放 `third_party_scripts/` 对应目录。
- `aec.py` / `aecmos.py`：AEC 专用指标（ERLE 能量和、far-delay 估计等）与 AECMOS wrapper（加载 `third_party_scripts/AECMOS/` 的 ONNX）。
- `batch.py`：批量客观指标编排入口。

开发期指标（validation）走 YAML `trainer.val_metrics`，由 `MetricsCapableTrainer` 消费，与评测流水线相互独立。

## 第三方指标与资源约定

第三方实现统一放 [`third_party_scripts/`](../../../third_party_scripts/)，按 import guard 隔离：可选依赖缺失时对应指标明确跳过或降级，不让评测整体失败。已接入的入口包括 DNSMOS、AECMOS、NISQA、SCOREQ；TSE 评测额外约定 wespeaker（spksim 后端，缺失时按官方 Hub 机制下载到 `~/.wespeaker/`）、sherpa-onnx Zipformer（TER，可选依赖，未安装则跳过并 warning）、FireRedASR2S / FireRedVAD（timing 官方口径）。模型文件不进 git，各目录 README 有下载方式。

## 评估资产纪律

- checkpoint 加载 fail-closed：[`checkpoint_loading.py`](../../../evaluate/checkpoint_loading.py) 对 0 命中、`missing_keys`/`unexpected_keys`、shape 不匹配、EMA `shadow_params` 不一致都在复制权重前报错；禁止评测部分随机初始化模型。
- 评测 artifact 绑定内容身份：TSE 评测额外绑定 REAL-T release README、split、meta、mapping、overlap JSON 与 `scene_channel.csv` 的 sha256，任一变化使旧推理产物失配。
- 论文与榜单指标不等于项目复现：比赛榜单、第三方报告和 `docs/paper/` 中的数字，只有在本仓库按上述流水线实测后才能写成项目结果（「未验证」是合法状态，不能写成通过）。
- 环境能力如实标注：socket/Gloo、GPU、外部权重缺失时，对应路径只能写「未验证」。

## 权威口径

- 代码 SSOT：[`evaluate/`](../../../evaluate/)、[`core/metrics/`](../../../core/metrics/)。
- 真实验证状态清单：[`docs/architecture/README.md`](../../architecture/README.md)「当前真实验证状态」。
