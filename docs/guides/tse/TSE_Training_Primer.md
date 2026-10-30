# TSE Training Primer

本文把 TSE 的训练链路串一遍：先选模型与 cue，再理解批合同如何进 trainer，最后看评测入口。所有「怎么跑」的命令以 [`../../../tasks/tse/README.md`](../../../tasks/tse/README.md) 为准，本文只做导读与定位。

## 1. 选模型：两个 registry 身份

TSE registry 当前只有 `TSE_BSRNN_SPK` 与 `TSE_Modular`（[`../../../tasks/tse/README.md`](../../../tasks/tse/README.md) 的模型清单表是 SSOT）。选择逻辑很简单：

- **`TSE_BSRNN_SPK`**：REAL-TSE Challenge baseline（`wesep-real-tse`）移植，内置 5 种可插拔 speaker cue 特征（`listen`/`usef`/`tfmap`/`context`/`spkemb`），走 causal streaming 路径，有 ONNX / TorchScript 部署链路。只需要 speaker waveform / embedding 时选它。
- **`TSE_Modular`**：BSRNN 模块化组合，`model.args` 里按 frontend 装配 cue——speaker（`usef`/`tfmap`/`context`/`spkemb`）、spatial（`modules/spatial/`，IPD/CDF/SDF/ΔSTFT）、visual（`modules/visual/`，仅预提取 features）、textual（`modules/textual/`，phoneme ids）。full-sequence forward，**不支持 streaming**；`require_implemented=True` 时未实现的组合在构造期抛 `NotImplementedError`，`listen` 与 raw video 由配置边界拒绝。

模型注册和 frontend 接线不等于端到端闭环完成。四模态的本机完成度（frontend / Dataset 解码 / 真实数据 / 闭环）逐格记录在任务 README 的「四模态完成度」表：当前只有 speaker 与 textual 完成了真实数据闭环，spatial 与 visual 没有已整理数据集。

## 2. cue 从哪来

- **speaker**：dynamic 模式从 nested enrollment 清单（`spk2enroll.json`）采样注册音；offline 模式从 `cues.yaml` 读取 waveform 或 embedding。Libri2Mix 的 dynamic speaker 清单已按当前合同生成（见任务 README 的数据现状节），offline speaker recipe 还需要 `cues.yaml`。
- **textual**：WeSep `json/dae_keyword_phoneme` 资源，`scope=speaker`、`selection=random` 选词；KCE backbone 用官方 DAE-TSE 权重，缺失时构造即 fail-fast，无随机初始化兜底。
- **spatial / visual**：offline 分别解码 `npy/fields` 与 `npy/muse_frontend`；dynamic spatial 可按 `actor_uid` 生成方向 cue。两者本机都没有已整理数据，不要按「frontend 已接线」推断可以训练。

## 3. 批合同怎么进 trainer

Dataset 产出 canonical dict item（`key`/`mixture`/`sr`/`length`/`targets`，可选 geometry/timebase/provenance），`tse_collate_fn` 拼成 `TSEBatch` 并 `validate()`。trainer 侧有三个实现约定容易踩：

1. **`_unpack_batch` 只接受 `TSEBatch`**，tuple 显式报错；设备搬运走 `TSEBatch.to()`，包括全部 typed cue tensor。新增字段必须同时进 canonical item、`TSEBatch`、`validate()` 和实际消费链路。
2. **多目标批训练**（`simulation_cfg.targets_per_mixture > 1`）：对 `[B,C,T]` mixture 只算一次 STFT，按 `[B]` `num_targets` 用 `repeat_interleave` 展开，与 `[R,T]` target、speaker cue、`spk_labels` 逐行对齐，loss 在 `R=sum(num_targets)` 行上计算；GPU 增强只作用于原始 B 条 mixture。
3. **SSA 与 multi-task CE**：SSA（`trainer.ssa_prob`，默认 0.0 关闭）以概率整批先 `no_grad` 预前向得 est0、再当作 enrollment 训练，随机性由 `(meta.seed, epoch, batch_idx)` 派生保证 DDP 一致；`joint_training: false` 时 SSA 不适用（warning 一次后跳过）。multi-task speaker CE 在 `CompositeLoss.components` 加 `SpeakerCrossEntropyLoss` 后由 entry 自动推导 `model.args.multi_task: true` 并注入 `spks_in_train`，未知说话人（label=-1）由 `ignore_index` 跳过。

## 4. 训练与评测入口

- 训练：`python train.py -C configs/tse/dynamic/tse_bsrnn_spk.yaml`（dynamic speaker）、`configs/tse/textual/tse_modular_textual_keywords.yaml`（offline textual）。WeSep 参数对齐配方（`tse_bsrnn_spk_wesep_baseline.yaml` 等四份）覆盖 chunk、epoch、optimizer 与 dither 语义，但不保证 processor/随机流 bit-exact。
- 评测：REAL-TSE 评测走 `evaluate/eval_pipeline_tse.py`，读取 `*_meta.csv` 逐条推理并算比赛四指标（`spksim`/`dnsmos`/`timing`/`ter`）及可选有参考指标；指标口径、依赖与 artifact 身份绑定见任务 README 的「评测（REAL-TSE）」节。常规开发指标（SI-SDR/PESQ/ESTOI）走 YAML `trainer.val_metrics`，与 SE/AEC 一致。

## 权威口径

- 数据清单与命令：[`../../../tasks/tse/README.md`](../../../tasks/tse/README.md)。
- 模型挂载与 core 追加点：[`../../architecture/TSE_Architecture.md`](../../architecture/TSE_Architecture.md)。
- 批合同与 entry 约束：根 `AGENTS.md` 任务合同矩阵、`tasks/tse/data/`、`tasks/tse/trainer/`。
- TSE 仿真：[`../simulation/Multi_Source_Simulation.md`](../simulation/Multi_Source_Simulation.md)。
