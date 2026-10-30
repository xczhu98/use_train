# 当前任务化仓库目录结构（2026-09-21）

本文按当前工作区实际目录整理。`data/`、`exp/`、`runs/`、`third_party_scripts/` 中的数据集、checkpoint、日志和第三方源码不展开；它们不是 Python 任务实现。

## 一、仓库边界

```text
use_train/
├── core/                 # 共享执行基础设施：data、losses、metrics、models、trainer、inference、utils
├── tasks/                # 五个任务的 Python 实现与任务注册
│   ├── se/               # Speech Enhancement
│   ├── aec/              # Acoustic Echo Cancellation
│   ├── tse/              # Target Speaker Extraction
│   ├── ss/               # Speech Separation
│   └── ssl/              # Spatial SSL（固定阵列 IPD 特征与空间监督）
├── configs/              # 实验配置；仍按 configs/<task>/ 组织
│   ├── se/
│   ├── aec/
│   ├── tse/
│   ├── ss/
│   └── ssl/
├── evaluate/             # 跨任务评测编排、checkpoint 和 artifact 生命周期
├── onnx/                 # 跨任务 ONNX 导出、runtime、推理和发布验证
├── tools/                # 数据准备、模型工具、验证脚本和共享工具
├── tests/                # 跨层与五任务 unit / integration / DDP 测试
├── docs/                 # 架构、指南、论文和报告
├── train.py              # 统一训练入口
├── inference.py          # 根文件推理入口；公共生命周期位于 core/inference
└── README.md / AGENTS.md # 仓库说明与开发合同
```

## 二、任务公共入口

```text
tasks/
├── __init__.py           # 任务命名空间导出
├── descriptors.py        # TaskDescriptor 与五任务静态描述
├── registry.py           # task / data.mode 路由与训练入口、data module 惰性加载
└── task_api.py           # descriptor 驱动的统一 dataset/collate API
```

当前路由为：

| 任务 | Python 包 | data.mode | 训练入口 | 配置目录 |
|---|---|---|---|---|
| SE | `tasks.se` | `dynamic` / `offline` | `tasks.se.trainer.entry.SETrainingEntry` | `configs/se/` |
| AEC | `tasks.aec` | `dynamic_aec` / `offline_aec` | `tasks.aec.trainer.entry.AECTrainingEntry` | `configs/aec/` |
| TSE | `tasks.tse` | `dynamic_tse` / `offline_tse` | `tasks.tse.trainer.entry.TSETrainingEntry` | `configs/tse/` |
| SS | `tasks.ss` | `dynamic_ss` / `offline_ss` | `tasks.ss.trainer.entry.SSTrainingEntry` | `configs/ss/` |
| SSL | `tasks.ssl` | `dynamic_ssl` | `tasks.ssl.trainer.entry.SSLTrainingEntry` | `configs/ssl/` |

## 三、任务包实际结构

### SE：`tasks/se/`

```text
tasks/se/
├── __init__.py
├── data/__init__.py
├── data/batch.py
├── data/dynamic_dataset.py
├── data/pase_preprocess.py
├── data/post_mix_adapter.py
├── models/__init__.py
├── models/base_model.py
├── models/discriminative/bs_roformer.py
├── models/discriminative/bsrnn.py
├── models/discriminative/clearervoice.py
├── models/discriminative/clearervoice_dsp.py
├── models/discriminative/conformer.py
├── models/discriminative/dccrn.py
├── models/discriminative/deepfilternet.py
├── models/discriminative/fast_ulcnet.py
├── models/discriminative/fastenhancer.py
├── models/discriminative/gtcrn.py
├── models/discriminative/gtcrn/open_model_checkpoint/README.md
├── models/discriminative/gtcrn/open_model_checkpoint/config.json
├── models/discriminative/gtcrn/open_model_checkpoint/model_trained_on_dns3.tar
├── models/discriminative/gtcrn/open_model_checkpoint/model_trained_on_vctk.tar
├── models/discriminative/mossformer2.py
├── models/discriminative/mtfaa_net.py
├── models/discriminative/rt_semamba.py
├── models/discriminative/tf_gridnet.py
├── models/discriminative/tf_skimnet.py
├── models/discriminative/ulunas.py
├── models/discriminative/ulunas/open_model_checkpoint/README.md
├── models/discriminative/ulunas/open_model_checkpoint/config.json
├── models/discriminative/ulunas/open_model_checkpoint/model_trained_on_dns3.tar
├── models/discriminative/zipenhancer.py
├── models/gan/__init__.py
├── models/gan/discriminators/__init__.py
├── models/gan/discriminators/discriminator.py
├── models/gan/discriminators/metric_mag_discriminator.py
├── models/gan/generators/__init__.py
├── models/gan/generators/mambattention.py
├── models/gan/generators/mossformer_gan.py
├── models/gan/generators/mp_senet.py
├── models/gan/generators/rwsa_mamba_unet.py
├── models/generative/meanflowse/LICENSE-Apache-2.0-score-sde
├── models/generative/meanflowse/LICENSE-MIT-MeanFlowSE
├── models/generative/meanflowse/PROVENANCE
├── models/generative/meanflowse/__init__.py
├── models/generative/meanflowse/model.py
├── models/generative/meanse/PROVENANCE
├── models/generative/meanse/__init__.py
├── models/generative/meanse/model.py
├── models/modules/bsrnn_impl/__init__.py
├── models/modules/bsrnn_impl/bsrnn_core.py
├── models/modules/clearervoice_impl/__init__.py
├── models/modules/clearervoice_impl/frcrn/__init__.py
├── models/modules/clearervoice_impl/frcrn/complex_nn.py
├── models/modules/clearervoice_impl/frcrn/conv_stft.py
├── models/modules/clearervoice_impl/frcrn/frcrn.py
├── models/modules/clearervoice_impl/frcrn/se_layer.py
├── models/modules/clearervoice_impl/frcrn/unet.py
├── models/modules/clearervoice_impl/mossformer2/__init__.py
├── models/modules/clearervoice_impl/mossformer2/conv_module.py
├── models/modules/clearervoice_impl/mossformer2/fsmn.py
├── models/modules/clearervoice_impl/mossformer2/layer_norm.py
├── models/modules/clearervoice_impl/mossformer2/mossformer2.py
├── models/modules/clearervoice_impl/mossformer2/mossformer2_block.py
├── models/modules/clearervoice_impl/mossformer_gan/conformer.py
├── models/modules/clearervoice_impl/mossformer_gan/conv_module.py
├── models/modules/clearervoice_impl/mossformer_gan/fsmn.py
├── models/modules/clearervoice_impl/mossformer_gan/generator.py
├── models/modules/clearervoice_impl/mossformer_gan/get_layer_from_string.py
├── models/modules/clearervoice_impl/mossformer_gan/mossformer.py
├── models/modules/clearervoice_impl/mossformer_gan/se_layer.py
├── models/modules/dccrn_impl/LICENSE
├── models/modules/dccrn_impl/PROVENANCE
├── models/modules/dccrn_impl/__init__.py
├── models/modules/dccrn_impl/complex_layers.py
├── models/modules/dccrn_impl/network.py
├── models/modules/deepfilternet_impl.py
├── models/modules/fast_ulcnet_impl/LICENSE.CRM_pytorch
├── models/modules/fast_ulcnet_impl/LICENSE.Fast-ULCNet
├── models/modules/fast_ulcnet_impl/PROVENANCE
├── models/modules/fast_ulcnet_impl/__init__.py
├── models/modules/fast_ulcnet_impl/comfi_fast_grnn.py
├── models/modules/fast_ulcnet_impl/crm.py
├── models/modules/mambattention_impl/codec_module.py
├── models/modules/mambattention_impl/generator.py
├── models/modules/mambattention_impl/lsigmoid.py
├── models/modules/mambattention_impl/mambattention_block.py
├── models/modules/mp_senet_impl/discriminator.py
├── models/modules/mp_senet_impl/generator.py
├── models/modules/mp_senet_impl/transformer.py
├── models/modules/mp_senet_impl/utils.py
├── models/modules/mtfaa_impl/LICENSE
├── models/modules/mtfaa_impl/PROVENANCE
├── models/modules/mtfaa_impl/__init__.py
├── models/modules/mtfaa_impl/core.py
├── models/modules/rt_semamba_impl/LICENSE
├── models/modules/rt_semamba_impl/PROVENANCE
├── models/modules/rt_semamba_impl/__init__.py
├── models/modules/rt_semamba_impl/network.py
├── models/modules/rwsa_mamba_impl/LICENSE
├── models/modules/rwsa_mamba_impl/PROVENANCE
├── models/modules/rwsa_mamba_impl/codec_module.py
├── models/modules/rwsa_mamba_impl/generator.py
├── models/modules/rwsa_mamba_impl/lsigmoid.py
├── models/modules/rwsa_mamba_impl/mambattention_block.py
├── models/modules/tf_skimnet_impl/PROVENANCE
├── models/modules/tf_skimnet_impl/__init__.py
├── models/modules/tf_skimnet_impl/core.py
├── models/modules/zipenhancer_impl/__init__.py
├── models/modules/zipenhancer_impl/generator.py
├── models/modules/zipenhancer_impl/scaling.py
├── models/modules/zipenhancer_impl/zipenhancer_layer.py
├── models/modules/zipenhancer_impl/zipformer.py
├── models/pase_family/__init__.py
├── models/pase_family/common/__init__.py
├── models/pase_family/common/dewavlm_distill.py
├── models/pase_family/common/discriminator.py
├── models/pase_family/common/utils.py
├── models/pase_family/common/vocoder_backend.py
├── models/pase_family/common/vocos/__init__.py
├── models/pase_family/common/vocos/backbone.py
├── models/pase_family/common/vocos/conv.py
├── models/pase_family/common/vocos/head.py
├── models/pase_family/common/wavlm/WavLM.py
├── models/pase_family/common/wavlm/__init__.py
├── models/pase_family/common/wavlm/modules.py
├── models/pase_family/common/wavlm_frontend.py
├── models/pase_family/gap_urgenet/__init__.py
├── models/pase_family/gap_urgenet/gan_wrappers.py
├── models/pase_family/gap_urgenet/gap_urgenet_model.py
├── models/pase_family/gap_urgenet/open_model_checkpoint/Adapter.pt
├── models/pase_family/gap_urgenet/open_model_checkpoint/DeWavLM-Omni.pt
├── models/pase_family/gap_urgenet/open_model_checkpoint/PostNet.pt
├── models/pase_family/gap_urgenet/open_model_checkpoint/Predictor.pt
├── models/pase_family/gap_urgenet/open_model_checkpoint/README.md
├── models/pase_family/gap_urgenet/open_model_checkpoint/Vocoder.pt
├── models/pase_family/gap_urgenet/open_model_checkpoint/config.json
├── models/pase_family/gap_urgenet/postnet/__init__.py
├── models/pase_family/gap_urgenet/postnet/tfgrid_cws.py
├── models/pase_family/gap_urgenet/predictor/__init__.py
├── models/pase_family/gap_urgenet/predictor/tfgridnet.py
├── models/pase_family/pase/__init__.py
├── models/pase_family/pase/model.py
├── models/pase_family/pase/open_model_checkpoint/DeWavLM.pt
├── models/pase_family/pase/open_model_checkpoint/README.md
├── models/pase_family/pase/open_model_checkpoint/Vocoder_Dual.pt
├── models/pase_family/pase/open_model_checkpoint/Vocoder_L24.pt
├── models/pase_family/pase/open_model_checkpoint/config.json
├── models/pase_family/pase/open_model_checkpoint/framework_all.png
├── models/pase_family/stupase/__init__.py
├── models/pase_family/stupase/flow/__init__.py
├── models/pase_family/stupase/flow/cfm.py
├── models/pase_family/stupase/flow/dit.py
├── models/pase_family/stupase/flow/modules.py
├── models/pase_family/stupase/flow/utils.py
├── models/pase_family/stupase/flow_train_wrapper.py
├── models/pase_family/stupase/mel_feature_extractor.py
├── models/pase_family/stupase/mel_gan_wrapper.py
├── models/pase_family/stupase/open_model_checkpoint/CFM.pt
├── models/pase_family/stupase/open_model_checkpoint/DeWavLM-R.pt
├── models/pase_family/stupase/open_model_checkpoint/README.md
├── models/pase_family/stupase/open_model_checkpoint/Vocoder_Mel-16k.pt
├── models/pase_family/stupase/open_model_checkpoint/config.json
├── models/pase_family/stupase/open_model_checkpoint/framework_stupase.png
├── models/pase_family/stupase/stupase_model.py
├── models/pase_family/stupase/vocos_head.py
├── models/pase_family/stupase/vocos_vocoder.py
├── models/pase_family/unipase/__init__.py
├── models/pase_family/unipase/adapter.py
├── models/pase_family/unipase/adapter_discriminators.py
├── models/pase_family/unipase/gan_wrappers.py
├── models/pase_family/unipase/open_model_checkpoint/Adapter.pt
├── models/pase_family/unipase/open_model_checkpoint/DeWavLM-Omni.pt
├── models/pase_family/unipase/open_model_checkpoint/PostNet.pt
├── models/pase_family/unipase/open_model_checkpoint/README.md
├── models/pase_family/unipase/open_model_checkpoint/Vocoder_DWO-L1.pt
├── models/pase_family/unipase/open_model_checkpoint/Vocoder_WavLM-L24.pt
├── models/pase_family/unipase/open_model_checkpoint/config.json
├── models/pase_family/unipase/postnet/__init__.py
├── models/pase_family/unipase/postnet/tfgrid_cws_res.py
├── models/pase_family/unipase/unipase_model.py
├── models/pase_family/unipase/vocoder.py
├── models/pase_family/unipase/wavlm_frontend_plc.py
├── trainer/__init__.py
├── trainer/dual_input_loss_trainer.py
├── trainer/entry.py
├── trainer/gan_trainer.py
├── trainer/meanflowse_trainer.py
├── trainer/metricgan_trainer.py
├── trainer/metrics_trainer.py
├── trainer/pase_gan_trainer.py
├── trainer/rt_semamba_distill_trainer.py
├── trainer/se_trainer.py
├── trainer/unipase_adapter_gan_trainer.py
├── inference/__init__.py
├── inference/adapter.py
├── inference/streaming.py
```

### AEC：`tasks/aec/`

```text
tasks/aec/
├── README.md
├── data/__init__.py
├── data/audio_path_utils.py
├── data/batch.py
├── data/collate.py
├── data/dynamic_aec_build.py
├── data/dynamic_dataset.py
├── data/echo_clipping.py
├── data/echo_device_eq.py
├── data/erl_coupling.py
├── data/linear_aec_frontend.py
├── data/offline_dataset.py
├── data/pfdkf.py
├── data/pfdkf_torch.py
├── data/post_mix_adapter.py
├── data/state_label.py
├── data/time_delay_jitter.py
├── inference/__init__.py
├── inference/_native/delay_estimator_ext.cpython-310-x86_64-linux-gnu.so
├── inference/aec_frontend.py
├── inference/csrc/delay_estimator_ext.cpp
├── inference/csrc/pocketfft_hdronly.h
├── inference/csrc/pyproject.toml
├── inference/csrc/setup.py
├── inference/gcc_phat_core.py
├── inference/matched_filter_delay_estimator.py
├── inference/offline_alignment.py
├── inference/streaming_aec_frontend.py
├── inference/streaming_delay_estimator.py
├── metrics/__init__.py
├── metrics/state_metrics.py
├── models/__init__.py
├── models/discriminative/cagcrn.py
├── models/discriminative/dbgnet.py
├── models/discriminative/deepvqe.py
├── models/discriminative/localvqe.py
├── models/discriminative/ultra_dpc_aec.py
├── models/modules/__init__.py
├── models/modules/mtfaa_mask.py
├── models/state_detector.py
├── trainer/__init__.py
├── trainer/aec_trainer.py
├── trainer/entry.py
```

### TSE：`tasks/tse/`

```text
tasks/tse/
├── README.md
├── __init__.py
├── cue_contract.py
├── data/__init__.py
├── data/batch.py
├── data/collate.py
├── data/dynamic_dataset.py
├── data/enrollment_augment.py
├── data/offline_dataset.py
├── data/scene_dataset.py
├── data/spatial_cues.py
├── data/wesep_adapter.py
├── inference/__init__.py
├── inference/adapter.py
├── inference/artifacts.py
├── inference/extractor.py
├── inference/sidecar.py
├── inference/streaming.py
├── inference/torchscript.py
├── model_io.py
├── objectives.py
├── models/__init__.py
├── models/discriminative/tse_bsrnn_spk.py
├── models/discriminative/tse_modular.py
├── models/modular_spec.py
├── models/modules/common/deep_update.py
├── models/modules/common/norm.py
├── models/modules/composition.py
├── models/modules/fusion/film.py
├── models/modules/fusion/speech.py
├── models/modules/separator/adapters.py
├── models/modules/separator/bsrnn.py
├── models/modules/spatial/doa_features.py
├── models/modules/spatial/fixed_features.py
├── models/modules/spatial/pos_encoding.py
├── models/modules/spatial/spatial_frontend.py
├── models/modules/speaker/spk_frontend.py
├── models/modules/speaker/usef.py
├── models/modules/textual/__init__.py
├── models/modules/textual/kce.py
├── models/modules/textual/textual_frontend.py
├── models/modules/visual/__init__.py
├── models/modules/visual/muse.py
├── models/modules/visual/visual_frontend.py
├── models/speaker/config.py
├── models/speaker/ecapa.py
├── models/speaker/encoder.py
├── models/speaker/fbank.py
├── models/speaker/resnet.py
├── pretrained_models/README.md
├── pretrained_models/spk_emb_100/avg_model.pt
├── pretrained_models/spk_emb_100/config.yaml
├── pretrained_models/spk_emb_causal_100/avg_model.pt
├── pretrained_models/spk_emb_causal_100/config.yaml
├── pretrained_models/tfmap_context_100/avg_model.pt
├── pretrained_models/tfmap_context_100/config.yaml
├── pretrained_models/tfmap_context_causal_100/avg_model.pt
├── pretrained_models/tfmap_context_causal_100/config.yaml
├── speaker_embedding_models/README.md
├── speaker_embedding_models/ecapa_tdnn_glob_c512/.gitkeep
├── speaker_embedding_models/ecapa_tdnn_glob_c512/avg_model.pt
├── speaker_embedding_models/resnet34/.gitkeep
├── trainer/__init__.py
├── trainer/entry.py
├── trainer/tse_trainer.py
```

### SS：`tasks/ss/`

```text
tasks/ss/
├── .gitignore
├── README.md
├── __init__.py
├── data/__init__.py
├── data/collate.py
├── data/config.py
├── data/dynamic_dataset.py
├── data/offline_dataset.py
├── data/source_pool.py
├── inference/__init__.py
├── inference/extractor.py
├── inference/adapter.py
├── model_io.py
├── models/__init__.py
├── models/_third_party/NOTICE.md
├── models/_third_party/__init__.py
├── models/_third_party/tdanet.py
├── models/_third_party/tiger.py
├── models/_third_party/tiger_layers/__init__.py
├── models/_third_party/tiger_layers/activations.py
├── models/_third_party/tiger_layers/normalizations.py
├── models/tdanet.py
├── models/tiger.py
├── trainer/__init__.py
├── trainer/entry.py
├── trainer/ss_trainer.py
```

## 四、共享层职责

| 目录 | 职责 |
|---|---|
| `core/data/` | 共享数据合同、仿真基础、采样、collate 原语和时间轴处理 |
| `core/losses/` | 共享 loss registry、Composite、PIT/频域/时域损失 |
| `core/metrics/` | 共享指标和 metric batch 合同 |
| `core/models/` | 模型构造、IO contract、execution plan 和任务 registry facade |
| `core/trainer/` | 训练生命周期、DDP、AMP、EMA、checkpoint、scheduler 和验证编排 |
| `core/utils/` | 配置、安全加载、artifact fingerprint、registry 等通用能力 |
| `evaluate/` | 评测 pipeline、checkpoint loader、summary 和 artifact 生命周期 |
| `onnx/` | streaming wrapper、runtime metadata、导出和三路验证 gate |
| `tools/` | 数据生成、模型审计、任务验证、共享 IO 和工程脚本 |

## 五、当前边界

- 新 Python 代码使用 `tasks.<task>.*`；旧顶层 `se/`、`aec/`、`tse/`、`ss/` 不再存在。
- `configs/<task>/`、`data/<task>/`、`exp/<task>/` 是资源和实验路径，不能因为 Python 包迁移而改成 `tasks/<task>/`。
- TSE 的 speaker checkpoint 和配套说明位于 `tasks/tse/`，权重仍由 `.gitignore` 隔离。
- 复杂度审计 `tools/model_tools/audit_metrics.py` 当前支持 SE、AEC、TSE、SS 四个任务（SSL 不在其支持范围）：SE/AEC/TSE/SS 分别走任务专用输入 adapter；SS 区分 TDANet waveform-native 与 TIGER external-STFT。
- `evaluate/` 和 `onnx/` 仍是跨任务目录；本次没有复制为 `tasks/<task>/evaluate` 或 `tasks/<task>/onnx`。
- 文件推理已收敛到统一抽象：`core/inference/contracts.py` 的 `InferenceAdapter` 协议（`infer(FileInferenceItem) -> PreparedOutput`），manifest、output lock、`skip-existing` 与 sidecar 由公共生命周期负责；任务 Adapter 只处理任务模型 IO。SE/AEC/TSE 均有根入口，其中 TSE 使用严格 JSONL cue manifest 并从 `TSEExtractor` 派生 sidecar；SS 走 `SSExtractor`/execution plan 而无根文件推理入口，SSL 无公开音频文件入口。

## 六、检查命令

```bash
python -c "from tasks import iter_tasks; print([t.name for t in iter_tasks()])"
python tools/model_tools/audit_metrics.py -C configs/ss/tdanet/tdanet_echoset_offline.yaml --task ss --no_profile
python -m pytest -m unit tests/test_model_complexity_tools.py tests/test_task_namespace.py -q
```
