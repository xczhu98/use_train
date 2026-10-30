# Speech Separation（SS）

`tasks/ss/` 是仓库里的多源语音分离任务包。它和 `tasks/se/`（语音增强）的区别在输出：SE 给回一路增强后的语音，SS 输入一路 mixture（多人混在一起、不指定要哪一路的录音），输出固定数量的 source（逐路分离出的语音）；「哪一路输出对应哪个说话人」不预先指定，由训练时的 PIT 处理。

先统一几个会反复出现的词：

- **mixture / source**：mixture 是混合录音，也就是输入；source 是要分离出的单路语音，数量由配置 `ss.num_sources` 固定（当前版本不支持每个样本源数不同的 variable-K）。
- **PIT**（Permutation Invariant Training，置换不变训练）：模型输出的 K 路顺序是任意的，PIT 在训练时把 K! 种「输出—监督」配对都算一遍，只保留 loss 最小的那种，这样就不必预先指定顺序。本仓库用 `PITLoss` 包住 pairwise loss，可选 `PairwiseL1Loss` 或 `PairwiseNegSISDRLoss`。
- **SI-SDR**（Scale-Invariant Signal-to-Distortion Ratio，尺度不变信噪失真比）：分离质量的常用指标，越高越好。
- **waveform-native / external STFT**：前者指模型直接吃波形；后者指模型吃框架算好的频谱（STFT 是短时傅里叶变换，把波形切成帧再变到频域）。
- **strict JSONL**：清单文件格式，每行一个 JSON 对象，字段缺失或多出不合规内容都会直接报错。
- **fail-fast**：配置不合法就立刻报错退出，不带着错误继续跑。

## 当前边界

- `data.mode: offline_ss`：读取严格 JSONL mixture/source manifest（每行一条混合音频与它的 K 路源）。
- `data.mode: dynamic_ss`：从共享 SCP/JSONL source pool 采样固定数量 source，在线执行共享 Layer A、SS Layer B（语音 / 语音+背景）、逐源 RIR（房间冲激响应）、post-RIR SIR、背景混合与 D–F。mono online moving scene 的全部 speech source 共用一次采样的 room、microphone position 和 timebase，每路保留独立轨迹。
- 固定 `ss.num_sources`，不支持 variable-K output。
- `TDANet` 使用 waveform-native `[B, T] -> [B, K, T]` 合同。
- `TIGER` 使用框架 external STFT，模型输出 `[B, K, F, T, 2]` source spectra。
- `SSModelRuntime.separate()` 是 trainer、inference 与 verifier 共用的模型调用与 source waveform 输出入口。
- `PITLoss` 通过统一 `loss:` 配置构造，内部包装 `PairwiseL1Loss` 或 `PairwiseNegSISDRLoss`。
- `SSExtractor` 统一 waveform-native 和 external-STFT 模型的多源 waveform 输出。
- `ResolvedExecutionPlan`（执行计划）在启动期统一校验 Dataset、Model、Loss、采样率、source count 与 STFT 所有权；训练和验证不再各自从 YAML 重新推断模型输入域。
- Trainer 的 loss 直接消费模型原始输出（只检查是否有限），不做幅度 clamp（裁剪）；只有写音频时才在发布边界裁剪。

## 入口

```bash
python train.py -C configs/ss/tdanet/tdanet_echoset_offline.yaml
python train.py -C configs/ss/tiger/tiger_echoset_external_stft.yaml
```

两个 offline 配置当前都指向 `data/ss/EchoSet/{train,val}.jsonl`（这份 EchoSet 清单与音频随仓库提供，可直接跑）；checkpoint 与日志写到各自配置的 `trainer.exp_path` 下。`dynamic_ss` 需要自己为 train/validation 配置 strict source-pool JSONL。带身份的 JSONL 每行可写为：

```json
{"uid":"speaker-001-utt-01","speaker_id":"speaker-001","fs":16000,"length":48000,"path":"audio/utt-01.wav"}
```

Dataset 先采样互不相同的 `(dataset_id, speaker_id)`，再从各说话人选择 utterance；旧清单没有 `dataset_id` 时用清单路径隔离 namespace。`speaker_id` 可省略，此时只保证抽到的 source 互不相同。可用身份或 source 数不足时启动失败。

动态配置见 `configs/ss/tdanet/tdanet_dynamic_sisdr.yaml`；`configs/ss/spatial/` 下另有 OnlineSpatialNet 的 `mhsa` / `ret` / `mamba` 三份 8 kHz NBSS 论文对齐配方。Layer B、sampling、时长过滤、timeline、音乐、RIR、SIR/SNR、source-scene audit 与 worker 随机性详见 [多源共享仿真](../../docs/guides/simulation/Multi_Source_Simulation.md)。

## 官方 checkpoint parity

官方权重不进入仓库。parity 指用官方权重和自己的实现跑同一段输入、比对结果一致，用来确认移植没有跑偏；准备权重后显式运行：

```bash
USE_TRAIN_TDANET_CHECKPOINT=/path/TDANetBest-2ms-LRS2.bin \
USE_TRAIN_TIGER_CHECKPOINT=/path/TIGER-speech.safetensors \
python -m pytest -m external tests/test_ss_official_checkpoint_parity.py -q
```

`-m external` 只选中需要外部权重的用例；未设置对应环境变量的用例会 skip。

TDANet checkpoint 里的 `model_args` 才是实际的模型合同；EchoSet baseline 使用与 checkpoint 兼容的 16 kHz/2 ms 拓扑。TIGER parity 固定使用 `640/160/640` 的 external-STFT 几何（window / hop / n_fft）。

训练初始化的 `--checkpoint` 同时支持安全的 torch model-only checkpoint 和 `.safetensors`；两种格式都会在复制参数前执行完整 key、dtype、shape 校验。

## 🧠 模型清单

| 模型 | 说明 | 参考来源 |
|------|------|----------|
| **TDANet** | waveform-native 固定 source count 分离模型 | [项目实现](models/tdanet.py) |
| **TIGER** | external-STFT 固定 source count 分离模型 | [项目实现](models/tiger.py) |
| **SpatialNet** | 离线多通道谱进、K 路源谱出分离模型，参考公开 NBSS 项目（`Audio-WestlakeU/NBSS`）设计 | [项目实现](models/spatialnet.py) |
| **OnlineSpatialNet** | 因果（只看历史帧、可用于实时）多通道分离模型，窄带分支与 SE backend 相同（`mhsa`/`ret`/`mamba`） | [项目实现](models/online_spatialnet.py) |

## 🔍 推理

SS 没有根文件推理入口：把 SS 配置交给根 `inference.py` 会在模型构造前直接报错。分离推理走任务包内的 `SSExtractor`（`tasks/ss/inference/extractor.py`），它把 waveform-native 与 external-STFT 模型的多源 waveform 输出统一成一份接口：构造需要 `SSExtractor(model, plan=..., stft_module=...)`（`plan` 是已解析的 `ResolvedExecutionPlan`），调用 `extractor(mixture, sample_rate=...)` 返回 `[K, T]` 波形。模型调用与 source 输出合同由 `SSModelRuntime.separate()` 统一；`tasks/ss/inference/adapter.py` 提供同一边界的 adapter。分离指标使用共享的 `core/metrics/separation.py`。
