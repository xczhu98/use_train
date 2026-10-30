# RIR 生成器配置

本目录存放 RIR 生成代码、backend 实现和生成 profile。RIR 生成配置不属于训练/推理配置，不放在仓库顶层 `configs/`。

## 文件命名

配置文件按 backend 各保留一份，采样率通过 YAML 的 `sample_rates_hz` 配置，不写入文件名：

```text
rir_<backend>.yaml
```

例如：

```text
rir_gpurir.yaml
rir_fram.yaml
rir_rirgen.yaml
```

文件名全部使用小写 `snake_case`。文件名只表达 backend；数据用途、采样率和具体参数以 YAML 字段及 `rirs_meta.json` 的 `effective_config` 为准。

## 运行方式

推荐生产数据使用 PKL：

```bash
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --output-dir /data/rir/realistic_speech_16k \
    --validate
```

多麦克风离线 RIR 使用独立的 scene-level 入口，不改变上面的单通道 legacy
生成器。配置 `scene.geometry.array_profile` 或 `scene.geometry.custom_geometry`
且解析出 `C>1` 后执行：

当前 scene-level 生成入口使用 `gpurir` backend；`fram` 和 `rirgen` 仍明确保持
legacy mono backend，不会伪造 `[A,C,L]` 空间输出。

```bash
python -m tools.prepare_data_tools.generators.se.generate_scene_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --output-dir /data/rir/spatial_scene_16k \
    --num-scenes 1000 \
    --sample-rate 16000 \
    --field far
```

该入口生成：

```text
scene_bundles/<shard>/<scene_uid>.npz
scene_manifests/<scene_uid>.json
scenes.jsonl
scenes_meta.json
generation_complete.json
```

每个 scene 绑定同一个 room、绝对麦克风阵列位置、actor order、actor positions、采样率
和 full/direct/early content hash。输出使用一个 scene `.npz` bundle 和每个 scene 的 JSON
manifest，避免为每个 actor/track 创建大量小文件。没有 geometry 或解析得到 `C=1` 时必须继续使用
`generate_simulated_rirs` 的 mono `rirs.scp` 合同，不能把 legacy 单通道数据伪装成 spatial scene。

查看配置默认值：

```bash
python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
    --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
    --dump-effective-config
```

`--output-dir` 可以覆盖 YAML 中的 `output_dir: null`。生成参数统一写在 YAML 中；CLI 不再提供旧版参数别名，避免同一语义出现两套命名。

## 配置约定

- YAML 字段统一使用 `snake_case`。
- 带单位的数值在字段名中标注 `_hz`、`_m`、`_sec`、`_ms`、`_mps` 或 `_samples`。
- 每个范围字段的注释必须说明单位、采样方式、边界和上下字段关系。
- `sample_rates_hz` 必须唯一；near/far 距离默认应互斥。
- `alpha`、`beta` 和 `reflect_coef` 由房间、RT60、声速和 backend 推导，不作为普通顶层输入。
- `output.format` 默认使用 `pkl`；FLOAT WAV 可保存超范围 float32，FLAC 超出 `[-1, 1]` 时直接失败。
- `validation.enabled` 建议生产开启，发布前检查采样率、帧数、有限性和移动 segment 峰值。
- 配置加载遇到未知字段会 fail-fast，不会静默忽略拼写错误。

## Profile 选择

`rir_gpurir.yaml`、`rir_fram.yaml` 和 `rir_rirgen.yaml` 分别对应三个 backend。
需要压力测试时直接覆盖同一 YAML 中的 RT60、房间、距离和移动字段，不另建采样率文件。

当前公共默认值为 `total_rirs=100000`、`rt60_range_sec=[0.1,1.0]`、
`static_moving_ratio=[1.0,0.0]`、`motion.enabled=false`、
`motion.n_segments_range=[3,6]` 和 `execution.shards=10`。启用移动模式后，
每个场景按 scene seed 在闭区间 `[3,6]` 内采样段数。

spatial moving scene（多麦克风阵列 + 逐 actor 轨迹）由
`generate_scene_rirs.py --moving` 生成：输出 schema v3 typed manifest，
track 布局 `[P,A,C,L]`，generator version `se.scene-rir-v3.0`，轨迹与
timebase 物化在 manifest 中，消费侧按 manifest 权威渲染。`--moving` 要求
YAML 中 `motion.enabled: true`（复用 `n_segments_range` /
`segment_duration_sec` / `speed_range_mps` / `trajectory_type` 采样轨迹），
否则在 GPU 导入前 fail-fast。

生成器内容语义发生改变时必须递增 `GENERATOR_VERSION`，使旧 manifest 指纹与新 profile 语义失配。

## mc_sim 空间评估集

`tools.prepare_data_tools.generators.se.generate_mc_sim_testset` 使用共享 scene plan 和空间渲染器生成静态双源、多麦克风 SE 集。命令行接收动态 SE 配置、目标与干扰两份四列 SCP、输出目录及样本数；两份 SCP 的 UID 必须不交叠，但 UID 不交叠不能证明 speaker-disjoint，speaker 隔离由数据准备方保证。生成器发布 `mixture.scp`、同内容的 `noisy.scp`、`clean.scp`、`target_image.scp` 和 `mc_sim_manifest.json`。SCP 来源可使用小型 fixture 检查工具链；fixture 结果不代表真实 WSJ0 验证，真实 WSJ0 数据也不保证本地可用。

```bash
python -m tools.prepare_data_tools.generators.se.generate_mc_sim_testset \
    --config configs/se/spatial/spatialnet_dynamic.yaml \
    --target-scp <target_speech.scp> \
    --interference-scp <interference_speech.scp> \
    --output-dir data/se/mc_sim_testset \
    --num-samples 100 --seed 42 --duration-seconds 4 \
    --sir-range-db -5 5
```

SE offline spatial split 支持 mono clean 与 `[C,T]` noisy。split 必须显式声明 `spatial_enabled`、`array_profile` 或 `custom_geometry`、`reference_channel` 和 `channel_order`；加载时按 geometry 核验音频实际通道数，不会根据首个文件推断或重排通道。

`ComplexIdealRatioMaskLoss` 是复数理想比率掩码（cIRM）MSE loss，不接入 NBC/NBC2 模型。它以模型的 `enhanced_spectra` 和 direct `target_spectra`，结合 noisy reference waveform 在选定参考通道计算目标/预测 mask 并计算 MSE；参考通道必须与数据阵列配置一致。
