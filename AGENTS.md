# AGENTS.md — use_train

## 核心指令

1. **增量修改优先**：用 `Edit` 做增量替换，不要用全文覆盖重写已有文件。
2. **单步一义**：一次修改只做一个逻辑变更；不夹带无关重构。
3. **先读后改**：修改任何文件前必须先用 `Read` 确认当前内容。
4. **Git 变更需明确授权且按任务范围提交**：未经用户明确请求，不执行 `git add`、`git commit`、`git push`、`git reset`、`git rebase` 等 Git mutation；用户明确要求提交时，默认仅暂存并提交与当前任务相关的改动（先 `git status`/`git diff` 确认范围并明确列出），避免卷入工作区中无关的并行改动；仅当用户明确要求全量提交时才使用 `git add -A` 纳入全部已跟踪和未跟踪变更。
5. **Push 默认同步三个远端**：用户明确要求 `push` 时，默认将当前分支推送到 `17u`、`xaudiolabs`、`xczhu` 三个远端；仅当用户明确指定远端或限制推送范围时，才只推送指定远端。任一远端推送失败时必须报告各远端结果，不得把部分成功表述为全部完成。
6. **同步文档**：代码变更后检查并更新相关 README / docstring，避免文档漂移。
7. **语言**：
   - 提交信息（commit message）使用英文。
   - 文档（README、`AGENTS.md` 等 Markdown）使用中文，专有名词、模型名称、代码术语保留英文。
   - 代码注释使用中文。
   - 回答用户问题时使用中文。

## 当前架构事实（2026-09-23）

本仓库是通用语音信号训练、推理和部署框架，当前支持五个任务：`se`、`aec`、`tse`、`ss`、`ssl`；公开 `data.mode` 共九种：`dynamic` / `offline`、`dynamic_aec` / `offline_aec`、`dynamic_tse` / `offline_tse`、`dynamic_ss` / `offline_ss`、`dynamic_ssl`（SSL 仅保留动态仿真模式）。

- 共享训练、数据、仿真、loss、metrics、artifact identity 和推理生命周期实现位于 `core/`；
- 任务实现只位于 `tasks/{se,aec,tse,ss,ssl}/`，不得恢复旧顶层 `se/`、`aec/`、`tse/`、`ss/`、`ssl` 兼容包；
- 训练入口统一由 `train.py` 根据 `data.mode` 和 `tasks.registry` 路由；
- Dynamic waveform 数据集的 Layer F 是 CPU-only 的 observation/device degradation；总开关关闭时整层跳过，训练侧不另设 GPU Layer F；
- SE 多通道 dynamic 仍由 `core/data/simulation/` 统一渲染；offline 配对数据支持 mono clean + channel-first `[C,T]` noisy，必须在 split 显式声明 `spatial_enabled`、array geometry、`reference_channel` 和 `channel_order`，collate 仅在 Dataset 为 spatial 时打开 `allow_spatial_noisy`，不重排通道；
- SE diffuse background 使用 `habets_spherical` coherence policy；`prob_point_source_background`（默认 0，要求 `spatial_enabled` + `rir_backend=online`）把空间背景按概率升级为同 room 点源 RIR，噪声 actor 由 `core/data/simulation/background_resolver.py` 在同 room 内采样并构造派生 noise scene plan 注入 `build_background_spec`；`prob_wind_noise_background`（默认 0，要求 `spatial_enabled`）把 wind 资产作为可分解的空间加性背景并复用 diffuse/point-source 链，和非加性 `prob_wind_noise_sidechain` 分开；空间 sidechain 第一阶段只开放 SSL/DOA，并通过 `NonDecomposableObservationContract` 生成严格 `[C,T]` control image；两项为 0 时零 rng 消耗、行为与 diffuse 默认完全一致；阵列 yaw / mic-jitter 配置显式进入 RIR plan identity；`target_type` 支持 `early` / `dry` / `rts`，其中 `rts` 仅由 SE spatial path 消费；spatial 背景能力的任务开放清单以 `core/data/simulation/capabilities.py` 的 `SPATIAL_FEATURE_CAPABILITIES` 声明矩阵为单一事实源（capability 门禁、配置期校验与文档消费同一张表），C-3 彩色噪声已接入全部五任务的 spatial 路径（合成 colored waveform 作为 noise actor 复用 diffuse/point-source BackgroundSpec 链，mono 路径实现不变）；`moving_source_speed_range_mps` 是房间内声源速度，独立于 Layer A 的 `prob_speed_perturb` / `speed_rate_range` 语速扰动；SE 轨迹采样带自适应缩速（重试按 0.9^n 缩速 + 端点投影回房间），实际速度可能低于配置下限；
- `ComplexIdealRatioMaskLoss` 是当前唯一从 NBSS 吸收的损失组件，使用 SE 参考通道与模型直接谱，不等于接入 NBC/NBC2 模型；`evaluate/oracle_beamformer.py` 提供 MVDR oracle 基线，mc_sim 生成器输出 mixture、clean reference 与 multichannel target image；本机 WSJ0 语料缺失时不得声称已完成真实语料评估。
- 文件推理统一使用 `core.inference.InferenceSpec`、`FileInferenceItem`、`PreparedOutput` 和 `InferenceReport`；
- 任务 Adapter 只处理任务模型输入输出，不处理 manifest、output lock、`skip-existing` 或 sidecar 发布；
- `core.inference` 负责 input inventory、artifact/source identity、reuse、原子输出、`run_manifest.json` 和 `inf.scp`；已加载模型的入口必须把 loader 冻结的身份（`ArtifactBinding.from_fingerprint` / `SourceBinding.from_fingerprint`）原样交给 runner，禁止改用路径声明并在 runner 内重新 fingerprint；item 的 `inputs` 与 `attributes` 一同进入严格 canonical JSON inventory，任一变化使整个 run 失效；音频 manifest 以 `MANIFEST_SCHEMA_VERSION` 显式版本化，旧版本按复用策略重推或报错；
- streaming 使用 `core.inference.streaming` 的 session seam，任务内部 cache/state 不得泄漏到公共合同；
- TSE sidecar 由 `tasks/tse/inference/sidecar.py` 根据公共 `InferenceReport` 发布；
- ONNX 是独立 artifact/runtime 发布 Module，使用统一 source identity，但不复用音频文件 manifest；
- 当前不保留旧训练、旧推理或旧场景配置的自动转换。旧字段必须在配置边界显式拒绝，旧测试只能验证 fail-fast，不能作为生产回退入口。

真实验证资产：

- 历史实验模型未纳入当前支持矩阵时，必须在配置/registry 边界显式拒绝；不得通过模型别名映射到结构不同的模型。外部历史 checkpoint 不等于当前架构支持。
- AEC DeepVQE 历史 checkpoint 已验证真实推理、streaming 和 ONNX 三路对齐；从历史配置重建训练时必须使用当前 canonical RIR 路径、split lifecycle 字段和场景概率键；
- AEC DeepVQE 一 epoch CPU smoke 已覆盖 dataset、collate、STFT、forward、loss、optimizer、validation 和 checkpoint 保存；
- 真实硬件/分布式门禁必须显式记录环境能力，不能把 socket/Gloo 或外部资源缺失描述为代码通过。

> 维护要求：新增/移除任务或 `data.mode` 时，必须同步更新本节（任务数、mode 清单）、「新增任务接入指南 → 任务合同矩阵」以及「新增任务接入指南 → 已知硬编码枚举点」中受影响的条目。

## 执行权限

| 操作 | 权限 |
|------|------|
| 读取文件、搜索代码、运行只读命令 | 自主执行 |
| 编辑文件 | 自主执行 |
| 任意 Git mutation（`add`、`commit`、`push`、`reset`、`rebase` 等） | **必须先经用户明确请求** |
| 删除文件或目录（agent 自己创建的测试/调试临时产物除外，见红线规范「测试临时产物必须清理」） | **必须先经用户确认** |
| 修改 `core/data/simulation_configs.py` 中的 `SimulationConfigs` 字段定义 | **必须先经用户确认** |
| 修改共享训练、数据、部署或安全基础设施：`core/trainer/{base_trainer,entry,metrics_trainer}.py`、`core/data/{collate,dataset_build,config_flags,simulation_configs}.py`、任务侧 `data/collate.py`/`data/batch.py`、`onnx/export.py`、`onnx/verify/verify_common.py`、`core/utils/safe_load.py`、`tools/shared/*` | **必须先经用户确认** |
| 安装/卸载 Python 包、修改环境 | **必须先经用户确认** |

用户对具体改动的明确请求即视为该改动本身的确认；「必须先经用户确认」仅适用于 agent 在执行过程中自主发起的、超出用户明确请求范围的改动。熔断机制第 3、4 条同样适用此原则：仅当改动是 agent 自行判断需要、而非用户直接要求时才触发暂停。

## 红线规范

### 安全红线（违反即阻断）

- **禁止**在代码中硬编码密码、token、API key、内网地址。
- **禁止**执行 `rm -rf`、`git push --force`、`git reset --hard` 等不可逆操作，除非用户显式授权。
- **本机实际路径允许硬编码**：允许在代码和配置中直接填写当前机器的数据、checkpoint、日志和评估资源绝对路径；数据集资源路径同时支持相对仓库根目录的 `data/...` 写法，与绝对路径等价，runtime 两种形式均支持。资源已存在时优先使用清晰、可直接运行的实际路径，未准备的资源可保留 `null` / `~`。允许提交路径字面量不等于允许提交资源本体；训练数据、checkpoint、日志和生成产物仍须由 `.gitignore` 隔离。
- **checkpoint/pickle 安全加载**：自有代码必须使用 `core.utils.safe_load.safe_torch_load` 或 `tools.shared.pickle_io.load_pkl_file`；禁止新增裸 `torch.load`/`pickle.load`。`allow_unsafe=True` 或 `PREPARE_SCP_ALLOW_UNSAFE_PICKLE=1` 仅限可信来源，并必须注明 `purpose`，禁止静默回退到不安全反序列化。
- **公共 checkpoint loader 必须 fail-closed**：`evaluate/checkpoint_loading.py` 对命中参数为 0、`missing_keys`、`unexpected_keys`、tensor shape 不匹配，以及 EMA `shadow_params` 数量、类型、dtype、shape 与 `model.parameters()` 不一致均必须在复制权重前报错；禁止导出、验证或推理部分随机初始化模型。
- **Trainer 加载语义按用途隔离**：完整 `resume` 为恢复模型、optimizer、scheduler、AMP/EMA 等可信训练状态，因 optimizer/scheduler 可能依赖 pickle，才允许显式 `allow_unsafe=True` 并注明 `purpose`；model-only pretrained 初始化必须保持 `weights_only=True` 安全模式，两条路径都严格校验模型 state，禁止把 resume 的 unsafe 权限复用于预训练权重加载。
- **推理 fingerprint 必须绑定实际 loader**：普通 checkpoint 在 loader 前后复核预先计算的内容身份；waveform-native 模型在构造后复核预先固化的全部 `*_ckpt_path` artifact。推理 source boundary 必须包含 `core/utils/inference_artifacts.py`，保证指纹、manifest 与 `--skip-existing` 校验实现的变化会使旧产物失配。冻结身份必须贯穿加载、执行与发布三处：runner 在推理前、发布 manifest 前（含 `allow_partial` 的中止路径）都要复核同一份身份，身份失配不得被降级为可发布结果，也不得按当前路径重新绑定后继续发布。
- **ONNX 发布是图与 manifest 的同一文件集事务**：ONNX 产物、manifest 与 source binding 必须在同一 per-target 锁内 staging、提交、复核与回滚，manifest 的 artifact identity 取自已验证 snapshot 而非重读当前 target，source binding 取自实际 loader 冻结身份；`require_manifest=True` 或调用方提供 manifest 时，publisher 不接受该参数必须直接报错，禁止静默发布无 provenance 的图。
- **显式例外（第三方官方比赛 ckpt 兼容加载）**：如 TSE `load_pretrained` 加载第三方官方比赛 checkpoint，允许在 `safe_torch_load` 安全模式 + 注明 `purpose` + 0 匹配 fail-closed 前提下保留非严格匹配加载；除此之外禁止新增裸 `torch.load` / 静默回退到不安全反序列化。
- **外部资源传输必须验证 TLS**：禁止新增使用 `ssl.CERT_NONE` 或关闭 hostname 校验的资源下载逻辑。
- **可选依赖必须隔离**：`evaluate/`、`onnx/`、`tools/` 中的可选第三方库必须使用 import guard；缺失时对应功能应明确跳过或降级，不得让基础 import、主训练路径或默认 unit 测试整体失败。

### 架构红线（违反即阻断）

- **DDP collective 一致性**：任何含 `all_reduce`/`broadcast`/`barrier` 的训练或验证路径，所有 rank 必须执行相同调用序列；门控只能依赖各 rank 一致的配置、`world_size` 或模型静态属性，禁止依赖本地数据为空或本地计数。空 rank 必须以零值参与聚合，不得提前 return。validation 的严格非空门禁必须在任何 collective 之前、按全局 SR 分布对所有 rank 对称判定并抛出同一错误；允许 rank-local dummy padding 的 train/default sampler 保持既有语义，不得把任一无真实 batch 的 rank 单独判死。
- **异常同步 collective 顺序**：DDP batch/validation 边界的异常同步必须先做状态 tensor 归约，只有全局失败时才收集异常字符串；成功路径不得执行 `all_gather_object`。所有 rank 在同一静态阶段调用相同序列。
- **DDP epoch-tail 梯度**：任何新 Trainer 的 epoch-tail 残差处理**必须**经过 `_handle_epoch_tail_gradients()`，不得自行实现 `world_size > 1` 守卫。遇到 NaN/Inf loss 时必须使用 `_graph_preserving_zero_loss()`：它只沿本次 forward 的可微输出图（由 `_capture_forward_graph()` 捕获）构造零 loss，未进入该图的参数交由 DDP reducer 处理；禁止图外连接参数或回退到 detached zero leaf 做 backward，无任何可微输出时直接报错。
- **GAN / 多优化器 step 配对**：每个 accumulation boundary（含 epoch-tail）必须复用统一 optimizer-step 语义：discriminator 先 `update_scaler=False, increment_step=False`，generator 后 `update_scaler=True, increment_step=True`；无关模型需先冻结，`generator_steps` 当前只允许为 1。
- **`_raw_model` 解包顺序**：DDP wrapping **必须**在 `torch.compile` 之后（当前 `core/trainer/base_trainer.py`）；`_raw_model()` 必须同时正确解包 `DDP.module` 与 compile 产生的 `_orig_mod`。改动包装顺序时需同步验证解包逻辑。
- **任务批合同（collate/batch）通用红线**：每个任务必须显式枚举合法样本格式；五任务 collate 统一返回已执行 `validate()` 的 Batch 对象，实现 `BatchContract` 的 `validate()`、`pin_memory()`、`.to(device, non_blocking=...)` 和 `batch_size`。SE/AEC 的样本仍为 tuple，TSE/SS/SSL 为 dict；Trainer 只接受对应 Batch 类型，位置 tuple/list 批显式拒绝；非法 tuple、缺失键或畸形字段必须在 collate/batch 边界 fail-fast，禁止静默截断或透传忽略；每个字段必须由消费端显式消费；已移除的旧格式必须保持显式拒绝，不得作为生产回退入口。各任务当前的合法格式与字段语义登记在「新增任务接入指南 → 任务合同矩阵」，新任务接入或合同变更必须同步更新该矩阵。
- **任务模型 I/O 通用红线**：STFT/iSTFT 由 trainer 或顶层模块统一管理，禁止模型内部重复引入；各任务的接口基线（基类签名、dict 谱进谱出字段集合或 waveform-native 合同）登记在「新增任务接入指南 → 任务合同矩阵」；dual-input、loss-returning、GAN 等例外形态必须由对应 Trainer 驱动并在 docstring 声明。
- **模型构造参数**：新增模型必须显式消费 `build_model` 注入的 STFT 参数（显式值与框架有效 STFT 上下文冲突时由 `core/models/build.py` 拒绝），对未知 kwargs 抛出 `TypeError`，禁止静默吞掉 YAML 拼写错误或废弃字段。
- **模型注册契约**：新增模型必须同时更新对应 `models/__init__.py` lazy map 和模块内 `@MODEL_REGISTRY.register()`，名称必须一致；别名也必须显式注册，并增加 registry/build smoke test。新任务的注册命名统一为 `_MODEL_REGISTRY_MAP` + `<TASK>_MODEL_TYPES` + `register_<task>_models(only=None)`，经 `core.models.register_lazy_models` 注册（可选依赖用 `optional_dependencies` 显式声明）；存量的 `_AEC_MODEL_REGISTRY_MAP` / `_TSE_MODEL_REGISTRY_MAP` 前缀命名是 legacy 形态，保留但不要求回填。
- **`find_unused_parameters`**：存在条件分支、按 batch 跳过模块或多路径输出的模型，必须在 entry/config 层显式推导和校验；无条件单路径模型不得随意设为 `true`。
- **流式 cache 与 I/O 契约**：所有 `onnx/models/streaming_*.py` wrapper 的 cache shape、顺序、输入输出名称和动态轴必须由 `init_cache()`/`init_cache_flat()`/`get_input_names()`/`get_output_names()` 等接口推导，禁止在 `onnx/export.py` 或验证脚本中硬编码。
- **ONNX cache metadata 写入**：支持 cache metadata 的流式导出分支必须统一通过 `write_streaming_runtime_metadata()` 从 `StreamingRuntimeSpec` 一次写入完整 runtime metadata；底层多 key 更新统一调用 `onnx.utils.write_onnx_metadata()`，禁止复制 subprocess 逻辑。对应 ORT 推理脚本必须启动时校验 metadata，缺失即 fail-fast。
- **ONNX 发布门禁**：正式发布 ONNX 必须经过对应验证脚本的完整 gate，禁止使用 `--no-verify` 发布；所有 `--save-onnx` 路径必须调用 `publish_verified_onnx()`，验证失败不得覆盖既有资产，临时 owned artifact 必须在异常路径清理。

### 数据、配置与仓库卫生约定

- **任务命名空间**：任务源码统一放在 `tasks/{se,aec,tse,ss,ssl}/`，导入使用 `tasks.<task>`，不提供旧顶层任务包兼容；配置、实验和数据目录分别保留 `configs/<task>/`、`exp/<task>/`、`data/<task>/` 语义，跨任务共享资产位于 `data/shared/`（clean、noise、rirs、LibriSpeech、urgent26_track1_se_dataset），任务独有大资产按任务归位（`data/aec/echo`、`data/tse/spk`），迁移时按路径用途逐项核对。

- **`data.mode` 单一事实来源**：新增模式必须在 `core/data/constants.py` 增加 `DATA_MODE_*` 常量，全仓库引用该常量；配置必须显式声明 `data.mode`，不得依赖各 entry 不一致的默认值；Dataset 配置统一使用 `data.train` / `data.validation`，已移除的 `data.offline` / `data.dynamic` section 不得再作为回退入口。
- **`SimulationConfigs` 字段校验**：新增区间型字段必须加入 `core/data/simulation_configs.py` 的 `_RANGE_ATTRS`；概率字段使用 `prob_*`/`*_prob` 命名并纳入 `[0,1]` 校验；仍需支持的字段重命名必须在 `from_dict` 保留兼容映射并补充 unit 测试；已明确移除的字段不得自动派生或静默接受，必须在配置边界显式拒绝。
- **离线生成器版本**：任何改变离线数据内容的 generator 修改都必须递增 generator version，使旧 manifest 指纹失效；禁止代码和数据生成语义改变后继续复用旧 manifest。
- **配置继承语义**：`base_config` 中 list 字段按整体替换，不会自动追加；需要追加时必须显式写出完整列表。配置命名、`exp_path` 和推理配置细节以 `configs/README.md` 为准。
- **TSE 模板白名单**：可分发训练模板仅限 `tests/test_configs.py::TSE_TEMPLATE_CONFIGS` 精确列出的配置；新增/改名必须同步更新白名单，禁止用未被 runtime 消费的模板标记绕过校验。
- **`.gitignore` 例外验证**：在 `core/data/`、`tasks/se/data/`、`tasks/aec/data/` 等可能被 `data/` 规则覆盖的源码目录新增文件后，必须用 `git check-ignore -v <path>` 或 `git status --ignored` 验证未被静默忽略；任何 `!` 例外必须验证实际生效。
- **Git 提交边界**：Git 提交不得包含实验结果、训练/评估数据集、数据清单、checkpoint、ONNX、日志、HTML、报告、图片、中间 artifact 或临时文件；上述内容默认通过 `.gitignore` 隔离。仅提交生成脚本、必要的小型 fixture，或用户明确要求纳入版本控制的固定报告。
- **测试临时产物必须清理**：运行测试、调试或验证时生成的临时文件和临时代码（如 `runs/`、`output/`、临时输出目录、一次性调试脚本）必须在任务完成后全部清理，不得残留在工作区。此类由 agent 自己创建的临时产物，清理时无需用户确认，不受「删除文件或目录需确认」限制；删除不属于本次任务产生的文件仍需用户确认。
- **测试目录与第三方目录卫生**：`tests/` 新增文件应是 pytest 测试并显式声明 marker；非 pytest 的运维、比较和验证 CLI 放在 `tools/ops/` 或 `tools/verify/`。第三方目录必须明确选择 submodule（含 `.gitmodules`）、占位目录或整目录忽略形态，不得混用。
- **`.kimi-code/skills/` 一律不提交**：`.kimi-code/skills/` 下的任何 skill（含 `ai-speech-paper`、`experiment-report` 及未来新增 skill）均为本机 agent 工作配置，禁止 git 提交入库；`.gitignore` 不得为该目录添加 `!` 例外，新增 skill 后必须用 `git check-ignore -v` 验证已被忽略。

## 新增任务接入指南

新增任务（下称 `<task>`）遵循**纯新增原则**：只新增 `tasks/<task>/`、`configs/<task>/` 与对应测试文件，外加三类登记点——`core/data/constants.py`（`DATA_MODE_*` 常量 + `_DATA_MODE_PROTOCOL` 条目）、`tasks/descriptors.py`（`TaskDescriptor`）、「已知硬编码枚举点」清单列出的位置。登记点的纯追加属于任务接入本身，不触发执行权限表中的共享设施确认；改动若超出纯追加（修改合同语义、校验逻辑或既有任务行为），按执行权限表与熔断机制第 3 条处理。`train.py`、`tasks/registry.py` 按 descriptor 自动路由，禁止为单个任务在共享层打补丁。

### 任务合同矩阵

各任务当前的批格式与模型 I/O 基线。**新增任务必须登记一行；既有任务合同变更必须同步更新本表**，并保证 collate/batch 代码、矩阵、契约测试三者一致。

| 任务 | data.mode | 批合同（collate → trainer） | 模型 I/O 基线 | streaming | 根文件推理 |
|---|---|---|---|---|---|
| `se` | `dynamic` / `offline` | tuple 样本 4/7（7 = 追加 `is_pure_noise`/`is_pure_echo`/`is_pure_echo_noise`）→ `core.data.collate.collate_fn` → `SEBatch` + `validate()`（定义于 `core/data/speech_enhancement_batch.py`）；5 元组旧样本显式报错；Trainer 不接受 tuple/list 批；offline spatial 只接受 mono clean + `[C,T]` noisy，并要求 split 显式给出 `spatial_enabled`、geometry、`reference_channel`、`channel_order` | 判别式继承 `BaseSEModel`（`tasks/se/models/base_model.py`）：`forward(noisy_spec, lengths=None, *, fs=None)`；`SpatialNet_SE` / `OnlineSpatialNet_SE` 接收 `[B,C,F,T,2]` 并输出 reference 谱；dual-input / loss-returning / GAN 为显式例外；`MossFormer2_SE_48K` 内部 iSTFT→Kaldi fbank 前端为登记例外；`CoFiLite` 为 waveform-native 判别式例外（模型内部 sqrt-Hann STFT/iSTFT，`meta.allow_no_stft: true`）；`ComplexIdealRatioMaskLoss` 仅消费显式 direct spectra 与 reference-channel noisy | ✓ | ✓ |
| `aec` | `dynamic_aec` / `offline_aec` | tuple 样本 7/11/12 → `aec_collate_fn` → `AECBatch` + `validate()`（7 = offline 无 flags，11 = 标准动态，12 = 追加帧级 `state`：`(T,)` int64 互斥类别索引，供 DBGNet 类 state detector）；混合有无 `state` 的样本显式拒绝；`fs`/`length` 为精确正整数标量，旗标为标量 bool 或有限数值 0/1；其余长度显式拒绝；Trainer 不接受 tuple/list 批，`state` 补齐为 `[B,T_state]` int64、padding=-1 | dict 谱进谱出：输入至少 `mic_spectra`/`farend_spectra`，输出 `nearend_estimate_spectra`；可选 `error_spectra`/`error_mask`/`state_probs` | ✓ | ✓ |
| `tse` | `dynamic_tse` / `offline_tse` | canonical dict item（`key`、`mixture`、`sr`、`length`、`targets`，可选 geometry/timebase/provenance）→ `tse_collate_fn` → `TSEBatch` + `validate()`；`mixture` `[B,C,T]`、`target` `[R,T]`、`num_targets` `[B]` int32 且 `R=sum(num_targets)`；cue 按 R 行 typed 对齐，启用但缺失的 slot 在 item 使用 `value=None/present=False`、batch 使用零占位 + `present=False`；tuple item/batch 一律拒绝 | registry 含两个模型：`TSE_BSRNN_SPK` / `TSE_Modular` 由 model I/O 边界按 speaker waveform/embedding 装配 `mix_spectra`、`enroll_spectra`/`enroll_wav` 或 `spk_embedding`，输出 `target_estimate_spectra`；双 self-estimated objective 由 `TSETrainer` 的 `trainer.objectives.dual_self_estimated` 消费；`TSE_Modular`（`models/modular_spec.py` 解析 + `models/discriminative/tse_modular.py`）已在 `models/__init__.py` 注册：runtime STFT 的 BSRNN，frontend 已接线 speaker（`usef`/`tfmap`/`context`/`spkemb`，`listen` 未接线）、spatial（`modules/spatial/`，IPD/CDF/SDF/ΔSTFT + cyclic/initstate）、visual（`modules/visual/`）与 textual（`modules/textual/`）；未实现组合由 `require_implemented=True` 在构造期抛 `NotImplementedError` | ✓（仅 `TSE_BSRNN_SPK` causal 路径；`TSE_Modular` 不支持 streaming） | ✓（`inference.py --tse-manifest` 使用严格 JSONL mixture/cue 清单，`TSEExtractor`/`TSEInferenceAdapter` 按模型 IO 合同读取 canonical cue role，公共 runner 管理 manifest/reuse，`sidecar.py` 发布 TSE metadata） |
| `ss` | `dynamic_ss` / `offline_ss` | dict item（`mixture` [T]/[C,T]、`sources`、`sr`、`length`，可选 `source_ids`）→ `ss_collate_fn` → `SSBatch`（`core/data/source_separation_batch.py`）+ `validate()`；批内固定 source 数、固定麦克风数、统一 sr；`source_mask` 全 True，拒绝 partial mask；item 的 `sr`/`length` 是正整数，`source_ids` 全批提供或全批省略且每项为非空字符串 | mixture → 固定 K 路源估计（顶层 `ss.num_sources` 配置，PIT 由 trainer 管理）；waveform-native（TDANet `[B,T]`）、external-STFT（TIGER `mixture_spectra` [B,F,T,2]，必须传 `lengths`）或多通道谱进谱出（`SpatialNet` 离线 / `OnlineSpatialNet` 因果，`[B,C,F,T,2]` → `est_sources_spectra` [B,K,F,T,2]） | ✗ | ✗（`SSExtractor`/`SSInferenceAdapter` + execution plan） |
| `ssl` | `dynamic_ssl` | dict item（必填 `features`/`target_dpipd`/`doa`/`vad`/`pair_index`/`pair_mask`/`geometry`/`input_lengths`/`output_lengths`/`frame_origin`/`time_stride`/`sample_rate`/`provenance`/`input_representation`/`target_representation`/`coordinate_frame`，可选 `source_distances`/`input_frame_index`/`output_frame_index`）→ `ssl_collate_fn` → `SSLBatch` + `validate()`：双特征合同——`multichannel_stft_real_imag` features `[B,T_in,F,2C]` 与 `reference_ipd` features `[B,T_in,F,2(C-1)]`（projection 单一入口 `project_ssl_scene` 按 `input_representation` 分支）；`target_dpipd` `[B,T_out,F,K,P,2]`（pair 轴不求均值，末尾 2 为复数实虚部）、`doa` `[B,T_out,K,2]`、`vad` `[B,T_out,K]`∈[0,1]、`pair_index` `[B,P,2]`、`pair_mask` `[B,P]` bool、`geometry` `[B,C,3]`，provenance 含 `backend`、sha256、`normalization`、`sound_speed_mps` 与 `geometry.reference_channel` 摘要字段；item 的 `sample_rate`/`input_lengths`/`output_lengths` 为精确整数且等于未补齐帧数，collate 统一补齐 `receptive_field`/`algorithmic_latency`/`valid_output_mask`（补齐尾帧在所有帧字段为零）；`input_frame_index`/`output_frame_index` 若提供必须为整数、严格递增且补齐帧为零；loss 门禁按模型类型区分：native `FixedArrayIPDNet` → `SSLSpatialMSELoss`、official（`FNSSL`/`IPDNetFixed`/`IPDNetVariable`/`IPDNet2`）→ `SSLDPIPDPITLoss`，均可显式选择 `SSLDOACompositeLoss`（辅助项要求模型声明并产出对应 head）；`ssl.num_tracks` 统一 train/validation/model 声源数，重复声明必须一致，loss `max_tracks` 是容量上限；`ssl.vad_mode` 当前只接受 `direct_path`，VAD 来源与阈值进入 provenance；`strict_load=true`、train/validation seed=`meta.seed` 不变，拒绝 waveform `trainer.val_metrics` | `FixedArrayIPDNet`（GRU 骨干，仅消费 `reference_ipd`）`forward(features [B,T,F,2(C-1)])` → `[B,T,F,K,P,2]`；`FNSSL`（官方 full/narrow LSTM 结构移植）`forward(multichannel_stft_real_imag [B,T,F,2C])` → 恢复 `[B,T_out,F,K,P,2]`，公共 `target_layout=trailing_ri`；`IPDNetFixed`（官方 `FNblock×2 + CausCnnBlock`，时间压缩 12 倍；online/offline 区分点是 narrow-band LSTM 方向性 `bidirectional = not is_online` 与归一化而非 CausCnnBlock，官方 online 输出布局与上游逐位一致；offline 变体使用 `global_mean` observation 归一化，Fixed 在 eval 时可显式调用 `offline_inference=True` 按 `n_seg` 分块，段长须为 12 的正整数倍）与 `IPDNetVariable`（官方 pair-batched pair-sharing，pair 内 4 通道由 `pair_channel_order` 选择、默认 `per_mic_interleaved`；上游 `VariableArrayIPDnet.py` 存在真实 shape bug，本地修正并登记）位于 `tasks/ssl/models/ipdnet_official.py`；`IPDNet2`（官方 OnlineSpatialNet + mamba_ssm Mamba 双槽位 8 层，频率压缩在 layer0 内、时间压缩 AvgPool(5,1)，与官方参考实现逐元素 bit-exact）位于 `tasks/ssl/models/ipdnet2.py`，`mamba_ssm` 为 optional import + 构造期显式 ImportError、CUDA-only、CPU 不可跑；near-field mapping 的 range 原点固定为 `geometry.reference_channel` 麦克风；`input_size` 口径按模型区分（native/FNSSL/IPDNet2 为 `C-1`，IPDNet2 内部换算 `dim_input=2C`；IPDNetFixed 为 `2C` real+imag；IPDNetVariable 固定 4），不匹配即 fail-fast；`SSLRuntimeInputs.model_kwargs` 为 forward 声明了 `pair_index` 形参的模型补 canonical `pair_index`（variable 的 pair-batched 合同）；模型与 loss 只接受 canonical `dpipd` 输出，`prediction`/`ipd_prediction` 旧别名直接 fail-fast；帧域特征由 dataset 产出，不消费外部 STFT | ✗ | ✗（无公开音频文件入口） |

TSE 补充合同：`TSE_Modular` 的 spatial frontend 位于 `tasks/tse/models/modules/spatial/`，包含 IPD/CDF/SDF/ΔSTFT、cyclic/initstate；visual frontend 仅接受预提取 features，textual frontend 仅接受 phoneme ids。`implementation_available` 之外的 tfgridnet/nbc2/dpccn/spex_plus 与 raw video 在 `require_implemented=True` 构造期抛 `NotImplementedError`，模型声明 `supports_multi_task=False`。TSE 推理的 cue role、representation、文件/inline JSON 来源在 `TSEExtractor`/`TSEInferenceAdapter` 边界校验；cue identity 与 speaker frontend 绑定到 `tse-inference-artifact-v6` / `tse-inference-manifest-v4`，sidecar 由 `tasks/tse/inference/sidecar.py` 发布。

### 标准目录骨架

```
tasks/<task>/
├── data/            # dataset（dynamic/offline 按需）+ collate + batch
├── models/          # __init__.py 注册门面 + 具体模型（@MODEL_REGISTRY.register()）
├── trainer/         # entry.py（UnifiedTrainingEntry 子类）+ <task>_trainer.py
└── inference/       # 可选：adapter.py（无根文件推理入口时省略）
```

能复用 `core/` 就不在任务侧新建：SE 的 collate/Batch 直接复用 `core.data.collate` / `core.data.speech_enhancement_batch`、SS 的 `SSBatch` 直接复用 `core.data.source_separation_batch`。任务侧文件只承载任务特有语义。

### 接入 Checklist

1. **data.mode**：`core/data/constants.py` 增加 `DATA_MODE_DYNAMIC_<TASK>` / `DATA_MODE_OFFLINE_<TASK>`（按需）并登记 `_DATA_MODE_PROTOCOL`（task/storage/task_mode）；全仓库只引用常量，配置显式声明 `data.mode`。
2. **TaskDescriptor**：在 `tasks/descriptors.py` 按字段语义登记；registry 启动自动校验任务名与 mode 唯一。
3. **数据层**：dataset 按 mode 提供（优先复用 `core/data` 的 dynamic/offline 基座与 `_build_loaders_common`）；样本格式及字段在矩阵登记，collate 返回已校验的任务 Batch，实现 `BatchContract` 全部接口；锁页和设备搬运覆盖每个 tensor（含嵌套 cue），并保留类型、主样本数和元数据；合法格式显式枚举、非法 fail-fast；新增字段必须被消费端显式消费。
4. **模型层**：`models/__init__.py` 提供 `_MODEL_REGISTRY_MAP` + `<TASK>_MODEL_TYPES` + `register_<task>_models(only=None)`；模型类 `@MODEL_REGISTRY.register()` 且与 map 键一致；显式消费 `build_model` 注入的 STFT 上下文（与有效 STFT 冲突会被 `core/models/build.py` 拒绝），未知 kwargs 抛 `TypeError`；需要执行计划/几何校验的模型按 `core/models/io_contract.py` 声明 `IO_CONTRACT`；接口基线在矩阵登记。
5. **Trainer**：`entry.py` 继承 `core.trainer.entry.UnifiedTrainingEntry`，实现抽象方法 `build_loaders`（返回 `(train_loader, val_loader, train_sampler, train_ds)`）、`build_model`、`get_trainer_cls`；支持 GAN 时覆写 `_default_gan_trainer_cls`；需要 train dataset 事实时覆写 `build_model_for_training`；Trainer 类继承 `MetricsCapableTrainer`/`BaseTrainer`。epoch-tail 走 `_handle_epoch_tail_gradients()`、NaN/Inf 用 `_graph_preserving_zero_loss()`、DDP collective 一致性与 `find_unused_parameters` 推导按架构红线执行。
6. **推理（可选）**：实现 `core.inference.contracts.InferenceAdapter` 协议（`infer(FileInferenceItem) -> PreparedOutput`），工厂 `"tasks.<task>.inference.adapter:build_inference_adapter"` 写入 descriptor；manifest、output lock、`skip-existing`、sidecar 由 `core.inference` 负责，adapter 只处理任务模型 IO。streaming 实现 `StreamingAdapter`（`input_roles` + `open_session()`），任务 cache/state 不得泄漏进公共合同。无根文件推理入口时 `file_inference_adapter_factory=None` 并配好 `root_file_inference_error` 文案；推理 fingerprint 与 source boundary 按安全红线登记。
7. **ONNX（可选）**：streaming wrapper 的 cache shape、I/O 名称与动态轴由 `init_cache*()`/`get_*_names()` 推导；metadata 经 `write_streaming_runtime_metadata()` 一次写入；发布必须过 `publish_verified_onnx()` gate，禁止 `--no-verify`。
8. **配置**：`configs/<task>/` 下配置显式声明 `data.mode`；推理配置用 `*_infer.yaml` + `meta.inference_only: true`；`model.args` 键必须是模型 `__init__` 的显式形参（`tests/test_configs.py` 静态校验）；任务级顶层 section（如 SS 的 `ss.num_sources`）在 entry 的 `_validate_config` 显式校验；命名、`base_config` 继承（list 整体替换）、`exp_path` 约定以 `configs/README.md` 为准，并同步更新其任务总览表与目录结构。
9. **契约测试与枚举更新**：逐项更新「已知硬编码枚举点」；新增测试全部显式声明 marker；至少包含：collate/batch 合同测试（合法格式 + 非法拒绝）、registry/build smoke（任务包 lazy 导入不加载具体模型、`only=` 注册）、trainer entry 合同。
10. **文档**：更新「当前架构事实」（任务数、mode 清单）、「任务合同矩阵」、`configs/README.md`；任务特有规则沉淀在任务内 README（受 `.gitignore` 的 Markdown 例外约束），跨任务规则回本文件。

### TaskDescriptor 字段语义

在 `tasks/descriptors.py` 追加一条 `TaskDescriptor`；`tasks/registry.py` 启动时自动校验任务名与 `data_modes` 不重复。字段填写规则：

| 字段 | 填写规则 |
|---|---|
| `name` | 任务短名，与 `tasks/<task>/`、`configs/<task>/` 目录名一致 |
| `data_modes` | `frozenset` 的 `DATA_MODE_*` 常量；每个 mode 全仓库只能归属一个任务 |
| `training_entrypoint` | `tasks.<task>.trainer.entry.<Task>TrainingEntry` 点路径，registry lazy import |
| `model_registry_module` / `model_registrar` | `tasks.<task>.models` / `register_<task>_models` |
| `config_root` | `configs/<task>`，必须真实存在（`tests/test_task_namespace.py` 校验） |
| `supports_streaming` / `supports_external_stft` | 是否实现 streaming session seam / 模型是否消费框架 STFT 上下文 |
| `file_inference_adapter_factory` | `"tasks.<task>.inference.adapter:build_inference_adapter"`；无根文件推理入口置 `None` |
| `root_file_inference_error` | factory 为 `None` 时的报错文案，必须指向任务专用入口（参照 tse/ss/ssl） |

### 已知硬编码枚举点（新任务必须同步更新）

**代码与配置登记点**：

- `core/data/constants.py`：`DATA_MODE_*` 常量 + `_DATA_MODE_PROTOCOL`
- `tasks/descriptors.py`：`TaskDescriptor`
- `core/models/io_contract.py`：`ModelIOContract.tasks` 支持任务集合
- `core/data/dataset_contract.py`、`core/data/simulation_config_contract.py`：支持任务集合（`dataset_contract.py` 当前不含 ssl，SSL 走 spatial 专用路径；`simulation_config_contract.py` 已含全部五任务；新任务是否加入取决于是否复用对应合同）
- `core/data/simulation_views.py`、`core/data/simulation/spatial_background.py`：五任务空间仿真门控
- `core/models/execution_plan.py`：`_DIRECT_TRAINER_DEFAULT_MODES`（Trainer 直连兼容路径默认值）
- `evaluate/latency_verification.py`：延迟验证任务门控（仅按需扩展）

**契约测试**：

- `tests/test_task_namespace.py`：`len(iter_tasks()) == 5` 断言、资源忽略任务列表、data.mode 路由与仿真 trace 的 parametrize
- `tests/test_model_registry_contract.py`：lazy 导入断言的任务包列表、register/build parametrize、lazy map 合并断言
- `tests/test_inference_routing_contract.py`：mode/model/task 路由 parametrize
- `tests/test_inference_factory_contract.py`：无根入口任务 fail-fast parametrize（新任务无根文件推理时）
- `tests/test_configs.py`：per-task entry dispatch 链与 `_validate_inference_only_contract` 的 register dispatch；如需可分发训练模板，参照 `TSE_TEMPLATE_CONFIGS` 新建 `<TASK>_TEMPLATE_CONFIGS` 白名单

**文档与忽略规则**：

- `configs/README.md`：任务总览表与目录结构
- 本文件「当前架构事实」「任务合同矩阵」
- `.gitignore`：资源忽略依赖全局后缀规则与 `!tasks/*/data/` 例外自动生效，但必须用 `git check-ignore -v` 验证；任务内 README 由 `!tasks/*/README.md` 例外统一纳入版本控制（默认 `*.md` 被忽略，深层 `tasks/tse/*/README.md` 单独放开）

### 验收门禁

全部通过才算接入完成：

- `python -m pytest -m unit tests/test_task_namespace.py tests/test_model_registry_contract.py tests/test_inference_routing_contract.py tests/test_inference_factory_contract.py tests/test_configs.py -v`
- `python -c "from tasks.<task>.trainer.entry import <Task>TrainingEntry; print('OK')"`
- 任务新增 unit 测试全绿
- 真实数据可用后补一 epoch CPU smoke，覆盖面参照 AEC DeepVQE smoke（dataset/collate/STFT/forward/loss/optimizer/validation/checkpoint 保存）

## 验证策略

修改后**必须**执行与其影响面匹配的验证。无测试套件的模块至少执行 import 检查。

| 变更类型 | 必须执行的验证 |
|----------|---------------|
| 任何 `.py` 修改 | `python -c "import <changed_module>; print('OK')"`；仅修改 CLI 说明时另执行 `python <script> --help` |
| 模型架构（`tasks/aec/models/`、`tasks/se/models/`、`tasks/tse/models/`、`core/models/`） | import 检查 + 与改动相关的显式 `unit` 测试；需要数据、GPU 或外部模型的测试按资源条件显式选用对应 marker |
| UniPASE/PASE 系列（`tasks/se/models/pase_family/`，含 GAP-URGENet） | import 检查 + `python -m pytest -m unit -k "pase or stupase or unipase or gap"` |
| 配置文件（`configs/`） | `python -m pytest -m unit tests/test_configs.py -v` |
| 数据管道（`tasks/aec/data/`、`tasks/se/data/`、`tasks/tse/data/`、`core/data/`） | import 检查 + 相关 `unit` 测试；例如 `python -c "from tasks.se.data.dynamic_dataset import DynamicMixingDataset; print('Dataset import OK')"` |
| ONNX 导出/验证（`onnx/`） | `python onnx/export.py --help` + `python -m pytest -m unit tests/test_verify_3way_checkpoint_gate.py -v`；实际导出和三路验证须显式准备对应模型与资源 |
| Trainer（`core/trainer/`、`tasks/se/trainer/`、`tasks/aec/trainer/`、`tasks/tse/trainer/`） | entry import 检查 + 相关 `unit` 测试；共享训练骨架变更还需覆盖 `trainer`/`ddp`/`scheduler` 相关测试 |
| TSE 模块（`tasks/tse/`） | `python -m pytest -m unit -k "tse"` + `python -c "from tasks.tse.trainer.entry import TSETrainingEntry; print('TSE trainer entry OK')"` |
| 新增任务接入 | 「新增任务接入指南 → 验收门禁」列出的全部命令 |
| Audio Filter Pipeline | `PYTHONPATH=tools python -m pytest tools/audio_filter_pipeline/tests -q` |
| Noise Filter Pipeline | 先激活 `noise_filter_pipeline` 环境（或设置等价导入路径），再显式运行 `python -m pytest tools/noise_filter_pipeline/tests -q -m "pipeline or unit"`（根 `pyproject.toml` 的 `addopts = -m "unit"` 会静默反选仅带 `pipeline` marker 的用例） |
| 多文件变更 | 执行上述相关检查，并用 `git diff --check` 与 `git diff -- <files>` 核对范围和意图 |

根目录 `pyproject.toml` 默认通过 `addopts` 仅选择 `unit` 测试；新增 pytest 文件/函数必须显式声明 marker（清单以 `pyproject.toml` 的 `markers` 为准：`unit`/`contract`/`integration`/`subprocess`/`ddp`/`ort`/`gpu`/`external`/`pipeline`），未标记测试会被默认过滤。非 `unit` 测试必须显式选择 marker，并确认对应数据、硬件、环境或外部资源可用。

验证失败时**不得**继续后续修改；先定位并修复问题，或向用户报告。

> **配置约定**：推理配置使用 `*_infer.yaml` 命名并标记 `meta.inference_only: true`；所有配置显式声明 `data.mode`，模式常量统一来自 `core/data/constants.py`。`base_config` 继承中的 list 字段按整体替换，不会自动追加。TSE 可分发训练模板仅限 `tests/test_configs.py::TSE_TEMPLATE_CONFIGS` 精确白名单，禁止使用未被 runtime 消费的模板标记绕过校验。YAML 注释遵循 `configs/README.md`「YAML 注释规范」，检查器为 `python tools/verify/check_config_comments.py`，并由 `tests/test_configs.py::test_yaml_comment_contract` 门禁。

## 熔断机制

以下任一条件触发，**立即停止并请求人工确认**：

1. **连续报错 ≥ 3 次**：同一问题修复 3 次仍未通过验证，停止循环修补。
2. **破坏性操作**：任何涉及 `rm`、`git reset --hard`、`git push --force` 的操作。
3. **训练/共享框架变更**：修改 `tasks/se/trainer/entry.py`、`tasks/aec/trainer/entry.py`、`tasks/tse/trainer/entry.py`、`core/trainer/{base_trainer,entry,metrics_trainer}.py`，或修改 YAML 配置的**结构语义**（字段重命名/删除、`base_config` 继承关系、`data.mode` 等被 runtime 消费的键），或修改**正在被运行中实验引用**的配置。新增实验配置或仅调整未运行实验的超参数值不触发本条，但仍须通过 `tests/test_configs.py` 验证。
4. **部署共享基础设施变更**：修改 `onnx/export.py`、`onnx/verify/verify_common.py` 或共享导出/验证测试，可能影响已发布资产时暂停确认。
5. **依赖变更**：修改 `requirements.txt` 或 import 新的第三方库。
6. **范围漂移**：修复过程中发现需要改动 > 5 个文件或 > 200 行时，暂停并报告影响面，等用户确认。统计口径以 `git diff --stat` 为准，计入源码与既有测试文件的改动；纯文档改动、批量机械性替换以及本任务新建的附属测试文件不计入阈值。

## 提交信息规范

仅在用户明确要求提交时，使用以下 Conventional Commit 格式：

```
<type>(<scope>): <subject>
```

- **type**：`feat` / `fix` / `docs` / `refactor` / `test` / `chore`
- **scope**：`aec` / `se` / `tse` / `onnx` / `data` / `trainer` / `core`
- **subject**：英文小写，动词开头，≤ 72 字符

示例：

```
fix(aec): guard ddp epoch-tail gradients with world_size > 1
feat(onnx): add unified streaming metadata writer
```
