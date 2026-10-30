# `tools/` 目录治理

`tools/` 不是一个单一模块，而是几个不同生命周期的工具域。整理时应先固定
各域的接口和依赖方向，再逐个搬迁文件；不要按文件名或创建时间把所有脚本重新
归档。

## 当前域划分

| 域 | 当前目录 | 允许承载的内容 | 主要调用者 |
|---|---|---|---|
| 数据生产 | `prepare_data_tools/` | SCP、离线数据、RIR、manifest 和 publication | 训练数据层、数据准备 CLI |
| 模型资产 | `model_tools/` | checkpoint 转换、提取、平均、复杂度审计 | 评估、部署和人工迁移流程 |
| 验证门禁 | `verify/` | 只读检查、质量门、兼容性验证、审计合同 | 测试和人工验证流程 |
| 独立管道 | `audio_filter_pipeline/`、`noise_filter_pipeline/` | 有自己配置、阶段、运行状态和测试的长流程 | 对应管道 CLI |
| 通用适配 | `shared/` | 被 `core/`、`tasks/`、`evaluate/` 直接复用的文件/音频/SCP/受限 pickle 适配器 | 主框架与工具域 |
| 运维与分析 CLI | `ops/` | 按数据、报告、评估、任务和维护分组的一次性 CLI | 人工执行 |

环境能力探测和论文图工具分别归入 `environment/` 与 `visualization/`；调用方必须
直接依赖 canonical 模块，根路径不再提供兼容入口。

## 依赖方向

工具域之间只允许向下依赖：

```text
core / tasks / evaluate
          │
          ▼
       shared
          ▲
          │
prepare_data_tools   model_tools   verify   ops   独立管道
```

实际约束如下：

1. `shared/` 只能提供稳定的小接口，不能放任务规则、实验逻辑或某条管道的
   状态机。它已经被主框架直接使用，因此是一个需要兼容期的公共 seam，不能
   直接整体改名。
2. `verify/` 可以读取生产域并复用其合同，但生产代码不能反向依赖某个验证
   CLI 的实现。
3. `prepare_data_tools/` 负责生成和发布数据事实；只读检查放在 `verify/`，
   不把生成逻辑复制到验证脚本。
4. `audio_filter_pipeline/` 和 `noise_filter_pipeline/` 是独立的深模块：它们
   的外部接口是配置、CLI 和发布产物，内部的 `core/`、`processors/`、
   `stages/` 不应被其他工具直接导入。需要复用的逻辑应下沉到 `shared/` 或
   仓库的 `core/`，并经过单独合同测试。
## 目标目录

当前实现已经收敛到下面的概念结构：

```text
tools/
├── prepare_data_tools/       # 数据生产与发布
├── model_tools/              # 模型资产
├── verify/                   # 验证与审计门禁
├── audio_filter_pipeline/    # 独立长流程
├── noise_filter_pipeline/    # 独立长流程
├── shared/                   # 兼容期公共适配器，禁止继续膨胀
├── ops/                      # 一次性运维与分析 CLI，按域分组
├── environment/              # 环境能力探测
└── visualization/            # 论文方法图工具
```

目录名本身不是重点，重点是让每个模块只有一个归属，并且让调用者只依赖域的
外部接口。

## 入口治理

新增 CLI 必须归入已有域，并以 `python -m tools.<domain>...` 或管道自己的 canonical CLI 运行。

## 分阶段修复顺序

### 第 0 阶段：冻结边界（现在执行）

- 新增 CLI 必须归入已有域；禁止继续向 `tools/` 根目录添加业务 Python 文件。
- 新 CLI 使用 `python -m tools.<domain>...` 作为规范入口，并在域 README 中
  记录配置、输入、输出和失败语义。
- 运行产物、HTML、图片、checkpoint 和日志放在仓库外的输出根；源码目录只
  保留必要的静态图源和文档。
- 每次迁移先用 `rg` 找到 import、subprocess 和文档命令，再移动实现、增加
  兼容包装，最后删除包装；不直接批量改名。

### 第 1 阶段：清理可见入口

后续只维护 `ops/` 中的实现；新增或迁移 CLI 按组运行对应单元测试和
`python -m ... --help`。这一步不会触碰 `shared/`，避免影响主训练和推理路径。

### 第 2 阶段：收紧公共 seam

为 `shared/` 中的每个模块写出最小接口和输入/输出合同，禁止新增管道专用函
数。需要改变路径时，先在 `core/utils` 或 `core/data` 建立新 seam，再让旧
模块转发；完成全仓库 import 迁移和测试后才删除旧路径。

### 第 3 阶段：处理静态资产

保留会影响发布或文档的 DOT/SVG/PNG 源文件，把运行时生成的 HTML、报告、缓存
和临时音频移到输出根并加入精确 `.gitignore`。不要用宽泛规则掩盖源码；新增
例外后用 `git check-ignore -v <path>` 验证。

## 每次迁移的验收条件

- `rg` 不再发现生产代码或测试依赖已移除的旧路径。
- 每个 CLI 只有一个 canonical 实现，不保留第二份业务入口。
- 域 README、根 `tools/README.md` 和命令示例同步更新。
- 运行相关的 unit/contract 测试，并清理临时输出。
- `git diff --stat` 只包含本次域迁移；不提交数据、checkpoint、日志或生成
  产物。
