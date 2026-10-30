# Audio Filter Pipeline 内部结构

本管道按“入口 → 配置 → 阶段实现 → 工具与产物”的方向组织。新的调用应从
`cli/` 进入，阶段实现保持在 `processors/stages_v3/`，避免把编排逻辑重新放回
脚本入口。

```text
audio_filter_pipeline/
├── cli/                         # canonical CLI（python -m ...cli）
├── config/                      # schema、selection policy、YAML 与生成器
│   ├── schema.py
│   ├── selection_policy.py
│   ├── generate_configs.py
│   └── *.yaml
├── processors/                  # 阶段处理器与可选 runtime
│   └── stages_v3/               # Step0–Step6 DAG 实现
├── utils/                       # artifact、锁、路径和身份工具
└── tests/                       # 管道契约与阶段测试
```

`config/__init__.py` 继续导出原有 `audio_filter_pipeline.config` 符号，因此外部
调用应直接使用 canonical 包；配置实现放进 `config/schema.py`，策略解析放进
`config/selection_policy.py`。

推荐入口：

```bash
PYTHONPATH=tools python -m audio_filter_pipeline.cli --help
```
