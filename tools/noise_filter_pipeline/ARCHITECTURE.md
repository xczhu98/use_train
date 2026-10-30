# Noise Filter Pipeline 内部结构

本管道按“CLI → core 编排 → stages → processors”的方向组织。`core/` 是本管道
自己的运行时核心，负责配置、manifest、锁和阶段调度；仓库根目录的 `core/`
仍是训练框架核心，两者不要互相移动或混用。

```text
noise_filter_pipeline/
├── cli/                         # canonical Stage A–D CLI
├── config/                      # registry 与运行配置
├── core/                        # 本管道的 manifest、配置、锁和调度器
├── stages/                      # Stage A、B、C、D 编排层
├── processors/                  # 可复用处理器与模型 runtime
├── scripts/                     # 维护脚本与 HTML 生成器
├── utils/                       # 管道级小工具
└── tests/                       # 管道契约与阶段测试
```

阶段之间通过 `core` 的 manifest 和 execution contract 传递事实；不要在
`scripts/` 里直接调用 processor 绕过 stage runner。HTML 是文档产物，生成逻辑
位于 `scripts/generate_pipeline_html.py`，配置和输出路径由管道根目录解析。

推荐入口：

```bash
PYTHONPATH=tools python -m noise_filter_pipeline.cli --help
PYTHONPATH=tools python -m noise_filter_pipeline.scripts.generate_pipeline_html --help
```
