"""Noise filter pipeline 测试包收集兼容。"""

from __future__ import annotations

import sys

from tools import noise_filter_pipeline as _noise_filter_pipeline

# 从仓库根收集时包名为 tools.noise_filter_pipeline；为产品代码使用的 canonical
# 绝对导入注册同一包对象，避免依赖外部 PYTHONPATH=tools。
sys.modules.setdefault("noise_filter_pipeline", _noise_filter_pipeline)
