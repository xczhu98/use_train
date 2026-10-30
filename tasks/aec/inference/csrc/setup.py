"""Build script for the C++ delay estimator extension.

Cross-platform (Linux / Windows / macOS).

Usage:
    cd <project_root>                       # e.g. d:/code/use_train
    pip install -e tasks/aec/inference/csrc/      # recommended: editable install
    # or:
    python tasks/aec/inference/csrc/setup.py build_ext --inplace
"""

import os
import sys
from setuptools import setup, Extension

import pybind11
import numpy as np
from setuptools.command.build_ext import build_ext as _build_ext

_csrc_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_csrc_dir, "../../.."))
_native_dir = os.path.join(_project_root, "aec", "inference", "_native")
_src = os.path.join(_csrc_dir, "delay_estimator_ext.cpp")


class _NativeDirBuildExt(_build_ext):
    """将 inplace 扩展固定写入 tasks/aec/inference/_native/，避免加载同名旧产物。"""

    def _get_inplace_equivalent(self, build_py, ext):
        _, regular_file = super()._get_inplace_equivalent(build_py, ext)
        filename = os.path.basename(self.get_ext_filename(ext.name))
        os.makedirs(_native_dir, exist_ok=True)
        return os.path.join(_native_dir, filename), regular_file

if sys.platform == "win32":
    _compile_args = ["/std:c++17", "/O2", "/DNDEBUG", "/EHsc", "/D_USE_MATH_DEFINES"]
    _link_args = []
else:
    _compile_args = ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC"]
    _link_args = []

ext = Extension(
    "delay_estimator_ext",
    sources=[_src],
    include_dirs=[pybind11.get_include(), np.get_include()],
    language="c++",
    extra_compile_args=_compile_args,
    extra_link_args=_link_args,
)

setup(
    name="delay_estimator_ext",
    version="0.1.0",
    description="C++ accelerated AEC delay estimators (GCC-PHAT+ACPS & MatchedFilter)",
    ext_modules=[ext],
    cmdclass={"build_ext": _NativeDirBuildExt},
    zip_safe=False,
)
