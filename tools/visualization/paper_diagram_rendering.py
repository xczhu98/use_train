"""ArchScope 论文方法图的窄域检查、栅格化与发布工具。"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any


FONT_SANS = "Noto Sans CJK SC, Noto Sans CJK, sans-serif"
FONT_MONO = "Noto Sans Mono CJK SC, Noto Sans Mono CJK, monospace"


@dataclass(frozen=True)
class RenderReceipt:
    svg_path: Path
    png_path: Path
    resvg_bin: Path
    pair_atomic: bool = False


def load_archscope() -> Any:
    """按可选依赖加载 ArchScope，并提供可操作的缺失诊断。"""
    try:
        return importlib.import_module("archscope")
    except ImportError as exc:
        raise RuntimeError(
            "无法导入 ArchScope；请在当前 Python 环境安装或提供 archscope。"
            "Markdown 中的 Mermaid 仍是可独立阅读的语义真相。"
        ) from exc


def resolve_resvg(explicit: str | Path | None = None) -> Path:
    """按显式参数、RESVG_BIN、PATH 的顺序查找可执行的 resvg。"""
    candidates = (
        ("--resvg", explicit),
        ("RESVG_BIN", os.environ.get("RESVG_BIN")),
        ("PATH", shutil.which("resvg")),
    )
    diagnostics: list[str] = []
    for source, candidate in candidates:
        if not candidate:
            diagnostics.append(f"{source}=未设置")
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        diagnostics.append(f"{source}={path}（不是可执行文件）")
    raise RuntimeError(
        "找不到 resvg；请通过 --resvg 显式传入、设置 RESVG_BIN，"
        f"或将 resvg 加入 PATH。诊断：{'; '.join(diagnostics)}"
    )


def _check_diagram(diagram: Any) -> None:
    checks = {
        "check": diagram.check(),
        "check_arrows": diagram.check_arrows(),
        "check_edges_through_boxes": diagram.check_edges_through_boxes(),
    }
    failures = {name: result for name, result in checks.items() if result}
    if failures:
        raise RuntimeError(f"ArchScope 检查失败：{failures}")


def _normalize_svg_fonts(path: Path) -> None:
    source = path.read_text(encoding="utf-8")

    def replacement(match: re.Match[str]) -> str:
        family = FONT_MONO if "mono" in match.group(1).lower() else FONT_SANS
        return f'font-family="{family}"'

    normalized = re.sub(r'font-family="([^"]*)"', replacement, source)
    path.write_text(normalized, encoding="utf-8")


def _validate_targets(svg_path: Path, png_path: Path) -> None:
    if svg_path == png_path:
        raise ValueError("SVG 与 PNG 发布路径必须不同")
    if svg_path.suffix.lower() != ".svg":
        raise ValueError(f"SVG 发布路径必须以 .svg 结尾：{svg_path}")
    if png_path.suffix.lower() != ".png":
        raise ValueError(f"PNG 发布路径必须以 .png 结尾：{png_path}")
    if svg_path.parent != png_path.parent or svg_path.stem != png_path.stem:
        raise ValueError("SVG 与 PNG 必须使用同目录、同 stem 的成对发布路径")


def _restore_target(path: Path, previous: bytes | None) -> None:
    if previous is None:
        if path.exists():
            path.unlink()
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.rollback-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(previous)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_pair(staged_svg: Path, staged_png: Path, svg_path: Path, png_path: Path) -> None:
    previous = {
        svg_path: svg_path.read_bytes() if svg_path.exists() else None,
        png_path: png_path.read_bytes() if png_path.exists() else None,
    }
    try:
        os.replace(staged_svg, svg_path)
        os.replace(staged_png, png_path)
    except BaseException:
        restore_errors: list[str] = []
        for path in (svg_path, png_path):
            try:
                _restore_target(path, previous[path])
            except BaseException as exc:
                restore_errors.append(f"{path}: {exc}")
        if restore_errors:
            raise RuntimeError(
                "方法图发布失败，且回滚不完整：" + "; ".join(restore_errors)
            )
        raise


def render_diagram(
    diagram: Any,
    *,
    svg_path: str | Path,
    png_path: str | Path,
    resvg_bin: str | Path | None = None,
    zoom: float = 2.0,
) -> RenderReceipt:
    """检查并发布 SVG/PNG；单文件 rename 原子，但双文件切换并非瞬时原子。"""
    svg_target = Path(svg_path).expanduser().resolve()
    png_target = Path(png_path).expanduser().resolve()
    _validate_targets(svg_target, png_target)
    if zoom <= 0:
        raise ValueError(f"zoom 必须大于 0，实际为 {zoom}")

    _check_diagram(diagram)
    resvg = resolve_resvg(resvg_bin)
    svg_target.parent.mkdir(parents=True, exist_ok=True)
    png_target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".paper-diagram-render-", dir=svg_target.parent
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        staged_svg = temporary_root / "method-overview.svg"
        staged_png = temporary_root / "method-overview.png"
        diagram.save(staged_svg, png=False, on_overlap="raise")
        if not staged_svg.is_file() or staged_svg.stat().st_size == 0:
            raise RuntimeError("ArchScope 未生成非空 SVG")
        _normalize_svg_fonts(staged_svg)
        subprocess.run(
            [
                str(resvg),
                "--sans-serif-family",
                "Noto Sans CJK SC",
                "--monospace-family",
                "Noto Sans Mono CJK SC",
                "--languages",
                "zh-CN",
                "--zoom",
                str(zoom),
                str(staged_svg),
                str(staged_png),
            ],
            check=True,
        )
        if not staged_png.is_file() or staged_png.stat().st_size == 0:
            raise RuntimeError("resvg 未生成非空 PNG")
        _publish_pair(staged_svg, staged_png, svg_target, png_target)

    return RenderReceipt(
        svg_path=svg_target,
        png_path=png_target,
        resvg_bin=resvg,
    )
