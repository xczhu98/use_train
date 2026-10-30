#!/usr/bin/env python3
"""生成 URGENT2026 训练、推理与评估工作流解释图（非论文原图）。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from archscope import Block, Diagram, IOLabel, style


ARCHSCOPE_SOURCE = "HaozheZhang6/archscope@04bae8488aa30fb694c9c14abc35e6f4d8900328"


def find_repo_root(script_path: Path) -> Path:
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "AGENTS.md").is_file():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


ROOT = find_repo_root(Path(__file__).resolve())
OUTPUT_SVG = Path(__file__).with_suffix(".svg")
OUTPUT_PNG = Path(__file__).with_suffix(".png")
RESVG = ROOT / ".kimi-code/tools/resvg"


def place(diagram: Diagram, label: str, *, ident: str, x: float, y: float,
          kind: str, sub: str, min_w: float = 0, h: float | None = None) -> None:
    diagram.place(
        Block(label, id=ident, kind=kind, sub=sub, min_w=min_w, h=h), x=x, y=y
    )


def frame(diagram: Diagram, *, x: float, y: float, width: float, height: float,
          title: str, tint: str, stroke: str) -> None:
    diagram.doc.rect("frames", x, y, width, height, tint, stroke, 1.2, rx=12)
    diagram.doc.text(
        "labels", x + 16, y + 20, title, style.T_SECTION, stroke, "600", anchor="start"
    )


def normalize_svg_fonts() -> None:
    svg = OUTPUT_SVG.read_text(encoding="utf-8")
    svg = svg.replace(
        "Helvetica Neue, Helvetica, Arial, sans-serif", "Noto Sans CJK SC"
    ).replace(
        "SF Mono, Menlo, Consolas, monospace", "Noto Sans Mono CJK SC"
    )
    OUTPUT_SVG.write_text(svg, encoding="utf-8")


def render_png() -> None:
    subprocess.run(
        [
            str(RESVG), "--zoom", "2", "--background", "white",
            "--sans-serif-family", "Noto Sans CJK SC",
            "--monospace-family", "Noto Sans Mono CJK SC", "--languages", "zh-CN",
            str(OUTPUT_SVG), str(OUTPUT_PNG),
        ],
        check=True,
    )


def main() -> None:
    diagram = Diagram(title="URGENT2026：训练 → 推理 → 评估")

    frame(
        diagram, x=24, y=46, width=1680, height=202,
        title="主路径：训练产物经统一入口推理，再按显式开关评估",
        tint="#F8FAFC", stroke="#64748B",
    )
    diagram.place(IOLabel("训练", id="train", modality="state"), x=60, y=133)
    place(
        diagram, "URGENT2026 config", ident="config", x=168, y=98, kind="io",
        sub="train.py + 训练配置\nval_metrics 与周期 checkpoint", min_w=198, h=96,
    )
    place(
        diagram, "checkpoint", ident="checkpoint", x=436, y=98, kind="io",
        sub="推荐 best_model.tar\n或 latest_model.tar / model_XXXX.pth", min_w=220, h=96,
    )
    place(
        diagram, "统一推理", ident="infer", x=752, y=98, kind="model",
        sub="inference.py\n目录或 SCP 输入", min_w=170, h=96,
    )
    place(
        diagram, "推理 artifact", ident="artifacts", x=967, y=98, kind="io",
        sub="增强 wav + inf.scp\n只记录成功或复用 UID", min_w=202, h=96,
    )
    place(
        diagram, "显式请求评估", ident="evaluate", x=1238, y=98, kind="model",
        sub="evaluate/evaluate_urgent.py\n按 CLI 开关请求指标", min_w=210, h=96,
    )
    diagram.place(IOLabel("结果", id="result", modality="state"), x=1530, y=133)
    diagram.chain(
        ["train", "config", "checkpoint", "infer", "artifacts", "evaluate", "result"],
        labels=["", "", "", "", "", ""],
    )

    frame(
        diagram, x=24, y=282, width=1680, height=220,
        title="评估门：请求某指标不等于该指标必然产出",
        tint="#FFF7ED", stroke="#C2410C",
    )
    place(
        diagram, "侵入式指标门", ident="reference", x=69, y=350, kind="known",
        sub="PESQ / STOI / ESTOI / SI-SDR\n需要 --ref_scp 与配对参考", min_w=244, h=100,
    )
    place(
        diagram, "依赖 / 权重 / 资产门", ident="optional", x=425, y=350, kind="known",
        sub="DNSMOS / NISQA / UTMOS / SCOREQ 等\n缺失时对应项失败或缺失", min_w=270, h=100,
    )
    place(
        diagram, "WER / CER 额外前提", ident="wer", x=818, y=350, kind="known",
        sub="ESPnet / OWSM、--meta_tsv、--utt2lang\n官方 checkout 的 owsm_utils.py", min_w=292, h=100,
    )
    place(
        diagram, "实际结果汇总", ident="summary", x=1276, y=350, kind="io",
        sub="RESULTS.txt\n成功项 + 缺项/失败计数\nNormAvg: k/6 或 N/A", min_w=244, h=100,
    )
    diagram.edge("reference.t@0.22", "evaluate.b@0.24", label="满足配对条件", style_name="cond")
    diagram.edge("optional.t@0.50", "evaluate.b@0.50", label="资产可用", style_name="cond")
    diagram.edge("wer.t@0.78", "evaluate.b@0.76", label="全部前提可用", style_name="cond")
    diagram.edge("evaluate", "summary", label="逐项执行")
    diagram.note(
        54, 478,
        "事实边界：仅 6/6 是官方 NormAvg 口径；k/6 为本仓库对实际可用项的部分平均。",
        size=10.2, color="#9A3412", anchor="start",
    )

    checks = {
        "check": diagram.check(),
        "check_arrows": diagram.check_arrows(),
        "check_edges_through_boxes": diagram.check_edges_through_boxes(),
    }
    if any(checks.values()):
        raise RuntimeError(f"ArchScope geometry checks failed: {checks}")
    diagram.save(OUTPUT_SVG, png=False, on_overlap="raise")
    normalize_svg_fonts()
    render_png()
    print(f"wrote {OUTPUT_SVG} and {OUTPUT_PNG}")
    print(f"ArchScope source: {ARCHSCOPE_SOURCE}")
    print("ArchScope checks: all empty")


if __name__ == "__main__":
    main()
