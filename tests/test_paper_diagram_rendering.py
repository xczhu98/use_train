"""论文方法图的 clone-safe 语义与渲染发布合同。"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess

import pytest

from tools.visualization import paper_diagram_rendering as rendering


pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[1]
CHAT_MD = ROOT / "docs/paper/fullduplex/Xiaohongshu/01_FireRedChat.md"
ASR_MD = ROOT / "docs/paper/asr/Xiaohongshu/01_FireRedASR.md"
UNISE_MD = ROOT / "docs/paper/se/Alibaba/28_UniSE.md"
CHAT_SCRIPT = (
    ROOT
    / "docs/paper/fullduplex/Xiaohongshu/figures/generated/fireredchat/method-overview.py"
)
ASR_SCRIPT = (
    ROOT
    / "docs/paper/asr/Xiaohongshu/figures/generated/fireredasr/method-overview.py"
)
UNISE_SCRIPT = (
    ROOT / "docs/paper/se/Alibaba/figures/generated/unise/method-overview.py"
)


class _Diagram:
    def __init__(self, diagnostics: dict[str, object] | None = None) -> None:
        self.diagnostics = diagnostics or {}

    def check(self):
        return self.diagnostics.get("check", [])

    def check_arrows(self):
        return self.diagnostics.get("check_arrows", [])

    def check_edges_through_boxes(self):
        return self.diagnostics.get("check_edges_through_boxes", [])

    def save(self, path, **kwargs) -> None:
        assert kwargs == {"png": False, "on_overlap": "raise"}
        Path(path).write_text(
            '<svg><text font-family="wrong sans">方法图</text></svg>',
            encoding="utf-8",
        )


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{path} 缺少可静态审计的 {name}")


def test_resolve_resvg_priority_is_explicit_then_environment_then_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    explicit = _make_executable(tmp_path / "explicit-resvg")
    environment = _make_executable(tmp_path / "environment-resvg")
    from_path = _make_executable(tmp_path / "path-resvg")
    monkeypatch.setenv("RESVG_BIN", str(environment))
    monkeypatch.setattr(rendering.shutil, "which", lambda name: str(from_path))

    assert rendering.resolve_resvg(explicit) == explicit.resolve()
    assert rendering.resolve_resvg() == environment.resolve()

    monkeypatch.delenv("RESVG_BIN")
    assert rendering.resolve_resvg() == from_path.resolve()


def test_resolve_resvg_reports_all_supported_discovery_routes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("RESVG_BIN", raising=False)
    monkeypatch.setattr(rendering.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="--resvg.*RESVG_BIN.*PATH"):
        rendering.resolve_resvg()


def test_load_archscope_reports_optional_dependency_without_private_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    def missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(rendering.importlib, "import_module", missing)

    with pytest.raises(RuntimeError, match="ArchScope.*Mermaid") as exc_info:
        rendering.load_archscope()

    assert ".kimi-code" not in str(exc_info.value)
    assert "conda" not in str(exc_info.value).lower()


def test_archscope_checks_fail_before_any_publication(tmp_path: Path):
    svg = tmp_path / "method.svg"
    png = tmp_path / "method.png"
    diagram = _Diagram({"check_arrows": ["missing arrow"]})

    with pytest.raises(RuntimeError, match="check_arrows"):
        rendering.render_diagram(
            diagram,
            svg_path=svg,
            png_path=png,
            resvg_bin=tmp_path / "unused-resvg",
        )

    assert not svg.exists()
    assert not png.exists()


def test_render_requires_same_directory_and_stem_for_the_pair(tmp_path: Path):
    with pytest.raises(ValueError, match="同目录、同 stem"):
        rendering.render_diagram(
            _Diagram(),
            svg_path=tmp_path / "one.svg",
            png_path=tmp_path / "other.png",
            resvg_bin=tmp_path / "unused-resvg",
        )


def test_render_stages_both_artifacts_before_per_file_atomic_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    resvg = _make_executable(tmp_path / "resvg")
    svg = tmp_path / "method.svg"
    png = tmp_path / "method.png"

    def fake_run(args, **kwargs):
        assert kwargs == {"check": True}
        assert Path(args[-2]).is_file()
        assert not svg.exists()
        assert not png.exists()
        Path(args[-1]).write_bytes(b"stable-png")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(rendering.subprocess, "run", fake_run)
    receipt = rendering.render_diagram(
        _Diagram(), svg_path=svg, png_path=png, resvg_bin=resvg, zoom=2.0
    )

    assert 'font-family="Noto Sans CJK SC, Noto Sans CJK, sans-serif"' in svg.read_text(
        encoding="utf-8"
    )
    assert png.read_bytes() == b"stable-png"
    assert receipt.svg_path == svg.resolve()
    assert receipt.png_path == png.resolve()
    assert receipt.pair_atomic is False


def test_second_publish_failure_rolls_back_both_previous_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    resvg = _make_executable(tmp_path / "resvg")
    svg = tmp_path / "method.svg"
    png = tmp_path / "method.png"
    svg.write_text("previous-svg", encoding="utf-8")
    png.write_bytes(b"previous-png")

    def fake_run(args, **kwargs):
        Path(args[-1]).write_bytes(b"new-png")
        return subprocess.CompletedProcess(args, 0)

    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(rendering.subprocess, "run", fake_run)
    monkeypatch.setattr(rendering.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="injected second publish failure"):
        rendering.render_diagram(
            _Diagram(), svg_path=svg, png_path=png, resvg_bin=resvg
        )

    assert svg.read_text(encoding="utf-8") == "previous-svg"
    assert png.read_bytes() == b"previous-png"


@pytest.mark.parametrize("script", [CHAT_SCRIPT, ASR_SCRIPT, UNISE_SCRIPT])
def test_method_scripts_use_the_portable_render_helper(script: Path):
    source = script.read_text(encoding="utf-8")
    assert "from tools.visualization.paper_diagram_rendering import" in source
    assert "render_diagram(" in source
    assert ".kimi-code" not in source
    assert "miniconda" not in source
    assert "cairosvg" not in source.lower()


def test_fireredchat_topology_isolates_voiceprint_from_raw_audio_consumers():
    nodes = set(_literal_assignment(CHAT_SCRIPT, "TOPOLOGY_NODES"))
    edges = set(_literal_assignment(CHAT_SCRIPT, "TOPOLOGY_EDGES"))
    assert len(nodes) <= 12
    assert {"raw_audio", "target_voiceprint", "pvad", "asr", "audiollm"} <= nodes
    assert {edge for edge in edges if edge[0] == "target_voiceprint"} == {
        ("target_voiceprint", "pvad", "condition")
    }
    assert {edge for edge in edges if edge[0] == "raw_audio"} == {
        ("raw_audio", "pvad", "signal"),
        ("raw_audio", "asr", "signal"),
        ("raw_audio", "audiollm", "signal"),
    }
    assert ("pvad", "asr", "control") in edges
    assert ("pvad", "audiollm", "signal") not in edges


def test_fireredchat_renderer_implements_every_declared_topology_edge_once():
    edges = tuple(_literal_assignment(CHAT_SCRIPT, "TOPOLOGY_EDGES"))
    rendering_edges = _literal_assignment(CHAT_SCRIPT, "EDGE_RENDERING")

    assert len(edges) == len(set(edges))
    assert set(rendering_edges) == set(edges)


def test_fireredasr_topology_separates_encoders_and_marks_initialization_only():
    nodes = set(_literal_assignment(ASR_SCRIPT, "TOPOLOGY_NODES"))
    edges = set(_literal_assignment(ASR_SCRIPT, "TOPOLOGY_EDGES"))
    assert len(nodes) <= 12
    assert {"aed_encoder", "llm_encoder"} <= nodes
    assert ("aed_encoder", "llm_encoder", "initialization") in edges
    assert ("aed_encoder", "llm_encoder", "signal") not in edges


def test_unise_topology_separates_conditions_codec_ss_and_training_stages():
    nodes = set(_literal_assignment(UNISE_SCRIPT, "TOPOLOGY_NODES"))
    edges = set(_literal_assignment(UNISE_SCRIPT, "TOPOLOGY_EDGES"))
    assert len(nodes) <= 12
    assert {
        "degraded",
        "reference",
        "bicodec_target",
        "bicodec_decoder",
        "ss_composition",
        "pretrain",
        "s1",
        "s2",
    } <= nodes
    assert ("bicodec_target", "lm", "training") in edges
    assert ("lm", "bicodec_decoder", "signal") in edges
    assert ("pretrain", "s1", "training") in edges
    assert ("s1", "s2", "training") in edges


@pytest.mark.parametrize("markdown", [CHAT_MD, ASR_MD, UNISE_MD])
def test_tracked_markdown_is_clone_safe_and_mermaid_first(markdown: Path):
    text = markdown.read_text(encoding="utf-8")
    local_binary_link = re.compile(
        r"!?\[[^\]]*\]\((?!https?://)[^)\s]+\.(?:png|svg|pdf)(?:#[^)]*)?\)",
        re.IGNORECASE,
    )
    assert len(re.findall(r"```mermaid\n", text)) == 1
    assert not local_binary_link.search(text)
    assert "method-overview.png" not in text
    assert "本地渲染资产仅是 ignored optional evidence cache" in text


def test_markdown_semantics_state_the_three_corrected_method_boundaries():
    chat = CHAT_MD.read_text(encoding="utf-8")
    asr = ASR_MD.read_text(encoding="utf-8")
    unise = UNISE_MD.read_text(encoding="utf-8")

    assert "原始语音" in chat and "pVAD 只输出 timestamps / barge-in 控制" in chat
    assert "RAW -->|原始语音| PVAD" in chat
    assert "RAW -->|原始语音| ASR" in chat
    assert "RAW -->|原始语音| ALLM" in chat
    assert "VOICE -->|speaker condition| PVAD" in chat
    assert "VOICE --> ASR" not in chat
    assert "VOICE --> ALLM" not in chat
    assert "PVAD -. timestamps .-> ASR" in chat
    assert "ArchScope 与 resvg" in chat and "optional cache renderer" in chat
    assert "未固定版本" in chat and "不承诺跨环境 bit-for-bit 复现" in chat
    assert "AED encoder 与 LLM encoder 是两个独立运行时模块" in asr
    assert "虚线只表示权重初始化" in asr
    assert "IN --> AEDENC --> AEDDEC --> OUT" in asr
    assert "IN --> LLMENC --> ADAPT --> QWEN --> OUT" in asr
    assert "AEDENC -. 仅权重初始化 .-> LLMENC" in asr
    assert "条件输入、BiCodec target token 与 BiCodec decoder 三者分离" in unise
    assert "DEG --> COND --> LM --> DEC --> OUT" in unise
    assert "TARGET -. 训练 target .-> LM" in unise
    assert "SS -. 按次选择任务模式 .-> TASK" in unise
    assert "PRE --> S1 --> S2" in unise
    assert "Pretrain → S1 → S2" in unise
