"""本机 paper skill 与论文资产的一致性检查。"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_SCRIPT = (
    REPO_ROOT / ".kimi-code/skills/ai-speech-paper/scripts/discover_new_papers.py"
)

pytestmark = pytest.mark.external


@pytest.fixture
def paper_discovery_module(monkeypatch):
    """按文件路径隔离加载本机 skill，并在测试后恢复模块缓存。"""
    if not SKILL_SCRIPT.is_file():
        pytest.skip(f"本机 paper skill 不存在: {SKILL_SCRIPT}")
    spec = importlib.util.spec_from_file_location("test_discover_new_papers", SKILL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_paper_source_aliases_are_contained_and_one_to_one(
    tmp_path, paper_discovery_module,
):
    """alias 可为空；存在时隔离于仓库且每个 PDF 唯一映射 canonical。"""
    module = paper_discovery_module
    aliases = module.load_source_aliases(REPO_ROOT)
    assert all(len(docs) == 1 for docs in aliases.values())
    for source, docs in aliases.items():
        assert (REPO_ROOT / source).is_file()
        assert (REPO_ROOT / docs[0]).is_file()

    alias_file = tmp_path / module.ALIAS_RELATIVE_PATH
    alias_file.parent.mkdir(parents=True)
    alias_file.write_text(
        json.dumps({"docs/paper/aec/Paper.pdf": ["../../outside.md"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="越界|允许范围"):
        module.load_source_aliases(tmp_path)

    source = tmp_path / "docs/paper/aec/Paper.pdf"
    direct_doc = tmp_path / "docs/paper/aec/01_Direct.md"
    alias_doc = tmp_path / "docs/paper/aec/02_Alias.md"
    source.parent.mkdir(parents=True)
    source.touch()
    direct_doc.write_text("[PDF](Paper.pdf)\n", encoding="utf-8")
    alias_doc.write_text("# Alias candidate\n", encoding="utf-8")
    alias_file.write_text(
        json.dumps(
            {
                "docs/paper/aec/Paper.pdf": [
                    "docs/paper/aec/02_Alias.md"
                ]
            }
        ),
        encoding="utf-8",
    )
    assert module.load_source_aliases(tmp_path) != aliases
    result = module.scan(tmp_path)
    assert result["counts"]["ambiguous"] == 1
    assert result["ambiguous"][0]["coverage_docs"] == (
        "docs/paper/aec/01_Direct.md",
        "docs/paper/aec/02_Alias.md",
    )


def test_paper_discovery_is_deterministic_and_maps_every_local_pdf(
    paper_discovery_module,
):
    """discovery 必须稳定枚举全部 PDF，并完整描述覆盖或待处理映射。"""
    module = paper_discovery_module
    result = module.scan(REPO_ROOT)
    assert result == module.scan(REPO_ROOT)
    assert result["canonical_tasks"] == sorted(
        {"aec", "asr", "fullduplex", "loss", "se", "survey", "sv", "tse"}
    )

    groups = ("covered", "new_candidates", "ambiguous", "unmapped_tasks")
    records = [record for group in groups for record in result[group]]
    discovered_sources = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "docs/paper").rglob("*.pdf")
    )
    record_sources = [record["source"] for record in records]
    assert sorted(record_sources) == discovered_sources
    assert len(record_sources) == len(set(record_sources))
    assert all(len(record["coverage_docs"]) == 1 for record in result["covered"])
    sources_by_doc = {}
    for record in result["covered"]:
        sources_by_doc.setdefault(record["coverage_docs"][0], set()).add(record["source"])
    assert all(
        module.sources_match_canonical(REPO_ROOT, doc, sources)
        for doc, sources in sources_by_doc.items()
    )
    assert all(
        record["target_dir"].startswith(f"docs/paper/{record['task']}")
        for record in result["covered"]
    )


def test_project_skills_remain_gitignored():
    """所有项目 skill 均为本机 agent 配置，必须保持整目录忽略。"""
    ignored = (
        ".kimi-code/skills/ai-speech-paper/SKILL.md",
        ".kimi-code/skills/ai-speech-paper/references/project-paper-format.md",
        ".kimi-code/skills/ai-speech-paper/references/source-aliases.json",
        ".kimi-code/skills/ai-speech-paper/scripts/discover_new_papers.py",
        ".kimi-code/skills/ai-speech-paper/scripts/validate_paper_reading.py",
        ".kimi-code/skills/ai-speech-paper/scripts/__pycache__/discover_new_papers.pyc",
        ".kimi-code/skills/experiment-report/SKILL.md",
    )
    for path in ignored:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", path],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, f"应保持忽略: {path}"
