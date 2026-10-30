"""论文发布合同的 unit 与 CLI contract 测试。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.verify import paper_publication_contract as contract

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)


def _add(root: Path, *paths: str) -> None:
    subprocess.run(["git", "-C", str(root), "add", "--", *paths], check=True)


def test_canonical_discovery_excludes_derived_markdown(tmp_path: Path) -> None:
    paper = tmp_path / "docs" / "paper" / "se"
    expected = paper / "Lab" / "01_Paper.md"
    excluded = [
        paper / "README.md",
        paper / "analysis" / "report.md",
        paper / "illustrated" / "story.md",
        paper / "Lab" / "00_核心观点.md",
    ]
    for path in [expected, *excluded]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test\n", encoding="utf-8")

    assert contract.canonical_markdown_files(tmp_path / "docs" / "paper") == [expected]


def test_markdown_scan_allows_https_and_code_span_provenance(tmp_path: Path) -> None:
    markdown = tmp_path / "docs" / "paper" / "se" / "Lab" / "01_Paper.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text(
        "[paper](https://arxiv.org/pdf/1234.5678)\n"
        "https://example.com/public-asset.png\n"
        "`docs/paper/se/Lab/01_Paper.pdf`\n"
        "`![cached](figures/table.png)`\n",
        encoding="utf-8",
    )

    assert contract.scan_markdown(markdown, tmp_path) == []


@pytest.mark.parametrize(
    ("link", "rule"),
    [
        ("[PDF](01_Paper.pdf)", "local-pdf-missing"),
        ("![missing](figures/table.png)", "local-image-missing"),
        ("[root](/etc/passwd)", "absolute-path"),
        ("[drive](C:/secret.txt)", "absolute-path"),
        ("[file](file:///tmp/paper.pdf)", "file-url"),
        ("[escape](../../../../../secret.txt)", "path-escape"),
        ("http://example.com/paper.pdf", "unsafe-scheme"),
    ],
)
def test_markdown_scan_rejects_non_clone_safe_targets(
    tmp_path: Path, link: str, rule: str
) -> None:
    markdown = tmp_path / "docs" / "paper" / "se" / "Lab" / "01_Paper.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text(f"{link}\n", encoding="utf-8")

    violations = contract.scan_markdown(markdown, tmp_path)

    assert [violation.rule for violation in violations] == [rule]


def test_existing_local_image_is_clone_safe(tmp_path: Path) -> None:
    paper = tmp_path / "docs" / "paper" / "se" / "Lab"
    paper.mkdir(parents=True)
    markdown = paper / "01_Paper.md"
    image = paper / "figure.png"
    markdown.write_text("![figure](figure.png)\n", encoding="utf-8")
    image.write_bytes(b"not-a-real-png")
    _init_repo(tmp_path)
    _add(tmp_path, "docs/paper/se/Lab/01_Paper.md", "docs/paper/se/Lab/figure.png")

    canonical, violations = contract.audit_repository(tmp_path)

    assert canonical == [markdown]
    assert violations == []
    assert contract.tracked_paper_pdfs(tmp_path, tmp_path / "docs/paper") == []


def test_parent_relative_image_is_rejected_even_when_inside_repo(
    tmp_path: Path,
) -> None:
    paper = tmp_path / "docs" / "paper" / "se" / "Lab"
    figures = tmp_path / "docs" / "paper" / "se" / "figures"
    paper.mkdir(parents=True)
    figures.mkdir(parents=True)
    markdown = paper / "01_Paper.md"
    image = figures / "figure.png"
    markdown.write_text("![figure](../figures/figure.png)\n", encoding="utf-8")
    image.write_bytes(b"not-a-real-png")
    _init_repo(tmp_path)
    _add(
        tmp_path,
        "docs/paper/se/Lab/01_Paper.md",
        "docs/paper/se/figures/figure.png",
    )

    _, violations = contract.audit_repository(tmp_path)

    assert [item.rule for item in violations] == ["path-escape"]


@pytest.mark.parametrize("ignored", [False, True])
def test_untracked_or_ignored_existing_image_is_clone_safe(
    tmp_path: Path, ignored: bool
) -> None:
    paper = tmp_path / "docs" / "paper" / "se" / "Lab"
    paper.mkdir(parents=True)
    markdown = paper / "01_Paper.md"
    image = paper / "figure.png"
    markdown.write_text("![figure](figure.png)\n", encoding="utf-8")
    image.write_bytes(b"not-a-real-png")
    if ignored:
        (tmp_path / ".gitignore").write_text("*.png\n", encoding="utf-8")
    _init_repo(tmp_path)
    _add(tmp_path, "docs/paper/se/Lab/01_Paper.md")

    _, violations = contract.audit_repository(tmp_path)

    assert violations == []


def test_untracked_local_pdf_link_is_clone_safe(tmp_path: Path) -> None:
    paper = tmp_path / "docs" / "paper" / "se" / "Lab"
    paper.mkdir(parents=True)
    markdown = paper / "01_Paper.md"
    pdf = paper / "01_Paper.pdf"
    markdown.write_text("[PDF](01_Paper.pdf)\n", encoding="utf-8")
    pdf.write_bytes(b"not-a-real-pdf")
    _init_repo(tmp_path)
    _add(tmp_path, "docs/paper/se/Lab/01_Paper.md")

    _, violations = contract.audit_repository(tmp_path)

    assert violations == []


def test_tracked_pdf_is_rejected(tmp_path: Path) -> None:
    paper = tmp_path / "docs" / "paper" / "se" / "Lab"
    paper.mkdir(parents=True)
    markdown = paper / "01_Paper.md"
    pdf = paper / "01_Paper.pdf"
    markdown.write_text("[PDF](01_Paper.pdf)\n", encoding="utf-8")
    pdf.write_bytes(b"not-a-real-pdf")
    _init_repo(tmp_path)
    _add(tmp_path, "docs/paper/se/Lab/01_Paper.md", "docs/paper/se/Lab/01_Paper.pdf")

    _, violations = contract.audit_repository(tmp_path)

    assert [item.rule for item in violations] == ["tracked-paper-pdf"]


def test_tracked_reading_pdf_mirror_is_allowed(tmp_path: Path) -> None:
    paper = tmp_path / "docs" / "paper" / "se" / "Lab"
    paper.mkdir(parents=True)
    markdown = paper / "01_Paper.md"
    mirror = paper / "01_Paper_reading.pdf"
    markdown.write_text("# test\n", encoding="utf-8")
    mirror.write_bytes(b"not-a-real-pdf")
    _init_repo(tmp_path)
    _add(
        tmp_path,
        "docs/paper/se/Lab/01_Paper.md",
        "docs/paper/se/Lab/01_Paper_reading.pdf",
    )

    _, violations = contract.audit_repository(tmp_path)

    assert violations == []
    assert contract.tracked_paper_pdfs(tmp_path, tmp_path / "docs/paper") == []


def test_git_query_failure_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paper = tmp_path / "docs" / "paper"
    paper.mkdir(parents=True)

    def _missing_git(*_args, **_kwargs):
        raise OSError("git executable missing")

    monkeypatch.setattr(contract.subprocess, "run", _missing_git)
    with pytest.raises(contract.GitUnavailableError, match="cannot execute git"):
        contract.tracked_paper_pdfs(tmp_path, paper)


def test_cli_accepts_clone_safe_repository(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    markdown = tmp_path / "docs" / "paper" / "se" / "Lab" / "01_Paper.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text(
        "[paper](https://arxiv.org/pdf/1234.5678)\n"
        "source_pdf: `docs/paper/se/Lab/01_Paper.pdf`\n",
        encoding="utf-8",
    )
    _init_repo(tmp_path)
    _add(tmp_path, "docs/paper/se/Lab/01_Paper.md")

    assert contract.main(
        ["--repo-root", str(tmp_path), "docs/paper/se/Lab/01_Paper.md"]
    ) == 0
    assert (
        "1 Markdown files, links are clone-safe and every linked local PDF/image exists"
        in capsys.readouterr().out
    )
