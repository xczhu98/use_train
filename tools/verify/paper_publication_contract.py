#!/usr/bin/env python3
"""审计论文 Markdown 与本地证据缓存的发布边界。"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
PDF_SUFFIX = ".pdf"
# 精读 md 渲染出的阅读版 PDF 镜像允许 tracked（见 .gitignore 例外）；其余 PDF 仍是本地证据缓存
READING_PDF_SUFFIX = "_reading.pdf"
_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*(?P<target><[^>]+>|[^\s)]+)")
_BARE_URL_RE = re.compile(r"(?P<target>(?:https?|file):[^\s<>()]+)", re.IGNORECASE)
_FENCE_RE = re.compile(r"(?ms)^[ \t]{0,3}(`{3,}|~{3,}).*?^\1[ \t]*$")
_CODE_SPAN_RE = re.compile(r"(`+)(?:(?!\1).)*\1", re.DOTALL)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    rule: str
    detail: str

    def render(self, repo_root: Path) -> str:
        try:
            display = self.path.relative_to(repo_root)
        except ValueError:
            display = self.path
        return f"{display}:{self.line}: {self.rule}: {self.detail}"


class GitUnavailableError(RuntimeError):
    """Git 缺失、仓库无效或查询失败。"""


def canonical_markdown_files(paper_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in paper_root.rglob("*.md"):
        relative = path.relative_to(paper_root)
        if path.name == "README.md" or path.name.startswith("00_"):
            continue
        if "analysis" in relative.parts or "illustrated" in relative.parts:
            continue
        files.append(path)
    return sorted(files)


def _mask_code(markdown: str) -> str:
    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    return _CODE_SPAN_RE.sub(blank, _FENCE_RE.sub(blank, markdown))


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _target_suffix(target: str) -> str:
    path = unquote(urlsplit(target).path)
    return PurePosixPath(path.replace("\\", "/")).suffix.lower()


def tracked_repository_files(repo_root: Path) -> frozenset[Path]:
    repo_root = repo_root.resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise GitUnavailableError(f"cannot execute git: {exc}") from exc

    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitUnavailableError(message or "git ls-files failed")

    tracked = result.stdout.decode("utf-8", errors="strict").split("\0")
    return frozenset(repo_root / item for item in tracked if item)


def scan_markdown(
    path: Path,
    repo_root: Path,
) -> list[Violation]:
    markdown = path.read_text(encoding="utf-8")
    masked = _mask_code(markdown)
    violations: list[Violation] = []
    resolved_root = repo_root.resolve()

    link_matches = list(_LINK_RE.finditer(masked))
    for match in link_matches:
        target = match.group("target").strip("<>")
        line = _line_number(masked, match.start())
        if target.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE_RE.match(target):
            violations.append(Violation(path, line, "absolute-path", target))
            continue

        parsed = urlsplit(target)
        scheme = parsed.scheme.lower()
        if scheme:
            if scheme == "https":
                continue
            rule = "unsafe-scheme" if scheme != "file" else "file-url"
            violations.append(Violation(path, line, rule, target))
            continue

        local_path = unquote(parsed.path).replace("\\", "/")
        if not local_path or local_path.startswith("#"):
            continue
        if ".." in PurePosixPath(local_path).parts:
            violations.append(Violation(path, line, "path-escape", target))
            continue

        lexical_target = Path(os.path.normpath(path.parent / local_path))
        if not _is_within(lexical_target, resolved_root):
            violations.append(Violation(path, line, "path-escape", target))
            continue
        resolved_target = lexical_target.resolve()
        if not _is_within(resolved_target, resolved_root):
            violations.append(Violation(path, line, "path-escape", target))
            continue

        suffix = _target_suffix(target)
        if suffix == PDF_SUFFIX:
            if not resolved_target.is_file():
                violations.append(Violation(path, line, "local-pdf-missing", target))
            continue
        if suffix not in IMAGE_SUFFIXES:
            continue
        if not resolved_target.is_file():
            violations.append(Violation(path, line, "local-image-missing", target))

    prose = list(masked)
    for match in link_matches:
        prose[match.start() : match.end()] = " " * (match.end() - match.start())
    prose_text = "".join(prose)

    bare_url_matches = list(_BARE_URL_RE.finditer(prose_text))
    for match in bare_url_matches:
        target = match.group("target")
        scheme = urlsplit(target).scheme.lower()
        if scheme == "https":
            continue
        rule = "file-url" if scheme == "file" else "unsafe-scheme"
        violations.append(
            Violation(path, _line_number(prose_text, match.start()), rule, target)
        )

    return violations


def tracked_paper_pdfs(repo_root: Path, paper_root: Path) -> list[Path]:
    resolved_root = repo_root.resolve()
    resolved_paper_root = paper_root.resolve()
    if not _is_within(resolved_paper_root, resolved_root):
        raise GitUnavailableError("paper root must be inside repository root")
    return sorted(
        path
        for path in tracked_repository_files(resolved_root)
        if _is_within(path, resolved_paper_root)
        and path.suffix.lower() == PDF_SUFFIX
        and not path.name.endswith(READING_PDF_SUFFIX)
    )


def _selected_markdown_files(
    repo_root: Path, markdown_paths: list[Path] | None
) -> list[Path]:
    if not markdown_paths:
        return canonical_markdown_files(repo_root / "docs" / "paper")

    selected: list[Path] = []
    for candidate in markdown_paths:
        path = candidate if candidate.is_absolute() else repo_root / candidate
        path = path.resolve()
        if not _is_within(path, repo_root):
            raise ValueError(f"Markdown path escapes repository root: {candidate}")
        if path.suffix.lower() != ".md":
            raise ValueError(f"publication input must be Markdown: {candidate}")
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append(path)
    return selected


def audit_repository(
    repo_root: Path, markdown_paths: list[Path] | None = None
) -> tuple[list[Path], list[Violation]]:
    repo_root = repo_root.resolve()
    markdown_files = _selected_markdown_files(repo_root, markdown_paths)
    violations = [
        violation
        for markdown in markdown_files
        for violation in scan_markdown(markdown, repo_root)
    ]
    violations.extend(
        Violation(path, 1, "tracked-paper-pdf", "PDF files are local-only evidence")
        for path in tracked_paper_pdfs(repo_root, repo_root / "docs" / "paper")
    )
    return markdown_files, sorted(
        violations, key=lambda item: (str(item.path), item.line, item.rule)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit clone-safe publication rules for docs/paper Markdown."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "markdown",
        nargs="*",
        type=Path,
        help="Markdown paths to audit (default: all canonical docs/paper Markdown)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        markdown_files, violations = audit_repository(args.repo_root, args.markdown)
    except (GitUnavailableError, OSError, UnicodeError, ValueError) as exc:
        print(f"paper publication contract: ERROR: {exc}", file=sys.stderr)
        return 2

    if violations:
        print(
            f"paper publication contract: FAIL ({len(violations)} violations in "
            f"{len(markdown_files)} Markdown files)",
            file=sys.stderr,
        )
        for violation in violations:
            print(violation.render(args.repo_root.resolve()), file=sys.stderr)
        return 1

    print(
        f"paper publication contract: OK ({len(markdown_files)} Markdown files, "
        "links are clone-safe and every linked local PDF/image exists)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
