"""tools 目录分域合同。"""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

def test_legacy_scripts_directory_is_removed() -> None:
    assert not (TOOLS / "scripts").exists()


def test_tools_root_has_no_unowned_business_modules() -> None:
    assert {
        path.name for path in TOOLS.glob("*.py")
    } == {"__init__.py"}


def test_ops_domains_have_explicit_packages() -> None:
    expected = {"aec", "data", "evaluation", "reporting"}
    assert {
        path.name
        for path in (TOOLS / "ops").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    } == expected
    for domain in expected:
        assert (TOOLS / "ops" / domain / "__init__.py").is_file()
