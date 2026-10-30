"""Ensure AEC collate re-exports the SE collate_fn instead of duplicating it."""

from __future__ import annotations

import pytest

from tasks.aec.data.collate import collate_fn as aec_collate_fn
from core.data.collate import collate_fn as se_collate_fn


def test_aec_collate_fn_is_se_collate_fn() -> None:
    """Regression guard against future copy-paste drift."""
    assert aec_collate_fn is se_collate_fn

pytestmark = [pytest.mark.unit]
