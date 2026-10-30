"""SS optional source identity contract tests."""

import pytest
import torch

from tasks.ss.data.collate import ss_collate_fn


pytestmark = pytest.mark.unit


def _item(*, source_ids="omitted", length=3):
    item = {
        "mixture": torch.tensor([1.0, 2.0, 3.0, 4.0])[:length],
        "sources": [
            torch.tensor([0.1, 0.2, 0.3, 0.4])[:length],
            torch.tensor([0.4, 0.3, 0.2, 0.1])[:length],
        ],
        "sr": 16000,
        "length": length,
    }
    if source_ids != "omitted":
        item["source_ids"] = source_ids
    return item


def test_all_samples_without_source_ids_are_valid_and_padded() -> None:
    result = ss_collate_fn([_item(length=3), _item(length=4)])

    assert result.source_ids is None
    assert result.mixture.shape == (2, 4)
    assert result.sources.shape == (2, 2, 4)
    assert result.lengths.tolist() == [3, 4]
    assert torch.equal(result.mixture[0], torch.tensor([1.0, 2.0, 3.0, 0.0]))


def test_partial_source_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="present for every item"):
        ss_collate_fn([_item(), _item(source_ids=("a", "b"))])


@pytest.mark.parametrize(
    "source_ids",
    [(), ("a",), ("a", "b", "c"), ("a", ""), ("a", 1)],
)
def test_explicit_source_ids_must_match_nonempty_source_identity_contract(source_ids) -> None:
    with pytest.raises(ValueError, match="source_ids|source_ids values"):
        ss_collate_fn([_item(source_ids=source_ids), _item(source_ids=source_ids)])


@pytest.mark.parametrize("field", ["sr", "length"])
def test_ss_collate_rejects_non_exact_positive_metadata(field) -> None:
    item = _item()
    item[field] = 16000.0 if field == "sr" else 3.0
    with pytest.raises(ValueError, match=field):
        ss_collate_fn([item])
