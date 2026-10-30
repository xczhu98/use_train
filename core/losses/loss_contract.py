"""Loss 输入能力的共享合同。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossContract:
    inputs: tuple[str, ...]
    permutation_invariant: bool = False
    supports_lengths: bool = False
    supports_source_mask: bool = False

    def __post_init__(self) -> None:
        if not self.inputs or len(set(self.inputs)) != len(self.inputs):
            raise ValueError("LossContract inputs must be non-empty and unique")
        if any(not isinstance(name, str) or not name for name in self.inputs):
            raise ValueError("LossContract inputs must be non-empty strings")
        for value in (
            self.permutation_invariant,
            self.supports_lengths,
            self.supports_source_mask,
        ):
            if type(value) is not bool:
                raise TypeError("LossContract capability flags must be bool")


def resolve_loss_contract(loss) -> LossContract | None:
    """Unwrap strict wrappers and return an explicit loss contract when present."""
    inner = getattr(loss, "loss", loss)
    contract = getattr(inner, "LOSS_CONTRACT", None)
    if contract is None:
        return None
    if not isinstance(contract, LossContract):
        raise TypeError("loss.LOSS_CONTRACT must be a LossContract")
    return contract
