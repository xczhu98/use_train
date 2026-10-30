"""Task-neutral rendered-track planning for dynamic speech scenes.

``SceneSpec`` describes Layer B presence. ``ScenePlan`` describes the rendered
track roles consumed by Layer C/D/F and audit code. It deliberately does not
know task-specific names such as nearend, mic, or enrollment.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.data.simulation.scene_spec import SceneSpec


@dataclass(frozen=True, slots=True)
class ScenePlan:
    """Immutable role plan behind a small rendered-scene Interface."""

    scene: SceneSpec

    @property
    def observation_role(self) -> str:
        return "mixture"

    @property
    def background_role(self) -> str:
        return "background"

    @staticmethod
    def source_full_rir_role(index: int) -> str:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"source index must be a non-negative integer, got {index!r}")
        return f"source_full_rir/{index}"

    @staticmethod
    def target_role(index: int) -> str:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"target index must be a non-negative integer, got {index!r}")
        return f"target/{index}"

    @staticmethod
    def absent_target_role() -> str:
        return "target/absent"

    def source_roles(self) -> tuple[str, ...]:
        return tuple(self.source_full_rir_role(i) for i in range(self.scene.source_count))

    def target_roles(self) -> tuple[str, ...]:
        return tuple(self.target_role(i) for i in range(self.scene.source_count))

    def linear_roles(self) -> tuple[str, ...]:
        return (*self.source_roles(), self.background_role, self.observation_role)

    def to_dict(self) -> dict[str, object]:
        return {
            "scene": self.scene.to_dict(),
            "observation_role": self.observation_role,
            "background_role": self.background_role,
            "source_roles": list(self.source_roles()),
            "target_roles": list(self.target_roles()),
            "linear_roles": list(self.linear_roles()),
        }


__all__ = ["ScenePlan"]
