"""任务无关的 waveform scene 合同与投影。

本模块只组合 :class:`WaveformBundle`，不重复实现 waveform 的快照、dtype、
finite 值和 timeline 校验。这里的 role 是公共仿真职责，不替代任务物理名：
SE ``noisy``、AEC ``mic``、TSE/SS ``mixture`` 都投影为 observation；AEC
``nearend``、TSE ``target``、SS K 路 source 则投影为 supervision。任务侧通过
``WaveformSceneContract`` 声明允许和必须出现的 rendered track，再由
``WaveformSceneBundle`` 做统一合同校验。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import re

import numpy as np

from core.data.waveform_bundle import WaveformBundle


_INDEXED_ROLE = re.compile(r"^(source_full_rir|target)/(\d+)$")
_SOURCE_FULL_RIR_PREFIX = "source_full_rir/"
_TARGET_PREFIX = "target/"
_TARGET_ABSENT = "target/absent"


def _normalize_role(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-empty role names")
    return value


def _normalize_role_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of role names")
    try:
        normalized = tuple(_normalize_role(value, field_name) for value in values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of role names") from exc
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate roles")
    return normalized


def _normalize_count(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be a non-negative integer or None")
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _indexed_roles(roles: Iterable[str], family: str) -> tuple[str, ...]:
    indexed: list[tuple[int, str]] = []
    for role in roles:
        match = _INDEXED_ROLE.fullmatch(role)
        if match is not None and match.group(1) == family:
            indexed.append((int(match.group(2)), role))
    return tuple(role for _, role in sorted(indexed))


def _validate_indexed_family(roles: tuple[str, ...], family: str) -> tuple[str, ...]:
    indexed: list[tuple[int, str]] = []
    prefix = f"{family}/"
    for role in roles:
        if not role.startswith(prefix):
            continue
        if family == "target" and role == _TARGET_ABSENT:
            continue
        match = _INDEXED_ROLE.fullmatch(role)
        if match is None or match.group(1) != family:
            raise ValueError(
                f"{family} role {role!r} must use the form {family}/N"
            )
        indexed.append((int(match.group(2)), role))

    indexed.sort()
    expected = tuple(range(len(indexed)))
    actual = tuple(index for index, _ in indexed)
    if actual != expected:
        raise ValueError(
            f"{family} roles must have contiguous zero-based indices; got {actual!r}"
        )
    return tuple(role for _, role in indexed)


def _role_matches_declaration(role: str, declaration: str) -> bool:
    if declaration.endswith("/*"):
        return role.startswith(declaration[:-1])
    return role == declaration


@dataclass(frozen=True)
class WaveformSceneContract:
    """任务对 scene 角色和 source/target 数量的声明。

    ``declared_roles`` 支持精确角色名以及 ``source_full_rir/*``、``target/*``
    形式的角色族声明。未提供 ``declared_roles`` 时不限制额外角色，但保留
    ``required_roles``、计数和 observation/decomposition 的合同校验。
    """

    required_roles: tuple[str, ...] = ()
    source_count: int | None = None
    target_count: int | None = None
    observation_role: str | None = None
    decomposition_roles: tuple[str, ...] = ()
    declared_roles: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        required_roles = _normalize_role_tuple(self.required_roles, "required_roles")
        decomposition_roles = _normalize_role_tuple(
            self.decomposition_roles, "decomposition_roles"
        )
        declared_roles = (
            None
            if self.declared_roles is None
            else _normalize_role_tuple(self.declared_roles, "declared_roles")
        )
        observation_role = self.observation_role
        if observation_role is not None:
            observation_role = _normalize_role(observation_role, "observation_role")
        object.__setattr__(self, "required_roles", required_roles)
        object.__setattr__(self, "source_count", _normalize_count(self.source_count, "source_count"))
        object.__setattr__(self, "target_count", _normalize_count(self.target_count, "target_count"))
        object.__setattr__(self, "observation_role", observation_role)
        object.__setattr__(self, "decomposition_roles", decomposition_roles)
        object.__setattr__(self, "declared_roles", declared_roles)


@dataclass(frozen=True)
class WaveformSceneBundle:
    """带有任务角色合同的不可变 waveform scene。

    底层 waveform 始终由 ``WaveformBundle`` 保存和冻结。构造时会重新建立
    一个 ``WaveformBundle`` 快照，因此调用方不能通过传入对象改变 scene。
    所有角色最后一维必须共享同一 timeline；角色本身不携带任务物理语义。
    ``source_full_rir/N`` 是实际进入线性 observation 的 full-RIR source，
    ``target/N`` 是 early/dry supervision，不能把二者视为同一波形。
    """

    bundle: WaveformBundle
    contract: WaveformSceneContract = field(default_factory=WaveformSceneContract)

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, WaveformBundle):
            raise TypeError("bundle must be a WaveformBundle")
        if not isinstance(self.contract, WaveformSceneContract):
            raise TypeError("contract must be a WaveformSceneContract")
        # 重新组合底层 bundle；已由 WaveformBundle 标记的 bytes-backed 快照会
        # 直接复用，外部可写数组仍在底层构造处建立独立的不可变快照。
        snapshot = WaveformBundle(self.bundle.tracks, self.bundle.metadata)
        object.__setattr__(self, "bundle", snapshot)
        self.validate()

    @classmethod
    def from_tracks(
        cls,
        tracks: Mapping[str, np.ndarray],
        *,
        metadata: Mapping[str, object] | None = None,
        contract: WaveformSceneContract | None = None,
    ) -> "WaveformSceneBundle":
        """从命名 waveform 创建 scene；waveform 合同仍由 WaveformBundle 执行。"""
        return cls(
            WaveformBundle(tracks, {} if metadata is None else metadata),
            WaveformSceneContract() if contract is None else contract,
        )

    @property
    def tracks(self) -> Mapping[str, np.ndarray]:
        """返回底层只读 track 映射。"""
        return self.bundle.tracks

    @property
    def metadata(self) -> Mapping[str, object]:
        return self.bundle.metadata

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self.bundle.tracks)

    @property
    def num_samples(self) -> int:
        return self.bundle.num_samples

    @property
    def required_roles(self) -> tuple[str, ...]:
        return self.contract.required_roles

    @property
    def observation_role(self) -> str | None:
        return self.contract.observation_role

    @property
    def decomposition_roles(self) -> tuple[str, ...]:
        return self.contract.decomposition_roles

    @property
    def source_roles(self) -> tuple[str, ...]:
        return _indexed_roles(self.roles, "source_full_rir")

    @property
    def target_roles(self) -> tuple[str, ...]:
        return _indexed_roles(self.roles, "target")

    @property
    def source_count(self) -> int:
        return (
            self.contract.source_count
            if self.contract.source_count is not None
            else len(self.source_roles)
        )

    @property
    def target_count(self) -> int:
        return (
            self.contract.target_count
            if self.contract.target_count is not None
            else len(self.target_roles)
        )

    def require(self, role: str) -> np.ndarray:
        """读取一个已通过底层 WaveformBundle 校验的角色。"""
        return self.bundle.require(role)

    def has(self, role: str) -> bool:
        if not isinstance(role, str):
            raise TypeError("role must be a string")
        return role in self.bundle.tracks

    def validate(self) -> None:
        """验证任务角色、source/target 索引和 observation 合同。"""
        roles = self.roles
        role_set = set(roles)

        for role in self.contract.required_roles:
            if role not in role_set:
                raise KeyError(f"required scene role {role!r} is missing")

        if self.contract.observation_role is not None:
            if self.contract.observation_role not in role_set:
                raise KeyError(
                    f"observation role {self.contract.observation_role!r} is missing"
                )

        for role in self.contract.decomposition_roles:
            if role not in role_set:
                raise KeyError(f"decomposition role {role!r} is missing")

        if self.contract.declared_roles is not None:
            undeclared = tuple(
                role
                for role in roles
                if not any(
                    _role_matches_declaration(role, declaration)
                    for declaration in self.contract.declared_roles
                )
            )
            if undeclared:
                raise ValueError(f"scene contains undeclared roles: {undeclared!r}")

        source_roles = _validate_indexed_family(roles, "source_full_rir")
        target_roles = _validate_indexed_family(roles, "target")

        if (
            self.contract.source_count is not None
            and self.contract.source_count != len(source_roles)
        ):
            raise ValueError(
                "source_count does not match source_full_rir/N roles: "
                f"declared {self.contract.source_count}, found {len(source_roles)}"
            )
        if (
            self.contract.target_count is not None
            and self.contract.target_count != len(target_roles)
        ):
            raise ValueError(
                "target_count does not match target/N roles: "
                f"declared {self.contract.target_count}, found {len(target_roles)}"
            )

    def to_task_view(self, roles: Iterable[str] | None = None) -> WaveformBundle:
        """投影为任务所需的 ``WaveformBundle`` 快照。

        ``roles`` 为 ``None`` 时返回包含全部角色的新快照；传入角色序列时
        只保留这些角色，并在投影前检查每个角色存在。该方法不改变 scene
        合同，也不改变原 bundle。
        """
        selected = self.roles if roles is None else _normalize_role_tuple(roles, "roles")
        missing = tuple(role for role in selected if role not in self.bundle.tracks)
        if missing:
            raise KeyError(f"task view roles are missing: {missing!r}")
        return WaveformBundle(
            {role: self.bundle.require(role) for role in selected},
            self.bundle.metadata,
        )

    def project(self, roles: Iterable[str] | None = None) -> WaveformBundle:
        """``to_task_view`` 的简洁别名。"""
        return self.to_task_view(roles)


__all__ = ["WaveformSceneBundle", "WaveformSceneContract"]
