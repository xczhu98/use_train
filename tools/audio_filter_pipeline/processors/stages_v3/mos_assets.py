"""Generation-3 Step3 的 selected-only 本地 MOS 资产身份。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from ..mos_contract import MOS_ASSET_PATH_FIELDS, MOS_ASSET_PROVENANCE_FIELDS
from ..mos import (
    normalize_selectable_mos_models,
    snapshot_local_checkpoint_identity,
    verify_urgent_pk_trusted_assets,
)
from ..optional_runtime import local_model_tree_sha256


def selected_mos_asset_provenance(
    selected_models: Sequence[str],
    options_by_model: Mapping[str, Mapping[str, Any] | None],
    *,
    tree_digest=local_model_tree_sha256,
    urgent_verifier=verify_urgent_pk_trusted_assets,
) -> dict[str, Any]:
    """验证且仅记录 selected model 的 canonical 本地资产，不加载 estimator。"""
    selected = (
        normalize_selectable_mos_models(selected_models)
        if selected_models else ()
    )
    provenance: dict[str, Any] = {}
    for model in selected:
        options = options_by_model.get(model)
        if options is None:
            raise ValueError(f"Step3 selected {model} without configured assets")
        path_fields = MOS_ASSET_PATH_FIELDS[model]
        if "checkpoint" in path_fields:
            identity = urgent_verifier(
                options["checkpoint"],
                options["source_dir"],
                options.get("expected_sha256"),
                options.get("expected_commit"),
            )
            if (
                not isinstance(identity, Mapping)
                or any(type(key) is not str for key in identity)
                or set(identity) != MOS_ASSET_PROVENANCE_FIELDS[model]
                or any(
                    type(identity.get(key)) is not str
                    or len(identity[key]) != length
                    or any(char not in "0123456789abcdef" for char in identity[key])
                    for key, length in (
                        ("checkpoint_sha256", 64),
                        ("source_commit", 40),
                        ("source_package_sha256", 64),
                    )
                )
            ):
                raise ValueError("Step3 urgent_pk verifier provenance is invalid")
            provenance[model] = dict(identity)
            continue
        if "model_dir" in path_fields:
            model_dir = os.path.realpath(os.path.abspath(options["model_dir"]))
            if not os.path.isdir(model_dir):
                raise ValueError(f"Step3 {model} model_dir must be an existing directory")
            provenance[model] = {
                "model_dir": model_dir,
                "verified_model_tree_sha256": tree_digest(model_dir),
            }
            continue
        model_path = os.path.realpath(os.path.abspath(options["model_path"]))
        if not os.path.isfile(model_path):
            raise ValueError(f"Step3 {model} model_path must be an existing file")
        parent = os.path.dirname(model_path)
        expected = options.get("model_sha256")
        actual_tree = tree_digest(parent)
        if expected != actual_tree:
            raise ValueError(
                f"Step3 {model} model tree digest mismatch: expected={expected}, actual={actual_tree}"
            )
        checkpoint_identity = snapshot_local_checkpoint_identity(model_path)
        expected_checkpoint = options.get("checkpoint_sha256")
        if (
            expected_checkpoint is not None
            and checkpoint_identity["checkpoint_sha256"] != expected_checkpoint
        ):
            raise ValueError(
                f"Step3 {model} checkpoint digest mismatch: "
                f"expected={expected_checkpoint}, "
                f"actual={checkpoint_identity['checkpoint_sha256']}"
            )
        provenance[model] = {
            "model_path": model_path,
            "model_version": options.get("model_version"),
            "expected_model_tree_sha256": expected,
            "verified_model_tree_sha256": actual_tree,
            **checkpoint_identity,
        }
    return {"selected_models": list(selected), "models": provenance}
