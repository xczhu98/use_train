"""TSE 外部 speaker checkpoint 配置解析契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tasks.tse.models.speaker.config import resolve_external_speaker_checkpoint


pytestmark = pytest.mark.unit


def _external_speaker_args(pretrained: object) -> dict:
    return {
        "joint_training": False,
        "features": {"spkemb": {"enabled": True}},
        "speaker_model": {"speaker_encoder": {"pretrained": pretrained}},
    }


def test_repo_relative_checkpoint_uses_explicit_repo_root_outside_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo_root = tmp_path / "repo"
    checkpoint = repo_root / "assets" / "speaker.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"speaker")
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    resolved = resolve_external_speaker_checkpoint(
        _external_speaker_args("assets/speaker.pt"), repo_root=repo_root
    )

    assert resolved == checkpoint.resolve()


@pytest.mark.parametrize(
    ("model_args", "error", "match"),
    [
        ([], TypeError, r"model\.args must be a mapping"),
        (
            {"joint_training": False, "features": []},
            TypeError,
            r"model\.args\.features must be a mapping",
        ),
        (
            {"joint_training": False, "features": {"spkemb": []}},
            TypeError,
            r"model\.args\.features\.spkemb must be a mapping",
        ),
        (
            {
                "joint_training": False,
                "features": {"spkemb": {"enabled": True}},
                "speaker_model": [],
            },
            TypeError,
            r"model\.args\.speaker_model must be a mapping",
        ),
        (
            {
                "joint_training": False,
                "features": {"spkemb": {"enabled": True}},
                "speaker_model": {"speaker_encoder": []},
            },
            TypeError,
            r"speaker_encoder must be a mapping",
        ),
    ],
)
def test_invalid_mapping_sections_fail_fast(
    tmp_path: Path, model_args: object, error: type[Exception], match: str
):
    with pytest.raises(error, match=match):
        resolve_external_speaker_checkpoint(model_args, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("joint_training", "false", r"model\.args\.joint_training.*bool"),
        ("joint_training", 0, r"model\.args\.joint_training.*bool"),
        ("spkemb.enabled", "true", r"spkemb\.enabled.*bool"),
        ("spkemb.enabled", 1, r"spkemb\.enabled.*bool"),
    ],
)
def test_non_boolean_flags_fail_fast(
    tmp_path: Path, key: str, value: object, match: str
):
    model_args = _external_speaker_args("speaker.pt")
    if key == "joint_training":
        model_args["joint_training"] = value
    else:
        model_args["features"]["spkemb"]["enabled"] = value

    with pytest.raises(TypeError, match=match):
        resolve_external_speaker_checkpoint(model_args, repo_root=tmp_path)


@pytest.mark.parametrize(
    "pretrained",
    [
        None,
        True,
        False,
        1,
        Path("speaker.pt"),
        "",
        " speaker.pt",
        "speaker.pt ",
        "missing.pt",
    ],
)
def test_invalid_checkpoint_paths_fail_fast(tmp_path: Path, pretrained: object):
    with pytest.raises((TypeError, ValueError), match=r"pretrained"):
        resolve_external_speaker_checkpoint(
            _external_speaker_args(pretrained), repo_root=tmp_path
        )


def test_checkpoint_directory_is_rejected(tmp_path: Path):
    directory = tmp_path / "speaker.pt"
    directory.mkdir()

    with pytest.raises(ValueError, match=r"existing file"):
        resolve_external_speaker_checkpoint(
            _external_speaker_args("speaker.pt"), repo_root=tmp_path
        )
