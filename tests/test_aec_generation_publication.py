"""AEC GenerationManifest publication 下线合同。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.prepare_data_tools.generators.aec import (
    generate_ood_stress_set as ood_generator,
)
from tools.prepare_data_tools.generators.aec import (
    generate_simulated_testset as simulated_generator,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.mark.parametrize(
    ("generator", "argv"),
    [
        (
            simulated_generator,
            [
                "--config",
                "missing.yaml",
                "--output-dir",
                "unused-output",
                "--num-samples",
                "1",
            ],
        ),
        (
            ood_generator,
            [
                "--output-dir",
                "unused-output",
                "--scenarios",
                "reference_polarity_inversion",
                "--num-per-scenario",
                "1",
            ],
        ),
    ],
)
def test_aec_generation_manifest_publication_is_retired(
    tmp_path: Path,
    generator: object,
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "published"
    resolved_argv = [str(output) if value == "unused-output" else value for value in argv]
    if generator is simulated_generator:
        config = tmp_path / "config.yaml"
        config.write_text("data:\n  train: {}\n", encoding="utf-8")
        resolved_argv[resolved_argv.index("missing.yaml")] = str(config)

    assert generator.main(resolved_argv) == 1
    assert "GenerationManifest v1 publication 已下线" in capsys.readouterr().err
    assert not output.exists()


def test_aec_generator_versions_remain_explicit() -> None:
    history = simulated_generator.GENERATOR_VERSION_HISTORY
    assert simulated_generator.GENERATOR_VERSION == history[-1]
    versions = [tuple(map(int, item.split("."))) for item in history]
    assert versions == sorted(set(versions))
    assert versions[-1] >= (2, 18)  # canonical 场景目录发布语义的最低版本
    assert ood_generator.GENERATOR_VERSION == "1.2"
