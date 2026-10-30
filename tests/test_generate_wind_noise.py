"""Tests for tools/prepare_data_tools/generators/noise/generate_wind_noise.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.prepare_data_tools.generators.noise._wind_noise_backend import (  # noqa: E402
    WindNoiseGenerator,
    lsf2poly,
)
from tools.prepare_data_tools._common import parse_float_pair  # noqa: E402
from core.data.resource_paths import resolve_manifest_resource_path  # noqa: E402
from tools.prepare_data_tools.generators.noise.generate_wind_noise import (  # noqa: E402
    _make_tasks,
    _make_uid,
    main,
)
from tools.shared.scp_io import read_scp_dict  # noqa: E402


@pytest.mark.unit
def test_parse_float_pair():
    assert parse_float_pair("3,10") == (3.0, 10.0)
    assert parse_float_pair(" 1 , 5.5 ") == (1.0, 5.5)
    with pytest.raises(ValueError):
        parse_float_pair("3")


@pytest.mark.unit
def test_make_uid_encoding():
    uid = _make_uid(fs=16000, gustiness=6.549, idx=7)
    assert uid == "wind_fs16000_gust6.5_00000007"


@pytest.mark.unit
def test_task_iterator_is_lazy_and_preserves_seed_sequence(tmp_path):
    args = argparse.Namespace(
        fs=[8000, 16000],
        num_data=2,
        duration=0.5,
        gustiness_range=(3.0, 10.0),
        seed=42,
        out_dir=tmp_path,
        shards=4,
    )
    tasks = _make_tasks(args)
    assert iter(tasks) is tasks
    materialized = list(tasks)
    assert [task["idx"] for task in materialized] == [0, 1, 2, 3]
    assert [task["seed"] for task in materialized] == [42, 43, 44, 45]
    assert [task["fs"] for task in materialized] == [8000, 8000, 16000, 16000]


@pytest.mark.unit
@pytest.mark.parametrize(
    "args",
    [
        ["--fs", "0"],
        ["--fs", "16000", "16000"],
        ["--num_data", "0"],
        ["--duration", "nan"],
        ["--workers", "0"],
        ["--shards", "0"],
        ["--gustiness_range", "0.5,2"],
    ],
)
def test_main_rejects_invalid_generation_parameters(tmp_path, args):
    with pytest.raises(SystemExit) as exc_info:
        main(["--out_dir", str(tmp_path / "out"), *args])
    assert exc_info.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.unit
def test_overwrite_rejects_output_symlink(tmp_path):
    referent = tmp_path / "referent"
    referent.mkdir()
    link = tmp_path / "out"
    try:
        link.symlink_to(referent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台不支持符号链接")

    with pytest.raises(SystemExit) as exc_info:
        main(["--out_dir", str(link), "--overwrite"])
    assert exc_info.value.code == 2
    assert not list(referent.iterdir())


@pytest.mark.unit
def test_worker_failure_does_not_publish_wind_indexes(monkeypatch, tmp_path):
    class FakePool:
        def __init__(self, _workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def imap_unordered(self, _fn, tasks):
            return (None for _task in tasks)

    monkeypatch.setattr(
        "tools.prepare_data_tools.generators.noise.generate_wind_noise.mp.Pool",
        FakePool,
    )
    out_dir = tmp_path / "out"
    assert main(
        [
            "--out_dir",
            str(out_dir),
            "--num_data",
            "1",
            "--workers",
            "1",
        ]
    ) == 1
    assert not (out_dir / "wind_noise_sources.scp").exists()
    assert not (out_dir / "wind_noise_meta.json").exists()


@pytest.mark.unit
def test_lsf2poly_known_case():
    # spectrum.lsf2poly 的 doctest 参考值（Kondoz 标准算法）
    lsf = [0.7842, 1.5605, 1.8776, 1.8984, 2.3593]
    a = lsf2poly(lsf)
    expected = np.array(
        [
            1.00000000e00,
            6.14837835e-01,
            9.89884967e-01,
            9.31594056e-05,
            3.13713832e-03,
            -8.12002261e-03,
        ]
    )
    assert a.shape == (6,)
    np.testing.assert_allclose(a, expected, atol=1e-6)


@pytest.mark.unit
def test_lsf2poly_properties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = int(rng.integers(2, 12))
        lsf = np.sort(rng.uniform(0.01, np.pi - 0.01, size=p))
        a = lsf2poly(lsf)
        # 输出长度为 p+1，首系数为 1，系数为实数（虚部为 0）
        assert a.shape == (p + 1,)
        assert a[0] == pytest.approx(1.0)
        assert np.isrealobj(a) or np.all(np.abs(np.imag(a)) < 1e-12)


@pytest.mark.unit
def test_lsf2poly_rejects_out_of_range():
    with pytest.raises(ValueError):
        lsf2poly([-0.1, 1.0])
    with pytest.raises(ValueError):
        lsf2poly([1.0, np.pi + 0.1])


@pytest.mark.unit
def test_backend_generate_wind_noise_deterministic():
    kwargs = dict(fs=16000, duration=0.5, generate=True, gustiness=5.0, start_seed=123)
    sig1, profile1 = WindNoiseGenerator(**kwargs).generate_wind_noise()
    sig2, profile2 = WindNoiseGenerator(**kwargs).generate_wind_noise()
    assert sig1.shape == (8000,)
    np.testing.assert_array_equal(sig1, sig2)
    np.testing.assert_array_equal(profile1, profile2)
    # 输出归一化到 0.95 峰值，且非静音
    assert np.max(np.abs(sig1)) == pytest.approx(0.95, rel=1e-6)
    assert np.std(sig1) > 1e-3


@pytest.mark.external
def test_main_end_to_end(tmp_path):
    out_dir = tmp_path / "wind_out"
    rc = main(
        [
            "--out_dir",
            str(out_dir),
            "--fs",
            "16000",
            "--num_data",
            "2",
            "--duration",
            "0.5",
            "--workers",
            "2",
            "--seed",
            "7",
        ]
    )
    assert rc == 0

    # scp 为 4 列且可被共享 scp_io 解析
    scp_path = out_dir / "wind_noise_sources.scp"
    entries = read_scp_dict(scp_path)
    assert len(entries) == 2
    for line in scp_path.read_text().strip().split("\n"):
        assert len(line.split()) == 4

    # wav 采样率 / 帧数正确，路径相对 manifest parent 可恢复
    for path in entries.values():
        assert not Path(path).is_absolute()
        resolved = Path(resolve_manifest_resource_path(path, scp_path))
        info = sf.info(resolved)
        assert info.samplerate == 16000
        assert info.frames == 8000

    # meta json 汇总
    meta = json.loads((out_dir / "wind_noise_meta.json").read_text())
    assert meta["total_files"] == 2
    assert meta["per_fs"]["fs16000"]["count"] == 2

    # 已存在的非空目录默认拒绝写入
    rc2 = main(
        [
            "--out_dir",
            str(out_dir),
            "--fs",
            "16000",
            "--num_data",
            "1",
            "--duration",
            "0.5",
            "--workers",
            "1",
        ]
    )
    assert rc2 == 1


@pytest.mark.external
def test_main_reproducible(tmp_path):
    def run(out_dir: Path) -> dict[str, np.ndarray]:
        rc = main(
            [
                "--out_dir",
                str(out_dir),
                "--fs",
                "16000",
                "--num_data",
                "2",
                "--duration",
                "0.5",
                "--workers",
                "2",
                "--seed",
                "99",
            ]
        )
        assert rc == 0
        scp_path = out_dir / "wind_noise_sources.scp"
        return {
            uid: sf.read(
                resolve_manifest_resource_path(path, scp_path), dtype="float32"
            )[0]
            for uid, path in read_scp_dict(scp_path).items()
        }

    a = run(tmp_path / "run_a")
    b = run(tmp_path / "run_b")
    assert a.keys() == b.keys()
    for uid in a:
        np.testing.assert_array_equal(a[uid], b[uid])
