"""MuSE visual frontend：kernel 上游对齐、时间对齐、缺失门控与配置边界。

上游 fixture 由固定 WeSep commit ``dc11cb18`` 的 ``wesep/modules/visual/muse.py``
在同一 seed 下直接生成（本测试不依赖上游 checkout）：构造顺序与 seed 必须与 fixture
生成时一致，否则参数初始化不同。
"""

from __future__ import annotations

import os

import pytest
import torch

from tasks.tse.models.modules.visual import (
    MUSE_FRONTEND_FEATURE_DIM,
    VisualFrontend,
    VisualFrontendOutput,
    build_visual_adapter,
    load_muse_frontend,
)
from tasks.tse.models.modules.visual.muse import (
    Muse_LipROIProcessor,
    Muse_VisualFrontend,
    compute_muse_frontend,
)


pytestmark = [pytest.mark.unit]


# --- 上游 fixture（WeSep dc11cb18，同一 seed 下的实际输出） ---------------------
# 重新生成方式（在固定 upstream checkout 上运行，仅用于产出常量）：
#   PYTHONPATH=<upstream> python -c "import torch; from wesep.modules.visual.muse \
#       import Muse_LipROIProcessor, Muse_VisualFrontend, build_visual_adapter, \
#       compute_muse_frontend; ..."
# 各 fixture 的输入：
#   ROI      seed 7001 → rand(1,16,16,3,4)，Muse_LipROIProcessor(roi_size=8, resize_size=16)
#   ENCODER  seed 7002 → rand(1,224,224,3,4) 先于 Muse_VisualFrontend() 构造
#   ADAPTER  构造 seed / 输入 seed 见各 fixture；输入 shape 见对应测试
_UPSTREAM_ROI = {
    "seed": 7001,
    "first8": [
        -1.1427310705, -0.7357692122, 2.1087474823, 0.2233454585,
        0.0013462246, -1.3133841753, -0.9699932337, 1.4212468863,
    ],
    "mean": 0.3265878260,
    "std": 1.0787218809,
}
_UPSTREAM_ENCODER = {
    "seed": 7002,
    "samples": [
        (0, 0, 0.0576228909), (0, 3, 0.0232569948),
        (7, 0, 0.0012278927), (7, 3, 0.0024043610),
        (255, 0, 0.1030916199), (255, 3, 0.0412447490),
        (511, 0, 0.0219828784), (511, 3, 0.0193920769),
    ],
    "frame_mean": [0.0298920441, 0.0349785052, 0.0357540995, 0.0309562199],
    "frame_std": [0.0395495743, 0.0465447605, 0.0487350225, 0.0425113961],
}
_UPSTREAM_MUSE_ADAPTER = {
    "seed": 7101,
    "input_seed": 7102,
    "samples": [
        -1.4578789473, -0.3224201202, -0.2332022190, 1.1359287500,
        -0.7318459749, -0.6494370699, 0.3417099118, -0.0833093971,
        0.9132648706,
    ],
    "mean": 0.1223372594,
    "std": 0.9761830568,
}
_UPSTREAM_CLEARER_ADAPTER = {
    "seed": 7201,
    "input_seed": 7202,
    "samples": [
        -0.3181561530, -1.0684754848, 0.7885979414, -0.3866823316,
        -1.2353316545, 0.0249068290, 0.2651495039, 0.2918572128,
        0.3383347690, -0.6694550514, 0.7475197315, -0.1019195020,
    ],
    "mean": 0.0150313657,
    "std": 0.5550592542,
}
_UPSTREAM_DIRECT_ADAPTER = {
    "seed": 7301,
    "input_seed": 7302,
    "samples": [
        -0.6150752902, 0.5665283203, -0.1744606197, -1.0759801865,
        1.1408326626, -1.6028399467, 0.1021106020, 0.1134701744,
    ],
}


def _frontend(*, adapter=None, representation="features", freeze=None, **extra):
    config: dict[str, object] = dict(extra)
    if adapter is not None:
        config["adapter"] = adapter
    if freeze is not None:
        config["freeze_frontend"] = freeze
    return VisualFrontend(
        representation=representation,
        features={"muse_visual": config},
        sample_rate=16000,
        hop_length=1600,
    )


def _direct(dim: int = 2) -> dict[str, object]:
    return {"type": "direct", "input_dim": dim, "output_dim": dim}


def _muse_adapter(dim: int = 3) -> dict[str, object]:
    return {"type": "muse", "input_dim": dim, "output_dim": dim, "num_layers": 1}


def _times(*rows: list[float]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64)


# --- 上游 kernel 对齐 ---------------------------------------------------------


def test_upstream_lip_roi_processor_matches_pinned_commit():
    torch.manual_seed(_UPSTREAM_ROI["seed"])
    video = torch.rand(1, 16, 16, 3, 4)

    roi = Muse_LipROIProcessor(roi_size=8, resize_size=16)(video)

    assert roi.shape == (1, 4, 8, 8)
    assert roi.dtype == torch.float32
    torch.testing.assert_close(
        roi.flatten()[:8], torch.tensor(_UPSTREAM_ROI["first8"]), rtol=0, atol=2e-7
    )
    assert float(roi.mean()) == pytest.approx(_UPSTREAM_ROI["mean"], abs=1e-6)
    assert float(roi.std()) == pytest.approx(_UPSTREAM_ROI["std"], abs=1e-6)


def test_upstream_muse_encoder_matches_pinned_commit():
    torch.manual_seed(_UPSTREAM_ENCODER["seed"])
    video = torch.rand(1, 224, 224, 3, 4)
    encoder = Muse_VisualFrontend().eval()

    with torch.no_grad():
        features = compute_muse_frontend(video, Muse_LipROIProcessor(), encoder)

    assert features.shape == (1, MUSE_FRONTEND_FEATURE_DIM, 4)
    expected = torch.tensor([value for _, _, value in _UPSTREAM_ENCODER["samples"]])
    actual = torch.tensor(
        [features[0, channel, frame] for channel, frame, _ in _UPSTREAM_ENCODER["samples"]]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
    torch.testing.assert_close(
        features.mean(dim=1)[0], torch.tensor(_UPSTREAM_ENCODER["frame_mean"]),
        rtol=0, atol=1e-6,
    )
    torch.testing.assert_close(
        features.std(dim=1)[0], torch.tensor(_UPSTREAM_ENCODER["frame_std"]),
        rtol=0, atol=1e-6,
    )


def test_upstream_muse_adapter_matches_pinned_commit():
    torch.manual_seed(_UPSTREAM_MUSE_ADAPTER["seed"])
    adapter = build_visual_adapter(
        {"type": "muse", "input_dim": 6, "output_dim": 6, "num_layers": 2}
    ).eval()
    torch.manual_seed(_UPSTREAM_MUSE_ADAPTER["input_seed"])
    value = torch.randn(2, 6, 5)

    with torch.no_grad():
        output = adapter(value)

    assert output.shape == (2, 6, 5)
    torch.testing.assert_close(
        output.flatten()[::7], torch.tensor(_UPSTREAM_MUSE_ADAPTER["samples"]),
        rtol=0, atol=2e-7,
    )
    assert float(output.mean()) == pytest.approx(_UPSTREAM_MUSE_ADAPTER["mean"], abs=1e-6)
    assert float(output.std()) == pytest.approx(_UPSTREAM_MUSE_ADAPTER["std"], abs=1e-6)


def test_upstream_clearer_adapter_matches_pinned_commit():
    torch.manual_seed(_UPSTREAM_CLEARER_ADAPTER["seed"])
    adapter = build_visual_adapter(
        {"type": "clearer", "input_dim": 4, "output_dim": 6, "hidden_dim": 8, "num_layers": 1}
    ).eval()
    torch.manual_seed(_UPSTREAM_CLEARER_ADAPTER["input_seed"])
    value = torch.randn(2, 4, 5)

    with torch.no_grad():
        output = adapter(value)

    assert output.shape == (2, 6, 5)
    torch.testing.assert_close(
        output.flatten()[::5], torch.tensor(_UPSTREAM_CLEARER_ADAPTER["samples"]),
        rtol=0, atol=2e-7,
    )
    assert float(output.mean()) == pytest.approx(_UPSTREAM_CLEARER_ADAPTER["mean"], abs=1e-6)
    assert float(output.std()) == pytest.approx(_UPSTREAM_CLEARER_ADAPTER["std"], abs=1e-6)


def test_direct_adapter_passes_upstream_samples_through():
    torch.manual_seed(_UPSTREAM_DIRECT_ADAPTER["seed"])
    adapter = build_visual_adapter({"type": "direct", "input_dim": 4, "output_dim": 4}).eval()
    torch.manual_seed(_UPSTREAM_DIRECT_ADAPTER["input_seed"])
    value = torch.randn(2, 4, 5)

    with torch.no_grad():
        output = adapter(value)

    assert torch.equal(output, value)
    torch.testing.assert_close(
        output.flatten()[::5], torch.tensor(_UPSTREAM_DIRECT_ADAPTER["samples"]),
        rtol=0, atol=2e-7,
    )


# --- 时间对齐 ----------------------------------------------------------------


def test_frame_times_drive_step_hold_alignment():
    frontend = _frontend(adapter=_direct())
    value = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])
    present = torch.tensor([True])

    output = frontend(
        value, present=present, num_frames=5,
        frame_times=_times([0.0, 0.15, 0.4]),
    )

    # hop/sr = 0.1s：音频帧时间 0,0.1,0.2,0.3,0.4 → 视频帧 0,0,1,1,2
    torch.testing.assert_close(
        output.embedding,
        torch.tensor([[[1.0, 1.0, 2.0, 2.0, 3.0], [10.0, 10.0, 20.0, 20.0, 30.0]]]),
    )
    torch.testing.assert_close(
        output.frame_times,
        torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4]], dtype=torch.float64),
        rtol=0, atol=1e-12,
    )
    assert output.present is present
    assert output.representation == "features"
    assert frontend.frame_interval == pytest.approx(0.1)


def test_fps_path_holds_the_last_valid_video_frame():
    frontend = _frontend(adapter=_direct())
    value = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])

    output = frontend(
        value, present=torch.tensor([True]), num_frames=6, fps=10.0,
    )

    # 视频帧时间 0,0.1,0.2；音频帧 0,0.1,...,0.5 → 索引 0,1,2,2,2,2
    torch.testing.assert_close(
        output.embedding[0, 0], torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0, 3.0])
    )
    assert bool(torch.isfinite(output.embedding).all())


def test_lengths_restrict_alignment_to_the_valid_prefix():
    frontend = _frontend(adapter=_direct())
    value = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])

    output = frontend(
        value,
        present=torch.tensor([True]),
        num_frames=4,
        frame_times=_times([0.0, 0.1, 0.0]),
        lengths=torch.tensor([2], dtype=torch.int32),
    )

    # 第 3 帧不在有效前缀内（padding 时间戳必须为 0），0.1s 之后的音频帧保持第 2 帧
    torch.testing.assert_close(
        output.embedding[0, 0], torch.tensor([1.0, 2.0, 2.0, 2.0])
    )


def test_forward_matches_compute():
    frontend = _frontend(adapter=_direct())
    value = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])
    present = torch.tensor([True])

    called = frontend(value, present=present, num_frames=3, fps=25.0)
    computed = frontend.compute(value, present=present, num_frames=3, fps=25.0)

    assert isinstance(called, VisualFrontendOutput)
    torch.testing.assert_close(called.embedding, computed.embedding, rtol=0, atol=0)
    torch.testing.assert_close(called.frame_times, computed.frame_times, rtol=0, atol=0)


# --- 缺失行与冻结语义 --------------------------------------------------------


def _mixed_batch():
    value = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    present = torch.tensor([True, False])
    frame_times = _times([0.0, 0.15, 0.4], [0.0, 0.0, 0.0])
    lengths = torch.tensor([3, 0], dtype=torch.int32)
    return value, present, frame_times, lengths


def test_missing_rows_are_gated_to_zero_without_polluting_batch_norm():
    frontend = _frontend(adapter=_muse_adapter())
    value, present, frame_times, lengths = _mixed_batch()
    running_before = {
        key: tensor.clone()
        for key, tensor in frontend.adapter.state_dict().items()
        if "running" in key
    }
    assert running_before

    frontend.train()
    output = frontend(
        value, present=present, num_frames=5, frame_times=frame_times, lengths=lengths,
    )

    assert output.embedding.shape == (2, 3, 5)
    assert bool((output.embedding[1] == 0).all())
    assert bool((output.frame_times[1] == 0).all())
    assert bool(torch.isfinite(output.embedding).all())
    assert bool((output.embedding[0] != 0).any())
    for key, before in running_before.items():
        assert torch.equal(frontend.adapter.state_dict()[key], before), key


def test_frozen_frontend_keeps_eval_and_stops_gradients():
    frontend = _frontend(adapter=_muse_adapter(), freeze=True)
    value, present, frame_times, lengths = _mixed_batch()

    assert all(not parameter.requires_grad for parameter in frontend.parameters())
    frontend.train()
    assert frontend.adapter.training is False
    assert not frontend.training or frontend.adapter.training is False

    output = frontend(
        value, present=present, num_frames=4, frame_times=frame_times, lengths=lengths,
    )

    assert output.embedding.requires_grad is False
    assert all(parameter.grad is None for parameter in frontend.adapter.parameters())


def test_trainable_frontend_keeps_gradients_while_staying_eval():
    frontend = _frontend(adapter=_muse_adapter(), freeze=False)
    value, present, frame_times, lengths = _mixed_batch()

    assert all(parameter.requires_grad for parameter in frontend.parameters())
    frontend.train()
    assert frontend.adapter.training is False

    output = frontend(
        value, present=present, num_frames=4, frame_times=frame_times, lengths=lengths,
    )
    output.embedding.sum().backward()

    assert output.embedding.requires_grad is True
    assert all(parameter.grad is not None for parameter in frontend.adapter.parameters())
    assert bool((output.embedding[1] == 0).all())


def test_frontend_owns_no_stft_state():
    frontend = _frontend(adapter=_direct())

    assert not hasattr(frontend, "stft")
    assert not hasattr(frontend, "istft")
    assert not hasattr(frontend, "window")
    assert not [key for key in frontend.state_dict() if "window" in key or "stft" in key]


# --- 非法输入 ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("fps", "error"),
    [
        (None, ValueError),
        (0, ValueError),
        (-25.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("25", TypeError),
        (torch.tensor(25.0), TypeError),
    ],
)
def test_invalid_fps_is_rejected(fps, error):
    frontend = _frontend(adapter=_direct())
    value = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    with pytest.raises(error):
        frontend(value, present=torch.tensor([True]), num_frames=3, fps=fps)


def test_frame_times_and_fps_are_mutually_exclusive():
    frontend = _frontend(adapter=_direct())
    value = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    with pytest.raises(ValueError, match="exactly one of frame_times or fps"):
        frontend(
            value, present=torch.tensor([True]), num_frames=3,
            frame_times=_times([0.0, 0.1]), fps=25.0,
        )


@pytest.mark.parametrize(
    ("present", "frame_times", "lengths", "match"),
    [
        ([True], [[0.0, 0.3, 0.2]], None, "monotonic"),
        ([True], [[0.0, -0.1, 0.2]], None, "non-negative"),
        ([True], [[0.05, 0.1, 0.2]], None, "start at 0"),
        ([True], [[0.0, float("inf"), 0.2]], None, "finite"),
        ([True, False], [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]], None, "zero frame_times placeholders"),
        ([True], [[0.0, 0.1, 0.2]], [2], "after the valid length must be zero"),
    ],
)
def test_invalid_frame_times_are_rejected(present, frame_times, lengths, match):
    frontend = _frontend(adapter=_direct())
    rows = len(present)
    value = torch.ones(rows, 2, 3)
    if not all(present):
        value[-1] = 0
    length_tensor = (
        torch.tensor(lengths, dtype=torch.int32) if lengths is not None else None
    )

    with pytest.raises(ValueError, match=match):
        frontend(
            value, present=torch.tensor(present), num_frames=3,
            frame_times=_times(*frame_times), lengths=length_tensor,
        )


def test_frame_times_shape_and_dtype_are_validated():
    frontend = _frontend(adapter=_direct())
    value = torch.ones(1, 2, 3)

    with pytest.raises(ValueError, match=r"shape \[R,3\]"):
        frontend(value, present=torch.tensor([True]), num_frames=3, frame_times=_times([0.0, 0.1]))
    with pytest.raises(TypeError, match="floating-point"):
        frontend(
            value, present=torch.tensor([True]), num_frames=3,
            frame_times=torch.zeros(1, 3, dtype=torch.int64),
        )
    with pytest.raises(TypeError, match="floating-point"):
        frontend(value, present=torch.tensor([True]), num_frames=3, frame_times=[0.0, 0.1, 0.2])


@pytest.mark.parametrize(
    ("lengths", "error", "match"),
    [
        (torch.tensor([3], dtype=torch.int64), TypeError, "int32"),
        (torch.tensor([[3]], dtype=torch.int32), ValueError, r"shape \[R\]"),
        (torch.tensor([0], dtype=torch.int32), ValueError, r"within \[1,V\]"),
        (torch.tensor([4], dtype=torch.int32), ValueError, r"within \[1,V\]"),
    ],
)
def test_invalid_lengths_are_rejected(lengths, error, match):
    frontend = _frontend(adapter=_direct())
    value = torch.ones(1, 2, 3)

    with pytest.raises(error, match=match):
        frontend(
            value, present=torch.tensor([True]), num_frames=3,
            frame_times=_times([0.0, 0.1, 0.2]), lengths=lengths,
        )


def test_missing_rows_must_use_zero_lengths():
    frontend = _frontend(adapter=_direct())
    value = torch.ones(2, 2, 3)
    present = torch.tensor([True, False])

    with pytest.raises(ValueError, match="zero otherwise"):
        frontend(
            value, present=present, num_frames=3,
            frame_times=_times([0.0, 0.1, 0.2], [0.0, 0.0, 0.0]),
            lengths=torch.tensor([3, 3], dtype=torch.int32),
        )


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (torch.ones(1, 2, 3, dtype=torch.int64), TypeError, "floating-point"),
        (torch.ones(1, 2, 3, 1), ValueError, r"shape \[R,D,V\]"),
        (torch.ones(1, 4, 3), ValueError, "provide 2 channels"),
        (torch.full((1, 2, 3), float("nan")), ValueError, "finite"),
        (torch.ones(2, 2, 3), ValueError, r"row axis \[1\]"),
    ],
)
def test_invalid_feature_values_are_rejected(value, error, match):
    frontend = _frontend(adapter=_direct())

    with pytest.raises(error, match=match):
        frontend(value, present=torch.tensor([True]), num_frames=3, fps=25.0)


def test_invalid_num_frames_and_present_are_rejected():
    frontend = _frontend(adapter=_direct())
    value = torch.ones(1, 2, 3)

    with pytest.raises(ValueError, match="num_frames"):
        frontend(value, present=torch.tensor([True]), num_frames=0, fps=25.0)
    with pytest.raises(TypeError, match="torch.bool"):
        frontend(value, present=torch.tensor([1]), num_frames=3, fps=25.0)
    with pytest.raises(ValueError, match=r"shape \[R\]"):
        frontend(value, present=torch.tensor([[True]]), num_frames=3, fps=25.0)


def test_frontend_rejects_undecoded_video_paths():
    frontend = _frontend(adapter=_direct())

    with pytest.raises(TypeError, match="data layer"):
        frontend("/tmp/clip.mp4", present=torch.tensor([True]), num_frames=3, fps=25.0)


# --- 配置边界 ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("representation", "features", "error", "match"),
    [
        ("video", {"muse_visual": {}}, ValueError, "visual representation"),
        ("features", {}, ValueError, "non-empty mapping"),
        ("features", {"visual_other": {}}, ValueError, "unknown features"),
        (
            "features",
            {"muse_visual": {"fusion": "concat", "adapter": _direct()}},
            ValueError,
            "CueComposer",
        ),
        (
            "features",
            {"muse_visual": {"multi_fuse": True, "adapter": _direct()}},
            ValueError,
            "CueComposer",
        ),
        (
            "features",
            {"muse_visual": {"upsample": False, "adapter": _direct()}},
            ValueError,
            "upsample=False is unsupported",
        ),
        (
            "features",
            {"muse_visual": {"mystery": 1, "adapter": _direct()}},
            ValueError,
            "unknown keys",
        ),
        ("features", {"muse_visual": {}}, ValueError, "explicit adapter"),
        (
            "features",
            {"muse_visual": {"adapter": {"type": "dummy", "input_dim": 2, "output_dim": 2}}},
            ValueError,
            "unsupported visual adapter",
        ),
        (
            "features",
            {"muse_visual": {"adapter": {"type": "direct", "input_dim": 2, "output_dim": 3}}},
            ValueError,
            "input_dim == output_dim",
        ),
        (
            "features",
            {"muse_visual": {"adapter": {"type": "muse", "input_dim": 2, "output_dim": 2}}},
            ValueError,
            "requires num_layers",
        ),
        (
            "features",
            {"muse_visual": {"adapter": {"type": "clearer", "input_dim": 2, "output_dim": 2,
                                         "num_layers": 1}}},
            ValueError,
            "requires hidden_dim",
        ),
        (
            "features",
            {"muse_visual": {"adapter": {"type": "direct", "input_dim": 2, "output_dim": 2,
                                         "hidden_dim": 4}}},
            ValueError,
            "does not consume",
        ),
        (
            "features",
            {"muse_visual": {"adapter": {"type": "direct", "input_dim": 2, "output_dim": 2,
                                         "mystery": 1}}},
            ValueError,
            "unknown keys",
        ),
        (
            "features",
            {"muse_visual": {"adapter": {"type": "direct", "input_dim": 2}}},
            ValueError,
            "requires output_dim",
        ),
        (
            "features",
            {"muse_visual": {"adapter": _direct(), "checkpoint": "/tmp/x.pt"}},
            ValueError,
            "only consumed by raw visual input",
        ),
        (
            "features",
            {"muse_visual": {"adapter": _direct(), "checkpoint": ""}},
            ValueError,
            "non-empty string",
        ),
        (
            "features",
            {"muse_visual": {"adapter": _direct(), "freeze_frontend": "yes"}},
            TypeError,
            "must be a bool",
        ),
        ("raw", {"muse_visual": {}}, ValueError, "MuSE encoder checkpoint"),
        ("features", {"muse_visual": []}, TypeError, "must be a mapping"),
        ("features", {"muse_visual": {"adapter": []}}, TypeError, "must be a mapping"),
    ],
)
def test_visual_frontend_config_boundary(representation, features, error, match):
    with pytest.raises(error, match=match):
        VisualFrontend(
            representation=representation,
            features=features,
            sample_rate=16000,
            hop_length=1600,
        )


@pytest.mark.parametrize(
    ("sample_rate", "hop_length"),
    [(0, 1600), (16000, 0), (16000.0, 1600), (16000, None)],
)
def test_alignment_context_must_be_positive_ints(sample_rate, hop_length):
    with pytest.raises(ValueError, match="positive exact int"):
        VisualFrontend(
            representation="features",
            features={"muse_visual": {"adapter": _direct()}},
            sample_rate=sample_rate,
            hop_length=hop_length,
        )


def test_upsample_true_is_accepted_as_alignment_declaration():
    frontend = _frontend(adapter=_direct(), upsample=True)

    assert frontend.frame_interval == pytest.approx(0.1)


# --- raw 输入路径 ------------------------------------------------------------


def _write_encoder_checkpoint(path, *, wrapper=True, prefix=True, drop=None, extra=None):
    torch.manual_seed(4242)
    encoder = Muse_VisualFrontend()
    state = {}
    for key, value in encoder.state_dict().items():
        if drop is not None and key == drop:
            continue
        state[f"module.{key}" if prefix else key] = value
    if extra is not None:
        state[extra] = torch.zeros(1)
    torch.save({"state_dict": state} if wrapper else state, path)
    return encoder


def test_raw_input_requires_checkpoint_and_loads_strictly(tmp_path):
    checkpoint = tmp_path / "muse_frontend.pt"
    _write_encoder_checkpoint(checkpoint)

    frontend = VisualFrontend(
        representation="raw",
        features={"muse_visual": {"checkpoint": str(checkpoint)}},
        sample_rate=16000,
        hop_length=640,
    )

    assert frontend.input_dim == MUSE_FRONTEND_FEATURE_DIM
    assert frontend.output_dim == MUSE_FRONTEND_FEATURE_DIM
    assert frontend.roi is not None
    assert frontend.encoder is not None
    assert frontend.encoder.training is False
    assert all(not parameter.requires_grad for parameter in frontend.parameters())

    torch.manual_seed(515)
    video = torch.rand(1, 224, 224, 3, 3)
    output = frontend(
        video, present=torch.tensor([True]), num_frames=3,
        frame_times=_times([0.0, 0.04, 0.08]),
    )

    assert output.embedding.shape == (1, MUSE_FRONTEND_FEATURE_DIM, 3)
    with torch.no_grad():
        native = compute_muse_frontend(video, frontend.roi, frontend.encoder)
    torch.testing.assert_close(output.embedding, native, rtol=0, atol=1e-6)


def test_raw_input_accepts_uint8_video(tmp_path):
    checkpoint = tmp_path / "muse_frontend.pt"
    _write_encoder_checkpoint(checkpoint)
    frontend = VisualFrontend(
        representation="raw",
        features={"muse_visual": {"checkpoint": str(checkpoint)}},
        sample_rate=16000,
        hop_length=640,
    )
    video = torch.randint(0, 256, (1, 224, 224, 3, 3), dtype=torch.uint8)

    output = frontend(video, present=torch.tensor([True]), num_frames=2, fps=25.0)

    assert output.embedding.shape == (1, MUSE_FRONTEND_FEATURE_DIM, 2)
    assert bool(torch.isfinite(output.embedding).all())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"drop": "resnet.layer1.conv1a.weight"}, "missing="),
        ({"extra": "resnet.unknown"}, "unexpected="),
    ],
)
def test_raw_input_checkpoint_mismatch_fails_closed(tmp_path, kwargs, match):
    checkpoint = tmp_path / "muse_frontend.pt"
    _write_encoder_checkpoint(checkpoint, **kwargs)

    with pytest.raises(ValueError, match=match):
        load_muse_frontend(Muse_VisualFrontend(), str(checkpoint))


def test_raw_input_accepts_plain_state_dict_without_ddp_prefix(tmp_path):
    checkpoint = tmp_path / "muse_frontend.pt"
    encoder = _write_encoder_checkpoint(checkpoint, wrapper=False, prefix=False)

    frontend = Muse_VisualFrontend()
    load_muse_frontend(frontend, str(checkpoint))

    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(frontend.state_dict()[key], value, rtol=0, atol=0)


def test_raw_input_checkpoint_shape_and_missing_file_fail_fast(tmp_path):
    checkpoint = tmp_path / "muse_frontend.pt"
    _write_encoder_checkpoint(checkpoint)

    with pytest.raises(FileNotFoundError):
        VisualFrontend(
            representation="raw",
            features={"muse_visual": {"checkpoint": str(tmp_path / "absent.pt")}},
            sample_rate=16000,
            hop_length=640,
        )

    broken = tmp_path / "broken.pt"
    torch.manual_seed(4242)
    torch.save({"resnet.layer1.conv1a.weight": torch.zeros(1)}, broken)
    with pytest.raises(ValueError, match="shape"):
        load_muse_frontend(Muse_VisualFrontend(), str(broken))

    empty = tmp_path / "empty.pt"
    torch.save({}, empty)
    with pytest.raises(ValueError, match="non-empty state dict"):
        load_muse_frontend(Muse_VisualFrontend(), str(empty))


def test_raw_value_shape_and_dtype_are_validated(tmp_path):
    checkpoint = tmp_path / "muse_frontend.pt"
    _write_encoder_checkpoint(checkpoint)
    frontend = VisualFrontend(
        representation="raw",
        features={"muse_visual": {"checkpoint": str(checkpoint)}},
        sample_rate=16000,
        hop_length=640,
    )

    with pytest.raises(ValueError, match=r"shape \[R,H,W,3,V\]"):
        frontend(
            torch.rand(1, 16, 16, 4, 2), present=torch.tensor([True]),
            num_frames=2, fps=25.0,
        )
    with pytest.raises(TypeError, match="uint8 or float32"):
        frontend(
            torch.rand(1, 16, 16, 3, 2, dtype=torch.float64), present=torch.tensor([True]),
            num_frames=2, fps=25.0,
        )


def test_visual_facade_exports_frontend_api():
    import tasks.tse.models.modules.visual as visual

    assert visual.VisualFrontend is VisualFrontend
    assert visual.VisualFrontendOutput is VisualFrontendOutput
    assert set(visual.__all__) == {
        "MUSE_FRONTEND_FEATURE_DIM",
        "VisualFrontend",
        "VisualFrontendOutput",
        "VisualRepresentation",
        "build_visual_adapter",
        "load_muse_frontend",
    }
    assert os.path.basename(visual.__file__) == "__init__.py"
