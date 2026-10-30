"""TSE typed cue contract tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tasks.tse.cue_contract import (
    TSECue,
    TSECueSpec,
    TSECueSpecs,
    TSECues,
    collate_tse_cues,
)


def _full_specs() -> TSECueSpecs:
    return TSECueSpecs(
        speaker_waveform=TSECueSpec("waveform", (6,)),
        speaker_embedding=TSECueSpec("embedding", (3,)),
        spatial=TSECueSpec("track", (2, 4)),
        visual=TSECueSpec("features", (5, 3)),
        textual=TSECueSpec("phoneme_ids", (7,)),
    )


def _present_cues() -> TSECues:
    return TSECues(
        speaker_waveform=TSECue(
            "waveform", np.arange(5, dtype=np.float32), True, 5, None
        ),
        speaker_embedding=TSECue(
            "embedding", np.ones(3, dtype=np.float32), True, None, None
        ),
        spatial=TSECue(
            "track",
            np.ones((2, 3), dtype=np.float32),
            True,
            3,
            np.array([0, 160, 320], dtype=np.int64),
        ),
        visual=TSECue(
            "features",
            np.ones((5, 2), dtype=np.float32),
            True,
            2,
            np.array([0.0, 0.04], dtype=np.float64),
        ),
        textual=TSECue(
            "phoneme_ids", np.array([1, 2, 3], dtype=np.int64), True, 3, None
        ),
    )


def _missing_cues() -> TSECues:
    return TSECues(
        speaker_waveform=TSECue("waveform", None, False, 0, None),
        speaker_embedding=TSECue("embedding", None, False, None, None),
        spatial=TSECue("track", None, False, 0, None),
        visual=TSECue("features", None, False, 0, None),
        textual=TSECue("phoneme_ids", None, False, 0, None),
    )


class TestTSECueContracts:
    def test_all_typed_slots_collate_with_fixed_shapes_and_dtypes(self):
        batches = collate_tse_cues([_present_cues(), _missing_cues()], _full_specs())

        waveform = batches.speaker_waveform
        embedding = batches.speaker_embedding
        spatial = batches.spatial
        visual = batches.visual
        textual = batches.textual
        assert waveform is not None and waveform.value.shape == (2, 6)
        assert waveform.value.dtype == torch.float32
        assert waveform.present.tolist() == [True, False]
        assert waveform.lengths is not None and waveform.lengths.tolist() == [5, 0]
        assert embedding is not None and embedding.value.shape == (2, 3)
        assert embedding.lengths is None and embedding.timebase is None
        assert spatial is not None and spatial.value.shape == (2, 2, 4)
        assert spatial.timebase is not None and spatial.timebase.dtype == torch.int64
        assert spatial.timebase.tolist() == [[0, 160, 320, 0], [0, 0, 0, 0]]
        assert visual is not None and visual.value.shape == (2, 5, 3)
        assert visual.timebase is not None and visual.timebase.dtype == torch.float64
        assert textual is not None and textual.value.dtype == torch.int64
        assert textual.value.tolist()[0][:3] == [1, 2, 3]
        assert torch.count_nonzero(textual.value[1]) == 0

    def test_all_missing_rows_use_configured_shape(self):
        specs = TSECueSpecs(visual=TSECueSpec("features", (8, 11)))
        rows = [
            TSECues(visual=TSECue("features", None, False, 0, None)),
            TSECues(visual=TSECue("features", None, False, 0, None)),
        ]
        visual = collate_tse_cues(rows, specs).visual
        assert visual is not None
        assert visual.value.shape == (2, 8, 11)
        assert visual.present.tolist() == [False, False]
        assert visual.lengths is not None and visual.lengths.tolist() == [0, 0]
        assert visual.timebase is not None and visual.timebase.shape == (2, 11)

    def test_speaker_waveform_and_embedding_can_both_be_present(self):
        specs = TSECueSpecs(
            speaker_waveform=TSECueSpec("waveform", (4,)),
            speaker_embedding=TSECueSpec("embedding", (2,)),
        )
        cues = TSECues(
            speaker_waveform=TSECue(
                "waveform", np.ones(4, dtype=np.float32), True, 4, None
            ),
            speaker_embedding=TSECue(
                "embedding", np.ones(2, dtype=np.float32), True, None, None
            ),
        )
        result = collate_tse_cues([cues], specs)
        assert result.speaker_waveform is not None
        assert result.speaker_embedding is not None
        assert result.speaker_waveform.present.tolist() == [True]
        assert result.speaker_embedding.present.tolist() == [True]

    def test_static_spatial_has_no_fake_length_or_timebase(self):
        specs = TSECueSpecs(spatial=TSECueSpec("static", (2,)))
        rows = [
            TSECues(
                spatial=TSECue(
                    "static", np.array([0.2, -0.1], dtype=np.float32), True, None, None
                )
            ),
            TSECues(spatial=TSECue("static", None, False, None, None)),
        ]
        spatial = collate_tse_cues(rows, specs).spatial
        assert spatial is not None
        assert spatial.value.shape == (2, 2)
        assert spatial.lengths is None
        assert spatial.timebase is None

    def test_raw_visual_preserves_uint8_and_frame_time_float64(self):
        specs = TSECueSpecs(visual=TSECueSpec("raw", (2, 3, 3, 4)))
        raw = np.ones((2, 3, 3, 2), dtype=np.uint8)
        rows = [
            TSECues(
                visual=TSECue(
                    "raw", raw, True, 2, np.array([0.0, 0.1], dtype=np.float64)
                )
            )
        ]
        visual = collate_tse_cues(rows, specs).visual
        assert visual is not None
        assert visual.value.dtype == torch.uint8
        assert visual.value.shape == (1, 2, 3, 3, 4)
        assert visual.timebase is not None and visual.timebase.dtype == torch.float64

    def test_unknown_or_wrong_representation_is_rejected(self):
        specs = TSECueSpecs(spatial=TSECueSpec("embedding", (2,)))
        with pytest.raises(ValueError, match="spatial representation"):
            specs.validate()

        specs = TSECueSpecs(spatial=TSECueSpec("static", (2,)))
        cues = TSECues(spatial=TSECue("track", None, False, 0, None))
        with pytest.raises(ValueError, match="representation mismatch"):
            cues.validate(specs)

    def test_enabled_slot_requires_explicit_missing_cue(self):
        specs = TSECueSpecs(textual=TSECueSpec("phoneme_ids", (4,)))
        with pytest.raises(ValueError, match="enabled textual slot"):
            TSECues().validate(specs)

    def test_disabled_slot_rejects_supplied_cue(self):
        cues = TSECues(
            speaker_embedding=TSECue("embedding", None, False, None, None)
        )
        with pytest.raises(ValueError, match="not enabled"):
            cues.validate(TSECueSpecs())

    @pytest.mark.parametrize(
        "cue",
        [
            TSECue("waveform", np.zeros(2, dtype=np.float32), False, 0, None),
            TSECue("waveform", None, False, 1, None),
            TSECue("waveform", None, True, 1, None),
        ],
    )
    def test_presence_value_and_length_are_not_inferred(self, cue):
        specs = TSECueSpecs(speaker_waveform=TSECueSpec("waveform", (2,)))
        with pytest.raises(ValueError):
            TSECues(speaker_waveform=cue).validate(specs)

    def test_temporal_value_can_grow_beyond_configured_fallback(self):
        specs = TSECueSpecs(textual=TSECueSpec("phoneme_ids", (2,)))
        cues = TSECues(
            textual=TSECue(
                "phoneme_ids",
                np.array([1, 2, 3, 99, 99], dtype=np.int64),
                True,
                3,
                None,
            )
        )

        batches = collate_tse_cues([cues], specs)

        assert batches.textual is not None
        assert batches.textual.value.shape == (1, 5)
        assert batches.textual.value.tolist() == [[1, 2, 3, 0, 0]]
        assert batches.textual.lengths is not None
        assert batches.textual.lengths.tolist() == [3]
        batches.validate(specs, 1)

    def test_temporal_capacity_uses_longest_present_storage_in_batch(self):
        specs = TSECueSpecs(speaker_waveform=TSECueSpec("waveform", (2,)))
        rows = [
            TSECues(
                speaker_waveform=TSECue(
                    "waveform", np.ones(5, dtype=np.float32), True, 3, None
                )
            ),
            TSECues(
                speaker_waveform=TSECue(
                    "waveform", np.ones(8, dtype=np.float32), True, 8, None
                )
            ),
            TSECues(
                speaker_waveform=TSECue("waveform", None, False, 0, None)
            ),
        ]

        waveform = collate_tse_cues(rows, specs).speaker_waveform

        assert waveform is not None
        assert waveform.value.shape == (3, 8)
        assert waveform.lengths is not None and waveform.lengths.tolist() == [3, 8, 0]
        assert torch.all(waveform.value[0, 3:] == 0)
        assert torch.all(waveform.value[2] == 0)

    def test_non_temporal_dimensions_still_match_config_exactly(self):
        specs = TSECueSpecs(visual=TSECueSpec("features", (4, 2)))
        cues = TSECues(
            visual=TSECue(
                "features",
                np.zeros((5, 7), dtype=np.float32),
                True,
                7,
                np.arange(7, dtype=np.float64),
            )
        )
        with pytest.raises(ValueError, match="non-temporal shape"):
            cues.validate(specs)

    def test_timebase_dtype_and_monotonicity_are_enforced(self):
        specs = TSECueSpecs(spatial=TSECueSpec("track", (2, 3)))
        wrong_dtype = TSECues(
            spatial=TSECue(
                "track",
                np.zeros((2, 3), dtype=np.float32),
                True,
                3,
                np.array([0.0, 1.0, 2.0], dtype=np.float32),
            )
        )
        with pytest.raises(TypeError, match="torch.int64"):
            wrong_dtype.validate(specs)

        non_monotonic = TSECues(
            spatial=TSECue(
                "track",
                np.zeros((2, 3), dtype=np.float32),
                True,
                3,
                np.array([0, 2, 1], dtype=np.int64),
            )
        )
        with pytest.raises(ValueError, match="monotonic"):
            non_monotonic.validate(specs)

    def test_visual_timebase_requires_float_input(self):
        specs = TSECueSpecs(visual=TSECueSpec("features", (2, 2)))
        cues = TSECues(
            visual=TSECue(
                "features",
                np.zeros((2, 2), dtype=np.float32),
                True,
                2,
                np.array([0, 1], dtype=np.int64),
            )
        )
        with pytest.raises(TypeError, match="floating-point"):
            cues.validate(specs)

    def test_timebase_must_be_relative_to_mixture_start(self):
        specs = TSECueSpecs(spatial=TSECueSpec("track", (2, 2)))
        cues = TSECues(
            spatial=TSECue(
                "track",
                np.zeros((2, 2), dtype=np.float32),
                True,
                2,
                np.array([-1, 0], dtype=np.int64),
            )
        )
        with pytest.raises(ValueError, match="non-negative"):
            cues.validate(specs)

    def test_batch_validate_rejects_nonzero_missing_placeholder(self):
        specs = TSECueSpecs(textual=TSECueSpec("phoneme_ids", (3,)))
        batches = collate_tse_cues(
            [TSECues(textual=TSECue("phoneme_ids", None, False, 0, None))],
            specs,
        )
        assert batches.textual is not None
        batches.textual.value[0, 0] = 1
        with pytest.raises(ValueError, match="zero value placeholders"):
            batches.validate(specs, 1)

    def test_batch_validate_rejects_nonzero_padded_tail(self):
        specs = TSECueSpecs(textual=TSECueSpec("phoneme_ids", (3,)))
        batches = collate_tse_cues(
            [
                TSECues(
                    textual=TSECue(
                        "phoneme_ids", np.array([1], dtype=np.int64), True, 1, None
                    )
                )
            ],
            specs,
        )
        assert batches.textual is not None
        batches.textual.value[0, 2] = 2
        with pytest.raises(ValueError, match="after valid length"):
            batches.validate(specs, 1)

    def test_batch_to_recurses_through_all_enabled_slots(self):
        batches = collate_tse_cues([_present_cues()], _full_specs())
        moved = batches.to("cpu")
        assert moved is not batches
        for slot in (
            "speaker_waveform",
            "speaker_embedding",
            "spatial",
            "visual",
            "textual",
        ):
            assert getattr(moved, slot) is not getattr(batches, slot)
        moved.validate(_full_specs(), 1)


pytestmark = [pytest.mark.unit]
