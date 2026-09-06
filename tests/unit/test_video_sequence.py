from pathlib import Path

import pytest

from digitalsreeni_image_annotator.video_sequence import FrameSequence


def test_frame_sequence_matches_sam3_numeric_stem_sort(tmp_path):
    for name in ["10.png", "2.png", "1.png"]:
        (tmp_path / name).touch()

    sequence = FrameSequence.from_folder(tmp_path)

    assert [frame.name for frame in sequence.frames] == ["1.png", "2.png", "10.png"]
    assert [frame.source_index for frame in sequence.frames] == [1, 2, 10]
    assert sequence.index_for_name("2.png") == 1
    assert sequence.name_for_index(2) == "10.png"


def test_frame_sequence_orders_frame_numbered_names_numerically(tmp_path):
    """"frame10" comes after "frame2", not between "frame1" and "frame2".

    Recording filenames in this lab carry the source frame number
    ("..._T1_f003367.png"), and lexical order puts f10 before f2 whenever
    the numbers are not zero-padded. Ordering by the parsed number is
    what makes A/D stepping and forward tracking follow real time.
    """
    for name in ["frame2.png", "frame10.png", "frame1.png"]:
        (tmp_path / name).touch()

    sequence = FrameSequence.from_folder(tmp_path)

    assert [frame.name for frame in sequence.frames] == [
        "frame1.png",
        "frame2.png",
        "frame10.png",
    ]
    assert [frame.source_index for frame in sequence.frames] == [1, 2, 10]


def test_frame_sequence_matches_sam3_lexical_fallback(tmp_path):
    """No parseable frame number anywhere: fall back to SAM 3's order.

    SAM 3 reads a frame folder in lexical filename order, so when this
    app cannot do better it has to match, or frame index N here would
    not be frame index N inside the tracker.
    """
    for name in ["shot_b.png", "shot_a.png", "shot_c.png"]:
        (tmp_path / name).touch()

    sequence = FrameSequence.from_folder(tmp_path)

    assert [frame.name for frame in sequence.frames] == [
        "shot_a.png",
        "shot_b.png",
        "shot_c.png",
    ]

def test_frame_sequence_lexical_fallback_is_case_sensitive_like_sam3(tmp_path):
    for name in ["a.png", "B.png", "c.png"]:
        (tmp_path / name).touch()

    sequence = FrameSequence.from_folder(tmp_path)

    assert [frame.name for frame in sequence.frames] == ["B.png", "a.png", "c.png"]


def test_frame_sequence_rejects_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="No supported image frames"):
        FrameSequence.from_folder(tmp_path)


def test_frame_sequence_from_paths_preserves_clip_order_and_source_indices(tmp_path):
    paths = [tmp_path / "frame_10.jpg", tmp_path / "frame_20.jpg"]

    sequence = FrameSequence.from_paths(tmp_path, paths, [10, 20])

    assert [frame.name for frame in sequence.frames] == [
        "frame_10.jpg",
        "frame_20.jpg",
    ]
    assert sequence.frame_for_name("frame_20.jpg").source_index == 20


def test_frame_sequence_from_paths_rejects_mismatched_source_indices(tmp_path):
    with pytest.raises(ValueError, match="equal lengths"):
        FrameSequence.from_paths(tmp_path, [tmp_path / "one.jpg"], [])


def test_er70s_frame_names_sort_by_source_frame_number(tmp_path):
    names = [
        "20260814_ER70S-6_300ipm_33V_T1_f001539.png",
        "20260814_ER70S-6_300ipm_33V_T1_f000049.png",
        "20260814_ER70S-6_300ipm_33V_T1_f000003.png",
        "20260814_ER70S-6_300ipm_33V_T1_f000014.png",
        "20260814_ER70S-6_300ipm_33V_T1_f000008.png",
    ]
    for name in names:
        (tmp_path / name).touch()

    sequence = FrameSequence.from_folder(tmp_path)

    assert [frame.source_index for frame in sequence.frames] == [
        3,
        8,
        14,
        49,
        1539,
    ]


def test_tracking_run_stops_before_a_large_source_frame_gap(tmp_path):
    sequence = FrameSequence.from_paths(
        tmp_path,
        [tmp_path / f"frame_{index}.png" for index in range(5)],
        [3, 8, 14, 49, 1539],
    )

    assert sequence.end_index_for_max_gap(0, max_gap=60) == 3
    assert sequence.source_gap_after(3) == 1490
    assert sequence.end_index_for_max_gap(4, max_gap=60) == 4
    assert sequence.source_gap_after(4) is None
