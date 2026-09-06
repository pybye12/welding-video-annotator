"""Telling a hand-drawn label apart from a model-generated one.

The whole review workflow rests on this distinction: manual labels are
ground truth for the lab, SAM 3 masks are suggestions. They share one
data structure, so the only thing separating them is the ``source`` key.
"""

from digitalsreeni_image_annotator.annotation_source import (
    count_ai_annotations,
    count_manual_annotations,
    frame_has_ai_annotations,
    is_ai_generated,
    is_manual,
    is_tracked,
    manual_classes,
    source_label,
)


MANUAL = {"segmentation": [0, 0, 4, 0, 4, 4], "category_name": "droplet"}
TRACKED = {
    "segmentation": [0, 0, 4, 0, 4, 4],
    "category_name": "droplet",
    "source": "sam3_track",
}
DINO = {"segmentation": [0, 0, 4, 0, 4, 4], "source": "dino"}


def test_an_annotation_without_a_source_key_is_a_human_label():
    """Every manual tool has always written a plain dict.

    Projects saved before provenance existed must keep reading as human
    work, or an old .iap would suddenly look machine-generated and the
    "never overwrite a human label" rule would protect nothing.
    """
    assert is_manual(MANUAL)
    assert not is_ai_generated(MANUAL)
    assert source_label(MANUAL) == ""


def test_tracked_and_detected_masks_are_both_recognised_as_ai():
    assert is_ai_generated(TRACKED)
    assert is_ai_generated(DINO)
    assert is_tracked(TRACKED)
    assert not is_tracked(DINO)


def test_source_label_is_what_the_annotation_list_shows():
    assert source_label(TRACKED) == "SAM 3"
    assert source_label(DINO) == "DINO"


def test_an_unknown_source_string_is_not_treated_as_ai():
    """Only sources this app actually produces count.

    An imported COCO file can carry any ``source`` value it likes.
    Guessing that an unrecognised one means "AI" would mark a
    collaborator's imported labels as machine output.
    """
    assert is_manual({"source": "colleague_export"})
    assert source_label({"source": "colleague_export"}) == ""


def test_counts_split_a_frame_between_human_and_machine():
    frame = {"droplet": [MANUAL, TRACKED], "molten_consumable": [TRACKED]}
    assert count_ai_annotations(frame) == 2
    assert count_manual_annotations(frame) == 1
    assert frame_has_ai_annotations(frame)


def test_an_empty_or_missing_frame_counts_as_nothing():
    for empty in ({}, None, {"droplet": []}):
        assert count_ai_annotations(empty) == 0
        assert count_manual_annotations(empty) == 0
        assert not frame_has_ai_annotations(empty)


def test_manual_classes_names_only_the_classes_a_person_labelled():
    """Drives the "you already labelled this class here" report.

    A class present only as a tracked mask must not appear, or the
    report would claim human coverage the frame does not have.
    """
    frame = {"droplet": [TRACKED], "molten_consumable": [MANUAL, TRACKED]}
    assert manual_classes(frame) == {"molten_consumable"}
