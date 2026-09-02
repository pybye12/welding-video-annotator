from digitalsreeni_image_annotator.annotation_history import AnnotationHistory


def _frame(count):
    return {"droplet": [{"segmentation": [0, 0, 1, 1, 2, 2], "n": i} for i in range(count)]}


def test_undo_returns_the_state_recorded_before_the_edit():
    history = AnnotationHistory()
    before = _frame(1)

    history.record("frame_0001.png", before, "adding a polygon")
    after = _frame(2)

    label, restored = history.undo("frame_0001.png", after)

    assert label == "adding a polygon"
    assert restored == before


def test_snapshots_are_detached_from_the_live_annotation_dict():
    """A snapshot must survive later in-place edits of the same dict."""
    history = AnnotationHistory()
    live = _frame(1)
    history.record("frame.png", live, "edit")

    live["droplet"][0]["segmentation"].append(99)
    live["droplet"].append({"segmentation": [5, 5]})

    _, restored = history.undo("frame.png", live)
    assert len(restored["droplet"]) == 1
    assert restored["droplet"][0]["segmentation"] == [0, 0, 1, 1, 2, 2]


def test_redo_replays_the_undone_state():
    history = AnnotationHistory()
    before, after = _frame(1), _frame(2)
    history.record("frame.png", before)

    history.undo("frame.png", after)
    assert history.can_redo("frame.png")

    _, replayed = history.redo("frame.png", before)
    assert replayed == after


def test_recording_a_new_edit_drops_the_redo_branch():
    history = AnnotationHistory()
    history.record("frame.png", _frame(1))
    history.undo("frame.png", _frame(2))
    assert history.can_redo("frame.png")

    history.record("frame.png", _frame(1))

    assert not history.can_redo("frame.png")


def test_history_is_per_frame():
    """Undo on one frame must never rewrite another."""
    history = AnnotationHistory()
    history.record("a.png", _frame(1))

    assert history.can_undo("a.png")
    assert not history.can_undo("b.png")
    assert history.undo("b.png", _frame(3)) is None


def test_depth_is_bounded_and_keeps_the_most_recent_states():
    history = AnnotationHistory(depth=3)
    for step in range(6):
        history.record("frame.png", _frame(step), f"step {step}")

    assert history.undo_label("frame.png") == "step 5"

    labels = []
    current = _frame(9)
    while history.can_undo("frame.png"):
        label, current = history.undo("frame.png", current)
        labels.append(label)

    assert labels == ["step 5", "step 4", "step 3"]


def test_forget_drops_history_for_a_removed_frame():
    history = AnnotationHistory()
    history.record("gone.png", _frame(1))

    history.forget("gone.png")

    assert not history.can_undo("gone.png")


def test_empty_frame_state_round_trips_as_no_annotations():
    history = AnnotationHistory()
    history.record("frame.png", None, "first polygon")

    _, restored = history.undo("frame.png", _frame(1))

    assert restored == {}
