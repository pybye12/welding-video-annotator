"""Reviewing a SAM 3 tracking run and taking it back.

A tracking run over a welding clip writes masks to dozens of frames at
once. When it goes wrong — and it does, a molten_consumable mask lands on
the background plate — the annotator needs three things: to see which
polygons the model produced, to remove one by clicking it, and to undo
the entire run in a single action so the frames can be labelled by hand.
"""

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor, QKeyEvent
from PyQt6.QtWidgets import QLabel, QListWidgetItem, QMessageBox

from digitalsreeni_image_annotator.annotator_window import ImageAnnotator


SQUARE = [0, 0, 10, 0, 10, 10, 0, 10]
FAR_SQUARE = [100, 100, 110, 100, 110, 110, 100, 110]


def _window(qtbot):
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.auto_save = lambda: None
    window.show_info = lambda *args: None
    window.show_warning = lambda *args: None
    return window


def _manual(class_name="droplet", segmentation=None, number=1):
    return {
        "segmentation": list(segmentation or SQUARE),
        "category_name": class_name,
        "category_id": 1,
        "number": number,
    }


def _tracked(class_name="droplet", segmentation=None, number=2, source_frame="a.png"):
    annotation = _manual(class_name, segmentation, number)
    annotation.update(
        {
            "source": "sam3_track",
            "sam3_source_frame": source_frame,
            "sam3_source_id": "abc123",
        }
    )
    return annotation


def _open_frame(window, name, annotations):
    """Put a frame on screen with the given ``{class: [annotation]}``."""
    window.image_file_name = name
    window.current_slice = None
    window.all_annotations[name] = annotations
    window.image_label.annotations = annotations
    for class_name in annotations:
        window.image_label.class_colors.setdefault(class_name, QColor("red"))
        # A class hidden in the class list is skipped by hit-testing, and
        # an absent one counts as hidden — so the class has to be there
        # and checked for a canvas click to find anything.
        if not window.class_list.findItems(class_name, Qt.MatchFlag.MatchExactly):
            item = QListWidgetItem(class_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            window.class_list.addItem(item)
    window.update_annotation_list()
    window.update_selection_readout()


def _press(window, key):
    window.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    )


# --- Issue 7: telling a machine mask from a human one ---------------------


def test_a_tracked_mask_is_tagged_in_the_annotation_list(qtbot):
    """The provenance is already in the data; the list has to show it.

    Without this the reviewer cannot tell which polygons to scrutinise,
    and a wrong SAM mask is indistinguishable from their own work.
    """
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_manual(), _tracked()]})

    rows = [
        window.annotation_list.item(i).text()
        for i in range(window.annotation_list.count())
    ]

    assert len(rows) == 2
    assert not any("[SAM 3]" in row for row in rows[:1])
    assert any("[SAM 3]" in row for row in rows)


def test_the_tag_survives_re_sorting_the_list(qtbot):
    """Sorting rebuilds the list through a different function.

    Two builders drifting apart is exactly how a tracked mask would look
    hand-drawn after one click on "Sort by Area".
    """
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_manual(), _tracked()]})

    window.sort_annotations_by_area()

    rows = [
        window.annotation_list.item(i).text()
        for i in range(window.annotation_list.count())
    ]
    assert sum("[SAM 3]" in row for row in rows) == 1


def test_the_readout_separates_frame_state_from_polygon_state(qtbot):
    """"A frame is open" and "a polygon is selected" are different.

    Confusing them is what makes "No valid polygon annotations selected"
    read as a bug when a frame row is highlighted in the Frames list.
    """
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_manual(), _tracked()]})

    assert "2 annotations on this frame" in window.annotation_selection_label.text()
    assert "1 from SAM 3" in window.annotation_selection_label.text()
    assert "none selected" in window.annotation_selection_label.text()

    window.annotation_list.setCurrentRow(0)

    assert "1 selected" in window.annotation_selection_label.text()


# --- Issues 2 and 4: selecting the polygon you clicked --------------------


def test_clicking_a_polygon_selects_it_without_entering_edit_mode(qtbot):
    """Selection used to require a double click, which also opened
    vertex editing. Reviewing forty tracked masks that way is unusable.
    """
    window = _window(qtbot)
    tracked = _tracked()
    _open_frame(window, "a.png", {"droplet": [tracked]})

    picked = window.image_label.select_annotation_at((5, 5))

    assert picked is tracked
    assert window.has_annotation_selection()
    assert window.selected_annotations() == [tracked]
    assert window.image_label.editing_polygon is None


def test_clicking_empty_image_clears_the_selection(qtbot):
    """"Nothing is selected" has to be reachable on purpose.

    Otherwise a stale selection keeps the track button live and the next
    run is seeded from a polygon the annotator forgot about.
    """
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_manual()]})
    window.image_label.select_annotation_at((5, 5))
    assert window.has_annotation_selection()

    picked = window.image_label.select_annotation_at((500, 500))

    assert picked is None
    assert not window.has_annotation_selection()


def test_the_topmost_polygon_wins_when_two_overlap(qtbot):
    """A tracked mask usually sits on top of the polygon it came from."""
    window = _window(qtbot)
    manual = _manual()
    tracked = _tracked()
    _open_frame(window, "a.png", {"droplet": [manual, tracked]})

    assert window.image_label.select_annotation_at((5, 5)) is tracked


def test_a_tracked_mask_is_never_confused_with_its_source_polygon(qtbot):
    """Same geometry, different provenance — selection must not mix them.

    Rows are matched by value (Qt copies a dict on its way through item
    data, so object identity is not available). That is safe only
    because a tracked mask carries provenance keys the hand-drawn
    polygon lacks. If it ever stopped doing so, deleting the AI copy
    could remove the human original instead — which is the one outcome
    this whole workflow exists to prevent.
    """
    window = _window(qtbot)
    manual = _manual(number=1)
    tracked = _tracked(number=1)  # identical geometry
    assert manual != tracked
    _open_frame(window, "a.png", {"droplet": [manual, tracked]})

    window.select_annotation_in_list(tracked)
    selected = window.selected_annotations()[0]

    assert selected["source"] == "sam3_track"

    window.select_annotation_in_list(manual)
    assert "source" not in window.selected_annotations()[0]


# --- Issue 6: Delete and Backspace, whatever has focus --------------------


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
def test_delete_and_backspace_remove_a_polygon_selected_on_the_canvas(qtbot, key):
    """Focus is on the canvas after clicking a polygon there.

    Previously Delete only fired while the annotation list had focus, so
    the natural click-then-delete gesture did nothing at all.
    """
    window = _window(qtbot)
    tracked = _tracked()
    _open_frame(window, "a.png", {"droplet": [_manual(), tracked]})
    window.image_label.select_annotation_at((5, 5))

    _press(window, key)

    remaining = window.all_annotations["a.png"]["droplet"]
    assert tracked not in remaining
    assert len(remaining) == 1


def test_deleting_one_annotation_does_not_stop_to_ask(qtbot, monkeypatch):
    """One confirm dialog per bad mask makes a review pass unbearable.

    The deletion is recorded in the undo history, so Ctrl+Z is the
    safety net rather than a modal.
    """
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_tracked()]})
    window.image_label.select_annotation_at((5, 5))

    def fail(*args, **kwargs):
        raise AssertionError("a single deletion must not prompt")

    monkeypatch.setattr(QMessageBox, "question", fail)

    _press(window, Qt.Key.Key_Delete)

    assert window.all_annotations["a.png"]["droplet"] == []


def test_deleting_several_annotations_still_asks(qtbot, monkeypatch):
    """A multi-select delete is the destructive one worth confirming."""
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_manual(), _tracked()]})
    window.annotation_list.selectAll()
    asked = []

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: asked.append(True) or QMessageBox.StandardButton.No,
    )

    _press(window, Qt.Key.Key_Delete)

    assert asked
    assert len(window.all_annotations["a.png"]["droplet"]) == 2


def test_backspace_never_deletes_a_frame(qtbot):
    """Binding Backspace to the Frames list would put "remove this
    frame" one stray keystroke from the labelling hand.
    """
    window = _window(qtbot)
    window.image_list.addItem("a.png")
    window.image_list.setCurrentRow(0)
    removed = []
    window.delete_selected_image = lambda: removed.append(True)

    _press(window, Qt.Key.Key_Backspace)

    assert removed == []


# --- Issues 1, 2 and 7: the track button says what it needs ---------------


def test_track_selected_is_disabled_until_frames_are_prepared(qtbot):
    """Step 1 is genuinely mandatory, so step 2 should not look ready.

    Leaving it enabled is what produced "Please initialize the tracker
    first" as the answer to a button press.
    """
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_manual()]})
    window.annotation_list.setCurrentRow(0)

    assert not window.sam3_track_forward_btn.isEnabled()
    assert not window.sam3_track_all_btn.isEnabled()
    assert "Prepare Loaded Frames" in window.sam3_track_forward_btn.toolTip()


class _PreparedTracker:
    """Stands in for a tracker with frames already prepared."""

    is_initialized = True

    def unload(self):
        return None

    def close_session(self):
        return None


def test_selecting_a_frame_row_is_not_selecting_a_polygon(qtbot):
    """The exact confusion behind "No valid polygon annotations selected".

    A highlighted row in the Frames list looks like a selection, and the
    tooltip has to say why it is not the one that counts.
    """
    window = _window(qtbot)
    window.sam3_tracker = _PreparedTracker()
    _open_frame(window, "a.png", {"droplet": [_manual()]})
    window.image_list.addItem("a.png")
    window.image_list.setCurrentRow(0)
    window.update_tracking_controls()

    assert not window.sam3_track_forward_btn.isEnabled()
    assert "Frames list is not the same thing" in (
        window.sam3_track_forward_btn.toolTip()
    )
    # Track All only needs a polygon on the frame, not a selection.
    assert window.sam3_track_all_btn.isEnabled()


def test_track_selected_turns_on_once_a_polygon_is_selected(qtbot):
    window = _window(qtbot)
    window.sam3_tracker = _PreparedTracker()
    _open_frame(window, "a.png", {"droplet": [_manual()]})

    window.image_label.select_annotation_at((5, 5))

    assert window.sam3_track_forward_btn.isEnabled()


def test_a_polygon_too_small_to_track_does_not_enable_the_button(qtbot):
    """Mirrors the filter inside the run: fewer than three points is
    rejected there, so the button must not promise otherwise.
    """
    window = _window(qtbot)
    window.sam3_tracker = _PreparedTracker()
    _open_frame(window, "a.png", {"droplet": [_manual(segmentation=[0, 0, 4, 4])]})
    window.annotation_list.setCurrentRow(0)

    assert not window.sam3_track_forward_btn.isEnabled()
    assert not window.sam3_track_all_btn.isEnabled()


def test_the_button_keeps_saying_what_it_does_while_disabled(qtbot):
    """The reason replaces nothing: a disabled button still has to
    explain its purpose, not only its complaint.
    """
    window = _window(qtbot)
    tip = window.sam3_track_forward_btn.toolTip()
    assert "nearby loaded frames" in tip
    assert "Prepare Loaded Frames" in tip


# --- Issue 5: clearing a frame SAM 3 got wrong ----------------------------


def _tracked_frame(window, name, extra=None):
    """A frame carrying a tracked mask, plus anything else given."""
    annotations = {"droplet": [_tracked()]}
    if extra:
        for class_name, items in extra.items():
            annotations.setdefault(class_name, []).extend(items)
    window.all_annotations[name] = annotations
    return annotations


def test_clearing_ai_masks_keeps_the_labels_you_drew(qtbot):
    """The review move: SAM got this frame wrong, take its masks off.

    Only model output goes. A hand-drawn polygon already on the frame is
    ground truth and must survive, or "clear and redo" would quietly
    destroy the very work it exists to protect.
    """
    window = _window(qtbot)
    manual = _manual("molten_consumable")
    _open_frame(
        window,
        "a.png",
        {"droplet": [_tracked(), _tracked(number=3)], "molten_consumable": [manual]},
    )

    assert window.clear_ai_annotations_on_frame() == 2

    remaining = window.all_annotations["a.png"]
    assert "droplet" not in remaining
    assert remaining["molten_consumable"] == [manual]


def test_clearing_ai_masks_is_a_single_undo_step(qtbot):
    """Ctrl+Z puts them back if the frame turns out to have been fine."""
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_tracked(), _tracked(number=3)]})

    window.clear_ai_annotations_on_frame()
    assert window.all_annotations.get("a.png", {}).get("droplet", []) == []

    window.undo_annotation_change()

    assert len(window.all_annotations["a.png"]["droplet"]) == 2


def test_clearing_touches_only_the_frame_in_view(qtbot):
    """Per frame, deliberately.

    The annotator asked to fix the frame in front of them, not to take
    back a whole run — a frame they have already checked and accepted
    must not be cleared behind their back.
    """
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_tracked()]})
    _tracked_frame(window, "b.png")

    window.clear_ai_annotations_on_frame()

    assert window.all_annotations.get("a.png", {}).get("droplet", []) == []
    assert len(window.all_annotations["b.png"]["droplet"]) == 1


def test_clearing_a_frame_with_no_ai_masks_does_nothing(qtbot):
    """Not even an empty history step, which would break the redo branch."""
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_manual()]})

    assert window.clear_ai_annotations_on_frame() == 0
    assert not window.annotation_history.can_undo("a.png")
    assert len(window.all_annotations["a.png"]["droplet"]) == 1


def test_shift_delete_clears_the_ai_masks_on_this_frame(qtbot):
    """Keeps a review pass on the canvas: D, look, Shift+Del, draw."""
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_tracked()]})

    window.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Delete,
            Qt.KeyboardModifier.ShiftModifier,
        )
    )

    assert window.all_annotations.get("a.png", {}).get("droplet", []) == []


def test_plain_delete_still_deletes_only_the_selection(qtbot):
    """Regression: the plain-Delete branch matches the key regardless of
    modifiers, so Shift+Delete has to be tested first — and, just as
    importantly, adding it must not turn an ordinary Delete into a
    clear-the-frame.
    """
    window = _window(qtbot)
    manual = _manual()
    tracked = _tracked()
    _open_frame(window, "a.png", {"droplet": [manual, tracked]})
    window.image_label.select_annotation_at((5, 5))  # picks the tracked one

    _press(window, Qt.Key.Key_Delete)

    remaining = window.all_annotations["a.png"]["droplet"]
    assert remaining == [manual]


def test_the_panel_names_the_keys_it_owns(qtbot):
    """Shift+Delete is the one nobody would guess.

    It is the whole per-frame review loop, and a shortcut that only
    exists in a tooltip and the Ctrl+/ dialog is a shortcut most people
    never find. The panel that owns the keys names them, the way the
    drawing panel above names P / R / B / E.
    """
    window = _window(qtbot)
    texts = [
        child.text()
        for child in window.findChildren(QLabel)
        if "Shift+Delete" in child.text()
    ]

    assert texts, "Shift+Delete must be named somewhere in the sidebar"
    # The tracking panel mentions it too, so pick the line that belongs to
    # the annotations panel rather than assuming an order among labels.
    helper = next(
        (text for text in texts if "removes the selection" in text), ""
    )
    assert helper, "the annotations panel must name the keys it owns"
    assert "clears every [SAM 3] mask on this frame" in helper
    assert "keeps the ones you drew" in helper
    assert "Ctrl+Z" in helper


def test_the_clear_button_is_off_when_there_is_nothing_to_clear(qtbot):
    window = _window(qtbot)
    _open_frame(window, "a.png", {"droplet": [_manual()]})
    assert not window.clear_ai_button.isEnabled()

    _open_frame(window, "b.png", {"droplet": [_manual(), _tracked()]})
    assert window.clear_ai_button.isEnabled()
    assert "1 model-generated" in window.clear_ai_button.toolTip()


def test_ctrl_z_undoes_one_frame_and_nothing_else(qtbot):
    """Ctrl+Z means "this frame", including right after tracking.

    Tracking records a history step per frame, so the annotator can step
    through and take back the frames they disagree with, one at a time,
    without a run-wide action reaching frames they already accepted.
    """
    window = _window(qtbot)
    _open_frame(window, "a.png", {})
    window.all_annotations["b.png"] = {}

    for frame_name in ("a.png", "b.png"):
        window._record_tracking_history_once(frame_name)
        window.all_annotations[frame_name] = {"droplet": [_tracked()]}
    window.image_label.annotations = window.all_annotations["a.png"]

    window.undo_annotation_change()

    assert window.all_annotations.get("a.png", {}).get("droplet", []) == []
    assert len(window.all_annotations["b.png"]["droplet"]) == 1


# --- Finding the frames a run touched -------------------------------------


def test_the_ai_filter_shows_only_frames_a_model_wrote_to(qtbot):
    """The review queue after a tracking run.

    Checking a 44-frame run without this means opening all 44 frames to
    find the handful the model actually reached.
    """
    window = _window(qtbot)
    for name in ("a.png", "b.png", "c.png"):
        window.image_list.addItem(name)
    window.all_annotations["a.png"] = {"droplet": [_manual()]}
    window.all_annotations["b.png"] = {"droplet": [_tracked()]}
    window.all_annotations["c.png"] = {}

    window.ai_only_button.setChecked(True)

    hidden = {
        window.image_list.item(i).text(): window.image_list.item(i).isHidden()
        for i in range(window.image_list.count())
    }
    assert hidden == {"a.png": True, "b.png": False, "c.png": True}

    window.ai_only_button.setChecked(False)
    assert not any(
        window.image_list.item(i).isHidden()
        for i in range(window.image_list.count())
    )


def test_todo_counts_a_tracked_mask_as_labelled(qtbot):
    """Pins today's meaning of "labelled" so a change is deliberate.

    ``frame_has_labels`` counts any committed annotation, so a frame
    holding only a SAM 3 mask reads as done — it leaves the Todo queue
    and it counts toward "N of M labeled". That is defensible (a mask is
    there) but it does mean one Track All run can take the progress bar
    to 100% with nothing human-verified. Combining Todo with AI
    therefore yields nothing, which is why the AI filter exists as its
    own toggle rather than as a refinement of Todo.
    """
    window = _window(qtbot)
    for name in ("a.png", "b.png"):
        window.image_list.addItem(name)
    window.all_annotations["a.png"] = {"droplet": [_manual(), _tracked()]}
    window.all_annotations["b.png"] = {"droplet": [_tracked()]}

    assert window.frame_has_labels("b.png")
    assert window.frame_has_ai_labels("b.png")

    window.ai_only_button.setChecked(True)
    window.unlabeled_only_button.setChecked(True)

    assert window.image_list.item(0).isHidden()
    assert window.image_list.item(1).isHidden()


# --- Why a run stopped ----------------------------------------------------


class _ReportingTracker:
    """A tracker that has just finished a run and recorded why it ended."""

    is_initialized = True

    def __init__(self, report):
        self.last_run_report = report

    def unload(self):
        return None

    def close_session(self):
        return None


def test_a_short_run_says_why_it_stopped(qtbot):
    """Nine frames of a sixty-frame clip used to be reported as silence.

    Whether that was the droplet leaving the frame or a gate being too
    strict changes what the annotator should do next, and only one of
    those is worth re-seeding from.
    """
    window = _window(qtbot)
    window.sam3_tracker = _ReportingTracker(
        {
            1: {
                "frames": 9,
                "last_frame": 30,
                "stopped": "5 masks in a row looked wrong (last: shrank to 1/6.0 in one frame)",
            }
        }
    )

    notes = window._tracking_stop_notes({1: ("droplet", "abc")})

    assert len(notes) == 1
    assert "droplet stopped after 9 frame(s)" in notes[0]
    assert "shrank" in notes[0]


def test_a_run_that_reached_the_end_reports_no_stop(qtbot):
    """Running out of clip is not a failure and must not read like one."""
    window = _window(qtbot)
    window.sam3_tracker = _ReportingTracker(
        {1: {"frames": 44, "last_frame": 60, "stopped": ""}}
    )

    assert window._tracking_stop_notes({1: ("droplet", "abc")}) == []


def test_a_full_run_reports_uncertain_masks_as_skipped(qtbot):
    window = _window(qtbot)
    window.sam3_tracker = _ReportingTracker(
        {
            1: {"frames": 40, "processed_through": 63, "skipped": 23,
                "last_issue": "no mask", "stopped": ""}
        }
    )

    notes = window._tracking_stop_notes({1: ("object", "abc")})

    assert len(notes) == 1
    assert "checked through frame 63" in notes[0]
    assert "skipped 23 uncertain mask(s)" in notes[0]
    assert "stopped" not in notes[0]

def test_stop_notes_survive_a_tracker_that_reports_nothing(qtbot):
    """Older trackers, and the guard tests' stand-ins, carry no report."""
    window = _window(qtbot)
    window.sam3_tracker = None
    assert window._tracking_stop_notes({}) == []
