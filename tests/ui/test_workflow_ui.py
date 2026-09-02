"""UI behaviour added for day-to-day labeling work.

These cover the things a lab annotator relies on every shift: knowing how
much of a clip is done, finding the frames still to label, undoing a
mistake, and reaching the tools without the mouse.
"""

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QLineEdit

from digitalsreeni_image_annotator.annotator_window import (
    ImageAnnotator,
    redo_shortcut_sequences,
)


def _window(qtbot):
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.auto_save = lambda: None
    window.show_info = lambda *args: None
    return window


def _load_frames(window, names):
    for name in names:
        window.image_list.addItem(name)
    window.annotations_changed()


def _label(window, name, class_name="droplet"):
    window.all_annotations[name] = {
        class_name: [{"segmentation": [0, 0, 4, 0, 4, 4], "category_name": class_name}]
    }


# --- Progress reporting ---------------------------------------------------


def test_frame_progress_counts_only_committed_labels(qtbot):
    window = _window(qtbot)
    _load_frames(window, ["a.png", "b.png", "c.png", "d.png"])
    _label(window, "a.png")
    # A model proposal awaiting review is not finished work.
    window.all_annotations["b.png"] = {"Temp-droplet": [{"segmentation": [0, 0, 1, 1]}]}

    window.annotations_changed()

    assert window.frame_has_labels("a.png")
    assert not window.frame_has_labels("b.png")
    assert "1 of 4 labeled (25%)" in window.frame_progress_label.text()
    assert "3 to go" in window.frame_progress_label.text()
    assert window.frame_progress.value() == 1
    assert window.frame_progress.maximum() == 4


def test_frame_progress_reports_an_empty_project(qtbot):
    window = _window(qtbot)

    window.refresh_frame_progress()

    assert window.frame_progress_label.text() == "No frames loaded"
    # isVisible() is False for every widget while the window is hidden;
    # isVisibleTo(parent) is what actually reflects the hidden flag.
    assert not window.frame_progress.isVisibleTo(window.image_list_widget)


def test_marker_and_count_cannot_disagree(qtbot):
    """Both come from one pass, so labelling a frame updates both."""
    window = _window(qtbot)
    _load_frames(window, ["a.png", "b.png"])
    before = window.image_list.item(1).data(window._MARKER_STATE_ROLE)

    _label(window, "b.png")
    window.refresh_frame_progress()

    assert window.image_list.item(1).data(window._MARKER_STATE_ROLE) != before
    assert "1 of 2 labeled" in window.frame_progress_label.text()


def test_marker_refresh_skips_rows_that_already_look_right(qtbot):
    """Navigation refreshes on every frame change; unchanged rows are free."""
    window = _window(qtbot)
    _load_frames(window, ["a.png", "b.png"])
    item = window.image_list.item(0)
    touched = []
    item_set_icon = item.setIcon
    item.setIcon = lambda icon: touched.append(1) or item_set_icon(icon)

    window.refresh_frame_progress()

    assert touched == []


def test_switching_theme_recolours_the_markers(qtbot):
    window = _window(qtbot)
    _load_frames(window, ["a.png"])
    light_state = window.image_list.item(0).data(window._MARKER_STATE_ROLE)

    window.toggle_dark_mode()

    assert window.image_list.item(0).data(window._MARKER_STATE_ROLE) != light_state


def test_labeled_frames_are_marked_in_the_list(qtbot):
    window = _window(qtbot)
    _load_frames(window, ["a.png", "b.png"])
    _label(window, "a.png")

    window.annotations_changed()

    done, todo = window.image_list.item(0), window.image_list.item(1)
    assert "Has labels" in done.toolTip()
    assert "No labels yet" in todo.toolTip()
    # The marker is an icon, so item text stays the plain file name that
    # findItems() and switch_image() look up.
    assert done.text() == "a.png"
    assert not done.icon().isNull()


# --- Filtering ------------------------------------------------------------


def test_filter_box_hides_frames_that_do_not_match(qtbot):
    window = _window(qtbot)
    _load_frames(window, ["clip_0001.png", "clip_0002.png", "still.png"])

    window.frame_filter_edit.setText("clip")

    hidden = [window.image_list.item(i).isHidden() for i in range(3)]
    assert hidden == [False, False, True]
    assert "2 of 3 shown" in window.frame_count_label.text()


def test_todo_filter_keeps_the_open_frame_visible(qtbot):
    """Hiding the frame on the canvas would look like it had been lost."""
    window = _window(qtbot)
    _load_frames(window, ["a.png", "b.png"])
    _label(window, "a.png")
    window.image_file_name = "a.png"

    window.unlabeled_only_button.setChecked(True)

    assert not window.image_list.item(0).isHidden()
    assert not window.image_list.item(1).isHidden()


def test_todo_filter_hides_finished_frames(qtbot):
    window = _window(qtbot)
    _load_frames(window, ["a.png", "b.png"])
    _label(window, "a.png")

    window.unlabeled_only_button.setChecked(True)

    assert window.image_list.item(0).isHidden()
    assert not window.image_list.item(1).isHidden()


def test_clearing_the_filter_restores_every_frame(qtbot):
    window = _window(qtbot)
    _load_frames(window, ["a.png", "b.png"])
    window.frame_filter_edit.setText("a")

    window.frame_filter_edit.setText("")

    assert not any(window.image_list.item(i).isHidden() for i in range(2))
    assert window.frame_count_label.text() == "2 loaded"


# --- Keyboard -------------------------------------------------------------


def test_number_keys_select_a_class(qtbot):
    window = _window(qtbot)
    window.add_class("molten_consumable", QColor(255, 128, 0))
    window.add_class("droplet", QColor(255, 0, 0))

    assert window.handle_workflow_key(Qt.Key.Key_2)

    assert window.current_class == "droplet"


def test_number_key_past_the_end_of_the_class_list_is_ignored(qtbot):
    window = _window(qtbot)
    window.add_class("droplet", QColor(255, 0, 0))

    assert not window.handle_workflow_key(Qt.Key.Key_5)


def test_tool_keys_toggle_the_matching_tool(qtbot):
    window = _window(qtbot)
    window.add_class("droplet", QColor(255, 0, 0))
    window.on_class_selected(window.class_list.item(0))

    assert window.handle_workflow_key(Qt.Key.Key_P)
    assert window.image_label.current_tool == "polygon"

    assert window.handle_workflow_key(Qt.Key.Key_B)
    assert window.image_label.current_tool == "paint_brush"

    # Pressing the same key again puts the tool away.
    assert window.handle_workflow_key(Qt.Key.Key_B)
    assert window.image_label.current_tool is None


def test_typing_in_a_text_field_is_left_alone(qtbot):
    """The frame filter has to receive 'p' and '1' as literal characters."""
    window = _window(qtbot)
    window.add_class("droplet", QColor(255, 0, 0))
    field = QLineEdit()
    qtbot.addWidget(field)

    assert not window.handle_workflow_key(Qt.Key.Key_P, field)
    assert not window.handle_workflow_key(Qt.Key.Key_1, field)


def test_workflow_keys_are_ignored_while_sam3_is_tracking(qtbot):
    window = _window(qtbot)
    window.add_class("droplet", QColor(255, 0, 0))
    window._sam3_inference_in_flight = True

    assert not window.handle_workflow_key(Qt.Key.Key_P)


def test_undo_and_redo_are_bound_once_so_the_shortcut_is_not_ambiguous(qtbot):
    """A sequence bound twice makes Qt fire neither binding.

    StandardKey.Redo resolves per platform (Ctrl+Shift+Z on Linux and
    macOS, Ctrl+Y on Windows), so a literal
    [StandardKey.Redo, "Ctrl+Y"] silently binds Ctrl+Y twice on Windows.
    Counting, not membership, is what catches that.
    """
    window = _window(qtbot)

    bound = []
    for action in (window.undo_action, window.redo_action):
        bound.extend(sequence.toString() for sequence in action.shortcuts())
    extra = [
        shortcut.key().toString()
        for shortcut in window.findChildren(type(window._snake_shortcut))
    ]

    assert "Ctrl+Z" in bound
    assert "Ctrl+Y" in bound
    assert "Ctrl+Shift+Z" in bound
    assert len(bound) == len(set(bound)), f"a sequence is bound twice: {bound}"
    assert not set(bound) & set(extra)


def test_redo_sequences_do_not_self_collide_on_windows():
    """The Windows case, exercised on any machine.

    On Windows StandardKey.Redo *is* Ctrl+Y, so a naive
    [StandardKey.Redo, "Ctrl+Y"] binds it twice and Qt fires neither.
    Passing the platform sequence in as an argument is what lets this be
    checked from Linux CI.
    """
    from PyQt6.QtGui import QKeySequence

    windows = [s.toString() for s in redo_shortcut_sequences(QKeySequence("Ctrl+Y"))]
    linux = [
        s.toString() for s in redo_shortcut_sequences(QKeySequence("Ctrl+Shift+Z"))
    ]

    for platform in (windows, linux):
        assert len(platform) == len(set(platform)), platform
        assert set(platform) == {"Ctrl+Y", "Ctrl+Shift+Z"}


def test_undo_and_redo_actually_fire_from_the_keyboard(qtbot):
    """Asserting on shortcut lists is how the ambiguity bug hid before."""
    from PyQt6.QtTest import QTest

    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    window.raise_()
    if not qtbot.waitUntil(window.isActiveWindow, timeout=2000) and not (
        window.isActiveWindow()
    ):  # pragma: no cover - depends on the window manager
        pytest.skip("window manager did not activate the window")
    _load_frames(window, ["a.png"])
    window.image_file_name = "a.png"
    _label(window, "a.png")
    window.record_annotation_history("adding a polygon")
    window.all_annotations["a.png"]["droplet"].append(
        {"segmentation": [9, 9, 5, 5, 1, 1], "category_name": "droplet"}
    )
    window.annotations_changed()

    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(window.all_annotations["a.png"]["droplet"]) == 1

    QTest.keyClick(
        window,
        Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert len(window.all_annotations["a.png"]["droplet"]) == 2


def test_bulk_frame_loading_does_not_refresh_once_per_frame(qtbot):
    """Project open adds frames one at a time; refreshes must coalesce."""
    window = _window(qtbot)
    calls = []
    original = window.refresh_frame_progress
    window.refresh_frame_progress = lambda: calls.append(1) or original()

    with window.suspended_progress_refresh():
        for index in range(50):
            window.image_list.addItem(f"frame_{index:05d}.png")
            window.annotations_changed()

    assert calls == [1], f"expected one refresh, got {len(calls)}"


def test_project_load_defers_refreshes_until_loading_completes(qtbot):
    window = _window(qtbot)
    window.is_loading_project = True
    _load_frames(window, ["a.png", "b.png"])
    _label(window, "a.png")

    # Nothing recomputed while the project is still being assembled.
    assert window.frame_progress_label.text() == "No frames loaded"

    window.is_loading_project = False
    window.annotations_changed()

    assert "1 of 2 labeled" in window.frame_progress_label.text()


def test_history_is_dropped_when_the_workspace_is_cleared(qtbot):
    """Undo must not carry one project's annotations into the next."""
    window = _window(qtbot)
    _load_frames(window, ["frame_0001.png"])
    window.image_file_name = "frame_0001.png"
    _label(window, "frame_0001.png")
    window.record_annotation_history("adding a polygon")
    assert window.annotation_history.can_undo("frame_0001.png")

    window.clear_all(show_messages=False)

    assert not window.annotation_history.can_undo("frame_0001.png")


def test_history_is_dropped_when_a_frame_is_removed(qtbot):
    window = _window(qtbot)
    _load_frames(window, ["a.png", "b.png"])
    window.image_file_name = "a.png"
    _label(window, "a.png")
    window.record_annotation_history("adding a polygon")

    window._remove_image_item(window.image_list.item(0), select_next=False)

    assert not window.annotation_history.can_undo("a.png")


# --- Undo / redo through the window --------------------------------------


def test_undo_restores_the_previous_annotations_for_the_frame(qtbot):
    window = _window(qtbot)
    _load_frames(window, ["a.png"])
    window.image_file_name = "a.png"
    _label(window, "a.png")

    window.record_annotation_history("adding a polygon")
    window.all_annotations["a.png"]["droplet"].append(
        {"segmentation": [9, 9, 5, 5, 1, 1], "category_name": "droplet"}
    )
    window.annotations_changed()
    assert window.undo_button.isEnabled()

    window.undo_annotation_change()

    assert len(window.all_annotations["a.png"]["droplet"]) == 1
    assert window.redo_button.isEnabled()

    window.redo_annotation_change()

    assert len(window.all_annotations["a.png"]["droplet"]) == 2


def test_undo_and_redo_start_disabled(qtbot):
    window = _window(qtbot)

    assert not window.undo_button.isEnabled()
    assert not window.redo_button.isEnabled()
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()


# --- Shell --------------------------------------------------------------


def test_status_bar_reports_tool_class_and_cursor(qtbot):
    window = _window(qtbot)
    window.add_class("droplet", QColor(255, 0, 0))
    window.on_class_selected(window.class_list.item(0))
    window.handle_workflow_key(Qt.Key.Key_B)

    window.update_status_bar((120.6, 44.2))

    assert window.status_tool_label.text() == "Paint brush"
    assert "droplet" in window.status_class_label.text()
    assert "120" in window.status_cursor_label.text()
    assert "44" in window.status_cursor_label.text()
    assert "brush" in window.status_brush_label.text()


def test_canvas_shows_the_placeholder_until_a_frame_is_open(qtbot):
    window = _window(qtbot)

    assert window.canvas_stack.currentIndex() == 0

    window.current_image = object()
    window._update_canvas_placeholder()

    assert window.canvas_stack.currentIndex() == 1


def test_slice_panel_is_hidden_when_there_are_no_slices(qtbot):
    window = _window(qtbot)

    assert not window.slice_list.isVisibleTo(window.image_list_widget)
    assert not window.slice_heading.isVisibleTo(window.image_list_widget)


def test_next_step_hint_follows_the_state_of_the_session(qtbot):
    window = _window(qtbot)

    window.update_next_step_hint()
    assert "Add frames" in window.workflow_hint.text()

    _load_frames(window, ["a.png"])
    window.update_next_step_hint()
    assert "label classes" in window.workflow_hint.text()

    window.add_class("droplet", QColor(255, 0, 0))
    window.on_class_selected(window.class_list.item(0))
    window.image_file_name = "a.png"
    _label(window, "a.png")
    window.update_next_step_hint()
    assert "1 label on this frame" in window.workflow_hint.text()


def test_panels_are_resizable_rather_than_pinned_to_fixed_widths(qtbot):
    """Fixed widths were what clipped the sidebar buttons."""
    window = _window(qtbot)

    assert window.main_splitter.count() == 3
    assert window.main_splitter.indexOf(window.sidebar) == 0
    assert window.main_splitter.indexOf(window.image_widget) == 1
    assert window.main_splitter.indexOf(window.image_list_widget) == 2


# --- Bugs found by clicking through the app ------------------------------


def test_arming_the_brush_before_the_mouse_enters_the_canvas_does_not_crash(qtbot):
    """paintEvent used to raise TypeError and Qt turned that into an abort.

    The guard tested `hasattr(self, "cursor_pos")`, which __init__ always
    satisfies. Clicking "Paint mask" (or pressing B) without having moved
    the mouse over the canvas left cursor_pos None, and the next repaint
    killed the app.
    """
    from PyQt6.QtGui import QPixmap, QPainter

    window = _window(qtbot)
    label = window.image_label
    label.setPixmap(QPixmap(64, 64))
    label.current_tool = "paint_brush"
    label.cursor_pos = None

    surface = QPixmap(64, 64)
    painter = QPainter(surface)
    try:
        label.draw_tool_size_indicator(painter)  # must not raise
    finally:
        painter.end()


def test_a_and_d_move_through_a_still_image_project(qtbot):
    """A / D did nothing without a video sequence, which the UI advertises."""
    window = _window(qtbot)
    switched = []
    window.switch_image = lambda item: switched.append(item.text())
    _load_frames(window, ["a.png", "b.png", "c.png"])
    window.image_list.setCurrentRow(0)

    window.go_to_next_frame()
    assert window.image_list.currentRow() == 1
    window.go_to_next_frame()
    assert window.image_list.currentRow() == 2
    # Does not run off the end.
    window.go_to_next_frame()
    assert window.image_list.currentRow() == 2

    window.go_to_previous_frame()
    assert window.image_list.currentRow() == 1

    # One load per keypress: setCurrentItem already triggers switch_image
    # through currentRowChanged, so stepping must not call it again.
    assert switched.count("c.png") == 1, switched


def test_frame_stepping_skips_rows_hidden_by_the_filter(qtbot):
    """The keys should walk what the annotator can actually see."""
    window = _window(qtbot)
    window.switch_image = lambda item: None
    _load_frames(window, ["a.png", "skip.png", "c.png"])
    window.image_list.item(1).setHidden(True)
    window.image_list.setCurrentRow(0)

    window.go_to_next_frame()

    assert window.image_list.currentRow() == 2


def test_enter_in_the_filter_box_returns_focus_to_the_canvas(qtbot):
    """Otherwise every shortcut looks broken right after filtering."""
    window = _window(qtbot)
    window.switch_image = lambda item: None
    _load_frames(window, ["a.png", "b.png"])
    window.frame_filter_edit.setFocus()

    window.frame_filter_edit.returnPressed.emit()

    assert not window.frame_filter_edit.hasFocus()
