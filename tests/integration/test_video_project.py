import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QDialog, QInputDialog, QMessageBox

from digitalsreeni_image_annotator import annotator_window
from digitalsreeni_image_annotator.annotator_window import ImageAnnotator
from digitalsreeni_image_annotator.video_clip import (
    ExtractedFrame,
    ExtractedVideoClip,
    VideoFrameSelection,
)
from digitalsreeni_image_annotator.video_sequence import FrameSequence


def test_video_session_survives_project_roundtrip(qtbot, tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frame_paths = []
    for source_index in (10, 12):
        frame_path = images_dir / f"clip_frame_{source_index:08d}.jpg"
        image = QImage(32, 24, QImage.Format.Format_RGB888)
        image.fill(source_index)
        assert image.save(str(frame_path))
        frame_paths.append(frame_path)

    project_path = tmp_path / "video_project.iap"
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(project_path)
    window.current_project_dir = str(tmp_path)
    window.is_loading_project = True
    window.add_images_to_list([str(path) for path in frame_paths])
    window.is_loading_project = False
    window.frame_sequence = FrameSequence.from_paths(
        images_dir, frame_paths, [10, 12]
    )
    window.active_video_session_id = "clip-a"
    window.video_sessions = {
        "clip-a": {
            "source_type": "video",
            "source_path": "source.mp4",
            "start_frame": 10,
            "end_frame": 12,
            "stride": 2,
            "frames": [
                {"name": path.name, "source_index": source_index}
                for path, source_index in zip(frame_paths, (10, 12))
            ],
        }
    }

    window.save_project(show_message=False)

    with project_path.open(encoding="utf-8") as project_file:
        project_data = json.load(project_file)
    assert project_data["video_sessions"]["clip-a"]["stride"] == 2
    assert project_data["active_video_session_id"] == "clip-a"

    restored = ImageAnnotator()
    qtbot.addWidget(restored)
    restored.hide()
    restored.current_project_file = str(project_path)
    restored.current_project_dir = str(tmp_path)
    restored.is_loading_project = True
    restored.load_project_data(project_data)
    restored.is_loading_project = False

    assert [frame.source_index for frame in restored.frame_sequence.frames] == [
        10,
        12,
    ]
    assert restored.frame_sequence.name_for_index(1) == frame_paths[1].name


def test_open_video_clip_extracts_copies_and_persists_multiple_sessions(
    qtbot, tmp_path, monkeypatch
):
    video_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (64, 48),
    )
    if not writer.isOpened():
        import pytest

        pytest.skip("OpenCV MJPG writer is unavailable in this environment")
    try:
        for index in range(10):
            writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()

    project_path = tmp_path / "video_project.iap"
    cache_root = tmp_path / "cache"

    class AcceptedClipDialog:
        def __init__(self, _metadata, _parent):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selection(self):
            return VideoFrameSelection(2, 8, 2)

    class TestStandardPaths:
        class StandardLocation:
            CacheLocation = object()

        @staticmethod
        def writableLocation(_location):
            return str(cache_root)

    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(video_path), ""),
    )
    monkeypatch.setattr(annotator_window, "VideoClipDialog", AcceptedClipDialog)
    monkeypatch.setattr(annotator_window, "QStandardPaths", TestStandardPaths)
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "information",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok,
    )

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(project_path)
    window.current_project_dir = str(tmp_path)

    window.open_video_clip()
    first_session_id = window.active_video_session_id
    first_names = [
        frame["name"]
        for frame in window.video_sessions[first_session_id]["frames"]
    ]

    assert len(first_names) == 4
    assert [
        frame["source_index"]
        for frame in window.video_sessions[first_session_id]["frames"]
    ] == [2, 4, 6, 8]
    assert all((tmp_path / "images" / name).is_file() for name in first_names)
    assert "cache_dir" not in window.video_sessions[first_session_id]

    window.open_video_clip()

    assert len(window.video_sessions) == 2
    assert len(window.all_images) == 8
    assert window.active_video_session_id != first_session_id

    ordinary_path = tmp_path / "images" / "ordinary.jpg"
    ordinary = QImage(64, 48, QImage.Format.Format_RGB888)
    ordinary.fill(255)
    assert ordinary.save(str(ordinary_path))
    window.add_images_to_list([str(ordinary_path)], auto_save=False)
    assert window.save_project(show_message=False)

    with project_path.open(encoding="utf-8") as project_file:
        project_data = json.load(project_file)
    assert len(project_data["video_sessions"]) == 2

    outside_cache = tmp_path / "outside-cache"
    outside_cache.mkdir()
    (outside_cache / ".digitalsreeni-video-cache").touch()
    project_data["video_sessions"][first_session_id]["cache_dir"] = str(
        outside_cache
    )

    restored = ImageAnnotator()
    qtbot.addWidget(restored)
    restored.hide()
    restored.current_project_file = str(project_path)
    restored.current_project_dir = str(tmp_path)
    restored.is_loading_project = True
    restored.load_project_data(project_data)
    restored.is_loading_project = False

    assert len(restored.video_sessions) == 2
    assert "cache_dir" not in restored.video_sessions[first_session_id]
    assert outside_cache.is_dir()

    first_item = next(
        restored.image_list.item(index)
        for index in range(restored.image_list.count())
        if restored.image_list.item(index).text() == first_names[0]
    )
    restored.switch_image(first_item)
    assert restored.active_video_session_id == first_session_id
    assert [
        frame.source_index for frame in restored.frame_sequence.frames
    ] == [2, 4, 6, 8]

    workspace = restored._prepare_sam3_frame_workspace()
    tracker_frames = sorted(
        path
        for path in workspace.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    expected_frames = [
        Path(restored.image_paths[name]) for name in first_names
    ]
    assert len(tracker_frames) == len(expected_frames)
    assert [path.read_bytes() for path in tracker_frames] == [
        path.read_bytes() for path in expected_frames
    ]
    assert ordinary_path.read_bytes() not in {
        path.read_bytes() for path in tracker_frames
    }


def test_frame_folder_adopts_preloaded_images_and_uses_unique_sessions(
    qtbot, tmp_path, monkeypatch
):
    frame_folder = tmp_path / "source-frames"
    frame_folder.mkdir()

    def create_frames(start):
        paths = []
        for index in range(start, start + 2):
            path = frame_folder / f"{index:06d}.jpg"
            image = QImage(32, 24, QImage.Format.Format_RGB888)
            image.fill(index * 20)
            assert image.save(str(path))
            paths.append(path)
        return paths

    first_paths = create_frames(0)
    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(frame_folder),
    )
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "critical",
        lambda *_args, **_kwargs: None,
    )

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)
    window.add_images_to_list(
        [str(path) for path in first_paths], auto_save=False
    )
    assert window.save_project(show_message=False)

    window.open_frame_folder()
    first_session_id = window.active_video_session_id

    assert len(window.video_sessions) == 1
    assert len(window.all_images) == 2
    assert set(window._video_session_by_frame) == {
        path.name.casefold() for path in first_paths
    }

    for path in first_paths:
        path.unlink()
    second_paths = create_frames(10)
    window.open_frame_folder()

    assert len(window.video_sessions) == 2
    assert window.active_video_session_id != first_session_id
    assert len(window.all_images) == 4
    assert all(
        (tmp_path / "images" / path.name).is_file() for path in second_paths
    )


@pytest.mark.parametrize("delete_method", ["remove_image", "delete_selected_image"])
def test_deleting_video_frame_invalidates_tracker_and_rebuilds_sequence(
    qtbot, tmp_path, monkeypatch, delete_method
):
    project_images = tmp_path / "images"
    project_images.mkdir()
    frame_paths = []
    for source_index in (10, 11, 12):
        frame_path = project_images / f"clip_{source_index}.jpg"
        image = QImage(32, 24, QImage.Format.Format_RGB888)
        image.fill(source_index)
        assert image.save(str(frame_path))
        frame_paths.append(frame_path)

    cache_root = tmp_path / "cache"

    class TestStandardPaths:
        class StandardLocation:
            CacheLocation = object()

        @staticmethod
        def writableLocation(_location):
            return str(cache_root)

    class FakeTracker:
        def __init__(self):
            self.close_calls = 0

        def close_session(self):
            self.close_calls += 1

        def unload(self):
            pass

    monkeypatch.setattr(annotator_window, "QStandardPaths", TestStandardPaths)
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "information",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok,
    )

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)
    window.is_loading_project = True
    window.add_images_to_list([str(path) for path in frame_paths])
    window.is_loading_project = False
    window.active_video_session_id = "clip-a"
    window.video_sessions = {
        "clip-a": {
            "source_type": "video",
            "frames": [
                {"name": path.name, "source_index": source_index}
                for path, source_index in zip(frame_paths, (10, 11, 12))
            ],
        }
    }
    window._restore_active_frame_sequence()
    workspace = window._prepare_sam3_frame_workspace()
    tracker = FakeTracker()
    window.sam3_tracker = tracker

    first_item = window.image_list.item(0)
    window.image_list.setCurrentItem(first_item)
    window.switch_image(first_item)
    getattr(window, delete_method)()

    assert tracker.close_calls == 1
    assert not workspace.exists()
    assert [frame.source_index for frame in window.frame_sequence.frames] == [
        11,
        12,
    ]
    assert window.frame_sequence.index_for_name(frame_paths[1].name) == 0
    assert [
        frame["name"] for frame in window.video_sessions["clip-a"]["frames"]
    ] == [frame_paths[1].name, frame_paths[2].name]


def test_video_import_rolls_back_when_project_commit_fails(
    qtbot, tmp_path, monkeypatch
):
    video_path = tmp_path / "rollback.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable in this environment")
    try:
        for index in range(4):
            writer.write(np.full((24, 32, 3), index * 40, dtype=np.uint8))
    finally:
        writer.release()

    class AcceptedClipDialog:
        def __init__(self, _metadata, _parent):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selection(self):
            return VideoFrameSelection(0, 3, 1)

    class TestStandardPaths:
        class StandardLocation:
            CacheLocation = object()

        @staticmethod
        def writableLocation(_location):
            return str(tmp_path / "cache")

    errors = []
    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(video_path), ""),
    )
    monkeypatch.setattr(annotator_window, "VideoClipDialog", AcceptedClipDialog)
    monkeypatch.setattr(annotator_window, "QStandardPaths", TestStandardPaths)
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "critical",
        lambda _parent, title, message: errors.append((title, message)),
    )

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)
    real_json_dump = annotator_window.json.dump
    dump_calls = 0

    def fail_during_second_project_write(data, file_object, *args, **kwargs):
        nonlocal dump_calls
        dump_calls += 1
        if dump_calls == 2:
            file_object.write('{"partial":')
            file_object.flush()
            raise OSError("simulated project write failure")
        return real_json_dump(data, file_object, *args, **kwargs)

    monkeypatch.setattr(annotator_window.json, "dump", fail_during_second_project_write)
    window.open_video_clip()

    assert dump_calls == 2
    assert not window.video_sessions
    assert window.image_list.count() == 0
    assert not list((tmp_path / "images").glob("*"))
    with (tmp_path / "project.iap").open(encoding="utf-8") as project_file:
        assert json.load(project_file)["images"] == []
    assert not list(tmp_path.glob(".project.iap.*.tmp"))
    assert errors and "rolled back" in errors[-1][1]


def test_sam3_tracking_preserves_a_non_droplet_class(qtbot, tmp_path):
    project_images = tmp_path / "images"
    project_images.mkdir()
    frame_paths = []
    for index in range(4):
        frame_path = project_images / f"frame_{index}.jpg"
        image = QImage(32, 24, QImage.Format.Format_RGB888)
        image.fill(index * 50)
        assert image.save(str(frame_path))
        frame_paths.append(frame_path)

    class FakeTracker:
        is_initialized = True

        def __init__(self):
            self.polygons = None
            self.window = None
            self.window_enabled_during_tracking = None
            self.canvas_enabled_during_tracking = None

        def track_polygons(
            self,
            frame_index,
            polygons,
            _frame_size,
            max_frame_num_to_track=None,
        ):
            self.polygons = (frame_index, polygons)
            self.max_frame_num_to_track = max_frame_num_to_track
            self.window_enabled_during_tracking = self.window.isEnabled()
            self.canvas_enabled_during_tracking = self.window.image_label.isEnabled()
            return [
                (frame_index, {}),
                (1, {1: [[4, 4, 12, 4, 12, 12, 4, 12]]}),
                (2, {1: [[5, 4, 13, 4, 13, 12, 5, 12]]}),
                (3, {1: [[6, 4, 14, 4, 14, 12, 6, 12]]}),
            ]

        def close_session(self):
            pass

        def unload(self):
            pass

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)
    window.is_loading_project = True
    window.add_images_to_list([str(path) for path in frame_paths])
    window.add_class("molten_consumable", QColor("magenta"))
    window.is_loading_project = False
    window.frame_sequence = FrameSequence.from_paths(
        project_images,
        frame_paths,
        [100, 101, 102, 103],
    )
    window.active_video_session_id = "clip-a"
    window.video_sessions = {
        "clip-a": {
            "source_type": "video",
            "frames": [
                {"name": path.name, "source_index": source_index}
                for path, source_index in zip(frame_paths, (100, 101, 102, 103))
            ],
        }
    }
    first_item = window.image_list.item(0)
    window.image_list.setCurrentItem(first_item)
    window.switch_image(first_item)
    source_annotation = {
        "segmentation": [4, 4, 12, 4, 12, 12, 4, 12],
        "category_name": "molten_consumable",
        "category_id": window.class_mapping["molten_consumable"],
    }
    window.image_label.annotations = {
        "molten_consumable": [source_annotation]
    }
    window.all_annotations[frame_paths[0].name] = window.image_label.annotations
    existing_manual = {
        "segmentation": [18, 4, 24, 4, 24, 10, 18, 10],
        "category_name": "molten_consumable",
        "category_id": window.class_mapping["molten_consumable"],
        "source": "manual",
    }
    window.all_annotations[frame_paths[2].name] = {
        "molten_consumable": [existing_manual]
    }
    tracker = FakeTracker()
    tracker.window = window
    window.sam3_tracker = tracker
    window.auto_save = lambda: True

    window.sam3_track_forward(all_objects=True)

    assert tracker.polygons[0] == 0
    assert tracker.max_frame_num_to_track == 3
    assert tracker.window_enabled_during_tracking is False
    assert tracker.canvas_enabled_during_tracking is False
    assert window.isEnabled()
    tracked = window.all_annotations[frame_paths[1].name]["molten_consumable"]
    assert tracked[0]["category_name"] == "molten_consumable"
    assert "droplet_event_id" not in tracked[0]
    source_id = tracked[0]["sam3_source_id"]
    for frame_path in frame_paths[1:]:
        generated = [
            annotation
            for annotation in window.all_annotations[frame_path.name][
                "molten_consumable"
            ]
            if annotation.get("source") == "sam3_track"
        ]
        assert len(generated) == 1
        assert generated[0]["sam3_source_id"] == source_id
    assert existing_manual in window.all_annotations[frame_paths[2].name][
        "molten_consumable"
    ]

    window.image_list.setCurrentItem(first_item)
    window.switch_image(first_item)
    warnings = []
    window.auto_save = lambda: False
    window.show_warning = lambda title, message: warnings.append((title, message))
    window.sam3_track_forward(all_objects=True)

    assert warnings
    assert warnings[-1][0] == "SAM 3 Tracking"
    assert "could not be saved" in warnings[-1][1]
    assert window.image_list.currentItem() is first_item


def test_late_progress_cancel_discards_a_completed_video_copy(
    qtbot, tmp_path, monkeypatch
):
    video_path = tmp_path / "late-cancel.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable in this environment")
    try:
        for index in range(3):
            writer.write(np.full((24, 32, 3), index * 50, dtype=np.uint8))
    finally:
        writer.release()

    class AcceptedClipDialog:
        def __init__(self, _metadata, _parent):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selection(self):
            return VideoFrameSelection(0, 2, 1)

    class FakeSignal:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

        def emit(self, *args):
            for callback in self.callbacks:
                callback(*args)

    class LateCancelledProgressDialog:
        def __init__(self, *_args, **_kwargs):
            self.canceled = FakeSignal()

        def setWindowTitle(self, _title):
            pass

        def setWindowModality(self, _modality):
            pass

        def setAutoClose(self, _enabled):
            pass

        def setAutoReset(self, _enabled):
            pass

        def setMaximum(self, _maximum):
            pass

        def setValue(self, _value):
            pass

        def accept(self):
            pass

        def exec(self):
            # The worker result already exists, but the dialog receives a
            # genuine cancel event before finalization consumes that result.
            self.canceled.emit()

    class CompletedVideoExtractionThread:
        def __init__(
            self,
            metadata,
            selection,
            _output_dir,
            project_images_dir,
            cache_root,
        ):
            self.metadata = metadata
            self.selection = selection
            self.project_images_dir = Path(project_images_dir)
            self.cache_root = cache_root
            self.progress_changed = FakeSignal()
            self.finished = FakeSignal()
            self.result = None
            self.error = None
            self.cancelled = False
            self.interrupted = False

        def start(self):
            self.project_images_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for position, source_index in enumerate(
                self.selection.source_indices(self.metadata.frame_count)
            ):
                name = f"late_cancel_frame_{position:012d}.jpg"
                path = self.project_images_dir / name
                path.write_bytes(b"completed-frame")
                frames.append(
                    ExtractedFrame(
                        source_index,
                        path,
                        name,
                        source_index / self.metadata.fps,
                    )
                )
            self.result = ExtractedVideoClip(
                self.metadata,
                self.selection,
                self.project_images_dir,
                tuple(frames),
            )

        def requestInterruption(self):
            self.interrupted = True

        def wait(self):
            pass

    class TestStandardPaths:
        class StandardLocation:
            CacheLocation = object()

        @staticmethod
        def writableLocation(_location):
            return str(tmp_path / "cache")

    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(video_path), ""),
    )
    monkeypatch.setattr(annotator_window, "VideoClipDialog", AcceptedClipDialog)
    monkeypatch.setattr(
        annotator_window,
        "QProgressDialog",
        LateCancelledProgressDialog,
    )
    monkeypatch.setattr(
        annotator_window,
        "VideoExtractionThread",
        CompletedVideoExtractionThread,
    )
    monkeypatch.setattr(annotator_window, "QStandardPaths", TestStandardPaths)

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)

    window.open_video_clip()

    assert not window.video_sessions
    assert window.image_list.count() == 0
    assert not list((tmp_path / "images").glob("*"))


def test_frame_folder_registration_rolls_back_after_project_write_failure(
    qtbot, tmp_path, monkeypatch
):
    frame_folder = tmp_path / "source-frames"
    frame_folder.mkdir()
    source_paths = []
    for index in range(2):
        path = frame_folder / f"{index:06d}.jpg"
        image = QImage(32, 24, QImage.Format.Format_RGB888)
        image.fill(index * 70)
        assert image.save(str(path))
        source_paths.append(path)

    errors = []
    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(frame_folder),
    )
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "critical",
        lambda _parent, title, message: errors.append((title, message)),
    )

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)
    real_json_dump = annotator_window.json.dump
    dump_calls = 0

    def fail_during_second_project_write(data, file_object, *args, **kwargs):
        nonlocal dump_calls
        dump_calls += 1
        if dump_calls == 2:
            file_object.write('{"partial":')
            raise OSError("simulated frame-folder commit failure")
        return real_json_dump(data, file_object, *args, **kwargs)

    monkeypatch.setattr(annotator_window.json, "dump", fail_during_second_project_write)

    window.open_frame_folder()

    assert dump_calls == 2
    assert not window.video_sessions
    assert window.image_list.count() == 0
    assert not list((tmp_path / "images").glob("*"))
    assert all(path.is_file() for path in source_paths)
    with (tmp_path / "project.iap").open(encoding="utf-8") as project_file:
        assert json.load(project_file)["images"] == []
    assert errors and "commit failure" in errors[-1][1]


def test_save_project_as_restores_identity_after_atomic_write_failure(
    qtbot, tmp_path, monkeypatch
):
    old_dir = tmp_path / "old"
    old_images = old_dir / "images"
    old_images.mkdir(parents=True)
    image_path = old_images / "frame.jpg"
    image = QImage(32, 24, QImage.Format.Format_RGB888)
    image.fill(100)
    assert image.save(str(image_path))

    old_project = old_dir / "old.iap"
    new_project = tmp_path / "new" / "new.iap"
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(old_project)
    window.current_project_dir = str(old_dir)
    window.add_images_to_list([str(image_path)], auto_save=False)
    assert window.save_project(show_message=False)
    original_paths = window.image_paths.copy()

    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(new_project), ""),
    )
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    errors = []
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "critical",
        lambda _parent, title, message: errors.append((title, message)),
    )

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated Save As failure")

    monkeypatch.setattr(annotator_window, "_write_json_atomically", fail_write)

    window.save_project_as()

    assert window.current_project_file == str(old_project)
    assert window.current_project_dir == str(old_dir)
    assert window.image_paths == original_paths
    assert old_project.is_file()
    assert not new_project.exists()
    assert errors and errors[-1][0] == "Save As Failed"


def test_save_project_as_rejects_a_different_same_named_destination_image(
    qtbot, tmp_path, monkeypatch
):
    old_dir = tmp_path / "old"
    old_images = old_dir / "images"
    old_images.mkdir(parents=True)
    source = old_images / "frame.jpg"
    source.write_bytes(b"source-image")
    old_project = old_dir / "project.iap"

    new_dir = tmp_path / "new"
    new_images = new_dir / "images"
    new_images.mkdir(parents=True)
    occupied = new_images / source.name
    occupied.write_bytes(b"different-image")
    new_project = new_dir / "project.iap"

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(old_project)
    window.current_project_dir = str(old_dir)
    window.image_paths = {source.name: str(source)}
    original_paths = window.image_paths.copy()
    errors = []
    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(new_project), ""),
    )

    def record_error(_parent, title, message):
        assert window.current_project_file == str(old_project)
        assert window.current_project_dir == str(old_dir)
        assert window.image_paths == original_paths
        errors.append((title, message))

    monkeypatch.setattr(annotator_window.QMessageBox, "critical", record_error)

    window.save_project_as()

    assert window.current_project_file == str(old_project)
    assert window.current_project_dir == str(old_dir)
    assert window.image_paths == original_paths
    assert occupied.read_bytes() == b"different-image"
    assert not new_project.exists()
    assert "different image" in window._last_project_save_error
    assert errors and errors[-1][0] == "Save As Failed"
    assert "different image" in errors[-1][1]


def test_save_project_as_adopts_a_verified_identical_destination_image(
    qtbot, tmp_path, monkeypatch
):
    old_dir = tmp_path / "old"
    old_images = old_dir / "images"
    old_images.mkdir(parents=True)
    source = old_images / "frame.jpg"
    source.write_bytes(b"same-image")

    new_dir = tmp_path / "new"
    new_images = new_dir / "images"
    new_images.mkdir(parents=True)
    occupied = new_images / source.name
    occupied.write_bytes(source.read_bytes())
    new_project = new_dir / "project.iap"

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(old_dir / "project.iap")
    window.current_project_dir = str(old_dir)
    window.image_paths = {source.name: str(source)}
    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(new_project), ""),
    )
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "information",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok,
    )

    window.save_project_as()

    assert window.current_project_file == str(new_project)
    assert window.image_paths == {source.name: str(occupied)}
    with new_project.open(encoding="utf-8") as project_file:
        project_data = json.load(project_file)
    assert project_data["image_paths"] == {source.name: str(occupied)}


def test_close_project_preserves_live_state_when_save_fails(
    qtbot, tmp_path, monkeypatch
):
    warnings = []
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)
    window.all_annotations = {
        "frame.jpg": {"Temp-review": [{"segmentation": [0, 0, 1, 0, 1, 1]}]}
    }
    window._last_project_save_error = "simulated failure"
    cleared = []
    window.clear_all = lambda *_args, **_kwargs: cleared.append(True)
    window.save_project = lambda *_args, **_kwargs: False
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    window.close_project()

    assert not cleared
    assert window.current_project_file == str(tmp_path / "project.iap")
    assert "Temp-review" in window.all_annotations["frame.jpg"]
    assert warnings and warnings[-1][0] == "Close Cancelled"


@pytest.mark.parametrize("mask_attribute", ["temp_paint_mask", "temp_eraser_mask"])
def test_close_project_cancel_preserves_uncommitted_canvas_masks(
    qtbot, tmp_path, monkeypatch, mask_attribute
):
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)
    pending_mask = np.ones((24, 32), dtype=np.uint8)
    setattr(window.image_label, mask_attribute, pending_mask)
    cleared = []
    window.clear_all = lambda *_args, **_kwargs: cleared.append(True)
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Cancel,
    )

    window.close_project()

    assert not cleared
    assert getattr(window.image_label, mask_attribute) is pending_mask
    assert window.current_project_file == str(tmp_path / "project.iap")


def test_save_project_catches_directory_creation_failure(
    qtbot, tmp_path, monkeypatch
):
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)

    def fail_makedirs(*_args, **_kwargs):
        raise PermissionError("simulated read-only destination")

    monkeypatch.setattr(annotator_window.os, "makedirs", fail_makedirs)

    assert window.save_project(show_message=False) is False
    assert "read-only" in window._last_project_save_error


def test_save_project_removes_partial_atomic_image_copy(
    qtbot, tmp_path, monkeypatch
):
    source = tmp_path / "source.jpg"
    image = QImage(32, 24, QImage.Format.Format_RGB888)
    image.fill(80)
    assert image.save(str(source))
    project_dir = tmp_path / "project"
    destination = project_dir / "images" / source.name

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(project_dir / "project.iap")
    window.current_project_dir = str(project_dir)
    window.add_images_to_list([str(source)], auto_save=False)
    original_paths = window.image_paths.copy()
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    def fail_copy(_source, target):
        Path(target).write_bytes(b"partial")
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(annotator_window, "_copy_file_atomically", fail_copy)

    assert window.save_project(show_message=False) is False
    assert not destination.exists()
    assert window.image_paths == original_paths


def test_save_project_keeps_committed_paths_when_post_commit_refresh_fails(
    qtbot, tmp_path, monkeypatch
):
    source = tmp_path / "source.jpg"
    image = QImage(32, 24, QImage.Format.Format_RGB888)
    image.fill(80)
    assert image.save(str(source))
    project_dir = tmp_path / "project"
    destination = project_dir / "images" / source.name

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(project_dir / "project.iap")
    window.current_project_dir = str(project_dir)
    window.add_images_to_list([str(source)], auto_save=False)
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    def fail_title_refresh():
        raise RuntimeError("simulated post-commit UI failure")

    monkeypatch.setattr(window, "update_window_title", fail_title_refresh)

    assert window.save_project(show_message=False) is True
    assert destination.is_file()
    assert window.image_paths == {source.name: str(destination)}
    with Path(window.current_project_file).open(encoding="utf-8") as project_file:
        project_data = json.load(project_file)
    assert project_data["image_paths"] == {source.name: str(destination)}


def test_save_project_restores_state_and_retry_rejects_locked_partial_file(
    qtbot, tmp_path, monkeypatch
):
    source = tmp_path / "source.jpg"
    image = QImage(32, 24, QImage.Format.Format_RGB888)
    image.fill(80)
    assert image.save(str(source))
    project_dir = tmp_path / "project"
    destination = project_dir / "images" / source.name

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(project_dir / "project.iap")
    window.current_project_dir = str(project_dir)
    window.add_images_to_list([str(source)], auto_save=False)
    original_paths = window.image_paths.copy()

    copy_calls = []

    def fail_copy(_source, target):
        copy_calls.append(Path(target))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"partial")
        raise OSError("original copy failure")

    original_unlink = Path.unlink

    def fail_destination_unlink(path, *args, **kwargs):
        if path == destination:
            raise PermissionError("simulated locked rollback file")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(annotator_window, "_copy_file_atomically", fail_copy)
    monkeypatch.setattr(Path, "unlink", fail_destination_unlink)

    assert window.save_project(show_message=False) is False
    assert window.image_paths == original_paths
    assert window._last_project_save_error == "original copy failure"
    assert destination.read_bytes() == b"partial"

    assert window.save_project(show_message=False) is False
    assert window.image_paths == original_paths
    assert len(copy_calls) == 1
    assert "different image" in window._last_project_save_error


def test_new_project_failure_does_not_replace_current_state(
    qtbot, tmp_path, monkeypatch
):
    old_project = tmp_path / "old.iap"
    new_project = tmp_path / "new.iap"
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(old_project)
    window.current_project_dir = str(tmp_path)
    window.all_annotations = {"frame.jpg": {"droplet": [{"number": 1}]}}
    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(new_project), ""),
    )
    monkeypatch.setattr(
        annotator_window.QInputDialog,
        "getMultiLineText",
        lambda *_args, **_kwargs: ("new notes", True),
    )
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "critical",
        lambda *_args, **_kwargs: None,
    )

    def fail_write(*_args, **_kwargs):
        raise PermissionError("simulated new-project failure")

    monkeypatch.setattr(annotator_window, "_write_json_atomically", fail_write)

    window.new_project()

    assert window.current_project_file == str(old_project)
    assert window.all_annotations["frame.jpg"]["droplet"][0]["number"] == 1
    assert not new_project.exists()


def test_video_removed_after_range_dialog_reports_controlled_error(
    qtbot, tmp_path, monkeypatch
):
    video_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable in this environment")
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    class RemovingClipDialog:
        def __init__(self, _metadata, _parent):
            pass

        def exec(self):
            video_path.unlink()
            return QDialog.DialogCode.Accepted

        def selection(self):
            return VideoFrameSelection(0, 0, 1)

    errors = []
    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(video_path), ""),
    )
    monkeypatch.setattr(annotator_window, "VideoClipDialog", RemovingClipDialog)
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "critical",
        lambda _parent, title, message: errors.append((title, message)),
    )

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)

    window.open_video_clip()

    assert errors and errors[-1][0] == "Video Error"
    assert not window.video_sessions


def test_video_replaced_after_range_dialog_reports_controlled_error(
    qtbot, tmp_path, monkeypatch
):
    video_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable in this environment")
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    class ReplacingClipDialog:
        def __init__(self, _metadata, _parent):
            pass

        def exec(self):
            with video_path.open("ab") as video_file:
                video_file.write(b"changed-after-probe")
            return QDialog.DialogCode.Accepted

        def selection(self):
            return VideoFrameSelection(0, 0, 1)

    errors = []
    monkeypatch.setattr(
        annotator_window.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(video_path), ""),
    )
    monkeypatch.setattr(annotator_window, "VideoClipDialog", ReplacingClipDialog)
    monkeypatch.setattr(
        annotator_window.QMessageBox,
        "critical",
        lambda _parent, title, message: errors.append((title, message)),
    )

    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.current_project_file = str(tmp_path / "project.iap")
    window.current_project_dir = str(tmp_path)

    window.open_video_clip()

    assert errors and errors[-1][0] == "Video Error"
    assert "changed" in errors[-1][1]
    assert not window.video_sessions
