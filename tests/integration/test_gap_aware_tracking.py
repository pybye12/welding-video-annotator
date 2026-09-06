from PyQt6.QtGui import QColor, QImage

from digitalsreeni_image_annotator.annotator_window import ImageAnnotator
from digitalsreeni_image_annotator.video_sequence import FrameSequence


def test_tracking_stops_at_gap_and_moves_to_next_manual_seed(qtbot, tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frame_paths = []
    for index in range(4):
        path = images_dir / f"recording_f{index:06d}.png"
        image = QImage(32, 24, QImage.Format.Format_RGB888)
        image.fill(index * 40)
        assert image.save(str(path))
        frame_paths.append(path)

    class FakeTracker:
        is_initialized = True

        def __init__(self):
            self.limit = None
            self.last_run_report = {}

        def track_polygons(
            self,
            frame_index,
            _polygons,
            _frame_size,
            max_frame_num_to_track=None,
        ):
            self.limit = max_frame_num_to_track
            self.last_run_report = {
                1: {"frames": 2, "last_frame": 2, "stopped": ""}
            }
            square = [[4, 4, 12, 4, 12, 12, 4, 12]]
            return [
                (frame_index, {}),
                (1, {1: square}),
                (2, {1: square}),
                (3, {1: square}),
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
    window.add_class("molten_consumable", QColor("orange"))
    window.is_loading_project = False
    window.frame_sequence = FrameSequence.from_paths(
        images_dir,
        frame_paths,
        [3, 8, 14, 1539],
    )
    first_item = window.image_list.item(0)
    window.image_list.setCurrentItem(first_item)
    window.switch_image(first_item)
    source = {
        "segmentation": [4, 4, 12, 4, 12, 12, 4, 12],
        "category_name": "molten_consumable",
        "category_id": window.class_mapping["molten_consumable"],
    }
    window.image_label.annotations = {"molten_consumable": [source]}
    window.all_annotations[frame_paths[0].name] = window.image_label.annotations
    tracker = FakeTracker()
    window.sam3_tracker = tracker
    window.auto_save = lambda: True
    messages = []
    window.show_info = lambda title, message: messages.append((title, message))

    window.sam3_track_forward(all_objects=True)

    assert tracker.limit == 2
    assert frame_paths[1].name in window.all_annotations
    assert frame_paths[2].name in window.all_annotations
    assert frame_paths[3].name not in window.all_annotations
    assert window.image_list.currentItem().text() == frame_paths[3].name
    assert messages[-1][0] == "Manual Mask Needed"
    assert "1525 source frames away" in messages[-1][1]
