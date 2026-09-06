from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import QLabel

from digitalsreeni_image_annotator.annotator_window import ImageAnnotator


def test_er70s6_controls_and_presets_are_available(qtbot):
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()
    window.auto_save = lambda: None
    window.show_info = lambda *args: None

    action_labels = [action.text() for action in window.findChildren(QAction)]
    assert "Add ER70S-6 &Full Arc Classes" in action_labels
    assert "Add ER70S-6 &CAVITAR Classes" in action_labels
    assert "Show ER70S-6 Labeling &Protocol" in action_labels
    assert window.brightness_slider.minimum() == -100
    assert window.brightness_slider.maximum() == 100
    assert window.contrast_slider.minimum() == -100
    assert window.contrast_slider.maximum() == 100
    assert window.export_format_selector.findText("RGB Semantic Masks") >= 0

    window.add_class("droplet", QColor(0, 255, 255))
    window.add_cavitar_welding_classes()

    assert set(window.class_mapping) == {"droplet", "molten_consumable"}
    assert window.image_label.class_colors["droplet"].getRgb()[:3] == (255, 0, 0)
    assert window.image_label.class_colors["molten_consumable"].getRgb()[:3] == (
        255,
        128,
        0,
    )


def test_sidebar_separates_labeling_from_optional_ai_tools(qtbot):
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()

    assert window.sidebar_tabs.count() == 2
    assert window.sidebar_tabs.tabText(0) == "Label"
    assert window.sidebar_tabs.tabText(1) == "Auto-track"
    assert window.labeling_page.isAncestorOf(window.polygon_button)
    assert window.labeling_page.isAncestorOf(window.annotation_list)
    assert window.labeling_page.isAncestorOf(window.export_button)
    assert window.ai_page.isAncestorOf(window.sam_box_button)
    assert window.ai_page.isAncestorOf(window.sam3_init_btn)
    assert window.ai_page.isAncestorOf(window.dino_model_selector)
    assert (
        window.labeling_scroll.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert (
        window.ai_scroll.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )


def test_sidebar_exposes_common_loading_and_welding_actions(qtbot):
    window = ImageAnnotator()
    qtbot.addWidget(window)
    window.hide()

    assert window.open_video_button.text() == "Open Video Clip..."
    assert window.cavitar_preset_button.text() == "Droplets only"
    assert window.full_arc_preset_button.text() == "Droplets + arc"
    assert window.add_class_button.text() == "Add Custom Class"
    assert window.class_list.maximumHeight() == 120
    assert window.annotation_list.maximumHeight() == 150
    assert "on-screen preview" in window.brightness_slider.toolTip()
    assert "exported mask" in window.brightness_slider.toolTip()
    assert "press Enter" in window.polygon_button.toolTip()
    assert "selected class" in window.eraser_button.toolTip()
    assert window.export_button.property("buttonRole") == "primary"
    assert "reviewed labels" in window.export_button.toolTip()
    assert window.review_package_button.text() == "Create Review Package"
    assert window.review_package_button.property("buttonRole") == "accent"
    assert "review page" in window.review_package_button.toolTip()
    assert window.add_images_button.property("buttonRole") == "primary"
    assert window.delete_button.property("buttonRole") == "danger"
    assert window.clear_all_button.property("buttonRole") == "danger"
    assert window.windowTitle() == "Annotation Studio"
    assert window.image_widget.objectName() == "canvasPanel"
    assert window.image_list_widget.objectName() == "framesPanel"
    assert window.frame_count_label.text() == "0 loaded"

    ai_hint = window.findChild(QLabel, "aiWorkflowHint")
    sam3_scope = window.findChild(QLabel, "sam3ScopeLabel")
    assert ai_hint is not None and "TRACK ACROSS FRAMES" in ai_hint.text()
    assert sam3_scope is not None
    assert "every later loaded frame" in sam3_scope.text()
    assert "numeric filename order" in sam3_scope.text()
