import copy
import filecmp
import json
import os
import shutil
import tempfile
import traceback
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import shapely
from czifile import CziFile
from PyQt6.QtCore import (
    QEvent,
    QObject,
    QStandardPaths,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPalette,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from tifffile import TiffFile

from .annotation_statistics import show_annotation_statistics, summarize_annotations
from .coco_json_combiner import show_coco_json_combiner
from .dino_phrase_editor import ClassThresholdTable, PhraseEditorPanel
from .dino_utils import DINOUtils
from .dataset_splitter import DatasetSplitterTool
from .default_stylesheet import default_stylesheet
from .dicom_converter import DicomConverter
from .dino_merge_dialog import show_dino_merge_dialog
from .export_formats import (
    export_coco_json,
    export_labeled_images,
    export_pascal_voc_bbox,
    export_pascal_voc_both,
    export_rgb_semantic_masks,
    export_semantic_labels,
    export_yolo_v4,
    export_yolo_v5plus,
)
from .help_window import HelpWindow
from .image_augmenter import show_image_augmenter
from .image_label import ImageLabel
from .image_patcher import show_image_patcher
from .import_formats import (
    import_coco_json,
    import_yolo_v4,
    import_yolo_v5plus,
    process_import_format,
)
from .review_package import export_review_package
from .sam_utils import InferenceBusyError, SAMUtils, _run_sync
from .slice_registration import SliceRegistrationTool
from .snake_game import SnakeGame
from .soft_dark_stylesheet import soft_dark_stylesheet
from .annotation_history import AnnotationHistory
from .shortcuts import ShortcutReferenceDialog
from .theme import tokens_for
from .stack_interpolator import StackInterpolator
from .stack_to_slices import show_stack_to_slices
from .utils import calculate_area, calculate_bbox
from .video_clip import (
    TRACKER_WORKSPACE_MARKER,
    cleanup_managed_video_directory,
    create_tracker_frame_workspace,
    probe_video,
    validate_video_source,
    video_clip_cache_directory,
)
from .video_clip_dialog import VideoClipDialog, VideoExtractionThread
from .video_sequence import FrameSequence
from .welding_defaults import (
    ER70S6_CAVITAR_CLASSES,
    ER70S6_CLASSES,
    ER70S6_PROTOCOL,
)
from .yolo_trainer import LoadPredictionModelDialog, TrainingInfoDialog, YOLOTrainer

warnings.filterwarnings("ignore", category=UserWarning)


def redo_shortcut_sequences(standard_redo):
    """Every sequence that should reach redo, with no duplicates.

    ``standard_redo`` is the platform's own binding, passed in rather
    than read here so a test can supply the Windows value (Ctrl+Y) on any
    machine — which is the case that matters, since binding Ctrl+Y twice
    makes Qt treat it as ambiguous and fire neither.
    """
    sequences = []
    for candidate in (standard_redo, QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")):
        if candidate not in sequences:
            sequences.append(candidate)
    return sequences


def _canonical_image_name(name):
    """Return a portable identity key for a project image filename."""
    return os.path.basename(str(name)).casefold()


def _write_json_atomically(file_path, data):
    """Replace a project JSON file only after a complete durable temp write."""
    destination = Path(file_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(data, temp_file, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, destination)
        if os.name != "nt":
            directory_fd = None
            try:
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                os.fsync(directory_fd)
            except OSError:
                # Some filesystems do not support directory fsync. The atomic
                # replacement still protects against partial JSON contents.
                pass
            finally:
                if directory_fd is not None:
                    os.close(directory_fd)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError as cleanup_error:
                print(f"Could not remove temporary project file {temp_path}: {cleanup_error}")
        raise


def _copy_file_atomically(source, destination):
    """Copy a file through a same-directory temporary path."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        shutil.copy2(source, temp_path)
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError as cleanup_error:
                print(f"Could not remove temporary image file {temp_path}: {cleanup_error}")
        raise


class TrainingThread(QThread):
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, yolo_trainer, epochs, imgsz):
        super().__init__()
        self.yolo_trainer = yolo_trainer
        self.epochs = epochs
        self.imgsz = imgsz

    def run(self):
        try:
            results = self.yolo_trainer.train_model(
                epochs=self.epochs, imgsz=self.imgsz
            )
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit(str(e))


class DimensionDialog(QDialog):
    def __init__(self, shape, file_name, parent=None, default_dimensions=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Dimensions")
        layout = QVBoxLayout(self)

        # Add file name label
        file_name_label = QLabel(f"File: {file_name}")
        file_name_label.setWordWrap(True)
        layout.addWidget(file_name_label)

        # Add dimension assignment widgets
        dim_widget = QWidget()
        dim_layout = QGridLayout(dim_widget)
        self.combos = []
        self.shape = shape
        dimensions = ["T", "Z", "C", "S", "H", "W"]
        for i, dim in enumerate(shape):
            dim_layout.addWidget(QLabel(f"Dimension {i} (size {dim}):"), i, 0)
            combo = QComboBox()
            combo.addItems(dimensions)
            if default_dimensions and i < len(default_dimensions):
                combo.setCurrentText(default_dimensions[i])
            dim_layout.addWidget(combo, i, 1)
            self.combos.append(combo)
        layout.addWidget(dim_widget)

        self.button = QPushButton("OK")
        self.button.clicked.connect(self.accept)
        layout.addWidget(self.button)

        self.setMinimumWidth(300)

    def get_dimensions(self):
        return [combo.currentText() for combo in self.combos]


class _DINOReviewEventFilter(QObject):
    """Application-wide event filter that lets Enter / Escape accept or
    reject pending DINO temp_annotations regardless of which widget has
    focus. Without this, clicking a slice/image entry in a list moves
    focus there and Enter is consumed by the list's itemActivated
    handler before it can reach ImageLabel.keyPressEvent.

    Suppressed when a modal dialog is active or focus is on a text-input
    widget so we don't break dialog default-button behaviour or
    in-cell editing.
    """

    def __init__(self, main_window: "ImageAnnotator"):
        super().__init__(main_window)
        self.main_window = main_window

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.KeyPress:
            return False
        key = event.key()
        if key not in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            return False
        app = QApplication.instance()
        if app is None or app.activeModalWidget() is not None:
            return False
        focused = app.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit)):
            return False
        if (
            self.main_window._sam3_inference_in_flight
            or not self.main_window.isEnabled()
        ):
            return True
        temp = self.main_window.image_label.temp_annotations
        if not temp or not any(a.get("source") == "dino" for a in temp):
            return False
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.main_window.accept_dino_results()
        else:
            self.main_window.reject_dino_results()
        return True


class _WorkflowKeyFilter(QObject):
    """Application-wide handler for the unmodified tool and class keys.

    P / R / B / E pick a drawing tool and 1-9 pick a label class. These
    have to work while focus sits on the frame list, the class list or a
    button — none of which forward a plain key press to the canvas — which
    is the same reason A / D / C and F2 are registered globally.

    Suppressed while a modal dialog is up, while the caret is in a text
    field (so a filter box still accepts the literal character), and while
    SAM 3 is mid-inference.
    """

    def __init__(self, main_window: "ImageAnnotator"):
        super().__init__(main_window)
        self.main_window = main_window

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.KeyPress:
            return False
        if event.modifiers() not in (
            Qt.KeyboardModifier.NoModifier,
            Qt.KeyboardModifier.KeypadModifier,
        ):
            return False
        app = QApplication.instance()
        if app is None or app.activeModalWidget() is not None:
            return False
        if not self.main_window.isEnabled():
            return False
        # Only when the main window is the one being typed into. Several
        # child windows are shown non-modally (help, the Snake easter egg,
        # the YOLO training dialog); without this an "E" aimed at one of
        # them would silently arm the eraser on the canvas behind it.
        if not self.main_window.isActiveWindow():
            return False
        # `obj` is the widget the key press was delivered to, i.e. the one
        # with focus. Testing it directly is more reliable than asking the
        # application for its focus widget, which is null whenever the
        # window is not active — including under an offscreen/test server.
        return self.main_window.handle_workflow_key(event.key(), obj)


class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.is_loading_project = False
        self.backup_project_path = None

        self.setWindowTitle("Annotation Studio")
        self.setGeometry(100, 100, 1400, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.create_menu_bar()

        # Initialize image_label early
        self.image_label = ImageLabel()

        self.image_label.sam_box_active = False
        self.image_label.sam_points_active = False
        self.image_label.sam_positive_points = []
        self.image_label.sam_negative_points = []
        self.image_label.set_main_window(self)

        # Initialize attributes
        self.current_image = None
        self.current_class = None
        self.image_file_name = ""
        self.all_annotations = {}
        self.all_images = []
        self.image_paths = {}
        self.loaded_json = None
        self.class_mapping = {}
        self.editing_mode = False
        self.current_slice = None
        self.slices = []
        self.current_stack = None
        self.image_dimensions = {}
        self.image_slices = {}
        self.image_shapes = {}

        
        # For paint brush and eraser
        self.paint_brush_size = 10
        self.eraser_size = 10
        # Initialize SAM utils
        self.current_sam_model = None
        self.sam_utils = SAMUtils()

        # Initialize DINO utils for LLM-assisted detection.
        # Phrases and thresholds are owned by the widgets (PhraseEditorPanel
        # and ClassThresholdTable); the project save/load reads/writes them
        # through the widget APIs, not through a shadow dict on self.
        self.dino_utils = DINOUtils()
        self.dino_model_loaded = False
        self.dino_custom_model_path = None

        # Debounce timer for SAM points: wait 1s after last click before inference
        self.sam_inference_timer = QTimer(self)
        self.sam_inference_timer.setSingleShot(True)
        self.sam_inference_timer.timeout.connect(self.apply_sam_prediction)

        # Guards against re-entrant `apply_sam_prediction` calls — the
        # debounce timer can fire while an earlier inference is still
        # pumping inside _run_sync. See apply_sam_prediction().
        self._sam_inference_in_flight = False

        # Create sam_magic_wand_button
        self.sam_magic_wand_button = QPushButton("Magic Wand")
        self.sam_magic_wand_button.setCheckable(True)
        self.sam_magic_wand_button.setEnabled(False)  # Initially disable the button

        # Initialize tool group
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)

        # Font size control
        self.font_sizes = {
            "Small": 8,
            "Medium": 10,
            "Large": 12,
            "XL": 14,
            "XXL": 16,
        }  # Also, add the options in create_menu_bar method
        self.current_font_size = "Medium"

        # Dark mode control. Default on — matches the look most users
        # expect from a 2025-era desktop annotation tool; toggle with
        # Settings → Toggle Dark Mode (Ctrl+D).
        self.dark_mode = True

        # Default annotations sorting
        self.current_sort_method = "class"  # Default sorting method

        # DINO batch review state. Initialised eagerly here so the
        # consumers don't each carry a `hasattr` check (one forgotten
        # check would crash with AttributeError).
        self.dino_batch_results: dict[str, list] = {}

        # Per-frame undo/redo over annotation snapshots. See
        # annotation_history.py for why this snapshots rather than
        # recording inverse commands.
        self.annotation_history = AnnotationHistory()

        # Progress-refresh coalescing. Bulk paths add frames one at a
        # time, and a refresh walks the whole list, so without this the
        # cost of opening a project is quadratic in frame count.
        self._progress_refresh_depth = 0
        self._progress_refresh_pending = False
        self._hidden_frame_count = 0

        # Setup UI components
        self.setup_ui()

        # Apply theme and font (this includes stylesheet and font size application)
        self.apply_theme_and_font()

        # Connect sam_magic_wand_button
        self.sam_magic_wand_button.clicked.connect(self.toggle_tool)

        self.class_list.itemChanged.connect(self.toggle_class_visibility)

        # YOLO Trainer
        self.yolo_trainer = None
        self.setup_yolo_menu()

        self.frame_sequence = None
        self.video_sessions = {}
        self._video_session_by_frame = {}
        self.active_video_session_id = None
        self._sam3_frame_workspace = None
        self._sam3_frame_workspace_root = None
        self.sam3_tracker = None
        self._sam3_inference_in_flight = False

        # F2 → Snake game (Easter egg). Registered as a global QShortcut
        # so it fires regardless of which widget has focus — putting it
        # in keyPressEvent didn't work because QTableWidget (DINO
        # threshold table) and other focusable children consume F2
        # before it bubbles up to the main window.
        self._snake_shortcut = QShortcut(QKeySequence("F2"), self)
        self._snake_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._snake_shortcut.activated.connect(self.launch_snake_game)

        self._video_shortcuts = []
        for key, callback in (
            ("A", self.go_to_previous_frame),
            ("D", self.go_to_next_frame),
            ("C", self.copy_selected_annotation_to_next_frame),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(
                lambda callback=callback: self._trigger_video_shortcut(callback)
            )
            self._video_shortcuts.append(shortcut)

        # Enter/Escape for DINO temp_annotations need to work even when
        # focus is on slice_list / image_list / a button — none of which
        # forward the key to ImageLabel.keyPressEvent. Application-wide
        # event filter intercepts these keys but only when DINO results
        # are pending review, and skips modal dialogs + text inputs.
        self._dino_review_filter = _DINOReviewEventFilter(self)
        QApplication.instance().installEventFilter(self._dino_review_filter)

        self._install_workflow_shortcuts()
        self.refresh_frame_progress()
        self.update_status_bar()

        self.showMaximized()

    def setup_ui(self):
        # Initialize the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(0)

        # The three columns live in a splitter so the annotator can widen
        # the frame list for long file names, or collapse the sidebar and
        # give the whole window to the image. Fixed min/max widths alone
        # meant the sidebar clipped its own buttons on smaller displays.
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setObjectName("mainSplitter")
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)
        self.layout.addWidget(self.main_splitter)

        # Initialize tool group
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)

        # Setup UI components
        self.setup_sidebar()
        self.setup_image_area()
        self.setup_image_list()
        self.setup_slice_list()
        self.setup_status_bar()

        # Canvas takes the slack; the two side panels keep their widths.
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([380, 1000, 260])

        self.update_ui_for_current_tool()

    def update_window_title(self):
        base_title = "Annotation Studio"
        if hasattr(self, "current_project_file"):
            project_name = os.path.basename(self.current_project_file)
            project_name = os.path.splitext(project_name)[
                0
            ]  # Remove the file extension
            self.setWindowTitle(f"{base_title} - {project_name}")
        else:
            self.setWindowTitle(base_title)
        self.update_project_identity()

    def new_project(self):
        if self._sam3_inference_in_flight:
            self.show_warning("SAM 3 Tracker", "Wait for SAM 3 tracking to finish.")
            return
        project_file, _ = QFileDialog.getSaveFileName(
            self, "Create New Project", "", "Image Annotator Project (*.iap)"
        )
        if project_file:
            # Ensure the file has the correct extension
            if not project_file.lower().endswith(".iap"):
                project_file += ".iap"

            # Prompt for initial project notes
            notes, ok = QInputDialog.getMultiLineText(
                self, "Project Notes", "Enter initial project notes:"
            )
            notes = notes if ok else ""
            creation_date = datetime.now().isoformat()
            project_dir = os.path.dirname(project_file)
            try:
                os.makedirs(os.path.join(project_dir, "images"), exist_ok=True)
                _write_json_atomically(
                    project_file,
                    {
                        "classes": [],
                        "images": [],
                        "image_paths": {},
                        "notes": notes,
                        "creation_date": creation_date,
                        "last_modified": creation_date,
                    },
                )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "New Project Failed",
                    f"The new project could not be created:\n{exc}",
                )
                return

            # Replace the live state only after the empty project exists.
            self.clear_all(new_project=True, show_messages=False)
            self.current_project_file = project_file
            self.current_project_dir = project_dir
            self.project_notes = notes
            self.project_creation_date = creation_date

            # Keep only this message
            self.show_info(
                "New Project", f"New project created at {self.current_project_file}"
            )
            self.initialize_yolo_trainer()
            self.update_window_title()

    def show_project_search(self):
        if self._sam3_inference_in_flight:
            self.show_warning("SAM 3 Tracker", "Wait for SAM 3 tracking to finish.")
            return
        from .project_search import show_project_search

        show_project_search(self)

    def open_project(self):
        if self._sam3_inference_in_flight:
            self.show_warning("SAM 3 Tracker", "Wait for SAM 3 tracking to finish.")
            return
        print("open_project method called")  # Debug print
        self.remove_all_temp_annotations()  # Remove temp annotations from the previous project
        project_file, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Image Annotator Project (*.iap)"
        )
        print(f"Selected project file: {project_file}")  # Debug print
        if project_file:
            try:
                self.backup_project_before_open(project_file)
                self.open_specific_project(project_file)
            except Exception as e:
                self.restore_project_from_backup()
                QMessageBox.critical(
                    self,
                    "Error",
                    f"An error occurred while opening the project: {str(e)}\n"
                    f"The project file has been restored from backup.",
                )
        else:
            print("No project file selected")  # Debug print

    def backup_project_before_open(self, project_file):
        """Create a backup of the project file before opening it."""
        import os
        import shutil

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(project_file), ".project_backups")
        os.makedirs(backup_dir, exist_ok=True)

        self.backup_project_path = os.path.join(
            backup_dir, f"{os.path.basename(project_file)}.{timestamp}.backup"
        )
        shutil.copy2(project_file, self.backup_project_path)

    def restore_project_from_backup(self):
        """Restore the project file from its backup if available."""
        if self.backup_project_path and os.path.exists(self.backup_project_path):
            try:
                shutil.copy2(self.backup_project_path, self.current_project_file)
                print(f"Project restored from backup: {self.backup_project_path}")
            except Exception as e:
                print(f"Failed to restore from backup: {str(e)}")

    def open_specific_project(self, project_file):
        if self._sam3_inference_in_flight:
            self.show_warning("SAM 3 Tracker", "Wait for SAM 3 tracking to finish.")
            return
        print(f"Opening specific project: {project_file}")  # Debug print
        if os.path.exists(project_file):
            try:
                self.is_loading_project = True  # Set loading flag

                with open(project_file, "r") as f:
                    project_data = json.load(f)

                self.clear_all(show_messages=False)
                self.current_project_file = project_file
                self.current_project_dir = os.path.dirname(project_file)

                # Load project notes and metadata
                self.project_notes = project_data.get("notes", "")
                self.project_creation_date = project_data.get("creation_date", "")
                self.last_modified = project_data.get("last_modified", "")

                # Parse dates
                if self.project_creation_date:
                    self.project_creation_date = datetime.fromisoformat(
                        self.project_creation_date
                    ).strftime("%Y-%m-%d %H:%M:%S")
                if self.last_modified:
                    self.last_modified = datetime.fromisoformat(
                        self.last_modified
                    ).strftime("%Y-%m-%d %H:%M:%S")

                # Load all data without triggering auto-saves
                self.load_project_data(project_data)

                # Now save once after everything is loaded
                self.is_loading_project = False  # Clear loading flag
                # Reveal the phrase editor if any classes exist — the
                # per-class selectRow inside add_class was skipped during
                # load (see add_class). Selecting row 0 is enough; the
                # user can switch rows freely afterwards.
                if self.dino_class_table.rowCount() > 0:
                    self.dino_class_table.selectRow(0)
                normalized = self.save_project(show_message=False)
                if not normalized:
                    detail = getattr(self, "_last_project_save_error", None)
                    QMessageBox.warning(
                        self,
                        "Project Opened Read-Only",
                        "The project was opened, but its normalized state could "
                        f"not be saved{f': {detail}' if detail else '.'}",
                    )

                self.initialize_yolo_trainer()
                self.update_window_title()
                # Progress refreshes are suppressed while
                # is_loading_project is set (a project can hold thousands
                # of frames, each added one at a time). Bring the frame
                # markers, counters and filter up to date in one pass now
                # that the whole project is in memory.
                self.annotations_changed()

                print(f"Project opened successfully: {project_file}")
                QMessageBox.information(
                    self,
                    "Project Opened",
                    f"Project opened successfully: {os.path.basename(project_file)}",
                )

            except Exception as e:
                self.is_loading_project = False  # Make sure to clear flag on error
                # Whatever state the half-loaded project left behind, the
                # panels must describe it rather than the previous one.
                self.annotations_changed()
                raise e
        else:
            print(f"Project file not found: {project_file}")
            QMessageBox.critical(
                self, "Error", f"Project file not found: {project_file}"
            )

    def load_project_data(self, project_data):
        """Load project data without triggering auto-saves."""
        self._reset_sam3_video_state()
        self._clear_video_sessions(clean_clip_caches=True)
        loaded_sessions = copy.deepcopy(project_data.get("video_sessions", {}))
        self.video_sessions = (
            loaded_sessions if isinstance(loaded_sessions, dict) else {}
        )
        legacy_session = project_data.get("video_session")
        if legacy_session and not self.video_sessions:
            self.video_sessions["legacy"] = copy.deepcopy(legacy_session)
        self.video_sessions = {
            session_id: session
            for session_id, session in self.video_sessions.items()
            if isinstance(session, dict)
        }
        for session in self.video_sessions.values():
            # Cache directories are process-local runtime state. Never trust a
            # path supplied by a project file as a deletion target.
            session.pop("cache_dir", None)
        self.active_video_session_id = project_data.get(
            "active_video_session_id"
        )
        if self.active_video_session_id not in self.video_sessions:
            self.active_video_session_id = next(iter(self.video_sessions), None)
        self._rebuild_video_session_frame_index()
        if self.active_video_session_id not in self.video_sessions:
            self.active_video_session_id = next(iter(self.video_sessions), None)

        # Load classes
        self.class_mapping.clear()
        self.image_label.class_colors.clear()
        for class_info in project_data.get("classes", []):
            self.add_class(class_info["name"], QColor(class_info["color"]))

        # Load images
        self.all_images = project_data.get("images", [])
        self.image_paths = project_data.get("image_paths", {})

        # Load all annotations first
        self.all_annotations.clear()
        # Undo history belongs to the project that is being replaced.
        self.annotation_history.clear()
        for image_info in project_data["images"]:
            if image_info.get("is_multi_slice", False):
                for slice_info in image_info.get("slices", []):
                    self.all_annotations[slice_info["name"]] = slice_info["annotations"]
            else:
                self.all_annotations[image_info["file_name"]] = image_info.get(
                    "annotations", {}
                )

        # Handle missing images
        missing_images = []
        for image_info in project_data["images"]:
            image_path = os.path.join(
                self.current_project_dir, "images", image_info["file_name"]
            )

            if not os.path.exists(image_path):
                missing_images.append(image_info["file_name"])
                continue

            # Update image_paths
            self.image_paths[image_info["file_name"]] = image_path

            if image_info.get("is_multi_slice", False):
                dimensions = image_info.get("dimensions", [])
                shape = image_info.get("shape", [])
                self.load_multi_slice_image(image_path, dimensions, shape)
            else:
                self.add_images_to_list([image_path])

        # Restore DINO configuration if present. Classes were created above
        # via add_class(), so the threshold table already has rows for them;
        # we just push the saved values into the existing widgets. Filter
        # out any keys that reference classes no longer in the project
        # (hand-edited .iap, class deleted between sessions) so stale state
        # doesn't get round-tripped on the next save.
        dino_cfg = project_data.get("dino_config", {})
        valid_classes = set(self.class_mapping.keys())

        phrases = dino_cfg.get("phrases", {})
        if phrases:
            kept = {k: v for k, v in phrases.items() if k in valid_classes}
            for orphan in phrases.keys() - kept.keys():
                print(f"  Skipped saved DINO phrases for unknown class "
                      f"'{orphan}' — class is not in the current project.")
            self.dino_phrase_panel.set_phrases(kept)

        for cls_name, thr in dino_cfg.get("thresholds", {}).items():
            ok = self.dino_class_table.set_thresholds(
                cls_name,
                thr.get("box", 0.25),
                thr.get("txt", 0.25),
                thr.get("nms", 0.50),
            )
            if not ok:
                print(f"  Skipped saved DINO thresholds for unknown class "
                      f"'{cls_name}' — class is not in the current project.")

        # Update UI
        self.update_ui()

        # Handle missing images if any
        if missing_images:
            self.handle_missing_images(missing_images)

        self._restore_active_frame_sequence()

        # Select the first image if available
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
            first_item = self.image_list.item(0)
            if first_item:
                self.switch_image(first_item)

        # Select the first class if available
        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self.on_class_selected()

    def handle_missing_images(self, missing_images):
        message = "The following images have annotations but were not found in the project directory:\n\n"
        message += "\n".join(missing_images[:10])  # Show first 10 missing images
        if len(missing_images) > 10:
            message += f"\n... and {len(missing_images) - 10} more."
        message += "\n\nWould you like to locate these images now?"

        reply = QMessageBox.question(
            self,
            "Missing Images",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.load_missing_images(missing_images)
        else:
            self.remove_missing_images(missing_images)

    def remove_missing_images(self, missing_images):
        for image_name in missing_images:
            # Remove from all_images
            self.all_images = [
                img for img in self.all_images if img["file_name"] != image_name
            ]

            # Remove from image_paths
            self.image_paths.pop(image_name, None)

            # Remove from all_annotations, and the undo history with it —
            # history is keyed by bare frame name and undo auto-saves, so
            # a stale entry can write these annotations back later.
            self.all_annotations.pop(image_name, None)
            self.annotation_history.forget(image_name)

            # If it's a multi-slice image, remove all related slices
            base_name = os.path.splitext(image_name)[0]
            if base_name in self.image_slices:
                for slice_name, _ in self.image_slices[base_name]:
                    self.all_annotations.pop(slice_name, None)
                    self.annotation_history.forget(slice_name)
                del self.image_slices[base_name]

        self._prune_video_sessions_to_project_images()
        self._restore_active_frame_sequence()
        self.update_ui()
        QMessageBox.information(
            self,
            "Images Removed",
            f"{len(missing_images)} missing images and their annotations have been removed from the project.",
        )

    def prompt_load_missing_images(self, missing_images):
        message = "The following images have annotations but were not found in the project directory:\n\n"
        message += "\n".join(missing_images[:10])  # Show first 10 missing images
        if len(missing_images) > 10:
            message += f"\n... and {len(missing_images) - 10} more."
        message += "\n\nWould you like to locate these images now?"

        reply = QMessageBox.question(
            self,
            "Load Missing Images",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.load_missing_images(missing_images)

    def load_missing_images(self, missing_images):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Missing Images",
            "",
            "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)",
        )
        if files:
            images_loaded = 0
            for file_path in files:
                file_name = os.path.basename(file_path)
                if file_name in missing_images:
                    dst_path = os.path.join(
                        self.current_project_dir, "images", file_name
                    )
                    shutil.copy2(file_path, dst_path)
                    self.image_paths[file_name] = dst_path

                    # Add the image to all_images if it's not already there
                    if not any(
                        img["file_name"] == file_name for img in self.all_images
                    ):
                        self.all_images.append(
                            {
                                "file_name": file_name,
                                "height": 0,
                                "width": 0,
                                "id": len(self.all_images) + 1,
                                "is_multi_slice": False,
                            }
                        )
                    images_loaded += 1
                    missing_images.remove(file_name)

            self.update_image_list()
            if images_loaded > 0:
                self.image_list.setCurrentRow(0)  # Select the first image
                self.switch_image(self.image_list.item(0))  # Display the first image
            QMessageBox.information(
                self,
                "Images Loaded",
                f"Successfully copied and loaded {images_loaded} out of {len(files)} selected images.",
            )

            # If there are still missing images, prompt again
            if missing_images:
                self.prompt_load_missing_images(missing_images)

    def update_image_list(self):
        self.image_list.clear()
        for image_info in self.all_images:
            self.image_list.addItem(image_info["file_name"])
        # Rebuilt rows carry no marker and no hidden state until this runs.
        self.annotations_changed()

    def select_class(self, index):
        if 0 <= index < self.class_list.count():
            item = self.class_list.item(index)
            self.class_list.setCurrentItem(item)
            self.current_class = item.text()
            print(f"Selected class: {self.current_class}")
        else:
            print("Invalid class index")

    def close_project(self):
        if self._sam3_inference_in_flight:
            self.show_warning("SAM 3 Tracker", "Wait for SAM 3 tracking to finish.")
            return
        if not self.image_label.check_unsaved_changes():
            return
        if hasattr(self, "current_project_file"):
            reply = QMessageBox.question(
                self,
                "Close Project",
                "Do you want to save the current project before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Yes:
                if not self.save_project(show_message=False):
                    detail = getattr(self, "_last_project_save_error", None)
                    QMessageBox.warning(
                        self,
                        "Close Cancelled",
                        "The project could not be saved, so it was not closed"
                        f"{f': {detail}' if detail else '.'}",
                    )
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                return  # User cancelled the operation

        # Clear all data
        self.clear_all(new_project=True, show_messages=False)

        # Reset project-related attributes
        if hasattr(self, "current_project_file"):
            del self.current_project_file
        if hasattr(self, "current_project_dir"):
            del self.current_project_dir

        # Update the window title
        self.update_window_title()

    def delete_selected_class(self):
        selected_items = self.class_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "No Selection", "Please select a class to delete."
            )
            return

        class_name = selected_items[0].text()
        reply = QMessageBox.question(
            self,
            "Delete Class",
            f"Are you sure you want to delete the class '{class_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_class(
                class_name
            )  # Sreeni note: Implement this method to handle class deletion

    def check_missing_images(self):
        missing_images = [
            img["file_name"]
            for img in self.all_images
            if img["file_name"] not in self.image_paths
            or not os.path.exists(self.image_paths[img["file_name"]])
        ]
        if missing_images:
            self.prompt_load_missing_images(missing_images)

    def convert_to_serializable(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {
                key: self.convert_to_serializable(value) for key, value in obj.items()
            }
        else:
            return obj

    def save_project(self, show_message=True):
        if self.is_loading_project:
            return False
        self._last_project_save_error = None

        had_project_file = hasattr(self, "current_project_file")
        had_project_dir = hasattr(self, "current_project_dir")
        original_project_file = getattr(self, "current_project_file", None)
        original_project_dir = getattr(self, "current_project_dir", "")
        original_image_paths = self.image_paths.copy()
        created_destinations = []

        try:
            saved = self._save_project_impl(
                show_message,
                created_destinations,
            )
        except Exception as exc:
            self._last_project_save_error = str(exc)
            print(f"Failed to save project: {exc}")
            if show_message:
                QMessageBox.critical(
                    self,
                    "Project Save Failed",
                    f"The project could not be saved:\n{exc}",
                )
            saved = False

        if not saved:
            try:
                for destination in reversed(created_destinations):
                    try:
                        Path(destination).unlink()
                    except OSError as cleanup_error:
                        print(
                            "Could not remove an incomplete project image "
                            f"{destination}: {cleanup_error}"
                        )
            finally:
                self.image_paths = original_image_paths
                if had_project_file:
                    self.current_project_file = original_project_file
                elif hasattr(self, "current_project_file"):
                    del self.current_project_file
                if had_project_dir:
                    self.current_project_dir = original_project_dir
                elif hasattr(self, "current_project_dir"):
                    del self.current_project_dir

        # The status bar is the only place that reports whether what is on
        # screen has reached disk, so it has to hear about failures too.
        self.set_saved_state(
            saved,
            "" if saved else "Save failed — check the project folder",
        )
        return saved

    def _save_project_impl(self, show_message, created_destinations):

        if not hasattr(self, "current_project_file") or not self.current_project_file:
            self.current_project_file, _ = QFileDialog.getSaveFileName(
                self, "Save Project", "", "Image Annotator Project (*.iap)"
            )
            if not self.current_project_file:
                return False

        self.current_project_dir = os.path.dirname(self.current_project_file)

        # Check if images are in the correct directory structure
        images_dir = os.path.join(self.current_project_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        images_to_copy = []
        candidate_image_paths = {}
        destination_names = {}
        for file_name, src_path in self.image_paths.items():
            dst_path = os.path.join(images_dir, file_name)
            destination_key = _canonical_image_name(file_name)
            previous_name = destination_names.get(destination_key)
            if previous_name is not None and previous_name != file_name:
                raise ValueError(
                    "The project contains image names that collide on this "
                    f"filesystem: {previous_name} and {file_name}."
                )
            destination_names[destination_key] = file_name
            candidate_image_paths[file_name] = dst_path

            if os.path.abspath(src_path) == os.path.abspath(dst_path):
                continue
            if os.path.exists(dst_path):
                try:
                    same_image = filecmp.cmp(src_path, dst_path, shallow=False)
                except OSError as exc:
                    raise OSError(
                        f"Could not verify existing project image {dst_path}: {exc}"
                    ) from exc
                if not same_image:
                    raise FileExistsError(
                        "The destination already contains a different image named "
                        f"{file_name}. Choose another project folder or remove the "
                        "conflicting file."
                    )
                continue
            images_to_copy.append((file_name, src_path, dst_path))

        if images_to_copy:
            reply = QMessageBox.question(
                self,
                "Image Directory Structure",
                f"The project structure requires all images to be in an 'images' subdirectory. "
                f"{len(images_to_copy)} images need to be copied to the correct location. "
                f"Do you want to copy these images?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                for file_name, src_path, dst_path in images_to_copy:
                    created_destinations.append(Path(dst_path))
                    _copy_file_atomically(src_path, dst_path)
            else:
                self._last_project_save_error = "Project image copy was declined."
                QMessageBox.warning(
                    self,
                    "Save Cancelled",
                    "Project cannot be saved without the correct directory structure.",
                )
                return False

        # Prepare image data
        images_data = []
        for image_info in self.all_images:
            file_name = image_info["file_name"]
            image_data = {
                "file_name": file_name,
                "width": image_info["width"],
                "height": image_info["height"],
                "is_multi_slice": image_info["is_multi_slice"],
            }

            if image_data["is_multi_slice"]:
                base_name_without_ext = os.path.splitext(file_name)[0]
                image_data["slices"] = []
                for slice_name, _ in self.image_slices.get(base_name_without_ext, []):
                    slice_data = {
                        "name": slice_name,
                        "annotations": self.convert_to_serializable(
                            self.all_annotations.get(slice_name, {})
                        ),
                    }
                    image_data["slices"].append(slice_data)

                image_data["dimensions"] = self.convert_to_serializable(
                    self.image_dimensions.get(base_name_without_ext, [])
                )
                image_data["shape"] = self.convert_to_serializable(
                    self.image_shapes.get(base_name_without_ext, [])
                )
            else:
                image_data["annotations"] = {}
                for class_name, annotations in self.all_annotations.get(
                    file_name, {}
                ).items():
                    if class_name.startswith("Temp-"):
                        continue
                    image_data["annotations"][class_name] = [
                        ann.copy() for ann in annotations
                    ]

            images_data.append(image_data)

        # Create project data
        project_data = {
            "classes": [
                {"name": name, "color": color.name()}
                for name, color in self.image_label.class_colors.items()
                if not name.startswith("Temp-")
            ],
            "images": images_data,
            "image_paths": {
                k: v
                for k, v in candidate_image_paths.items()
                if os.path.exists(v)
            },
            "notes": getattr(self, "project_notes", ""),
            "creation_date": getattr(
                self, "project_creation_date", datetime.now().isoformat()
            ),
            "last_modified": datetime.now().isoformat(),
        }

        # Persist DINO configuration by snapshotting the widgets that own it.
        dino_cfg = {
            "phrases": self.dino_phrase_panel.get_all_phrases(),
            "thresholds": self.dino_class_table.get_thresholds_dict(),
        }
        if dino_cfg["phrases"] or dino_cfg["thresholds"]:
            project_data["dino_config"] = dino_cfg

        video_sessions = self._video_sessions_for_save()
        if video_sessions:
            project_data["video_sessions"] = video_sessions
            if self.active_video_session_id in video_sessions:
                project_data["active_video_session_id"] = (
                    self.active_video_session_id
                )

        # Replace the live project only after a complete same-directory write.
        _write_json_atomically(
            self.current_project_file,
            self.convert_to_serializable(project_data),
        )
        self.image_paths = candidate_image_paths

        # The disk and required in-memory commit are complete. Failures in
        # optional UI/cache refresh must not be reported as a failed commit.
        try:
            if show_message:
                self.show_info(
                    "Project Saved", f"Project saved to {self.current_project_file}"
                )
            self.update_window_title()
            self._cleanup_video_clip_caches()
            self._restore_active_frame_sequence()
        except Exception as exc:
            print(f"Project saved, but post-save refresh failed: {exc}")
        return True

    def save_project_as(self):
        new_project_file, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "Image Annotator Project (*.iap)"
        )
        if new_project_file:
            # Ensure the file has the correct extension
            if not new_project_file.lower().endswith(".iap"):
                new_project_file += ".iap"

            # Store the original project identity. A failed Save As must not
            # leave the live window pointing at an uncommitted destination.
            had_project_file = hasattr(self, "current_project_file")
            had_project_dir = hasattr(self, "current_project_dir")
            original_project_file = getattr(self, "current_project_file", None)
            original_project_dir = getattr(self, "current_project_dir", "")
            original_image_paths = self.image_paths.copy()
            committed = False
            failure_detail = None

            try:
                self.current_project_file = new_project_file
                self.current_project_dir = os.path.dirname(new_project_file)
                committed = self.save_project(show_message=False)
                if not committed:
                    failure_detail = getattr(self, "_last_project_save_error", None)
            finally:
                if not committed:
                    if had_project_file:
                        self.current_project_file = original_project_file
                    elif hasattr(self, "current_project_file"):
                        del self.current_project_file
                    if had_project_dir:
                        self.current_project_dir = original_project_dir
                    elif hasattr(self, "current_project_dir"):
                        del self.current_project_dir
                    self.image_paths = original_image_paths

            if not committed:
                QMessageBox.critical(
                    self,
                    "Save As Failed",
                    "The project could not be saved to the selected location"
                    f"{f': {failure_detail}' if failure_detail else '.'}",
                )
                return

            # Update the window title
            self.update_window_title()

            # Show a success message
            QMessageBox.information(
                self, "Project Saved As", f"Project saved as:\n{new_project_file}"
            )

            # If this was originally a new unsaved project, update the original project file
            if original_project_file is None:
                self.current_project_file = new_project_file

    def auto_save(self):
        if self.is_loading_project:
            return False

        if not hasattr(self, "current_project_file"):
            reply = QMessageBox.question(
                self,
                "No Project",
                "You need to save the project before auto-saving. Would you like to save now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                saved = self.save_project()
                if saved:
                    print("Project auto-saved.")
                return saved
            else:
                return False

        if hasattr(self, "current_project_file"):
            saved = self.save_project(show_message=False)
            if saved:
                print("Project auto-saved.")
            return saved
        return False

    def show_project_details(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        from .annotation_statistics import AnnotationStatisticsDialog
        from .project_details import ProjectDetailsDialog

        # Generate annotation statistics
        stats_dialog = AnnotationStatisticsDialog(self)
        stats_dialog.generate_statistics(self.all_annotations)

        dialog = ProjectDetailsDialog(self, stats_dialog)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.were_changes_made():
                previous_notes = getattr(self, "project_notes", "")
                self.project_notes = dialog.get_notes()
                if self.save_project(show_message=False):
                    QMessageBox.information(
                        self, "Project Details", "Project details have been updated."
                    )
                else:
                    self.project_notes = previous_notes
                    detail = getattr(self, "_last_project_save_error", None)
                    QMessageBox.warning(
                        self,
                        "Project Details Not Saved",
                        "The project details were not updated"
                        f"{f': {detail}' if detail else '.'}",
                    )
            else:
                print("No changes made to project details.")

    def load_multi_slice_image(self, image_path, dimensions=None, shape=None):

        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        print(f"Loading multi-slice image: {image_path}")
        print(f"Base name: {base_name}")

        if dimensions and shape:
            print(f"Using stored dimensions: {dimensions}")
            print(f"Using stored shape: {shape}")
            self.image_dimensions[base_name] = dimensions
            self.image_shapes[base_name] = shape
            if image_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(image_path, dimensions, shape)
            elif image_path.lower().endswith(".czi"):
                self.load_czi(image_path, dimensions, shape)
        else:
            print("No stored dimensions or shape, loading as new image")
            if image_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(image_path)
            elif image_path.lower().endswith(".czi"):
                self.load_czi(image_path)

        print(f"Loaded multi-slice image: {file_name}")
        print(f"Dimensions: {self.image_dimensions.get(base_name, 'Not found')}")
        print(f"Shape: {self.image_shapes.get(base_name, 'Not found')}")
        print(f"Number of slices: {len(self.slices)}")

        if self.slices:
            self.current_image = self.slices[0][1]
            self.current_slice = self.slices[0][0]

            self.update_slice_list()
            self.slice_list.setCurrentRow(0)
            self.activate_slice(self.current_slice)
            print(f"Activated first slice: {self.current_slice}")
        else:
            print("No slices were loaded")
            self.current_image = None
            self.current_slice = None

        self.update_slice_list()
        self.image_label.update()

    # print(f"Loaded slices: {[slice_name for slice_name, _ in self.slices]}")

    def activate_sam_magic_wand(self):
        # Uncheck all other tools
        for button in self.tool_group.buttons():
            if button != self.sam_magic_wand_button:
                button.setChecked(False)

        # Set the current tool
        self.image_label.current_tool = "sam_magic_wand"
        self.image_label.sam_magic_wand_active = True
        self.image_label.setCursor(Qt.CursorShape.CrossCursor)

        # Update UI based on the current tool
        self.update_ui_for_current_tool()

        # If a class is not selected, select the first one (if available)
        if self.current_class is None and self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self.current_class = self.class_list.currentItem().text()
        elif self.class_list.count() == 0:
            QMessageBox.warning(
                self,
                "No Class Selected",
                "Please add a class before using annotation tools.",
            )
            self.sam_magic_wand_button.setChecked(False)
            self.deactivate_sam_magic_wand()

    def deactivate_sam_magic_wand(self):
        self.image_label.current_tool = None
        self.image_label.sam_magic_wand_active = False
        self.sam_magic_wand_button.setChecked(False)
        self.sam_magic_wand_button.setEnabled(False)  # Disable the button
        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

        # Clear any SAM-related temporary data
        self.image_label.sam_bbox = None
        self.image_label.drawing_sam_bbox = False
        self.image_label.temp_sam_prediction = None

        # Update UI based on the current tool
        self.update_ui_for_current_tool()

    def toggle_sam_assisted(self):
        if not self.current_sam_model:
            QMessageBox.warning(
                self,
                "No SAM Model Selected",
                "Please pick a SAM model before using the SAM-Assisted tool.",
            )
            self.sam_magic_wand_button.setChecked(False)
            return

        if self.sam_magic_wand_button.isChecked():
            self.activate_sam_magic_wand()
        else:
            self.deactivate_sam_magic_wand()

        self.image_label.clear_temp_sam_prediction()  # Clear temporary prediction

    def toggle_sam_magic_wand(self):
        if self.sam_magic_wand_button.isChecked():
            if self.current_class is None:
                QMessageBox.warning(
                    self,
                    "No Class Selected",
                    "Please select a class before using SAM2 Magic Wand.",
                )
                self.sam_magic_wand_button.setChecked(False)
                return
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.image_label.sam_magic_wand_active = True
        else:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.image_label.sam_magic_wand_active = False
            self.image_label.sam_bbox = None

        self.image_label.clear_temp_sam_prediction()  # Clear temporary prediction

    def schedule_sam_prediction(self):
        """Restart the debounce timer; inference fires 1s after last click."""
        self.sam_inference_timer.stop()
        self.sam_inference_timer.start(1000)

    def apply_sam_prediction(self):
        # Re-entry guard: if a previous SAM call is still in flight, the
        # event-loop pump inside _run_sync can deliver this timer fire
        # before the first call returns. Bail and rely on the user
        # clicking again (which restarts the debounce) to issue a fresh
        # inference with the up-to-date point set.
        if self._sam_inference_in_flight:
            return
        self._sam_inference_in_flight = True
        try:
            try:
                if self.image_label.current_tool == "sam_box":
                    if self.image_label.sam_bbox is None:
                        print("SAM bbox is None")
                        return
                    x1, y1, x2, y2 = self.image_label.sam_bbox
                    bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    prediction = self.sam_utils.apply_sam_prediction(self.current_image, bbox)
                    self.image_label.sam_bbox = None
                elif self.image_label.current_tool == "sam_points":
                    # Always use all points!
                    pos_points = self.image_label.sam_positive_points
                    neg_points = self.image_label.sam_negative_points
                    print(
                        f"[SAM-POINTS] Predicting with {len(pos_points)} positive points: {pos_points} "
                        f"and {len(neg_points)} negative points: {neg_points}"
                    )
                    if not pos_points:
                        print("No positive points for SAM-points")
                        return
                    prediction = self.sam_utils.apply_sam_points(
                        self.current_image,
                        pos_points,
                        neg_points,
                    )
                else:
                    return
            except InferenceBusyError:
                # Re-entry safety net from sam_utils. The call-site flag
                # above should catch this first, but if a different
                # caller drives inference concurrently we just skip —
                # the user keeps interacting; their next click will
                # restart the debounce.
                return
            except Exception as exc:
                traceback.print_exc()
                QMessageBox.critical(
                    self,
                    "SAM Error",
                    f"SAM inference failed:\n\n{exc}\n\n"
                    "See the log for details.",
                )
                return

            if prediction:
                temp_annotation = {
                    "segmentation": prediction["segmentation"],
                    "category_id": self.class_mapping[self.current_class],
                    "category_name": self.current_class,
                    "score": prediction["score"],
                }
                self.image_label.temp_sam_prediction = temp_annotation
                self.image_label.update()
            elif prediction is None:
                QMessageBox.information(
                    self,
                    "SAM",
                    "No mask matches the given constraints. "
                    "Try adjusting the box or point positions."
                )
            else:
                print("Failed to generate prediction")

            # Only clear box/points for box mode, not for points mode!
            if self.image_label.current_tool == "sam_box":
                self.image_label.sam_bbox = None
                self.image_label.update()
        finally:
            self._sam_inference_in_flight = False

    def accept_sam_prediction(self):
        if self.image_label.temp_sam_prediction:
            self.record_annotation_history("accepting a SAM mask")
            new_annotation = self.image_label.temp_sam_prediction
            self.image_label.annotations.setdefault(
                new_annotation["category_name"], []
            ).append(new_annotation)
            self.add_annotation_to_list(new_annotation)
            self.save_current_annotations()
            self.update_slice_list_colors()
            self.image_label.temp_sam_prediction = None
            # --- Clear points after accepting
            self.image_label.sam_positive_points = []
            self.image_label.sam_negative_points = []
            self.image_label.update()
            print("SAM prediction accepted, points cleared, and added to annotations.")

    def setup_slice_list(self):
        self.slice_list = QListWidget()
        self.slice_list.setObjectName("sliceList")
        self.slice_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.slice_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.slice_list.itemClicked.connect(self.switch_slice)
        self.slice_list.setMaximumHeight(190)
        self.slice_heading = QLabel("SLICES")
        self.slice_heading.setProperty("class", "eyebrow")
        # Inserted above "Clear Workspace" rather than appended, so the
        # destructive button stays at the bottom of the panel.
        insert_at = self.image_list_layout.count() - 1
        self.image_list_layout.insertWidget(insert_at, self.slice_heading)
        self.image_list_layout.insertWidget(insert_at + 1, self.slice_list)
        self._update_slice_panel_visibility()

    def open_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Images",
            "",
            "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)",
        )
        if file_names:
            self.image_list.clear()
            self.image_paths.clear()
            self.all_images.clear()
            self.slice_list.clear()
            self.slices.clear()
            self.current_stack = None
            self.current_slice = None
            self.add_images_to_list(file_names)

    def convert_to_8bit_rgb(self, image_array):
        if image_array.ndim == 2:
            # Grayscale image
            image_8bit = self.normalize_array(image_array)
            return np.stack((image_8bit,) * 3, axis=-1)
        elif image_array.ndim == 3:
            if image_array.shape[2] == 3:
                # Already RGB, just normalize
                return self.normalize_array(image_array)
            elif image_array.shape[2] > 3:
                # Multi-channel image, use first three channels
                rgb_array = image_array[:, :, :3]
                return self.normalize_array(rgb_array)

        raise ValueError(f"Unsupported image shape: {image_array.shape}")

    def add_images_to_list(self, file_names, known_size=None, auto_save=True):
        with self.suspended_progress_refresh():
            return self._add_images_to_list(file_names, known_size, auto_save)

    def _add_images_to_list(self, file_names, known_size, auto_save):
        first_added_item = None
        added_names = []
        existing_names = {
            _canonical_image_name(name): name for name in self.image_paths
        }
        for file_name in file_names:
            base_name = os.path.basename(file_name)
            image_key = _canonical_image_name(base_name)
            if image_key not in existing_names:
                image_info = {
                    "file_name": base_name,
                    "height": 0,
                    "width": 0,
                    "id": len(self.all_images) + 1,
                    "is_multi_slice": False,
                }

                # Detect multi-slice images and set dimensions
                if file_name.lower().endswith((".tif", ".tiff", ".czi")):
                    self.load_multi_slice_image(file_name)
                    base_name_without_ext = os.path.splitext(base_name)[0]
                    if (
                        base_name_without_ext in self.image_slices
                        and self.image_slices[base_name_without_ext]
                    ):
                        first_slice_name, first_slice = self.image_slices[
                            base_name_without_ext
                        ][0]
                        image_info["height"] = first_slice.height()
                        image_info["width"] = first_slice.width()
                        image_info["is_multi_slice"] = True
                        image_info["dimensions"] = self.image_dimensions.get(
                            base_name_without_ext, []
                        )
                        image_info["shape"] = self.image_shapes.get(
                            base_name_without_ext, []
                        )
                else:
                    # For regular images
                    if known_size:
                        image_info["width"], image_info["height"] = known_size
                    else:
                        image = QImage(file_name)
                        image_info["height"] = image.height()
                        image_info["width"] = image.width()

                self.all_images.append(image_info)
                added_names.append(base_name)
                item = QListWidgetItem(base_name)
                self.image_list.addItem(item)
                if first_added_item is None:
                    first_added_item = item

                # Update image_paths with the original file path
                self.image_paths[base_name] = file_name
                existing_names[image_key] = base_name

        if first_added_item:
            self.image_list.setCurrentItem(first_added_item)
            self.switch_image(first_added_item)

        if auto_save and not self.is_loading_project:
            self.auto_save()
        return added_names

    def update_all_images(self, new_image_info):
        for info in new_image_info:
            if not any(
                img["file_name"] == info["file_name"] for img in self.all_images
            ):
                self.all_images.append(info)

    def closeEvent(self, event):
        if self._sam3_inference_in_flight:
            event.ignore()
            return
        if not self.image_label.check_unsaved_changes():
            event.ignore()
            return
        event.accept()

        if (
            self.image_label.temp_paint_mask is not None
            or self.image_label.temp_eraser_mask is not None
        ):
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save them before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Yes:
                if self.image_label.temp_paint_mask is not None:
                    self.image_label.commit_paint_annotation()
                if self.image_label.temp_eraser_mask is not None:
                    self.image_label.commit_eraser_changes()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        self._reset_sam3_video_state(unload=True)
        self._clear_video_sessions(clean_clip_caches=True)
        event.accept()

    def switch_slice(self, item):
        if self._sam3_inference_in_flight:
            return
        if item is None:
            return
        if not self.image_label.check_unsaved_changes():
            return

        # Check for unsaved changes
        if (
            self.image_label.temp_paint_mask is not None
            or self.image_label.temp_eraser_mask is not None
        ):
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Yes:
                if self.image_label.temp_paint_mask is not None:
                    self.image_label.commit_paint_annotation()
                if self.image_label.temp_eraser_mask is not None:
                    self.image_label.commit_eraser_changes()
            elif reply == QMessageBox.StandardButton.Cancel:
                return
            else:
                self.image_label.discard_paint_annotation()
                self.image_label.discard_eraser_changes()

        self.save_current_annotations()
        self.image_label.clear_temp_sam_prediction()

        slice_name = item.text()
        for name, qimage in self.slices:
            if name == slice_name:
                self.current_image = qimage
                self.current_slice = name
                self.display_image()
                self.load_image_annotations()
                self.update_annotation_list()
                self.clear_highlighted_annotation()
                self.image_label.highlighted_annotations.clear()  # Add this line
                self.image_label.reset_annotation_state()
                self.image_label.clear_current_annotation()
                self.update_image_info()
                break

        # Ensure the UI is updated
        self.image_label.update()
        self.update_slice_list_colors()

        # Reset zoom level to default (1.0)
        self.set_zoom(1.0)

        # Sync DINO temp_annotations to the new slice (carry over masks
        # from the previous slice was a reported bug).
        self._refresh_dino_temp_for_current()

    def switch_image(self, item):
        if self._sam3_inference_in_flight:
            return
        if item is None:
            return
        if not self.image_label.check_unsaved_changes():
            return

        # Store the current item before checking temp annotations
        current_item = self.image_list.currentItem()

        if not self.check_temp_annotations():
            # If the user chooses not to discard temp annotations, revert the selection
            self.image_list.setCurrentItem(current_item)
            return

        self.save_current_annotations()
        self.image_label.clear_temp_sam_prediction()
        self.image_label.exit_editing_mode()

        file_name = item.text()
        print(f"\nSwitching to image: {file_name}")

        image_info = next(
            (img for img in self.all_images if img["file_name"] == file_name), None
        )

        if image_info:
            self.image_file_name = file_name
            self._activate_video_session_for_frame(file_name)
            image_path = self.image_paths.get(file_name)

            if not image_path:
                image_path = os.path.join(self.current_project_dir, "images", file_name)

            if image_path and os.path.exists(image_path):
                if image_info.get("is_multi_slice", False):
                    base_name = os.path.splitext(file_name)[0]
                    if base_name in self.image_slices:
                        self.slices = self.image_slices[base_name]
                        if self.slices:
                            self.current_image = self.slices[0][1]
                            self.current_slice = self.slices[0][0]
                            self.update_slice_list()
                            self.activate_slice(self.current_slice)
                    else:
                        self.load_multi_slice_image(
                            image_path,
                            image_info.get("dimensions"),
                            image_info.get("shape"),
                        )
                else:
                    self.load_regular_image(image_path)
                    self.display_image()
                    self.clear_slice_list()

                self.load_image_annotations()
                self.update_annotation_list()
                self.clear_highlighted_annotation()

                self.image_label.highlighted_annotations.clear()
                self.image_label.update()
                self.image_label.reset_annotation_state()
                self.image_label.clear_current_annotation()
                self.update_image_info()

                self.adjust_zoom_to_fit()
            else:
                self.current_image = None
                self.image_label.clear()
                self.load_image_annotations()
                self.update_annotation_list()
                self.update_image_info()

            self.image_list.setCurrentItem(item)
            self.image_label.update()
            self.update_slice_list_colors()
        else:
            self.current_image = None
            self.current_slice = None
            self.image_label.clear()
            self.update_image_info()
            self.clear_slice_list()

        # Sync DINO temp_annotations to the new image (mask carry-over
        # bug from single-image review and batch review).
        self._refresh_dino_temp_for_current()

    def adjust_zoom_to_fit(self):
        if not self.current_image:
            return

        # Get the dimensions of the image and the display area
        image_width = self.current_image.width()
        image_height = self.current_image.height()
        display_width = self.scroll_area.viewport().width()
        display_height = self.scroll_area.viewport().height()

        # Calculate and apply the zoom factor to fit the longest side
        zoom_factor = min(display_width / image_width, display_height / image_height)
        self.set_zoom(zoom_factor)

    def activate_current_slice(self):
        if self.current_slice:
            # Ensure the current slice is selected in the slice list
            items = self.slice_list.findItems(self.current_slice, Qt.MatchFlag.MatchExactly)
            if items:
                self.slice_list.setCurrentItem(items[0])

            # Load annotations for the current slice
            self.load_image_annotations()

            # Update the image label
            self.image_label.update()

            # Update the annotation list
            self.update_annotation_list()

    def load_image(self, image_path):
        extension = os.path.splitext(image_path)[1].lower()
        if extension in [".tif", ".tiff"]:
            self.load_tiff(image_path)
        elif extension == ".czi":
            self.load_czi(image_path)
        else:
            self.load_regular_image(image_path)

    def load_tiff(
        self, image_path, dimensions=None, shape=None, force_dimension_dialog=False
    ):
        print(f"Loading TIFF file: {image_path}")
        axes_hint = None
        with TiffFile(image_path) as tif:
            print(f"TIFF tags: {tif.pages[0].tags}")

            # Try to access metadata if available
            try:
                metadata = tif.pages[0].tags["ImageDescription"].value
                print(f"TIFF metadata: {metadata}")
            except KeyError:
                print("No ImageDescription metadata found")

            # Try to read axis labels from the tifffile series. ImageJ /
            # OME-TIFF stores axes like "TZCYX" — we can prefill the
            # dimension dialog with the right labels so the user just
            # clicks OK instead of guessing per axis. Map tifffile's
            # axes vocabulary (T,Z,C,S,Y,X) to the app's (T,Z,C,S,H,W).
            try:
                series_axes = tif.series[0].axes if tif.series else None
                if series_axes:
                    axis_map = {
                        "T": "T", "Z": "Z", "C": "C", "S": "S",
                        "Y": "H", "X": "W",
                    }
                    mapped = [axis_map.get(a) for a in series_axes]
                    if all(a is not None for a in mapped):
                        axes_hint = mapped
                        print(f"TIFF series axes: {series_axes} → dimension hint: {axes_hint}")
                    else:
                        unknown = [a for a in series_axes if axis_map.get(a) is None]
                        print(f"TIFF series axes had unknown labels {unknown}, no hint applied")
            except Exception as e:
                print(f"Could not read TIFF series axes: {e}")

            # Check if it's a multi-page TIFF
            if len(tif.pages) > 1:
                print(f"Multi-page TIFF detected. Number of pages: {len(tif.pages)}")
                # Read all pages into a 3D array
                image_array = tif.asarray()
            else:
                print("Single-page TIFF detected.")
                image_array = tif.pages[0].asarray()

            print(f"Image array shape: {image_array.shape}")
            print(f"Image array dtype: {image_array.dtype}")
            print(f"Image min: {image_array.min()}, max: {image_array.max()}")

        if dimensions and shape and not force_dimension_dialog:
            # Use stored dimensions and shape
            print(f"Using stored dimensions: {dimensions}")
            print(f"Using stored shape: {shape}")
            image_array = image_array.reshape(shape)
        else:
            # Process as before for new images or when forcing dimension dialog
            print("Processing as new image or forcing dimension dialog.")
            dimensions = None

        self.process_multidimensional_image(
            image_array, image_path, dimensions, force_dimension_dialog,
            axes_hint=axes_hint,
        )

    def load_czi(
        self, image_path, dimensions=None, shape=None, force_dimension_dialog=False
    ):
        print(f"Loading CZI file: {image_path}")
        with CziFile(image_path) as czi:
            image_array = czi.asarray()
            print(f"CZI array shape: {image_array.shape}")
            print(f"CZI array dtype: {image_array.dtype}")
            print(f"CZI array min: {image_array.min()}, max: {image_array.max()}")

        if dimensions and shape and not force_dimension_dialog:
            # Use stored dimensions and shape
            print(f"Using stored dimensions: {dimensions}")
            print(f"Using stored shape: {shape}")
            image_array = image_array.reshape(shape)
        else:
            # Process as before for new images or when forcing dimension dialog
            print("Processing as new image or forcing dimension dialog.")
            dimensions = None

        self.process_multidimensional_image(
            image_array, image_path, dimensions, force_dimension_dialog
        )

    def load_regular_image(self, image_path):
        self.current_image = QImage(image_path)
        self.slices = []
        self.slice_list.clear()
        self.current_slice = None

    def process_multidimensional_image(
        self, image_array, image_path, dimensions=None,
        force_dimension_dialog=False, axes_hint=None,
    ):
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        print(f"Processing file: {file_name}")
        print(f"Image array shape: {image_array.shape}")
        print(f"Image array dtype: {image_array.dtype}")

        if dimensions is None or force_dimension_dialog:
            if image_array.ndim > 2:
                # Prefer the loader's metadata-derived hint (e.g. ImageJ
                # TIFF axes='TZCYX'). Fall back to a hand-crafted default
                # that covers ndim 3..6 so a user clicking OK without
                # tweaking the combos gets a sensible result. The earlier
                # `default_dimensions[-ndim:]` slice silently degraded for
                # ndim≥5: one axis ended up unset and inherited the combo
                # box's first item ("T"), producing 2560 wrong slices for
                # a 5D TZCYX file.
                if axes_hint and len(axes_hint) == image_array.ndim:
                    default_dimensions = list(axes_hint)
                    print(f"Applying axes hint as default dims: {default_dimensions}")
                else:
                    if axes_hint and len(axes_hint) != image_array.ndim:
                        print(
                            f"Ignoring axes hint (length {len(axes_hint)} "
                            f"vs ndim {image_array.ndim})"
                        )
                    ndim_defaults = {
                        3: ["Z", "H", "W"],
                        4: ["T", "Z", "H", "W"],
                        5: ["T", "Z", "C", "H", "W"],
                        6: ["T", "Z", "C", "S", "H", "W"],
                    }
                    # ndim ≥ 7 falls into the generic case: pad with
                    # "T" at the front so H / W are still the last two
                    # axes — that way "click OK" still produces a
                    # sensible 2D slice even on exotic inputs.
                    default_dimensions = ndim_defaults.get(
                        image_array.ndim,
                        ["T"] * max(0, image_array.ndim - 2) + ["H", "W"],
                    )

                # Show a progress dialog
                progress = QProgressDialog(
                    "Assigning dimensions...", "Cancel", 0, 100, self
                )
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(10)
                QApplication.processEvents()

                while True:
                    dialog = DimensionDialog(
                        image_array.shape, file_name, self, default_dimensions
                    )
                    # Qt6 no longer shows the "?" help button by default;
                    # the old WindowContextHelpButtonHint clear is gone.
                    progress.setValue(50)
                    QApplication.processEvents()
                    if dialog.exec():
                        dimensions = dialog.get_dimensions()
                        print(f"Assigned dimensions: {dimensions}")
                        if "H" in dimensions and "W" in dimensions:
                            self.image_dimensions[base_name] = dimensions
                            break
                        else:
                            QMessageBox.warning(
                                self,
                                "Invalid Dimensions",
                                "You must assign both H and W dimensions.",
                            )
                    else:
                        progress.close()
                        return
                progress.setValue(100)
                progress.close()
            else:
                dimensions = ["H", "W"]
                self.image_dimensions[base_name] = dimensions

        self.image_shapes[base_name] = image_array.shape
        print(f"Final assigned dimensions: {self.image_dimensions[base_name]}")
        print(f"Image shape: {self.image_shapes[base_name]}")

        if self.image_dimensions[base_name]:
            self.create_slices(
                image_array, self.image_dimensions[base_name], image_path
            )
        else:
            rgb_image = self.convert_to_8bit_rgb(image_array)
            self.current_image = self.array_to_qimage(rgb_image)
            self.slices = []
            self.slice_list.clear()

        if self.slices:
            self.current_image = self.slices[0][1]
            self.current_slice = self.slices[0][0]
            self.slice_list.setCurrentRow(0)
            self.load_image_annotations()
            self.image_label.update()

        self.update_image_info()

        # Update UI
        self.update_slice_list()
        self.update_annotation_list()
        self.image_label.update()

    def create_slices(self, image_array, dimensions, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        slices = []
        self.slice_list.clear()

        print(f"Creating slices for {base_name}")
        print(f"Dimensions: {dimensions}")
        print(f"Image array shape: {image_array.shape}")

        # Create and show progress dialog
        progress = QProgressDialog("Loading slices...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately

        # Handle 2D images
        if image_array.ndim == 2:
            progress.setValue(50)  # Update progress
            QApplication.processEvents()  # Allow GUI to update
            normalized_array = self.normalize_array(image_array)
            qimage = self.array_to_qimage(normalized_array)
            slice_name = f"{base_name}"
            slices.append((slice_name, qimage))
            self.add_slice_to_list(slice_name)
        else:
            # For 3D or higher dimensional arrays
            slice_indices = [
                i for i, dim in enumerate(dimensions) if dim not in ["H", "W"]
            ]

            total_slices = np.prod([image_array.shape[i] for i in slice_indices])
            for idx, _ in enumerate(
                np.ndindex(tuple(image_array.shape[i] for i in slice_indices))
            ):
                if progress.wasCanceled():
                    break

                full_idx = [slice(None)] * len(dimensions)
                for i, val in zip(slice_indices, _):
                    full_idx[i] = val

                slice_array = image_array[tuple(full_idx)]
                rgb_slice = self.convert_to_8bit_rgb(slice_array)
                qimage = self.array_to_qimage(rgb_slice)

                slice_name = f"{base_name}_{'_'.join([f'{dimensions[i]}{val+1}' for i, val in zip(slice_indices, _)])}"
                slices.append((slice_name, qimage))

                self.add_slice_to_list(slice_name)

                # Update progress
                progress_value = int((idx + 1) / total_slices * 100)
                progress.setValue(progress_value)
                QApplication.processEvents()  # Allow GUI to update

        progress.setValue(100)  # Ensure progress reaches 100%

        self.image_slices[base_name] = slices
        self.slices = slices

        if slices:
            self.current_image = slices[0][1]
            self.current_slice = slices[0][0]
            self.slice_list.setCurrentRow(0)

            self.activate_slice(self.current_slice)

            slice_info = f"Total slices: {len(slices)}"
            for dim, size in zip(dimensions, image_array.shape):
                if dim not in ["H", "W"]:
                    slice_info += f", {dim}: {size}"
            self.update_image_info(additional_info=slice_info)
        else:
            print("No slices were created")

        print(f"Created {len(slices)} slices for {base_name}")
        return slices

    def add_slice_to_list(self, slice_name):
        item = QListWidgetItem(slice_name)
        # Marker via the shared helper so the two code paths that build
        # this list cannot disagree about what a labeled slice looks like.
        # Only this row is touched: a stack can produce thousands of
        # slices, and re-scanning the whole project per row would make
        # loading quadratic.
        self._apply_labeled_marker(item, slice_name)
        self.slice_list.addItem(item)
        self._update_slice_panel_visibility()

    def normalize_array(self, array):
        # print(f"Normalizing array. Shape: {array.shape}, dtype: {array.dtype}")
        # print(f"Array min: {array.min()}, max: {array.max()}, mean: {array.mean()}")

        array_float = array.astype(np.float32)

        if array.dtype == np.uint16:
            array_normalized = (array_float - array.min()) / (array.max() - array.min())
        elif array.dtype == np.uint8:
            # For 8-bit images, use a simple contrast stretching
            p_low, p_high = np.percentile(
                array_float, (0, 100)
            )  # Change these to 1, 99 or something to stretch the contrast for visualizing 8 bit images
            array_normalized = np.clip(array_float, p_low, p_high)
            array_normalized = (array_normalized - p_low) / (p_high - p_low)
        else:
            array_normalized = (array_float - array.min()) / (array.max() - array.min())

        # Apply gamma correction
        gamma = 1.0  # Adjust this value to fine-tune brightness (> 1 for darker, < 1 for brighter)
        array_normalized = np.power(array_normalized, gamma)

        return (array_normalized * 255).astype(np.uint8)

    def adjust_contrast(self, image, low_percentile=1, high_percentile=99):
        if image.dtype != np.uint8:
            p_low, p_high = np.percentile(image, (low_percentile, high_percentile))
            image_adjusted = np.clip(image, p_low, p_high)
            image_adjusted = (image_adjusted - p_low) / (p_high - p_low)
            return (image_adjusted * 255).astype(np.uint8)
        return image

    def activate_slice(self, slice_name):
        self.current_slice = slice_name
        self.image_file_name = slice_name
        self.load_image_annotations()
        self.update_annotation_list()

        for name, qimage in self.slices:
            if name == slice_name:
                self.current_image = qimage
                self.display_image()
                break

        self.image_label.update()

        items = self.slice_list.findItems(slice_name, Qt.MatchFlag.MatchExactly)
        if items:
            self.slice_list.setCurrentItem(items[0])

    def array_to_qimage(self, array):
        if array.ndim == 2:
            height, width = array.shape
            return QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
        elif array.ndim == 3 and array.shape[2] == 3:
            height, width, _ = array.shape
            bytes_per_line = 3 * width
            return QImage(
                array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            )
        else:
            raise ValueError(
                f"Unsupported array shape {array.shape} for conversion to QImage"
            )

    def update_slice_list(self):
        self.slice_list.clear()
        for slice_name, _ in self.slices:
            self.slice_list.addItem(QListWidgetItem(slice_name))
        self.update_slice_list_colors()

        # Select the current slice
        if self.current_slice:
            items = self.slice_list.findItems(self.current_slice, Qt.MatchFlag.MatchExactly)
            if items:
                self.slice_list.setCurrentItem(items[0])

    def clear_slice_list(self):
        self.slice_list.clear()
        self.slices = []
        self.current_slice = None
        self._update_slice_panel_visibility()

    def reset_tool_buttons(self):
        for button in self.tool_group.buttons():
            button.setChecked(False)

    def keyPressEvent(self, event):
        # Check if the current focus is on a text editing widget
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, (QLineEdit, QTextEdit)):
            super().keyPressEvent(event)
            return

        # F2 (Snake game) is wired as a global QShortcut in __init__
        # so it works when child widgets have focus. Don't re-handle here.
        if event.key() == Qt.Key.Key_Delete:
            # Handle deletions
            if self.class_list.hasFocus() and self.class_list.currentItem():
                self.delete_class(self.class_list.currentItem())
            elif (
                self.annotation_list.hasFocus() and self.annotation_list.selectedItems()
            ):
                self.delete_selected_annotations()
            elif self.image_list.hasFocus() and self.image_list.currentItem():
                self.delete_selected_image()
        elif event.key() == Qt.Key.Key_A:
            self.go_to_previous_frame()
        elif event.key() == Qt.Key.Key_D:
            self.go_to_next_frame()
        elif event.key() == Qt.Key.Key_C:
            self.copy_selected_annotation_to_next_frame()
        elif event.key() == Qt.Key.Key_Up or event.key() == Qt.Key.Key_Down:
            # Handle slice navigation
            if self.slice_list.hasFocus():
                current_row = self.slice_list.currentRow()
                if event.key() == Qt.Key.Key_Up and current_row > 0:
                    self.slice_list.setCurrentRow(current_row - 1)
                elif (
                    event.key() == Qt.Key.Key_Down
                    and current_row < self.slice_list.count() - 1
                ):
                    self.slice_list.setCurrentRow(current_row + 1)
                self.switch_slice(self.slice_list.currentItem())
            else:
                # Pass the event to the parent for default handling
                super().keyPressEvent(event)
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # Handle accepting visible temporary classes
            if self.has_visible_temp_classes():
                self.accept_visible_temp_classes()
            else:
                super().keyPressEvent(event)
        elif event.key() == Qt.Key.Key_Escape:
            # Handle rejecting visible temporary classes
            if self.has_visible_temp_classes():
                self.reject_visible_temp_classes()
            else:
                super().keyPressEvent(event)
        else:
            # Pass any other key events to the parent for default handling
            super().keyPressEvent(event)

    def _trigger_video_shortcut(self, callback):
        # Text entry only. A non-editable QComboBox keeps focus after its
        # popup closes, so including combo boxes here would silently stop
        # A / D advancing frames for the rest of the session — combo
        # type-ahead is protected in the event-filter path instead, where
        # the key is genuinely being delivered to the combo.
        if isinstance(
            QApplication.focusWidget(), (QLineEdit, QTextEdit, QAbstractSpinBox)
        ):
            return
        callback()

    def has_visible_temp_classes(self):
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.text().startswith("Temp-") and item.checkState() == Qt.CheckState.Checked:
                return True
        return False

    def launch_snake_game(self):
        # print("Launching Snake game")
        if not hasattr(self, "snake_game") or not self.snake_game.isVisible():
            self.snake_game = SnakeGame()
        self.snake_game.show()
        self.snake_game.setFocus()

    def import_annotations(self):
        if self._reject_while_sam3_busy():
            return
        if not self.image_label.check_unsaved_changes():
            return
        print("Starting import_annotations")
        import_format = self.import_format_selector.currentText()
        print(f"Import format: {import_format}")

        if import_format == "COCO JSON":
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Import COCO JSON Annotations", "", "JSON Files (*.json)"
            )
            if not file_name:
                print("No file selected, returning")
                return

            print(f"Selected file: {file_name}")
            json_dir = os.path.dirname(file_name)
            images_dir = os.path.join(json_dir, "images")
            imported_annotations, image_info = import_coco_json(
                file_name, self.class_mapping
            )

        elif import_format in ["YOLO (v4 and earlier)", "YOLO (v5+)"]:
            yaml_file, _ = QFileDialog.getOpenFileName(
                self, "Select YOLO Dataset YAML", "", "YAML Files (*.yaml *.yml)"
            )
            if not yaml_file:
                print("No YAML file selected, returning")
                return

            print(f"Selected YAML file: {yaml_file}")
            try:
                imported_annotations, image_info = process_import_format(
                    import_format, yaml_file, self.class_mapping
                )
                yaml_dir = os.path.dirname(yaml_file)
                if import_format == "YOLO (v4 and earlier)":
                    images_dir = os.path.join(yaml_dir, "train", "images")
                else:  # YOLO (v5+)
                    images_dir = os.path.join(
                        yaml_dir, "images", "train"
                    )  # Preferring train over val
            except ValueError as e:
                QMessageBox.warning(self, "Import Error", str(e))
                return

        else:
            QMessageBox.warning(
                self,
                "Unsupported Format",
                f"The selected format '{import_format}' is not implemented for import.",
            )
            return

        print(
            f"JSON/YOLO directory: {json_dir if import_format == 'COCO JSON' else os.path.dirname(yaml_file)}"
        )
        print(f"Images directory: {images_dir}")
        print(f"Imported annotations count: {len(imported_annotations)}")
        print(f"Image info count: {len(image_info)}")

        images_loaded = 0
        images_not_found = []

        for info in image_info.values():
            print(f"Processing image: {info['file_name']}")
            image_path = os.path.join(images_dir, info["file_name"])

            if os.path.exists(image_path):
                print(f"Image found at: {image_path}")
                self.image_paths[info["file_name"]] = image_path
                self.all_images.append(
                    {
                        "file_name": info["file_name"],
                        "height": info["height"],
                        "width": info["width"],
                        "id": info["id"],
                        "is_multi_slice": False,
                    }
                )
                images_loaded += 1
            else:
                print(f"Image not found at: {image_path}")
                images_not_found.append(info["file_name"])

        print(f"Images loaded: {images_loaded}")
        print(f"Images not found: {len(images_not_found)}")

        if images_not_found:
            message = f"The following {len(images_not_found)} images were not found in the 'images' directory:\n\n"
            message += "\n".join(images_not_found[:10])
            if len(images_not_found) > 10:
                message += f"\n... and {len(images_not_found) - 10} more."
            message += "\n\nDo you want to proceed and ignore annotations for these missing images?"
            reply = QMessageBox.question(
                self,
                "Missing Images",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                print("Import cancelled due to missing images")
                QMessageBox.information(
                    self,
                    "Import Cancelled",
                    "Import cancelled. Please ensure all images are in the 'images' directory and try again.",
                )
                return

        # Update annotations (only for found images)
        for image_name, annotations in imported_annotations.items():
            if image_name not in self.image_paths:
                continue
            self.all_annotations[image_name] = {}
            for category_name, category_annotations in annotations.items():
                self.all_annotations[image_name][category_name] = []
                for i, ann in enumerate(category_annotations, start=1):
                    new_ann = {
                        "segmentation": ann.get("segmentation"),
                        "bbox": ann.get("bbox"),
                        "category_id": ann["category_id"],
                        "category_name": category_name,
                        "number": i,
                        "type": ann.get("type", "polygon"),
                    }
                    self.all_annotations[image_name][category_name].append(new_ann)

        # Update class mapping and colors
        for annotations in self.all_annotations.values():
            for category_name in annotations.keys():
                if category_name not in self.class_mapping:
                    new_id = len(self.class_mapping) + 1
                    self.class_mapping[category_name] = new_id
                    self.image_label.class_colors[category_name] = QColor(
                        Qt.GlobalColor(new_id % 16 + 7)
                    )

        print("Updating UI")
        # Update UI
        self.update_class_list()
        self.update_image_list()
        self.update_annotation_list()

        # Highlight and display the first image
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
            self.switch_image(self.image_list.item(0))

        # Select the first class if available
        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self.on_class_selected()

        self.image_label.update()

        message = f"Annotations have been imported successfully from {file_name if import_format == 'COCO JSON' else yaml_file}.\n"
        message += f"{images_loaded} images were loaded from the 'images' directory.\n"
        if images_not_found:
            message += (
                f"Annotations for {len(images_not_found)} missing images were ignored."
            )

        print("Import complete, showing message")
        QMessageBox.information(self, "Import Complete", message)
        self.auto_save()  # Auto-save after importing annotations

    def export_annotations(self):
        if not self.image_label.check_unsaved_changes():
            return
        export_format = self.export_format_selector.currentText()

        supported_formats = [
            "COCO JSON",
            "YOLO (v4 and earlier)",
            "YOLO (v5+)",
            "Labeled Images",
            "Semantic Labels",
            "RGB Semantic Masks",
            "Pascal VOC (BBox)",
            "Pascal VOC (BBox + Segmentation)",
        ]

        if export_format not in supported_formats:
            QMessageBox.warning(
                self,
                "Unsupported Format",
                f"The selected format '{export_format}' is not implemented.",
            )
            return

        if export_format == "COCO JSON":
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Export COCO JSON Annotations", "", "JSON Files (*.json)"
            )
        else:
            file_name = QFileDialog.getExistingDirectory(
                self, f"Select Output Directory for {export_format} Export"
            )

        if not file_name:
            return

        self.save_current_annotations()

        if export_format == "COCO JSON":
            output_dir = os.path.dirname(file_name)
            json_filename = os.path.basename(file_name)
            json_file, images_dir = export_coco_json(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                output_dir,
                json_filename,
            )
            message = (
                "Annotations have been exported successfully in COCO JSON format.\n"
            )
            message += f"JSON file: {json_file}\nImages directory: {images_dir}"

        elif export_format == "YOLO (v4 and earlier)":
            labels_dir, yaml_path = export_yolo_v4(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = "Annotations have been exported successfully in YOLO (v4 and earlier) format.\n"
            message += f"Labels: {labels_dir}\nYAML: {yaml_path}"

        elif export_format == "YOLO (v5+)":
            output_dir, yaml_path = export_yolo_v5plus(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = (
                "Annotations have been exported successfully in YOLO (v5+) format.\n"
            )
            message += f"Output directory: {output_dir}\nYAML: {yaml_path}"

        elif export_format == "Labeled Images":
            labeled_images_dir = export_labeled_images(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = f"Labeled images have been exported successfully.\nLabeled Images: {labeled_images_dir}\n"
            message += f"A class summary has been saved in: {os.path.join(labeled_images_dir, 'class_summary.txt')}"

        elif export_format == "Semantic Labels":
            semantic_labels_dir = export_semantic_labels(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = f"Semantic labels have been exported successfully.\nSemantic Labels: {semantic_labels_dir}\n"
            message += f"A class-pixel mapping has been saved in: {os.path.join(semantic_labels_dir, 'class_pixel_mapping.txt')}"

        elif export_format == "RGB Semantic Masks":
            try:
                rgb_masks_dir = export_rgb_semantic_masks(
                    self.all_annotations,
                    self.image_label.class_colors,
                    self.image_paths,
                    self.slices,
                    self.image_slices,
                    file_name,
                )
            except ValueError as exc:
                QMessageBox.warning(self, "RGB Mask Export", str(exc))
                return
            message = (
                "RGB semantic masks have been exported successfully.\n"
                f"Output: {rgb_masks_dir}\n"
                "Unlabeled pixels are black; class pixels use the configured "
                "class colors."
            )

        elif export_format == "Pascal VOC (BBox)":
            voc_dir = export_pascal_voc_bbox(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = "Annotations have been exported successfully in Pascal VOC format (BBox only).\n"
            message += f"Pascal VOC Annotations: {voc_dir}"

        elif export_format == "Pascal VOC (BBox + Segmentation)":
            voc_dir = export_pascal_voc_both(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = "Annotations have been exported successfully in Pascal VOC format (BBox + Segmentation).\n"
            message += f"Pascal VOC Annotations: {voc_dir}"

        QMessageBox.information(self, "Export Complete", message)


    def create_review_package(self):
        """Create a small, self-contained sample for team approval."""
        if not self.image_label.check_unsaved_changes():
            return
        self.save_current_annotations()

        def has_real_annotations(image_name):
            return any(
                annotations
                for class_name, annotations in self.all_annotations.get(
                    image_name, {}
                ).items()
                if not class_name.startswith("Temp-")
            )

        ordered_names = [
            self.image_list.item(index).text()
            for index in range(self.image_list.count())
            if has_real_annotations(self.image_list.item(index).text())
        ]
        ordered_names.extend(
            name
            for name in self.all_annotations
            if name not in ordered_names and has_real_annotations(name)
        )
        if not ordered_names:
            self.show_warning(
                "Nothing to Review",
                "Label at least one frame before creating a review package.",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Choose Labels for Team Review")
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        heading = QLabel("Choose a small pilot batch")
        heading.setProperty("class", "dialog-title")
        explanation = QLabel(
            "The first five labeled frames are selected. Adjust the selection "
            "if needed, then create a package Hira and Lucas can review without "
            "installing this app."
        )
        explanation.setWordWrap(True)
        explanation.setProperty("class", "help-text")
        layout.addWidget(heading)
        layout.addWidget(explanation)

        review_list = QListWidget()
        review_list.setObjectName("reviewFrameList")
        review_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        current_name = self.current_slice or self.image_file_name
        default_names = ordered_names[:5]
        if current_name in ordered_names and current_name not in default_names:
            default_names[-1] = current_name
        for image_name in ordered_names:
            item = QListWidgetItem(image_name)
            review_list.addItem(item)
            if image_name in default_names:
                item.setSelected(True)
        layout.addWidget(review_list, 1)

        selection_hint = QLabel(
            "Tip: Ctrl+click adds or removes individual frames from the selection."
        )
        selection_hint.setProperty("class", "help-text")
        layout.addWidget(selection_hint)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Create Package"
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected_names = [item.text() for item in review_list.selectedItems()]
        if not selected_names:
            self.show_warning(
                "No Frames Selected",
                "Select at least one labeled frame for the review package.",
            )
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Choose an Empty Folder for the Review Package"
        )
        if not output_dir:
            return
        try:
            result = export_review_package(
                self.all_annotations,
                self.image_label.class_colors,
                self.image_paths,
                self.slices,
                self.image_slices,
                output_dir,
                selected_names,
            )
        except (OSError, ValueError) as exc:
            self.show_warning("Review Package", str(exc))
            return

        self.show_info(
            "Review Package Ready",
            f"{result['frame_count']} labels were packaged for review.\n\n"
            f"Preview in a browser:\n{result['review_path']}\n\n"
            f"Ready-to-send ZIP:\n{result['archive_path']}",
        )

    def save_slices(self, directory):
        slices_saved = False
        for image_file, image_slices in self.image_slices.items():
            for slice_name, qimage in image_slices:
                if (
                    slice_name in self.all_annotations
                    and self.all_annotations[slice_name]
                ):
                    file_path = os.path.join(directory, f"{slice_name}.png")
                    qimage.save(file_path, "PNG")
                    slices_saved = True

        return slices_saved

    def create_coco_annotation(self, ann, image_id, annotation_id):
        coco_ann = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": ann["category_id"],
            "area": calculate_area(ann),
            "iscrowd": 0,
        }

        if "segmentation" in ann:
            coco_ann["segmentation"] = [ann["segmentation"]]
            coco_ann["bbox"] = calculate_bbox(ann["segmentation"])
        elif "bbox" in ann:
            coco_ann["bbox"] = ann["bbox"]

        return coco_ann

    def update_all_annotation_lists(self):
        for image_name in self.all_annotations.keys():
            self.update_annotation_list(image_name)
        self.update_annotation_list()  # Update for the current image/slice

    def update_annotation_list(self, image_name=None):
        self.annotation_list.clear()
        current_name = image_name or self.current_slice or self.image_file_name
        annotations = self.all_annotations.get(current_name, {})
        for class_name, class_annotations in annotations.items():
            if not class_name.startswith(
                "Temp-"
            ):  # Only show non-temporary annotations
                color = self.image_label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
                for annotation in class_annotations:
                    number = annotation.get("number", 0)
                    area = calculate_area(annotation)
                    item_text = f"{class_name} - {number:<3} Area: {area:.2f}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, annotation)
                    item.setForeground(color)
                    self.annotation_list.addItem(item)

        # Force the annotation list to repaint
        self.annotation_list.repaint()

    def update_slice_list_colors(self):
        """Kept as the name every existing mutation path already calls.

        `annotations_changed()` is the real hook — use that in new code.
        This delegates so the two dozen existing call sites keep working
        without each one having to be edited.

        Slices no longer get a filled row colour. The old hardcoded
        steel-blue and light-blue fills were written per theme, ignored
        the stylesheet's own selection colour, and made the selected row
        indistinguishable from a labeled one. A marker dot carries the
        same information and leaves the theme in charge.
        """
        self.annotations_changed()

    #: Where a row caches the appearance it was last given, so a refresh
    #: over thousands of rows only touches the ones that actually changed.
    _MARKER_STATE_ROLE = Qt.ItemDataRole.UserRole + 1

    def _apply_labeled_marker(self, item, name, labeled=None):
        """Give one list row its labeled / still-to-do appearance.

        Returns immediately when the row already looks right, so a
        refresh over an unchanged list costs one comparison per row and
        no widget calls at all. The cached value carries the theme as
        well as the label state, so toggling dark mode re-colours every
        marker without a separate invalidation step.

        ``labeled`` lets a caller that already computed the answer pass
        it in rather than paying for the lookup twice.
        """
        if labeled is None:
            labeled = self.frame_has_labels(name)
        state = (labeled, self.dark_mode)
        if item.data(self._MARKER_STATE_ROLE) == state:
            return

        tokens = tokens_for(self.dark_mode)
        item.setIcon(
            self._status_dot(
                QColor(tokens["success"] if labeled else tokens["border_strong"])
            )
        )
        item.setForeground(
            QColor(tokens["text"] if labeled else tokens["text_muted"])
        )
        item.setBackground(QColor(Qt.GlobalColor.transparent))
        item.setToolTip(
            f"{name}\n{'Has labels' if labeled else 'No labels yet'}"
        )
        item.setData(self._MARKER_STATE_ROLE, state)

    def _update_slice_panel_visibility(self):
        """Hide the slice list unless the open image actually has slices.

        Multi-dimensional stacks are the exception in this workflow, not
        the rule, so an always-visible empty "Slices:" box just ate a
        third of the frame panel for every video project.
        """
        if not hasattr(self, "slice_list") or not hasattr(self, "slice_heading"):
            return
        has_slices = self.slice_list.count() > 0
        self.slice_heading.setVisible(has_slices)
        self.slice_list.setVisible(has_slices)

    def update_annotation_list_colors(self, class_name=None, color=None):
        for i in range(self.annotation_list.count()):
            item = self.annotation_list.item(i)
            annotation = item.data(Qt.ItemDataRole.UserRole)
            # Update only the item for the specific class if class_name is provided
            if class_name is None or annotation["category_name"] == class_name:
                item_color = (
                    color
                    if class_name
                    else self.image_label.class_colors.get(
                        annotation["category_name"], QColor(Qt.GlobalColor.white)
                    )
                )
                item.setForeground(item_color)

    def load_image_annotations(self):
        # print(f"Loading annotations for: {self.current_slice or self.image_file_name}")
        self.image_label.annotations.clear()
        current_name = self.current_slice or self.image_file_name
        # print(f"Current name for annotations: {current_name}")
        # print(f"All annotations keys: {list(self.all_annotations.keys())}")
        if current_name in self.all_annotations:
            self.image_label.annotations = copy.deepcopy(
                self.all_annotations[current_name]
            )
            # print(f"Loaded annotations: {self.image_label.annotations}")
        else:
            print(f"No annotations found for {current_name}")
        self.image_label.update()

    def save_current_annotations(self):
        if self.current_slice:
            current_name = self.current_slice
        elif self.image_file_name:
            current_name = self.image_file_name
        else:
            # print("Error: No current slice or image file name set")
            return

        # print(f"Saving annotations for: {current_name}")
        if self.image_label.annotations:
            self.all_annotations[current_name] = self.image_label.annotations.copy()
            # print(f"Saved {len(self.image_label.annotations)} annotations for {current_name}")
        elif current_name in self.all_annotations:
            del self.all_annotations[current_name]
            # print(f"Removed annotations for {current_name}")

        self.update_slice_list_colors()

        # print(f"All annotations now: {self.all_annotations.keys()}")
        # print(f"Current slice: {self.current_slice}")
        # print(f"Current image_file_name: {self.image_file_name}")

    def setup_class_list(self):
        """Set up the class list widget."""
        self.class_list = QListWidget()
        self.class_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self.show_class_context_menu)
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.sidebar_layout.addWidget(QLabel("Classes:"))
        self.sidebar_layout.addWidget(self.class_list)

    def setup_tool_buttons(self):
        """Set up the tool buttons with grouped manual and automated tools."""
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)

        # Create a widget for manual tools
        manual_tools_widget = QWidget()
        manual_layout = QVBoxLayout(manual_tools_widget)
        manual_layout.setSpacing(5)

        manual_label = QLabel("Manual Tools")
        manual_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        manual_layout.addWidget(manual_label)

        manual_buttons_layout = QHBoxLayout()
        self.polygon_button = QPushButton("Polygon")
        self.polygon_button.setCheckable(True)
        self.rectangle_button = QPushButton("Rectangle")
        self.rectangle_button.setCheckable(True)
        manual_buttons_layout.addWidget(self.polygon_button)
        manual_buttons_layout.addWidget(self.rectangle_button)
        manual_layout.addLayout(manual_buttons_layout)

        self.tool_group.addButton(self.polygon_button)
        self.tool_group.addButton(self.rectangle_button)
        self.polygon_button.clicked.connect(self.toggle_tool)
        self.rectangle_button.clicked.connect(self.toggle_tool)

        # Create a widget for automated tools
        automated_tools_widget = QWidget()
        automated_layout = QVBoxLayout(automated_tools_widget)
        automated_layout.setSpacing(5)

        automated_label = QLabel("Automated Tools")
        automated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        automated_layout.addWidget(automated_label)

        automated_buttons_layout = QHBoxLayout()
        self.sam_magic_wand_button = QPushButton("Magic Wand")
        self.sam_magic_wand_button.setCheckable(True)
        automated_buttons_layout.addWidget(self.sam_magic_wand_button)
        automated_layout.addLayout(automated_buttons_layout)

        self.tool_group.addButton(self.sam_magic_wand_button)
        self.sam_magic_wand_button.clicked.connect(self.toggle_tool)

        # Add the grouped tools to the sidebar layout
        self.sidebar_layout.addWidget(manual_tools_widget)
        self.sidebar_layout.addWidget(automated_tools_widget)

        # Set a fixed size for all buttons to make them smaller
        for button in [
            self.polygon_button,
            self.rectangle_button,
            self.load_sam2_button,
            self.sam_magic_wand_button,
        ]:
            button.setFixedSize(100, 30)

    def setup_annotation_list(self):
        """Set up the annotation list widget."""
        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.annotation_list.itemSelectionChanged.connect(
            self.update_highlighted_annotations
        )

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # Video Menu
        video_menu = menu_bar.addMenu("&Video")
        open_video_action = QAction("Open Video &Clip...", self)
        open_video_action.triggered.connect(self.open_video_clip)
        video_menu.addAction(open_video_action)

        open_folder_action = QAction("Open Frame &Folder...", self)
        open_folder_action.triggered.connect(self.open_frame_folder)
        video_menu.addAction(open_folder_action)

        next_frame_action = QAction("&Next Frame (D)", self)
        next_frame_action.triggered.connect(self.go_to_next_frame)
        video_menu.addAction(next_frame_action)

        prev_frame_action = QAction("&Previous Frame (A)", self)
        prev_frame_action.triggered.connect(self.go_to_previous_frame)
        video_menu.addAction(prev_frame_action)

        copy_anno_action = QAction("&Copy Selected Annotation to Next Frame (C)", self)
        copy_anno_action.triggered.connect(self.copy_selected_annotation_to_next_frame)
        video_menu.addAction(copy_anno_action)

        # Welding Menu
        welding_menu = menu_bar.addMenu("&Welding")
        add_welding_action = QAction("Add ER70S-6 &Full Arc Classes", self)
        add_welding_action.triggered.connect(self.add_default_welding_classes)
        welding_menu.addAction(add_welding_action)

        add_cavitar_action = QAction("Add ER70S-6 &CAVITAR Classes", self)
        add_cavitar_action.triggered.connect(self.add_cavitar_welding_classes)
        welding_menu.addAction(add_cavitar_action)

        welding_menu.addSeparator()
        protocol_action = QAction("Show ER70S-6 Labeling &Protocol", self)
        protocol_action.triggered.connect(self.show_er70s6_protocol)
        welding_menu.addAction(protocol_action)

        # Project Menu
        project_menu = menu_bar.addMenu("&Project")

        new_project_action = QAction("&New Project", self)
        new_project_action.setShortcut(QKeySequence.StandardKey.New)
        new_project_action.triggered.connect(self.new_project)
        project_menu.addAction(new_project_action)

        open_project_action = QAction("&Open Project", self)
        open_project_action.setShortcut(QKeySequence.StandardKey.Open)
        open_project_action.triggered.connect(self.open_project)
        project_menu.addAction(open_project_action)

        save_project_action = QAction("&Save Project", self)
        save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        save_project_action.triggered.connect(self.save_project)
        project_menu.addAction(save_project_action)

        save_project_as_action = QAction("Save Project &As...", self)
        save_project_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_project_as_action.triggered.connect(self.save_project_as)
        project_menu.addAction(save_project_as_action)

        close_project_action = QAction("&Close Project", self)
        close_project_action.setShortcut(QKeySequence("Ctrl+W"))
        close_project_action.triggered.connect(self.close_project)
        project_menu.addAction(close_project_action)

        project_details_action = QAction("Project &Details", self)
        project_details_action.setShortcut(QKeySequence("Ctrl+I"))
        project_details_action.triggered.connect(self.show_project_details)
        project_menu.addAction(project_details_action)

        search_projects_action = QAction("&Search Projects", self)
        search_projects_action.setShortcut(QKeySequence("Ctrl+F"))
        search_projects_action.triggered.connect(self.show_project_search)
        project_menu.addAction(search_projects_action)

        # Edit Menu — undo lives here because that is the first place
        # anyone looks for it.
        edit_menu = menu_bar.addMenu("&Edit")

        self.undo_action = QAction("&Undo Annotation Change", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        # Application context so Ctrl+Z still reaches the annotator while
        # focus sits in the frame list, the class list or a tool panel.
        self.undo_action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.undo_action.triggered.connect(self.undo_annotation_change)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("&Redo Annotation Change", self)
        # Ctrl+Shift+Z and Ctrl+Y both reach redo; people arrive from
        # different editors. Both live on the one action so neither is
        # bound twice — a sequence bound by an action *and* a QShortcut
        # is ambiguous, and Qt then triggers neither.
        #
        # Built by de-duplicating rather than listing literally, because
        # StandardKey.Redo resolves per platform: Ctrl+Shift+Z on Linux
        # and macOS, but Ctrl+Y on Windows. A literal
        # [StandardKey.Redo, "Ctrl+Y"] therefore binds Ctrl+Y twice on
        # Windows — which is exactly the ambiguity this comment warns
        # about, on the platform the lab actually runs.
        self.redo_action.setShortcuts(
            redo_shortcut_sequences(QKeySequence(QKeySequence.StandardKey.Redo))
        )
        self.redo_action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.redo_action.triggered.connect(self.redo_annotation_change)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        delete_annotations_action = QAction("&Delete Selected Annotations", self)
        delete_annotations_action.triggered.connect(self.delete_selected_annotations)
        edit_menu.addAction(delete_annotations_action)

        # Settings Menu
        settings_menu = menu_bar.addMenu("&Settings")

        font_size_menu = settings_menu.addMenu("&Font Size")
        for size in ["Small", "Medium", "Large", "XL", "XXL"]:
            action = QAction(size, self)
            action.triggered.connect(lambda checked, s=size: self.change_font_size(s))
            font_size_menu.addAction(action)

        toggle_dark_mode_action = QAction("Toggle &Dark Mode", self)
        toggle_dark_mode_action.setShortcut(QKeySequence("Ctrl+D"))
        toggle_dark_mode_action.triggered.connect(self.toggle_dark_mode)
        settings_menu.addAction(toggle_dark_mode_action)

        fit_action = QAction("&Fit Frame to Window", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self.adjust_zoom_to_fit)
        settings_menu.addAction(fit_action)

        # Tools Menu
        tools_menu = menu_bar.addMenu("&Tools")

        annotation_stats_action = QAction("Annotation Statistics", self)
        annotation_stats_action.triggered.connect(self.show_annotation_statistics)
        annotation_stats_action.setShortcut(QKeySequence("Ctrl+Alt+S"))
        tools_menu.addAction(annotation_stats_action)

        coco_json_combiner_action = QAction("COCO JSON Combiner", self)
        coco_json_combiner_action.triggered.connect(self.show_coco_json_combiner)
        tools_menu.addAction(coco_json_combiner_action)

        dataset_splitter_action = QAction("Dataset Splitter", self)
        dataset_splitter_action.triggered.connect(self.open_dataset_splitter)
        tools_menu.addAction(dataset_splitter_action)

        dino_merge_action = QAction("Merge COCO for Training", self)
        dino_merge_action.triggered.connect(self.show_dino_merge_dialog)
        tools_menu.addAction(dino_merge_action)

        stack_to_slices_action = QAction("Stack to Slices", self)
        stack_to_slices_action.triggered.connect(self.show_stack_to_slices)
        tools_menu.addAction(stack_to_slices_action)

        image_patcher_action = QAction("Image Patcher", self)
        image_patcher_action.triggered.connect(self.show_image_patcher)
        tools_menu.addAction(image_patcher_action)

        image_augmenter_action = QAction("Image Augmenter", self)
        image_augmenter_action.triggered.connect(self.show_image_augmenter)
        tools_menu.addAction(image_augmenter_action)

        slice_registration_action = QAction("Slice Registration", self)
        slice_registration_action.triggered.connect(self.show_slice_registration)
        tools_menu.addAction(slice_registration_action)

        stack_interpolator_action = QAction("Stack Interpolator", self)
        stack_interpolator_action.triggered.connect(self.show_stack_interpolator)
        tools_menu.addAction(stack_interpolator_action)

        dicom_converter_action = QAction("DICOM Converter", self)
        dicom_converter_action.triggered.connect(self.show_dicom_converter)
        tools_menu.addAction(dicom_converter_action)

        tools_menu.addSeparator()

        unload_models_action = QAction("Unload AI Models (Free GPU Memory)", self)
        unload_models_action.triggered.connect(self.unload_ai_models)
        tools_menu.addAction(unload_models_action)

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")

        help_action = QAction("&Show Help", self)
        help_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.setShortcut(QKeySequence("Ctrl+/"))
        shortcuts_action.triggered.connect(self.show_shortcut_reference)
        help_menu.addAction(shortcuts_action)

    def change_font_size(self, size):
        self.current_font_size = size
        self.apply_theme_and_font()

    def unload_ai_models(self):
        """Drop cached SAM/DINO model objects to free GPU/CPU memory.

        Useful on constrained GPUs (e.g. 8 GB) where SAM 2 base + DINO
        base together exhaust VRAM. After unload, the next inference
        call will re-load the model from disk (~1-3 s).
        """
        if self._sam3_inference_in_flight:
            self.show_warning("SAM 3 Tracker", "Wait for SAM 3 tracking to finish.")
            return
        self.sam_utils.unload()
        self.dino_utils.unload()
        if self.sam3_tracker is not None:
            self.sam3_tracker.unload()
            self.sam3_tracker = None
        # Reset the dropdowns to a neutral state so the user knows they
        # need to re-pick the model.
        self.sam_model_selector.setCurrentIndex(0)
        if hasattr(self, "dino_model_selector"):
            self.dino_model_selector.setCurrentIndex(0)
            self.dino_model_loaded = False
            self.lbl_dino_status.setText("No DINO model loaded")
            self.btn_detect_single.setEnabled(False)
            self.btn_detect_batch.setEnabled(False)
        QMessageBox.information(
            self,
            "Models Unloaded",
            "SAM, SAM 3, and DINO models have been unloaded from memory.\n\n"
            "Note: PyTorch keeps a per-process CUDA context that survives "
            "this unload (typically a few hundred MB visible in Task Manager / "
            "nvidia-smi). To fully reclaim GPU memory, restart the app.\n\n"
            "Re-select a SAM/DINO model to use AI tools again.",
        )

    def setup_sidebar(self):
        self.sidebar = QWidget()
        self.sidebar.setObjectName("controlPanel")
        # Minimum sized so the widest two-button row still fits; the
        # splitter lets the annotator go wider. Before this the panel was
        # pinned at 360px and clipped "Open Video Clip..." off its right
        # edge, with horizontal scrolling switched off.
        self.sidebar.setMinimumWidth(340)
        self.sidebar.setMaximumWidth(620)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        self.sidebar_layout.setSpacing(8)
        self.main_splitter.addWidget(self.sidebar)

        # Project identity: what is open and whether it is saved. The
        # previous header was a product name and tagline, which told the
        # annotator nothing they did not already know.
        identity = QWidget()
        identity.setObjectName("productIdentity")
        identity_layout = QVBoxLayout(identity)
        identity_layout.setContentsMargins(2, 0, 2, 2)
        identity_layout.setSpacing(1)
        self.project_name_label = QLabel("No project")
        self.project_name_label.setProperty("class", "product-title")
        self.project_meta_label = QLabel("Create or open a project to begin")
        self.project_meta_label.setProperty("class", "product-subtitle")
        self.project_meta_label.setWordWrap(True)
        identity_layout.addWidget(self.project_name_label)
        identity_layout.addWidget(self.project_meta_label)
        self.sidebar_layout.addWidget(identity)

        def help_text(text):
            label = QLabel(text)
            label.setProperty("class", "help-text")
            label.setWordWrap(True)
            return label

        def describe(widget, text):
            """Expose plain-language help to mouse and keyboard users."""
            widget.setToolTip(text)
            widget.setStatusTip(text)
            widget.setAccessibleDescription(text)

        def group(title):
            box = QGroupBox(title.upper())
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(0, 6, 0, 4)
            box_layout.setSpacing(6)
            return box, box_layout

        def side_by_side(*widgets):
            """Lay widgets across one row without letting them overflow.

            Buttons report their full label as a minimum width, and the
            sidebar scroll area has horizontal scrolling switched off, so
            a row wider than the panel used to be clipped at the right
            edge — "Open Video Clip..." lost its last characters and
            "Erase mask" disappeared entirely. An explicit minimum lets Qt
            shrink and elide the label instead of overflowing; the full
            text stays available in the tooltip.
            """
            row = QHBoxLayout()
            row.setSpacing(6)
            for widget in widgets:
                widget.setMinimumWidth(1)
                row.addWidget(widget, 1)
            return row

        def scroll_page():
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(2, 4, 6, 6)
            page_layout.setSpacing(6)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            scroll.setWidget(page)
            return scroll, page, page_layout

        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setObjectName("workflowTabs")
        self.sidebar_layout.addWidget(self.sidebar_tabs, 1)

        (
            self.labeling_scroll,
            self.labeling_page,
            labeling_layout,
        ) = scroll_page()
        self.sidebar_tabs.addTab(self.labeling_scroll, "Label")

        # Replaces the static "YOUR WORKFLOW: add frames -> choose a class
        # -> draw -> review -> export" banner. That text was correct on day
        # one and dead weight from day two. This line reports the next
        # useful action for the state the session is actually in, and is
        # refreshed by update_next_step_hint().
        self.workflow_hint = QLabel()
        self.workflow_hint.setObjectName("labelWorkflowHint")
        self.workflow_hint.setProperty("class", "workflow-hint")
        self.workflow_hint.setWordWrap(True)
        labeling_layout.addWidget(self.workflow_hint)

        data_group, data_layout = group("Start a labeling session")

        self.import_format_selector = QComboBox()
        self.import_format_selector.addItem("COCO JSON")
        self.import_format_selector.addItem("YOLO (v4 and earlier)")
        self.import_format_selector.addItem("YOLO (v5+)")
        self.import_format_selector.setToolTip(
            "Format of an existing labeled dataset to import."
        )

        self.import_button = QPushButton("Import Labels")
        self.import_button.clicked.connect(self.import_annotations)
        self.import_button.setToolTip(
            "Import images together with existing COCO or YOLO labels."
        )
        data_layout.addLayout(
            side_by_side(self.import_format_selector, self.import_button)
        )

        self.add_images_button = QPushButton("Add New Images")
        self.add_images_button.clicked.connect(self.add_images)
        self.add_images_button.setProperty("buttonRole", "primary")
        self.add_images_button.setToolTip("Add still images or an image sequence.")

        self.open_video_button = QPushButton("Open Video Clip...")
        self.open_video_button.clicked.connect(self.open_video_clip)
        self.open_video_button.setToolTip(
            "Load only a selected frame range from a large video."
        )
        data_layout.addLayout(
            side_by_side(self.add_images_button, self.open_video_button)
        )

        self.cavitar_preset_button = QPushButton("Droplets only")
        self.cavitar_preset_button.clicked.connect(
            self.add_cavitar_welding_classes
        )
        self.cavitar_preset_button.setToolTip(
            "Add only molten_consumable and droplet with the agreed RGB colors."
        )
        self.full_arc_preset_button = QPushButton("Droplets + arc")
        self.full_arc_preset_button.clicked.connect(
            self.add_default_welding_classes
        )
        self.full_arc_preset_button.setToolTip(
            "Add molten_consumable, droplet, external_arc, and internal_arc."
        )
        data_layout.addLayout(
            side_by_side(self.cavitar_preset_button, self.full_arc_preset_button)
        )

        labeling_layout.addWidget(data_group)

        class_group, class_layout = group("Label classes")

        self.class_list = QListWidget()
        self.class_list.setObjectName("classList")
        self.class_list.setMinimumHeight(90)
        self.class_list.setMaximumHeight(120)
        self.class_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self.show_class_context_menu)
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.class_list.setToolTip(
            "Select the class to assign before drawing an annotation. "
            "Press 1-9 to switch class without leaving the canvas."
        )
        class_layout.addWidget(self.class_list)

        self.add_class_button = QPushButton("Add Custom Class")
        self.add_class_button.clicked.connect(lambda: self.add_class())
        self.add_class_button.setProperty("buttonRole", "quiet")
        class_layout.addWidget(self.add_class_button)
        class_layout.addWidget(
            help_text(
                "Press 1-9 to switch class. Right-click a class to rename it, "
                "recolour it, or delete it."
            )
        )
        labeling_layout.addWidget(class_group)

        display_group, display_group_layout = group("Make boundaries easier to see")
        display_controls = QWidget()
        display_layout = QGridLayout(display_controls)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setHorizontalSpacing(8)
        display_layout.setVerticalSpacing(4)
        display_layout.setColumnStretch(1, 1)

        self.brightness_value_label = QLabel("+0")
        self.brightness_value_label.setProperty("class", "mono")
        self.brightness_value_label.setMinimumWidth(30)
        self.brightness_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_display_adjustments)
        describe(
            self.brightness_slider,
            "Lighten or darken only the on-screen preview. The source image "
            "and exported mask are not changed.",
        )
        display_layout.addWidget(QLabel("Brightness"), 0, 0)
        display_layout.addWidget(self.brightness_slider, 0, 1)
        display_layout.addWidget(self.brightness_value_label, 0, 2)

        self.contrast_value_label = QLabel("+0")
        self.contrast_value_label.setProperty("class", "mono")
        self.contrast_value_label.setMinimumWidth(30)
        self.contrast_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.update_display_adjustments)
        describe(
            self.contrast_slider,
            "Change preview contrast to make faint boundaries easier to see. "
            "The source image and exports are not changed.",
        )
        display_layout.addWidget(QLabel("Contrast"), 1, 0)
        display_layout.addWidget(self.contrast_slider, 1, 1)
        display_layout.addWidget(self.contrast_value_label, 1, 2)

        self.reset_display_button = QPushButton("Reset Display")
        self.reset_display_button.clicked.connect(self.reset_display_adjustments)
        self.reset_display_button.setProperty("buttonRole", "quiet")
        display_layout.addWidget(self.reset_display_button, 2, 0, 1, 3)
        display_group_layout.addWidget(display_controls)
        display_group_layout.addWidget(
            help_text("Preview only. Source images, masks, and exports stay unchanged.")
        )
        labeling_layout.addWidget(display_group)

        manual_group, manual_layout = group("Draw or correct a mask")

        self.polygon_button = QPushButton("Draw polygon")
        self.polygon_button.setCheckable(True)
        self.polygon_button.setProperty("buttonRole", "tool")
        describe(
            self.polygon_button,
            "Click around one object boundary, then press Enter to finish the mask.",
        )
        self.rectangle_button = QPushButton("Draw box")
        self.rectangle_button.setCheckable(True)
        self.rectangle_button.setProperty("buttonRole", "tool")
        describe(
            self.rectangle_button,
            "Drag a rectangle around an object to create a box annotation.",
        )

        self.paint_brush_button = QPushButton("Paint mask")
        self.paint_brush_button.setCheckable(True)
        self.paint_brush_button.setProperty("buttonRole", "tool")
        describe(
            self.paint_brush_button,
            "Paint pixels into the selected class. Display adjustments remain "
            "preview-only while painting.",
        )
        self.eraser_button = QPushButton("Erase mask")
        self.eraser_button.setCheckable(True)
        self.eraser_button.setProperty("buttonRole", "tool")
        describe(
            self.eraser_button,
            "Remove pixels only from the selected class without changing other classes.",
        )

        manual_layout.addLayout(
            side_by_side(self.polygon_button, self.rectangle_button)
        )
        manual_layout.addLayout(
            side_by_side(self.paint_brush_button, self.eraser_button)
        )

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_annotation_change)
        self.undo_button.setEnabled(False)
        describe(
            self.undo_button,
            "Undo the last annotation change on this frame (Ctrl+Z).",
        )
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo_annotation_change)
        self.redo_button.setEnabled(False)
        describe(
            self.redo_button,
            "Redo the annotation change you just undid (Ctrl+Shift+Z).",
        )
        manual_layout.addLayout(side_by_side(self.undo_button, self.redo_button))

        manual_layout.addWidget(
            help_text(
                "Shortcuts: P polygon, R box, B paint, E erase, Esc cancel. "
                "Polygon finishes on Enter. The eraser changes only the "
                "selected class. Use - and = for brush size."
            )
        )
        labeling_layout.addWidget(manual_group)

        annotations_group, annotations_layout = group("Review this frame")
        self.annotation_list = QListWidget()
        self.annotation_list.setMinimumHeight(110)
        self.annotation_list.setMaximumHeight(150)
        self.annotation_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.annotation_list.itemSelectionChanged.connect(
            self.update_highlighted_annotations
        )
        annotations_layout.addWidget(self.annotation_list)

        self.sort_by_class_button = QPushButton("Sort by Class")
        self.sort_by_class_button.clicked.connect(self.sort_annotations_by_class)
        self.sort_by_area_button = QPushButton("Sort by Area")
        self.sort_by_area_button.clicked.connect(self.sort_annotations_by_area)
        annotations_layout.addLayout(
            side_by_side(self.sort_by_class_button, self.sort_by_area_button)
        )

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_selected_annotations)
        self.delete_button.setProperty("buttonRole", "danger")
        self.merge_button = QPushButton("Merge")
        self.merge_button.clicked.connect(self.merge_annotations)
        self.change_class_button = QPushButton("Change Class")
        self.change_class_button.clicked.connect(self.change_annotation_class)
        annotations_layout.addLayout(
            side_by_side(
                self.delete_button, self.merge_button, self.change_class_button
            )
        )
        labeling_layout.addWidget(annotations_group)

        export_group, export_layout = group("Export and share")
        self.export_format_selector = QComboBox()
        self.export_format_selector.addItem("COCO JSON")
        self.export_format_selector.addItem("YOLO (v4 and earlier)")
        self.export_format_selector.addItem("YOLO (v5+)")
        self.export_format_selector.addItem("Labeled Images")
        self.export_format_selector.addItem("Semantic Labels")
        self.export_format_selector.addItem("RGB Semantic Masks")
        self.export_format_selector.addItem("Pascal VOC (BBox)")
        self.export_format_selector.addItem("Pascal VOC (BBox + Segmentation)")
        export_layout.addWidget(self.export_format_selector)

        self.export_button = QPushButton("Export Training Labels")
        self.export_button.clicked.connect(self.export_annotations)
        self.export_button.setProperty("buttonRole", "primary")
        describe(
            self.export_button,
            "Write the reviewed labels to a folder without modifying source images.",
        )
        export_layout.addWidget(self.export_button)

        self.review_package_button = QPushButton("Create Review Package")
        self.review_package_button.clicked.connect(self.create_review_package)
        self.review_package_button.setProperty("buttonRole", "accent")
        describe(
            self.review_package_button,
            "Choose a few labeled frames and create source images, exact RGB "
            "masks, overlays, and a review page your team can open in a browser.",
        )
        export_layout.addWidget(self.review_package_button)
        export_layout.addWidget(
            help_text(
                "Training export creates model-ready files. Review Package creates "
                "an easy sample for your team to approve first."
            )
        )
        labeling_layout.addWidget(export_group)
        labeling_layout.addStretch(1)

        self.ai_scroll, self.ai_page, ai_layout = scroll_page()
        self.sidebar_tabs.addTab(self.ai_scroll, "Auto-track")
        ai_workflow_hint = QLabel(
            "TRACK ACROSS FRAMES\n"
            "1  Draw one clean mask     2  Prepare frames\n"
            "3  Track forward           4  Review every frame"
        )
        ai_workflow_hint.setObjectName("aiWorkflowHint")
        ai_workflow_hint.setProperty("cardRole", "notice")
        ai_workflow_hint.setWordWrap(True)
        ai_layout.addWidget(ai_workflow_hint)
        ai_layout.addWidget(
            help_text("AI masks are suggestions, not final labels. Check every frame.")
        )

        sam2_group, sam2_layout = group("Improve one frame")
        self.sam_model_selector = QComboBox()
        self.sam_model_selector.addItem("Pick a SAM Model")
        self.sam_model_selector.addItems(list(self.sam_utils.sam_models.keys()))
        self.sam_model_selector.currentTextChanged.connect(self.change_sam_model)
        sam2_layout.addWidget(self.sam_model_selector)

        self.sam_box_button = QPushButton("Box Prompt")
        self.sam_box_button.setCheckable(True)
        self.sam_box_button.setProperty("buttonRole", "tool")
        self.sam_box_button.clicked.connect(self.toggle_sam_box)
        self.sam_points_button = QPushButton("Point Prompts")
        self.sam_points_button.setCheckable(True)
        self.sam_points_button.setProperty("buttonRole", "tool")
        self.sam_points_button.clicked.connect(self.toggle_sam_points)
        sam2_layout.addLayout(
            side_by_side(self.sam_box_button, self.sam_points_button)
        )
        sam2_layout.addWidget(
            help_text("Use a box or positive/negative points to segment one image.")
        )
        ai_layout.addWidget(sam2_group)

        sam3_group, sam3_layout = group("Continue a mask through frames")
        sam3_scope = QLabel(
            "Tracks from the current frame to the end of the frames currently "
            "loaded in the Images list."
        )
        sam3_scope.setObjectName("sam3ScopeLabel")
        sam3_scope.setProperty("cardRole", "info")
        sam3_scope.setWordWrap(True)
        sam3_layout.addWidget(sam3_scope)

        self.sam3_init_btn = QPushButton("1. Prepare Loaded Frames")
        self.sam3_init_btn.clicked.connect(self.init_sam3_tracker)
        self.sam3_init_btn.setProperty("buttonRole", "primary")
        describe(
            self.sam3_init_btn,
            "Load the current image sequence into SAM 3. Only frames shown in "
            "the Images list are prepared.",
        )
        sam3_layout.addWidget(self.sam3_init_btn)

        sam3_buttons_layout = QHBoxLayout()
        self.sam3_track_forward_btn = QPushButton("2. Track Selected to End")
        self.sam3_track_forward_btn.clicked.connect(self.sam3_track_forward)
        describe(
            self.sam3_track_forward_btn,
            "Select one polygon in the annotation list, then predict its mask on "
            "later loaded frames. Tracking stops after two consecutive misses.",
        )
        self.sam3_track_all_btn = QPushButton("Track All to End")
        self.sam3_track_all_btn.clicked.connect(
            lambda: self.sam3_track_forward(all_objects=True)
        )
        describe(
            self.sam3_track_all_btn,
            "Predict every valid polygon on this frame across later loaded frames. "
            "Existing manual annotations are preserved.",
        )
        sam3_buttons_layout.addWidget(self.sam3_track_forward_btn)
        sam3_buttons_layout.addWidget(self.sam3_track_all_btn)
        sam3_layout.addLayout(sam3_buttons_layout)
        sam3_layout.addWidget(
            help_text(
                "Start with a clean polygon on the current frame. SAM 3 follows it "
                "forward until the sequence ends or the object is missed twice. "
                "Correct mistakes in Labeling before export."
            )
        )
        ai_layout.addWidget(sam3_group)

        dino_group, dino_layout = group("Advanced: find objects from text")

        self.dino_model_selector = QComboBox()
        self.dino_model_selector.addItem("Pick a DINO Model")
        self.dino_model_selector.addItem("grounding-dino-base")
        self.dino_model_selector.addItem("grounding-dino-tiny")
        self.dino_model_selector.addItem("Custom / fine-tuned (browse)")
        self.dino_model_selector.currentTextChanged.connect(self._on_dino_model_changed)
        dino_layout.addWidget(self.dino_model_selector)

        # Custom model browse row (hidden by default)
        self.dino_browse_row = QWidget()
        dino_browse_layout = QHBoxLayout(self.dino_browse_row)
        dino_browse_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_dino_custom = QLabel("No path set")
        self.lbl_dino_custom.setWordWrap(True)
        # Themed via the "help-text" class rather than a hardcoded
        # "color:#555", which was invisible against the dark sidebar.
        self.lbl_dino_custom.setProperty("class", "help-text")
        btn_dino_browse = QPushButton("Browse")
        btn_dino_browse.setMinimumWidth(1)
        btn_dino_browse.clicked.connect(self.browse_dino_model)
        dino_browse_layout.addWidget(self.lbl_dino_custom, 1)
        dino_browse_layout.addWidget(btn_dino_browse)
        self.dino_browse_row.setVisible(False)
        dino_layout.addWidget(self.dino_browse_row)

        self.lbl_dino_status = QLabel("No DINO model loaded")
        self.lbl_dino_status.setWordWrap(True)
        self.lbl_dino_status.setProperty("cardRole", "status-idle")
        dino_layout.addWidget(self.lbl_dino_status)

        self.dino_class_table = ClassThresholdTable()
        self.dino_class_table.itemSelectionChanged.connect(self.on_dino_class_row_changed)
        dino_layout.addWidget(self.dino_class_table)

        self.dino_phrase_panel = PhraseEditorPanel()
        dino_layout.addWidget(self.dino_phrase_panel)

        self.btn_detect_single = QPushButton("Detect Current Image")
        self.btn_detect_single.clicked.connect(self.run_dino_detection_single)
        self.btn_detect_single.setEnabled(False)

        self.btn_detect_batch = QPushButton("Detect All Images")
        self.btn_detect_batch.clicked.connect(self.run_dino_detection_batch)
        self.btn_detect_batch.setEnabled(False)
        dino_layout.addLayout(
            side_by_side(self.btn_detect_single, self.btn_detect_batch)
        )

        # Batch mode
        self.dino_batch_mode = QComboBox()
        self.dino_batch_mode.addItem("Review before accepting")
        self.dino_batch_mode.addItem("Auto-accept all detections")
        dino_layout.addWidget(self.dino_batch_mode)
        ai_layout.addWidget(dino_group)
        ai_layout.addStretch(1)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)
        self.tool_group.addButton(self.polygon_button)
        self.tool_group.addButton(self.rectangle_button)
        self.tool_group.addButton(self.paint_brush_button)
        self.tool_group.addButton(self.eraser_button)
        self.tool_group.addButton(self.sam_box_button)
        self.tool_group.addButton(self.sam_points_button)

        self.polygon_button.clicked.connect(self.toggle_tool)
        self.rectangle_button.clicked.connect(self.toggle_tool)
        self.paint_brush_button.clicked.connect(self.toggle_tool)
        self.eraser_button.clicked.connect(self.toggle_tool)

    def update_display_adjustments(self):
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()
        self.brightness_value_label.setText(f"{brightness:+d}")
        self.contrast_value_label.setText(f"{contrast:+d}")
        self.image_label.set_display_adjustments(brightness, contrast)

    def reset_display_adjustments(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.update_display_adjustments()

    def toggle_sam_box(self):
        if self.sam_box_button.isChecked():
            self.sam_points_button.setChecked(False)
            self.image_label.current_tool = "sam_box"
            self.image_label.sam_box_active = True
            self.image_label.sam_points_active = False
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.image_label.current_tool = None
            self.image_label.sam_box_active = False
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
        self.update_ui_for_current_tool()

    def toggle_sam_points(self):
        if self.sam_points_button.isChecked():
            self.sam_box_button.setChecked(False)
            self.image_label.current_tool = "sam_points"
            self.image_label.sam_points_active = True
            self.image_label.sam_box_active = False
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.image_label.sam_positive_points = []
            self.image_label.sam_negative_points = []
        else:
            self.sam_inference_timer.stop()
            self.image_label.current_tool = None
            self.image_label.sam_points_active = False
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.image_label.sam_positive_points = []
            self.image_label.sam_negative_points = []
        self.update_ui_for_current_tool()

    def sort_annotations_by_class(self):
        current_name = self.current_slice or self.image_file_name
        if current_name not in self.all_annotations:
            QMessageBox.information(
                self,
                "No Annotations",
                "There are no annotations to sort for this image.",
            )
            return

        annotations = self.all_annotations[current_name]
        sorted_annotations = []
        for class_name in sorted(annotations.keys()):
            if not class_name.startswith("Temp-"):  # Skip temporary classes
                class_annotations = sorted(
                    annotations[class_name], key=lambda x: x.get("number", 0)
                )
                sorted_annotations.extend(class_annotations)

        self.update_annotation_list_with_sorted(sorted_annotations)

    def sort_annotations_by_area(self):
        current_name = self.current_slice or self.image_file_name
        if current_name not in self.all_annotations:
            QMessageBox.information(
                self,
                "No Annotations",
                "There are no annotations to sort for this image.",
            )
            return

        annotations = self.all_annotations[current_name]
        sorted_annotations = []
        for class_name in annotations.keys():
            if not class_name.startswith("Temp-"):  # Skip temporary classes
                class_annotations = sorted(
                    annotations[class_name],
                    key=lambda x: calculate_area(x),
                    reverse=True,
                )
                sorted_annotations.extend(class_annotations)

        self.update_annotation_list_with_sorted(sorted_annotations)

    def update_annotation_list_with_sorted(self, sorted_annotations):
        self.annotation_list.clear()
        for annotation in sorted_annotations:
            class_name = annotation["category_name"]
            if not class_name.startswith("Temp-"):  # Only add non-temporary annotations
                number = annotation.get("number", 0)
                area = calculate_area(annotation)
                item_text = f"{class_name} - {number:<3} Area: {area:.2f}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, annotation)
                color = self.image_label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
                item.setForeground(color)
                self.annotation_list.addItem(item)

        self.image_label.update()

    def change_sam_model(self, model_name):
        try:
            self.sam_utils.change_sam_model(model_name)
        except Exception as e:
            QMessageBox.critical(
                self,
                "SAM Model Error",
                f"Failed to load SAM model '{model_name}':\n\n{str(e)}\n\n"
                "Check that the model weights are downloadable and that torch "
                "is correctly installed for your platform / GPU."
            )
            self.sam_model_selector.setCurrentIndex(0)
            return

        self.current_sam_model = self.sam_utils.current_sam_model

        if model_name != "Pick a SAM Model":
            # Enable the SAM Magic Wand button
            self.sam_magic_wand_button.setEnabled(True)

            # Activate the SAM Magic Wand tool
            self.sam_magic_wand_button.setChecked(True)
            self.activate_sam_magic_wand()

            print(f"Changed SAM model to: {model_name}")
        else:
            # Disable and deactivate the SAM Magic Wand button
            self.sam_magic_wand_button.setEnabled(False)
            self.sam_magic_wand_button.setChecked(False)
            self.deactivate_sam_magic_wand()
            print("SAM model unset")

    # --- DINO / LLM-Assisted Detection Methods ---

    def _resolve_dino_model_path(self, model_name: str) -> str | None:
        """Return the canonical local path for a preset DINO model, or None if unknown."""
        from .dino_utils import GDINO_MODEL_PATHS
        # GDINO_MODEL_PATHS now returns absolute paths from models_base_dir().
        return GDINO_MODEL_PATHS.get(model_name)

    def _on_dino_model_changed(self, text):
        """Selection → ready state. Downloads happen lazily on first Detect."""
        self.dino_browse_row.setVisible(text == "Custom / fine-tuned (browse)")

        if text == "Pick a DINO Model":
            self.dino_model_loaded = False
            self.lbl_dino_status.setText("No DINO model loaded")
            self.btn_detect_single.setEnabled(False)
            self.btn_detect_batch.setEnabled(False)
            return

        if text == "Custom / fine-tuned (browse)":
            if self.dino_custom_model_path and os.path.exists(self.dino_custom_model_path):
                self.dino_model_loaded = True
                self.lbl_dino_status.setText(
                    f"Ready: {os.path.basename(self.dino_custom_model_path)}"
                )
                self.btn_detect_single.setEnabled(True)
                self.btn_detect_batch.setEnabled(True)
            else:
                self.dino_model_loaded = False
                self.lbl_dino_status.setText("Browse for a custom model folder")
                self.btn_detect_single.setEnabled(False)
                self.btn_detect_batch.setEnabled(False)
            return

        # Standard preset (grounding-dino-base/tiny)
        self.dino_model_loaded = True
        self.btn_detect_single.setEnabled(True)
        self.btn_detect_batch.setEnabled(True)
        model_path = self._resolve_dino_model_path(text)
        if model_path and os.path.exists(model_path):
            self.lbl_dino_status.setText(f"Ready: {text}")
        else:
            self.lbl_dino_status.setText(f"{text} — will download on first detection")

    def _ensure_dino_model_downloaded(self, model_name: str) -> bool:
        """If the preset model isn't on disk yet, download it. Returns success."""
        if model_name in ("Pick a DINO Model", "Custom / fine-tuned (browse)"):
            return True  # Custom path is validated elsewhere; no download for it.
        model_path = self._resolve_dino_model_path(model_name)
        if model_path and os.path.exists(model_path):
            return True

        # huggingface_hub is the only way to fetch the weights. Surface the
        # actionable install hint if it's missing rather than the generic
        # "Could not download" message.
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                f"Cannot download {model_name}: the huggingface_hub package "
                "is not installed.\n\nRun:\n    pip install huggingface_hub",
            )
            return False

        self.lbl_dino_status.setText(f"Downloading {model_name}...")
        QApplication.processEvents()
        try:
            downloaded = self.dino_utils.download_model(model_name)
        except Exception as e:
            QMessageBox.critical(self, "Download Failed", f"{model_name}:\n{e}")
            return False
        if not downloaded:
            QMessageBox.critical(
                self, "Download Failed",
                f"Could not download {model_name} from Hugging Face Hub.",
            )
            return False
        return True

    def browse_dino_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select DINO Model Folder")
        if path:
            self.dino_custom_model_path = path
            self.lbl_dino_custom.setText(os.path.basename(path))
            # Refresh ready state now that a path is set.
            self._on_dino_model_changed(self.dino_model_selector.currentText())

    def on_dino_class_row_changed(self):
        name = self.dino_class_table.selected_class_name()
        self.dino_phrase_panel.set_active_class(name)

    def _build_dino_class_configs(self) -> list[dict]:
        """Build class_configs from threshold table + phrase panel."""
        configs = []
        for cfg in self.dino_class_table.get_class_configs():
            phrases = self.dino_phrase_panel.get_phrases_for(cfg["name"])
            configs.append({
                "name": cfg["name"],
                "phrases": phrases,
                "box_thr": cfg["box_thr"],
                "txt_thr": cfg["txt_thr"],
                "nms_thr": cfg["nms_thr"],
            })
        return configs

    def run_dino_detection_single(self):
        if not self.dino_model_loaded:
            QMessageBox.warning(self, "No DINO Model",
                                "Please pick a DINO model first.")
            return
        if not self.sam_utils.current_sam_model:
            QMessageBox.warning(
                self, "No SAM Model",
                "DINO produces bounding boxes; SAM is needed to convert them "
                "into segmentation masks. Please pick a SAM model first.",
            )
            return
        if not self.current_image or self.current_image.isNull():
            QMessageBox.warning(self, "No Image",
                                "Please load an image first.")
            return

        model_name = self.dino_model_selector.currentText()
        class_configs = self._build_dino_class_configs()
        if not class_configs:
            QMessageBox.warning(self, "No Classes",
                                "Please add at least one class with phrases.")
            return

        self.btn_detect_single.setEnabled(False)
        self.btn_detect_batch.setEnabled(False)

        if not self._ensure_dino_model_downloaded(model_name):
            self.btn_detect_single.setEnabled(True)
            self.btn_detect_batch.setEnabled(True)
            return

        self.lbl_dino_status.setText("Detecting...")
        QApplication.processEvents()

        print(f"[DINO] detect_single: model={model_name!r} class_configs={class_configs}")
        try:
            results = self.dino_utils.detect(
                self.current_image, class_configs,
                model_name=model_name,
                custom_model_path=self.dino_custom_model_path,
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "DINO Error", str(e))
            self.btn_detect_single.setEnabled(True)
            self.btn_detect_batch.setEnabled(True)
            self.lbl_dino_status.setText("Detection failed.")
            return

        self.btn_detect_single.setEnabled(True)
        self.btn_detect_batch.setEnabled(True)

        if results is None:
            print("[DINO] detect_single: results=None (model resolution failure)")
            self.lbl_dino_status.setText("No detections.")
            return

        print(f"[DINO] detect_single: got {len(results)} result(s)")
        if results:
            for i, r in enumerate(results[:3]):
                print(f"[DINO]   result[{i}] class={r['class_name']!r} score={r['score']:.3f} bbox={r['bbox']}")

        if not results:
            self.lbl_dino_status.setText("No detections found.")
            return

        self.lbl_dino_status.setText(f"{len(results)} detection(s). Running SAM...")
        QApplication.processEvents()

        # Batch SAM segmentation. Wrap in try/except for the same reason
        # as the DINO call above — sam_utils raises on model load
        # failure / CUDA OOM / re-entry now, instead of returning None.
        bboxes = [r["bbox"] for r in results]
        print(f"[SAM] batch call: {len(bboxes)} bbox(es), first 3 = {bboxes[:3]}")
        try:
            sam_results = self.sam_utils.apply_sam_predictions_batch(
                self.current_image, bboxes
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "SAM Error", str(e))
            self.lbl_dino_status.setText("SAM segmentation failed.")
            return

        if sam_results is None:
            print("[SAM] batch returned None (no SAM model loaded)")
            QMessageBox.warning(self, "SAM Error",
                                "Failed to segment detections with SAM.")
            self.lbl_dino_status.setText("SAM segmentation failed.")
            return

        n_errors = sum(1 for s in sam_results if "error" in s)
        n_ok = sum(1 for s in sam_results if "segmentation" in s)
        print(f"[SAM] batch returned {len(sam_results)} result(s): {n_ok} ok, {n_errors} error(s)")

        # Honor the batch-mode dropdown for the single-image case too:
        # "Auto-accept" means commit straight to annotations without
        # showing the temp-review overlay. The dropdown name is "batch"
        # historically but it controls both paths.
        image_name = self.current_slice or self.image_file_name
        auto_accept = (
            self.dino_batch_mode.currentText() == "Auto-accept all detections"
        )
        if auto_accept:
            self._commit_dino_results(image_name, results, sam_results)
            n_committed = sum(1 for s in sam_results if "error" not in s)
            self.image_label.temp_annotations = []
            self.image_label.update()
            self.update_annotation_list()
            # Refresh slice list so the freshly-annotated slice picks
            # up the highlight color; review-mode's accept_dino_results
            # already does this, the auto-accept path didn't.
            self.update_slice_list_colors()
            self.auto_save()
            self.lbl_dino_status.setText(
                f"Loaded: {model_name}  |  {n_committed} mask(s) auto-accepted"
            )
            print(f"[DINO] auto-accept: committed {n_committed} mask(s) to {image_name}")
            return

        # Review mode — build temp annotations and let user accept/reject
        temp_annotations = []
        for r, s in zip(results, sam_results):
            if "error" in s:
                print(f"[SAM]   failed for {r['class_name']}: {s['error']}")
                continue
            temp_annotations.append({
                "segmentation": s["segmentation"],
                "category_name": r["class_name"],
                "score": r["score"],
                "source": "dino",
                "temp": True,
            })

        self.image_label.temp_annotations = temp_annotations
        # Defer setFocus until after the click event chain settles —
        # synchronous setFocus often loses to whatever widget is still
        # processing the original click.
        QTimer.singleShot(0, self.image_label.setFocus)
        self.image_label.update()
        self.lbl_dino_status.setText(
            f"Loaded: {model_name}  |  {len(temp_annotations)} mask(s) ready"
        )
        print(f"[DINO] detection complete: {len(results)} boxes, {len(temp_annotations)} masks attached to canvas")

    def run_dino_detection_batch(self):
        if not self.dino_model_loaded:
            QMessageBox.warning(self, "No DINO Model",
                                "Please pick a DINO model first.")
            return
        if not self.sam_utils.current_sam_model:
            QMessageBox.warning(
                self, "No SAM Model",
                "DINO produces bounding boxes; SAM is needed to convert them "
                "into segmentation masks. Please pick a SAM model first.",
            )
            return
        if not self.all_images:
            QMessageBox.warning(self, "No Images",
                                "Please load images first.")
            return

        model_name = self.dino_model_selector.currentText()
        class_configs = self._build_dino_class_configs()
        if not class_configs:
            QMessageBox.warning(self, "No Classes",
                                "Please add at least one class with phrases.")
            return

        if not self._ensure_dino_model_downloaded(model_name):
            return

        auto_accept = self.dino_batch_mode.currentText() == "Auto-accept all detections"

        # Build a flat list of (display_name, qimage) work items covering
        # both regular images (loaded from disk) and multi-dim image
        # slices (already QImages in memory). Slices live in
        # self.image_slices[base_name], indexed by their slice_name
        # (e.g. "stack_T1_Z1_C1"). The earlier implementation only
        # iterated self.all_images and skipped multi-slice entries with
        # a console warning, leaving slice-based projects unable to use
        # Detect All.
        work_items = self._collect_dino_batch_work_items()
        if not work_items:
            QMessageBox.information(
                self, "Detect All Images",
                "No images or slices available to process."
            )
            return
        total = len(work_items)

        progress = QProgressDialog("Running LLM Detection...", "Cancel", 0, total, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        for idx, (image_name, qimage) in enumerate(work_items):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            QApplication.processEvents()

            try:
                results = self.dino_utils.detect(
                    qimage, class_configs,
                    model_name=model_name,
                    custom_model_path=self.dino_custom_model_path,
                )
            except Exception as e:
                print(f"  DINO failed for {image_name}: {e}")
                continue

            if not results:
                continue

            bboxes = [r["bbox"] for r in results]
            try:
                sam_results = self.sam_utils.apply_sam_predictions_batch(qimage, bboxes)
            except Exception as e:
                print(f"  SAM failed for {image_name}: {e}")
                continue
            if sam_results is None:
                continue

            if auto_accept:
                self._commit_dino_results(image_name, results, sam_results)
            else:
                # Store for later review
                self._store_dino_batch_results(image_name, results, sam_results)

        progress.setValue(total)
        progress.close()

        if auto_accept:
            QMessageBox.information(
                self, "Batch Detection Complete",
                "Detections have been saved to annotations."
            )
            self.update_annotation_list()
            # Multi-dim stacks commonly auto-accept across dozens of
            # slices; the slice list must show which ones gained
            # annotations or the user can't tell what happened.
            self.update_slice_list_colors()
            self.auto_save()
        else:
            self._show_dino_batch_review()

    def _collect_dino_batch_work_items(self):
        """Return a flat ``[(name, QImage), …]`` list for batch DINO.

        Regular images are loaded from disk via PIL → QImage. Multi-dim
        images contribute one entry per slice from ``self.image_slices``;
        slices that haven't been materialised yet (the parent image was
        never opened in this session) are skipped with a console log.
        """
        from PIL import Image as PILImage
        items = []
        for img_info in self.all_images:
            file_name = img_info["file_name"]
            if img_info.get("is_multi_slice", False):
                base_name = os.path.splitext(file_name)[0]
                slices = self.image_slices.get(base_name, [])
                if not slices:
                    print(f"  Skipping multi-slice image '{file_name}': "
                          "no slices loaded (open the image first to "
                          "materialise its slices).")
                    continue
                for slice_name, qimage in slices:
                    items.append((slice_name, qimage))
            else:
                image_path = self.image_paths.get(file_name)
                if not image_path or not os.path.exists(image_path):
                    print(f"  Skipping '{file_name}': missing image path.")
                    continue
                try:
                    pil_img = PILImage.open(image_path).convert("RGB")
                    qimage = QImage(
                        pil_img.tobytes(),
                        pil_img.width,
                        pil_img.height,
                        pil_img.width * 3,
                        QImage.Format.Format_RGB888,
                    )
                    items.append((file_name, qimage))
                except Exception as e:
                    print(f"  Skipping '{file_name}': failed to load ({e}).")
        print(f"[DINO] batch work items: {len(items)} total")
        return items

    def _commit_dino_results(self, image_name, dino_results, sam_results):
        """Commit DINO+SAM results to annotations for a single image.

        If image_name is the currently-displayed image, route through
        image_label.annotations so the canvas reflects the change and the
        next save_current_annotations() doesn't overwrite the additions.
        Otherwise write directly to the project-level cache.
        """
        current_image = self.current_slice or self.image_file_name
        is_current = image_name == current_image

        # Snapshot the frame this write lands on, not the frame on screen:
        # the batch path commits into images the annotator is not looking
        # at, and undo is keyed by frame.
        self.record_annotation_history("accepting detections", image_name)

        if is_current:
            target = self.image_label.annotations
        else:
            if image_name not in self.all_annotations:
                self.all_annotations[image_name] = {}
            target = self.all_annotations[image_name]

        for r, s in zip(dino_results, sam_results):
            if "error" in s:
                continue
            class_name = r["class_name"]
            # DINO only returns labels that came from class_configs (which the
            # parent built from the class table), so this should never trigger.
            # Skip with a warning rather than auto-creating a class mid-batch
            # (which would fan out auto_save() per new class).
            if class_name not in self.class_mapping:
                print(f"  Skipping DINO result for unknown class '{class_name}'")
                continue
            existing = target.get(class_name, [])
            number = max((a.get("number", 0) for a in existing), default=0) + 1
            ann = {
                "segmentation": s["segmentation"],
                "category_id": self.class_mapping[class_name],
                "category_name": class_name,
                "score": r["score"],
                "source": "dino",
                "number": number,
            }
            target.setdefault(class_name, []).append(ann)

        if is_current:
            # Sync image_label.annotations -> all_annotations[current] for save.
            self.save_current_annotations()
            self.image_label.update()

    def _store_dino_batch_results(self, image_name, dino_results, sam_results):
        """Store results for batch review mode."""
        valid = []
        for r, s in zip(dino_results, sam_results):
            if "error" not in s:
                valid.append({
                    "segmentation": s["segmentation"],
                    "category_name": r["class_name"],
                    "score": r["score"],
                    "source": "dino",
                    "temp": True,
                })
        self.dino_batch_results[image_name] = valid

    def _show_dino_batch_review(self):
        """Navigate to first image with batch results for review.

        If the next entry refers to an image/slice that's no longer in
        the project (e.g. the source was removed between detection and
        review), pop the orphan and try the next entry so the user
        doesn't get stuck with un-reviewable results.
        """
        if not self.dino_batch_results:
            QMessageBox.information(self, "Batch Detection",
                                    "No detections found in any image.")
            return
        # Drain orphans up front. Navigate to the entry: it may be a
        # regular image (key in image_list) or a slice (key in some
        # image_slices[base_name]). _navigate_to_image_or_slice handles
        # both. After the switch, switch_image / switch_slice's tail
        # call to _refresh_dino_temp_for_current copies
        # dino_batch_results[first] into image_label.temp_annotations
        # and defers setFocus on the canvas — nothing to repeat here.
        while self.dino_batch_results:
            first = next(iter(self.dino_batch_results))
            if self._navigate_to_image_or_slice(first):
                return
            print(f"[DINO] dropping orphan batch result for {first!r} "
                  "(no matching image or slice in project)")
            self.dino_batch_results.pop(first, None)
        # Drained all entries without a single navigable target.
        QMessageBox.warning(
            self, "Batch Detection",
            "Detections were produced but none of them map to an image "
            "or slice still in the project. Results discarded.",
        )

    def _navigate_to_image_or_slice(self, name: str) -> bool:
        """Switch the UI to a regular image or a slice by name.

        Returns True if a match was found and the switch was issued.
        Used by batch-review navigation, which mixes regular image
        names and slice names in ``dino_batch_results``.
        """
        # Regular image — match in image_list directly
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item and item.text() == name:
                if self.image_list.currentRow() == i:
                    self.switch_image(item)
                else:
                    self.image_list.setCurrentRow(i)
                return True
        # Slice — find which multi-dim image contains it, switch to
        # that parent image first, then activate the specific slice
        # via slice_list.
        for base_name, slices in self.image_slices.items():
            if not any(s_name == name for s_name, _ in slices):
                continue
            # Find the parent file in image_list. The file_name in the
            # list includes the extension (e.g. "stack.tif") while
            # base_name is the stem ("stack"), so match by stripping
            # the extension and comparing for equality.
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if not item:
                    continue
                file_name = item.text()
                if os.path.splitext(file_name)[0] == base_name:
                    self.image_list.setCurrentRow(i)
                    self.switch_image(item)
                    # switch_image populates slice_list. Now find the slice.
                    for s_i in range(self.slice_list.count()):
                        s_item = self.slice_list.item(s_i)
                        if s_item and s_item.text() == name:
                            self.slice_list.setCurrentRow(s_i)
                            self.switch_slice(s_item)
                            return True
                    break
            return False
        return False

    def _refresh_dino_temp_for_current(self):
        """Sync ``image_label.temp_annotations`` to whatever the
        currently-displayed image/slice has stored in
        ``dino_batch_results``. Called from switch_slice / switch_image.

        Why this exists: ``temp_annotations`` is a single field on
        ``ImageLabel``, not a per-image cache. Without this sync, masks
        from the previously-viewed image bleed onto every slice the
        user navigates to. During a batch review the user expects each
        image to show its own pending detections; outside batch review,
        switching simply discards the pending overlay.
        """
        new_image = self.current_slice or self.image_file_name
        pending = self.dino_batch_results.get(new_image, []) if new_image else []
        if pending:
            # Re-stamp the "temp" flag in case it was stripped by a
            # previous accept path; this list also feeds the paintEvent
            # which expects dicts with "segmentation" + "category_name".
            self.image_label.temp_annotations = list(pending)
            self.lbl_dino_status.setText(
                f"Review: {new_image}  ({len(pending)} detection(s))"
            )
            QTimer.singleShot(0, self.image_label.setFocus)
        else:
            if self.image_label.temp_annotations:
                print("[DINO] temp annotations cleared on switch "
                      f"(no pending batch results for {new_image!r})")
            self.image_label.temp_annotations = []
        self.image_label.update()

    def accept_dino_results(self):
        """Accept current temp_annotations (called from keyPressEvent)."""
        if not self.image_label.temp_annotations:
            return
        image_name = self.current_slice or self.image_file_name
        self.record_annotation_history(
            f"accepting {len(self.image_label.temp_annotations)} detection(s)",
            image_name,
        )

        for ann in self.image_label.temp_annotations:
            class_name = ann["category_name"]
            # DINO only returns labels from class_configs (built from the
            # class table), so unknown classes should never reach this point.
            # Skip with a warning rather than auto-creating mid-accept.
            if class_name not in self.class_mapping:
                print(f"  Skipping DINO result for unknown class '{class_name}'")
                continue
            new_ann = {
                "segmentation": ann["segmentation"],
                "category_id": self.class_mapping[class_name],
                "category_name": class_name,
                "score": ann.get("score", 0.0),
                "source": "dino",
            }
            # Append to the live image_label dict; save_current_annotations()
            # below syncs it into self.all_annotations. add_annotation_to_list
            # assigns the per-class "number" used for display.
            self.image_label.annotations.setdefault(class_name, []).append(new_ann)
            self.add_annotation_to_list(new_ann)

        self.image_label.temp_annotations = []
        # Clear batch results if reviewing
        self.dino_batch_results.pop(image_name, None)
        if self.dino_batch_results:
            self._show_dino_batch_review()
        self.save_current_annotations()
        self.update_slice_list_colors()
        self.image_label.update()
        self.lbl_dino_status.setText("Results accepted.")
        print("DINO results accepted.")

    def reject_dino_results(self):
        """Discard current temp_annotations."""
        self.image_label.temp_annotations = []
        image_name = self.current_slice or self.image_file_name
        self.dino_batch_results.pop(image_name, None)
        if self.dino_batch_results:
            self._show_dino_batch_review()
        self.image_label.update()
        self.lbl_dino_status.setText("Results discarded.")
        print("DINO results discarded.")

    # --- END DINO Methods ---

    def setup_font_size_selector(self):
        font_size_label = QLabel("Font Size:")
        self.font_size_selector = QComboBox()
        self.font_size_selector.addItems(["Small", "Medium", "Large"])
        self.font_size_selector.setCurrentText("Medium")
        self.font_size_selector.currentTextChanged.connect(self.on_font_size_changed)

        self.sidebar_layout.addWidget(font_size_label)
        self.sidebar_layout.addWidget(self.font_size_selector)

    def on_font_size_changed(self, size):
        self.current_font_size = size
        self.apply_theme_and_font()

    def apply_theme_and_font(self):
        font_size = self.font_sizes[self.current_font_size]
        if self.dark_mode:
            style = soft_dark_stylesheet
        else:
            style = default_stylesheet

        # Combine the theme stylesheet with font size
        combined_style = f"{style}\nQWidget {{ font-size: {font_size}pt; }}"
        self.setStyleSheet(combined_style)

        # Apply font size to all widgets
        for widget in self.findChildren(QWidget):
            font = widget.font()
            font.setPointSize(font_size)
            widget.setFont(font)

        self.image_label.setFont(QFont("Arial", font_size))
        self.update()

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme_and_font()

        # Update slice list colors
        self.update_slice_list_colors()

        # Update other UI elements if necessary
        self.update_class_list()
        self.update_annotation_list()

        # Force a repaint of the main window
        self.repaint()

    def apply_stylesheet(self):
        if self.dark_mode:
            self.setStyleSheet(soft_dark_stylesheet)
        else:
            self.setStyleSheet(default_stylesheet)

    def update_ui_colors(self):
        # Update colors for elements that need to retain their functionality
        self.update_annotation_list_colors()
        self.update_slice_list_colors()
        self.image_label.update()

    def setup_image_area(self):
        """Set up the central annotation canvas."""
        self.image_widget = QWidget()
        self.image_widget.setObjectName("canvasPanel")
        self.image_widget.setMinimumWidth(320)
        self.image_layout = QVBoxLayout(self.image_widget)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_layout.setSpacing(0)
        self.main_splitter.addWidget(self.image_widget)

        canvas_header = QWidget()
        canvas_header.setObjectName("canvasHeader")
        canvas_header_layout = QHBoxLayout(canvas_header)
        canvas_header_layout.setContentsMargins(12, 7, 12, 7)
        canvas_header_layout.setSpacing(10)
        self.canvas_file_label = QLabel("No frame loaded")
        self.canvas_file_label.setProperty("class", "canvas-file")
        # Frame position sits next to the file name because "where am I in
        # the clip" is the question an annotator asks most often.
        self.canvas_position_label = QLabel("")
        self.canvas_position_label.setProperty("class", "mono")
        shortcut_hint = QLabel("A / D  previous / next frame")
        shortcut_hint.setProperty("class", "shortcut-pill")
        canvas_header_layout.addWidget(self.canvas_file_label)
        canvas_header_layout.addWidget(self.canvas_position_label)
        canvas_header_layout.addStretch(1)
        canvas_header_layout.addWidget(shortcut_hint)
        self.image_layout.addWidget(canvas_header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("canvasViewport")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        # With nothing loaded the canvas used to be an unexplained black
        # rectangle taking two thirds of the window. The placeholder says
        # what to do instead, and swaps out the moment a frame is open.
        self.canvas_stack = QStackedWidget()
        self.canvas_stack.addWidget(self._build_canvas_placeholder())
        self.canvas_stack.addWidget(self.scroll_area)
        self.image_layout.addWidget(self.canvas_stack, 1)

        canvas_footer = QWidget()
        canvas_footer.setObjectName("canvasFooter")
        footer_layout = QHBoxLayout(canvas_footer)
        footer_layout.setContentsMargins(12, 5, 12, 5)
        footer_layout.setSpacing(8)

        self.zoom_fit_button = QPushButton("Fit")
        self.zoom_fit_button.setProperty("buttonRole", "ghost")
        self.zoom_fit_button.setToolTip("Scale the frame to fit the viewport.")
        self.zoom_fit_button.clicked.connect(self.adjust_zoom_to_fit)
        self.zoom_actual_button = QPushButton("1:1")
        self.zoom_actual_button.setProperty("buttonRole", "ghost")
        self.zoom_actual_button.setToolTip(
            "Show the frame at one screen pixel per image pixel."
        )
        self.zoom_actual_button.clicked.connect(lambda: self.zoom_slider.setValue(100))

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(220)
        self.zoom_slider.valueChanged.connect(self.zoom_image)
        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setObjectName("zoomValueLabel")
        self.zoom_value_label.setProperty("class", "mono")
        self.zoom_value_label.setMinimumWidth(42)
        self.zoom_slider.valueChanged.connect(
            lambda value: self.zoom_value_label.setText(f"{value}%")
        )
        self.image_info_label = QLabel()
        self.image_info_label.setProperty("class", "mono")

        footer_layout.addWidget(self.zoom_fit_button)
        footer_layout.addWidget(self.zoom_actual_button)
        footer_layout.addWidget(self.zoom_slider)
        footer_layout.addWidget(self.zoom_value_label)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.image_info_label)
        self.image_layout.addWidget(canvas_footer)

    def _build_canvas_placeholder(self):
        """What the canvas shows before any frame is open."""
        placeholder = QWidget()
        placeholder.setObjectName("canvasPlaceholder")
        outer = QVBoxLayout(placeholder)
        outer.setContentsMargins(40, 40, 40, 40)
        outer.addStretch(1)

        # The content sits in a fixed-width column. A word-wrapped label
        # placed straight into a stretch-padded layout has no definite
        # width to wrap against and collapses on top of its neighbour.
        column = QWidget()
        column.setObjectName("canvasPlaceholderColumn")
        # Maximum, not fixed: QStackedWidget takes the widest page as its
        # minimum size hint, so a fixed width here would put a permanent
        # floor under the whole window even once a frame is loaded and
        # this page is never shown again.
        column.setMaximumWidth(480)
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(10)

        heading = QLabel("No frames loaded")
        heading.setProperty("class", "section-header")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column_layout.addWidget(heading)

        body = QLabel(
            "Pull a frame range out of a recording, or add stills that are "
            "already on disk. Save the project first and labels are written "
            "to the .iap file as you work."
        )
        body.setProperty("class", "help-text")
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.setWordWrap(True)
        column_layout.addWidget(body)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        placeholder_video = QPushButton("Open Video Clip...")
        placeholder_video.setProperty("buttonRole", "primary")
        placeholder_video.clicked.connect(self.open_video_clip)
        placeholder_images = QPushButton("Add New Images")
        placeholder_images.clicked.connect(self.add_images)
        placeholder_folder = QPushButton("Open Frame Folder...")
        placeholder_folder.clicked.connect(self.open_frame_folder)
        for button in (placeholder_video, placeholder_images, placeholder_folder):
            button.setMinimumWidth(1)
            actions.addWidget(button, 1)
        column_layout.addSpacing(4)
        column_layout.addLayout(actions)

        hint = QLabel("Ctrl+/  lists every keyboard shortcut")
        hint.setProperty("class", "shortcut-pill")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column_layout.addSpacing(10)
        column_layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignHCenter)

        outer.addWidget(column, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)
        return placeholder

    def _update_canvas_placeholder(self):
        """Show the canvas only once there is something to draw on."""
        if not hasattr(self, "canvas_stack"):
            return
        self.canvas_stack.setCurrentIndex(1 if self.current_image else 0)

    def setup_image_list(self):
        """Set up the frame navigator."""
        self.image_list_widget = QWidget()
        self.image_list_widget.setObjectName("framesPanel")
        self.image_list_widget.setMinimumWidth(220)
        self.image_list_widget.setMaximumWidth(460)
        self.image_list_layout = QVBoxLayout(self.image_list_widget)
        self.image_list_layout.setContentsMargins(10, 10, 10, 10)
        self.image_list_layout.setSpacing(6)
        self.main_splitter.addWidget(self.image_list_widget)

        frames_heading = QLabel("FRAMES")
        frames_heading.setProperty("class", "eyebrow")
        self.frame_count_label = QLabel("0 loaded")
        self.frame_count_label.setProperty("class", "panel-count")
        heading_row = QHBoxLayout()
        heading_row.addWidget(frames_heading)
        heading_row.addStretch(1)
        heading_row.addWidget(self.frame_count_label)
        self.image_list_layout.addLayout(heading_row)

        # Labeling a clip is a long job, so the panel reports how much of
        # it is done. Without this the only way to tell was to click every
        # frame in turn.
        self.frame_progress = QProgressBar()
        self.frame_progress.setObjectName("frameProgress")
        self.frame_progress.setTextVisible(False)
        self.frame_progress.setRange(0, 100)
        self.frame_progress.setValue(0)
        self.image_list_layout.addWidget(self.frame_progress)

        self.frame_progress_label = QLabel("No frames loaded")
        self.frame_progress_label.setProperty("class", "help-text")
        self.image_list_layout.addWidget(self.frame_progress_label)

        filter_row = QWidget()
        filter_row.setObjectName("framesToolbar")
        filter_layout = QHBoxLayout(filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(6)
        self.frame_filter_edit = QLineEdit()
        self.frame_filter_edit.setPlaceholderText("Filter frames...")
        self.frame_filter_edit.setClearButtonEnabled(True)
        self.frame_filter_edit.setToolTip(
            "Show only frames whose file name contains this text."
        )
        self.frame_filter_edit.textChanged.connect(self.apply_frame_filter)
        # Enter hands focus back to the canvas. Without this the caret
        # stays in the box after filtering, and A / D / P / 1-9 are all
        # correctly treated as typing — so the keys look broken right
        # after the most common reason to use the filter.
        self.frame_filter_edit.returnPressed.connect(self._leave_frame_filter)
        self.unlabeled_only_button = QPushButton("Todo")
        self.unlabeled_only_button.setCheckable(True)
        self.unlabeled_only_button.setProperty("buttonRole", "ghost")
        self.unlabeled_only_button.setToolTip(
            "Show only frames that have no labels yet — the queue of work left."
        )
        self.unlabeled_only_button.toggled.connect(lambda _: self.apply_frame_filter())
        filter_layout.addWidget(self.frame_filter_edit, 1)
        filter_layout.addWidget(self.unlabeled_only_button)
        self.image_list_layout.addWidget(filter_row)

        self.image_list = QListWidget()
        self.image_list.setObjectName("frameList")
        # Frame file names are long and repetitive; elide the middle so the
        # sequence number at the end stays readable and the list never
        # grows a horizontal scrollbar.
        self.image_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.image_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.image_list.setUniformItemSizes(True)
        self.image_list.itemClicked.connect(self.switch_image)
        self.image_list.currentRowChanged.connect(
            lambda row: self.switch_image(self.image_list.currentItem())
        )
        self.image_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self.show_image_context_menu)
        self.image_list.setToolTip(
            "Frames in this project. A filled marker means the frame has labels."
        )
        self.image_list_layout.addWidget(self.image_list, 1)

        self.clear_all_button = QPushButton("Clear Workspace")
        self.clear_all_button.clicked.connect(self.clear_all)
        self.clear_all_button.setProperty("buttonRole", "danger")
        self.image_list_layout.addWidget(self.clear_all_button)

    # ------------------------------------------------------------------
    # Session status: progress, filtering, the status bar and undo.
    # ------------------------------------------------------------------

    def setup_status_bar(self):
        """A single line of live session state along the bottom edge.

        Tool, class, cursor position and save state were previously only
        discoverable by looking at four different places in the sidebar,
        or not at all. Acquisition and annotation software puts them on
        one status line because the annotator's eyes are on the image.
        """
        status = QStatusBar()
        status.setSizeGripEnabled(False)
        self.setStatusBar(status)

        def metric(text="", strong=False):
            label = QLabel(text)
            label.setProperty(
                "class", "status-strong" if strong else "status-metric"
            )
            return label

        def separator():
            label = QLabel("|")
            label.setObjectName("statusSeparator")
            return label

        self.status_tool_label = metric("No tool", strong=True)
        self.status_class_label = metric("No class")
        self.status_cursor_label = metric("x -, y -")
        self.status_brush_label = metric("")
        self.status_save_label = metric("Not saved yet")

        status.addWidget(self.status_tool_label)
        status.addWidget(separator())
        status.addWidget(self.status_class_label)
        status.addWidget(separator())
        status.addWidget(self.status_cursor_label)
        status.addWidget(self.status_brush_label)
        status.addPermanentWidget(self.status_save_label)

    def update_cursor_readout(self, cursor_pos):
        """Update only the coordinate field.

        Called from every mouse-move over the canvas, so it deliberately
        does not rebuild the rest of the status line.
        """
        if not hasattr(self, "status_cursor_label"):
            return
        text = (
            f"x {int(cursor_pos[0]):>5}, y {int(cursor_pos[1]):>5}"
            if cursor_pos
            else "x     -, y     -"
        )
        if text != self.status_cursor_label.text():
            self.status_cursor_label.setText(text)

    def update_status_bar(self, cursor_pos=None):
        """Refresh the status line. Safe to call before the bar exists."""
        if not hasattr(self, "status_tool_label"):
            return

        tool_names = {
            "polygon": "Polygon",
            "rectangle": "Box",
            "paint_brush": "Paint brush",
            "eraser": "Eraser",
            "sam_box": "SAM box prompt",
            "sam_points": "SAM point prompts",
        }
        tool = getattr(self.image_label, "current_tool", None)
        self.status_tool_label.setText(tool_names.get(tool, "No tool"))

        if self.current_class:
            self.status_class_label.setText(f"Class: {self.current_class}")
        else:
            self.status_class_label.setText("No class selected")

        if cursor_pos is None:
            cursor_pos = getattr(self.image_label, "cursor_pos", None)
        self.update_cursor_readout(cursor_pos)

        if tool == "paint_brush":
            self.status_brush_label.setText(f"| brush {self.paint_brush_size} px")
        elif tool == "eraser":
            self.status_brush_label.setText(f"| eraser {self.eraser_size} px")
        else:
            self.status_brush_label.setText("")

    def set_saved_state(self, saved: bool, detail: str = ""):
        """Record whether the project on disk matches what is on screen."""
        if not hasattr(self, "status_save_label"):
            return
        if not hasattr(self, "current_project_file"):
            self.status_save_label.setText("No project — nothing is being saved")
            return
        if saved:
            stamp = datetime.now().strftime("%H:%M:%S")
            self.status_save_label.setText(detail or f"Saved {stamp}")
        else:
            self.status_save_label.setText(detail or "Unsaved changes")

    def frame_has_labels(self, frame_name) -> bool:
        """True when a frame carries at least one committed annotation.

        ``Temp-`` classes are model proposals awaiting review, so a frame
        holding only those is not finished and must not count as done.
        """
        classes = self.all_annotations.get(frame_name) or {}
        return any(
            annotations
            for class_name, annotations in classes.items()
            if not str(class_name).startswith("Temp-")
        )

    def _status_dot(self, color: QColor) -> QIcon:
        """A small filled circle used as the labeled/unlabeled marker.

        Drawn rather than set as an item background: the frame list is
        looked up by ``item.text()`` and ``findItems`` in a dozen places,
        so the marker has to live outside the text, and a full-row colour
        fill fights the theme's own selection colours.
        """
        key = color.name()
        cache = getattr(self, "_status_dot_cache", None)
        if cache is None:
            cache = self._status_dot_cache = {}
        if key not in cache:
            pixmap = QPixmap(10, 10)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(1, 1, 8, 8)
            painter.end()
            cache[key] = QIcon(pixmap)
        return cache[key]

    @contextmanager
    def suspended_progress_refresh(self):
        """Collapse many progress refreshes into one.

        Bulk paths add frames one at a time — ``load_project_data`` calls
        ``add_images_to_list`` once per image, and each of those switches
        image twice. Each refresh walks the whole list, so without this
        the cost of opening a project is quadratic in frame count. One
        refresh runs when the outermost block exits.
        """
        self._progress_refresh_depth += 1
        try:
            yield
        finally:
            self._progress_refresh_depth -= 1
            if self._progress_refresh_depth == 0 and self._progress_refresh_pending:
                self._progress_refresh_pending = False
                # Full refresh, not just the counters: a suspended block
                # may have changed label state, and the markers and filter
                # have to catch up with it.
                self.annotations_changed()

    def _defer_refresh(self) -> bool:
        """Record that a refresh is owed, when one cannot run right now.

        Named as a command because it mutates: callers read it as "should
        I skip, and if so, remember that I did".
        """
        if self._progress_refresh_depth > 0 or self.is_loading_project:
            self._progress_refresh_pending = True
            return True
        return False

    def annotations_changed(self):
        """Call after annotations are added or removed. Full refresh.

        Walks every frame, so this belongs to *mutation* paths only.
        Navigation calls ``refresh_session_status()`` instead, which does
        no per-frame work — switching frames in a 2000-frame clip must
        not pay for a whole-project scan.
        """
        self._update_slice_panel_visibility()
        if self._defer_refresh():
            return
        self._progress_refresh_pending = False
        self.apply_frame_filter()
        self.refresh_frame_progress()

    def refresh_session_status(self):
        """Cheap refresh for navigation: no per-frame work.

        Updates what depends on *which* frame is open, rather than on how
        many frames are labeled.
        """
        self.update_next_step_hint()
        self._sync_history_buttons()
        self._update_slice_panel_visibility()

    def refresh_slice_markers(self):
        """Mark every slice row as labeled or still to do."""
        if not hasattr(self, "slice_list"):
            return
        for index in range(self.slice_list.count()):
            item = self.slice_list.item(index)
            self._apply_labeled_marker(item, item.text())

    def refresh_frame_progress(self):
        """Update the row markers, frame counter, progress bar and hint.

        Markers and counts come from one pass, so a dot can never
        disagree with the number beside it. Rows that already look right
        cost one comparison and no widget calls, which is what keeps the
        pass affordable on a long clip.
        """
        if not hasattr(self, "image_list"):
            return
        if self._defer_refresh():
            return
        self.refresh_slice_markers()

        total = self.image_list.count()
        labeled = 0
        for index in range(total):
            item = self.image_list.item(index)
            name = item.text()
            has_labels = self.frame_has_labels(name)
            if has_labels:
                labeled += 1
            self._apply_labeled_marker(item, name, has_labels)

        if hasattr(self, "frame_progress"):
            self.frame_progress.setMaximum(max(total, 1))
            self.frame_progress.setValue(labeled)
            self.frame_progress.setVisible(total > 0)
        if hasattr(self, "frame_progress_label"):
            if total == 0:
                self.frame_progress_label.setText("No frames loaded")
            else:
                percent = round(100 * labeled / total)
                remaining = total - labeled
                self.frame_progress_label.setText(
                    f"{labeled} of {total} labeled ({percent}%) — "
                    f"{remaining} to go"
                )

        self._update_frame_count_label()
        self.update_next_step_hint()
        self._sync_history_buttons()

    def _leave_frame_filter(self):
        """Return focus to the canvas so the shortcut keys work again."""
        target = None
        for index in range(self.image_list.count()):
            if not self.image_list.item(index).isHidden():
                target = self.image_list.item(index)
                break
        if target is not None and self.image_list.currentItem() is None:
            self.image_list.setCurrentItem(target)
            self.switch_image(target)
        self.image_label.setFocus()

    def apply_frame_filter(self):
        """Hide frame rows that do not match the filter box or Todo toggle.

        Rows are hidden rather than removed so every existing lookup by
        ``findItems`` / ``item(index)`` keeps seeing the full project.
        """
        if not hasattr(self, "image_list") or not hasattr(self, "frame_filter_edit"):
            return
        if self._defer_refresh():
            return
        needle = self.frame_filter_edit.text().strip().lower()
        todo_only = self.unlabeled_only_button.isChecked()
        hidden = 0
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            name = item.text()
            matches = needle in name.lower() if needle else True
            if todo_only and self.frame_has_labels(name):
                matches = False
            # Never hide what is currently open. The list holds stack file
            # names rather than slice names, so this also covers the case
            # where a slice of that stack is the thing on the canvas. It
            # would look as though the open frame had left the project.
            if name == self.image_file_name:
                matches = True
            item.setHidden(not matches)
            hidden += 0 if matches else 1

        self._hidden_frame_count = hidden
        self._update_frame_count_label()

    def _update_frame_count_label(self):
        """Show the plain frame count, or how many survive the filter."""
        if not hasattr(self, "frame_count_label"):
            return
        total = self.image_list.count()
        hidden = min(getattr(self, "_hidden_frame_count", 0), total)
        # Restore the plain count when nothing is filtered out — otherwise
        # clearing the filter box left a stale "2 of 5 shown".
        self.frame_count_label.setText(
            f"{total - hidden} of {total} shown" if hidden else f"{total} loaded"
        )

    def update_next_step_hint(self):
        """Say what to do next, based on where the session actually is."""
        if not hasattr(self, "workflow_hint"):
            return
        if self.image_list.count() == 0:
            text = (
                "Add frames to start — “Add New Images” for stills, "
                "“Open Video Clip...” to pull a range out of a recording."
            )
        elif not self.class_mapping:
            text = (
                "Add label classes — “Droplets only” or “Droplets + arc” "
                "load the agreed ER70S-6 colours."
            )
        elif not self.current_class:
            text = "Select a class in the list below, then pick a drawing tool."
        else:
            count = sum(
                len(annotations)
                for class_name, annotations in (
                    self.all_annotations.get(self.image_file_name) or {}
                ).items()
                if not str(class_name).startswith("Temp-")
            )
            if count == 0:
                text = (
                    f"Drawing “{self.current_class}”. P polygon, B paint, "
                    "Enter to finish a polygon."
                )
            else:
                text = (
                    f"{count} label{'s' if count != 1 else ''} on this frame. "
                    "D moves to the next frame, C copies the selection forward."
                )
        self.workflow_hint.setText(text)

    def update_project_identity(self):
        """Keep the sidebar header in step with the open project."""
        if not hasattr(self, "project_name_label"):
            return
        if hasattr(self, "current_project_file"):
            name = os.path.splitext(os.path.basename(self.current_project_file))[0]
            self.project_name_label.setText(name)
        else:
            self.project_name_label.setText("No project")

        frames = self.image_list.count() if hasattr(self, "image_list") else 0
        classes = len(self.class_mapping)
        if not hasattr(self, "current_project_file"):
            self.project_meta_label.setText("Create or open a project to begin")
        else:
            self.project_meta_label.setText(
                f"{frames} frame{'s' if frames != 1 else ''} · "
                f"{classes} class{'es' if classes != 1 else ''}"
            )

    # --- Undo / redo ---------------------------------------------------

    def record_annotation_history(self, label: str = "", frame_name: str = None):
        """Snapshot the current frame before an edit changes it.

        Returns the frame the snapshot was taken for, so a caller whose
        edit may still be abandoned can undo the recording with
        ``discard_annotation_history``.
        """
        target = frame_name or self.current_slice or self.image_file_name
        if not target:
            return None
        self.annotation_history.record(
            target, self.all_annotations.get(target), label
        )
        self._sync_history_buttons()
        return target

    def discard_annotation_history(self, frame_name):
        """Undo a recording for an edit that did not happen after all.

        Recording clears the redo stack, so a snapshot taken before a
        dialog the user then cancels would silently throw a redo branch
        away and leave an undo step that restores an identical state.
        """
        if not frame_name:
            return
        self.annotation_history.discard_last(frame_name)
        self._sync_history_buttons()

    def _current_history_key(self):
        return self.current_slice or self.image_file_name

    def _sync_history_buttons(self):
        if not hasattr(self, "undo_button"):
            return
        key = self._current_history_key()
        can_undo = bool(key) and self.annotation_history.can_undo(key)
        can_redo = bool(key) and self.annotation_history.can_redo(key)
        self.undo_button.setEnabled(can_undo)
        self.redo_button.setEnabled(can_redo)
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(can_undo)
            self.redo_action.setEnabled(can_redo)

    def _restore_annotation_state(self, key, restored):
        """Put a snapshot back and refresh everything that reads from it."""
        if restored:
            self.all_annotations[key] = restored
        else:
            self.all_annotations.pop(key, None)

        self.image_label.annotations = copy.deepcopy(
            self.all_annotations.get(key, {})
        )
        self.image_label.reset_annotation_state()
        self.image_label.clear_current_annotation()
        self.image_label.highlighted_annotations.clear()
        self.update_annotation_list()
        # update_slice_list_colors ends in annotations_changed(), which
        # covers the markers, the filter and the counters.
        self.update_slice_list_colors()
        self.image_label.update()
        self.update_status_bar()

    def undo_annotation_change(self):
        """Step one annotation edit back on the current frame."""
        if self._sam3_inference_in_flight:
            return
        key = self._current_history_key()
        if not key:
            return
        step = self.annotation_history.undo(key, self.all_annotations.get(key))
        if step is None:
            return
        label, restored = step
        self._restore_annotation_state(key, restored)
        self._sync_history_buttons()
        self.statusBar().showMessage(
            f"Undid {label}" if label else "Undid the last change", 2500
        )
        self._save_after_history_step()

    def redo_annotation_change(self):
        """Re-apply the annotation edit that was just undone."""
        if self._sam3_inference_in_flight:
            return
        key = self._current_history_key()
        if not key:
            return
        step = self.annotation_history.redo(key, self.all_annotations.get(key))
        if step is None:
            return
        label, restored = step
        self._restore_annotation_state(key, restored)
        self._sync_history_buttons()
        self.statusBar().showMessage(
            f"Redid {label}" if label else "Redid the last change", 2500
        )
        self._save_after_history_step()

    def _save_after_history_step(self):
        """Persist an undo/redo, but never interrupt it with a dialog.

        ``auto_save`` prompts "you need to save the project first" when
        there is no project file. Ctrl+Z is a reflex key; a modal on it
        would be unusable in a workspace the annotator has not saved yet.
        """
        if hasattr(self, "current_project_file"):
            self.auto_save()
        else:
            self.set_saved_state(False)

    # --- Keyboard ------------------------------------------------------

    def _install_workflow_shortcuts(self):
        """Plain-key tool and class shortcuts, via one event filter.

        Two things rule out the obvious "one QShortcut per key" approach:

        * The Ctrl combinations are already owned by menu actions
          (Edit > Undo / Redo, Help > Keyboard Shortcuts). Binding the
          same sequence a second time makes it ambiguous, and Qt then
          fires *neither* — measured: with both bindings present, Ctrl+Z
          did nothing at all.
        * Application-context QShortcuts are cheap individually but a set
          this size (four tool letters plus nine class digits, on top of
          the existing A/D/C/F2) crashed Qt's shortcut map during teardown
          on PyQt6 6.11 here.

        One filter also matches how DINO review keys are already handled
        (see ``_DINOReviewEventFilter``), and puts the "is the user typing
        in a text box" test in a single place.
        """
        self._workflow_key_filter = _WorkflowKeyFilter(self)
        QApplication.instance().installEventFilter(self._workflow_key_filter)

    def handle_workflow_key(self, key, target=None) -> bool:
        """Act on a plain tool or class key. Returns True if consumed."""
        if self._sam3_inference_in_flight or self._typing_in_text_field(target):
            return False

        tool_keys = {
            Qt.Key.Key_P: "polygon_button",
            Qt.Key.Key_R: "rectangle_button",
            Qt.Key.Key_B: "paint_brush_button",
            Qt.Key.Key_E: "eraser_button",
        }
        if key in tool_keys:
            self._activate_tool_shortcut(tool_keys[key])
            return True

        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            index = key - Qt.Key.Key_1
            if index < self.class_list.count():
                self.select_class_by_index(index)
                return True
        return False

    #: Widgets that must receive letters and digits themselves.
    _TEXT_ENTRY_WIDGETS = (QLineEdit, QTextEdit, QAbstractSpinBox)

    #: Additionally exempt when the key is being delivered straight to
    #: them: a combo box uses letters for type-ahead, which is how the
    #: SAM and DINO model pickers are normally driven. Not exempt via
    #: focusWidget(), because a non-editable combo keeps focus after its
    #: popup closes and would then swallow every tool key.
    _TYPEAHEAD_WIDGETS = (QComboBox,)

    def _typing_in_text_field(self, target=None) -> bool:
        """True while a text input is taking the keystroke.

        Checks the widget the event was delivered to when one is given —
        the application's focus widget is null while the window is not
        active, which would let a tool key steal a character out of the
        frame filter box.
        """
        exempt = self._TEXT_ENTRY_WIDGETS + self._TYPEAHEAD_WIDGETS
        if isinstance(target, exempt):
            return True
        if target is not None and isinstance(target, QWidget):
            if isinstance(target.parentWidget(), exempt):
                return True
        return isinstance(QApplication.focusWidget(), self._TEXT_ENTRY_WIDGETS)

    def _activate_tool_shortcut(self, button_name):
        """Toggle a drawing tool from its letter key.

        Goes through ``click()`` rather than calling ``toggle_tool``
        directly: ``toggle_tool`` identifies the tool from ``sender()``,
        so a direct call would look like it came from nowhere and would
        fall back to the magic-wand button.
        """
        if self._typing_in_text_field() or self._sam3_inference_in_flight:
            return
        button = getattr(self, button_name, None)
        if button is None or not button.isEnabled():
            return
        button.click()

    def select_class_by_index(self, index):
        """Select the class at ``index`` in the class list (keys 1-9)."""
        if self._typing_in_text_field():
            return
        if 0 <= index < self.class_list.count():
            item = self.class_list.item(index)
            self.class_list.setCurrentItem(item)
            self.on_class_selected(item)

    def show_shortcut_reference(self):
        dialog = ShortcutReferenceDialog(self)
        dialog.exec()

    ##########    ### Tools  ########## I love useful image processing tools :)
    def open_dataset_splitter(self):
        self.dataset_splitter = DatasetSplitterTool(self)
        self.dataset_splitter.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.dataset_splitter.show_centered(self)

    def show_annotation_statistics(self):
        if not self.all_annotations:
            QMessageBox.warning(
                self, "No Annotations", "There are no annotations to analyze."
            )
            return
        try:
            self.annotation_stats_dialog = show_annotation_statistics(
                self, self.all_annotations
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while showing annotation statistics: {str(e)}",
            )

    def show_coco_json_combiner(self):
        self.coco_json_combiner_dialog = show_coco_json_combiner(self)

    def show_dino_merge_dialog(self):
        show_dino_merge_dialog(self)

    def show_stack_to_slices(self):
        self.stack_to_slices_dialog = show_stack_to_slices(self)

    def show_image_patcher(self):
        self.image_patcher_dialog = show_image_patcher(self)

    def show_image_augmenter(self):
        self.image_augmenter_dialog = show_image_augmenter(self)

    def show_slice_registration(self):
        self.slice_registration_dialog = SliceRegistrationTool(self)
        self.slice_registration_dialog.show_centered(self)

    def show_stack_interpolator(self):
        self.stack_interpolator_dialog = StackInterpolator(self)
        self.stack_interpolator_dialog.show_centered(self)

    def show_dicom_converter(self):
        self.dicom_converter_dialog = DicomConverter(self)
        self.dicom_converter_dialog.show_centered(self)

    ###################################################################

    # update the show_help method:
    def show_help(self):
        self.help_window = HelpWindow(
            dark_mode=self.dark_mode, font_size=self.font_sizes[self.current_font_size]
        )
        self.help_window.show_centered(self)

    def add_images(self):
        if self._reject_while_sam3_busy():
            return
        if not self.image_label.check_unsaved_changes():
            return
        file_names, _ = QFileDialog.getOpenFileNames(
            self, "Add Images", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)"
        )
        if file_names:
            self.add_images_to_list(file_names)

    def clear_all(self, new_project=False, show_messages=True):
        if self._sam3_inference_in_flight:
            self.show_warning("SAM 3 Tracker", "Wait for SAM 3 tracking to finish.")
            return
        if not new_project and show_messages:
            reply = self.show_question(
                "Clear All",
                "Are you sure you want to clear all images and annotations? This action cannot be undone.",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._reset_sam3_video_state(unload=True)
        self._clear_video_sessions(clean_clip_caches=True)

        # Clear images
        self.image_list.clear()
        self.image_paths.clear()
        self.all_images.clear()
        self.current_image = None
        self.image_file_name = ""

        # Clear the image display
        self.image_label.clear()
        self.image_label.setPixmap(QPixmap())  # Set an empty pixmap
        self.image_label.original_pixmap = None
        self.image_label.scaled_pixmap = None

        # Clear annotations
        self.all_annotations.clear()
        # History is keyed by bare frame name, so leaving it behind
        # would let one Ctrl+Z inject the closed project's
        # annotations into a new project that reuses a file name —
        # and undo auto-saves, so that reaches the .iap file.
        self.annotation_history.clear()
        self.annotation_list.clear()
        self.image_label.annotations.clear()
        self.image_label.highlighted_annotations.clear()

        # Clear current class
        self.current_class = None

        # Reset class-related data
        self.class_list.clear()
        self.image_label.class_colors.clear()
        self.class_mapping.clear()

        # Reset DINO state
        self.dino_class_table.clear_classes()
        self.dino_phrase_panel.clear()
        self.dino_model_loaded = False
        self.dino_custom_model_path = None
        self.dino_model_selector.setCurrentIndex(0)
        self.lbl_dino_status.setText("No DINO model loaded")
        self.btn_detect_single.setEnabled(False)
        self.btn_detect_batch.setEnabled(False)

        # Clear slices
        self.image_slices.clear()
        self.slices = []
        self.slice_list.clear()
        self.current_slice = None
        self.current_stack = None

        # Reset zoom
        self.image_label.zoom_factor = 1.0
        self.zoom_slider.setValue(100)

        # Reset tools
        self.image_label.current_tool = None
        self.polygon_button.setChecked(False)
        self.rectangle_button.setChecked(False)
        self.sam_magic_wand_button.setChecked(False)
        self.sam_magic_wand_button.setEnabled(False)  # Disable the SAM-Assisted button
        self.image_label.sam_magic_wand_active = False  # Deactivate SAM magic wand

        # Reset SAM-related attributes
        self.image_label.sam_bbox = None
        self.image_label.drawing_sam_bbox = False
        self.image_label.temp_sam_prediction = None

        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)  # Reset cursor to default
        self.sam_model_selector.setCurrentIndex(0)  # Reset to "Pick a SAM Model"
        self.current_sam_model = None  # Reset the current SAM model

        # Reset project-related attributes
        if not new_project:
            if hasattr(self, "current_project_file"):
                del self.current_project_file
            if hasattr(self, "current_project_dir"):
                del self.current_project_dir

        # Update UI
        self.image_label.update()
        self.update_image_info()

        # Force a repaint of the main window
        self.repaint()
        self.update_window_title()

    def show_warning(self, title, message):
        QMessageBox.warning(self, title, message)

    def _reject_while_sam3_busy(self):
        if not self._sam3_inference_in_flight:
            return False
        self.show_warning("SAM 3 Tracker", "Wait for SAM 3 tracking to finish.")
        return True

    def _run_sam3_ui_locked(self, operation, *args):
        """Run synchronous SAM work without allowing nested UI mutations."""
        was_enabled = self.isEnabled()
        self.setEnabled(False)
        try:
            return _run_sync(operation, *args)
        finally:
            self.setEnabled(was_enabled)

    def show_info(self, title, message):
        QMessageBox.information(self, title, message)

    def update_image_info(self, additional_info=None):
        if hasattr(self, "canvas_file_label"):
            self.canvas_file_label.setText(
                self.current_slice or self.image_file_name or "No frame loaded"
            )

        position = ""
        if self.frame_sequence:
            frame = self.frame_sequence.frame_for_name(self.image_file_name)
            if frame:
                position = f"{frame.index + 1} / {len(self.frame_sequence.frames)}"
                if frame.source_index is not None:
                    position += f"  (source {frame.source_index})"
        elif self.image_file_name and hasattr(self, "image_list"):
            matches = self.image_list.findItems(
                self.image_file_name, Qt.MatchFlag.MatchExactly
            )
            if matches:
                row = self.image_list.row(matches[0])
                position = f"{row + 1} / {self.image_list.count()}"
        if hasattr(self, "canvas_position_label"):
            self.canvas_position_label.setText(position)

        if self.current_image:
            width = self.current_image.width()
            height = self.current_image.height()
            info = f"{width} x {height} px"
            if additional_info:
                info += f"  ·  {additional_info}"
            self.image_info_label.setText(info)
        else:
            self.image_info_label.setText("No image loaded")

        self._update_canvas_placeholder()
        # Navigation only. The whole-project scan belongs to
        # annotations_changed(), which the mutation paths call.
        self.refresh_session_status()
        self.update_status_bar()

    def show_question(self, title, message):
        return QMessageBox.question(
            self, title, message, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
        )

    def show_image_context_menu(self, position):
        menu = QMenu()
        current_item = self.image_list.itemAt(position)
        if current_item:
            file_name = current_item.text()
            delete_action = menu.addAction("Remove Image")

            if not self.is_multi_dimensional(file_name):
                predict_action = menu.addAction("Predict using YOLO")

            if self.is_multi_dimensional(file_name):
                redefine_dimensions_action = menu.addAction("Redefine Dimensions")

            action = menu.exec(self.image_list.mapToGlobal(position))

            if action == delete_action:
                self.remove_image()
            elif not self.is_multi_dimensional(file_name) and action == predict_action:
                self.predict_single_image(file_name)
            elif (
                self.is_multi_dimensional(file_name)
                and action == redefine_dimensions_action
            ):
                self.redefine_dimensions(file_name)

    def is_multi_dimensional(self, file_name):
        return file_name.lower().endswith((".tif", ".tiff", ".czi"))

    def predict_single_image(self, file_name):
        if self.is_multi_dimensional(file_name):
            return  # Do nothing for multi-dimensional images

        if not self.yolo_trainer or not self.yolo_trainer.model:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO model first from the YOLO > Prediction Settings > Load Model menu.",
            )
            return

        # Deactivate SAM tool before prediction
        self.deactivate_sam_magic_wand()

        image_path = self.image_paths[file_name]
        try:
            results = self.yolo_trainer.predict(image_path)
            self.process_yolo_results(results, file_name)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Prediction Error",
                f"An error occurred during prediction: {str(e)}\n\n"
                "This might be due to a mismatch between the model and the YAML file classes. "
                "Please check that the YAML file corresponds to the loaded model.",
            )

    def redefine_dimensions(self, file_name):
        file_path = self.image_paths.get(file_name)
        if not file_path or not file_path.lower().endswith((".tif", ".tiff", ".czi")):
            return  # Exit the method if it's not a TIFF or CZI file

        reply = QMessageBox.warning(
            self,
            "Redefine Dimensions",
            "Redefining dimensions will cause all associated annotations to be lost. "
            "Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Remove existing annotations for this file
            base_name = os.path.splitext(file_name)[0]

            print(f"Removing annotations for image: {base_name}")
            # print(f"Current annotations: {list(self.all_annotations.keys())}")

            # Create a list of keys to remove, using a more specific matching condition
            keys_to_remove = [
                key
                for key in self.all_annotations.keys()
                if key == base_name
                or (
                    key.startswith(f"{base_name}_")
                    and not key.startswith(f"{base_name}_8bit")
                )
            ]

            print(f"Keys to remove: {keys_to_remove}")

            # Remove the annotations, and the undo history that describes
            # them. Re-confirming the dialog usually reproduces the same
            # slice names, so a surviving entry would let one Ctrl+Z
            # restore annotations the user was told had been removed —
            # and undo auto-saves, so it would reach the .iap.
            for key in keys_to_remove:
                del self.all_annotations[key]
                self.annotation_history.forget(key)

            # print(f"Annotations after removal: {list(self.all_annotations.keys())}")

            # Remove existing slices
            if base_name in self.image_slices:
                del self.image_slices[base_name]

            # Clear current image if it's the one being redefined
            if self.image_file_name == file_name:
                self.current_image = None
                self.image_label.clear()

            # Reload the image with new dimension dialog
            if file_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(file_path, force_dimension_dialog=True)
            elif file_path.lower().endswith(".czi"):
                self.load_czi(file_path, force_dimension_dialog=True)

            # Update UI
            self.update_slice_list()
            self.update_annotation_list()
            self.image_label.update()

            # print(f"Final annotations: {list(self.all_annotations.keys())}")

            QMessageBox.information(
                self,
                "Dimensions Redefined",
                "The dimensions have been redefined and the image reloaded. "
                "All previous annotations for this image have been removed.",
            )

    def remove_image(self):
        if self._reject_while_sam3_busy():
            return
        current_item = self.image_list.currentItem()
        if current_item and self._remove_image_item(current_item):
            self.auto_save()

    def _remove_image_item(self, item, select_next=True):
        """Remove one image through the shared project/video state transition."""
        if item is None:
            return None

        file_name = item.text()
        self.image_list.takeItem(self.image_list.row(item))
        self.image_paths.pop(file_name, None)
        self.all_images = [
            image
            for image in self.all_images
            if image["file_name"] != file_name
        ]
        self.all_annotations.pop(file_name, None)
        # Drop the undo history with the annotations. History is keyed by
        # bare frame name, so a stale entry would let one Ctrl+Z re-insert
        # a removed frame's annotations into a frame that reuses the name.
        self.annotation_history.forget(file_name)
        self._remove_frame_from_video_sessions(file_name)

        base_name = os.path.splitext(file_name)[0]
        if base_name in self.image_slices:
            for slice_name, _ in self.image_slices[base_name]:
                self.all_annotations.pop(slice_name, None)
                self.annotation_history.forget(slice_name)
            del self.image_slices[base_name]
            self.slice_list.clear()

        if self.image_file_name == file_name:
            self.current_image = None
            self.image_file_name = ""
            self.current_slice = None
            self.image_label.clear()
            self.annotation_list.clear()

        if select_next and self.image_list.count() > 0:
            next_item = self.image_list.item(0)
            self.image_list.setCurrentItem(next_item)
            self.switch_image(next_item)
        elif self.image_list.count() == 0:
            self.current_image = None
            self.image_file_name = ""
            self.current_slice = None
            self.image_label.clear()
            self.annotation_list.clear()
            self.slice_list.clear()

        self.update_ui()
        return file_name

    def load_annotations(self):
        if self._reject_while_sam3_busy():
            return
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Annotations", "", "JSON Files (*.json)"
        )
        if file_name:
            with open(file_name, "r") as f:
                self.loaded_json = json.load(f)

            # Load categories
            self.class_list.clear()
            self.image_label.class_colors.clear()
            self.class_mapping.clear()
            for category in self.loaded_json["categories"]:
                class_name = category["name"]
                self.class_mapping[class_name] = category["id"]

                # Assign a color if not already assigned
                if class_name not in self.image_label.class_colors:
                    color = QColor(
                        Qt.GlobalColor(len(self.image_label.class_colors) % 16 + 7)
                    )
                    self.image_label.class_colors[class_name] = color

                # Add item to class list with color indicator
                item = QListWidgetItem(class_name)
                self.update_class_item_color(
                    item, self.image_label.class_colors[class_name]
                )
                self.class_list.addItem(item)

            # Create a mapping of image IDs to file names
            image_id_to_filename = {
                img["id"]: img["file_name"] for img in self.loaded_json["images"]
            }

            # Load image information
            json_images = {img["file_name"]: img for img in self.loaded_json["images"]}

            # Update existing images and add new ones from JSON
            updated_all_images = []
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                file_name = item.text()
                if file_name in json_images:
                    updated_image = self.all_images[i].copy()
                    updated_image.update(json_images[file_name])
                    updated_all_images.append(updated_image)
                    del json_images[file_name]
                else:
                    updated_all_images.append(self.all_images[i])

            # Add remaining images from JSON
            for img in json_images.values():
                updated_all_images.append(img)
                self.image_list.addItem(img["file_name"])

            self.all_images = updated_all_images

            # Load annotations
            self.all_annotations.clear()
            # The imported set replaces what history refers to.
            self.annotation_history.clear()
            for annotation in self.loaded_json["annotations"]:
                image_id = annotation["image_id"]
                file_name = image_id_to_filename.get(image_id)
                if file_name:
                    if file_name not in self.all_annotations:
                        self.all_annotations[file_name] = {}

                    category = next(
                        (
                            cat
                            for cat in self.loaded_json["categories"]
                            if cat["id"] == annotation["category_id"]
                        ),
                        None,
                    )
                    if category:
                        category_name = category["name"]
                        if category_name not in self.all_annotations[file_name]:
                            self.all_annotations[file_name][category_name] = []

                        ann = {
                            "category_id": annotation["category_id"],
                            "category_name": category_name,
                        }

                        if "segmentation" in annotation:
                            ann["segmentation"] = annotation["segmentation"][0]
                            ann["type"] = "polygon"
                        elif "bbox" in annotation:
                            ann["bbox"] = annotation["bbox"]
                            ann["type"] = "bbox"

                        # Add number field if it's missing
                        if "number" not in ann:
                            ann["number"] = (
                                len(self.all_annotations[file_name][category_name]) + 1
                            )

                        self.all_annotations[file_name][category_name].append(ann)

            # Check for missing images
            missing_images = [
                img["file_name"]
                for img in self.loaded_json["images"]
                if img["file_name"] not in self.image_paths
            ]
            if missing_images:
                self.show_warning(
                    "Missing Images",
                    "The following images are missing:\n" + "\n".join(missing_images),
                )

            # Reload the current image if it exists, otherwise load the first image
            if self.image_file_name and self.image_file_name in self.all_annotations:
                self.switch_image(
                    self.image_list.findItems(self.image_file_name, Qt.MatchFlag.MatchExactly)[0]
                )
            elif self.all_images:
                self.switch_image(self.image_list.item(0))

            self.image_label.highlighted_annotations = []  # Clear existing highlights
            self.update_annotation_list()  # This will repopulate the annotation list
            self.image_label.update()  # Force a redraw of the image label

    def clear_highlighted_annotation(self):
        self.image_label.highlighted_annotation = None
        self.image_label.update()

    def update_highlighted_annotations(self):
        selected_items = self.annotation_list.selectedItems()
        self.image_label.highlighted_annotations = [
            item.data(Qt.ItemDataRole.UserRole) for item in selected_items
        ]
        self.image_label.update()  # Force a redraw of the image label

        # Enable/disable merge and change class buttons based on selection
        self.merge_button.setEnabled(len(selected_items) >= 2)
        self.change_class_button.setEnabled(len(selected_items) > 0)

    def renumber_annotations(self):
        current_name = self.current_slice or self.image_file_name
        if current_name in self.all_annotations:
            for class_name, annotations in self.all_annotations[current_name].items():
                for i, ann in enumerate(annotations, start=1):
                    ann["number"] = i
        self.update_annotation_list()

    def delete_selected_annotations(self):
        if self._reject_while_sam3_busy():
            return
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "No Selection", "Please select an annotation to delete."
            )
            return

        reply = QMessageBox.question(
            self,
            "Delete Annotations",
            f"Are you sure you want to delete {len(selected_items)} annotation(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.record_annotation_history(
                f"deleting {len(selected_items)} annotation(s)"
            )
            # Create a list of annotations to remove
            annotations_to_remove = []
            for item in selected_items:
                annotation = item.data(Qt.ItemDataRole.UserRole)
                annotations_to_remove.append((annotation["category_name"], annotation))

            # Remove annotations from image_label.annotations
            for category_name, annotation in annotations_to_remove:
                if category_name in self.image_label.annotations:
                    if annotation in self.image_label.annotations[category_name]:
                        self.image_label.annotations[category_name].remove(annotation)

            # Update all_annotations
            current_name = self.current_slice or self.image_file_name
            self.all_annotations[current_name] = self.image_label.annotations

            # Sort and update the annotation list based on the current sorting method
            if self.current_sort_method == "area":
                self.sort_annotations_by_area()
            else:
                self.sort_annotations_by_class()

            self.image_label.highlighted_annotations.clear()
            self.image_label.update()

            # Update slice list colors
            self.update_slice_list_colors()

            QMessageBox.information(
                self,
                "Annotations Deleted",
                f"{len(selected_items)} annotation(s) have been deleted.",
            )
            self.auto_save()  # Auto-save after deleting annotations

    def merge_annotations(self):
        if self._reject_while_sam3_busy():
            return
        if self.image_label.editing_polygon is not None:
            QMessageBox.warning(
                self,
                "Edit Mode Active",
                "Please exit the annotation edit mode before merging annotations.",
            )
            return

        selected_items = self.annotation_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(
                self,
                "Not Enough Annotations",
                "Please select at least two annotations to merge.",
            )
            return

        class_name = selected_items[0].data(Qt.ItemDataRole.UserRole)["category_name"]
        if not all(
            item.data(Qt.ItemDataRole.UserRole)["category_name"] == class_name
            for item in selected_items
        ):
            QMessageBox.warning(
                self,
                "Mixed Classes",
                "All selected annotations must be from the same class.",
            )
            return

        polygons = []
        original_annotations = []
        for item in selected_items:
            annotation = item.data(Qt.ItemDataRole.UserRole)
            original_annotations.append(annotation)
            if "segmentation" in annotation:
                points = zip(
                    annotation["segmentation"][0::2], annotation["segmentation"][1::2]
                )
                polygon = Polygon(points)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                polygons.append(polygon)

        def are_all_polygons_connected(polygons):
            if len(polygons) < 2:
                return True

            connected = set([0])  # Start with the first polygon
            to_check = set(range(1, len(polygons)))

            while to_check:
                newly_connected = set()
                for i in connected:
                    for j in to_check:
                        if polygons[i].intersects(polygons[j]) or polygons[i].touches(
                            polygons[j]
                        ):
                            newly_connected.add(j)

                if not newly_connected:
                    return (
                        False  # If no new connections found, they're not all connected
                    )

                connected.update(newly_connected)
                to_check -= newly_connected

            return True  # All polygons are connected

        if not are_all_polygons_connected(polygons):
            QMessageBox.warning(
                self,
                "Disconnected Polygons",
                "Not all selected annotations are connected. Please select only connected annotations to merge.",
            )
            return

        try:
            merged_polygon = unary_union(polygons)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Merge Error",
                f"Unable to merge the selected annotations due to an error: {str(e)}",
            )
            return

        merge_history_frame = self.record_annotation_history(
            f"merging {len(selected_items)} annotation(s)"
        )
        new_annotation = {
            "segmentation": [],
            "category_id": self.class_mapping[class_name],
            "category_name": class_name,
        }

        if isinstance(merged_polygon, Polygon):
            new_annotation["segmentation"] = [
                coord for point in merged_polygon.exterior.coords for coord in point
            ]
        elif isinstance(merged_polygon, MultiPolygon):
            largest_polygon = max(merged_polygon.geoms, key=lambda p: p.area)
            new_annotation["segmentation"] = [
                coord for point in largest_polygon.exterior.coords for coord in point
            ]

        # Ask user about keeping original annotations
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Merge Annotations")
        msg_box.setText("Do you want to keep the original annotations?")
        msg_box.setIcon(QMessageBox.Icon.Question)

        keep_button = msg_box.addButton("Keep", QMessageBox.ButtonRole.YesRole)
        delete_button = msg_box.addButton("Delete", QMessageBox.ButtonRole.NoRole)
        cancel_button = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

        msg_box.setDefaultButton(cancel_button)
        msg_box.setEscapeButton(cancel_button)

        msg_box.exec()

        if msg_box.clickedButton() == cancel_button:
            # The snapshot was taken before this dialog; unwind it, or the
            # abandoned merge would still have cleared the redo branch.
            self.discard_annotation_history(merge_history_frame)
            return

        if msg_box.clickedButton() == delete_button:
            for annotation in original_annotations:
                if annotation in self.image_label.annotations[class_name]:
                    self.image_label.annotations[class_name].remove(annotation)

        self.image_label.annotations.setdefault(class_name, []).append(new_annotation)

        current_name = self.current_slice or self.image_file_name
        self.all_annotations[current_name] = self.image_label.annotations

        self.renumber_annotations()
        self.update_annotation_list()
        self.save_current_annotations()
        self.update_slice_list_colors()
        self.image_label.update()

        QMessageBox.information(
            self, "Merge Complete", "Annotations have been merged successfully."
        )
        self.auto_save()  # Auto-save after merging annotations

    def delete_selected_image(self):
        if self._sam3_inference_in_flight:
            return
        current_item = self.image_list.currentItem()
        if current_item:
            file_name = current_item.text()
            reply = QMessageBox.question(
                self,
                "Delete Image",
                f"Are you sure you want to delete the image '{file_name}'?\n\n"
                "This will remove the image and all its associated annotations.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                removed_name = self._remove_image_item(current_item)
                if removed_name:
                    self.auto_save()
                    QMessageBox.information(
                        self,
                        "Image Deleted",
                        f"The image '{file_name}' has been deleted.",
                    )

    def display_image(self):
        if self.current_image:
            if isinstance(self.current_image, QImage):
                pixmap = QPixmap.fromImage(self.current_image)
            elif isinstance(self.current_image, QPixmap):
                pixmap = self.current_image
            else:
                print(f"Unexpected image type: {type(self.current_image)}")
                return

            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap)
                self.image_label.adjustSize()
            else:
                print("Error: Null pixmap")
        else:
            self.image_label.clear()
            print("No current image to display")

    def update_ui(self):
        self.update_image_list()
        self.update_slice_list()
        self.update_class_list()
        self.update_annotation_list()
        self.image_label.update()
        self.update_image_info()

    def add_class(self, class_name=None, color=None):
        if self._reject_while_sam3_busy():
            return
        if not self.image_label.check_unsaved_changes():
            return

        if class_name is None:
            while True:
                class_name, ok = QInputDialog.getText(
                    self, "Add Class", "Enter class name:"
                )
                if not ok:
                    print("Class addition cancelled")
                    return
                if not class_name.strip():
                    QMessageBox.warning(
                        self,
                        "Invalid Input",
                        "Please enter a class name or press Cancel.",
                    )
                    continue
                if class_name in self.class_mapping:
                    QMessageBox.warning(
                        self,
                        "Duplicate Class",
                        f"The class '{class_name}' already exists. Please choose a different name.",
                    )
                    continue
                break
        else:
            # For programmatic addition (e.g., from YOLO predictions)
            if class_name in self.class_mapping:
                print(f"Class '{class_name}' already exists. Skipping addition.")
                return

        if not isinstance(class_name, str):
            print(
                f"Warning: class_name is not a string. Converting {class_name} to string."
            )
            class_name = str(class_name)

        if color is None:
            color = QColor(Qt.GlobalColor(len(self.image_label.class_colors) % 16 + 7))
        elif isinstance(color, str):
            color = QColor(color)

        print(f"Adding class: {class_name}, color: {color.name()}")

        self.image_label.class_colors[class_name] = color
        self.class_mapping[class_name] = len(self.class_mapping) + 1

        try:
            item = QListWidgetItem(class_name)

            # Create a color indicator
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))

            # Set visibility state
            item.setData(Qt.ItemDataRole.UserRole, True)

            # Set checkbox
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)

            self.class_list.addItem(item)

            self.class_list.setCurrentItem(item)
            self.current_class = class_name
            print(f"Class added successfully: {class_name}")

            # Sync DINO phrase/threshold state. Select the newly added
            # row so the phrase editor below the table reveals itself —
            # it hides by default and only becomes visible when a row is
            # selected (set_active_class). Skip the row-select during
            # project load: classes are added in a loop and we don't want
            # N row-selection signals firing during bulk restoration; the
            # caller will select an appropriate row after load completes.
            row_added = self.dino_class_table.add_class(class_name)
            self.dino_phrase_panel.on_class_added(class_name)
            if row_added and not self.is_loading_project:
                self.dino_class_table.selectRow(self.dino_class_table.rowCount() - 1)

            if not self.is_loading_project:
                self.auto_save()
        except Exception as e:
            print(f"Error adding class: {e}")
            traceback.print_exc()

    def update_class_item_color(self, item, color):
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        item.setIcon(QIcon(pixmap))

    def update_class_list(self):
        self.class_list.clear()
        for class_name, color in self.image_label.class_colors.items():
            item = QListWidgetItem(class_name)

            # Create a color indicator
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))

            # Store the visibility state
            item.setData(
                Qt.ItemDataRole.UserRole, self.image_label.class_visibility.get(class_name, True)
            )

            # Set checkbox
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) else Qt.CheckState.Unchecked)

            self.class_list.addItem(item)

        # Re-select the current class if it exists
        if self.current_class:
            items = self.class_list.findItems(self.current_class, Qt.MatchFlag.MatchExactly)
            if items:
                self.class_list.setCurrentItem(items[0])
        elif self.class_list.count() > 0:
            # If no class is selected, select the first one
            self.class_list.setCurrentItem(self.class_list.item(0))

        print(f"Updated class list with {self.class_list.count()} items")

    def update_class_selection(self):
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.text() == self.current_class:
                item.setSelected(True)
            else:
                item.setSelected(False)

    def toggle_class_visibility(self, item):
        class_name = item.text()
        is_visible = item.checkState() == Qt.CheckState.Checked
        self.image_label.set_class_visibility(class_name, is_visible)
        item.setData(Qt.ItemDataRole.UserRole, is_visible)
        self.image_label.update()

    def change_annotation_class(self):
        if self._reject_while_sam3_busy():
            return
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select one or more annotations to change class.",
            )
            return

        class_dialog = QDialog(self)
        class_dialog.setWindowTitle("Change Class")
        layout = QVBoxLayout(class_dialog)

        class_combo = QComboBox()
        for class_name in self.class_mapping.keys():
            class_combo.addItem(class_name)
        layout.addWidget(class_combo)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(class_dialog.accept)
        button_box.rejected.connect(class_dialog.reject)
        layout.addWidget(button_box)

        if class_dialog.exec() == QDialog.DialogCode.Accepted:
            new_class = class_combo.currentText()
            current_name = self.current_slice or self.image_file_name
            self.record_annotation_history(f"changing class to {new_class}")

            # Get the current maximum number for the new class
            max_number = max(
                [
                    ann.get("number", 0)
                    for ann in self.image_label.annotations.get(new_class, [])
                ]
                + [0]
            )

            for item in selected_items:
                annotation = item.data(Qt.ItemDataRole.UserRole)
                old_class = annotation["category_name"]

                # Remove from old class
                self.image_label.annotations[old_class].remove(annotation)
                if not self.image_label.annotations[old_class]:
                    del self.image_label.annotations[old_class]

                # Add to new class with updated number
                annotation["category_name"] = new_class
                annotation["category_id"] = self.class_mapping[new_class]
                max_number += 1
                annotation["number"] = max_number
                if new_class not in self.image_label.annotations:
                    self.image_label.annotations[new_class] = []
                self.image_label.annotations[new_class].append(annotation)

            # Update all_annotations
            self.all_annotations[current_name] = self.image_label.annotations

            # Renumber all annotations for consistency
            self.renumber_annotations()

            self.update_annotation_list()
            self.image_label.update()
            self.save_current_annotations()
            self.update_slice_list_colors()
            self.auto_save()

            QMessageBox.information(
                self,
                "Class Changed",
                f"Selected annotations have been changed to class '{new_class}'.",
            )

    def toggle_tool(self):
        if not self.image_label.check_unsaved_changes():
            return

        sender = self.sender()
        if sender is None:
            sender = self.sam_magic_wand_button

        if not self.current_class:
            QMessageBox.warning(
                self,
                "No Class Selected",
                "Please select a class before using annotation tools.",
            )
            sender.setChecked(False)
            return

        if self.current_class and self.current_class.startswith("Temp-"):
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Cannot use annotation tools with temporary classes.",
            )
            sender.setChecked(False)
            return

        other_buttons = [btn for btn in self.tool_group.buttons() if btn != sender]

        # Deactivate SAM if we're switching to a different tool
        if (
            sender != self.sam_magic_wand_button
            and self.image_label.sam_magic_wand_active
        ):
            self.deactivate_sam_magic_wand()

        if sender.isChecked():
            # Uncheck all other buttons
            for btn in other_buttons:
                btn.setChecked(False)

            # Set the current tool based on the checked button
            if sender == self.polygon_button:
                self.image_label.current_tool = "polygon"
            elif sender == self.rectangle_button:
                self.image_label.current_tool = "rectangle"
            elif sender == self.sam_magic_wand_button:
                self.image_label.current_tool = "sam_magic_wand"
                self.activate_sam_magic_wand()
            elif sender == self.paint_brush_button:
                self.image_label.current_tool = "paint_brush"
                self.image_label.setFocus()  # Set focus on the image label
            elif sender == self.eraser_button:
                self.image_label.current_tool = "eraser"
                self.image_label.setFocus()  # Set focus on the image label
        else:
            self.image_label.current_tool = None
            if sender == self.sam_magic_wand_button:
                self.deactivate_sam_magic_wand()

        # Update UI based on the current tool
        self.update_ui_for_current_tool()

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if self.image_label.current_tool == "paint_brush":
                self.paint_brush_size = max(1, self.paint_brush_size + delta // 120)
                print(f"Paint brush size: {self.paint_brush_size}")
            elif self.image_label.current_tool == "eraser":
                self.eraser_size = max(1, self.eraser_size + delta // 120)
                print(f"Eraser size: {self.eraser_size}")
        else:
            super().wheelEvent(event)

    def update_ui_for_current_tool(self):
        # Disable finish_polygon_button if it still exists in your code
        if hasattr(self, "finish_polygon_button"):
            self.finish_polygon_button.setEnabled(
                self.image_label.current_tool in ["polygon", "rectangle"]
            )

        # Update button states
        self.polygon_button.setChecked(self.image_label.current_tool == "polygon")
        self.rectangle_button.setChecked(self.image_label.current_tool == "rectangle")
        self.sam_magic_wand_button.setChecked(
            self.image_label.current_tool == "sam_magic_wand"
        )

        # Enable/disable SAM button based on model availability
        self.sam_magic_wand_button.setEnabled(self.current_sam_model is not None)

        # Disable all tools if no class is selected
        tools_enabled = (
            self.current_class is not None
            and not self.current_class.startswith("Temp-")
        )
        for button in self.tool_group.buttons():
            button.setEnabled(tools_enabled)

        # Crosshair for every tool that places a point on the image, not
        # just the magic wand — an arrow tip is a poor aiming reticle when
        # the boundary being traced is a few pixels wide.
        precise_tools = {
            "polygon",
            "rectangle",
            "paint_brush",
            "eraser",
            "sam_box",
            "sam_points",
        }
        wand_active = (
            self.image_label.current_tool == "sam_magic_wand"
            and self.sam_magic_wand_button.isEnabled()
        )
        if wand_active or self.image_label.current_tool in precise_tools:
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

        self.update_status_bar()

    def on_class_selected(self, current=None, previous=None):
        if not self.image_label.check_unsaved_changes():
            return

        if current is None:
            current = self.class_list.currentItem()

        if current:
            self.current_class = current.text()
            print(f"Class selected: {self.current_class}")
            if hasattr(self, "canvas_file_label"):
                self.canvas_file_label.setToolTip(
                    f"Active class: {self.current_class}"
                )

            if self.current_class.startswith("Temp-"):
                self.disable_annotation_tools()
            else:
                self.enable_annotation_tools()
        else:
            self.current_class = None
            if hasattr(self, "canvas_file_label"):
                self.canvas_file_label.setToolTip(
                    "Choose a class before drawing."
                )
            self.disable_annotation_tools()

    def disable_annotation_tools(self):
        for button in self.tool_group.buttons():
            button.setChecked(False)
            button.setEnabled(False)
        self.image_label.current_tool = None

    def enable_annotation_tools(self):
        for button in self.tool_group.buttons():
            button.setEnabled(True)

    def show_class_context_menu(self, position):
        menu = QMenu()
        rename_action = menu.addAction("Rename Class")
        change_color_action = menu.addAction("Change Color")
        delete_action = menu.addAction("Delete Class")

        item = self.class_list.itemAt(position)
        if item:
            action = menu.exec(self.class_list.mapToGlobal(position))

            if action == rename_action:
                self.rename_class(item)
            elif action == change_color_action:
                self.change_class_color(item)
            elif action == delete_action:
                self.delete_class(item)
        else:
            QMessageBox.warning(
                self, "No Selection", "Please select a class to perform actions."
            )

    def change_class_color(self, item):
        class_name = item.text()
        current_color = self.image_label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
        color = QColorDialog.getColor(
            current_color, self, f"Select Color for {class_name}"
        )

        if color.isValid():
            self.image_label.class_colors[class_name] = color

            # Update the color indicator
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))

            self.update_annotation_list_colors(class_name, color)
            self.image_label.update()
            self.auto_save()  # Auto-save after changing class color

    def rename_class(self, item):
        if self._sam3_inference_in_flight:
            return
        old_name = item.text()
        new_name, ok = QInputDialog.getText(
            self, "Rename Class", "Enter new class name:", text=old_name
        )
        if ok and new_name and new_name != old_name:
            # Update class mapping
            if old_name in self.class_mapping:
                old_id = self.class_mapping[old_name]
                self.class_mapping[new_name] = old_id
                del self.class_mapping[old_name]
            else:
                print(f"Warning: Class '{old_name}' not found in class_mapping")
                return

            # Update class colors
            if old_name in self.image_label.class_colors:
                self.image_label.class_colors[new_name] = (
                    self.image_label.class_colors.pop(old_name)
                )
            else:
                print(f"Warning: Class '{old_name}' not found in class_colors")
                return

            # Update annotations for all images and slices
            for image_name, image_annotations in self.all_annotations.items():
                if old_name in image_annotations:
                    image_annotations[new_name] = image_annotations.pop(old_name)
                    for annotation in image_annotations[new_name]:
                        annotation["category_name"] = new_name

            # Update current image annotations
            if old_name in self.image_label.annotations:
                self.image_label.annotations[new_name] = (
                    self.image_label.annotations.pop(old_name)
                )
                for annotation in self.image_label.annotations[new_name]:
                    annotation["category_name"] = new_name

            # Update current class if it's the renamed one
            if self.current_class == old_name:
                self.current_class = new_name

            # Update annotation list for all images and slices
            self.update_all_annotation_lists()

            # Update class list
            item.setText(new_name)

            # Update the image label
            self.image_label.update()
            self.auto_save()  # Auto-save after renaming a class

            print(f"Class renamed from '{old_name}' to '{new_name}'")

    def delete_class(self, item=None):
        if self._sam3_inference_in_flight:
            return
        if item is None:
            item = self.class_list.currentItem()

        if item is None:
            QMessageBox.warning(
                self, "No Selection", "Please select a class to delete."
            )
            return

        class_name = item.text()

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Delete Class",
            f"Are you sure you want to delete the class '{class_name}'?\n\n"
            "This will remove all annotations associated with this class.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Proceed with deletion
            # Remove class color
            self.image_label.class_colors.pop(class_name, None)

            # Remove class from mapping
            self.class_mapping.pop(class_name, None)

            # Remove annotations for this class from all images
            for image_annotations in self.all_annotations.values():
                image_annotations.pop(class_name, None)

            # Remove annotations for this class from current image
            self.image_label.annotations.pop(class_name, None)

            # Sync DINO state
            self.dino_class_table.remove_class(class_name)
            self.dino_phrase_panel.on_class_removed(class_name)

            # Update annotation list
            self.update_annotation_list()

            # Remove class from list
            row = self.class_list.row(item)
            self.class_list.takeItem(row)

            # Update current_class
            if self.current_class == class_name:
                self.current_class = None
                if self.class_list.count() > 0:
                    self.class_list.setCurrentRow(0)
                    self.on_class_selected(self.class_list.item(0))
                else:
                    self.disable_annotation_tools()

            self.image_label.update()

            # Inform the user
            QMessageBox.information(
                self, "Class Deleted", f"The class '{class_name}' has been deleted."
            )
            self.auto_save()  # Auto-save after deleting a class
        else:
            # User cancelled the operation
            QMessageBox.information(
                self, "Deletion Cancelled", "The class deletion was cancelled."
            )

    def finish_polygon(self):
        if (
            self.image_label.current_tool == "polygon"
            and len(self.image_label.current_annotation) > 2
        ):
            if self.current_class is None:
                QMessageBox.warning(
                    self,
                    "No Class Selected",
                    "Please select a class before finishing the annotation.",
                )
                return

            # Create a polygon from the current annotation
            polygon = Polygon(self.image_label.current_annotation)

            # Define the image boundary as a rectangle
            image_boundary = Polygon(
                [
                    (0, 0),
                    (self.current_image.width(), 0),
                    (self.current_image.width(), self.current_image.height()),
                    (0, self.current_image.height()),
                ]
            )

            # Intersect the polygon with the image boundary
            clipped_polygon = polygon.intersection(image_boundary)

            if clipped_polygon.is_empty:
                QMessageBox.warning(
                    self,
                    "Invalid Annotation",
                    "The annotation is completely outside the image boundaries.",
                )
                self.image_label.clear_current_annotation()
                self.image_label.update()
                return

            # Convert the clipped polygon to a segmentation format
            if isinstance(clipped_polygon, Polygon):
                segmentation = [
                    coord
                    for point in clipped_polygon.exterior.coords
                    for coord in point
                ]
            elif isinstance(clipped_polygon, MultiPolygon):
                largest_polygon = max(clipped_polygon.geoms, key=lambda p: p.area)
                segmentation = [
                    coord
                    for point in largest_polygon.exterior.coords
                    for coord in point
                ]
            else:
                QMessageBox.warning(
                    self, "Invalid Annotation", "The annotation could not be processed."
                )
                return

            self.record_annotation_history("adding a polygon")
            new_annotation = {
                "segmentation": segmentation,
                "category_id": self.class_mapping[self.current_class],
                "category_name": self.current_class,
            }
            self.image_label.annotations.setdefault(self.current_class, []).append(
                new_annotation
            )
            self.add_annotation_to_list(new_annotation)
            self.image_label.clear_current_annotation()
            self.image_label.drawing_polygon = False  # Reset the drawing_polygon flag
            self.image_label.reset_annotation_state()
            self.image_label.update()

            # Save the current annotations
            self.save_current_annotations()

            # Update the slice list colors
            self.update_slice_list_colors()
            self.auto_save()  # Auto-save after adding a polygon annotation

    def highlight_annotation(self, item):
        self.image_label.highlighted_annotation = item.data(Qt.ItemDataRole.UserRole)
        self.image_label.update()

    def delete_annotation(self):
        current_item = self.annotation_list.currentItem()
        if current_item:
            annotation = current_item.data(Qt.ItemDataRole.UserRole)
            category_name = annotation["category_name"]
            self.image_label.annotations[category_name].remove(annotation)
            self.annotation_list.takeItem(self.annotation_list.row(current_item))
            self.image_label.highlighted_annotation = None
            self.image_label.update()

    def add_annotation_to_list(self, annotation):
        class_name = annotation["category_name"]
        color = self.image_label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
        annotations = self.image_label.annotations.get(class_name, [])
        number = max([ann.get("number", 0) for ann in annotations] + [0]) + 1
        annotation["number"] = number
        area = calculate_area(annotation)
        item_text = f"{class_name} - {number:<3} Area: {area:.2f}"

        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, annotation)
        item.setForeground(color)
        self.annotation_list.addItem(item)

        # Clear the current selection
        self.annotation_list.clearSelection()
        self.image_label.highlighted_annotations.clear()
        self.image_label.update()

    def zoom_in(self):
        new_zoom = min(self.image_label.zoom_factor + 0.1, 5.0)
        self.set_zoom(new_zoom)

    def zoom_out(self):
        new_zoom = max(self.image_label.zoom_factor - 0.1, 0.1)
        self.set_zoom(new_zoom)

    def set_zoom(self, zoom_factor):
        self.image_label.set_zoom(zoom_factor)
        self.zoom_slider.setValue(int(zoom_factor * 100))
        self.image_label.update()

    def zoom_image(self):
        zoom_factor = self.zoom_slider.value() / 100
        self.set_zoom(zoom_factor)

    def disable_tools(self):
        self.polygon_button.setEnabled(False)
        self.rectangle_button.setEnabled(False)
        # self.finish_polygon_button.setEnabled(False)

    def enable_tools(self):
        self.polygon_button.setEnabled(True)
        self.rectangle_button.setEnabled(True)

    def finish_rectangle(self):
        if self.image_label.current_rectangle:
            x1, y1, x2, y2 = self.image_label.current_rectangle

            # Create a rectangle polygon from the annotation
            rectangle = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            # Define the image boundary as a rectangle
            image_boundary = Polygon(
                [
                    (0, 0),
                    (self.current_image.width(), 0),
                    (self.current_image.width(), self.current_image.height()),
                    (0, self.current_image.height()),
                ]
            )

            # Intersect the rectangle with the image boundary
            clipped_rectangle = rectangle.intersection(image_boundary)

            if clipped_rectangle.is_empty:
                QMessageBox.warning(
                    self,
                    "Invalid Annotation",
                    "The annotation is completely outside the image boundaries.",
                )
                self.image_label.current_rectangle = None
                self.image_label.update()
                return

            # Convert the clipped rectangle to a segmentation format
            if isinstance(clipped_rectangle, Polygon):
                segmentation = [
                    coord
                    for point in clipped_rectangle.exterior.coords
                    for coord in point
                ]
            elif isinstance(clipped_rectangle, MultiPolygon):
                largest_polygon = max(clipped_rectangle.geoms, key=lambda p: p.area)
                segmentation = [
                    coord
                    for point in largest_polygon.exterior.coords
                    for coord in point
                ]
            else:
                QMessageBox.warning(
                    self, "Invalid Annotation", "The annotation could not be processed."
                )
                return

            self.record_annotation_history("adding a box")
            new_annotation = {
                "segmentation": segmentation,
                "category_id": self.class_mapping[self.current_class],
                "category_name": self.current_class,
            }
            self.image_label.annotations.setdefault(self.current_class, []).append(
                new_annotation
            )
            self.add_annotation_to_list(new_annotation)
            self.image_label.start_point = None
            self.image_label.end_point = None
            self.image_label.current_rectangle = None
            self.image_label.update()

            # Save the current annotations
            self.save_current_annotations()

            # Update the slice list colors
            self.update_slice_list_colors()
            self.auto_save()

    def enter_edit_mode(self, annotation):
        self.editing_mode = True
        self.disable_tools()

        QMessageBox.information(
            self,
            "Edit Mode",
            "You are now in edit mode. Click and drag points to move them, Shift+Click to delete points, or click on edges to add new points.",
        )

    def exit_edit_mode(self):
        self.editing_mode = False
        self.enable_tools()

        self.image_label.editing_polygon = None
        self.image_label.editing_point_index = None
        self.image_label.hover_point_index = None
        self.update_annotation_list()
        self.image_label.update()

    def highlight_annotation_in_list(self, annotation):
        for i in range(self.annotation_list.count()):
            item = self.annotation_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == annotation:
                self.annotation_list.setCurrentItem(item)
                break

    def select_annotation_in_list(self, annotation):
        for i in range(self.annotation_list.count()):
            item = self.annotation_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == annotation:
                self.annotation_list.setCurrentItem(item)
                break

    ################################################################

    def setup_yolo_menu(self):
        yolo_menu = self.menuBar().addMenu("&YOLO (beta)")

        # Training submenu
        training_submenu = yolo_menu.addMenu("Training")

        load_pretrained_action = QAction("Load Pre-trained Model", self)
        load_pretrained_action.triggered.connect(self.load_yolo_model)
        training_submenu.addAction(load_pretrained_action)

        prepare_data_action = QAction("Prepare YOLO Dataset", self)
        prepare_data_action.triggered.connect(self.prepare_yolo_dataset)
        training_submenu.addAction(prepare_data_action)

        load_yaml_action = QAction("Load Dataset YAML", self)
        load_yaml_action.triggered.connect(self.load_yolo_yaml)
        training_submenu.addAction(load_yaml_action)

        train_action = QAction("Train Model", self)
        train_action.triggered.connect(self.show_train_dialog)
        training_submenu.addAction(train_action)

        save_model_action = QAction("Save Model", self)
        save_model_action.triggered.connect(self.save_yolo_model)
        training_submenu.addAction(save_model_action)

        # Prediction Settings submenu
        prediction_submenu = yolo_menu.addMenu("Prediction Settings")

        load_model_action = QAction("Load Model", self)
        load_model_action.triggered.connect(self.load_prediction_model)
        prediction_submenu.addAction(load_model_action)

        set_threshold_action = QAction("Set Confidence Threshold", self)
        set_threshold_action.triggered.connect(self.set_confidence_threshold)
        prediction_submenu.addAction(set_threshold_action)

    def load_yolo_model(self):
        if not hasattr(self, "current_project_dir"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer:
            self.initialize_yolo_trainer()

        if self.yolo_trainer.load_model():
            QMessageBox.information(
                self, "Model Loaded", "YOLO model loaded successfully."
            )
        else:
            QMessageBox.warning(self, "Load Cancelled", "Model loading was cancelled.")

    def prepare_yolo_dataset(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer:
            self.initialize_yolo_trainer()

        try:
            yaml_path = self.yolo_trainer.prepare_dataset()
            QMessageBox.information(
                self,
                "Dataset Prepared",
                f"YOLO dataset prepared successfully. YAML file: {yaml_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while preparing the dataset: {str(e)}",
            )

    def load_yolo_yaml(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer:
            self.initialize_yolo_trainer()

        try:
            if self.yolo_trainer.load_yaml():
                QMessageBox.information(
                    self, "YAML Loaded", "Dataset YAML loaded successfully."
                )
            else:
                QMessageBox.warning(
                    self, "Load Cancelled", "YAML loading was cancelled."
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while loading the YAML file: {str(e)}",
            )

    def save_yolo_model(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer or not self.yolo_trainer.model:
            QMessageBox.warning(
                self, "No Model", "Please train or load a YOLO model first."
            )
            return

        try:
            if self.yolo_trainer.save_model():
                QMessageBox.information(
                    self, "Model Saved", "YOLO model saved successfully."
                )
            else:
                QMessageBox.warning(
                    self, "Save Cancelled", "Model saving was cancelled."
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while saving the model: {str(e)}"
            )

    def load_prediction_model(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer:
            self.initialize_yolo_trainer()

        dialog = LoadPredictionModelDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            model_path = dialog.model_path
            yaml_path = dialog.yaml_path
            if model_path and yaml_path:
                try:
                    result, message = self.yolo_trainer.load_prediction_model(
                        model_path, yaml_path
                    )
                    if result:
                        QMessageBox.information(
                            self,
                            "Model Loaded",
                            "YOLO model and YAML file loaded successfully for prediction.",
                        )
                        if message:
                            QMessageBox.warning(self, "Class Mismatch Warning", message)
                    else:
                        QMessageBox.critical(
                            self,
                            "Error Loading Model",
                            f"Could not load the model or YAML file: {message}",
                        )
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            else:
                QMessageBox.warning(
                    self,
                    "Files Required",
                    "Both model and YAML files are required for prediction.",
                )

    def show_train_dialog(self):
        if not self.yolo_trainer:
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return
        if not self.yolo_trainer.model:
            QMessageBox.warning(
                self, "No Model", "Please load a pre-trained model first."
            )
            return
        if not self.yolo_trainer.yaml_path:
            QMessageBox.warning(
                self, "No Dataset", "Please prepare or load a dataset YAML first."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Train YOLO Model")
        layout = QVBoxLayout()

        epochs_label = QLabel("Number of Epochs:")
        epochs_input = QLineEdit("100")
        layout.addWidget(epochs_label)
        layout.addWidget(epochs_input)

        imgsz_label = QLabel("Image Size:")
        imgsz_input = QLineEdit("640")
        layout.addWidget(imgsz_label)
        layout.addWidget(imgsz_input)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            epochs = int(epochs_input.text())
            imgsz = int(imgsz_input.text())
            self.start_training(epochs, imgsz)

    def initialize_yolo_trainer(self):
        if hasattr(self, "current_project_dir"):
            self.yolo_trainer = YOLOTrainer(self.current_project_dir, self)
        else:
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )

    def start_training(self, epochs, imgsz):
        if not hasattr(self, "training_dialog"):
            self.training_dialog = TrainingInfoDialog(self)
        self.training_dialog.show()

        self.yolo_trainer.progress_signal.connect(self.training_dialog.update_info)
        self.yolo_trainer.set_progress_callback(self.training_dialog.update_info)
        self.training_dialog.stop_signal.connect(self.yolo_trainer.stop_training_signal)

        self.training_thread = TrainingThread(self.yolo_trainer, epochs, imgsz)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.start()

    def training_finished(self, results):
        self.training_dialog.stop_button.setEnabled(True)
        self.training_dialog.stop_button.setText("Stop Training")
        self.yolo_trainer.progress_signal.disconnect(self.training_dialog.update_info)
        self.training_dialog.stop_signal.disconnect(
            self.yolo_trainer.stop_training_signal
        )

        if isinstance(results, str):
            QMessageBox.critical(
                self, "Training Error", f"An error occurred during training: {results}"
            )
        else:
            QMessageBox.information(
                self, "Training Complete", "YOLO model training completed successfully."
            )

    def set_confidence_threshold(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer:
            self.initialize_yolo_trainer()

        current_threshold = self.yolo_trainer.conf_threshold
        new_threshold, ok = QInputDialog.getDouble(
            self,
            "Set Confidence Threshold",
            "Enter confidence threshold (0-1):",
            current_threshold,
            0,
            1,
            2,
        )
        if ok:
            self.yolo_trainer.set_conf_threshold(new_threshold)
            QMessageBox.information(
                self,
                "Threshold Updated",
                f"Confidence threshold set to {new_threshold}",
            )

    def show_predict_dialog(self):
        if not self.yolo_trainer or not self.yolo_trainer.model:
            QMessageBox.warning(self, "No Model", "Please load a YOLO model first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Predict with YOLO Model")
        layout = QVBoxLayout()

        image_list = QListWidget()
        for image_name in self.image_paths.keys():
            image_list.addItem(image_name)
        layout.addWidget(QLabel("Select images for prediction:"))
        layout.addWidget(image_list)

        conf_label = QLabel("Confidence Threshold:")
        conf_input = QDoubleSpinBox()
        conf_input.setRange(0, 1)
        conf_input.setSingleStep(0.01)
        conf_input.setValue(self.yolo_trainer.conf_threshold)
        layout.addWidget(conf_label)
        layout.addWidget(conf_input)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        predict_button = QPushButton("Predict")
        button_box.addButton(predict_button, QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_images = [item.text() for item in image_list.selectedItems()]
            conf = conf_input.value()
            self.yolo_trainer.set_conf_threshold(conf)
            self.run_predictions(selected_images)

    def run_predictions(self, selected_images):
        for image_name in selected_images:
            image_path = self.image_paths[image_name]
            results = self.yolo_trainer.predict(image_path)
            self.process_yolo_results(results, image_name)

    def process_yolo_results(self, results, image_name):
        image_path = self.image_paths[image_name]
        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.warning(self, "Error", f"Failed to load image: {image_name}")
            return
        original_height, original_width = image.shape[:2]

        temp_annotations = {}

        try:
            results, input_size, original_size = (
                results  # Unpack the results, input size, and original size
            )
            input_height, input_width = input_size
            orig_height, orig_width = original_size

            scale_x = original_width / orig_width
            scale_y = original_height / orig_height

            for result in results:
                boxes = result.boxes
                masks = result.masks

                if masks is None:
                    print(f"No masks found for {image_name}")
                    continue

                for mask, box in zip(masks, boxes):
                    try:
                        class_id = int(box.cls)
                        class_name = self.yolo_trainer.class_names[class_id]
                        score = float(box.conf)

                        mask_array = mask.data.cpu().numpy()[0]
                        # Resize mask to original image size
                        mask_array = cv2.resize(mask_array, (orig_width, orig_height))
                        contours, _ = cv2.findContours(
                            (mask_array > 0.5).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )

                        if contours:
                            epsilon = 0.005 * cv2.arcLength(contours[0], True)
                            approx = cv2.approxPolyDP(contours[0], epsilon, True)
                            polygon = approx.flatten().tolist()

                            # Scale the polygon coordinates
                            scaled_polygon = []
                            for i in range(0, len(polygon), 2):
                                x = polygon[i] * scale_x
                                y = polygon[i + 1] * scale_y
                                scaled_polygon.extend([x, y])

                            temp_class_name = f"Temp-{class_name}"
                            if temp_class_name not in temp_annotations:
                                temp_annotations[temp_class_name] = []

                            temp_annotation = {
                                "segmentation": scaled_polygon,
                                "category_name": temp_class_name,
                                "score": score,
                                "temp": True,
                            }
                            temp_annotations[temp_class_name].append(temp_annotation)
                    except IndexError:
                        QMessageBox.warning(
                            self,
                            "Class Mismatch",
                            "There is a mismatch between the model and the YAML file classes. "
                            "Please check that the YAML file corresponds to the loaded model.",
                        )
                        return

        except Exception as e:
            QMessageBox.warning(
                self,
                "Prediction Error",
                f"An error occurred during prediction: {str(e)}\n\n"
                "This might be due to a mismatch between the model and the YAML file classes. "
                "Please check that the YAML file corresponds to the loaded model.",
            )
            return

        self.add_temp_classes(temp_annotations)
        self.update_class_list()
        self.image_label.update()

        if temp_annotations:
            total_predictions = sum(len(anns) for anns in temp_annotations.values())
            QMessageBox.information(
                self,
                "Review Predictions",
                f"Found {total_predictions} predictions for {len(temp_annotations)} classes.\n"
                "Use class visibility checkboxes to review.\n"
                "Press Enter to accept or Esc to reject visible predictions.",
            )
        else:
            QMessageBox.information(
                self, "No Predictions", "No predictions were found for this image."
            )

        # Deactivate SAM tool
        self.deactivate_sam_magic_wand()

    def add_temp_classes(self, temp_annotations):
        for temp_class_name, annotations in temp_annotations.items():
            if temp_class_name not in self.image_label.class_colors:
                color = QColor(
                    Qt.GlobalColor(len(self.image_label.class_colors) % 16 + 7)
                )
                self.image_label.class_colors[temp_class_name] = color
            self.image_label.annotations[temp_class_name] = annotations

        self.update_class_list()

    def verify_current_class(self):
        if self.current_class is None or self.current_class not in self.class_mapping:
            if self.class_list.count() > 0:
                self.class_list.setCurrentRow(0)
                self.on_class_selected(self.class_list.item(0))
            else:
                self.current_class = None
                self.disable_annotation_tools()

    def accept_visible_temp_classes(self):
        visible_temp_classes = [
            item.text()
            for item in self.class_list.findItems("Temp-*", Qt.MatchFlag.MatchWildcard)
            if item.checkState() == Qt.CheckState.Checked
        ]
        if visible_temp_classes:
            self.record_annotation_history("accepting proposed masks")

        for temp_class_name in visible_temp_classes:
            permanent_class_name = temp_class_name[5:]  # Remove "Temp-" prefix
            if permanent_class_name not in self.image_label.annotations:
                self.add_class(
                    permanent_class_name, self.image_label.class_colors[temp_class_name]
                )

            # Get the current maximum number for this class
            current_max = max(
                [
                    ann.get("number", 0)
                    for ann in self.image_label.annotations.get(
                        permanent_class_name, []
                    )
                ]
                + [0]
            )

            for annotation in self.image_label.annotations[temp_class_name]:
                current_max += 1
                annotation["category_name"] = permanent_class_name
                annotation["number"] = current_max
                self.image_label.annotations.setdefault(
                    permanent_class_name, []
                ).append(annotation)

            del self.image_label.annotations[temp_class_name]
            del self.image_label.class_colors[temp_class_name]

        self.update_class_list()
        current_name = self.current_slice or self.image_file_name
        self.all_annotations[current_name] = self.image_label.annotations
        self.update_annotation_list()
        self.image_label.update()
        self.save_current_annotations()

        # Select the first primary class
        self.select_first_primary_class()
        self.verify_current_class()

        QMessageBox.information(
            self,
            "Annotations Accepted",
            "Temporary annotations have been accepted and added to the permanent classes.",
        )

    def select_first_primary_class(self):
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if not item.text().startswith("Temp-"):
                self.class_list.setCurrentItem(item)
                self.on_class_selected(item)
                break

    def reject_visible_temp_classes(self):
        visible_temp_classes = [
            item.text()
            for item in self.class_list.findItems("Temp-*", Qt.MatchFlag.MatchWildcard)
            if item.checkState() == Qt.CheckState.Checked
        ]

        for temp_class_name in visible_temp_classes:
            if temp_class_name in self.image_label.annotations:
                del self.image_label.annotations[temp_class_name]
            if temp_class_name in self.image_label.class_colors:
                del self.image_label.class_colors[temp_class_name]

        self.update_class_list()
        self.image_label.update()

    def is_class_visible(self, class_name):
        items = self.class_list.findItems(class_name, Qt.MatchFlag.MatchExactly)
        if items:
            return items[0].checkState() == Qt.CheckState.Checked
        return False

    def check_temp_annotations(self):
        temp_classes = [
            class_name
            for class_name in self.image_label.annotations.keys()
            if class_name.startswith("Temp-")
        ]
        if temp_classes:
            reply = QMessageBox.question(
                self,
                "Temporary Annotations",
                "There are temporary annotations that will be discarded. Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                for temp_class in temp_classes:
                    del self.image_label.annotations[temp_class]
                    del self.image_label.class_colors[temp_class]
                self.update_class_list()
                self.update_annotation_list()
                return True
            return False
        return True

    def _apply_welding_class_preset(self, classes, title):
        if self._reject_while_sam3_busy():
            return
        if not self.image_label.check_unsaved_changes():
            return

        added = 0
        updated = 0
        for class_name, color in classes:
            if class_name in self.class_mapping:
                current = self.image_label.class_colors.get(class_name)
                if current is None or current.rgba() != color.rgba():
                    self.image_label.class_colors[class_name] = QColor(color)
                    updated += 1
            else:
                self.add_class(class_name, QColor(color))
                added += 1

        self.update_class_list()
        self.update_annotation_list()
        self.image_label.update()
        self.auto_save()
        self.show_info(
            title,
            f"Preset ready: {added} class(es) added and {updated} color(s) updated.",
        )

    def add_default_welding_classes(self):
        self._apply_welding_class_preset(
            ER70S6_CLASSES,
            "ER70S-6 Full Arc Classes",
        )

    def add_cavitar_welding_classes(self):
        self._apply_welding_class_preset(
            ER70S6_CAVITAR_CLASSES,
            "ER70S-6 CAVITAR Classes",
        )

    def show_er70s6_protocol(self):
        QMessageBox.information(self, "ER70S-6 Labeling Protocol", ER70S6_PROTOCOL)

    def _application_cache_root(self):
        cache_root = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.CacheLocation
        )
        if not cache_root:
            cache_root = os.path.join(os.getcwd(), ".video_cache")
        return Path(cache_root)

    def _video_clip_cache_root(self):
        return self._application_cache_root() / "video_clips"

    def _sam3_tracker_cache_root(self):
        return self._application_cache_root() / "tracker_sessions"

    def _reset_sam3_video_state(self, unload=False):
        if self.sam3_tracker is not None:
            if unload:
                self.sam3_tracker.unload()
                self.sam3_tracker = None
            else:
                self.sam3_tracker.close_session()
        self._clear_sam3_frame_workspace()
        self.frame_sequence = None
        self.active_video_session_id = None

    def _clear_sam3_frame_workspace(self):
        if self._sam3_frame_workspace:
            cleanup_managed_video_directory(
                self._sam3_frame_workspace,
                self._sam3_frame_workspace_root,
                TRACKER_WORKSPACE_MARKER,
            )
            self._sam3_frame_workspace = None
            self._sam3_frame_workspace_root = None

    def _clear_video_sessions(self, clean_clip_caches=False):
        if clean_clip_caches:
            self._cleanup_video_clip_caches()
        self.video_sessions = {}
        self._video_session_by_frame = {}
        self.active_video_session_id = None

    def _rebuild_video_session_frame_index(self):
        frame_owners = {}
        empty_sessions = []
        for session_id, session in self.video_sessions.items():
            owned_frames = []
            for frame in session.get("frames", []):
                name = frame.get("name")
                frame_key = _canonical_image_name(name) if name else ""
                if not frame_key or frame_key in frame_owners:
                    continue
                frame_owners[frame_key] = session_id
                owned_frames.append(frame)
            session["frames"] = owned_frames
            if not owned_frames:
                empty_sessions.append(session_id)
        for session_id in empty_sessions:
            self.video_sessions.pop(session_id, None)
        self._video_session_by_frame = frame_owners

    def _prune_video_sessions_to_project_images(self):
        """Drop every persisted frame that no longer exists in the project."""
        project_names = {
            _canonical_image_name(name): name
            for name, path in self.image_paths.items()
            if path and os.path.exists(path)
        }
        for session in self.video_sessions.values():
            kept_frames = []
            for frame in session.get("frames", []):
                project_name = project_names.get(
                    _canonical_image_name(frame.get("name", ""))
                )
                if not project_name:
                    continue
                kept_frames.append({**frame, "name": project_name})
            session["frames"] = kept_frames
        self._rebuild_video_session_frame_index()
        if self.active_video_session_id not in self.video_sessions:
            self.active_video_session_id = next(iter(self.video_sessions), None)

    def _video_sessions_for_save(self):
        project_frame_keys = {
            _canonical_image_name(name)
            for name, path in self.image_paths.items()
            if path and os.path.exists(path)
        }
        saved_sessions = {}
        for session_id, source_session in self.video_sessions.items():
            session = copy.deepcopy(source_session)
            session.pop("cache_dir", None)
            session["frames"] = [
                frame
                for frame in session.get("frames", [])
                if _canonical_image_name(frame.get("name", ""))
                in project_frame_keys
            ]
            if session["frames"]:
                saved_sessions[session_id] = session
        return saved_sessions

    def _restore_active_frame_sequence(self):
        self.frame_sequence = None
        ImageAnnotator._prune_video_sessions_to_project_images(self)
        session = self.video_sessions.get(self.active_video_session_id)
        if not session:
            self.active_video_session_id = None
            self._rebuild_video_session_frame_index()
            return

        project_names = {
            _canonical_image_name(name): name for name in self.image_paths
        }
        frame_paths = []
        source_indices = []
        kept_frames = []
        for frame_data in session.get("frames", []):
            name = frame_data.get("name")
            project_name = project_names.get(_canonical_image_name(name or ""))
            path = self.image_paths.get(project_name) if project_name else None
            if not name or not path or not os.path.exists(path):
                continue
            if project_name != name:
                frame_data = {**frame_data, "name": project_name}
            frame_paths.append(path)
            source_indices.append(frame_data.get("source_index"))
            kept_frames.append(frame_data)

        session["frames"] = kept_frames
        if not frame_paths:
            self._rebuild_video_session_frame_index()
            if self.active_video_session_id not in self.video_sessions:
                self.active_video_session_id = next(iter(self.video_sessions), None)
                if self.active_video_session_id:
                    self._restore_active_frame_sequence()
            return

        self.frame_sequence = FrameSequence.from_paths(
            Path(frame_paths[0]).parent,
            frame_paths,
            source_indices,
        )
        self._rebuild_video_session_frame_index()
        if self.active_video_session_id not in self.video_sessions:
            self.active_video_session_id = next(iter(self.video_sessions), None)
            self._restore_active_frame_sequence()

    def _activate_video_session_for_frame(self, frame_name):
        matching_session_id = self._video_session_by_frame.get(
            _canonical_image_name(frame_name)
        )
        if not matching_session_id:
            return
        if matching_session_id != self.active_video_session_id:
            self._reset_sam3_video_state()
            self.active_video_session_id = matching_session_id
        if self.frame_sequence is None:
            self._restore_active_frame_sequence()

    def _remove_frame_from_video_sessions(self, frame_name):
        frame_key = _canonical_image_name(frame_name)
        active_session_id = self.active_video_session_id
        active_session = self.video_sessions.get(active_session_id, {})
        active_sequence_changed = any(
            _canonical_image_name(frame.get("name", "")) == frame_key
            for frame in active_session.get("frames", [])
        )
        if active_sequence_changed:
            self._reset_sam3_video_state()

        empty_sessions = []
        for session_id, session in self.video_sessions.items():
            session["frames"] = [
                frame
                for frame in session.get("frames", [])
                if _canonical_image_name(frame.get("name", "")) != frame_key
            ]
            if not session["frames"]:
                empty_sessions.append(session_id)
        for session_id in empty_sessions:
            self.video_sessions.pop(session_id, None)
        self._rebuild_video_session_frame_index()

        if active_session_id in self.video_sessions:
            self.active_video_session_id = active_session_id
            self._restore_active_frame_sequence()
        elif self.active_video_session_id not in self.video_sessions:
            self.active_video_session_id = None
            self.frame_sequence = None

    def _cleanup_video_clip_caches(self):
        allowed_root = self._video_clip_cache_root()
        for session in self.video_sessions.values():
            cache_dir = session.get("cache_dir")
            if not cache_dir:
                continue
            if cleanup_managed_video_directory(cache_dir, allowed_root):
                session.pop("cache_dir", None)

    def _sam3_frame_name_conflicts(self, frame_sequence):
        conflicts = []
        project_names = {
            _canonical_image_name(name): name for name in self.image_paths
        }
        seen_frame_keys = set()
        for frame in frame_sequence.frames:
            frame_key = _canonical_image_name(frame.name)
            if frame_key in seen_frame_keys:
                conflicts.append(frame.name)
                continue
            seen_frame_keys.add(frame_key)
            if frame_key in getattr(self, "_video_session_by_frame", {}):
                conflicts.append(frame.name)
                continue
            existing_name = project_names.get(frame_key)
            if existing_name is not None and existing_name != frame.name:
                conflicts.append(frame.name)
                continue
            existing_path = self.image_paths.get(existing_name)
            if not existing_path:
                continue
            existing_path = Path(existing_path)
            if existing_path.resolve() == frame.path.resolve():
                continue
            try:
                is_same_frame = filecmp.cmp(existing_path, frame.path, shallow=False)
            except OSError:
                is_same_frame = False
            if not is_same_frame:
                conflicts.append(frame.name)
        return conflicts

    def _clear_sam3_tracks_from_sources(self, source_frame, objects_to_track):
        """Remove prior generated results before replacing a tracking run.

        Snapshots each frame it is about to strip. This deletion is part
        of the tracking edit, and it reaches frames the results loop may
        never touch, so without a snapshot here undo on those frames
        would restore a state from *after* the deletion — losing the
        previous run's masks with no way back.
        """
        for frame_name, frame_annotations in self.all_annotations.items():
            for class_name, source_id in objects_to_track.values():
                annotations = frame_annotations.get(class_name)
                if annotations is None:
                    continue
                remaining = [
                    annotation
                    for annotation in annotations
                    if not (
                        annotation.get("source") == "sam3_track"
                        and annotation.get("sam3_source_frame") == source_frame
                        and annotation.get("sam3_source_id") == source_id
                    )
                ]
                if len(remaining) != len(annotations):
                    # Looked up defensively: the SAM 3 guard tests call
                    # this method unbound against a lightweight stand-in.
                    recorder = getattr(
                        self, "_record_tracking_history_once", None
                    )
                    if callable(recorder):
                        recorder(frame_name)
                annotations[:] = remaining

    def _record_tracking_history_once(self, frame_name):
        """Snapshot a frame the first time a tracking run touches it.

        One tracking run clears the previous run's masks and then writes
        new ones, and both halves reach the same frames. Recording each
        frame only once makes the whole run a single undo step, instead
        of leaving the annotator to press Ctrl+Z twice per frame and see
        a half-cleared state in between.
        """
        recorded = getattr(self, "_sam3_history_recorded", None)
        if recorded is None:
            recorded = self._sam3_history_recorded = set()
        if frame_name in recorded:
            return
        recorded.add(frame_name)
        self.record_annotation_history("SAM 3 tracking", frame_name)

    def open_frame_folder(self):
        if self._reject_while_sam3_busy():
            return
        if not self.image_label.check_unsaved_changes():
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Video Frame Folder")
        if folder:
            session_id = None
            added_names = []
            rollback_paths = []
            previous_active_session_id = None
            previous_image_name = None
            try:
                frame_sequence = FrameSequence.from_folder(folder)
                conflicts = self._sam3_frame_name_conflicts(frame_sequence)
                if conflicts:
                    preview = ", ".join(conflicts[:5])
                    raise ValueError(
                        "Frame names collide with images already loaded from "
                        f"another folder: {preview}"
                    )
                if not self.save_project(show_message=False):
                    detail = getattr(self, "_last_project_save_error", None)
                    QMessageBox.critical(
                        self,
                        "Project Save Required",
                        "The project must be saved before importing frames"
                        f"{f': {detail}' if detail else '.'}",
                    )
                    return

                previous_active_session_id = self.active_video_session_id
                previous_image_name = self.image_file_name
                project_images_dir = Path(self.current_project_dir) / "images"
                rollback_paths = [
                    project_images_dir / frame.name
                    for frame in frame_sequence.frames
                    if not (project_images_dir / frame.name).exists()
                ]
                self._reset_sam3_video_state()
                session_id = uuid.uuid4().hex[:12]
                while session_id in self.video_sessions:
                    session_id = uuid.uuid4().hex[:12]
                self.frame_sequence = frame_sequence
                self.active_video_session_id = session_id
                self.video_sessions[session_id] = {
                    "source_type": "frame_folder",
                    "source_path": str(frame_sequence.folder.resolve()),
                    "frames": [
                        {
                            "name": frame.name,
                            "source_index": frame.source_index,
                        }
                        for frame in frame_sequence.frames
                    ],
                }
                self._rebuild_video_session_frame_index()
                image_paths = [str(f.path) for f in frame_sequence.frames]
                added_names = self.add_images_to_list(
                    image_paths,
                    auto_save=False,
                )
                project_names = {
                    _canonical_image_name(name): name for name in self.image_paths
                }
                accepted_names = [
                    project_names.get(_canonical_image_name(frame.name))
                    for frame in frame_sequence.frames
                ]
                if any(name is None for name in accepted_names):
                    raise RuntimeError(
                        "Not all folder frames were registered with the project."
                    )
                if not self.save_project(show_message=False):
                    detail = getattr(self, "_last_project_save_error", None)
                    message = "The frame folder could not be saved to the project."
                    raise RuntimeError(f"{message} {detail}" if detail else message)
            except Exception as e:
                if session_id and session_id in self.video_sessions:
                    self._rollback_video_session_import(
                        session_id,
                        added_names,
                        rollback_paths,
                        previous_active_session_id,
                        previous_image_name,
                    )
                QMessageBox.critical(self, "Error", f"Failed to load frame folder: {str(e)}")

    def open_video_clip(self):
        if self._reject_while_sam3_busy():
            return
        if not self.image_label.check_unsaved_changes():
            return

        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All Files (*)",
        )
        if not video_path:
            return

        try:
            metadata = probe_video(video_path)
        except Exception as exc:
            QMessageBox.critical(self, "Video Error", str(exc))
            return

        dialog = VideoClipDialog(metadata, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selection = dialog.selection()

        try:
            validate_video_source(metadata)
        except Exception as exc:
            QMessageBox.critical(self, "Video Error", str(exc))
            return

        # Establish the project before doing expensive work. Imported frames
        # are copied into its images directory by the worker thread.
        if not self.save_project(show_message=False):
            detail = getattr(self, "_last_project_save_error", None)
            QMessageBox.critical(
                self,
                "Project Save Required",
                "The project must be saved before importing a video"
                f"{f': {detail}' if detail else '.'}",
            )
            return

        try:
            video_cache_root = self._video_clip_cache_root()
            output_dir = video_clip_cache_directory(
                metadata.path,
                selection,
                video_cache_root,
            )
            session_id = uuid.uuid4().hex[:12]
            output_dir = output_dir.with_name(f"{output_dir.name}_{session_id}")
            project_images_dir = Path(self.current_project_dir) / "images"

            total = selection.output_frame_count(metadata.frame_count)
            progress = QProgressDialog(
                "Importing selected video frames...",
                "Cancel",
                0,
                total * 2,
                self,
            )
            progress.setWindowTitle("Open Video Clip")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(False)
            progress.setAutoReset(False)

            worker = VideoExtractionThread(
                metadata,
                selection,
                output_dir,
                project_images_dir=project_images_dir,
                cache_root=video_cache_root,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Video Error", str(exc))
            return

        def update_extraction_progress(completed, frame_total):
            progress.setMaximum(frame_total)
            progress.setValue(completed)

        cancel_requested = {"value": False}

        def request_extraction_cancel():
            cancel_requested["value"] = True
            worker.requestInterruption()

        worker.progress_changed.connect(update_extraction_progress)
        progress.canceled.connect(request_extraction_cancel)
        worker.finished.connect(progress.accept)
        worker.start()
        progress.exec()
        if cancel_requested["value"]:
            worker.requestInterruption()
        worker.wait()

        if cancel_requested["value"] or worker.cancelled:
            if worker.result is not None:
                for frame in worker.result.frames:
                    try:
                        frame.path.unlink()
                    except OSError as cleanup_error:
                        print(
                            f"Could not remove cancelled frame {frame.path}: "
                            f"{cleanup_error}"
                        )
            return
        if worker.error:
            QMessageBox.critical(
                self,
                "Video Error",
                f"Failed to extract the selected frames:\n{worker.error}",
            )
            return

        clip = worker.result
        if clip is None:
            return

        frame_sequence = FrameSequence.from_paths(
            clip.output_dir,
            [frame.path for frame in clip.frames],
            [frame.source_index for frame in clip.frames],
        )
        conflicts = self._sam3_frame_name_conflicts(frame_sequence)
        if conflicts:
            preview = ", ".join(conflicts[:5])
            for frame in clip.frames:
                try:
                    frame.path.unlink()
                except OSError as cleanup_error:
                    print(
                        f"Could not remove conflicting frame {frame.path}: "
                        f"{cleanup_error}"
                    )
            QMessageBox.critical(
                self,
                "Video Error",
                "Extracted frame names collide with different images already "
                f"in this project: {preview}",
            )
            return

        previous_active_session_id = self.active_video_session_id
        previous_image_name = self.image_file_name
        self._reset_sam3_video_state()
        self.frame_sequence = frame_sequence
        self.active_video_session_id = session_id
        self.video_sessions[session_id] = {
            "source_type": "video",
            "source_path": str(metadata.path),
            "frame_count": metadata.frame_count,
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "start_frame": selection.start_frame,
            "end_frame": selection.end_frame,
            "stride": selection.stride,
            "frames": [
                {
                    "name": frame.name,
                    "source_index": frame.source_index,
                }
                for frame in frame_sequence.frames
            ],
        }
        self._rebuild_video_session_frame_index()
        added_names = []
        try:
            added_names = self.add_images_to_list(
                [str(frame.path) for frame in frame_sequence.frames],
                known_size=(metadata.width, metadata.height),
                auto_save=False,
            )
            if len(added_names) != len(frame_sequence.frames):
                raise RuntimeError("Not all extracted frames were added to the project.")
            if not self.save_project(show_message=False):
                detail = getattr(self, "_last_project_save_error", None)
                message = "The imported clip could not be saved to the project."
                raise RuntimeError(f"{message} {detail}" if detail else message)
        except Exception as exc:
            self._rollback_video_session_import(
                session_id,
                added_names,
                [frame.path for frame in frame_sequence.frames],
                previous_active_session_id,
                previous_image_name,
            )
            QMessageBox.critical(
                self,
                "Video Error",
                f"The video import was rolled back:\n{exc}",
            )
            return

        self.show_info(
            "Video Clip Loaded",
            f"Loaded {len(frame_sequence.frames):,} frames from "
            f"{metadata.path.name}. Use A and D to move frame by frame.",
        )

    def _rollback_video_session_import(
        self,
        session_id,
        added_names,
        frame_paths,
        previous_active_session_id,
        previous_image_name=None,
    ):
        if self.active_video_session_id == session_id:
            self._reset_sam3_video_state()
        self.video_sessions.pop(session_id, None)
        self._rebuild_video_session_frame_index()

        for name in added_names:
            items = self.image_list.findItems(name, Qt.MatchFlag.MatchExactly)
            if items:
                self._remove_image_item(items[0], select_next=False)
        for frame_path in frame_paths:
            try:
                Path(frame_path).unlink()
            except OSError as cleanup_error:
                print(
                    f"Could not remove rolled-back frame {frame_path}: "
                    f"{cleanup_error}"
                )

        if previous_active_session_id in self.video_sessions:
            self.active_video_session_id = previous_active_session_id
            self._restore_active_frame_sequence()
        if self.image_list.count() > 0:
            previous_items = (
                self.image_list.findItems(
                    previous_image_name,
                    Qt.MatchFlag.MatchExactly,
                )
                if previous_image_name
                else []
            )
            next_item = previous_items[0] if previous_items else self.image_list.item(0)
            self.image_list.setCurrentItem(next_item)
            self.switch_image(next_item)

    def go_to_next_frame(self):
        if self._sam3_inference_in_flight:
            return
        if not self.frame_sequence:
            self._step_through_image_list(1)
            return
        frame_idx = self.frame_sequence.index_for_name(self.image_file_name)
        next_name = self.frame_sequence.name_for_index(frame_idx + 1) if frame_idx is not None else None
        if next_name:
            self._navigate_to_image_or_slice(next_name)

    def _step_through_image_list(self, offset):
        """Move through the frame list when there is no video sequence.

        A / D previously did nothing at all for a project built from
        still images, while the canvas header and the frames panel both
        advertise them — so the app looked broken to anyone who had not
        opened a video clip. Frames hidden by the filter are skipped, so
        the keys walk what the annotator can actually see.
        """
        total = self.image_list.count()
        if total == 0:
            return
        row = self.image_list.currentRow()
        if row < 0:
            row = 0 if offset > 0 else total - 1
            target = row
        else:
            target = row + offset
        while 0 <= target < total and self.image_list.item(target).isHidden():
            target += offset
        if not (0 <= target < total):
            return
        # setCurrentItem fires currentRowChanged, which is already wired
        # to switch_image — calling it again here would load the frame
        # twice on every keypress.
        self.image_list.setCurrentItem(self.image_list.item(target))

    def go_to_previous_frame(self):
        if self._sam3_inference_in_flight:
            return
        if not self.frame_sequence:
            self._step_through_image_list(-1)
            return
        frame_idx = self.frame_sequence.index_for_name(self.image_file_name)
        previous_name = self.frame_sequence.name_for_index(frame_idx - 1) if frame_idx is not None else None
        if previous_name:
            self._navigate_to_image_or_slice(previous_name)

    def copy_selected_annotation_to_next_frame(self):
        if self._sam3_inference_in_flight:
            return
        if not self.image_list.currentItem():
            return

        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            self.show_warning("Copy Annotation", "No annotation selected.")
            return

        if not self.frame_sequence:
            return
        current_image_name = self.image_file_name
        current_frame_idx = self.frame_sequence.index_for_name(current_image_name)
        next_image_name = (
            self.frame_sequence.name_for_index(current_frame_idx + 1)
            if current_frame_idx is not None
            else None
        )
        if not next_image_name:
            return

        self.save_current_annotations()
        # The copy lands on the *next* frame, so that is the frame whose
        # state undo has to be able to restore.
        self.record_annotation_history(
            "copying an annotation forward", next_image_name
        )
        for item in selected_items:
            annotation = copy.deepcopy(item.data(Qt.ItemDataRole.UserRole))
            class_name = annotation.get("category_name")
            if not class_name:
                continue
            for key in (
                "source",
                "sam3_source_frame",
                "sam3_source_id",
                "sam3_object_id",
                "droplet_event_id",
            ):
                annotation.pop(key, None)
            self.all_annotations.setdefault(next_image_name, {}).setdefault(
                class_name, []
            ).append(annotation)

        self.auto_save()
        self.go_to_next_frame()

    def init_sam3_tracker(self):
        if not self.frame_sequence:
            self.show_warning(
                "SAM 3 Tracker", "Please open an active video sequence first."
            )
            return
        if self._sam3_inference_in_flight:
            self.show_warning("SAM 3 Tracker", "SAM 3 is already busy.")
            return

        try:
            self._sam3_inference_in_flight = True
            self.setCursor(Qt.CursorShape.WaitCursor)
            from .sam3_tracker import SAM3Tracker

            if self.sam3_tracker is None:
                ckpt_path = Path(__file__).resolve().parents[3] / "sam3-001.pt"
                if not ckpt_path.exists():
                    selected_path, _ = QFileDialog.getOpenFileName(
                        self,
                        "Select SAM 3 Checkpoint",
                        "",
                        "PyTorch Models (*.pt *.pth)",
                    )
                    if not selected_path:
                        return
                    ckpt_path = Path(selected_path)
                self.sam3_tracker = self._run_sam3_ui_locked(
                    SAM3Tracker, str(ckpt_path)
                )

            self._prepare_sam3_frame_workspace()
            self._run_sam3_ui_locked(
                self.sam3_tracker.init_state,
                str(self._sam3_frame_workspace),
            )
            frame_count = len(self.frame_sequence.frames)
            self.show_info(
                "SAM 3 Tracker",
                f"Prepared {frame_count} loaded frame(s). Draw or select a polygon "
                "on the current frame, then track it to the end.",
            )
        except InferenceBusyError:
            self.show_warning("SAM 3 Tracker", "Another AI inference is still running.")
        except Exception as e:
            self._clear_sam3_frame_workspace()
            QMessageBox.critical(self, "SAM 3 Error", f"Failed to initialize tracker:\n{str(e)}")
        finally:
            self._sam3_inference_in_flight = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _prepare_sam3_frame_workspace(self):
        if not self.frame_sequence:
            raise ValueError("A frame sequence is required before tracking.")
        self._clear_sam3_frame_workspace()
        tracker_cache_root = self._sam3_tracker_cache_root()
        self._sam3_frame_workspace = create_tracker_frame_workspace(
            [frame.path for frame in self.frame_sequence.frames],
            tracker_cache_root,
        )
        self._sam3_frame_workspace_root = tracker_cache_root
        return self._sam3_frame_workspace

    def sam3_track_forward(self, all_objects=False):
        if self.sam3_tracker is None or not self.sam3_tracker.is_initialized:
            self.show_warning("SAM 3 Tracker", "Please initialize the tracker first.")
            return
        if self._sam3_inference_in_flight:
            self.show_warning("SAM 3 Tracker", "SAM 3 is already busy.")
            return

        current_image_name = self.image_file_name
        current_idx = self.frame_sequence.index_for_name(current_image_name)
        if current_idx is None:
            self.show_warning(
                "Tracking",
                "The current image is not part of the active video sequence.",
            )
            return

        self.save_current_annotations()
        current_annotations = self.all_annotations.get(current_image_name, {})
        if not current_annotations:
            self.show_warning("Tracking", "No annotations on current frame to track.")
            return

        if all_objects:
            targets = [
                (class_name, annotation)
                for class_name, annotations in current_annotations.items()
                for annotation in annotations
            ]
        else:
            targets = []
            for item in self.annotation_list.selectedItems():
                selected = item.data(Qt.ItemDataRole.UserRole)
                class_name = selected.get("category_name")
                if not class_name:
                    continue
                for annotation in current_annotations.get(class_name, []):
                    if annotation is selected or (
                        annotation.get("number") == selected.get("number")
                        and annotation.get("segmentation") == selected.get("segmentation")
                    ):
                        targets.append((class_name, annotation))
                        break

        objects_to_track = {}
        object_polygons = []
        for object_id, (class_name, annotation) in enumerate(targets, start=1):
            segmentation = annotation.get("segmentation")
            if not segmentation or len(segmentation) < 6:
                continue
            polygon = make_valid(Polygon(np.asarray(segmentation).reshape(-1, 2)))
            if polygon.is_empty:
                continue
            if isinstance(polygon, MultiPolygon):
                polygon = max(polygon.geoms, key=lambda item: item.area)
            if not isinstance(polygon, Polygon):
                continue
            source_id = annotation.setdefault("sam3_source_id", uuid.uuid4().hex)
            if class_name == "droplet":
                annotation["droplet_event_id"] = source_id
            polygon_points = np.asarray(polygon.exterior.coords[:-1]).reshape(-1, 2)
            object_polygons.append((object_id, polygon_points.flatten().tolist()))
            objects_to_track[object_id] = (class_name, source_id)

        if not objects_to_track:
            message = (
                "No valid polygon annotations selected."
                if not all_objects
                else "No valid polygon annotations to track."
            )
            self.show_warning("Tracking", message)
            return

        next_name = None
        # One undo step per frame for the whole run — the clear below and
        # the write loop both reach the same frames.
        self._sam3_history_recorded = set()
        try:
            self._sam3_inference_in_flight = True
            self.setCursor(Qt.CursorShape.WaitCursor)
            tracked_annotation_count = 0
            frame_size = (self.current_image.width(), self.current_image.height())
            results = self._run_sam3_ui_locked(
                self.sam3_tracker.track_polygons,
                current_idx,
                object_polygons,
                frame_size,
            )
            self._clear_sam3_tracks_from_sources(
                current_image_name, objects_to_track
            )
            for out_frame_idx, segmentations_by_object in results:
                if out_frame_idx == current_idx:
                    continue
                frame_name = self.frame_sequence.name_for_index(out_frame_idx)
                if not frame_name:
                    continue
                # Snapshot before writing. Undo is per frame, so without
                # this a Ctrl+Z on a tracked frame would restore a state
                # from before the run and silently discard the tracked
                # masks along with whatever else changed since.
                self._record_tracking_history_once(frame_name)
                frame_annotations = self.all_annotations.setdefault(frame_name, {})

                for object_id, segmentations in segmentations_by_object.items():
                    object_info = objects_to_track.get(object_id)
                    if not object_info:
                        continue
                    class_name, source_id = object_info
                    annotations = frame_annotations.setdefault(class_name, [])

                    for segmentation in segmentations:
                        annotations.append(
                            {
                                "segmentation": segmentation,
                                "category_id": self.class_mapping[class_name],
                                "category_name": class_name,
                                "source": "sam3_track",
                                "sam3_source_frame": current_image_name,
                                "sam3_source_id": source_id,
                                "sam3_object_id": object_id,
                                **(
                                    {"droplet_event_id": source_id}
                                    if class_name == "droplet"
                                    else {}
                                ),
                            }
                        )
                        tracked_annotation_count += 1

            saved = self.auto_save()
            self.update_annotation_list()
            # Unconditional: the clear above changes label state even on a
            # run that produced nothing, and the navigation at the end of
            # this method — which used to be the only refresh — is skipped
            # when tracking produced no masks, the save failed, or this is
            # the last frame.
            self.annotations_changed()
            self.image_label.update()
            if tracked_annotation_count:
                if not saved:
                    self.show_warning(
                        "SAM 3 Tracking",
                        "Tracking annotations were generated but could not be "
                        "saved to the project. They remain in memory; save the "
                        "project before navigating away.",
                    )
                    return
                next_name = self.frame_sequence.name_for_index(current_idx + 1)
                if any(
                    class_name == "droplet"
                    for class_name, _ in objects_to_track.values()
                ):
                    droplet_count = summarize_annotations(self.all_annotations)[
                        "unique_droplet_events"
                    ]
                    self.show_info(
                        "SAM 3 Tracking",
                        f"Unique large-droplet count: {droplet_count}",
                    )
            else:
                self.show_warning(
                    "SAM 3 Tracking",
                    "SAM 3 could not reproduce the selected object on "
                    "the source frame. No tracking annotations were saved.",
                )
        except InferenceBusyError:
            self.show_warning("SAM 3 Tracker", "Another AI inference is still running.")
        except Exception as e:
            QMessageBox.critical(self, "SAM 3 Tracking Error", f"Tracking failed:\n{str(e)}")
        finally:
            self._sam3_inference_in_flight = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        if next_name:
            self._navigate_to_image_or_slice(next_name)

    def remove_all_temp_annotations(self):
        for image_name in list(self.all_annotations.keys()):
            for class_name in list(self.all_annotations[image_name].keys()):
                if class_name.startswith("Temp-"):
                    del self.all_annotations[image_name][class_name]
            if not self.all_annotations[image_name]:
                del self.all_annotations[image_name]

        for class_name in list(self.image_label.class_colors.keys()):
            if class_name.startswith("Temp-"):
                del self.image_label.class_colors[class_name]

        self.update_class_list()
        self.update_annotation_list()
        self.image_label.update()
