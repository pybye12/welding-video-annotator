"""The keyboard reference, in one place.

The shortcuts were previously spread across tooltips, the help window and
CLAUDE.md, so a new lab member had no single place to look them up. This
module is the complete list, rendered by ``ShortcutReferenceDialog``
(Help > Keyboard Shortcuts, or Ctrl+/).

Sidebar help text still names the two or three keys relevant to the panel
it sits in — that is deliberate, since the point there is proximity — but
this list is what must stay complete. Add a new shortcut here as well as
wherever it is bound.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


# (section title, [(keys, what it does), ...])
SHORTCUT_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Move through frames",
        [
            ("A", "Previous frame"),
            ("D", "Next frame"),
            ("C", "Copy the selected annotation to the next frame"),
            ("Up / Down", "Previous / next slice of a stack"),
        ],
    ),
    (
        "Draw and correct",
        [
            ("P", "Polygon tool"),
            ("R", "Box tool"),
            ("B", "Paint brush"),
            ("E", "Eraser"),
            ("1 - 9", "Select label class by position"),
            ("Enter", "Finish the polygon, or accept proposed masks"),
            ("Esc", "Cancel the current shape, or reject proposed masks"),
            ("Delete", "Delete the selected annotations"),
            ("- / =", "Brush or eraser size"),
        ],
    ),
    (
        "Undo",
        [
            ("Ctrl+Z", "Undo the last annotation change on this frame"),
            ("Ctrl+Shift+Z", "Redo"),
            ("Ctrl+Y", "Redo (alternate)"),
        ],
    ),
    (
        "View",
        [
            ("Ctrl+Wheel", "Zoom to the cursor"),
            ("Ctrl+Drag", "Pan"),
            ("Ctrl+0", "Fit the frame to the viewport"),
            ("Ctrl+D", "Switch between light and dark"),
        ],
    ),
    (
        "Project",
        [
            ("Ctrl+N / Ctrl+O / Ctrl+S", "New / open / save project"),
            ("Ctrl+Shift+S", "Save project as"),
            ("Ctrl+W", "Close project"),
            ("Ctrl+I", "Project details"),
            ("Ctrl+F", "Search projects"),
            ("Ctrl+Alt+S", "Annotation statistics"),
            ("F1", "Help"),
            ("Ctrl+/", "This shortcut list"),
            ("F2", "Break time"),
        ],
    ),
]


class ShortcutReferenceDialog(QDialog):
    """A read-only, scrollable list of every shortcut the app responds to."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(430, 520)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 12)
        outer.setSpacing(10)

        heading = QLabel("Keyboard Shortcuts")
        heading.setProperty("class", "dialog-title")
        outer.addWidget(heading)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 8, 0)
        page_layout.setSpacing(14)

        for title, rows in SHORTCUT_SECTIONS:
            section = QLabel(title.upper())
            section.setProperty("class", "eyebrow")
            page_layout.addWidget(section)

            grid_host = QWidget()
            grid = QGridLayout(grid_host)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(4)
            grid.setColumnStretch(1, 1)
            for row, (keys, description) in enumerate(rows):
                key_label = QLabel(keys)
                key_label.setProperty("class", "mono")
                key_label.setAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop
                )
                key_label.setMinimumWidth(112)
                text_label = QLabel(description)
                text_label.setWordWrap(True)
                grid.addWidget(key_label, row, 0)
                grid.addWidget(text_label, row, 1)
            page_layout.addWidget(grid_host)

        page_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        # Some platform styles put a themed glyph on standard buttons; it
        # reads as a stray tick mark next to "Close".
        close_button = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.setIcon(QIcon())
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)
