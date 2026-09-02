"""Design tokens and the QSS builder that both themes are generated from.

Why this module exists
----------------------
The light and dark stylesheets used to be two hand-maintained QSS strings
(``default_stylesheet.py`` and ``soft_dark_stylesheet.py``). They drifted:
the dark sheet gained panel styling, button roles and status colours that
the light sheet never got, so switching to light mode dropped the app back
to raw Qt widget defaults. Every new widget had to be styled twice or it
looked broken in one theme.

Both sheets are now *generated* from the token tables below by
``build_stylesheet``. A colour or metric is written once; both themes stay
in step by construction. The two original modules still export the same
names (``default_stylesheet``, ``soft_dark_stylesheet``) so nothing that
imports them has to change.

Palette intent
--------------
Neutral greys with a single functional accent, the way lab acquisition and
annotation software is normally coloured (napari, CVAT, Fiji). Colour
carries meaning here: accent = selection or the primary action, green =
frame finished, amber = machine-proposed and unreviewed, red = destructive.
Nothing is coloured for decoration, because the *image* is the thing the
annotator has to read, and a saturated interface competes with it.

Adding a new token
------------------
Add the key to BOTH ``LIGHT_TOKENS`` and ``DARK_TOKENS`` — ``build_stylesheet``
raises KeyError on a missing key, so a half-added token fails loudly at
import time instead of silently rendering an unstyled widget.
"""

from __future__ import annotations

# Font stacks. Each entry is quoted for QSS and falls back to a generic
# family so the app still renders sensibly on a machine with none of them.
# Ordered by what each platform actually ships, best first. Inter leads
# because where it is installed it is the closest match to the metrics
# the rest of this sheet was tuned against; Segoe UI Variable Text is the
# current Windows UI face and Segoe UI the older one; SF Pro Text covers
# macOS. DejaVu Sans is last because it is wide and low-contrast — a
# usable fallback, not a design target.
UI_FONT_STACK = (
    '"Inter", "Segoe UI Variable Text", "Segoe UI", "SF Pro Text", '
    '"Helvetica Neue", "Noto Sans", "Ubuntu", "Cantarell", '
    '"DejaVu Sans", sans-serif'
)
# Numeric read-outs (zoom %, frame index, coordinates, counts) use a
# monospaced stack so digits do not jitter as values change.
MONO_FONT_STACK = (
    '"JetBrains Mono", "Cascadia Mono", "SF Mono", "Consolas", '
    '"DejaVu Sans Mono", "Liberation Mono", monospace'
)


LIGHT_TOKENS = {
    "name": "light",
    # Surfaces, back to front.
    "app_bg": "#E9EBED",
    "panel_bg": "#FFFFFF",
    "panel_alt_bg": "#F4F5F7",
    "sunken_bg": "#F0F1F3",
    "canvas_bg": "#DFE2E5",
    "input_bg": "#FFFFFF",
    "header_bg": "#F4F5F7",
    # Lines.
    "border": "#D2D6DB",
    "border_strong": "#B4BAC1",
    "divider": "#E3E6E9",
    # Type.
    "text": "#1F2328",
    "text_muted": "#5A636D",
    "text_faint": "#818A94",
    "text_inverse": "#FFFFFF",
    # Accent and state.
    "accent": "#1F6FEB",
    "accent_hover": "#3B82F6",
    "accent_pressed": "#1A5FCC",
    "accent_soft": "#E8F0FE",
    "accent_border": "#A9C6F5",
    "focus_ring": "#1F6FEB",
    "success": "#1A7F37",
    "success_soft": "#E6F4EA",
    "warning": "#9A6700",
    "warning_soft": "#FFF6E0",
    "danger": "#CF222E",
    "danger_hover": "#B01B26",
    "danger_soft": "#FDECEE",
    # Interaction.
    "hover_bg": "#EDEFF2",
    "pressed_bg": "#E2E5E9",
    "selected_bg": "#DCE9FC",
    "selected_text": "#0B3C86",
    "disabled_bg": "#F2F3F5",
    "disabled_text": "#A5ACB4",
    "button_bg": "#FBFBFC",
    "button_hover_bg": "#F1F3F5",
    "button_pressed_bg": "#E5E8EB",
    "scrollbar": "#C4CAD1",
    "scrollbar_hover": "#A9B0B8",
    "tooltip_bg": "#24292F",
    "tooltip_text": "#FFFFFF",
}


DARK_TOKENS = {
    "name": "dark",
    "app_bg": "#181A1C",
    "panel_bg": "#212427",
    "panel_alt_bg": "#26292D",
    "sunken_bg": "#1B1E20",
    "canvas_bg": "#101214",
    "input_bg": "#1A1D1F",
    "header_bg": "#26292D",
    "border": "#33373B",
    "border_strong": "#474C52",
    "divider": "#2C3033",
    "text": "#E3E6E8",
    "text_muted": "#9BA3AB",
    "text_faint": "#727A82",
    "text_inverse": "#FFFFFF",
    "accent": "#3B82F6",
    "accent_hover": "#5A97F8",
    "accent_pressed": "#2E6BD1",
    "accent_soft": "#182842",
    "accent_border": "#2C4C7C",
    "focus_ring": "#5A97F8",
    "success": "#4CAF63",
    "success_soft": "#16271A",
    "warning": "#D6A233",
    "warning_soft": "#2A2314",
    "danger": "#E5534B",
    "danger_hover": "#F0655D",
    "danger_soft": "#2C1A1A",
    "hover_bg": "#2B2F33",
    "pressed_bg": "#33383D",
    "selected_bg": "#1F3557",
    "selected_text": "#DCE9FC",
    "disabled_bg": "#212427",
    "disabled_text": "#5E666D",
    "button_bg": "#2A2E32",
    "button_hover_bg": "#333840",
    "button_pressed_bg": "#3A4048",
    "scrollbar": "#3C4248",
    "scrollbar_hover": "#4E555C",
    "tooltip_bg": "#33383D",
    "tooltip_text": "#E3E6E8",
}


# Shared metrics. Kept out of the colour tables because they never differ
# between themes — a control that is 26px tall in dark mode must be 26px
# tall in light mode or the layout reflows when the user hits Ctrl+D.
METRICS = {
    "radius_control": "6px",
    "radius_panel": "10px",
    "radius_pill": "5px",
    "control_height": "28px",
    "control_pad": "5px 12px",
}


#: Point size the scale below is expressed against — the app's "Medium".
BASE_POINT_SIZE = 10


def type_scale(base_pt: int = BASE_POINT_SIZE) -> dict:
    """Five sizes derived from one base, in px.

    The Font Size menu used to be applied by appending
    ``QWidget { font-size: Npt; }`` after the whole sheet and calling
    setFont on every widget. Both of those flatten the hierarchy: a
    heading, a button label and a help line all ended up the same size,
    which is most of why the interface read as blocky. Sizes are derived
    here instead, so changing Font Size scales the scale.
    """
    body = max(9, round(base_pt * 4 / 3))
    return {
        "fs_title": body + 2,
        "fs_body": body,
        "fs_label": body - 1,
        "fs_small": body - 2,
        "fs_micro": body - 3,
    }


def _fmt(template: str, tokens: dict, base_pt: int = BASE_POINT_SIZE) -> str:
    """Substitute ``{token}`` placeholders, failing loudly on typos."""
    values = dict(METRICS)
    values.update(tokens)
    values.update({k: f"{v}px" for k, v in type_scale(base_pt).items()})
    values["ui_font"] = UI_FONT_STACK
    values["mono_font"] = MONO_FONT_STACK
    return template.format(**values)


# The template is one string so the generated QSS keeps a readable,
# reviewable order: base -> surfaces -> typography -> controls -> lists ->
# chrome. Braces that belong to QSS are doubled for str.format.
_TEMPLATE = """
/* ------------------------------------------------------------------ */
/* Generated from theme.py — edit the tokens there, not this string.   */
/* Theme: {name}                                                       */
/* ------------------------------------------------------------------ */

QWidget {{
    background-color: {app_bg};
    color: {text};
    font-family: {ui_font};
}}

QMainWindow, QDialog {{
    background-color: {app_bg};
}}

/* --- Panels ------------------------------------------------------- */

QWidget#controlPanel,
QWidget#framesPanel {{
    background-color: {panel_bg};
    border: 1px solid {border};
    border-radius: {radius_panel};
}}

QWidget#canvasPanel {{
    background-color: {panel_bg};
    border: 1px solid {border};
    border-radius: {radius_panel};
}}

QWidget#productIdentity,
QWidget#canvasHeader,
QWidget#canvasFooter,
QWidget#framesHeader,
QWidget#framesToolbar {{
    background-color: transparent;
}}

QWidget#canvasHeader {{
    border-bottom: 1px solid {divider};
}}

QWidget#canvasFooter {{
    border-top: 1px solid {divider};
}}

QScrollArea#canvasViewport {{
    background-color: {canvas_bg};
    border: none;
}}

QScrollArea#canvasViewport > QWidget > QWidget {{
    background-color: {canvas_bg};
}}

QWidget#canvasPlaceholder,
QWidget#canvasPlaceholderColumn {{
    background-color: {canvas_bg};
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}

/* --- Typography --------------------------------------------------- */

QLabel {{
    color: {text};
    background-color: transparent;
}}

QLabel.product-title {{
    color: {text};
    font-size: {fs_title};
    font-weight: 600;
}}

QLabel.product-subtitle,
QLabel.muted {{
    color: {text_muted};
    font-size: {fs_small};
}}

QLabel.eyebrow {{
    color: {text_muted};
    font-size: {fs_label};
    font-weight: 600;
}}

QLabel.canvas-file {{
    color: {text};
    font-weight: 600;
}}

QLabel.mono {{
    font-family: {mono_font};
    color: {text_muted};
}}

QLabel.panel-count {{
    font-family: {mono_font};
    color: {text_muted};
    font-size: {fs_small};
}}

QLabel.shortcut-pill {{
    color: {text_faint};
    background-color: {panel_alt_bg};
    border: 1px solid {border};
    border-radius: {radius_pill};
    padding: 2px 6px;
    font-size: {fs_micro};
}}

QLabel.section-header,
QLabel.dialog-title {{
    color: {text};
    font-size: {fs_title};
    font-weight: 600;
    padding: 2px 0;
}}

QLabel.help-text {{
    color: {text_faint};
    font-size: {fs_small};
}}

QLabel.workflow-hint,
QLabel[cardRole="notice"] {{
    background-color: {panel_alt_bg};
    border: 1px solid {border};
    border-left: 2px solid {accent};
    border-radius: {radius_control};
    padding: 8px 10px;
    color: {text_muted};
    font-size: {fs_small};
    font-weight: 400;
}}

QLabel[cardRole="info"] {{
    background-color: transparent;
    border: none;
    border-left: 2px solid {border_strong};
    padding: 2px 0 2px 8px;
    color: {text_muted};
    font-size: {fs_small};
}}

QLabel[cardRole="status-idle"] {{
    background-color: {panel_alt_bg};
    border: 1px solid {border};
    border-radius: {radius_control};
    padding: 5px 8px;
    color: {text_muted};
    font-size: {fs_small};
}}

QLabel[cardRole="status-ok"] {{
    background-color: {success_soft};
    border: 1px solid {success};
    border-radius: {radius_control};
    padding: 5px 8px;
    color: {success};
    font-size: {fs_small};
}}

QLabel[cardRole="status-warn"] {{
    background-color: {warning_soft};
    border: 1px solid {warning};
    border-radius: {radius_control};
    padding: 5px 8px;
    color: {warning};
    font-size: {fs_small};
}}

/* --- Group boxes -------------------------------------------------- */

QGroupBox {{
    background-color: transparent;
    border: none;
    margin-top: 18px;
    padding-top: 4px;
    font-size: {fs_label};
    font-weight: 500;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 0px;
    padding: 0 0 4px 0;
    color: {text_muted};
    font-size: {fs_label};
    font-weight: 600;
}}

/* --- Buttons ------------------------------------------------------ */

QPushButton {{
    background-color: transparent;
    border: 1px solid {border_strong};
    border-radius: {radius_control};
    padding: {control_pad};
    min-height: {control_height};
    color: {text};
    font-size: {fs_body};
}}

QPushButton:hover {{
    background-color: {hover_bg};
    border-color: {accent_border};
}}

QPushButton:pressed {{
    background-color: {button_pressed_bg};
}}

QPushButton:focus {{
    border-color: {focus_ring};
}}

QPushButton:disabled {{
    background-color: {disabled_bg};
    border-color: {border};
    color: {disabled_text};
}}

QPushButton:checked {{
    background-color: {accent_soft};
    border-color: {accent};
    color: {accent};
    font-weight: 600;
}}

QPushButton[buttonRole="tool"] {{
    text-align: left;
    padding-left: 9px;
}}

QPushButton[buttonRole="tool"]:checked {{
    background-color: {accent_soft};
    border-color: {accent};
    color: {accent};
    font-weight: 600;
}}

QPushButton[buttonRole="primary"] {{
    background-color: {accent};
    border: 1px solid {accent};
    color: {text_inverse};
    font-weight: 600;
}}

QPushButton[buttonRole="primary"]:hover {{
    background-color: {accent_hover};
    border-color: {accent_hover};
}}

QPushButton[buttonRole="primary"]:pressed {{
    background-color: {accent_pressed};
    border-color: {accent_pressed};
}}

QPushButton[buttonRole="primary"]:disabled {{
    background-color: {disabled_bg};
    border-color: {border};
    color: {disabled_text};
}}

QPushButton[buttonRole="accent"] {{
    background-color: {accent_soft};
    border: 1px solid {accent_border};
    color: {accent};
    font-weight: 600;
}}

QPushButton[buttonRole="accent"]:hover {{
    border-color: {accent};
}}

QPushButton[buttonRole="quiet"] {{
    background-color: transparent;
    border: 1px dashed {border_strong};
    color: {text_muted};
}}

QPushButton[buttonRole="quiet"]:hover {{
    background-color: {hover_bg};
    color: {text};
}}

QPushButton[buttonRole="danger"] {{
    background-color: transparent;
    border: 1px solid {border_strong};
    color: {danger};
}}

QPushButton[buttonRole="danger"]:hover {{
    background-color: {danger_soft};
    border-color: {danger};
    color: {danger_hover};
}}

QPushButton[buttonRole="ghost"] {{
    background-color: transparent;
    border: 1px solid {border};
    color: {text_muted};
    padding: 2px 8px;
    min-height: 20px;
}}

QPushButton[buttonRole="ghost"]:hover {{
    background-color: {hover_bg};
    color: {text};
}}

QPushButton[buttonRole="ghost"]:checked {{
    background-color: {accent_soft};
    border-color: {accent_border};
    color: {accent};
}}

QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: {radius_control};
    padding: 3px 6px;
    color: {text_muted};
}}

QToolButton:hover {{
    background-color: {hover_bg};
    color: {text};
}}

QToolButton:checked {{
    background-color: {accent_soft};
    border-color: {accent_border};
    color: {accent};
}}

/* --- Tabs --------------------------------------------------------- */

QTabWidget::pane {{
    border: none;
    border-top: 1px solid {divider};
    top: -1px;
}}

QTabBar::tab {{
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 2px;
    margin-right: 18px;
    color: {text_muted};
    font-size: {fs_body};
    font-weight: 500;
}}

QTabBar::tab:hover {{
    color: {text};
}}

QTabBar::tab:selected {{
    color: {accent};
    border-bottom: 2px solid {accent};
}}

/* --- Lists and tables --------------------------------------------- */

QListWidget,
QTreeWidget,
QTableWidget {{
    background-color: {input_bg};
    border: 1px solid {border};
    border-radius: {radius_control};
    outline: none;
}}

QListWidget::item,
QTreeWidget::item {{
    padding: 6px 8px;
    border-radius: {radius_pill};
    color: {text};
}}

QListWidget::item:hover,
QTreeWidget::item:hover {{
    background-color: {hover_bg};
}}

QListWidget::item:selected,
QTreeWidget::item:selected,
QTableWidget::item:selected {{
    background-color: {selected_bg};
    color: {selected_text};
}}

QListWidget#frameList::item {{
    padding: 4px 6px;
}}

QHeaderView::section {{
    background-color: {header_bg};
    color: {text_muted};
    border: none;
    border-bottom: 1px solid {border};
    padding: 4px 6px;
    font-size: {fs_small};
    font-weight: 600;
}}

QTableWidget {{
    gridline-color: {divider};
}}

/* --- Inputs ------------------------------------------------------- */

QLineEdit,
QTextEdit,
QPlainTextEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {{
    background-color: {input_bg};
    border: 1px solid {border_strong};
    border-radius: {radius_control};
    padding: 3px 8px;
    min-height: 22px;
    color: {text};
    selection-background-color: {accent};
    selection-color: {text_inverse};
}}

QLineEdit:focus,
QTextEdit:focus,
QPlainTextEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {{
    border-color: {focus_ring};
}}

QLineEdit:disabled,
QComboBox:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled {{
    background-color: {disabled_bg};
    color: {disabled_text};
}}

QComboBox:hover {{
    border-color: {border_strong};
}}

QComboBox::drop-down {{
    border: none;
    width: 18px;
}}

QComboBox QAbstractItemView {{
    background-color: {panel_bg};
    border: 1px solid {border_strong};
    selection-background-color: {selected_bg};
    selection-color: {selected_text};
    outline: none;
}}

/* --- Sliders ------------------------------------------------------ */

QSlider::groove:horizontal {{
    height: 3px;
    background: {border};
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {panel_bg};
    border: 1px solid {border_strong};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    border-color: {accent};
}}

/* --- Scrollbars --------------------------------------------------- */

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical,
QScrollBar::handle:horizontal {{
    background: {scrollbar};
    border-radius: 5px;
    min-height: 28px;
    min-width: 28px;
}}

QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {{
    background: {scrollbar_hover};
}}

QScrollBar::add-line,
QScrollBar::sub-line,
QScrollBar::add-page,
QScrollBar::sub-page {{
    background: none;
    border: none;
    height: 0;
    width: 0;
}}

/* --- Menus and chrome --------------------------------------------- */

QMenuBar {{
    background-color: {app_bg};
    color: {text};
    border-bottom: 1px solid {divider};
}}

QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
    color: {text_muted};
}}

QMenuBar::item:selected {{
    background-color: {hover_bg};
    color: {text};
}}

QMenu {{
    background-color: {panel_bg};
    border: 1px solid {border_strong};
    padding: 4px;
}}

QMenu::item {{
    padding: 5px 22px 5px 18px;
    border-radius: {radius_pill};
    color: {text};
}}

QMenu::item:selected {{
    background-color: {selected_bg};
    color: {selected_text};
}}

QMenu::separator {{
    height: 1px;
    background: {divider};
    margin: 4px 6px;
}}

QToolTip {{
    background-color: {tooltip_bg};
    color: {tooltip_text};
    border: 1px solid {border_strong};
    border-radius: {radius_control};
    padding: 5px 7px;
}}

QStatusBar {{
    background-color: {panel_alt_bg};
    border-top: 1px solid {divider};
    color: {text_muted};
}}

QStatusBar::item {{
    border: none;
}}

QStatusBar QLabel {{
    color: {text_muted};
    font-size: {fs_small};
    padding: 0 2px;
}}

QLabel#statusSeparator {{
    color: {border_strong};
}}

QLabel.status-metric {{
    font-family: {mono_font};
    color: {text_muted};
    font-size: {fs_small};
}}

QLabel.status-strong {{
    color: {text};
    font-size: {fs_small};
    font-weight: 600;
}}

QProgressBar {{
    background-color: {sunken_bg};
    border: 1px solid {border};
    border-radius: {radius_pill};
    text-align: center;
    color: {text_muted};
    font-size: {fs_micro};
    max-height: 14px;
}}

QProgressBar::chunk {{
    background-color: {accent};
    border-radius: 2px;
}}

QProgressBar#frameProgress::chunk {{
    background-color: {success};
}}

QCheckBox,
QRadioButton {{
    background-color: transparent;
    color: {text};
    spacing: 6px;
}}

QCheckBox::indicator,
QRadioButton::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {border_strong};
    background-color: {input_bg};
}}

QCheckBox::indicator {{
    border-radius: 3px;
}}

QRadioButton::indicator {{
    border-radius: 7px;
}}

QCheckBox::indicator:checked,
QRadioButton::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}

QSplitter::handle {{
    background-color: transparent;
}}

QSplitter::handle:horizontal {{
    width: 6px;
}}

QSplitter::handle:hover {{
    background-color: {accent_border};
}}

QDialogButtonBox QPushButton {{
    min-width: 76px;
}}
"""


def build_stylesheet(tokens: dict, base_pt: int = BASE_POINT_SIZE) -> str:
    """Render the QSS for one token table at one base font size."""
    return _fmt(_TEMPLATE, tokens, base_pt)


def tokens_for(dark_mode: bool) -> dict:
    """Return the active token table.

    Widgets that must paint colours themselves (list item backgrounds,
    canvas overlays) read from here rather than hardcoding hex values, so
    they follow the theme like everything else.
    """
    return DARK_TOKENS if dark_mode else LIGHT_TOKENS
