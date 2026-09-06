"""
Main entry point for the Image Annotator application.

This module creates and runs the main application window.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import os
import sys
import traceback

from PyQt6.QtWidgets import QApplication, QMessageBox

from .annotator_window import ImageAnnotator

# Legacy defensive cleanup from the PyQt5 era: a stale
# QT_QPA_PLATFORM_PLUGIN_PATH could shadow Qt's bundled XCB plugin and
# break startup on Linux. PyQt6 packaging is more robust about this, but
# the pop is cheap and harmless to keep.
if sys.platform.startswith("linux"):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


def install_crash_guard():
    """Report an unexpected error instead of vanishing.

    PyQt aborts the process when a Python exception escapes a slot, so
    one bad geometry or one missing file closes the whole window, with
    the traceback visible only to whoever launched it from a terminal.
    Every annotation edit is auto-saved, so the labels survive — what is
    lost is the annotator's place in the clip and any idea of what went
    wrong.

    The dialog is shown once per distinct fault. An exception raised
    from a paint or timer handler repeats every frame, and a modal per
    repeat is its own kind of crash.
    """
    reported = set()

    def handle(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        traceback.print_exception(exc_type, exc_value, exc_tb)
        details = "".join(
            traceback.format_exception(exc_type, exc_value, exc_tb)
        )

        last = traceback.extract_tb(exc_tb)[-1] if exc_tb else None
        signature = (exc_type, last.filename if last else "", last.lineno if last else 0)
        if signature in reported:
            return
        reported.add(signature)

        if QApplication.instance() is None:
            return
        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Something went wrong")
            box.setText(
                "The last action failed, but the app is still running."
            )
            box.setInformativeText(
                "Your labels are saved — every edit is written to the "
                "project as you make it. If this keeps happening, send the "
                "details below along with what you were doing.\n\n"
                "This message is shown once per problem."
            )
            box.setDetailedText(details)
            box.exec()
        except Exception:  # pragma: no cover - the reporter must never raise
            pass

    sys.excepthook = handle


def main():
    """
    Main function to run the Image Annotator application.
    """
    app = QApplication(sys.argv)
    install_crash_guard()
    window = ImageAnnotator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
