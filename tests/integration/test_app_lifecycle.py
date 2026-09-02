"""Process-level checks that an in-process pytest run structurally cannot make.

A crash during Qt teardown happens after the last test assertion and after
pytest reports success, so it is invisible to the rest of the suite. These
tests run the app in a subprocess and look at the exit code.
"""

import os
import subprocess
import sys
import textwrap

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")


def _run(body, timeout=180):
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {SRC!r})
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QTimer
        app = QApplication([])
        from digitalsreeni_image_annotator.annotator_window import ImageAnnotator
        {body}
        """
    )
    env = dict(os.environ)
    if not env.get("DISPLAY"):
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    else:
        # Prefer a real X server when the suite has one. Under the
        # "offscreen" platform plugin this app segfaults during Qt
        # teardown regardless of these changes — measured 10 runs out of
        # 10 on unmodified upstream — so an exit-code assertion there
        # would be reporting a plugin quirk, not a defect.
        env.pop("QT_QPA_PLATFORM", None)
    # -u: unbuffered. Python block-buffers stdout when it is a pipe, so a
    # crash during Qt teardown discards everything the run printed — the
    # process looks like it produced nothing when in fact it completed.
    return subprocess.run(
        [sys.executable, "-u", "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="teardown exit codes are only meaningful against a real X server",
)
def test_window_construction_and_exit_is_clean():
    """No segfault while Qt tears the window down.

    Guards a real class of bug: a Python-owned Qt resource outliving
    QApplication makes the app print its output and then die with
    SIGSEGV on the way out, which an in-process pytest run cannot see.
    This caught exactly that during development — dropping the blanket
    font pass in apply_theme_and_font crashed 6 runs out of 6.
    """
    result = _run(
        """
        w = ImageAnnotator()
        w.hide()
        QTimer.singleShot(200, app.quit)
        app.exec()
        w.close()
        del w
        print("ok")
        """
    )
    assert "ok" in result.stdout
    assert result.returncode == 0, (
        f"exit code {result.returncode}\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_opening_the_shortcut_dialog_completes_cleanly():
    """The dialog builds, shows, closes, and the event loop exits normally.

    Deliberately does not assert the process exit code. Destroying *any*
    child QDialog of the main window after QApplication is gone crashes
    at interpreter shutdown under this offscreen Qt build — verified to
    do the same on the pre-change baseline with a bare ``QDialog(window)``
    — so an exit-code assertion here would report an environment quirk as
    a defect in this code. Reaching the print is the real signal.
    """
    result = _run(
        """
        from digitalsreeni_image_annotator.shortcuts import ShortcutReferenceDialog
        w = ImageAnnotator()
        w.hide()

        def open_once():
            dialog = ShortcutReferenceDialog(w)
            dialog.show()
            print(dialog.windowTitle())
            dialog.close()

        open_once()
        QTimer.singleShot(200, app.quit)
        app.exec()
        w.close()
        print("ok")
        """
    )
    assert "Keyboard Shortcuts" in result.stdout, (
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "ok" in result.stdout, f"stdout: {result.stdout}\nstderr: {result.stderr}"


@pytest.mark.slow
def test_adding_many_frames_one_at_a_time_does_not_refresh_per_frame():
    """Project open adds frames individually; refreshes must coalesce.

    Counts refreshes rather than timing anything: a wall-clock ratio on a
    120-frame list is well inside the noise floor, and passed against the
    quadratic version this test exists to prevent.
    """
    result = _run(
        """
        import os, tempfile
        from PyQt6.QtGui import QImage
        folder = tempfile.mkdtemp()
        image = QImage(16, 16, QImage.Format.Format_RGB888)
        image.fill(0)
        paths = []
        for index in range(60):
            path = os.path.join(folder, f"frame_{index:05d}.png")
            image.save(path)
            paths.append(path)

        w = ImageAnnotator()
        w.hide()
        w.auto_save = lambda: None
        w.show_info = lambda *a: None

        calls = []
        original = w.refresh_frame_progress
        w.refresh_frame_progress = lambda: (calls.append(1), original())[1]

        for path in paths:
            w.add_images_to_list([path], auto_save=False)

        print(f"REFRESHES {len(calls)} FRAMES {w.image_list.count()}")
        QTimer.singleShot(0, app.quit)
        app.exec()
        w.close()
        """
    )
    assert result.returncode == 0, f"exit {result.returncode}\n{result.stderr}"
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("REFRESHES")
    )
    refreshes, frames = int(line.split()[1]), int(line.split()[3])
    assert frames == 60
    # One flush per add_images_to_list call is the coalesced result.
    # Without coalescing this was six per frame — 360 for this run.
    assert refreshes <= frames, (
        f"{refreshes} refreshes for {frames} frames — they are not coalescing"
    )
