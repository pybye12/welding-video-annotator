"""Undo/redo history for annotation edits.

Why a whole-frame snapshot rather than a command log
----------------------------------------------------
Annotations for one frame live in ``all_annotations[frame_name]`` as a
``{class_name: [annotation_dict, ...]}`` mapping, and a dozen different
code paths mutate it — polygon commit, rectangle commit, brush commit,
eraser commit, delete, merge, class change, SAM/DINO accept, SAM 3 track.
Writing an inverse operation for each of those is a lot of surface area to
get subtly wrong, and a wrong inverse silently corrupts a labelled frame,
which is worse than having no undo at all.

Snapshotting the frame's annotation mapping before each edit is O(number of
polygons on one frame) — a few hundred points at most in this workflow — so
the copy is cheap, and restoring it cannot desynchronise from the real
state because it *is* the real state.

Scope
-----
History is per frame. Undo restores the frame it was recorded on, so
switching frames and pressing Ctrl+Z never rewrites a frame the annotator
is not looking at. The depth cap keeps memory bounded on long sessions.
"""

from __future__ import annotations

import copy


DEFAULT_DEPTH = 40


class AnnotationHistory:
    """Bounded per-frame undo/redo stacks over annotation snapshots."""

    def __init__(self, depth: int = DEFAULT_DEPTH):
        self.depth = max(1, int(depth))
        self._undo: dict[str, list[tuple[str, dict]]] = {}
        self._redo: dict[str, list[tuple[str, dict]]] = {}

    @staticmethod
    def snapshot(annotations: dict | None) -> dict:
        """Deep-copy one frame's ``{class: [annotation, ...]}`` mapping."""
        return copy.deepcopy(annotations) if annotations else {}

    def record(self, frame_name: str, annotations: dict | None, label: str = "") -> None:
        """Store the pre-edit state of ``frame_name``.

        Call this immediately *before* mutating the frame. Recording a new
        edit clears the redo stack, which is the behaviour every editor
        has: once you take a new action, the branch you undid is gone.
        """
        if not frame_name:
            return
        stack = self._undo.setdefault(frame_name, [])
        stack.append((label, self.snapshot(annotations)))
        if len(stack) > self.depth:
            del stack[0 : len(stack) - self.depth]
        self._redo.pop(frame_name, None)

    def can_undo(self, frame_name: str) -> bool:
        return bool(self._undo.get(frame_name))

    def can_redo(self, frame_name: str) -> bool:
        return bool(self._redo.get(frame_name))

    def undo(self, frame_name: str, current: dict | None):
        """Pop one undo step. Returns ``(label, restored_state)`` or None."""
        stack = self._undo.get(frame_name)
        if not stack:
            return None
        label, previous = stack.pop()
        self._redo.setdefault(frame_name, []).append(
            (label, self.snapshot(current))
        )
        return label, previous

    def redo(self, frame_name: str, current: dict | None):
        """Pop one redo step. Returns ``(label, restored_state)`` or None."""
        stack = self._redo.get(frame_name)
        if not stack:
            return None
        label, following = stack.pop()
        self._undo.setdefault(frame_name, []).append(
            (label, self.snapshot(current))
        )
        return label, following

    def undo_label(self, frame_name: str) -> str:
        stack = self._undo.get(frame_name)
        return stack[-1][0] if stack else ""

    def redo_label(self, frame_name: str) -> str:
        stack = self._redo.get(frame_name)
        return stack[-1][0] if stack else ""

    def discard_last(self, frame_name: str) -> bool:
        """Drop the newest undo entry for a frame.

        For an edit that was recorded and then abandoned — the user hit
        Cancel, or the operation turned out to change nothing. Without
        this, ``record`` has already cleared the redo stack and left a
        step that restores an identical state, so Ctrl+Z appears to do
        nothing and the real redo branch is gone.
        """
        stack = self._undo.get(frame_name)
        if not stack:
            return False
        stack.pop()
        if not stack:
            self._undo.pop(frame_name, None)
        return True

    def forget(self, frame_name: str) -> None:
        """Drop history for a frame that was removed from the project."""
        self._undo.pop(frame_name, None)
        self._redo.pop(frame_name, None)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
