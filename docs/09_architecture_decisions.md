# Architecture Decisions

## ADR-001: GUI Framework Choice

**Status**: Superseded by [ADR-014](#adr-014-migrate-from-pyqt5-to-pyqt6)

**Original decision (historical)**: Use PyQt5 5.15.11. Chosen because the upstream project used PyQt5, PyQt5's ecosystem was more mature at the time, and migration carried risk.

**Superseding decision**: The project migrated to PyQt6 6.7+ in the same PR that introduced in-process AI inference. See [ADR-014](#adr-014-migrate-from-pyqt5-to-pyqt6) for the rationale (mainly: PyQt6 eliminated the WinError 1114 DLL load-order conflict that motivated ADR-011, unblocking the subprocess removal in ADR-013).

---

## ADR-002: Use Ultralytics for SAM Integration

**Status**: Accepted

**Context**: Need to integrate Segment Anything Model 2 for semi-automated annotation

**Decision**: Use Ultralytics library instead of direct SAM2 installation

**Rationale**:
- Simplifies SAM model loading (single line)
- Includes PyTorch dependencies
- Automatic model caching
- No manual model download required
- Supports both SAM 2.0 and SAM 2.1 variants

**Consequences**:
- ✅ Simplified installation (no separate SAM2 setup)
- ✅ Automatic model management
- ✅ Consistent API
- ⚠️ Dependency on Ultralytics release cycle

---

## ADR-003: Store Absolute Paths in Project Files

**Status**: Accepted

**Context**: Project files need to reference image locations

**Decision**: Store absolute paths to images in project JSON

**Rationale**:
- Images can be anywhere on filesystem
- No requirement to keep images with project file
- Simplifies project structure

**Consequences**:
- ✅ Flexible image locations
- ❌ Projects not portable between machines
- ❌ Moving images breaks projects

**Mitigation**: Export functions copy images to output directory

---

## ADR-004: Automated Regression Testing

**Status**: Accepted (supersedes the original manual-only decision)

**Context**: The application is GUI-heavy, but utility, persistence, export,
video-import, and selected Qt behavior can be tested deterministically.

**Decision**: Maintain a pytest suite with unit, integration, and focused
pytest-qt UI tests. Keep manual testing for visual quality, model inference,
platform behavior, and complete annotator workflows.

**Rationale**:
- Unit tests catch data and conversion regressions quickly.
- Integration tests exercise project transactions and video workflows.
- Focused UI tests cover event routing and the presence of critical controls.
- Visual mask quality and GPU behavior still require manual observation.

**Consequences**:
- Automated tests are required for changed deterministic behavior.
- Manual tests remain required for new annotation and inference features.
- The suite requires pytest-qt and an offscreen Qt platform in headless runs.

---

## ADR-005: Disable Autosave During Project Loading

**Status**: Accepted

**Context**: Projects were getting corrupted when application terminated during loading (v0.8.9 bug)

**Decision**: Set `is_loading_project` flag to disable autosave during load

**Rationale**:
- Autosave triggered with partially loaded state corrupts file
- Loading large projects is slow, increases risk
- Simple flag prevents the issue

**Consequences**:
- ✅ Prevents project corruption
- ✅ Minimal code change
- ⚠️ Users lose autosave protection during load window

---

## ADR-006: Use Shapely for Polygon Operations

**Status**: Accepted

**Context**: Need to merge, validate, and manipulate polygon geometries

**Decision**: Use Shapely library for all polygon operations

**Rationale**:
- Industry-standard computational geometry library
- Handles invalid polygons gracefully
- Efficient union/intersection operations
- Well-tested algorithms

**Consequences**:
- ✅ Robust polygon handling
- ✅ Easy merge operations
- ✅ Automatic polygon validation
- ⚠️ Additional dependency

---

## ADR-007: Flatten Polygon Coordinates in Storage

**Status**: Accepted

**Context**: Need to store polygon annotations

**Decision**: Store as flattened list `[x1, y1, x2, y2, ...]` instead of nested `[[x1, y1], [x2, y2], ...]`

**Rationale**:
- Compatible with COCO JSON format
- Smaller file size
- Standard in annotation tools

**Consequences**:
- ✅ COCO compatibility
- ✅ Compact representation
- ⚠️ Must convert to/from paired format for some operations

---

## ADR-008: Support Multiple Export Formats

**Status**: Accepted

**Context**: Users need annotations in different formats for various ML frameworks

**Decision**: Implement exporters for COCO, YOLO, Pascal VOC, labeled images, semantic labels

**Rationale**:
- Different frameworks have different input requirements
- YOLO and COCO are most common
- Labeled images useful for visual verification
- Semantic labels needed for segmentation models

**Consequences**:
- ✅ Wide compatibility
- ✅ Flexible workflow
- ⚠️ More code to maintain
- ⚠️ Must keep up with format changes (e.g., YOLOv11)

---

## ADR-009: Use Per-Slice Annotation Storage for Multi-dimensional Images

**Status**: Accepted

**Context**: TIFF stacks and CZI files have multiple slices that need individual annotations

**Decision**: Store annotations per slice with naming convention `{filename}_T{t}_Z{z}_C{c}`

**Rationale**:
- Each slice is effectively a separate 2D image
- Simple extension of existing single-image annotation
- User can navigate and annotate independently

**Consequences**:
- ✅ Simple mental model (each slice = image)
- ✅ Reuses existing annotation code
- ⚠️ Large stacks create many entries in annotations dict
- ⚠️ No 3D annotation support

---

## ADR-010: Normalize 16-bit Images to 8-bit for Display

**Status**: Accepted

**Context**: SAM and display require 8-bit images, but microscopy often uses 16-bit

**Decision**: Normalize 16-bit to 8-bit using percentile clipping

**Rationale**:
- SAM models trained on 8-bit RGB images
- Displays only show 8-bit effectively
- Percentile clipping (2nd-98th) provides better contrast than linear

**Consequences**:
- ✅ Better visual contrast
- ✅ SAM compatibility
- ⚠️ Information loss (quantization)
- ⚠️ Different normalization per image/slice

---

## ADR-011: Run Torch-based Workers in Isolated Subprocesses

**Status**: Superseded by [ADR-013](#adr-013-in-process-inference-with-qthread-wrapping)

**Context**: Both SAM 2 (via Ultralytics) and Grounding DINO (via transformers) load PyTorch into the process. On Windows + Python 3.14, importing PyQt5 first and then loading PyTorch causes `WinError 1114` (DLL load order conflict between Qt and Torch native dependencies). The application is fundamentally PyQt5-based, so we cannot reorder these imports.

**Decision**: Run each ML model in its own subprocess script that has no PyQt5 imports — `sam_worker.py` for SAM and `dino_worker.py` for DINO. The parent GUI process speaks to each worker over stdin/stdout with JSON requests and responses.

**Rationale**:
- The DLL conflict only manifests when both libraries are loaded in the same process. Splitting them across processes avoids the issue entirely.
- Keeps the GUI responsive: heavy model loading doesn't block PyQt's event loop in the same address space.
- Lets us swap or upgrade torch/transformers/ultralytics versions without worrying about Qt interactions.
- The JSON-over-stdio protocol is simple, language-agnostic, and easy to debug — just inspect what the worker prints.

**Consequences**:
- ✅ Works reliably on Windows + Python 3.14 (the original motivating bug)
- ✅ Worker scripts are PyQt-free; they can be tested independently
- ⚠️ Per-inference subprocess spawn cost (~1-2 s startup + first model load)
- ⚠️ Need UTF-8 forced on both ends of the pipe (`PYTHONIOENCODING=utf-8` in env, `encoding="utf-8", errors="replace"` on parent) — Windows cp1252 default crashes on non-ASCII bytes in torch warnings
- ⚠️ Two near-identical worker scripts to maintain (`sam_worker.py` mirrors the pattern from `dino_worker.py`)

**Superseded by**: Migrating to PyQt6 (ADR-013) eliminated the underlying DLL conflict. The subprocess hop, JSON marshalling, and `check_worker_isolation.py` tooling were removed in the same PR.

**Related**:
- Implementation (historical): `sam_utils.py` / `sam_worker.py`, `dino_utils.py` / `dino_worker.py`
- Original SAM-only version landed in #65 (Python 3.14 support)
- DINO subprocess pattern landed alongside the DINO feature

---

## ADR-012: Lazy Model Load on Dropdown Selection

**Status**: Accepted

**Context**: Both SAM and DINO model weights are large (SAM 2 tiny ~80 MB up to large ~400 MB; Grounding DINO base ~1.9 GB) and may not exist on first run. An earlier DINO flow required an explicit "Load" button click that did the resolve-or-download dance synchronously before the user could detect anything.

**Decision**: Selecting a model from the dropdown only updates state. Actual downloads happen on first use (first Detect call). UI feedback in the status label distinguishes "Ready: <model>" (weights present) from "<model> — will download on first detection".

**Rationale**:
- Matches the existing SAM behaviour (`change_sam_model` just stores the name; download happens in the worker).
- Removes a redundant click — one fewer thing for users to discover.
- Selecting a model the user picked by mistake is now free; only confirmed Detect triggers the (potentially heavy) download.

**Consequences**:
- ✅ Consistent UX between the SAM and DINO panels
- ✅ Faster perceived startup; no spurious downloads from idle browsing
- ⚠️ First Detect after selection blocks the UI while download runs (~1 min for DINO base); the status label shows progress but the dialog is otherwise unresponsive
- ⚠️ No async download progress dialog — `huggingface_hub` prints to stdout

---

## ADR-013: In-process Inference with QThread Wrapping

**Status**: Accepted

**Context**: ADR-011 introduced a subprocess hop for every SAM and DINO inference call to work around a PyQt5 + Torch DLL load-order conflict on Windows + Python 3.14. The workaround cost a fresh `python sam_worker.py` / `dino_worker.py` spawn per inference (~1-2 s warm latency, model reloaded from disk on every call) plus a temp-PNG marshal of the image.

Migrating the GUI from PyQt5 to PyQt6 (same PR) eliminates the DLL conflict — verified by `tools/check_pyqt6_torch_coexistence.py` importing PyQt6 → torch → transformers → ultralytics cleanly in one process on Windows+Py3.14 (the original failure case) and the Linux/macOS test matrix.

**Decision**: Run SAM and DINO inference directly inside the main Python process. Keep the model objects on the `SAMUtils` / `DINOUtils` singletons so they persist across calls. Wrap each inference in a short-lived `QThread` to keep the UI thread responsive; the public API blocks the caller via a nested `QEventLoop` so call sites in `annotator_window.py` stay synchronous-looking.

**Rationale**:
- The latency win is the whole point. Subprocess spawn + Python startup + model reload was ~1-2 s every call; in-process with a cached model is ~50-500 ms.
- Threading via a nested `QEventLoop` (the `_run_sync` helper in `sam_utils.py`) lets the calling thread keep pumping events — timers, repaints, progress dialog cancels still work — while inference runs on the QThread. Existing call sites need no refactor.
- Torch and transformers are imported lazily on first inference, so app startup stays fast for users who never touch SAM/DINO.
- `_qimage_to_numpy` already exists; converting the QImage on the calling thread (not on the worker) keeps Qt objects single-threaded as required.

**Consequences**:
- ✅ Each inference is ~1-2 s faster on Windows; less dramatic on macOS/Linux but still smoother.
- ✅ Cached model survives between calls — opening a DINO model once costs once. The DINO model stays on its compute device (CPU or CUDA) for its full lifetime; the old worker shuffled CPU↔GPU per call, defeating the caching gain on PCIe. Call `DINOUtils.unload()` / `SAMUtils.unload()` to free GPU memory explicitly.
- ✅ UI stays responsive during batch DINO+SAM runs (the calling thread's `QEventLoop` still processes events).
- ✅ One source of truth per model — no more keeping `sam_utils.py` and `sam_worker.py` aligned.
- ✅ Exceptions from the inference worker (model load failures, CUDA errors) propagate out of `_run_sync` rather than being printed and silently turned into `None`. The `change_sam_model` error path in `annotator_window.py` actually catches now.
- ⚠️ A crash in torch (CUDA OOM, segfault) now takes the app down where the subprocess used to absorb it. Mitigation: inference is wrapped in `try/except` at the `_run_sync` boundary; the user sees an error dialog instead of a frozen UI.
- ⚠️ Model RAM stays resident until the user closes the app (or invokes the `unload()` method).
- ⚠️ Re-entrancy is a real hazard, addressed with belt-and-braces:
   - `_run_sync` sets a module-level `_inference_in_flight` flag and raises `InferenceBusyError` if re-entered. Same-thread re-entry can happen because the calling thread pumps its event loop while waiting (a timer fire, a click on an un-disabled widget, etc.). A `QMutex` would not help — same-thread re-acquisition deadlocks on a non-recursive mutex and is meaningless on a recursive one.
   - The known re-entry vector — the SAM debounce timer firing during an in-flight inference — is guarded at the call site: `apply_sam_prediction` in `annotator_window.py` carries its own `_sam_inference_in_flight` flag and skips. Batch DINO already disables its trigger buttons.
   - The two-layer design is intentional: the call-site flag handles the common case quietly; the `_run_sync` flag is the safety net that surfaces unknown re-entry vectors as a real exception rather than corrupting the model with concurrent `.forward()` calls (torch / ultralytics / transformers model objects are not thread-safe).

**Related**:
- Implementation: `sam_utils.py`, `dino_utils.py` (both refactored in the same PR that retires ADR-011).
- Smoke test: `tools/check_pyqt6_torch_coexistence.py` (gate that gated this whole change).
- Supersedes: [ADR-011](#adr-011-run-torch-based-workers-in-isolated-subprocesses).

---

## ADR-014: Migrate from PyQt5 to PyQt6

**Status**: Accepted

**Context**: The project shipped on PyQt5 5.15+ (ADR-001) from inception. Two pressures combined to motivate a migration:
1. The PyQt5 + Torch DLL load-order conflict on Windows + Python 3.14 (ADR-011) forced an entire subprocess isolation layer (`sam_worker.py`, `dino_worker.py`, `check_worker_isolation.py`) that added ~1-2 s latency per inference. The conflict only manifests on PyQt5 — Qt6's packaging reshuffle eliminates it.
2. PyQt5 is in maintenance mode. PyQt6 is the actively developed line, gets new Qt6.x features, and has better Linux native integration (XCB plugin paths in particular).

**Decision**: Migrate the GUI binding from PyQt5 (`>=5.15.0`) to PyQt6 (`>=6.7.0`). Land in a single PR alongside the subprocess-removal work (ADR-013), gated behind `tools/check_pyqt6_torch_coexistence.py` to confirm the DLL conflict is actually gone on Windows + Python 3.14.

**Rationale**:
- Two coupled changes share most of their cost (touching every file that imports PyQt5) so doing them in one PR avoids paying the migration tax twice.
- Most PyQt5→PyQt6 differences are enum namespacing (`Qt.AlignCenter` → `Qt.AlignmentFlag.AlignCenter`) and module relocations (`QAction` moves from `QtWidgets` to `QtGui`) — mechanical, codemod-able. The behavioural risk is in event APIs (`event.pos()` → `event.position()`, returning `QPointF` not `QPoint`) and a handful of removed widgets (`QDesktopWidget` → `QGuiApplication.primaryScreen()`).
- The existing test suite (65 pytest-qt tests, mostly exercising coordinate transforms) serves as the regression safety net.

**Consequences**:
- ✅ Subprocess workers retired; inference is in-process with cached models (see [ADR-013](#adr-013-in-process-inference-with-qthread-wrapping)).
- ✅ Cleaner Linux story — `libxcb-cursor0` is required by Qt 6 (was optional under Qt 5), but the platform plugin path mess is gone.
- ✅ Long support runway: PyQt6 is the maintained binding.
- ⚠️ One-time migration cost: ~30 files touched, enum namespacing across `annotator_window.py` (300+ references), `event.pos()` → `event.position()` rewrite in `image_label.py`.
- ⚠️ PyQt6 is GPLv3 / commercial like PyQt5. Switching to PySide6 (LGPL) was considered and rejected to stay close to the existing `pyqtSignal`/`pyqtSlot` API.
- ✅ All `.exec_()` call sites in `src/` migrated to `.exec()` in the v0.9.0 fix-pack — the PyQt5 alias is gone from this codebase.

**Verification**:
- `tools/check_pyqt6_torch_coexistence.py` imports PyQt6 → torch → torchvision → transformers → ultralytics in that order. Run before merging on the Windows + Python 3.14 target.
- The full automated suite passed on the new binding under
  `QT_QPA_PLATFORM=offscreen` during the migration.
- Full app constructs and renders headlessly; snake-game easter egg validates the `QDesktopWidget` → `QGuiApplication.primaryScreen()` replacement.

**Related**:
- Supersedes: [ADR-001](#adr-001-gui-framework-choice).
- Unblocks: [ADR-013](#adr-013-in-process-inference-with-qthread-wrapping).

---

## ADR-015: Application-wide Event Filter for DINO Review Shortcuts

**Status**: Accepted (v0.9.0)

**Context**: During DINO batch / single-image review, the user has
to accept (Enter) or reject (Escape) pending masks. The keyboard
handling was originally in `ImageLabel.keyPressEvent`, which only
fires when the canvas has focus. In practice the user clicks slice
entries, image entries, or buttons during review — focus moves to
those widgets and Enter is consumed locally (e.g. `QListWidget`
emits `itemActivated`), never reaching the canvas. The result: Enter
and Escape silently failed during the most common review workflow.

Three options were considered:

1. **Force focus back to the canvas on every UI interaction** —
   intrusive, breaks normal navigation (Tab/Arrow keys on lists), and
   fragile because Qt's focus chain is not always predictable.
2. **Global `QShortcut` with ApplicationShortcut context** — fires
   regardless of focus but unconditionally hijacks Enter / Escape,
   breaking modal dialogs (Enter activates default button) and inline
   editing in `QLineEdit` / `QInputDialog`.
3. **Application-wide `QObject` event filter** that intercepts only
   when DINO temp_annotations are pending, and only when the focused
   widget is not a text input and no modal dialog is active.

**Decision**: Option 3. Implement `_DINOReviewEventFilter`, install it
on `QApplication.instance()` once at startup, and gate the
interception on three conditions: pending DINO temp_annotations,
no active modal widget, focus not on `QLineEdit`/`QTextEdit`. While synchronous
SAM 3 inference pumps Qt events, Enter/Escape is consumed without applying a
DINO review action.

**Consequences**:
- ✅ Enter/Escape works regardless of which widget holds focus during
  DINO review.
- ✅ Modal dialogs and text-input fields are unaffected.
- ✅ Pattern is reusable for any future "review pending state" feature.
- ⚠️ Adds a per-key-press function call cost to the entire app. The
  filter short-circuits in three cheap checks before any work, so the
  overhead is negligible (≤ a few μs per keystroke).
- ⚠️ Single global filter means future review-state features must
  share it or layer additional filters; if more review modes appear,
  collapse them into a strategy registry rather than installing
  multiple top-level filters.

**Related**:
- Implementation: `annotator_window.py` (`_DINOReviewEventFilter`
  class, `installEventFilter` call in `__init__`).
- Cross-cuts: documented in
  [Cross-cutting Concepts → DINO Temp Annotations](08_crosscutting_concepts.md#dino-temp-annotations--single-field-many-images).

---

## ADR-016: Use SAM 3 Session Predictor with Polygon-Derived Prompts

**Status**: Accepted (June 8, 2026)

**Context**: The welding workflow starts with manually labeled polygons and
must propagate several independently labeled objects through extracted video
frames. Meta's dense SAM 3 box prompt is a semantic visual prompt, resets dense
tracking state, and does not provide a one-box-to-one-object-ID contract.

**Decision**: Integrate Meta SAM 3 through `build_sam3_video_predictor` and its
session request API. Convert each source polygon to a binary mask and submit it
with an explicit object ID when the installed predictor exposes mask seeding.
Otherwise derive dense foreground/background refinement points from that mask.
Propagate forward and store results using the existing flattened
`segmentation` annotation schema.

**Consequences**:
- Multiple labeled objects retain independent object IDs and classes.
- Existing project saving, rendering, and export code remains reusable.
- Polygon-derived masks and refinement points preserve more spatial context
  than a single click, but users must still review propagated results.
- SAM 3 remains optional and has stricter CUDA/Python requirements than the
  base annotator.
- When neither `cc_torch` nor Triton is available, the adapter uses Meta's
  slower CPU connected-components fallback so interactive prompts work on
  Windows without modifying the installed SAM 3 package.

**Related**: `sam3_tracker.py`, `video_sequence.py`, and
[SAM 3 Welding Video Annotator Adaptation](SAM3_WELDING_VIDEO_ADAPTATION.md).

---

## ADR-017: Extract Selected Video Ranges to a Disk-Backed Frame Sequence

**Status**: Accepted (August 7, 2026)

**Context**: High-speed research videos can contain enough frames to exhaust
RAM if a labeling tool decodes the whole video up front. The existing
annotation, rendering, export, and tracking paths already operate on image
filenames, so introducing a second video-only annotation model would duplicate
state and increase project-format risk.

**Decision**: Probe video metadata first, then let the user choose an inclusive
frame range and stride. Decode that range sequentially on a cancellable
`QThread`, write selected frames to a fingerprinted application-cache folder,
and copy them into the project image directory on the same worker. Register the
session only after copying completes, and roll back registration and files if
the `.iap` commit fails. The commit flushes and fsyncs a same-directory
temporary file, atomically replaces the previous project, and attempts a
parent-directory fsync on POSIX. Persist only per-clip video metadata and
filename-to-source-frame mappings in `video_sessions`; runtime cache paths are
never deserialized.

**Consequences**:
- Peak decode memory is bounded to approximately one video frame plus encoder
  buffers, independent of source-video length.
- Large frame copies stay off the GUI thread and are cancellable.
- A failed JSON write cannot truncate the previous project file.
- Generic project image copies are staged and rolled back with the same save
  result contract, so a partial copy cannot become a committed project image.
- Save adopts an occupied project-image path only after verifying that its
  bytes match the source; mismatched same-named images abort the transaction.
- Smaller clips and temporal downsampling reduce disk use and annotation load.
- Manual correction, classes, exports, and current tracking code are reused.
- Imported frames are project-local before session registration, so projects
  remain self-contained.
- Multiple clips are persisted independently and selecting one of their frames
  activates the corresponding sequence.
- Tracking uses an isolated, position-numbered workspace rather than the
  project's shared image directory, preventing positional index drift.
- Extracted frames use lossless PNG by default so the import step does not add
  compression artifacts to faint scientific boundaries.
- SAM2 or EfficientTAM propagation can be added behind the frame-sequence
  workflow without changing the annotation schema.

**Related**: `video_clip.py`, `video_clip_dialog.py`, `video_sequence.py`, and
[Bounded-Memory Video Loading](08_crosscutting_concepts.md#bounded-memory-video-loading).

---

## Decisions Under Consideration

### Expand Model and Performance Testing

**Status**: Under Consideration

**Proposal**: Add benchmark videos and opt-in GPU model tests for long-running
SAM2, SAM3, EfficientTAM, and large-video workflows.

**Pros**:
- Detect memory and throughput regressions.
- Validate real model loading and propagation across supported platforms.

**Cons**:
- Large model downloads and specialized GPU requirements.
- Longer, less deterministic CI runs.

---

## ADR-018: Generate Both Stylesheets from One Token Table

**Status**: Accepted (v0.9.1)

**Context**: `default_stylesheet.py` (light) and
`soft_dark_stylesheet.py` (dark) were two hand-written QSS strings,
389 and 483 lines. They drifted. The dark sheet grew panel surfaces,
button roles (`primary` / `accent` / `quiet` / `danger` / `tool`),
tab styling and status colours; the light sheet never got most of
them, so Ctrl+D dropped half the window to raw Qt widget defaults.
Every new widget had to be styled twice or it looked broken in one
mode, and nothing failed when the second edit was forgotten.

**Decision**: Move the palette into `theme.py` as two token tables
with identical keys, plus `build_stylesheet(tokens)` which renders the
QSS from one template. The two stylesheet modules become three-line
wrappers that call it, keeping their existing module and variable
names so no import changes.

Rendering uses `str.format`, so a token missing from one table raises
`KeyError` at import — the failure is immediate and obvious rather
than a silently unstyled widget. `tests/unit/test_theme.py` also
asserts the two tables share their keys and that both sheets emit the
same selector set.

Interface chrome that paints colours from Python reads
`tokens_for(dark_mode)` rather than embedding hex literals, which is what
removed the hardcoded list-row colours from `update_slice_list_colors`.
Canvas overlay colours in `image_label.py` stay as they are: they are
picked for contrast against welding footage rather than against the UI,
so they are not a theming concern.

**Consequences**:
- ✅ Light and dark cannot diverge; a colour is written once.
- ✅ A new widget needs one rule, not two.
- ✅ Python-side colours (markers, overlays) follow the theme.
- ⚠️ QSS is now generated, so it cannot be read straight off disk —
  the file header points at `theme.py`.
- ⚠️ Braces in the template must be doubled for `str.format`. The
  test suite catches an unbalanced template immediately.

**Related**: `theme.py`, [Cross-cutting Concepts → One Token Table,
Two Stylesheets](08_crosscutting_concepts.md#one-token-table-two-stylesheets).

---

## ADR-019: Undo via Per-Frame Snapshots

**Status**: Accepted (v0.9.1)

**Context**: The app had no undo. A misplaced polygon, an eraser
stroke that took out the wrong class, or an accepted batch of model
proposals could only be repaired by hand, and annotators work through
hundreds of frames per session. Two designs were considered:

1. **Command log with inverse operations.** Smallest memory
   footprint, but every one of the dozen mutation paths (polygon,
   rectangle, brush, eraser, delete, merge, class change, SAM/DINO
   accept, SAM 3 track) needs its own correct inverse. A wrong
   inverse corrupts a labelled frame silently — worse than no undo.
2. **Snapshot the frame's annotation mapping before each edit.**

**Decision**: Option 2. `AnnotationHistory` keeps bounded per-frame
undo and redo stacks of deep-copied `{class: [annotation, ...]}`
mappings. Mutating paths call `record_annotation_history(label)`
before they change anything.

Per frame, not global: undo restores the frame it was recorded on, so
switching frames and pressing Ctrl+Z cannot rewrite a frame the
annotator is not looking at. Depth is capped (40) to bound memory
across a long session.

**Consequences**:
- ✅ Correct by construction — the restored state *is* a state the
  frame was previously in.
- ✅ New mutation paths need one line, not an inverse operation.
- ✅ Undo/redo restore is one code path (`_restore_annotation_state`),
  so the annotation list, markers, progress and canvas always refresh
  together.
- ⚠️ Memory is proportional to depth × polygons per frame. Acceptable
  here (hundreds of points per frame); would need revisiting for
  dense instance segmentation with thousands of objects.
- ⚠️ A mutation path that forgets to record is not undoable, and
  nothing warns. Mitigated by keeping the call adjacent to the commit
  in each path.

**Related**: `annotation_history.py`, [Cross-cutting Concepts →
Undo/Redo](08_crosscutting_concepts.md#undoredo--snapshots-not-inverse-commands).

---

## ADR-020: One Event Filter for Plain-Key Shortcuts

**Status**: Accepted (v0.9.1)

**Context**: Tool selection (P/R/B/E) and class selection (1-9) have
to work while focus sits on the frame list, the class list or a
button — none of which forward a plain key press to the canvas. The
obvious implementation is one `QShortcut` per key with
`ApplicationShortcut` context, matching how A/D/C and F2 are already
registered.

Two problems appeared:

1. **Ambiguity.** Ctrl+Z was bound by both the Edit menu action and a
   `QShortcut`. Qt treats a doubly-bound sequence as ambiguous and
   fires neither — measured: Ctrl+Z did nothing at all.
2. **Teardown instability.** While the per-key `QShortcut` version was
   in place, constructing the window and letting the interpreter exit
   segfaulted during Qt teardown on PyQt6 6.11 here, reproducibly, and
   stopped once the digit shortcuts were removed or switched to
   `WindowShortcut`. The root cause was never isolated, so treat this
   as an observation rather than a proven mechanism — the ambiguity in
   point 1 is on its own sufficient reason for the decision.

**Decision**: Handle unmodified tool and class keys in a single
application-wide event filter, `_WorkflowKeyFilter`, mirroring the
existing `_DINOReviewEventFilter` (ADR-015). Ctrl combinations stay on
their menu actions, with `ApplicationShortcut` context where they must
work regardless of focus, and alternates such as Ctrl+Y added to the
same action via `setShortcuts([...])` rather than a second object.

The filter's "is the user typing?" test inspects the widget the event
was **delivered to** rather than `QApplication.focusWidget()`, because
the focus widget is null while the window is inactive — which would
let a tool key steal a character out of the frame filter box.

**Decision, part two — de-duplicate platform bindings.** Redo is
reachable by Ctrl+Shift+Z and Ctrl+Y. `QKeySequence.StandardKey.Redo`
resolves *per platform*: Ctrl+Shift+Z on Linux and macOS, **Ctrl+Y on
Windows**. Listing `[StandardKey.Redo, "Ctrl+Y"]` therefore binds Ctrl+Y
twice on Windows — reintroducing the exact ambiguity above, on the
platform the lab runs. The action's sequences are built by appending
candidates only if not already present, and
`test_undo_and_redo_are_bound_once_so_the_shortcut_is_not_ambiguous`
asserts on the *count*, since a membership check cannot see a duplicate.

**Consequences**:
- ✅ Every sequence is bound exactly once on every platform.
- ✅ Tool and class keys work from any focused widget.
- ✅ Text fields, spin boxes and modal dialogs keep their keystrokes.
- ⚠️ A second global filter on top of ADR-015's. If a third appears,
  collapse them as ADR-015 already recommends.
- ⚠️ The legacy A/D/C/F2 `QShortcut`s were left alone to keep the
  change small; they could fold into this filter later. They do at
  least share the new `_typing_in_text_field` test.
- ⚠️ The filter is gated on `isActiveWindow()`, because several child
  windows (help, the Snake easter egg, the YOLO training dialog) are
  shown non-modally; without the gate an "E" aimed at one of them would
  arm the eraser on the canvas behind it.

**Related**: `annotator_window.py` (`_WorkflowKeyFilter`,
`handle_workflow_key`), [Cross-cutting Concepts → Keyboard
Shortcuts](08_crosscutting_concepts.md#keyboard-shortcuts--bind-each-sequence-once).

---

### Consider Relative Paths with Image Copying

**Status**: Under Consideration

**Proposal**: Copy images to project folder, store relative paths

**Pros**:
- Portable projects
- Self-contained

**Cons**:
- Disk space duplication
- Slow for large image sets
- Export already copies images
