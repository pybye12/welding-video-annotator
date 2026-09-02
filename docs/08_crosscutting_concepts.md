# Cross-cutting Concepts

## Coordinate Systems

### Screen Coordinates vs Image Coordinates

All mouse events are in screen coordinates and must be converted to image coordinates:

```python
# In ImageLabel
def screen_to_image_coords(self, screen_pos):
    # Account for offset (centering)
    image_x = screen_pos.x() - self.offset_x
    image_y = screen_pos.y() - self.offset_y

    # Account for zoom
    original_x = image_x / self.zoom_factor
    original_y = image_y / self.zoom_factor

    return (original_x, original_y)
```

### Annotation Storage Format

Annotations are stored in image coordinates (unzoomed, absolute pixels):
- **Polygon**: Flattened list `[x1, y1, x2, y2, ...]`
- **Rectangle**: COCO format `[x, y, width, height]`

### Pan + Zoom Reference Frames

Two non-obvious gotchas live in `ImageLabel.mouseMoveEvent` /
`wheelEvent`:

- **Pan must use `event.globalPosition()`, not `event.position()`.**
  Widget-local coords absorb half the cursor delta during a scrollbar
  move (the widget shifts under the cursor mid-drag) → effective
  half-speed pan. The global frame is stable.
- **Zoom-to-cursor must compute the post-zoom `offset_x/y`
  analytically from the viewport, not read `self.offset_x` after the
  zoom call.** `update_scaled_pixmap()` only *relaxes* the minimum
  size on zoom-out; the widget hasn't shrunk by the time
  `update_offset()` runs, so `self.width()` is stale and the offset
  comes out wrong. Use `viewport().width()` + `scaled_pixmap.width()`
  to derive the offset directly. Zoom-in worked by accident because
  the widget grows immediately when `setMinimumSize` enlarges it.

## Image Format Conversions

### QImage ↔ NumPy Array

**QImage to NumPy** (for SAM inference):
```python
def qimage_to_numpy(qimage):
    width = qimage.width()
    height = qimage.height()
    fmt = qimage.format()

    if fmt == QImage.Format_Grayscale16:
        # 16-bit → normalize to 8-bit → RGB
        buffer = qimage.constBits().asarray(height * width * 2)
        image = np.frombuffer(buffer, dtype=np.uint16)
        image_8bit = normalize_16bit_to_8bit(image)
        return np.stack((image_8bit,) * 3, axis=-1)

    elif fmt == QImage.Format_RGB888:
        # Direct conversion
        buffer = qimage.constBits().asarray(height * width * 3)
        return np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))

    # ... handle other formats
```

**16-bit Normalization**:
```python
def normalize_16bit_to_8bit(image):
    # Percentile-based normalization for better contrast
    p2, p98 = np.percentile(image, (2, 98))
    image_clipped = np.clip(image, p2, p98)
    return ((image_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
```

## Polygon Operations

### Shapely for Geometry

**Merge Annotations**:
```python
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# Convert segmentation lists to Shapely Polygons
polygons = []
for ann in selected_annotations:
    coords = [(ann["segmentation"][i], ann["segmentation"][i+1])
              for i in range(0, len(ann["segmentation"]), 2)]
    poly = Polygon(coords)
    poly = make_valid(poly)  # Fix invalid polygons
    polygons.append(poly)

# Merge
merged = unary_union(polygons)

# Convert back to segmentation format
coords = list(merged.exterior.coords)
segmentation = [coord for point in coords for coord in point]
```

### Minimum Area Threshold

Paint brush annotations filter out small artifacts:
```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) > 10:  # 10 pixels minimum
        # Accept annotation
```

## Autosave and Project Corruption Prevention

### Critical: Disable Autosave During Load

**Problem**: Autosave triggered during loading can corrupt project files

**Solution** (v0.8.12):
```python
class ImageAnnotator:
    def load_project_data(self, project_data):
        self.is_loading_project = True  # Disable autosave
        try:
            # ... load all data
        finally:
            self.is_loading_project = False  # Re-enable

    def save_project(self, show_message=True):
        if self.is_loading_project:
            return False  # Skip save during load
        # ... normal save logic
```

## SAM Model Management

### Model Caching

First use downloads models, subsequent uses load from cache:
```python
# Ultralytics automatically caches in:
# - Working directory (current implementation)
# - Or ~/.cache/ultralytics/ (default)

sam_model = SAM("sam2_t.pt")  # Downloads if not present
```

### Releasing Model GPU Memory

`SAMUtils.unload()` and `DINOUtils.unload()` must do **three** things,
in order:

1. Drop the cached Python references (`self._model = None`, etc.).
2. **`gc.collect()`** to break circular references inside Ultralytics
   / Transformers model objects (config ↔ model, processor ↔
   tokenizer). Without this, the C++/CUDA backing memory stays pinned
   until Python's cyclic GC runs on its own schedule, which can be
   many seconds or never. Task Manager / `nvidia-smi` will show zero
   drop in GPU memory.
3. **`torch.cuda.empty_cache()`** (plus `torch.cuda.ipc_collect()`) so
   the PyTorch allocator returns the freed blocks to the OS / driver.

Skipping step 2 was the cause of "Tools → Unload AI Models does
nothing visible" in v0.9.0 manual testing.

### Model Size Recommendations

| Model | Size | RAM Usage | Speed | Recommendation |
|-------|------|-----------|-------|----------------|
| SAM 2 tiny | ~40MB | Low | Fast | ✅ Recommended for most users |
| SAM 2 small | ~90MB | Medium | Medium | ✅ Good balance |
| SAM 2 base | ~150MB | Medium-High | Slow | ⚠️ Use with caution |
| SAM 2 large | ~400MB | High | Very Slow | ❌ Not recommended (crashes on limited resources) |

## Dark Mode Support

### One Token Table, Two Stylesheets

`theme.py` owns the palette. `LIGHT_TOKENS` and `DARK_TOKENS` hold the
same keys; `build_stylesheet(tokens)` renders the QSS; and
`default_stylesheet.py` / `soft_dark_stylesheet.py` are three-line
modules that call it. Both sheets therefore style exactly the same
selectors, always.

```python
# theme.py
LIGHT_TOKENS = {"panel_bg": "#FFFFFF", "success": "#1A7F37", ...}
DARK_TOKENS  = {"panel_bg": "#212427", "success": "#4CAF63", ...}

# soft_dark_stylesheet.py
soft_dark_stylesheet = build_stylesheet(DARK_TOKENS)
```

```python
# In ImageAnnotator
if dark_mode_enabled:
    self.setStyleSheet(soft_dark_stylesheet)
    self.image_label.set_dark_mode(True)
else:
    self.setStyleSheet(default_stylesheet)
    self.image_label.set_dark_mode(False)
```

**Rules**

- Add a token to **both** tables. `build_stylesheet` uses `str.format`,
  so a key present in one table only raises `KeyError` at import time —
  loud, and before anything renders.
- Chrome that paints colours in Python — list-item markers, panel
  decoration — reads `tokens_for(self.dark_mode)` instead of carrying
  its own hex literals. `_apply_labeled_marker` is the example to copy.
  The canvas overlays in `image_label.py` are deliberately *not* themed:
  the polygon fills, the SAM point markers and the brush ring are read
  against the welding footage, not against the interface, and their
  colours are chosen for contrast with the image. Converting those is a
  separate question from theming the shell.
- `tests/unit/test_theme.py` asserts that the two tables share their
  keys and that both sheets style the same selector set, so a
  half-finished theme change fails CI rather than shipping a widget that
  is only styled in one mode.

Before this, the two sheets were hand-maintained strings. The dark one
gained panel styling and button roles that the light one never got, so
Ctrl+D dropped half the window back to raw Qt defaults.

**Dark Mode Considerations**:
- Annotation rendering uses inverted colors for visibility
- Text labels use high-contrast colors
- Background grid adjusted for dark backgrounds

### Dark Mode — No Hardcoded Colors Rule

**Do not hardcode `background`, `color`, or other palette-dependent
values in widget `setStyleSheet(...)` calls.** They override both the
default OS look *and* `soft_dark_stylesheet.py`, leaving bright
rectangles on the dark sidebar. Past offenders that bit us:

- `ClassThresholdTable` header had `background: #e0e0e0;` → bright bar
  across the top of the DINO panel in dark mode.
- `lbl_dino_status` had `background: #f5f5f5;` → bright box where the
  "No DINO model loaded" status sat.

Either leave the property out of the inline stylesheet so the global
sheet wins, or use Qt's palette role functions (`palette(base)`,
`palette(mid)`, `palette(text)`, …) which resolve at paint time
against the active palette. Inline hardcoded greys are an anti-pattern.

When introducing a new widget type that doesn't have a rule in
`soft_dark_stylesheet.py` yet — add the rule there *first*, then build
the widget. Otherwise the widget uses the OS default in dark mode,
which on Windows means barely-visible radio-button indicators and
white-on-white headers (the dataset splitter radio buttons hit this
before they were styled).

## Type Scale and the Font Size Setting

`theme.type_scale(base_pt)` derives five px sizes from one base, and
`build_stylesheet(tokens, base_pt=...)` renders the sheet at that size.
`apply_theme_and_font` passes the user's Font Size choice in, so changing
it scales the whole hierarchy.

Before this the setting was applied by appending
`QWidget { font-size: Npt; }` to a fixed sheet. That rule is less
specific than almost every other rule in the sheet, so it only reached
widgets nothing else styled — headings, button labels and help text all
came out the same size, which is most of why the interface read as
blocky.

**Do not remove the blanket rule or the per-widget font pass that follow
the setStyleSheet call.** They look redundant beside the generated sheet.
Removing them made the app segfault during Qt teardown: 6 runs out of 6
under a real X server, against 0 out of 6 with them present, on
otherwise identical code. The mechanism was never isolated. The comment
in `apply_theme_and_font` says the same thing, because this is exactly
the kind of tidy-up a later reader will attempt.

Note also that the `offscreen` Qt platform plugin segfaults at teardown
for this app regardless — 10 runs out of 10 on unmodified upstream — so
process-exit assertions are only meaningful against a real X server.
`tests/integration/test_app_lifecycle.py` skips itself without a DISPLAY
for that reason.

## Frame Status and Session Progress

`all_annotations[frame_name]` is the single source of truth for whether
a frame is finished. `frame_has_labels()` answers that question in one
place, and deliberately ignores classes prefixed `Temp-`: those are
model proposals waiting for review, so a frame holding only proposals is
not done.

Everything that reports progress derives from it:

| Surface | Fed by |
|---------|--------|
| Marker dot on a frame or slice row | `_apply_labeled_marker` |
| "N of M labeled (P%)" + progress bar | `refresh_frame_progress` |
| Undo / redo button enablement | `_sync_history_buttons` |
| "Todo" filter in the frames panel | `apply_frame_filter` |
| Next-step line at the top of the Label tab | `update_next_step_hint` |

`annotations_changed()` is the hook to call after label state changes.
`update_slice_list_colors()` delegates to it, which is what lets the two
dozen existing mutation paths keep their existing call and stay correct;
new code should call `annotations_changed()` directly.

**Refreshes coalesce, because bulk paths are per-frame.**
`load_project_data` calls `add_images_to_list` once per image, and each
of those switches image twice — so an uncoalesced O(all frames) refresh
per call makes opening a project quadratic. Measured on 400 frames
before coalescing: 11.5 s, scaling 3.2× for 2× the frames. Two guards
fix it:

- `suspended_progress_refresh()` — a context manager wrapping bulk
  operations; `add_images_to_list` uses it. One refresh runs when the
  outermost block exits.
- `is_loading_project` suppresses refreshes outright, with a single
  `annotations_changed()` after the load finishes.

After both, the same 400-frame load takes 4.5 s — faster than the
pre-change baseline, because the per-row slice-colour loop is now
skipped during bulk work too.

**Markers and counts are computed in one pass**, so a dot can never
disagree with the number beside it. That is affordable because
`_apply_labeled_marker` returns immediately for a row that already looks
right: each row caches `(labeled, dark_mode)` in a custom item role, so a
refresh over an unchanged list costs one dict lookup per row and zero
`setIcon` calls. Caching the theme alongside the state is also what makes
Ctrl+D re-colour every marker without a separate invalidation pass.

`ImageLabel._notify_cursor` calls `update_cursor_readout`, not the whole
status-bar rebuild — it runs on every mouse-move.

**Markers are icons, never row colours.** The frame list is looked up by
`item.text()` and `findItems(name, MatchExactly)` in more than a dozen
places, so the marker has to stay out of the text; and a filled row
colour competes with the stylesheet's own selection colour, which made a
labeled row and the selected row indistinguishable.

**Filtering hides rows, it never removes them.** `apply_frame_filter`
calls `setHidden`, so `image_list.count()`, `item(index)` and
`findItems` keep seeing the whole project and no existing caller has to
learn about filtering. The frame currently on the canvas is always left
visible — hiding it looks like the open frame vanished.

## Undo/Redo — Snapshots, Not Inverse Commands

`AnnotationHistory` (`annotation_history.py`) stores a deep copy of one
frame's `{class: [annotation, ...]}` mapping before each edit.

A dozen paths mutate annotations — polygon, rectangle, brush, eraser,
delete, merge, class change, DINO accept, SAM 3 track. Writing a correct
inverse for each is a large surface area, and a wrong inverse silently
corrupts a labelled frame, which is worse than having no undo. A frame
snapshot is a few hundred points at most in this workflow, and restoring
it cannot desynchronise from the live state because it *is* the state.

- Call `record_annotation_history("what changed")` **before** mutating.
- History is per frame, so Ctrl+Z never rewrites a frame the annotator
  is not looking at. Paths that write to a frame other than the one on
  screen must pass that frame name explicitly — `_commit_dino_results`,
  `copy_selected_annotation_to_next_frame` and the per-frame loop in
  `sam3_track_forward` all do.
- **History must die with the annotations it describes.** It is keyed by
  bare frame name, and `undo_annotation_change` auto-saves, so a stale
  entry can inject a closed project's annotations into a new project
  that reuses a file name and write them to its `.iap`. `clear_all`,
  `load_project_data` and `import_annotations` call
  `annotation_history.clear()`; `_remove_image_item` calls `forget()`
  for the frame and each of its slices.
- `ImageLabel` reaches it through `_record_history`, which looks the
  method up defensively — the unit tests drive `ImageLabel` with
  lightweight stand-in main windows.

## Keyboard Shortcuts — Bind Each Sequence Once

Three mechanisms are in play, and mixing them wrongly silently breaks
the binding:

| Kind | Mechanism | Examples |
|------|-----------|----------|
| Menu commands | `QAction` with `setShortcut` | Ctrl+Z, Ctrl+S, Ctrl+D |
| Plain keys needing global reach | Application-wide event filter | P R B E, 1-9 |
| Legacy single keys | `QShortcut` (ApplicationShortcut) | A, D, C, F2 |

**A sequence bound twice is ambiguous and Qt fires neither.** This was
measured: with Ctrl+Z on both the Edit menu action and a `QShortcut`,
pressing Ctrl+Z did nothing at all. Ctrl+Y is therefore a second
sequence on the *same* redo action (`setShortcuts([...])`), not a
separate shortcut object.

Menu actions that must work while focus sits in the frame list or a tool
panel need `setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)`
— that is why the undo and redo actions set it.

`_WorkflowKeyFilter` handles the unmodified tool and class keys. Its
"is the user typing?" test inspects the widget the event was **delivered
to**, not `QApplication.focusWidget()`: the focus widget is null
whenever the window is not active, which would let a tool key steal a
character out of the frame filter box.

Beware platform-resolved standard keys: `StandardKey.Redo` is
Ctrl+Shift+Z on Linux and macOS but **Ctrl+Y on Windows**, so listing it
alongside a literal `"Ctrl+Y"` binds Ctrl+Y twice on Windows. Build such
lists by de-duplicating, and assert on the count — a membership check
cannot see a duplicate.

The filter is also gated on `isActiveWindow()`: several child windows
(help, the Snake easter egg, the YOLO training dialog) are shown
non-modally, and without the gate a tool key aimed at one of them would
act on the canvas behind it.

## Thread Safety for YOLO Training

### Training Thread

```python
class TrainingThread(QThread):
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(object)

    def run(self):
        try:
            results = self.yolo_trainer.train_model(
                epochs=self.epochs,
                imgsz=self.imgsz
            )
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit(str(e))
```

**UI Update**:
- Training runs in background thread
- Progress updates via Qt signals
- UI remains responsive during training

## Error Handling

### YOLO Model/Data Mismatch

**Problem**: Loading YOLO model trained on different classes

**Solution**:
```python
try:
    model = YOLO(model_path)
    model_classes = model.names
    yaml_classes = data_yaml['names']

    if model_classes != yaml_classes:
        QMessageBox.warning(
            self,
            "Class Mismatch",
            f"Model classes: {model_classes}\n"
            f"Data classes: {yaml_classes}"
        )
        return
except Exception as e:
    # Handle gracefully instead of crashing
```

## Multi-dimensional Image Slicing

### Dimension Assignment

User assigns meaning to each dimension:
```
TIFF shape: (10, 50, 3, 512, 512)
User assigns: T   Z   C   H    W

Result: 10 timepoints × 50 Z-slices × 3 channels = 1500 slices
Each slice: 512×512 pixels
```

### Slice Naming Convention

```python
def generate_slice_name(filename, t, z, c, s):
    parts = []
    if t is not None:
        parts.append(f"T{t}")
    if z is not None:
        parts.append(f"Z{z}")
    if c is not None:
        parts.append(f"C{c}")
    if s is not None:
        parts.append(f"S{s}")

    return f"{filename}_{'_'.join(parts)}"

# Example: "stack.tif_T0_Z5_C0"
```

## Keyboard Shortcuts

### Global Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New Project |
| Ctrl+O | Open Project |
| Ctrl+S | Save Project |
| Ctrl+W | Close Project |
| Ctrl+Shift+S | Annotation Statistics |
| F1 | Help Window |

### Canvas Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+Wheel | Zoom In/Out |
| Ctrl+Drag | Pan |
| Esc | Cancel Current Annotation |
| Enter | Finish/Accept Annotation |
| Up/Down | Navigate Slices (multi-dimensional) |
| -/= | Adjust Brush/Eraser Size |

## Logging and Debug Output

### Print Statements

Current implementation uses `print()` for debugging:
```python
print(f"Changed SAM model to: {model_name}")
print(f"SAM input points: {all_points}, labels: {all_labels}")
print(f"Loading project from: {project_path}")
```

**Note**: No formal logging framework is used. Output goes to console.

## DINO Temp Annotations — Single Field, Many Images

`ImageLabel.temp_annotations` is a **single list on the image_label**,
not a per-image cache. It holds the pending DINO+SAM masks shown as
an overlay while the user decides accept/reject. The per-image batch
cache is `ImageAnnotator.dino_batch_results` (a dict keyed by image
name) — `image_label.temp_annotations` is only ever set to one image's
slice of that dict at a time.

Consequences this codebase has tripped over:

- **Image/slice switches must re-sync** `temp_annotations` from
  `dino_batch_results` for the new image (load if pending, clear if
  not). Otherwise masks from the previously-viewed image visually
  bleed onto every slice the user navigates to. See
  `_refresh_dino_temp_for_current()`.
- **Enter / Escape during review** must work even when the focus is on
  slice_list / image_list / a button — `QListWidget` consumes
  Enter for itemActivated before `ImageLabel.keyPressEvent` ever sees
  it. Solved with an application-wide event filter
  (`_DINOReviewEventFilter`) that fires only while
  `temp_annotations` has DINO items and skips modal dialogs and text
  inputs. Setting `image_label.setFocus()` synchronously inside
  `_show_dino_batch_review` was not enough — Qt's focus handling
  raced the click event that opened the review and the canvas
  often didn't end up focused. `QTimer.singleShot(0, …)` defers until
  the current event chain settles.
- **Auto-accept dropdown applies to both paths.** The batch-mode
  combo ("Review before accepting" / "Auto-accept all detections")
  controls **both** "Detect Current Image" and "Detect All Images".
  Only checking it in `run_dino_detection_batch` and not
  `run_dino_detection_single` produced a confusing "auto-accept
  doesn't actually auto-accept for single image" bug.
- **Batch detection must enumerate slices, not just `all_images`.**
  Multi-dim images live in `all_images` as a single entry with
  `is_multi_slice=True`, and their actual slice QImages live under
  `self.image_slices[base_name]`. The first cut of
  `run_dino_detection_batch` iterated `all_images` and skipped the
  multi-slice entries with a console log — leaving stack-based
  projects unable to use "Detect All Images" at all. Batch jobs go
  through `_collect_dino_batch_work_items()` which flattens regular
  images + every loaded slice into a `(name, QImage)` list.
- **Review navigation must handle slice names.** Slice names like
  `stack_T1_Z1_C1` are not in `image_list`. After collecting batch
  results for slices, `_navigate_to_image_or_slice()` finds the
  parent image via `os.path.splitext` matching and then activates
  the specific row in `slice_list`. Without this, batch review on
  slices either silently no-op'd or showed the first regular
  image's masks on a slice.

## Multi-dimensional TIFF Axis Defaults

`load_tiff` extracts `tif.series[0].axes` (e.g. `"TZCYX"`) and maps
it through `{T:T, Z:Z, C:C, S:S, Y:H, X:W}` to populate the
`DimensionDialog` combo boxes. This is what lets a user open an
ImageJ-style 5D TIFF and just click OK.

When the metadata is missing or unfamiliar, fall back to the
hand-crafted defaults keyed on `ndim`:

| ndim | default labels |
|------|---------------|
| 3 | `Z H W` |
| 4 | `T Z H W` |
| 5 | `T Z C H W` |
| 6 | `T Z C S H W` |

**Do not** use `default_dimensions[-ndim:]` of a shorter list to
"extend" defaults — that silently degrades for `ndim ≥ 5`: the final
combo gets no default and inherits the first item ("T"), which is
the wrong axis. The 5D TZCYX bug that produced 2560 one-row slices
on a `(2,5,2,256,256)` file came from exactly this.

## SAM 3 Frame Identity and Annotation Compatibility

SAM 3 propagation outputs use integer frame indices, while the annotator stores
annotations under image filenames. `FrameSequence` is the translation layer
between these identities. Its ordering must match Meta's loader: numeric
filename stems sort numerically; all other names use lexical ordering.

SAM 3 masks must be converted into the existing annotation contract before
being committed:

```python
{
    "segmentation": [x1, y1, x2, y2, ...],
    "category_id": int,
    "category_name": str,
}
```

Do not introduce a parallel `polygon` field or nested point arrays. Existing
rendering, area calculation, project persistence, and exporters consume the
flattened `segmentation` field.

## Bounded-Memory Video Loading

Normal videos must not be decoded into an in-memory Python list. Video import
uses `cv2.VideoCapture` and holds only the current decoded frame while writing
selected lossless PNG frames to the application cache. The worker then copies the clip
to the project image directory before registration. The user controls the
inclusive start/end range and stride before decoding begins. Both phases run
on `VideoExtractionThread`; GUI widgets are updated only through Qt signals.

There are three related frame identities:

| Identity | Example | Purpose |
|----------|---------|---------|
| Clip index | `0` | Position used by A/D navigation and tracking APIs |
| Source index | `1057` | Original zero-based frame number in the video |
| Frame name | `weld_ab12_frame_000000000042.png` | Stable clip-position key used by `image_paths` and `all_annotations` |

`FrameSequence` owns the active mapping. The optional `.iap` `video_sessions`
field persists every clip, while old projects without that field continue to
load normally. Both `FrameSequence` and `ImageAnnotator` build filename indexes
when sequences or sessions change, avoiding full-frame scans during navigation.
The index also enforces one clip-session owner per case-folded project filename,
because a global annotation key cannot safely represent two different sequence
contexts. Case-folding is applied on every platform so a project created on a
case-sensitive filesystem cannot become ambiguous when moved to Windows.
Generated names include a cache fingerprint so two different videos with the
same basename cannot silently overwrite each other's annotations.

Clip caches carry an application marker. Recursive cleanup requires both that
marker and a resolved path strictly below the application's expected cache
root. Runtime cache paths are never restored from `.iap` data. Cancellation,
successful project copying, project clearing, and application close remove
only directories satisfying both checks.

SAM 3 never receives the project's shared `images/` directory. Before model
initialization, the application builds a managed tracker workspace containing
only the active clip's frames, named by zero-padded clip position. This keeps
the model's integer output indices aligned with `FrameSequence` even when the
project also contains unrelated images or other clips.

Any membership or ordering change to the active sequence closes the current
SAM 3 session before rebuilding `FrameSequence`. This prevents old model frame
indices from being mapped through a newly shortened sequence.

Project JSON uses a same-directory temporary file, file flush plus `fsync`, and
`os.replace`. POSIX builds also attempt a parent-directory `fsync`; Windows uses
same-volume atomic replacement. A mid-write failure leaves the previous `.iap`
intact. Image copies are staged to temporary files in the destination directory
and replaced atomically. Before committing, save builds a candidate image-path
mapping and adopts an occupied destination only when a byte comparison confirms
it is the same source image; a different same-named file aborts the save.
`save_project()` catches filesystem and serialization errors, removes files
created by the failed attempt, restores project identity and `image_paths`, and
returns `False`. Callers such as Close Project and Save As must stop their
workflow when that happens. Pending `Temp-*` review annotations are excluded
from serialization without mutating the live review state.

## Export Format Filename Matching

`export_formats.py` historically looked up image paths via substring
match:

```python
image_path = next(
    (path for name, path in image_paths.items() if image_name in name),
    None,
)
```

That is fragile — `"bee.jpg" in "honeybee.jpg"` returns True and you
write the wrong file. The COCO, YOLO v4, and YOLO v5+ exports all
share this code path.

**Always try the exact key first; fall back to substring only if no
exact key matches.** Pattern:

```python
image_path = image_paths.get(image_name)
if image_path is None:
    image_path = next(
        (path for name, path in image_paths.items() if image_name in name),
        None,
    )
```

The substring fallback is kept for backward compatibility with old
projects that may have stored normalised image names (e.g. without
extension); new code should prefer the exact-key path.
