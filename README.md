# Welding Video Annotator

A desktop application for drawing segmentation labels on images and long video
sequences. This fork is based on
[DigitalSreeni Image Annotator](https://github.com/bnsreenu/digitalsreeni-image-annotator)
and adds a practical welding-video workflow, optional SAM 3 propagation, and
safer training-mask export.

The app runs locally. Images, videos, projects, and exported labels are not
uploaded by the application.

## Main Features

- Draw and correct labels with Polygon, Rectangle, Paint Brush, and Eraser.
- Use SAM 2 point or box prompts for assisted image segmentation.
- Open selected ranges from large videos without loading the full video into RAM.
- Navigate extracted frames and preserve their original source-frame numbers.
- Propagate a selected polygon through later frames with optional SAM 3 tracking.
- Adjust preview brightness and contrast without changing source images or labels.
- Save work as an `.iap` project and continue later.
- Export COCO, YOLO segmentation, Pascal VOC, class-ID masks, or RGB masks.
- Keep custom classes for general datasets or apply the included ER70S-6 presets.
- Use the guided **Label** and **Auto-track** workspaces without model jargon.
- Undo and redo annotation changes per frame with `Ctrl+Z` / `Ctrl+Shift+Z`.
- See at a glance how much of a clip is labeled, and filter down to the frames
  still to do.
- Work keyboard-first: `A`/`D` to move between frames, `P`/`R`/`B`/`E` to pick a
  tool, `1`-`9` to switch class. `Ctrl+/` lists everything.
- Create a team review package with source frames, exact RGB masks, overlays,
  and a browser page before labeling an entire dataset.

## What This Fork Adds

| Area | Original DigitalSreeni tool | Changes in this fork |
|---|---|---|
| Input | Images and image stacks | Direct video-clip and frame-folder loading |
| Large videos | Image-oriented workflow | Select start/end frames and stride; decode one frame at a time |
| Navigation | Image list | Video sessions, source-frame numbers, and `A`/`D` frame navigation |
| Assisted labeling | SAM 2 on individual images | Optional SAM 3 forward mask propagation from a selected polygon |
| Review | Standard image display | Non-destructive display controls and shareable review packages |
| Welding setup | User-created classes | ER70S-6 Full Arc and CAVITAR class/color presets |
| Export | Existing annotation formats | Strict multiclass RGB masks, blank masks, overlap checks, and safe staged export |
| Reliability | Upstream behavior | Automated unit, UI, and integration tests for the added workflows |
| Correcting mistakes | No undo | Per-frame undo/redo across every annotation edit |
| Tracking progress | Plain file list | Labeled markers, a "N of M labeled" bar, a name filter, and a "Todo" toggle |
| Session state | Spread across panels | One status line: active tool, class, cursor position, brush size, save state |
| Appearance | Two hand-written stylesheets that drifted | Light and dark generated from one token table, so both stay complete |

SAM 3 does **not** decide whether a region is a droplet, molten consumable, or
arc. The annotator chooses the class and draws the first polygon; SAM 3 proposes
masks for later frames, which must still be reviewed and corrected.

## Quick Setup: Windows

### 1. Install the prerequisites

- [Python 3.11](https://www.python.org/downloads/) - enable **Add Python to PATH** during installation.
- [Git for Windows](https://git-scm.com/download/win) - this includes Git Bash.

### 2. Install the app

Open **Git Bash**, then run:

```bash
git clone https://github.com/pybye12/welding-video-annotator.git
cd welding-video-annotator
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The first installation can take several minutes because the image and machine
learning libraries are large.

On macOS or Linux, use `python3` instead of `py -3.11` and activate with
`source .venv/bin/activate`.

### 3. Start the app

```bash
sreeni
```

To open it again later:

```bash
cd welding-video-annotator
source .venv/Scripts/activate
sreeni
```

## Basic Labeling Workflow

1. Create a project and save the `.iap` file.
2. Click **Add New Images**, or use **Video > Open Video Clip** to choose a
   manageable frame range from a video.
3. Add your own classes, or choose a preset from the **Welding** menu.
4. Select a class and draw labels with **Polygon**, **Paint Brush**, or another
   annotation tool. Press **Enter** to finish a polygon.
5. Use the brightness and contrast controls when a boundary is faint. These
   controls affect only the screen preview; brush strokes no longer change the
   apparent brightness of untouched pixels.
6. Review every frame and correct inaccurate boundaries.
7. Select a class before using **Eraser**. It corrects only that class, so a
   nearby class is not accidentally removed.
8. Save the project regularly. The status line along the bottom shows when the
   project last reached disk.
9. Select an export format and click **Export Training Labels**. Use a new
   empty folder when exporting RGB semantic masks.
10. Before a full labeling run, click **Create Review Package**, choose about
    five labeled frames, and send the generated `.zip` file to the reviewer.

### ER70S-6 presets

Use **Welding > Add ER70S-6 CAVITAR Classes** for the CAVITAR assignment:

- `molten_consumable`: orange `(255, 128, 0)`, while attached to the wire.
- `droplet`: red `(255, 0, 0)`, only after detachment.

Use **Welding > Add ER70S-6 Full Arc Classes** for full-arc recordings. It adds:

- `external_arc`: blue `(0, 0, 255)`.
- `internal_arc`: yellow `(255, 255, 0)`.

Background, weld pool, and spatter remain black and are not labeled. The full
boundary rules are in the
[ER70S-6 labeling protocol](docs/ER70S6_LABELING_PROTOCOL.md).

## Optional SAM 3 Tracking

The normal manual tools, SAM 2 tools, video loading, display controls, and
exports work without SAM 3.

SAM 3 tracking additionally requires:

- [Meta's official SAM 3 package](https://github.com/facebookresearch/sam3),
  installed in the same virtual environment.
- A compatible SAM 3 `.pt` or `.pth` checkpoint.
- An NVIDIA GPU with CUDA for practical inference speed.

Place a checkpoint named `sam3-001.pt` in the repository root, or select the
checkpoint when prompted. Then:

1. Draw a polygon on a clear starting frame.
2. Click that polygon in the **Annotations** list.
3. Open the **Auto-track** tab.
4. Click **1. Prepare Loaded Frames**.
5. Click **2. Track Selected to End**.
6. Review all generated masks and fix drift with the manual tools.

If the app reports **No valid polygon annotations selected**, select the
finished polygon row in the Annotations panel before tracking.

## Keyboard

Press `Ctrl+/` in the app for the full list. The ones worth learning first:

| Key | Action |
|---|---|
| `A` / `D` | Previous / next frame |
| `C` | Copy the selected annotation to the next frame |
| `P` / `R` / `B` / `E` | Polygon / box / paint brush / eraser |
| `1` - `9` | Select label class by position in the list |
| `Enter` | Finish the polygon, or accept proposed masks |
| `Esc` | Cancel the current shape, or reject proposed masks |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo the last annotation change on this frame |
| `Ctrl+0` | Fit the frame to the window |
| `Ctrl+D` | Switch between light and dark |

Undo history is kept per frame, so undoing never rewrites a frame you are not
looking at.

## Tech Stack

| Purpose | Technology |
|---|---|
| Desktop interface | Python 3.10+ and PyQt6 |
| Images and video | OpenCV, Pillow, NumPy, scikit-image, Shapely |
| Assisted segmentation | PyTorch, Ultralytics SAM 2, optional Meta SAM 3 |
| Data formats | COCO, YOLO segmentation, Pascal VOC, semantic PNG masks |
| Verification | pytest and pytest-qt |

## Tests

From an activated development environment:

```bash
python -m pytest -q
```

Implementation details and known limitations are documented in
[SAM 3 Welding Video Annotator Adaptation](docs/SAM3_WELDING_VIDEO_ADAPTATION.md).

## Credits and License

This project adapts the open-source
[DigitalSreeni Image Annotator](https://github.com/bnsreenu/digitalsreeni-image-annotator)
created by Dr. Sreenivas Bhattiprolu. The original manual annotation, SAM 2,
project, import, and export capabilities remain the foundation of this fork.

Distributed under the [MIT License](LICENSE).
