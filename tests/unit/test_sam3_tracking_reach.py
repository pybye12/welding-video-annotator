"""How far a tracking run gets before it gives up.

Reported from the lab: "Track All to End" stopped after about nine
frames. It was not Meta's propagation limit — with
``max_frame_num_to_track=None`` SAM 3 uses the full video range. It was
this app's own plausibility gates, which were anchored to the polygon
the annotator drew and never moved. A droplet necking down before it
detaches is a real, gradual change, and the fixed floor could not tell
it apart from drift.
"""

import math

import numpy as np
import pytest

from digitalsreeni_image_annotator.sam3_tracker import SAM3Tracker


FRAME = (200, 200)
SOURCE_AREA = 3600.0  # a 60x60 droplet


def _square(area, centre=(60, 60), size=200):
    """A filled square of roughly ``area`` px centred on ``centre``."""
    mask = np.zeros((size, size), dtype=bool)
    half = max(1, int(round(math.sqrt(area) / 2)))
    cx, cy = centre
    mask[
        max(0, cy - half) : min(size, cy + half),
        max(0, cx - half) : min(size, cx + half),
    ] = True
    return mask


class ShrinkingDropletPredictor:
    """A droplet losing a fifth of its area every frame, then detaching.

    This is the shape of the real footage: the consumable necks down
    steadily before the droplet separates. Nothing here is drift.
    """

    def __init__(self, frames=30, decay=0.8):
        self.frames = frames
        self.decay = decay
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "s"}
        return {"is_success": True}

    def handle_stream_request(self, request):
        self.requests.append(request)
        area = SOURCE_AREA
        for index in range(self.frames):
            yield {
                "frame_index": 2 + index,
                "outputs": {
                    "out_obj_ids": np.array([7]),
                    "out_binary_masks": _square(area)[None, ...],
                },
            }
            area *= self.decay

    def shutdown(self):
        pass


def _tracker(predictor, tmp_path):
    tracker = SAM3Tracker("unused.pt", predictor=predictor)
    tracker.init_state(str(tmp_path))
    return tracker


def _source_polygon():
    return [(30, 30), (90, 30), (90, 90), (30, 90)]


def _flat(points):
    return [coordinate for point in points for coordinate in point]


def test_a_steadily_shrinking_droplet_is_followed_far_past_the_old_limit(tmp_path):
    """The bug, reproduced and fixed.

    Under the old source-anchored floor of 15%, a fifth lost per frame
    crosses the threshold on frame nine — which is exactly the "about
    nine images" that was reported. Judging each frame against the
    previous one instead lets the droplet shrink as far as it physically
    does, down to the absolute 50 px floor.
    """
    tracker = _tracker(ShrinkingDropletPredictor(), tmp_path)

    results = tracker.track_polygons(2, [(7, _flat(_source_polygon()))], FRAME)

    labelled = [index for index, objects in results if objects]
    # The old rule: max(50, 0.15 * 3600) = 540 px, reached on frame nine.
    old_limit = sum(1 for n in range(30) if SOURCE_AREA * 0.8**n >= 540)
    assert old_limit == 9, "the old floor is what produced the reported nine"
    # Now limited by the absolute 50 px floor rather than a ratio of the
    # starting size, so the run doubles its reach on the same footage.
    assert len(labelled) >= 2 * old_limit, (
        f"only reached {len(labelled)} frames; the whole point is to get "
        "well past the old nine"
    )


def test_noise_masks_are_skipped_without_ending_the_requested_run(tmp_path):
    """Tracking a two-pixel speck is rejected, but later frames are checked."""
    tracker = _tracker(ShrinkingDropletPredictor(frames=60), tmp_path)

    results = tracker.track_polygons(2, [(7, _flat(_source_polygon()))], FRAME)

    report = tracker.last_run_report[7]
    assert len(results) == 60
    assert report["processed_through"] == 61
    assert report["skipped"] > 0
    assert "px floor" in report["last_issue"]
    assert not report["stopped"]


class TeleportingPredictor(ShrinkingDropletPredictor):
    """The mask jumps to the bottom of the frame — drift onto the plate.

    Same size, so every area gate is satisfied. Only a position check
    can catch this one.
    """

    def handle_stream_request(self, request):
        self.requests.append(request)
        for index in range(10):
            centre = (60, 60) if index < 2 else (160, 185)
            yield {
                "frame_index": 2 + index,
                "outputs": {
                    "out_obj_ids": np.array([7]),
                    "out_binary_masks": _square(SOURCE_AREA, centre)[None, ...],
                },
            }


def test_a_mask_that_jumps_across_the_frame_is_refused(tmp_path):
    """The molten_consumable-on-the-background-plate failure.

    The area gates pass it happily — it is the right size. Before this
    check there was nothing at all comparing where the mask was to where
    the object had been.
    """
    tracker = _tracker(TeleportingPredictor(), tmp_path)

    results = tracker.track_polygons(2, [(7, _flat(_source_polygon()))], FRAME)

    labelled = [index for index, objects in results if objects]
    assert labelled == [2, 3], (
        "the jumped frames must not become annotations"
    )
    report = tracker.last_run_report[7]
    assert len(results) == 10
    assert report["processed_through"] == 11
    assert report["skipped"] == 8
    assert "jumped" in report["last_issue"]
    assert not report["stopped"]


def test_the_report_counts_the_frames_it_produced(tmp_path):
    tracker = _tracker(ShrinkingDropletPredictor(frames=5), tmp_path)

    results = tracker.track_polygons(2, [(7, _flat(_source_polygon()))], FRAME)

    labelled = [index for index, objects in results if objects]
    # The prompt frame is reported separately from propagated frames.
    assert tracker.last_run_report[7]["frames"] == len(labelled) - 1
    assert tracker.last_run_report[7]["last_frame"] == labelled[-1]


# --- the gate in isolation ------------------------------------------------


def _reference(area, centroid=(60.0, 60.0)):
    return {"area": area, "centroid": centroid}


def test_shape_drift_into_long_background_region_is_rejected():
    mask = np.zeros(FRAME[::-1], dtype=bool)
    mask[96:104, 20:180] = True

    segmentation, reason = SAM3Tracker.evaluate_tracked_mask(
        mask,
        {
            "area": 1280.0,
            "centroid": (100.0, 100.0),
            "source_elongation": 1.0,
        },
        FRAME,
    )

    assert segmentation is None
    assert "elongated" in reason


def test_gradual_change_is_accepted_and_a_single_jump_is_not():
    accepted, reason = SAM3Tracker.evaluate_tracked_mask(
        _square(2000), _reference(2500), FRAME
    )
    assert accepted and reason == ""

    _, reason = SAM3Tracker.evaluate_tracked_mask(
        _square(100), _reference(2500), FRAME
    )
    assert "shrank" in reason

    _, reason = SAM3Tracker.evaluate_tracked_mask(
        _square(20000), _reference(500), FRAME
    )
    assert "grew" in reason


def test_an_empty_mask_reports_the_absence_rather_than_a_gate():
    """The caller treats "nothing there" differently from "we said no"."""
    segmentation, reason = SAM3Tracker.evaluate_tracked_mask(
        np.zeros((200, 200), dtype=bool), _reference(2500), FRAME
    )
    assert segmentation is None
    assert reason == "no mask"


def test_a_mask_covering_half_the_frame_is_always_refused():
    huge = np.ones((200, 200), dtype=bool)
    _, reason = SAM3Tracker.evaluate_tracked_mask(huge, _reference(20000), FRAME)
    assert "half the frame" in reason


def test_a_droplet_moving_fast_between_frames_is_still_accepted():
    """The travel allowance has to be generous or fast droplets strand."""
    segmentation, reason = SAM3Tracker.evaluate_tracked_mask(
        _square(400, centre=(60, 100)), _reference(400, (60.0, 60.0)), FRAME
    )
    assert segmentation, reason
