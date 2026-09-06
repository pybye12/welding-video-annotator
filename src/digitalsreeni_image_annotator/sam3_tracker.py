"""SAM 3 video tracking adapter.

This module wraps Meta's supported session-based video predictor API. Existing
annotation polygons become mask or dense point prompts so the user's exact
object selection, rather than an ambiguous image region, seeds tracking.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from contextlib import nullcontext

import cv2
import numpy as np


class SAM3Tracker:
    """Small adapter around Meta's SAM 3 video predictor."""

    MIN_TRACKED_AREA = 50.0
    MIN_SOURCE_AREA_RATIO = 0.15

    # SAM returned no mask at all: the object has left the frame or merged
    # into the pool. Give up quickly — continuing risks re-using this
    # object's id for whatever appears next.
    MAX_CONSECUTIVE_MISSES = 2

    # SAM returned a mask and our own gates rejected it. Be slower to give
    # up here: the gates are heuristics, and two unlucky frames (an arc
    # flash, a moment of occlusion) should not end a run that is otherwise
    # tracking correctly.
    MAX_CONSECUTIVE_REJECTIONS = 5

    # How much the tracked area may change from the *previous accepted*
    # frame. A droplet necking down and detaching changes size gradually,
    # so a per-frame limit lets it shrink to any size over enough frames
    # while a single catastrophic jump is still refused.
    MAX_AREA_STEP = 4.0

    # A mask may never cover more than this fraction of the frame.
    MAX_FRAME_FRACTION = 0.5

    # How far the mask's centre may travel in one frame, as a fraction of
    # the frame diagonal. This is what catches drift onto the background
    # plate: the size gates alone cannot tell a plausible blob in the
    # wrong place from the right one.
    MAX_CENTROID_STEP_FRACTION = 0.25

    def __init__(self, checkpoint_path, predictor=None):
        self.checkpoint_path = checkpoint_path
        self.predictor = predictor
        self.session_id = None
        # Filled in by track_polygons: per object, how many frames it
        # produced and why it stopped. Read by the window so a short run
        # can explain itself instead of going quiet.
        self.last_run_report = {}
        if self.predictor is None:
            self._load_predictor()

    @property
    def is_initialized(self):
        return self.session_id is not None

    def _load_predictor(self):
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"SAM 3 checkpoint not found: {self.checkpoint_path}")

        try:
            import torch
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 3 is not installed. Install Meta's official sam3 package "
                "and its supported CUDA/PyTorch dependencies."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3 video tracking requires a CUDA-capable GPU.")

        self._configure_connected_components_fallback()
        try:
            self.predictor = build_sam3_video_predictor(
                checkpoint_path=self.checkpoint_path,
                gpus_to_use=[torch.cuda.current_device()],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SAM 3 predictor from {self.checkpoint_path}: {exc}"
            ) from exc

    @staticmethod
    def _cuda_autocast_context():
        """Keep SAM 3 CUDA ops in one dtype during GUI-driven inference.

        Some official SAM 3 builds run activations in bfloat16 on CUDA. Without
        an explicit autocast context, PyTorch can hit mixed bfloat16/float32
        convolution inputs during interactive prompts.
        """
        try:
            import torch

            if torch.cuda.is_available():
                return torch.autocast("cuda", dtype=torch.bfloat16)
        except Exception:
            pass
        return nullcontext()

    @staticmethod
    def _configure_connected_components_fallback():
        """Use Meta's CPU implementation when CUDA CC backends are unavailable.

        The official SAM 3 connected-components dispatcher tries ``cc_torch``
        and then Triton for CUDA tensors. Standard Windows environments often
        have neither. Meta also ships a correct CPU implementation, so route
        through it instead of failing at the first interactive point prompt.
        """
        if (
            importlib.util.find_spec("cc_torch") is not None
            or importlib.util.find_spec("triton") is not None
        ):
            return
        module = importlib.import_module("sam3.perflib.connected_components")
        module.connected_components = module.connected_components_cpu
        print(
            "[SAM3] cc_torch/Triton unavailable; using the slower CPU "
            "connected-components fallback."
        )

    def init_state(self, video_folder_path):
        """Start a predictor session for a directory of ordered video frames."""
        if not os.path.isdir(video_folder_path):
            raise ValueError(
                f"Path must be a directory containing frames: {video_folder_path}"
            )
        if self.predictor is None:
            raise RuntimeError("SAM 3 predictor is not loaded.")

        self.close_session()
        with self._cuda_autocast_context():
            response = self.predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": video_folder_path,
                    "offload_video_to_cpu": True,
                }
            )
        self.session_id = response["session_id"]
        return self.session_id

    def track_points(self, frame_idx, object_points, frame_size=None):
        """Add per-object point prompts and propagate them forward.

        Args:
            frame_idx: Frame index in the SAM 3 session.
            object_points: Iterable of ``(object_id, (x, y))`` entries.
            frame_size: Optional ``(width, height)``. When supplied, points are
                original-image pixels and are normalized for Meta's tracker.

        Returns:
            List of ``(frame_idx, segmentations_by_object_id)`` results.
            Full-resolution masks are converted while streaming so long videos
            do not retain every GPU/CPU mask until propagation finishes.
        """
        if not self.is_initialized:
            raise RuntimeError("Call init_state before tracking.")

        with self._cuda_autocast_context():
            self.predictor.handle_request(
                {"type": "reset_session", "session_id": self.session_id}
            )
            for object_id, point in object_points:
                prompt_point = list(point)
                rel_coordinates = False
                if frame_size is not None:
                    frame_width, frame_height = frame_size
                    prompt_point = [
                        float(point[0]) / frame_width,
                        float(point[1]) / frame_height,
                    ]
                    rel_coordinates = True
                self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": self.session_id,
                        "frame_index": frame_idx,
                        "obj_id": object_id,
                        "points": [prompt_point],
                        "point_labels": [1],
                        "rel_coordinates": rel_coordinates,
                    }
                )

            request = {
                "type": "propagate_in_video",
                "session_id": self.session_id,
                "propagation_direction": "forward",
                "start_frame_index": frame_idx,
            }
            results = []
            for response in self.predictor.handle_stream_request(request):
                segmentations_by_object = {
                    object_id: self.mask_to_segmentations(mask)
                    for object_id, mask in self.extract_masks_from_outputs(
                        response["outputs"]
                    ).items()
                }
                results.append((response["frame_index"], segmentations_by_object))
        return results

    def track_boxes(self, frame_idx, object_boxes, frame_size):
        """Initialize objects from tight boxes and propagate them forward."""
        if not self.is_initialized:
            raise RuntimeError("Call init_state before tracking.")

        frame_width, frame_height = frame_size
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame size must contain positive dimensions.")

        requested_boxes = []
        normalized_boxes = []
        for object_id, box in object_boxes:
            x, y, width, height = (float(value) for value in box)
            if width <= 0 or height <= 0:
                continue
            x = min(max(x, 0.0), float(frame_width - 1))
            y = min(max(y, 0.0), float(frame_height - 1))
            width = min(width, float(frame_width) - x)
            height = min(height, float(frame_height) - y)
            if width <= 0 or height <= 0:
                continue
            requested_boxes.append((int(object_id), (x, y, width, height)))
            normalized_boxes.append(
                [
                    x / frame_width,
                    y / frame_height,
                    width / frame_width,
                    height / frame_height,
                ]
            )
        if not requested_boxes:
            raise ValueError("No valid object boxes supplied.")

        with self._cuda_autocast_context():
            self.predictor.handle_request(
                {"type": "reset_session", "session_id": self.session_id}
            )
            prompt_response = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": self.session_id,
                    "frame_index": frame_idx,
                    "bounding_boxes": normalized_boxes,
                    "bounding_box_labels": [1] * len(normalized_boxes),
                }
            )
        model_to_requested = self.match_output_objects_to_boxes(
            prompt_response.get("outputs", {}), requested_boxes
        )
        if not model_to_requested:
            raise RuntimeError("SAM 3 did not find an object inside the selected box.")

        matched_masks = {
            model_to_requested[model_id]: mask
            for model_id, mask in self.extract_masks_from_outputs(
                prompt_response.get("outputs", {})
            ).items()
            if model_id in model_to_requested
        }
        with self._cuda_autocast_context():
            self.predictor.handle_request(
                {"type": "reset_session", "session_id": self.session_id}
            )
            for requested_id, mask in matched_masks.items():
                points, labels = self.mask_to_refinement_points(mask, frame_size)
                self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": self.session_id,
                        "frame_index": frame_idx,
                        "obj_id": requested_id,
                        "points": points,
                        "point_labels": labels,
                        "rel_coordinates": True,
                    }
                )

        source_areas = {
            object_id: width * height
            for object_id, (_, _, width, height) in requested_boxes
        }
        frame_area = frame_width * frame_height
        request = {
            "type": "propagate_in_video",
            "session_id": self.session_id,
            "propagation_direction": "forward",
            "start_frame_index": frame_idx,
        }
        results = []
        with self._cuda_autocast_context():
            for response in self.predictor.handle_stream_request(request):
                segmentations_by_object = {}
                for model_id, mask in self.extract_masks_from_outputs(
                    response["outputs"]
                ).items():
                    requested_id = int(model_id)
                    if requested_id not in source_areas:
                        continue
                    segmentation = self.largest_plausible_segmentation(
                        mask, source_areas[requested_id], frame_area
                    )
                    if segmentation:
                        segmentations_by_object[requested_id] = [segmentation]
                results.append((response["frame_index"], segmentations_by_object))
        return results

    def track_polygons(
        self,
        frame_idx,
        object_polygons,
        frame_size,
        max_frame_num_to_track=None,
    ):
        """Initialize objects from exact polygons and propagate forward.

        Each output object is reduced to its largest connected component.
        Components smaller than a conservative fraction of the source polygon
        are ignored as likely tracking noise. These geometric gates apply to
        every class; this adapter does not inspect class names.
        """
        if not self.is_initialized:
            raise RuntimeError("Call init_state before tracking.")

        frame_width, frame_height = frame_size
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame size must contain positive dimensions.")

        prompts = []
        source_areas = {}
        source_masks = {}
        for object_id, segmentation in object_polygons:
            source_mask = self.polygon_to_mask(segmentation, frame_size)
            source_area = float(np.count_nonzero(source_mask))
            if source_area < self.MIN_TRACKED_AREA:
                continue
            points, labels = self.mask_to_refinement_points(
                source_mask, frame_size, positive_count=6
            )
            prompts.append((int(object_id), points, labels))
            source_areas[int(object_id)] = source_area
            source_masks[int(object_id)] = source_mask
        if not prompts:
            raise ValueError("No valid object polygons supplied.")

        with self._cuda_autocast_context():
            self.predictor.handle_request(
                {"type": "reset_session", "session_id": self.session_id}
            )
            for object_id, points, labels in prompts:
                self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": self.session_id,
                        "frame_index": frame_idx,
                        "obj_id": object_id,
                        "points": points,
                        "point_labels": labels,
                        "rel_coordinates": True,
                    }
                )
                self._replace_tracker_prompt_with_mask(
                    frame_idx, object_id, source_masks[object_id]
                )

            frame_area = frame_width * frame_height
            request = {
                "type": "propagate_in_video",
                "session_id": self.session_id,
                "propagation_direction": "forward",
                "start_frame_index": frame_idx,
            }
            if max_frame_num_to_track is not None:
                request["max_frame_num_to_track"] = max_frame_num_to_track
            results = []
            active_object_ids = set(source_areas)
            consecutive_misses = {object_id: 0 for object_id in source_areas}
            consecutive_rejections = {object_id: 0 for object_id in source_areas}
            # Each object's last accepted frame, which every later frame is
            # judged against. Seeded from the drawn polygon.
            references = {
                object_id: {
                    "area": source_areas[object_id],
                    "centroid": self._mask_centroid(source_masks[object_id]),
                }
                for object_id in source_areas
            }
            report = {
                object_id: {"frames": 0, "last_frame": None, "stopped": ""}
                for object_id in source_areas
            }
            self.last_run_report = report

            for response in self.predictor.handle_stream_request(request):
                frame_index = response["frame_index"]
                segmentations_by_object = {}
                masks_by_object = self.extract_masks_from_outputs(response["outputs"])
                for object_id in list(active_object_ids):
                    mask = masks_by_object.get(object_id)

                    if frame_index == frame_idx:
                        # The prompt frame. The polygon the annotator drew
                        # is the right answer here, so keep the strict
                        # source-anchored gate: a mask that disagrees with
                        # it means the prompt itself failed.
                        segmentation = (
                            self.largest_plausible_segmentation(
                                mask, source_areas[object_id], frame_area
                            )
                            if mask is not None
                            else None
                        )
                        if not segmentation or self.segmentation_overlap_ratio(
                            segmentation, source_masks[object_id]
                        ) < 0.1:
                            active_object_ids.discard(object_id)
                            report[object_id]["stopped"] = (
                                "SAM 3 could not reproduce the polygon on the "
                                "frame it was given"
                            )
                            continue
                        segmentations_by_object[object_id] = [segmentation]
                        continue

                    if mask is None:
                        segmentation, reason = None, "no mask"
                    else:
                        segmentation, reason = self.evaluate_tracked_mask(
                            mask, references[object_id], frame_size
                        )

                    if segmentation:
                        consecutive_misses[object_id] = 0
                        consecutive_rejections[object_id] = 0
                        contour = np.asarray(segmentation, dtype=np.float32).reshape(
                            -1, 2
                        )
                        references[object_id] = {
                            "area": float(cv2.contourArea(contour)),
                            "centroid": self.contour_centroid(
                                contour.astype(np.int32)
                            ),
                        }
                        segmentations_by_object[object_id] = [segmentation]
                        report[object_id]["frames"] += 1
                        report[object_id]["last_frame"] = frame_index
                        continue

                    # Two different kinds of failure, with different
                    # tolerances: "SAM found nothing" means the object is
                    # gone, while "our gate said no" may just mean the
                    # gate is wrong about one awkward frame.
                    if reason == "no mask":
                        consecutive_misses[object_id] += 1
                        if (
                            consecutive_misses[object_id]
                            >= self.MAX_CONSECUTIVE_MISSES
                        ):
                            active_object_ids.discard(object_id)
                            report[object_id]["stopped"] = (
                                f"SAM 3 found nothing on "
                                f"{self.MAX_CONSECUTIVE_MISSES} frames in a row"
                            )
                    else:
                        consecutive_rejections[object_id] += 1
                        if (
                            consecutive_rejections[object_id]
                            >= self.MAX_CONSECUTIVE_REJECTIONS
                        ):
                            active_object_ids.discard(object_id)
                            report[object_id]["stopped"] = (
                                f"{self.MAX_CONSECUTIVE_REJECTIONS} masks in a "
                                f"row looked wrong (last: {reason})"
                            )
                results.append((frame_index, segmentations_by_object))
                if not active_object_ids:
                    break
        return results

    def _replace_tracker_prompt_with_mask(self, frame_idx, object_id, source_mask):
        """Seed Meta's tracker from the exact polygon when its mask API exists.

        Meta's session request layer currently exposes text, box, and point
        prompts but not the underlying tracker's mask prompt. The official
        model object does expose that capability, so use it defensively while
        retaining the point prompt as a compatibility fallback.
        """
        predictor_states = getattr(self.predictor, "_all_inference_states", None)
        model = getattr(self.predictor, "model", None)
        tracker = getattr(model, "tracker", None)
        get_states = getattr(model, "_get_tracker_inference_states_by_obj_ids", None)
        add_new_mask = getattr(tracker, "add_new_mask", None)
        preflight = getattr(tracker, "propagate_in_video_preflight", None)
        if not (
            isinstance(predictor_states, dict)
            and callable(get_states)
            and callable(add_new_mask)
            and callable(preflight)
        ):
            return False

        session = predictor_states.get(self.session_id)
        if not isinstance(session, dict) or "state" not in session:
            return False
        tracker_states = get_states(session["state"], [object_id])
        if len(tracker_states) != 1:
            return False

        try:
            import torch

            mask_tensor = torch.as_tensor(source_mask, dtype=torch.bool)
            with self._cuda_autocast_context():
                add_new_mask(tracker_states[0], frame_idx, object_id, mask_tensor)
                preflight(tracker_states[0], run_mem_encoder=True)
            return True
        except Exception as exc:
            print(f"[SAM3] exact polygon mask seed unavailable: {exc}")
            return False

    @classmethod
    def mask_to_refinement_points(cls, mask, frame_size, positive_count=4):
        """Create normalized positive and negative tracker points from a mask."""
        frame_width, frame_height = frame_size
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        positives = []
        suppression_radius = max(3, int(np.sqrt(np.count_nonzero(mask)) / 8))
        for _ in range(positive_count):
            y, x = np.unravel_index(np.argmax(distance), distance.shape)
            if distance[y, x] <= 0:
                break
            positives.append((float(x), float(y)))
            cv2.circle(distance, (int(x), int(y)), suppression_radius, 0, -1)

        box = cls.mask_bbox(mask)
        if not positives or box is None:
            raise RuntimeError("SAM 3 returned an empty prompt mask.")
        x, y, width, height = box
        margin = max(3.0, min(width, height) * 0.15)
        negatives = [
            (x - margin, y - margin),
            (x + width / 2, y - margin),
            (x + width + margin, y - margin),
            (x - margin, y + height / 2),
            (x + width + margin, y + height / 2),
            (x - margin, y + height + margin),
            (x + width / 2, y + height + margin),
            (x + width + margin, y + height + margin),
        ]
        points = positives + negatives
        normalized = [
            [
                min(max(px / frame_width, 0.0), 1.0),
                min(max(py / frame_height, 0.0), 1.0),
            ]
            for px, py in points
        ]
        return normalized, [1] * len(positives) + [0] * len(negatives)

    @classmethod
    def match_output_objects_to_boxes(cls, outputs, requested_boxes):
        """Greedily match SAM 3 prompt-frame masks to requested source boxes."""
        candidates = []
        for model_id, mask in cls.extract_masks_from_outputs(outputs).items():
            mask_box = cls.mask_bbox(mask)
            if mask_box is None:
                continue
            for requested_id, requested_box in requested_boxes:
                overlap = cls.bbox_iou(mask_box, requested_box)
                if overlap > 0:
                    candidates.append((overlap, model_id, requested_id))

        mapping = {}
        used_requested_ids = set()
        for _, model_id, requested_id in sorted(candidates, reverse=True):
            if model_id in mapping or requested_id in used_requested_ids:
                continue
            mapping[model_id] = requested_id
            used_requested_ids.add(requested_id)
        return mapping

    @staticmethod
    def mask_bbox(mask):
        """Return an ``(x, y, width, height)`` box for a non-empty mask."""
        ys, xs = np.nonzero(np.asarray(mask) > 0)
        if not len(xs):
            return None
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        return min_x, min_y, max_x - min_x + 1, max_y - min_y + 1

    @staticmethod
    def bbox_iou(first, second):
        """Calculate intersection over union for two xywh boxes."""
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        intersection = intersection_width * intersection_height
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def is_plausible_tracked_mask(mask, source_box_area, frame_area):
        """Reject catastrophic drift masks before they become annotations."""
        mask_area = int(np.count_nonzero(mask))
        if mask_area == 0:
            return False
        return (
            mask_area <= max(source_box_area * 20.0, 500.0)
            and mask_area <= frame_area * 0.5
        )

    @staticmethod
    def polygon_to_mask(segmentation, frame_size):
        """Rasterize an annotator polygon into a frame-sized binary mask."""
        frame_width, frame_height = frame_size
        points = np.asarray(segmentation, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            return np.zeros((frame_height, frame_width), dtype=np.uint8)
        points[:, 0] = np.clip(points[:, 0], 0, frame_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, frame_height - 1)
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        return mask

    @staticmethod
    def _mask_centroid(mask):
        """Centre of mass of a binary mask, or None when it is empty."""
        moments = cv2.moments((np.asarray(mask) > 0).astype(np.uint8), binaryImage=True)
        if not moments["m00"]:
            return None
        return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    @staticmethod
    def _largest_contour(mask):
        """Biggest external contour of a mask, with its area, or None."""
        binary_mask = (np.asarray(mask) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = [contour for contour in contours if len(contour) >= 3]
        if not contours:
            return None, 0.0
        largest = max(contours, key=cv2.contourArea)
        return largest, float(cv2.contourArea(largest))

    @staticmethod
    def contour_centroid(contour):
        """Centre of mass of a contour, falling back to its bounding box."""
        moments = cv2.moments(contour)
        if moments["m00"]:
            return (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            )
        x, y, width, height = cv2.boundingRect(contour)
        return (x + width / 2.0, y + height / 2.0)

    @classmethod
    def evaluate_tracked_mask(cls, mask, reference, frame_size):
        """Accept or reject one propagated mask, against the last good one.

        The gates are relative to the *previous accepted frame*, not to
        the polygon the annotator drew. Anchoring to the source made a
        real physical change — a droplet necking down before it detaches
        — indistinguishable from drift, so tracking gave up part way
        through exactly the event being studied. Comparing frame to
        frame instead allows any amount of gradual change while still
        refusing a single implausible jump.

        ``reference`` is ``{"area": float, "centroid": (x, y)}`` from the
        last accepted frame. Returns ``(segmentation | None, reason)``;
        ``reason`` is empty when the mask was accepted and otherwise says
        in plain words why it was not, so a run that stops early can
        explain itself.
        """
        frame_width, frame_height = frame_size
        frame_area = float(frame_width) * float(frame_height)
        contour, area = cls._largest_contour(mask)
        if contour is None:
            return None, "no mask"

        if area < cls.MIN_TRACKED_AREA:
            return None, f"only {int(area)} px, below the {int(cls.MIN_TRACKED_AREA)} px floor"
        if area > frame_area * cls.MAX_FRAME_FRACTION:
            return None, "covered more than half the frame"

        reference_area = float(reference["area"])
        if reference_area > 0:
            if area > reference_area * cls.MAX_AREA_STEP:
                return None, (
                    f"grew {area / reference_area:.1f}x in one frame"
                )
            if area * cls.MAX_AREA_STEP < reference_area:
                return None, (
                    f"shrank to 1/{reference_area / max(area, 1.0):.1f} in one frame"
                )

        centroid = cls.contour_centroid(contour)
        previous = reference.get("centroid")
        if previous is not None:
            travelled = float(
                np.hypot(centroid[0] - previous[0], centroid[1] - previous[1])
            )
            diagonal = float(np.hypot(frame_width, frame_height))
            # A plain fraction of the frame, with no term scaling to the
            # object's own size. An earlier version added ``4 * sqrt(area)``
            # so a small droplet could move several of its own widths —
            # but for a large object that term reached 240 px in a 200 px
            # frame and swallowed the check entirely, which is exactly the
            # drift this is here to catch. A quarter of the diagonal is
            # already far more travel than one frame of welding video
            # shows, small object or not.
            allowance = diagonal * cls.MAX_CENTROID_STEP_FRACTION
            if travelled > allowance:
                return None, (
                    f"jumped {int(travelled)} px across the frame in one step"
                )

        return contour.flatten().tolist(), ""

    @classmethod
    def largest_plausible_segmentation(cls, mask, source_area, frame_area):
        """Source-frame gate: strict, and anchored to the drawn polygon.

        Used only on the frame the annotator prompted from, where the
        polygon they drew *is* the right answer and a mask that differs
        wildly from it means the prompt failed. Propagated frames go
        through ``evaluate_tracked_mask`` instead.
        """
        binary_mask = (np.asarray(mask) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = [contour for contour in contours if len(contour) >= 3]
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        largest_area = float(cv2.contourArea(largest))
        minimum_area = max(
            cls.MIN_TRACKED_AREA,
            float(source_area) * cls.MIN_SOURCE_AREA_RATIO,
        )
        if largest_area < minimum_area:
            return None
        if largest_area > max(float(source_area) * 20.0, 500.0):
            return None
        if largest_area > float(frame_area) * 0.5:
            return None
        return largest.flatten().tolist()

    @staticmethod
    def segmentation_overlap_ratio(segmentation, source_mask):
        """Measure overlap against the source using the smaller region's area."""
        points = np.asarray(segmentation, dtype=np.int32).reshape(-1, 2)
        if len(points) < 3:
            return 0.0
        predicted_mask = np.zeros_like(source_mask, dtype=np.uint8)
        cv2.fillPoly(predicted_mask, [points], 1)
        predicted_area = int(np.count_nonzero(predicted_mask))
        source_area = int(np.count_nonzero(source_mask))
        smaller_area = min(predicted_area, source_area)
        if smaller_area == 0:
            return 0.0
        intersection = int(np.count_nonzero(predicted_mask & source_mask))
        return intersection / smaller_area

    def close_session(self):
        if self.predictor is None or self.session_id is None:
            self.session_id = None
            return
        try:
            self.predictor.handle_request(
                {"type": "close_session", "session_id": self.session_id}
            )
        finally:
            self.session_id = None

    def unload(self):
        self.close_session()
        predictor = self.predictor
        if predictor is not None:
            for candidate in (
                getattr(predictor, "model", None),
                getattr(getattr(predictor, "predictor", None), "model", None),
            ):
                try:
                    if candidate is not None and hasattr(candidate, "cpu"):
                        candidate.cpu()
                except Exception as exc:
                    print(f"[SAM3] unload: warning moving model to CPU: {exc}")
            if hasattr(predictor, "shutdown"):
                predictor.shutdown()
        self.predictor = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def extract_masks_from_outputs(outputs):
        """Return ``{object_id: uint8_mask}`` from a predictor output."""
        if not isinstance(outputs, dict):
            return {}
        out_obj_ids = outputs.get("out_obj_ids")
        out_masks = outputs.get("out_binary_masks")
        if out_obj_ids is None or out_masks is None:
            return {}

        if hasattr(out_obj_ids, "detach"):
            out_obj_ids = out_obj_ids.detach().cpu().numpy()
        else:
            out_obj_ids = np.asarray(out_obj_ids)
        if hasattr(out_masks, "detach"):
            out_masks = out_masks.detach().cpu().numpy()
        else:
            out_masks = np.asarray(out_masks)

        masks = {}
        for object_id, mask in zip(out_obj_ids, out_masks):
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[0]
            masks[int(object_id)] = (mask > 0).astype(np.uint8)
        return masks

    @staticmethod
    def mask_to_segmentations(mask, min_area=10):
        """Convert a binary mask to flattened polygons used by the annotator."""
        contours, _ = cv2.findContours(
            (np.asarray(mask) > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        return [
            contour.flatten().tolist()
            for contour in contours
            if len(contour) >= 3 and cv2.contourArea(contour) > min_area
        ]
