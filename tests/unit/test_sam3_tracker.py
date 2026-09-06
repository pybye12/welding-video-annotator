import numpy as np
from types import SimpleNamespace

import digitalsreeni_image_annotator.sam3_tracker as sam3_tracker_module
from digitalsreeni_image_annotator.sam3_tracker import SAM3Tracker


class FakePredictor:
    def __init__(self):
        self.requests = []
        self.shutdown_called = False

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "session-1"}
        if request["type"] == "add_prompt" and "bounding_boxes" in request:
            first = np.zeros((20, 20), dtype=bool)
            first[2:8, 2:8] = True
            second = np.zeros((20, 20), dtype=bool)
            second[11:18, 11:18] = True
            return {
                "outputs": {
                    "out_obj_ids": np.array([20, 10]),
                    "out_binary_masks": np.stack([second, first]),
                }
            }
        return {"is_success": True}

    def handle_stream_request(self, request):
        self.requests.append(request)
        object_id = next(
            (
                previous["obj_id"]
                for previous in reversed(self.requests)
                if previous.get("type") == "add_prompt" and "obj_id" in previous
            ),
            7,
        )
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:11, 2:11] = True
        yield {
            "frame_index": 3,
            "outputs": {
                "out_obj_ids": np.array([object_id]),
                "out_binary_masks": mask[None, ...],
            },
        }

    def shutdown(self):
        self.shutdown_called = True


class OffTargetPredictor(FakePredictor):
    def handle_stream_request(self, request):
        self.requests.append(request)
        off_target = np.zeros((20, 20), dtype=bool)
        off_target[11:20, 11:20] = True
        on_target = np.zeros((20, 20), dtype=bool)
        on_target[2:11, 2:11] = True
        for frame_index, mask in [(2, off_target), (3, on_target)]:
            yield {
                "frame_index": frame_index,
                "outputs": {
                    "out_obj_ids": np.array([7]),
                    "out_binary_masks": mask[None, ...],
                },
            }


class MergeThenNewDropletPredictor(FakePredictor):
    def handle_stream_request(self, request):
        self.requests.append(request)
        valid = np.zeros((20, 20), dtype=bool)
        valid[2:11, 2:11] = True
        empty = np.zeros((20, 20), dtype=bool)
        later_new_droplet = np.zeros((20, 20), dtype=bool)
        later_new_droplet[12:20, 12:20] = True
        for frame_index, mask in [
            (2, valid),
            (3, empty),
            (4, empty),
            (5, later_new_droplet),
        ]:
            yield {
                "frame_index": frame_index,
                "outputs": {
                    "out_obj_ids": np.array([7]),
                    "out_binary_masks": mask[None, ...],
                },
            }


class FakeMaskSeedTracker:
    def __init__(self):
        self.added_masks = []
        self.preflight_states = []

    def add_new_mask(self, state, frame_idx, object_id, mask):
        self.added_masks.append((state, frame_idx, object_id, mask))

    def propagate_in_video_preflight(self, state, run_mem_encoder):
        self.preflight_states.append((state, run_mem_encoder))


def test_tracker_uses_supported_session_and_point_prompt_api(tmp_path):
    predictor = FakePredictor()
    tracker = SAM3Tracker("unused.pt", predictor=predictor)

    assert tracker.init_state(str(tmp_path)) == "session-1"
    results = tracker.track_points(2, [(7, (12.5, 24.0))])

    assert results[0][0] == 3
    assert 7 in results[0][1]
    assert [request["type"] for request in predictor.requests] == [
        "start_session",
        "reset_session",
        "add_prompt",
        "propagate_in_video",
    ]
    prompt = predictor.requests[2]
    assert prompt["obj_id"] == 7
    assert prompt["points"] == [[12.5, 24.0]]
    assert prompt["point_labels"] == [1]
    assert prompt["rel_coordinates"] is False


def test_tracker_normalizes_original_image_points(tmp_path):
    predictor = FakePredictor()
    tracker = SAM3Tracker("unused.pt", predictor=predictor)
    tracker.init_state(str(tmp_path))

    tracker.track_points(2, [(7, (100, 50))], frame_size=(200, 100))

    prompt = predictor.requests[2]
    assert prompt["points"] == [[0.5, 0.5]]
    assert prompt["rel_coordinates"] is True


def test_extract_masks_and_convert_to_annotator_segmentation():
    mask = np.zeros((20, 20), dtype=bool)
    mask[3:12, 5:15] = True
    outputs = {
        "out_obj_ids": np.array([4]),
        "out_binary_masks": mask[None, ...],
    }

    masks = SAM3Tracker.extract_masks_from_outputs(outputs)
    segmentations = SAM3Tracker.mask_to_segmentations(masks[4])

    assert masks[4].dtype == np.uint8
    assert len(segmentations) == 1
    assert len(segmentations[0]) >= 6
    assert all(isinstance(value, int) for value in segmentations[0])


def test_tracker_uses_boxes_and_maps_model_ids_back_to_requested_ids(tmp_path):
    predictor = FakePredictor()
    tracker = SAM3Tracker("unused.pt", predictor=predictor)
    tracker.init_state(str(tmp_path))

    results = tracker.track_boxes(
        2,
        [(7, (1, 1, 8, 8)), (9, (10, 10, 9, 9))],
        (20, 20),
    )

    prompt = predictor.requests[2]
    assert prompt["bounding_boxes"] == [
        [0.05, 0.05, 0.4, 0.4],
        [0.5, 0.5, 0.45, 0.45],
    ]
    assert prompt["bounding_box_labels"] == [1, 1]
    assert set(results[0][1]) == {7}


def test_tracker_uses_source_polygon_as_normalized_prompt(tmp_path):
    predictor = FakePredictor()
    tracker = SAM3Tracker("unused.pt", predictor=predictor)
    tracker.init_state(str(tmp_path))

    results = tracker.track_polygons(
        2,
        [(7, [2, 2, 12, 2, 12, 12, 2, 12])],
        (20, 20),
        max_frame_num_to_track=4,
    )

    prompt = predictor.requests[2]
    assert prompt["obj_id"] == 7
    assert prompt["rel_coordinates"] is True
    assert prompt["point_labels"].count(1) == 6
    assert prompt["point_labels"].count(0) == 8
    assert all(0 <= coordinate <= 1 for point in prompt["points"] for coordinate in point)
    assert set(results[0][1]) == {7}
    propagate = next(
        request for request in predictor.requests
        if request["type"] == "propagate_in_video"
    )
    assert propagate["max_frame_num_to_track"] == 4


def test_tracker_uses_exact_polygon_mask_when_meta_api_is_available(tmp_path):
    predictor = FakePredictor()
    mask_seed_tracker = FakeMaskSeedTracker()
    predictor.model = SimpleNamespace(
        tracker=mask_seed_tracker,
        _get_tracker_inference_states_by_obj_ids=lambda state, object_ids: [
            "tracker-state"
        ],
    )
    predictor._all_inference_states = {
        "session-1": {"state": "session-state"},
    }
    tracker = SAM3Tracker("unused.pt", predictor=predictor)
    tracker.init_state(str(tmp_path))

    tracker.track_polygons(
        2,
        [(7, [2, 2, 12, 2, 12, 12, 2, 12])],
        (20, 20),
    )

    _, frame_idx, object_id, mask = mask_seed_tracker.added_masks[0]
    assert frame_idx == 2
    assert object_id == 7
    assert np.count_nonzero(mask.numpy()) > 0
    assert mask_seed_tracker.preflight_states == [("tracker-state", True)]


def test_match_output_objects_to_boxes_uses_mask_overlap():
    first = np.zeros((30, 30), dtype=bool)
    first[2:10, 2:10] = True
    second = np.zeros((30, 30), dtype=bool)
    second[18:27, 18:27] = True
    outputs = {
        "out_obj_ids": np.array([41, 42]),
        "out_binary_masks": np.stack([first, second]),
    }

    mapping = SAM3Tracker.match_output_objects_to_boxes(
        outputs, [(5, (0, 0, 12, 12)), (6, (16, 16, 13, 13))]
    )

    assert mapping == {41: 5, 42: 6}


def test_rejects_catastrophic_mask_growth():
    reasonable = np.zeros((100, 100), dtype=bool)
    reasonable[10:30, 10:30] = True
    drifted = np.ones((100, 100), dtype=bool)

    assert SAM3Tracker.is_plausible_tracked_mask(reasonable, 400, 10_000)
    assert not SAM3Tracker.is_plausible_tracked_mask(drifted, 400, 10_000)


def test_refinement_points_are_normalized_and_include_background():
    mask = np.zeros((100, 200), dtype=bool)
    mask[30:60, 80:120] = True

    points, labels = SAM3Tracker.mask_to_refinement_points(mask, (200, 100))

    assert len(points) == len(labels)
    assert labels.count(1) == 4
    assert labels.count(0) == 8
    assert all(0 <= coordinate <= 1 for point in points for coordinate in point)


def test_largest_component_is_kept_and_small_spatter_is_ignored():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:50, 20:50] = True
    mask[70:73, 70:73] = True

    segmentation = SAM3Tracker.largest_plausible_segmentation(
        mask, source_area=900, frame_area=10_000
    )

    points = np.asarray(segmentation).reshape(-1, 2)
    assert points[:, 0].max() < 60
    assert points[:, 1].max() < 60


def test_spatter_sized_component_is_rejected_relative_to_source_droplet():
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:48, 40:48] = True

    segmentation = SAM3Tracker.largest_plausible_segmentation(
        mask, source_area=900, frame_area=10_000
    )

    assert segmentation is None


def test_off_target_prompt_frame_rejects_entire_propagation(tmp_path):
    predictor = OffTargetPredictor()
    tracker = SAM3Tracker("unused.pt", predictor=predictor)
    tracker.init_state(str(tmp_path))

    results = tracker.track_polygons(
        2,
        [(7, [2, 2, 12, 2, 12, 12, 2, 12])],
        (20, 20),
    )

    assert results == [(2, {})]


def test_track_reaches_end_without_reusing_id_for_new_droplet(tmp_path):
    predictor = MergeThenNewDropletPredictor()
    tracker = SAM3Tracker("unused.pt", predictor=predictor)
    tracker.init_state(str(tmp_path))

    results = tracker.track_polygons(
        2,
        [(7, [2, 2, 12, 2, 12, 12, 2, 12])],
        (20, 20),
    )

    assert set(results[0][1]) == {7}
    assert len(results) == 4
    assert all(not objects for _, objects in results[1:])
    report = tracker.last_run_report[7]
    assert report["processed_through"] == 5
    assert report["skipped"] == 3
    assert not report["stopped"]


def test_segmentation_overlap_ratio_detects_correct_source_region():
    source_mask = SAM3Tracker.polygon_to_mask(
        [2, 2, 12, 2, 12, 12, 2, 12],
        (20, 20),
    )

    assert SAM3Tracker.segmentation_overlap_ratio(
        [3, 3, 11, 3, 11, 11, 3, 11], source_mask
    ) > 0.9
    assert (
        SAM3Tracker.segmentation_overlap_ratio(
            [14, 14, 19, 14, 19, 19, 14, 19], source_mask
        )
        == 0.0
    )


def test_polygon_to_mask_clips_points_to_frame():
    mask = SAM3Tracker.polygon_to_mask(
        [-10, -10, 15, -10, 15, 15, -10, 15],
        (20, 20),
    )

    assert mask.shape == (20, 20)
    assert np.count_nonzero(mask) > 0


def test_unload_closes_session_and_shuts_down(tmp_path):
    predictor = FakePredictor()
    tracker = SAM3Tracker("unused.pt", predictor=predictor)
    tracker.init_state(str(tmp_path))

    tracker.unload()

    assert predictor.requests[-1]["type"] == "close_session"
    assert predictor.shutdown_called
    assert tracker.predictor is None
    assert not tracker.is_initialized


def test_configures_cpu_connected_components_fallback(monkeypatch):
    cpu_fallback = object()
    module = SimpleNamespace(
        connected_components=object(),
        connected_components_cpu=cpu_fallback,
    )
    monkeypatch.setattr(
        sam3_tracker_module.importlib.util, "find_spec", lambda name: None
    )
    monkeypatch.setattr(
        sam3_tracker_module.importlib, "import_module", lambda name: module
    )

    SAM3Tracker._configure_connected_components_fallback()

    assert module.connected_components is cpu_fallback
