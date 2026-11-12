"""Tests for the dataset labeller's camera projection and the shape model hook."""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "controllers"
sys.path.insert(0, str(ROOT / "shape_dataset_collector"))
sys.path.insert(0, str(ROOT / "colour_detector"))
from projection import (CAMERA_OFFSET, Box3D, focal_length_px, overlap_fraction,  # noqa: E402
                        project_box, project_point)
from classifier import SHAPE_CLASSES, SHAPE_MODEL, AsyncClassifier, is_not_a_block, load_backend  # noqa: E402

W, H = 52, 39


def test_point_straight_ahead_at_camera_height_hits_image_centre():
    u, v, d = project_point((1.0, 0.0, CAMERA_OFFSET[2]), (0.0, 0.0), 0.0, W, H)
    assert u == pytest.approx(W / 2) and v == pytest.approx(H / 2)
    assert d == pytest.approx(1.0 - CAMERA_OFFSET[0])


def test_left_and_up_move_the_point_left_and_up_in_the_image():
    u, v, _ = project_point((1.0, 0.2, 0.2), (0.0, 0.0), 0.0, W, H)
    assert u < W / 2 and v < H / 2


def test_robot_heading_is_respected():
    # Robot at origin facing +y: a point on the +y axis is straight ahead.
    u, _, _ = project_point((0.0, 1.0, CAMERA_OFFSET[2]), (0.0, 0.0), math.pi / 2, W, H)
    assert u == pytest.approx(W / 2)


def test_point_behind_the_camera_is_not_projected():
    assert project_point((-1.0, 0.0, 0.0), (0.0, 0.0), 0.0, W, H) is None


def test_block_ahead_has_expected_width_and_is_fully_visible():
    distance = 1.0
    box = Box3D(distance, 0.0, 0.0, 0.3, 0.3, 0.1)
    p = project_box(box, (0.0, 0.0), 0.0, W, H)
    assert p is not None and p.visible_fraction == pytest.approx(1.0)
    # Width of the near face: 0.3 m at (distance - 0.15 - camera offset) metres.
    expected = focal_length_px(W) * 0.3 / (distance - 0.15 - CAMERA_OFFSET[0])
    assert abs(p.bbox[2] - expected) <= 2
    assert abs((p.bbox[0] + p.bbox[2] / 2) - W / 2) <= 1


def test_object_outside_the_field_of_view_is_none():
    assert project_box(Box3D(0.0, 1.0, 0.0, 0.2, 0.2, 0.2), (0.0, 0.0), 0.0, W, H) is None


def test_object_cut_by_the_image_edge_reports_partial_visibility():
    p = project_box(Box3D(1.0, 0.45, 0.0, 0.3, 0.3, 0.1), (0.0, 0.0), 0.0, W, H)
    assert p is not None and 0 < p.visible_fraction < 1 and p.bbox[0] == 0


def test_overlap_fraction():
    assert overlap_fraction((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(0.5)
    assert overlap_fraction((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_shape_classifier_uses_its_own_class_names():
    logits = np.array([0.0, 5.0, 0.0, 0.0], np.float32)  # "ball"
    clf = AsyncClassifier(Path("."), backend=lambda batch: logits, spec=SHAPE_MODEL)
    assert clf.submit("s", np.full((12, 12, 3), 100, np.uint8))
    got, end = [], time.time() + 2
    while not got and time.time() < end:
        got = clf.poll()
        time.sleep(0.01)
    assert got and got[0].label == SHAPE_CLASSES[1] and not got[0].error


def test_missing_shape_model_names_the_expected_file(tmp_path):
    backend, reason = load_backend(tmp_path, SHAPE_MODEL)
    assert backend is None and "webots_shapes_int8.tflite" in reason


def test_ball_outline_does_not_depend_on_its_own_rotation():
    boxes = [Box3D(1.0, 0.0, yaw, 0.2, 0.2, 0.2, round=True) for yaw in (0.0, 0.4, 0.785)]
    bboxes = {project_box(b, (0.0, 0.0), 0.0, W, H).bbox for b in boxes}
    assert len(bboxes) == 1


def test_ball_width_matches_its_radius_at_the_near_surface():
    r, d = 0.1, 1.0
    p = project_box(Box3D(d, 0.0, 0.0, 2 * r, 2 * r, 2 * r, round=True), (0.0, 0.0), 0.0, W, H)
    expected = 2 * focal_length_px(W) * r / (d - CAMERA_OFFSET[0] - r)
    assert abs(p.bbox[2] - expected) <= 2
    assert p.depth == pytest.approx(d - CAMERA_OFFSET[0] - r)


@pytest.mark.parametrize("verdict, rejected", [
    (None, False),                 # shape model off or failed: keep the HSV result
    (("block", 0.99), False),
    (("ball", 0.97), True),
    (("cylinder", 0.61), True),
    (("background", 0.80), True),
    (("ball", 0.40), False),       # unsure: keep the HSV result
])
def test_shape_verdict_rejects_only_confident_non_blocks(verdict, rejected):
    assert is_not_a_block(verdict, 0.6) is rejected


def test_model_spec_without_files_loads_nothing(tmp_path):
    from classifier import ModelSpec
    backend, reason = load_backend(tmp_path, ModelSpec("shape", SHAPE_CLASSES, ()))
    assert backend is None and "no model files" in reason
