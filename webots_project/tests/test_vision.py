"""Unit tests for the colour detector (no Webots needed).

Run from the webots_project folder:
    python -m pytest tests -q
"""

import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "controllers" / "colour_detector"))
import vision  # noqa: E402
from classifier import preprocess, softmax  # noqa: E402

W, H = 52, 39  # e-puck camera resolution

# Colours as they appear in the arena (BGR). Grass/water/sky values are from
# the course's "Webots with OpenCV" slides and must NOT be detected as blocks.
BGR = {
    "red": (25, 25, 220),
    "dark_red_shadow": (20, 20, 90),
    "green": (30, 200, 30),
    "blue": (220, 30, 30),
    "floor_grey": (166, 166, 166),
    "grass_yellow_green": (80, 224, 197),
    "water_cyan": (230, 180, 30),
    "sky_grey_blue": (150, 100, 80),
}


def scene(background="floor_grey", block=None, box=(20, 12, 12, 10)):
    img = np.full((H, W, 3), BGR[background], np.uint8)
    if block:
        x, y, w, h = box
        img[y:y + h, x:x + w] = BGR[block]
    return img


@pytest.mark.parametrize("colour", ["red", "green", "blue"])
def test_detects_each_block_colour(colour):
    dets = vision.detect_blocks(scene(block=colour))
    assert [d.colour for d in dets] == [colour]
    d = dets[0]
    assert d.bbox == (20, 12, 12, 10)
    assert d.area_fraction == pytest.approx(120 / (W * H), rel=0.05)  # blur trims corners
    assert d.centroid_x == pytest.approx(25.5)


def test_shadowed_red_face_still_red():
    assert [d.colour for d in vision.detect_blocks(scene(block="dark_red_shadow"))] == ["red"]


@pytest.mark.parametrize("bg", ["floor_grey", "grass_yellow_green", "water_cyan", "sky_grey_blue"])
def test_background_is_not_a_block(bg):
    assert vision.detect_blocks(scene(background=bg)) == []


def test_tiny_noise_is_ignored():
    img = scene()
    img[5, 5] = BGR["red"]  # a single red pixel
    assert vision.detect_blocks(img) == []


def test_largest_block_first():
    img = scene(block="blue", box=(2, 2, 6, 6))
    img[10:35, 20:50] = BGR["red"]
    assert [d.colour for d in vision.detect_blocks(img)] == ["red", "blue"]


def test_horizontal_offset_sign():
    left = vision.detect_blocks(scene(block="green", box=(0, 10, 8, 8)))[0]
    right = vision.detect_blocks(scene(block="green", box=(44, 10, 8, 8)))[0]
    assert left.horizontal_offset(W) < -0.5  # block on the left -> negative
    assert right.horizontal_offset(W) > 0.5  # block on the right -> positive


def test_webots_buffer_conversion():
    bgra = np.zeros((H, W, 4), np.uint8)
    bgra[..., 2] = 200  # red channel
    bgr = vision.webots_to_bgr(bgra.tobytes(), W, H)
    assert bgr.shape == (H, W, 3) and bgr[0, 0].tolist() == [0, 0, 200]
    assert vision.webots_to_bgr(None, W, H) is None
    with pytest.raises(ValueError):
        vision.webots_to_bgr(b"\x00" * 10, W, H)


@pytest.mark.parametrize("bbox", [None, (0, 0, 5, 5), (45, 30, 7, 9), (10, 5, 30, 30)])
def test_square_crop_is_square_and_inside(bbox):
    crop = vision.square_crop(scene(), bbox)
    assert crop.shape[0] == crop.shape[1] > 0
    assert crop.shape[0] <= min(W, H)


def test_cnn_preprocess_matches_training_format():
    x = preprocess(scene(block="red"))
    assert x.shape == (1, 32, 32, 3) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    # BGR -> RGB: the red block must be in channel 0 after conversion
    assert x[0, ..., 0].max() > 0.8


def test_softmax_is_a_distribution():
    p = softmax(np.array([1000.0, 1001.0, 999.0]))  # large logits must not overflow
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0) and p.argmax() == 1


def test_annotate_upscales_without_modifying_input():
    img = scene(block="red")
    before = img.copy()
    out = vision.annotate(img, vision.detect_blocks(img), "x")
    assert out.shape[1] >= 300 and np.array_equal(img, before)
    upscaled = cv.resize(img, out.shape[1::-1], interpolation=cv.INTER_NEAREST)
    assert np.any(out != upscaled)  # boxes/labels were drawn


# --------------------------------------------------------------- classifier
import time  # noqa: E402

from classifier import CLASS_NAMES, AsyncClassifier, load_backend  # noqa: E402


def _wait_for_results(clf, n, timeout=2.0):
    got, end = [], time.time() + timeout
    while len(got) < n and time.time() < end:
        got += clf.poll()
        time.sleep(0.01)
    return got


def test_async_classifier_returns_prediction():
    logits = np.zeros(10, np.float32)
    logits[5] = 5.0  # "dog"
    clf = AsyncClassifier(Path("."), backend=lambda batch: logits)
    assert clf.submit("t1", scene(block="red"))
    (pred,) = _wait_for_results(clf, 1)
    assert pred.tag == "t1" and pred.label == CLASS_NAMES[5] and not pred.error
    assert pred.confidence > 0.9 and len(pred.top3) == 3


def test_async_classifier_reports_failures_then_disables():
    def broken(batch):
        raise RuntimeError("boom")

    clf = AsyncClassifier(Path("."), backend=broken)
    results = []
    for i in range(3):
        clf.submit(f"t{i}", scene())
        results += _wait_for_results(clf, 1)
    assert all(r.error.startswith("RuntimeError") for r in results) and len(results) == 3
    assert not clf.enabled and "DISABLED" in clf.description
    assert clf.submit("late", scene()) is False


def test_async_classifier_rejects_wrong_output_size():
    clf = AsyncClassifier(Path("."), backend=lambda batch: np.zeros(7))
    clf.submit("t", scene())
    (pred,) = _wait_for_results(clf, 1)
    assert "invalid output" in pred.error


def test_missing_model_disables_cnn_with_reason(tmp_path):
    backend, reason = load_backend(tmp_path)
    assert backend is None and "no model file" in reason
    assert AsyncClassifier(tmp_path).submit("x", scene()) is False
