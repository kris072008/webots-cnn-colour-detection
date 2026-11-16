"""Tests for the animal photographer's background subtraction."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "controllers" / "animal_photographer"))
from silhouette import silhouette  # noqa: E402


def scene():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)


def test_box_and_mask_cover_exactly_the_changed_pixels():
    without = scene()
    with_animal = without.copy()
    with_animal[30:70, 50:110] += 128  # the "animal": every channel changes by 128 (uint8 wraps)
    mask, bbox = silhouette(with_animal, without)
    assert bbox == (50, 30, 60, 40)
    assert mask[30:70, 50:110].all() and mask.sum() == 60 * 40


def test_identical_frames_give_no_animal():
    frame = scene()
    assert silhouette(frame, frame.copy()) == (None, None)


def test_a_few_noisy_pixels_are_not_an_animal():
    without = scene()
    with_animal = without.copy()
    with_animal[10, 10] = 255 - with_animal[10, 10]
    with_animal[100, 150] = 255 - with_animal[100, 150]
    assert silhouette(with_animal, without) == (None, None)
