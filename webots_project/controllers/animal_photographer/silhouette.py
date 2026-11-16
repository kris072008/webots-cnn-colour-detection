"""Background subtraction for the animal photographer (pure, unit-tested)."""

from __future__ import annotations

import cv2 as cv
import numpy as np

DIFF_THRESHOLD = 30  # summed BGR difference that counts as "animal"
MIN_AREA_PX = 30


def silhouette(with_animal: np.ndarray, without: np.ndarray):
    """Mask and bounding box of the pixels that change when the animal is removed."""
    diff = cv.absdiff(with_animal, without).astype(np.int32).sum(axis=2)
    mask = (diff > DIFF_THRESHOLD).astype(np.uint8)
    # 3x3 is symmetric about its centre; an even-sized kernel would shift the box by a pixel.
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if int(mask.sum()) < MIN_AREA_PX:
        return None, None
    x, y, w, h = cv.boundingRect(mask)
    return mask, (int(x), int(y), int(w), int(h))
