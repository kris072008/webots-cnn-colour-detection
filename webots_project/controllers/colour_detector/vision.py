"""Colour-block detection for the e-puck camera, using OpenCV.

Everything in this module works on plain NumPy arrays, so it can be unit-tested
without Webots running (see tests/test_vision.py).

Pipeline (one call to ``detect_blocks``):
    BGRA camera buffer -> BGR -> light blur -> HSV
    -> one binary mask per colour (cv.inRange)
    -> morphological opening to remove speckle noise
    -> external contours -> keep the largest blob above MIN_AREA_FRACTION
    -> centroid, bounding box and area fraction for steering.

Why HSV instead of raw RGB thresholds?
    The original controller tested a single centre pixel with R > 100,
    G < 80, B < 80. RGB values change a lot with lighting and shadows, so the
    same red block reads differently near and far. HSV separates the colour
    (hue) from brightness (value), so one hue range works under all the
    lighting in the arena. The lecture results showed the plain RGB filter
    "blacking out" the red box completely; the ranges below fix that.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

# ---------------------------------------------------------------------------
# HSV ranges.  OpenCV uses H: 0-179 (degrees / 2), S: 0-255, V: 0-255.
# ---------------------------------------------------------------------------
# RED wraps around hue 0, so it needs two ranges (0-10 and 170-179).
# GREEN starts at H=45 so the yellow-green grass used in the course world
#   (RGB ~ 197,224,80 -> H ~ 36) is NOT detected as a green block.
# BLUE starts at H=105 so cyan "water" (H ~ 97) is excluded, and needs S>=150
#   so the grey-blue sky (S ~ 120) is excluded.
# S_MIN rejects grey walls/floor and white highlights; V_MIN=40 keeps the
#   shaded side of a block (dark but still saturated) and rejects black.
HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 120, 40), (10, 255, 255)), ((170, 120, 40), (179, 255, 255))],
    "green": [((45, 120, 40), (85, 255, 255))],
    "blue": [((105, 150, 40), (130, 255, 255))],
}

# A blob must cover at least this fraction of the image to count.
# 0.4 % of a 52x39 image is ~8 pixels: big enough to ignore noise,
# small enough to see a 0.3 m block from across a 3 m arena.
MIN_AREA_FRACTION = 0.004

_MORPH_KERNEL = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class Detection:
    """One coloured blob found in a camera frame."""

    colour: str
    area_fraction: float  # blob area / image area, 0..1 (grows as robot gets closer)
    centroid_x: float  # pixels from the left edge
    centroid_y: float  # pixels from the top edge
    bbox: tuple[int, int, int, int]  # x, y, width, height

    def horizontal_offset(self, image_width: int) -> float:
        """Blob position from -1 (far left) to +1 (far right); 0 = centred."""
        half = image_width / 2.0
        return (self.centroid_x - half) / half


def webots_to_bgr(raw: bytes | None, width: int, height: int) -> np.ndarray | None:
    """Convert the BGRA byte buffer from ``Camera.getImage()`` to a BGR array.

    Returns None (instead of raising) when the camera has no frame yet, which
    happens on the first simulation step after ``camera.enable()``.
    """
    if not raw:
        return None
    expected = width * height * 4
    if len(raw) != expected:
        raise ValueError(
            f"camera buffer has {len(raw)} bytes, expected {expected} "
            f"for {width}x{height} BGRA"
        )
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    return cv.cvtColor(bgra, cv.COLOR_BGRA2BGR)


def colour_mask(hsv: np.ndarray, colour: str) -> np.ndarray:
    """Binary mask (0/255) of pixels inside ``colour``'s HSV range(s)."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES[colour]:
        mask |= cv.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    # Opening = erode then dilate: removes isolated noisy pixels at block edges.
    return cv.morphologyEx(mask, cv.MORPH_OPEN, _MORPH_KERNEL)


def detect_blocks(
    bgr: np.ndarray,
    colours: tuple[str, ...] = tuple(HSV_RANGES),
    min_area_fraction: float = MIN_AREA_FRACTION,
) -> list[Detection]:
    """Find the largest blob of each colour. Returns biggest-first.

    An empty list means no block is visible; callers must handle that case
    (the controller keeps searching).
    """
    height, width = bgr.shape[:2]
    image_area = float(width * height)
    hsv = cv.cvtColor(cv.GaussianBlur(bgr, (3, 3), 0), cv.COLOR_BGR2HSV)

    detections: list[Detection] = []
    for colour in colours:
        mask = colour_mask(hsv, colour)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv.contourArea)
        # Count mask pixels inside the bounding box rather than contourArea:
        # contourArea under-counts tiny blobs (a 3x3 square has area 4).
        x, y, w, h = cv.boundingRect(largest)
        area = float(cv.countNonZero(mask[y : y + h, x : x + w]))
        if area / image_area < min_area_fraction:
            continue
        moments = cv.moments(mask[y : y + h, x : x + w], binaryImage=True)
        if moments["m00"] == 0:
            continue
        detections.append(
            Detection(
                colour=colour,
                area_fraction=area / image_area,
                centroid_x=x + moments["m10"] / moments["m00"],
                centroid_y=y + moments["m01"] / moments["m00"],
                bbox=(x, y, w, h),
            )
        )
    detections.sort(key=lambda d: d.area_fraction, reverse=True)
    return detections


def square_crop(bgr: np.ndarray, bbox: tuple[int, int, int, int] | None = None,
                margin: float = 0.25) -> np.ndarray:
    """Square crop for the CNN (CIFAR-10 images are square, 32x32).

    With a bbox, the crop is centred on the object and padded by ``margin``
    so the CNN sees the whole object, like a CIFAR photo. Without one, the
    centre square of the frame is used. Resizing a 52x39 frame straight to
    32x32 would squash the object, so we crop first.
    """
    height, width = bgr.shape[:2]
    if bbox is None:
        cx, cy, side = width / 2, height / 2, min(width, height)
    else:
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * (1 + 2 * margin)
    side = int(min(max(side, 8), width, height))
    x0 = int(round(min(max(cx - side / 2, 0), width - side)))
    y0 = int(round(min(max(cy - side / 2, 0), height - side)))
    return bgr[y0 : y0 + side, x0 : x0 + side]


def annotate(bgr: np.ndarray, detections: list[Detection], label: str = "") -> np.ndarray:
    """Copy of the frame with boxes and labels drawn, upscaled for readability."""
    scale = max(1, 320 // bgr.shape[1])
    out = cv.resize(bgr, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
    box_colours = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 0, 0)}
    for det in detections:
        x, y, w, h = (v * scale for v in det.bbox)
        colour = box_colours.get(det.colour, (255, 255, 255))
        cv.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        cv.putText(out, f"{det.colour} {det.area_fraction:.0%}", (x, max(12, y - 4)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv.LINE_AA)
    if label:
        cv.putText(out, label, (4, out.shape[0] - 6), cv.FONT_HERSHEY_SIMPLEX,
                   0.45, (255, 255, 255), 1, cv.LINE_AA)
    return out
