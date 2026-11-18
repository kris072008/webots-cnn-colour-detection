"""Pinhole projection of known 3-D objects into the e-puck's camera image.

Used to label dataset images automatically: the supervisor knows where every
object is, so its bounding box in the 52x39 image can be computed instead of
drawn by hand. Pure functions, unit-tested in tests/test_projection.py.

Frames: world x/y on the floor, z up. The robot's camera sits CAMERA_OFFSET
in front of and above the robot origin and looks along the robot's +x axis
(e-puck proto, Webots R2025a). In the image, u grows to the right and v
downwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CAMERA_OFFSET = (0.03, 0.0, 0.028)  # m, from E-puck.proto: Camera translation
FIELD_OF_VIEW = 0.84  # rad, horizontal
NEAR = 0.0055  # m, camera near plane


@dataclass(frozen=True)
class Box3D:
    """An object approximated by an oriented box resting on the floor."""

    x: float
    y: float
    yaw: float  # rad, about z
    size_x: float
    size_y: float
    height: float
    round: bool = False  # ball or cylinder: looks the same from every side (size_x = diameter)

    def corners(self) -> list[tuple[float, float, float]]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        out = []
        for dx in (-self.size_x / 2, self.size_x / 2):
            for dy in (-self.size_y / 2, self.size_y / 2):
                for z in (0.0, self.height):
                    out.append((self.x + c * dx - s * dy, self.y + s * dx + c * dy, z))
        return out


def focal_length_px(width: int, fov: float = FIELD_OF_VIEW) -> float:
    return (width / 2.0) / math.tan(fov / 2.0)


def project_point(point: tuple[float, float, float], robot_xy: tuple[float, float],
                  robot_yaw: float, width: int, height: int) -> tuple[float, float, float] | None:
    """(u, v, depth) of a world point, or None if it is behind the near plane."""
    px, py, pz = point
    rx, ry = robot_xy
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    forward = c * (px - rx) + s * (py - ry) - CAMERA_OFFSET[0]
    left = -s * (px - rx) + c * (py - ry) - CAMERA_OFFSET[1]
    up = pz - CAMERA_OFFSET[2]
    if forward <= NEAR:
        return None
    f = focal_length_px(width)
    return width / 2.0 - f * left / forward, height / 2.0 - f * up / forward, forward


@dataclass(frozen=True)
class Projection:
    bbox: tuple[int, int, int, int]  # x, y, w, h clipped to the image
    visible_fraction: float  # clipped box area / unclipped box area
    depth: float  # distance to the nearest point of the object, m


def _round_extent(box: Box3D, robot_xy: tuple[float, float], robot_yaw: float,
                  width: int, height: int) -> tuple[float, float, float, float, float] | None:
    """(u0, u1, v0, v1, near depth) of a ball or upright cylinder.

    Its outline does not depend on its own rotation: project the centre, then
    use the radius at the near surface for the half-width, and the bottom and
    top at that same depth for the vertical extent."""
    radius = box.size_x / 2.0
    centre = project_point((box.x, box.y, box.height / 2.0), robot_xy, robot_yaw, width, height)
    if centre is None:
        return None
    u_c, _, depth = centre
    near = depth - radius
    if near <= NEAR:
        return None
    f = focal_length_px(width)
    half = f * radius / near
    v_top = height / 2.0 - f * (box.height - CAMERA_OFFSET[2]) / near
    v_bottom = height / 2.0 - f * (0.0 - CAMERA_OFFSET[2]) / near
    return u_c - half, u_c + half, v_top, v_bottom, near


def project_box(box: Box3D, robot_xy: tuple[float, float], robot_yaw: float,
                width: int, height: int) -> Projection | None:
    """Image bounding box of an object, or None if it is not in view."""
    if box.round:
        extent = _round_extent(box, robot_xy, robot_yaw, width, height)
        if extent is None:
            return None
        u0, u1, v0, v1, nearest = extent
    else:
        pts = [project_point(p, robot_xy, robot_yaw, width, height) for p in box.corners()]
        if any(p is None for p in pts):
            return None  # partly behind the camera: skip rather than guess
        us, vs, ds = zip(*pts, strict=True)
        u0, u1, v0, v1, nearest = min(us), max(us), min(vs), max(vs), min(ds)
    full = (u1 - u0) * (v1 - v0)
    cu0, cu1 = max(u0, 0.0), min(u1, float(width))
    cv0, cv1 = max(v0, 0.0), min(v1, float(height))
    if cu1 <= cu0 or cv1 <= cv0 or full <= 0:
        return None
    clipped = (cu1 - cu0) * (cv1 - cv0)
    bbox = (int(math.floor(cu0)), int(math.floor(cv0)),
            max(1, int(math.ceil(cu1) - math.floor(cu0))), max(1, int(math.ceil(cv1) - math.floor(cv0))))
    return Projection(bbox, clipped / full, nearest)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    inter = overlap_fraction(a, b) * a[2] * a[3]
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def overlap_fraction(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Share of box a's area that box b covers."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    return (iw * ih) / float(aw * ah) if aw * ah else 0.0
