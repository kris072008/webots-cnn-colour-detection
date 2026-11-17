"""Find objects in a 360-degree lidar scan, and decide how to name them.

Pure NumPy, so it can be unit-tested without Webots (tests/test_lidar_objects.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOOR_Z = 0.04  # m: world heights below this are the floor (or the flat water patch)
JOIN_M = 1.5  # m: points closer than this belong to the same object (keeps a truck in one piece)
MIN_POINTS = 4  # fewer points than this is noise
VIEW_FACTOR = 2.0  # viewing distance = this x the object's visible width
MIN_VIEW_M, MAX_VIEW_M = 0.55, 12.0


@dataclass(frozen=True)
class Found:
    """One object found by the lidar, in world coordinates."""

    x: float  # centre of the points the lidar hit (the near face of the object)
    y: float
    width: float  # largest extent of those points in the ground plane, m
    points: int

    def view_distance(self) -> float:
        """How far from the object the camera should stand so it fills the square crop."""
        return float(np.clip(VIEW_FACTOR * self.width, MIN_VIEW_M, MAX_VIEW_M))


def to_world(points_xyz: np.ndarray, x: float, y: float, z: float, yaw: float) -> np.ndarray:
    """Lidar points (sensor frame, x forward, y left, z up) -> world frame, for a level robot."""
    c, s = np.cos(yaw), np.sin(yaw)
    px, py, pz = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    return np.column_stack([x + c * px - s * py, y + s * px + c * py, z + pz])


def find_objects(world_xyz: np.ndarray, max_range: float, origin: tuple[float, float]) -> list[Found]:
    """Group the lidar's non-floor hits into objects.

    Points that are not finite, are floor hits, or lie at the maximum range
    (no return) are dropped. The rest are joined into groups: two points are in
    the same group if a chain of points less than JOIN_M apart connects them.
    """
    pts = world_xyz[np.isfinite(world_xyz).all(axis=1)]
    pts = pts[pts[:, 2] > FLOOR_Z]
    dist = np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1])
    pts = pts[dist < 0.98 * max_range]
    if len(pts) == 0:
        return []
    xy = pts[:, :2]
    # Union-find over all pairs closer than JOIN_M (a few thousand points at most).
    parent = np.arange(len(xy))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(xy)):
        near = np.nonzero(np.hypot(*(xy[i + 1:] - xy[i]).T) < JOIN_M)[0] + i + 1
        for j in near:
            ri, rj = root(i), root(int(j))
            if ri != rj:
                parent[rj] = ri
    groups: dict[int, list[int]] = {}
    for i in range(len(xy)):
        groups.setdefault(root(i), []).append(i)
    found = []
    for idx in groups.values():
        if len(idx) < MIN_POINTS:
            continue
        g = xy[idx]
        lo, hi = g.min(axis=0), g.max(axis=0)
        found.append(Found(float(g[:, 0].mean()), float(g[:, 1].mean()), float(np.hypot(*(hi - lo))), len(idx)))
    # Visit order: by bearing from the scan point, anticlockwise from +x.
    found.sort(key=lambda f: np.arctan2(f.y - origin[1], f.x - origin[0]) % (2 * np.pi))
    return found


def vote(answers: list[tuple[str, float]], min_views: int = 2, min_mean: float = 0.6) -> tuple[str, float, bool]:
    """Combine the CNN's top-1 answers from several views.

    Each view adds its confidence to its class; the class with the largest total
    wins. The answer is only trusted ("sure") if it won at least ``min_views``
    views and its mean confidence over all views is at least ``min_mean``.
    A single answered view is therefore never "sure".
    Returns (class, mean confidence over all views, sure).
    """
    if not answers:
        raise ValueError("no answers to vote on")
    totals: dict[str, float] = {}
    wins: dict[str, int] = {}
    for label, conf in answers:
        totals[label] = totals.get(label, 0.0) + conf
        wins[label] = wins.get(label, 0) + 1
    best = max(totals, key=totals.get)
    mean = totals[best] / len(answers)
    return best, mean, wins[best] >= min_views and mean >= min_mean
