"""Tests for the object tour's lidar grouping and CNN vote (object_finder/lidar_objects.py)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "controllers" / "object_finder"))
from lidar_objects import JOIN_M, MAX_VIEW_M, MIN_VIEW_M, find_objects, to_world, vote  # noqa: E402


def arc(cx, cy, width, n=20, z=0.3):
    """Points along a short wall of the given width centred at (cx, cy)."""
    xs = np.linspace(cx - width / 2, cx + width / 2, n)
    return np.column_stack([xs, np.full(n, cy), np.full(n, z)])


def test_two_separate_objects_are_found_with_their_widths():
    pts = np.vstack([arc(3.0, 0.0, 1.0), arc(-4.0, 2.0, 2.0)])
    found = find_objects(pts, max_range=14.0, origin=(0.0, 0.0))
    assert len(found) == 2
    widths = sorted(round(f.width, 1) for f in found)
    assert widths == [1.0, 2.0]


def test_floor_hits_and_no_return_beams_are_ignored():
    floor = np.column_stack([np.linspace(0.5, 3, 30), np.zeros(30), np.full(30, 0.01)])
    no_return = np.array([[np.inf, np.inf, np.inf], [13.9, 0.0, 0.3]])  # inf, and at max range
    found = find_objects(np.vstack([floor, no_return, arc(2.0, 2.0, 0.5)]), 14.0, (0.0, 0.0))
    assert len(found) == 1 and found[0].x == pytest.approx(2.0, abs=0.05)


def test_points_within_join_distance_are_one_object():
    gap = JOIN_M * 0.9
    pts = np.vstack([arc(5.0, 0.0, 1.0), arc(5.0 + 1.0 + gap, 0.0, 1.0)])
    assert len(find_objects(pts, 14.0, (0.0, 0.0))) == 1


def test_a_few_stray_points_are_noise():
    assert find_objects(arc(3.0, 3.0, 0.2, n=3), 14.0, (0.0, 0.0)) == []


def test_view_distance_is_clamped():
    small = find_objects(arc(2.0, 0.0, 0.05), 14.0, (0.0, 0.0))[0]
    huge = find_objects(arc(8.0, 0.0, 12.0, n=200), 14.0, (0.0, 0.0))[0]
    assert small.view_distance() == MIN_VIEW_M and huge.view_distance() == MAX_VIEW_M


def test_objects_are_ordered_by_bearing():
    pts = np.vstack([arc(0.0, -3.0, 0.5), arc(3.0, 0.0, 0.5), arc(0.0, 3.0, 0.5)])
    bearings = [math.atan2(f.y, f.x) % (2 * math.pi) for f in find_objects(pts, 14.0, (0.0, 0.0))]
    assert bearings == sorted(bearings)


def test_to_world_rotates_and_shifts():
    out = to_world(np.array([[1.0, 0.0, -0.1]]), 2.0, 3.0, 0.1, math.pi / 2)
    assert out[0] == pytest.approx([2.0, 4.0, 0.0])


@pytest.mark.parametrize("answers, expected", [
    ([("truck", 1.0), ("truck", 0.62), ("truck", 0.98)], ("truck", True)),
    ([("horse", 0.98), ("deer", 0.95), ("horse", 1.0)], ("horse", True)),      # 2 of 3 agree, mean 0.66
    ([("automobile", 0.68), ("automobile", 0.86), ("airplane", 0.97)], ("automobile", False)),  # mean 0.51
    ([("frog", 0.41), ("ship", 0.54), ("bird", 0.30)], ("ship", False)),        # no two views agree
])
def test_vote_says_unsure_when_views_disagree_or_are_weak(answers, expected):
    label, _, sure = vote(answers)
    assert (label, sure) == expected


def test_a_single_answered_view_is_never_sure():
    assert vote([("truck", 0.99)])[2] is False


def test_vote_needs_answers():
    with pytest.raises(ValueError):
        vote([])
