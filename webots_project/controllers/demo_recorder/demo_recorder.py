"""Supervisor for demo videos and for scoring the robot's inspections.

Replaces block_randomiser in a copy of colour_blocks.wbt or
colour_blocks_distractors.wbt. It places the blocks (and any distractors)
exactly as block_randomiser does for the same seed, then:

* optionally records a video of the run with captions taken from the robot's
  own log (each "found" event and both CNN labels), and
* writes images/ground_truth.csv: for every "block_found" event in the
  robot's encounters.csv, which object the robot was actually looking at
  (the nearest object in front of it), so the robot's labels can be scored.

controllerArgs: --seed=1 --seconds=60 [--out=PATH.mp4 --width=1280 --height=720]
Leave --out empty to only score, without recording.
"""

from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path

from controller import Supervisor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "block_randomiser"))
sys.path.insert(0, str(HERE.parent / "shape_dataset_collector"))
from block_randomiser import BLOCK_DEFS, DISTRACTOR_DEFS, place_objects  # noqa: E402
from projection import Box3D, project_box  # noqa: E402

CAMERA_W, CAMERA_H = 52, 39  # e-puck camera resolution in colour_blocks*.wbt

IMAGES = HERE.parent / "colour_detector" / "images"
ENCOUNTERS = IMAGES / "encounters.csv"
GROUND_TRUTH = IMAGES / "ground_truth.csv"
WHITE = 0xFFFFFF
KIND = {"RED_BLOCK": "block", "GREEN_BLOCK": "block", "BLUE_BLOCK": "block",
        "DISTRACTOR_1": "ball", "DISTRACTOR_2": "cylinder", "DISTRACTOR_3": "ball"}
COLOUR = {"RED_BLOCK": "red", "GREEN_BLOCK": "green", "BLUE_BLOCK": "blue",
          "DISTRACTOR_1": "red", "DISTRACTOR_2": "green", "DISTRACTOR_3": "blue"}


def look_at(position, target):
    """Axis-angle orientation for a Webots Viewpoint (x forward, z up) at
    ``position`` looking at ``target``."""
    f = [t - p for t, p in zip(target, position)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    left = [-f[1], f[0], 0.0]  # z_world x f, then normalise
    n = math.sqrt(sum(c * c for c in left))
    left = [c / n for c in left]
    up = [f[1] * left[2] - f[2] * left[1], f[2] * left[0] - f[0] * left[2], f[0] * left[1] - f[1] * left[0]]
    m = [[f[0], left[0], up[0]], [f[1], left[1], up[1]], [f[2], left[2], up[2]]]  # columns: x, y, z
    angle = math.acos(max(-1.0, min(1.0, (m[0][0] + m[1][1] + m[2][2] - 1) / 2)))
    axis = [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]]
    n = math.sqrt(sum(c * c for c in axis)) or 1.0
    return [c / n for c in axis] + [angle]


def new_rows(path: Path, start: int) -> list[list[str]]:
    """Complete encounters.csv rows written since byte offset ``start``.

    Reads bytes and stops at the last newline, so a row the robot is still
    writing is never parsed half-finished."""
    if not path.exists():
        return []
    with path.open("rb") as fh:
        fh.seek(start)
        data = fh.read()
    data = data[: data.rfind(b"\n") + 1]
    rows = []
    for row in csv.reader(data.decode("utf-8").splitlines()):
        if len(row) >= 8:
            rows.append(row)
        elif row:
            print(f"[recorder] WARNING: skipped malformed log row {row!r}")
    return rows


def caption(row: list[str], truth: dict[str, str]) -> str | None:
    t, event, colour, image, label, conf = row[1], row[2], row[3], row[5], row[6], row[7]
    name = image.replace(".png", "")
    if name.startswith("startup"):
        return None
    if event == "block_found":
        actual = truth.get(name)
        return f"{t} s  HSV: {colour.upper()} object found" + (f"  (really a {actual})" if actual else "")
    if event == "cnn_prediction":
        return f"{t} s   CIFAR-10 CNN: {label} {float(conf):.0%}"
    if event == "shape_prediction":
        return f"{t} s   shape CNN: {label} {float(conf):.0%}"
    if event == "rejected_not_block":
        return f"{t} s   -> NOT a block: ignored, keep searching"
    if event == "block_confirmed":
        return f"{t} s   -> block confirmed"
    return None


def object_box(sup: Supervisor, name: str) -> Box3D:
    """The object's 3-D shape and pose, read from the world."""
    node = sup.getFromDef(name)
    x, y, _ = node.getField("translation").getSFVec3f()
    _, _, az, angle = node.getField("rotation").getSFRotation()
    geom = node.getField("children").getMFNode(0).getField("geometry").getSFNode()
    kind = geom.getTypeName()
    if kind == "Box":
        sx, sy, h = geom.getField("size").getSFVec3f()
        return Box3D(x, y, angle if az >= 0 else -angle, sx, sy, h)
    r = geom.getField("radius").getSFFloat()
    h = 2 * r if kind == "Sphere" else geom.getField("height").getSFFloat()
    return Box3D(x, y, 0.0, 2 * r, 2 * r, h, round=True)


def facing_object(sup: Supervisor, robot, placed: dict[str, tuple[float, float]],
                  colour: str) -> tuple[str, float]:
    """Which object of the inspected colour the robot is looking at.

    Only the objects of that colour are candidates (each colour has a block and
    at most one distractor). Among those in view, prefer the one whose
    projected outline covers the image's centre column (the robot steers its
    target there), otherwise the nearest (an IR-bump arrival can leave the
    object off-centre). Returns ("", inf) if none is in view."""
    rx, ry, _ = robot.getField("translation").getSFVec3f()
    _, _, az, angle = robot.getField("rotation").getSFRotation()
    yaw = angle if az >= 0 else -angle  # the e-puck only rotates about z
    in_view = []
    for name in placed:
        if COLOUR.get(name) != colour:
            continue
        p = project_box(object_box(sup, name), (rx, ry), yaw, CAMERA_W, CAMERA_H)
        if p is not None:
            bx, _, bw, _ = p.bbox
            in_view.append((not (bx <= CAMERA_W / 2 <= bx + bw), p.depth, name))
    if not in_view:
        return "", float("inf")
    _, depth, name = min(in_view)
    return name, depth


def main() -> None:
    args = {"seed": "1", "seconds": "60", "out": "", "width": "1280", "height": "720"}
    for arg in sys.argv[1:]:
        if arg.strip():
            key, _, value = arg.lstrip("-").partition("=")
            if key not in args:
                sys.exit(f"[recorder] unknown argument {arg!r}")
            args[key] = value
    sup = Supervisor()
    step = int(sup.getBasicTimeStep())
    robot = sup.getFromDef("EPUCK")
    if robot is None:
        sys.exit("[recorder] the world has no robot with DEF EPUCK")
    placed = place_objects(sup, random.Random(int(args["seed"])), int(args["seed"]))
    expected = set(BLOCK_DEFS) | {d for d in DISTRACTOR_DEFS if sup.getFromDef(d) is not None}
    if set(placed) != expected:
        print(f"[recorder] WARNING: not placed: {sorted(expected - set(placed))}")
    # Angled view of the whole arena (the world's own view is straight down).
    children = sup.getRoot().getField("children")
    for i in range(children.getCount()):
        node = children.getMFNode(i)
        if node.getTypeName() == "Viewpoint":
            position = [0.0, -4.3, 2.7]
            node.getField("position").setSFVec3f(position)
            node.getField("orientation").setSFRotation(look_at(position, [0.0, 0.35, -0.2]))
    sup.step(step)

    recording = bool(args["out"])
    out = Path(args["out"]).resolve() if recording else None
    if recording:
        out.parent.mkdir(parents=True, exist_ok=True)
        sup.movieStartRecording(str(out), int(args["width"]), int(args["height"]), 0, 90, 1, False)
        sup.setLabel(0, f"e-puck block search  |  seed {args['seed']}",
                     0.01, 0.01, 0.09, WHITE, 0.0, "Arial")
    log_start = ENCOUNTERS.stat().st_size if ENCOUNTERS.exists() else 0
    seen_rows = 0
    truth: dict[str, str] = {}
    truth_rows: list[list] = []
    captions: list[str] = []
    seconds = float(args["seconds"])
    finished = False
    try:
        while sup.getTime() < seconds:
            if sup.step(step) == -1:
                return
            rows = new_rows(ENCOUNTERS, log_start)
            for row in rows[seen_rows:]:
                if row[2] == "block_found":
                    name, dist = facing_object(sup, robot, placed, row[3])
                    tag = row[5].replace(".png", "")
                    kind = KIND.get(name, "nothing")
                    if kind == "nothing":
                        print(f"[recorder] WARNING: {tag}: no object inside the camera's view")
                    truth[tag] = f"{row[3]} {kind}" if kind != "block" else "block"
                    truth_rows.append([row[1], tag, row[3], name, kind,
                                       f"{dist:.2f}" if math.isfinite(dist) else ""])
                text = caption(row, truth)
                if text:
                    captions.append(text)
            seen_rows = len(rows)
            if recording:
                sup.setLabel(1, f"time {sup.getTime():5.1f} s (real-time)", 0.01, 0.07,
                             0.08, WHITE, 0.0, "Arial")
                for i, text in enumerate(captions[-6:]):
                    sup.setLabel(2 + i, text, 0.5, 0.012 + 0.031 * i, 0.066, WHITE, 0.0, "Arial")
        finished = True
    finally:
        # Always keep what was measured, even if Webots stopped the run early.
        with GROUND_TRUTH.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sim_time_s", "tag", "hsv_colour", "object_def", "true_object", "distance_m"])
            writer.writerows(truth_rows)
        unmatched = sum(1 for r in truth_rows if r[4] == "nothing")
        print(f"[recorder] ground truth for {len(truth_rows)} inspections ({unmatched} unmatched)"
              f"{'' if finished else ' - RUN STOPPED EARLY'} -> {GROUND_TRUTH}")
        if recording:
            sup.movieStopRecording()
    if recording:
        while not sup.movieIsReady():
            if sup.step(step) == -1:
                return
        print(f"[recorder] {'FAILED to save the video' if sup.movieFailed() else 'saved ' + str(out)}")
    sup.simulationQuit(0)
    sup.step(step)


if __name__ == "__main__":
    main()
