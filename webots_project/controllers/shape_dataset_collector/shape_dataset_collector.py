"""Supervisor e-puck that builds a labelled shape dataset from Webots itself.

Used with worlds/shape_dataset.wbt. For each random layout it places three
blocks, three balls and three cylinders in random colours (red, green or
blue), sizes and positions, and randomises the light level (domain
randomisation). It then teleports the robot to many viewpoints, mostly facing
an object from 0.2-1.6 m, takes a camera frame and labels every visible
object automatically by projecting its known 3-D shape into the image
(projection.py). Because every object comes in every colour, colour says
nothing about shape: the classes can only be told apart by shape.

Crops (saved as <out>/<split>/<class>/<name>.png at 32x32):
  --crop=box (default)  the projected box, jittered like an imperfect detector,
                        then vision.square_crop exactly as the robot crops.
                        "background" crops go through the same square_crop
                        from a random box that touches no object.
  --crop=hsv            the robot's own HSV detector finds the object's blob
                        (it must overlap the projected box, IoU >= 0.3) and that
                        blob is cropped with no jitter, exactly as on the robot.
                        Objects only; used as a deployment-style test set.

controllerArgs (all optional):
    --split=NAME         output sub-folder (train, test, test_hsv, ...)
    --seed-base=N        first layout seed (train 0, test 100000)
    --layouts=N          random layouts (default 80)
    --poses=K            camera views per layout (default 40)
    --crop=box|hsv       crop procedure, see above
    --out=PATH           dataset root (default ../../../cnn/webots_shapes)
    --debug=N            also save the first N full frames with boxes drawn

Writes <split>/labels.csv row by row and prints how many views and objects
were skipped and why, then quits the simulation.
"""

from __future__ import annotations

import csv
import math
import random
import sys
from collections import Counter
from pathlib import Path

import cv2 as cv
from controller import Supervisor

from projection import Box3D, iou, overlap_fraction, project_box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "colour_detector"))
import vision  # noqa: E402  (same frame conversion and cropping as the robot)

CLASSES = ("block", "ball", "cylinder", "background")
COLOURS = {"red": (0.9, 0.05, 0.05), "green": (0.05, 0.8, 0.05), "blue": (0.05, 0.1, 0.9)}
ARENA_LIMIT = 1.5 - 0.25  # keep objects off the walls
ROBOT_LIMIT = 1.5 - 0.08
MIN_GAP = 0.5  # centre-to-centre distance between objects
MIN_VISIBLE = 0.7  # skip objects cut off by the image edge
MAX_OCCLUSION = 0.3  # skip objects more than this much covered by a nearer one
MIN_BOX_PX = 4
MIN_HSV_IOU = 0.3
HEADER = ["file", "class", "colour", "depth_m", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
          "visible_fraction", "layout", "luminosity", "hsv_iou"]


def parse_args(argv: list[str]) -> dict[str, str]:
    values = {"split": "train", "seed-base": "", "layouts": "80", "poses": "40", "crop": "box",
              "debug": "0", "out": str(HERE.parents[2] / "cnn" / "webots_shapes")}
    for arg in argv:
        if not arg.strip():
            continue
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"bad argument {arg!r}, expected --key=value")
        key, value = arg[2:].split("=", 1)
        if key not in values:
            raise ValueError(f"unknown argument --{key}")
        values[key] = value
    if values["crop"] not in ("box", "hsv"):
        raise ValueError("--crop must be box or hsv")
    if not values["seed-base"]:
        values["seed-base"] = "0" if values["split"] == "train" else "100000"
    return values


class Collector:
    def __init__(self, args: dict[str, str]) -> None:
        self.sup = Supervisor()
        self.step_ms = int(self.sup.getBasicTimeStep())
        self.camera = self.sup.getDevice("camera")
        self.camera.enable(self.step_ms)
        self.width, self.height = self.camera.getWidth(), self.camera.getHeight()
        self.me = self.sup.getSelf()
        self.split, self.crop_mode = args["split"], args["crop"]
        self.seed_base = int(args["seed-base"])
        self.layouts, self.poses = int(args["layouts"]), int(args["poses"])
        self.debug_left = int(args["debug"])
        self.out = Path(args["out"]) / self.split
        for cls in CLASSES:
            (self.out / cls).mkdir(parents=True, exist_ok=True)
        self.objects = []  # (kind, node, geometry node, colour field)
        for kind, prefix in (("block", "BLOCK"), ("ball", "BALL"), ("cylinder", "CYLINDER")):
            for i in range(1, 4):
                node = self.sup.getFromDef(f"{prefix}_{i}")
                if node is None:
                    raise RuntimeError(f"world has no node DEF {prefix}_{i}")
                shape = node.getField("children").getMFNode(0)
                colour = shape.getField("appearance").getSFNode().getField("baseColor")
                self.objects.append((kind, node, shape.getField("geometry").getSFNode(), colour))
        self.sky = self.sup.getFromDef("SKY")
        self.sky_light = self.sup.getFromDef("SKY_LIGHT")
        if self.sky is None or self.sky_light is None:
            print("[dataset] WARNING: no DEF SKY / SKY_LIGHT - light level will not be randomised")
        self.counts: Counter[str] = Counter()
        self.skipped: Counter[str] = Counter()
        self.luminosity = 1.0
        # Labels are written row by row, so an early stop still leaves a usable CSV.
        self._csv_file = (self.out / "labels.csv").open("w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(HEADER)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.sup.step(self.step_ms) == -1:
                raise SystemExit(0)

    # -------------------------------------------------------------- layout
    def place_layout(self, rng: random.Random) -> list[tuple[str, str, Box3D]]:
        placed: list[tuple[str, str, Box3D]] = []
        for kind, node, geom, colour_field in self.objects:
            colour = rng.choice(list(COLOURS))
            colour_field.setSFColor([min(1.0, max(0.0, c + rng.uniform(-0.05, 0.05)))
                                     for c in COLOURS[colour]])
            if kind == "block":
                sx = sy = 0.3  # the brief's block size
                h = 0.1
            elif kind == "ball":
                r = rng.uniform(0.06, 0.11)
                geom.getField("radius").setSFFloat(r)
                sx = sy = h = 2 * r
            else:
                r, h = rng.uniform(0.05, 0.09), rng.uniform(0.12, 0.25)
                geom.getField("radius").setSFFloat(r)
                geom.getField("height").setSFFloat(h)
                sx = sy = 2 * r
            for _ in range(1000):
                x, y = rng.uniform(-ARENA_LIMIT, ARENA_LIMIT), rng.uniform(-ARENA_LIMIT, ARENA_LIMIT)
                if all(math.dist((x, y), (b.x, b.y)) >= MIN_GAP for _, _, b in placed):
                    break
            else:
                raise RuntimeError("could not place all objects; reduce MIN_GAP")
            yaw = rng.uniform(0, math.pi)
            node.getField("translation").setSFVec3f([x, y, h / 2])
            node.getField("rotation").setSFRotation([0, 0, 1, yaw])
            placed.append((kind, colour, Box3D(x, y, yaw, sx, sy, h, round=kind != "block")))
        self.luminosity = rng.uniform(0.6, 1.4)
        for node in (self.sky, self.sky_light):
            if node is not None:
                node.getField("luminosity").setSFFloat(self.luminosity)
        return placed

    def random_pose(self, rng: random.Random, placed) -> tuple[float, float, float]:
        for _ in range(200):
            if rng.random() < 0.75:  # look at a chosen object from some distance
                _, _, target = rng.choice(placed)
                dist, bearing = rng.uniform(0.2, 1.6), rng.uniform(-math.pi, math.pi)
                x, y = target.x + dist * math.cos(bearing), target.y + dist * math.sin(bearing)
                yaw = math.atan2(target.y - y, target.x - x) + rng.uniform(-0.35, 0.35)
            else:
                x, y = rng.uniform(-ROBOT_LIMIT, ROBOT_LIMIT), rng.uniform(-ROBOT_LIMIT, ROBOT_LIMIT)
                yaw = rng.uniform(-math.pi, math.pi)
            inside = abs(x) < ROBOT_LIMIT and abs(y) < ROBOT_LIMIT
            clear = all(math.dist((x, y), (b.x, b.y)) > max(b.size_x, b.size_y) / 2 * 1.5 + 0.05
                        for _, _, b in placed)
            if inside and clear:
                return x, y, yaw
        raise RuntimeError("no free robot pose found")

    # -------------------------------------------------------------- capture
    def save(self, crop, cls: str, name: str, row: list) -> None:
        small = cv.resize(crop, (32, 32), interpolation=cv.INTER_AREA)
        path = self.out / cls / f"{name}.png"
        if not cv.imwrite(str(path), small):
            raise OSError(f"could not write {path}")
        self._csv.writerow([f"{self.split}/{cls}/{name}.png", cls, *row])
        self._csv_file.flush()
        self.counts[cls] += 1

    def save_debug(self, frame, views, layout_id: int, pose_id: int) -> None:
        big = cv.resize(frame, None, fx=8, fy=8, interpolation=cv.INTER_NEAREST)
        for kind, _, p in views:
            bx, by, bw, bh = p.bbox
            cv.rectangle(big, (bx * 8, by * 8), ((bx + bw) * 8, (by + bh) * 8), (255, 255, 255), 2)
            cv.putText(big, kind, (bx * 8 + 3, by * 8 + 18), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        debug_dir = self.out.parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        path = debug_dir / f"{self.split}_{layout_id:03d}_{pose_id:03d}.png"
        if not cv.imwrite(str(path), big):
            print(f"[dataset] WARNING: could not write debug frame {path}")

    def hsv_match(self, frame, colour: str, bbox):
        """The HSV detector's blob of this colour and its IoU with the projected
        box, or (None, None) if the detector sees no blob of that colour."""
        dets = vision.detect_blocks(frame, (colour,))
        if not dets:
            return None, None
        return dets[0], iou(bbox, dets[0].bbox)

    def capture_view(self, rng: random.Random, layout_id: int, pose_id: int, placed) -> None:
        x, y, yaw = self.random_pose(rng, placed)
        self.me.getField("translation").setSFVec3f([x, y, 0.0])
        self.me.getField("rotation").setSFRotation([0, 0, 1, yaw])
        self.me.resetPhysics()
        self.step(2)  # let the camera render the new view
        frame = vision.webots_to_bgr(self.camera.getImage(), self.width, self.height)
        if frame is None:
            self.skipped["view: no camera frame"] += 1
            return
        views = []
        for kind, colour, box in placed:
            p = project_box(box, (x, y), yaw, self.width, self.height)
            if p is not None:
                views.append((kind, colour, p))
        views.sort(key=lambda v: v[2].depth)
        if self.debug_left > 0 and views:
            self.debug_left -= 1
            self.save_debug(frame, views, layout_id, pose_id)
        boxes_in_view = [v[2].bbox for v in views]
        for i, (kind, colour, p) in enumerate(views):
            bx, by, bw, bh = p.bbox
            if p.visible_fraction < MIN_VISIBLE:
                self.skipped["object: cut off by the image edge"] += 1
                continue
            if min(bw, bh) < MIN_BOX_PX:
                self.skipped["object: too small"] += 1
                continue
            if any(overlap_fraction(p.bbox, views[j][2].bbox) > MAX_OCCLUSION for j in range(i)):
                self.skipped["object: hidden behind a nearer object"] += 1
                continue
            det, hsv_iou = self.hsv_match(frame, colour, p.bbox)
            if self.crop_mode == "hsv":
                if det is None or hsv_iou < MIN_HSV_IOU:
                    self.skipped["object: HSV blob missing or not this object"] += 1
                    continue
                crop = vision.square_crop(frame, det.bbox)  # exactly what the robot crops
            else:
                crop = vision.square_crop(frame, self.jitter(rng, p.bbox))
            self.save(crop, kind, f"{self.split}_{layout_id:03d}_{pose_id:03d}_{i}",
                      [colour, round(p.depth, 3), *p.bbox, round(p.visible_fraction, 2), layout_id,
                       round(self.luminosity, 2), "" if hsv_iou is None else round(hsv_iou, 2)])
        if self.crop_mode == "box" and (not views or rng.random() < 0.35):
            self.save_background(rng, frame, boxes_in_view, layout_id, pose_id)

    @staticmethod
    def jitter(rng: random.Random, bbox):
        """Shift and scale a box like an imperfect detector would."""
        bx, by, bw, bh = bbox
        s = rng.uniform(0.9, 1.15)
        jw, jh = max(2, int(round(bw * s))), max(2, int(round(bh * s)))
        jx = int(round(bx + bw / 2 - jw / 2 + rng.uniform(-0.15, 0.15) * bw))
        jy = int(round(by + bh / 2 - jh / 2 + rng.uniform(-0.15, 0.15) * bh))
        return jx, jy, jw, jh

    def save_background(self, rng, frame, boxes_in_view, layout_id: int, pose_id: int) -> None:
        """A crop of empty arena, made by the same square_crop as the objects."""
        for _ in range(30):
            bw, bh = rng.randint(MIN_BOX_PX, 30), rng.randint(MIN_BOX_PX, 25)
            box = self.jitter(rng, (rng.randint(0, self.width - bw), rng.randint(0, self.height - bh), bw, bh))
            crop = vision.square_crop(frame, box)
            # Where square_crop actually cut from (it clamps to the image).
            side = crop.shape[0]
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            x0 = int(round(min(max(cx - side / 2, 0), self.width - side)))
            y0 = int(round(min(max(cy - side / 2, 0), self.height - side)))
            region = (x0, y0, side, side)
            if all(overlap_fraction(region, b) < 0.02 and overlap_fraction(b, region) < 0.02
                   for b in boxes_in_view):
                self.save(crop, "background", f"{self.split}_{layout_id:03d}_{pose_id:03d}_bg",
                          ["", "", *region, "", layout_id, round(self.luminosity, 2), ""])
                return
        self.skipped["background: no empty region in view"] += 1

    def run(self) -> None:
        done = 0
        try:
            for layout_id in range(self.layouts):
                rng = random.Random(self.seed_base + layout_id)
                placed = self.place_layout(rng)
                self.step(1)
                for pose_id in range(self.poses):
                    self.capture_view(rng, layout_id, pose_id, placed)
                done = layout_id + 1
                if done % 10 == 0:
                    print(f"[dataset] {self.split}: {done}/{self.layouts} layouts, {dict(self.counts)}")
        finally:
            self._csv_file.close()
            status = "done" if done == self.layouts else f"STOPPED EARLY after {done}/{self.layouts} layouts"
            print(f"[dataset] {self.split} {status}: {dict(self.counts)} -> {self.out}")
            print(f"[dataset] skipped: {dict(self.skipped) or 'nothing'}")
        self.sup.simulationQuit(0)
        self.step(1)


if __name__ == "__main__":
    try:
        args = parse_args(sys.argv[1:])
    except ValueError as exc:
        sys.exit(f"[dataset] bad controllerArgs: {exc}")
    Collector(args).run()
