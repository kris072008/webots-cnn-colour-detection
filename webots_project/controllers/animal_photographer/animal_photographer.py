"""Supervisor e-puck that photographs every Webots animal for the CNN.

Used with worlds/animal_safari.wbt. The world holds the eight animals in the
Webots object library: cat, deer, dog and horse are CIFAR-10 classes; cow,
fox, rabbit and sheep are not, so they test what the CNN does with animals
it was never taught.

For each animal the controller hides the others, then drives its camera
round the animal in ``--views`` equal steps, always facing it. At each view
it takes two frames: one with the animal and one with the animal briefly
moved away. The difference between them is the animal's silhouette, which
gives an exact bounding box with no hand labelling (background
subtraction). If the silhouette touches the image edge the robot backs off
and tries again, so the whole animal is always in view. The crop round the
box is made with the robot's own ``vision.square_crop``, as for the blocks.

Saves, in --out (default ../../../cnn/webots_animals):
  <animal>_<angle>.png          full camera frame
  <animal>_<angle>_crop.png     square crop the CNN classifies
  <animal>_<angle>_mask.png     the animal's silhouette in that crop
  labels.csv                    one row per view
Then quits the simulation.

controllerArgs (optional): --views=N (default 24), --out=PATH
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from controller import Supervisor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "colour_detector"))
import vision  # noqa: E402  (same frame conversion and cropping as the robot)

from silhouette import silhouette  # noqa: E402

# Animal DEF name -> CIFAR-10 class, or "" if CIFAR-10 has no such class.
ANIMALS = {"CAT": "cat", "DEER": "deer", "DOG": "dog", "HORSE": "horse",
           "COW": "", "FOX": "", "RABBIT": "", "SHEEP": ""}
HIDDEN = [100.0, 100.0, -50.0]  # far away and under the floor
EDGE_PX = 1
START_DISTANCE = 0.6  # m; the robot backs off from here until the animal fits
BACK_OFF = 1.2
MAX_DISTANCE = 15.0
HEADER = ["file", "animal", "cifar_class", "angle_deg", "distance_m",
          "bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def parse_args(argv: list[str]) -> dict[str, str]:
    values = {"views": "24", "out": str(HERE.parents[2] / "cnn" / "webots_animals")}
    for arg in argv:
        if not arg.strip():
            continue
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"bad argument {arg!r}, expected --key=value")
        key, value = arg[2:].split("=", 1)
        if key not in values:
            raise ValueError(f"unknown argument --{key}")
        values[key] = value
    return values


class Photographer:
    def __init__(self, args: dict[str, str]) -> None:
        self.sup = Supervisor()
        self.step_ms = int(self.sup.getBasicTimeStep())
        self.camera = self.sup.getDevice("camera")
        self.camera.enable(self.step_ms)
        self.width, self.height = self.camera.getWidth(), self.camera.getHeight()
        self.me = self.sup.getSelf()
        self.views = int(args["views"])
        self.out = Path(args["out"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.nodes = {}
        for name in ANIMALS:
            node = self.sup.getFromDef(name)
            if node is None:
                raise RuntimeError(f"world has no node DEF {name}")
            self.nodes[name] = node
        self.saved = 0
        self.problems: list[str] = []

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.sup.step(self.step_ms) == -1:
                print(f"[animals] simulation stopped early: only {self.saved} views saved, labels.csv is incomplete")
                raise SystemExit(1)

    def frame(self) -> np.ndarray:
        self.step(2)  # let the camera render the new scene
        bgr = vision.webots_to_bgr(self.camera.getImage(), self.width, self.height)
        if bgr is None:
            raise RuntimeError("camera returned no image")
        return bgr

    def place(self, name: str, position: list[float]) -> None:
        self.nodes[name].getField("translation").setSFVec3f(position)

    def look_from(self, angle: float, distance: float) -> None:
        """Put the robot ``distance`` m from the origin at ``angle``, facing it."""
        x, y = distance * math.cos(angle), distance * math.sin(angle)
        self.me.getField("translation").setSFVec3f([x, y, 0.0])
        self.me.getField("rotation").setSFRotation([0, 0, 1, angle + math.pi])
        self.me.resetPhysics()

    def capture(self, name: str, angle: float, distance: float):
        """Back off until the whole animal is in view; return frame, mask, bbox, distance."""
        while distance <= MAX_DISTANCE:
            self.look_from(angle, distance)
            self.place(name, [0.0, 0.0, 0.0])
            with_animal = self.frame()
            self.place(name, HIDDEN)
            without = self.frame()
            mask, bbox = silhouette(with_animal, without)
            if bbox is not None:
                x, y, w, h = bbox
                touches = (x <= EDGE_PX or y <= EDGE_PX or x + w >= self.width - EDGE_PX
                           or y + h >= self.height - EDGE_PX)
                # The square crop cannot be wider than the frame is tall: back off if it would cut the animal.
                if not touches and max(w, h) <= self.height:
                    return with_animal, mask, bbox, distance
            distance *= BACK_OFF
        return None, None, None, distance

    def run(self) -> None:
        with (self.out / "labels.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(HEADER)
            for name in ANIMALS:
                self.place(name, HIDDEN)
            for name, cifar in ANIMALS.items():
                distance = START_DISTANCE
                for view in range(self.views):
                    angle_deg = round(360.0 * view / self.views, 1)
                    frame, mask, bbox, used = self.capture(name, math.radians(angle_deg), distance)
                    if frame is None:
                        self.problems.append(f"{name} at {angle_deg} deg: never fully in view")
                        continue
                    distance = max(START_DISTANCE, used / BACK_OFF ** 2)  # next view starts a bit closer
                    stem = f"{name.lower()}_{int(angle_deg):03d}"
                    # The mask gets the same crop, for the background-swap test in the notebook.
                    images = {f"{stem}.png": frame,
                              f"{stem}_crop.png": vision.square_crop(frame, bbox, margin=0.15),
                              f"{stem}_mask.png": vision.square_crop(cv.cvtColor(mask * 255, cv.COLOR_GRAY2BGR),
                                                                     bbox, margin=0.15)}
                    for file, image in images.items():
                        if not cv.imwrite(str(self.out / file), image):
                            raise RuntimeError(f"could not write {self.out / file}")
                    writer.writerow([stem, name.lower(), cifar, angle_deg, round(used, 3), *bbox])
                    fh.flush()
                    self.saved += 1
                print(f"[animals] {name.lower()}: done, {self.saved} views saved so far")
        print(f"[animals] saved {self.saved} views to {self.out}")
        # Keep a record next to the data, so a short dataset can never pass unnoticed.
        (self.out / "problems.txt").write_text("\n".join(self.problems) + ("\n" if self.problems else ""))
        print(f"[animals] problems: {self.problems or 'none'}")
        self.sup.simulationQuit(1 if self.problems else 0)
        self.step(1)


if __name__ == "__main__":
    Photographer(parse_args(sys.argv[1:])).run()
