"""CNN object tour: the e-puck drives to each object in worlds/cnn_objects.wbt
and ticks it off a checklist when its camera recognises it.

Separate demo world, not part of the main results. Three coloured blocks are
checked with the same HSV detector as the colour-block search (vision.py);
five CIFAR-10 objects (car, truck, horse, deer, boat), scaled down to toy
size, are named by the same int8 CIFAR-10 CNN, on the same background thread
(classifier.py).

What is and is not autonomous: the robot is a supervisor and reads where each
object is, like a map, so the route is planned from known positions. The
driving is real (wheel speeds from a proportional heading controller), and the
recognition is real: the verdict comes only from the camera image. At each
object the robot takes three views on one line (normal distance, 15% closer,
15% further) and adds up the CNN's top-3 confidences, so one bad view does not
decide.

controllerArgs (optional): --views=N (default 3), --quit (quit when done; for
batch testing).
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2 as cv
from controller import Supervisor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "colour_detector"))
import vision  # noqa: E402  (same HSV detection and cropping as the block search)
from classifier import AsyncClassifier  # noqa: E402  (same CNN, same background thread)

# Route order goes round the ring. kind "block": expected HSV colour;
# kind "object": expected CIFAR-10 class. The objects are real size, so each
# has its own viewing distance (m from its centre) at which it fills the
# camera's square crop. The world places each object so that this viewpoint is
# about 1.2 m from the arena centre (blocks: 2.5 m, so they stay out of other
# shots). The truck's 15% further view lies past the centre.
TARGETS = [
    ("RED_BLOCK", "block", "red", 1.0),
    ("TRUCK", "object", "truck", 9.0),
    ("GREEN_BLOCK", "block", "green", 1.0),
    ("HORSE", "object", "horse", 5.5),
    ("BLUE_BLOCK", "block", "blue", 1.0),
    ("CAR", "object", "automobile", 7.8),
    ("BOAT", "object", "ship", 0.55),
    ("DEER", "object", "deer", 4.0),
]
VIEW_SCALES = (1.0, 0.85, 1.15)  # three views on one line: normal, closer, further
# The truck's origin is at its rear axle, so aim 4.5 m forward (its local x) at the cab.
AIM_FORWARD_M = {"TRUCK": 4.5}
# Keep-out radius round each object. A side view whose straight path from the
# centre would pass this close to another object is skipped, not driven into.
CLEARANCE_M = {"RED_BLOCK": 0.6, "GREEN_BLOCK": 0.6, "BLUE_BLOCK": 0.6, "BOAT": 0.4,
               "HORSE": 1.6, "DEER": 1.2, "CAR": 2.7, "TRUCK": 4.0}
MAX_SPEED = 6.28  # rad/s, e-puck wheels
DRIVE_SPEED = MAX_SPEED  # full speed: the routes are long and clear
TURN_SPEED = 0.35 * MAX_SPEED
STEER_GAIN = 1.2 * MAX_SPEED  # wheel speed difference per radian of heading error
REACHED_M = 0.03
FACING_RAD = 0.03
LEG_TIMEOUT_S = 120.0
CNN_TIMEOUT_S = 3.0


def parse_args(argv: list[str]) -> dict[str, str]:
    values = {"views": "3", "quit": ""}
    for arg in argv:
        if not arg.strip():
            continue
        if arg == "--quit":
            values["quit"] = "1"
            continue
        if arg.startswith("--quit="):
            raise ValueError("use --quit on its own (no value)")
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"bad argument {arg!r}, expected --key=value or --quit")
        key, value = arg[2:].split("=", 1)
        if key not in values:
            raise ValueError(f"unknown argument --{key}")
        values[key] = value
    return values


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class ObjectFinder:
    def __init__(self, args: dict[str, str]) -> None:
        self.sup = Supervisor()
        self.step_ms = int(self.sup.getBasicTimeStep())
        self.camera = self.sup.getDevice("camera")
        self.camera.enable(self.step_ms)
        self.width, self.height = self.camera.getWidth(), self.camera.getHeight()
        self.left = self.sup.getDevice("left wheel motor")
        self.right = self.sup.getDevice("right wheel motor")
        for motor in (self.left, self.right):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
        self.me = self.sup.getSelf()
        if not args["views"].isdigit() or not 1 <= int(args["views"]) <= len(VIEW_SCALES):
            raise ValueError(f"--views must be 1 to {len(VIEW_SCALES)}, not {args['views']!r}")
        self.views = int(args["views"])
        self.quit_when_done = bool(args["quit"])
        self.nodes = {}
        for name, *_ in TARGETS:
            node = self.sup.getFromDef(name)
            if node is None:
                raise RuntimeError(f"world has no node DEF {name}")
            self.nodes[name] = node
        self.cnn = AsyncClassifier(HERE.parent / "colour_detector" / "models")
        print(f"[init] camera {self.width}x{self.height}, CNN {'ON' if self.cnn.enabled else 'OFF'} - "
              f"{self.cnn.description}")
        self.images = HERE / "images"
        self.images.mkdir(exist_ok=True)
        self.results: dict[str, tuple[str, str]] = {}  # name -> (status, what was seen)
        self.capture_count = 0

    # ----- low-level ---------------------------------------------------------
    def step(self) -> None:
        if self.sup.step(self.step_ms) == -1:
            raise SystemExit(0)

    def wheels(self, left: float, right: float) -> None:
        self.left.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
        self.right.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))

    def pose(self) -> tuple[float, float, float]:
        x, y, _ = self.me.getPosition()
        m = self.me.getOrientation()  # 3x3 rotation matrix, row-major
        return x, y, math.atan2(m[3], m[0])

    def target_xy(self, name: str) -> tuple[float, float]:
        """Where to aim: the object's origin, moved forward for the truck."""
        x, y, _ = self.nodes[name].getPosition()
        forward = AIM_FORWARD_M.get(name, 0.0)
        if forward:
            m = self.nodes[name].getOrientation()
            x, y = x + forward * m[0], y + forward * m[3]
        return x, y

    def path_is_clear(self, name: str, vx: float, vy: float) -> bool:
        """Does the straight path from the centre to (vx, vy) keep clear of the other objects?"""
        length = math.hypot(vx, vy)
        for other, *_ in TARGETS:
            if other == name:
                continue
            ox, oy = self.target_xy(other)  # the truck's body centre, not its rear axle
            t = 0.0 if length == 0 else max(0.0, min(1.0, (ox * vx + oy * vy) / length ** 2))
            if math.hypot(ox - t * vx, oy - t * vy) < CLEARANCE_M[other]:
                return False
        return True

    # ----- motion ------------------------------------------------------------
    def turn_to(self, heading: float) -> bool:
        """Turn on the spot to ``heading``. False if it timed out."""
        deadline = self.sup.getTime() + LEG_TIMEOUT_S
        ok = False
        while self.sup.getTime() < deadline:
            err = wrap(heading - self.pose()[2])
            if abs(err) < FACING_RAD:
                ok = True
                break
            speed = TURN_SPEED * max(0.15, min(1.0, abs(err) / 0.5))
            self.wheels(-speed if err > 0 else speed, speed if err > 0 else -speed)
            self.step()
        self.wheels(0, 0)
        self.step()
        if not ok:
            print(f"[drive] gave up turning to {math.degrees(heading):.0f} deg after {LEG_TIMEOUT_S:.0f} s")
        return ok

    def drive_to(self, gx: float, gy: float) -> bool:
        """Turn towards the goal, then drive with proportional heading control.
        False if the goal was not reached within LEG_TIMEOUT_S of simulated time."""
        deadline = self.sup.getTime() + LEG_TIMEOUT_S
        ok = False
        while self.sup.getTime() < deadline:
            x, y, heading = self.pose()
            dist = math.hypot(gx - x, gy - y)
            if dist < REACHED_M:
                ok = True
                break
            err = wrap(math.atan2(gy - y, gx - x) - heading)
            if abs(err) > 0.6:  # far off course (or just starting): turn on the spot first
                self.turn_to(math.atan2(gy - y, gx - x))
                continue
            base = DRIVE_SPEED * min(1.0, dist / 0.15 + 0.2)
            self.wheels(base - STEER_GAIN * err, base + STEER_GAIN * err)
            self.step()
        self.wheels(0, 0)
        self.step()
        if not ok:
            print(f"[drive] gave up reaching ({gx:.2f}, {gy:.2f}) after {LEG_TIMEOUT_S:.0f} s")
        return ok

    def go_and_face(self, vx: float, vy: float, face: float) -> bool:
        return self.drive_to(vx, vy) and self.turn_to(face)

    # ----- vision ------------------------------------------------------------
    def frame(self):
        for _ in range(3):  # let the robot settle and the camera render
            self.step()
        bgr = vision.webots_to_bgr(self.camera.getImage(), self.width, self.height)
        if bgr is None:
            raise RuntimeError("camera returned no image")
        return bgr

    def save(self, name: str, view: int, bgr) -> None:
        self.capture_count += 1
        path = self.images / f"{self.capture_count:03d}_{name.lower()}_view{view}.png"
        self.images.mkdir(exist_ok=True)  # in case it was deleted while the tour runs
        if not cv.imwrite(str(path), bgr):
            raise RuntimeError(f"could not write {path}")

    def check_block(self, name: str, expected: str, view: tuple[float, float, float]) -> None:
        if not self.go_and_face(*view):
            self.results[name] = ("NOT REACHED", "robot did not reach its viewpoint")
            return
        bgr = self.frame()
        self.save(name, 1, bgr)
        seen = vision.detect_blocks(bgr)
        if not seen:
            self.results[name] = ("MISSED", "HSV sees no coloured blob")
            print(f"[hsv] {name.lower()}: no coloured blob in view")
            return
        top = seen[0]
        self.results[name] = ("FOUND" if top.colour == expected else "MISSED", f"HSV sees {top.colour}")
        print(f"[hsv] {name.lower()}: biggest blob is {top.colour} ({top.area_fraction:.0%} of the image)")

    def classify(self, name: str, view: int, bgr) -> tuple[object | None, str]:
        """Submit to the background CNN and step until the answer comes back.
        Returns (prediction, "") or (None, why there is none)."""
        tag = f"{name.lower()}_view{view}"
        if not self.cnn.submit(tag, vision.square_crop(bgr)):
            return None, "CNN off"
        for _ in range(int(CNN_TIMEOUT_S * 1000 / self.step_ms)):
            for result in self.cnn.poll():
                if result.tag == tag:
                    return (None, f"CNN error: {result.error}") if result.error else (result, "")
            self.step()
        return None, f"no CNN answer within {CNN_TIMEOUT_S:.0f} s"

    def check_object(self, name: str, expected: str, views: list[tuple[float, float, float]]) -> None:
        # Vote: each view adds its top-1 confidence to its top-1 class.
        votes: dict[str, float] = defaultdict(float)
        answered, problems = 0, []
        for i, view in enumerate(views, start=1):
            if not self.path_is_clear(name, view[0], view[1]):
                problems.append(f"view {i} blocked")
                print(f"[tour] {name.lower()} view {i}: skipped, another object is in the way")
                continue
            if not self.go_and_face(*view):  # views share one line through the centre
                problems.append(f"view {i} not reached")
                continue
            bgr = self.frame()
            self.save(name, i, bgr)
            result, why = self.classify(name, i, bgr)
            if result is None:
                problems.append(f"view {i}: {why}")
                print(f"[cnn] {name.lower()} view {i}: {why}")
                continue
            answered += 1
            top3 = ", ".join(f"{label} {p:.0%}" for label, p in result.top3)
            print(f"[cnn] {name.lower()} view {i}/{len(views)}: {result.label} ({result.confidence:.0%})"
                  f"   top-3: {top3}   [{result.latency_ms:.1f} ms, off the control loop]")
            votes[result.label] += result.confidence
        if not answered:
            status = "SKIPPED" if any("CNN off" in p for p in problems) else "NOT REACHED"
            self.results[name] = (status, "; ".join(problems) or "no view")
            return
        best = max(votes, key=votes.get)
        self.results[name] = ("FOUND" if best == expected else "MISSED",
                              f"CNN says {best} (won {answered}-view vote)")

    # ----- tour --------------------------------------------------------------
    def views_for(self, name: str, distance: float) -> list[tuple[float, float, float]]:
        """Viewpoints on the line from the object through the arena centre, at
        slightly different distances, so the CNN sees it at three sizes. For far
        objects (the truck) the further view can lie past the centre."""
        tx, ty = self.target_xy(name)
        back = math.atan2(-ty, -tx)  # from the object towards the centre
        points = []
        for scale in VIEW_SCALES[: self.views]:
            d = distance * scale
            vx, vy = tx + d * math.cos(back), ty + d * math.sin(back)
            points.append((vx, vy, math.atan2(ty - vy, tx - vx)))
        return points

    def checklist(self) -> str:
        parts = []
        for name, *_ in TARGETS:
            label = name.lower().replace("_block", "")
            status = self.results[name][0] if name in self.results else "not yet"
            parts.append(f"{label} {status}")
        return " · ".join(parts)

    def run(self) -> None:
        if not self.cnn.enabled:
            print(f"[init] WARNING: CNN OFF - objects will be SKIPPED, only the blocks are checked. "
                  f"Reason: {self.cnn.description}")
        print("[tour] visiting 3 blocks (HSV colour check) and 5 objects (CIFAR-10 CNN)")
        try:
            for name, kind, expected, distance in TARGETS:
                print(f"[tour] -> {name.lower()} (expect {expected})")
                if not self.drive_to(0.0, 0.0):  # every leg starts at the centre
                    self.results[name] = ("NOT REACHED", "could not get back to the centre")
                else:
                    views = self.views_for(name, distance)
                    if kind == "block":
                        self.check_block(name, expected, views[0])
                    else:
                        self.check_object(name, expected, views)
                status, seen = self.results[name]
                print(f"[result] {name.lower()}: {seen} -> {status} (expected {expected})")
                print(f"[checklist] {self.checklist()}")
            self.drive_to(0.0, 0.0)
        finally:
            self.wheels(0, 0)  # never leave the robot driving after an error
            self.summary()
        if self.quit_when_done:
            all_found = all(self.results.get(n, ("",))[0] == "FOUND" for n, *_ in TARGETS)
            self.sup.simulationQuit(0 if all_found else 1)
            self.step()
        while True:
            self.step()

    def summary(self) -> None:
        def count(kind: str) -> str:
            names = [n for n, k, *_ in TARGETS if k == kind]
            found = sum(self.results.get(n, ("",))[0] == "FOUND" for n in names)
            return f"{found}/{len(names)}"
        print("=" * 70)
        print(f"TOUR SUMMARY: CNN objects recognised {count('object')} · colour blocks found {count('block')}")
        for name, _, expected, _ in TARGETS:
            status, seen = self.results.get(name, ("NOT VISITED", "tour stopped before this"))
            print(f"  {name.lower():12s} expected {expected:11s} {seen:38s} {status}")
        if self.cnn.dropped_frames or self.cnn.failures:
            print(f"  CNN: {self.cnn.failures} failed inferences, {self.cnn.dropped_frames} dropped frames")
        print(f"  images saved in {self.images}")
        print("=" * 70)


if __name__ == "__main__":
    ObjectFinder(parse_args(sys.argv[1:])).run()
