"""CNN object tour: the e-puck finds the objects in worlds/cnn_objects.wbt by
itself, drives to each one, names it, and ticks it off a checklist.

Separate demo world, not part of the main results.

How it works
------------
1. Find: from the arena centre the robot takes one 360-degree lidar scan. Hits
   on the floor are dropped and the rest are grouped into objects
   (lidar_objects.py). The robot knows its own position and heading from its
   GPS and inertial unit; it is never told where the objects are.
2. Look: for each object it drives along the line from the centre towards the
   object, to a distance that depends on the object's width, so the object fills
   the camera's square crop.
3. Name: if the view is dominated by one coloured blob (HSV, vision.py), it is a
   coloured block. Otherwise it takes three photos (normal distance, 15% closer,
   15% further), the int8 CIFAR-10 CNN names each on its background thread
   (classifier.py), and the answers vote. If fewer than two views agree, or the
   winner's mean confidence is below 60%, the robot answers "unsure".
4. Show: the checklist and the current answer are drawn on the 3D view. If
   TensorFlow is available (see runtime.ini), a Grad-CAM heatmap of the best
   view of each named object is computed on a background thread with the full
   Keras model and shown on the robot's display, also drawn on the 3D view.

The robot is a supervisor only so that, at the end, its answers can be scored
against what each object really is. Navigation and recognition use only the
robot's own sensors.

controllerArgs (optional): --quit (quit when done; for batch testing).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from controller import Supervisor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "colour_detector"))
import vision  # noqa: E402  (same HSV detection and cropping as the block search)
from classifier import AsyncClassifier  # noqa: E402  (same CNN, same background thread)

from lidar_objects import find_objects, to_world, vote  # noqa: E402
from gradcam_live import LiveGradCam  # noqa: E402  (optional: needs TensorFlow)

# Ground truth, used ONLY to score the robot's answers at the end.
TRUTH = {"RED_BLOCK": "red block", "GREEN_BLOCK": "green block", "BLUE_BLOCK": "blue block",
         "TRUCK": "truck", "HORSE": "horse", "CAR": "automobile", "BOAT": "ship", "DEER": "deer"}
# The truck's origin is its rear axle; its body centre is about 3 m forward.
BODY_FORWARD_M = {"TRUCK": 3.0}
MATCH_M = 4.0  # an object found within this distance of a real object's body can be that object
OVERSHOOT_M = 3.0  # a viewpoint may lie at most this far past the start point

BLOCK_FILL = 0.25  # a blob covering this much of the frame means a coloured block (blocks fill ~32%)
VIEW_SCALES = (1.0, 0.85, 1.15)  # three views on one line: normal, closer, further
MAX_SPEED = 6.28  # rad/s, e-puck wheels
DRIVE_SPEED = MAX_SPEED
TURN_SPEED = 0.35 * MAX_SPEED
STEER_GAIN = 1.2 * MAX_SPEED
REACHED_M = 0.03
FACING_RAD = 0.03
LEG_TIMEOUT_S = 120.0
CNN_TIMEOUT_S = 3.0
BLACK, YELLOW = 0x101010, 0xE0A000


def parse_args(argv: list[str]) -> dict[str, bool]:
    quit_when_done = False
    for arg in argv:
        if not arg.strip():
            continue
        if arg != "--quit":
            raise ValueError(f"unknown argument {arg!r} (only --quit is accepted)")
        quit_when_done = True
    return {"quit": quit_when_done}


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class ObjectFinder:
    def __init__(self, args: dict[str, bool]) -> None:
        self.sup = Supervisor()
        self.step_ms = int(self.sup.getBasicTimeStep())

        def device(name):
            dev = self.sup.getDevice(name)
            if dev is None:
                raise RuntimeError(f"the e-puck has no device {name!r} (check worlds/cnn_objects.wbt)")
            return dev

        self.camera = device("camera")
        self.camera.enable(self.step_ms)
        self.width, self.height = self.camera.getWidth(), self.camera.getHeight()
        self.lidar = device("lidar")
        self.lidar.enable(self.step_ms)
        self.lidar.enablePointCloud()
        self.gps = device("gps")
        self.gps.enable(self.step_ms)
        self.imu = device("inertial unit")
        self.imu.enable(self.step_ms)
        self.left, self.right = device("left wheel motor"), device("right wheel motor")
        for motor in (self.left, self.right):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
        self.quit_when_done = args["quit"]
        self.cnn = AsyncClassifier(HERE.parent / "colour_detector" / "models")
        print(f"[init] camera {self.width}x{self.height}, CNN {'ON' if self.cnn.enabled else 'OFF'} - "
              f"{self.cnn.description}")
        self.gradcam = LiveGradCam()
        state = "ON (Keras model, background thread)" if self.gradcam.available else "OFF - " + self.gradcam.reason
        print(f"[init] Grad-CAM {state}")
        self.display = self.sup.getDevice("display")  # optional screen for the heatmap
        self.images = HERE / "images"
        self.capture_count = 0
        self.found = []
        self.answers: list[dict] = []  # one per object found
        self.hud_status = "starting"
        self.home = (0.0, 0.0)

    # ----- low-level ---------------------------------------------------------
    def step(self) -> None:
        if self.sup.step(self.step_ms) == -1:
            raise SystemExit(0)
        for tag, img, label, prob, error in self.gradcam.poll():
            if error:
                print(f"[gradcam] {tag}: failed - {error}")
            else:
                self.show_heatmap(tag, img, label, prob)

    def show_heatmap(self, tag: str, img, label: str, prob: float) -> None:
        """Save the Grad-CAM overlay and paste it on the robot's display."""
        print(f"[gradcam] {tag}: {label} {prob:.0%}, heatmap shown on the display")
        self.images.mkdir(exist_ok=True)
        cv.imwrite(str(self.images / f"gradcam_{tag}.png"), img)
        if self.display is None:
            return
        w, h = self.display.getWidth(), self.display.getHeight()
        bgra = cv.cvtColor(cv.resize(img, (w, h)), cv.COLOR_BGR2BGRA)
        ref = self.display.imageNew(bgra.tobytes(), self.display.BGRA, w, h)
        self.display.imagePaste(ref, 0, 0, False)
        self.display.imageDelete(ref)
        self.display.setColor(0x000000)
        self.display.fillRectangle(0, 0, w, 26)
        self.display.setColor(0xFFFFFF)
        self.display.setFont("Arial", 16, True)
        self.display.drawText(f"Grad-CAM: {label} {prob:.0%}", 6, 5)

    def wheels(self, left: float, right: float) -> None:
        self.left.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
        self.right.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))

    def pose(self) -> tuple[float, float, float]:
        """Position from GPS, heading (yaw) from the inertial unit."""
        x, y, _ = self.gps.getValues()
        return x, y, self.imu.getRollPitchYaw()[2]

    # ----- on-screen display -------------------------------------------------
    def hud(self, status: str | None = None) -> None:
        if status is not None:
            self.hud_status = status
        lines = [f"Objects found by lidar: {len(self.found)}"]
        for i, a in enumerate(self.answers, start=1):
            lines.append(f"{i}. {a['said']}")
        # Right-hand side: the Grad-CAM display sits in the top-left corner of the 3D view.
        self.sup.setLabel(0, "CNN OBJECT TOUR", 0.66, 0.01, 0.14, YELLOW, 0.0, "Arial Black")
        self.sup.setLabel(1, "\n".join(lines), 0.66, 0.08, 0.12, BLACK, 0.0, "Arial Black")
        self.sup.setLabel(2, self.hud_status, 0.01, 0.9, 0.13, BLACK, 0.0, "Arial Black")

    # ----- motion ------------------------------------------------------------
    def turn_to(self, heading: float) -> bool:
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
        deadline = self.sup.getTime() + LEG_TIMEOUT_S
        ok = False
        while self.sup.getTime() < deadline:
            x, y, heading = self.pose()
            dist = math.hypot(gx - x, gy - y)
            if dist < REACHED_M:
                ok = True
                break
            err = wrap(math.atan2(gy - y, gx - x) - heading)
            if abs(err) > 0.6:
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

    # ----- sensing -----------------------------------------------------------
    def scan(self) -> None:
        """One 360-degree lidar scan from where the robot stands."""
        for _ in range(3):
            self.step()
        cloud = self.lidar.getPointCloud()
        raw = np.array([[p.x, p.y, p.z] for p in cloud], dtype=float).reshape(-1, 3)
        raw = raw[np.isfinite(raw).all(axis=1)]  # no-return beams come back as infinity
        if len(raw) == 0:
            raise RuntimeError("the lidar returned no points - is it enabled and in the turret slot?")
        x, y, yaw = self.pose()
        self.home = (x, y)  # every leg starts here, so paths never cross another object
        self.found = find_objects(to_world(raw, x, y, self.lidar_height, yaw), self.lidar.getMaxRange(), (x, y))
        print(f"[lidar] {len(raw)} points, {len(self.found)} objects found")
        for i, f in enumerate(self.found, start=1):
            print(f"[lidar]   object {i}: at ({f.x:+.1f}, {f.y:+.1f}) m, {f.width:.1f} m wide, "
                  f"{f.points} points -> view from {f.view_distance():.1f} m")

    @property
    def lidar_height(self) -> float:
        """Height of the lidar above the floor, from where it is mounted on the robot."""
        if not hasattr(self, "_lidar_h"):
            slot = self.sup.getSelf().getField("turretSlot")
            nodes = [slot.getMFNode(i) for i in range(slot.getCount())]
            lidars = [n for n in nodes if n.getTypeName() == "Lidar"]
            if not lidars:
                raise RuntimeError("no Lidar node in the e-puck's turretSlot")
            self._lidar_h = lidars[0].getPosition()[2]
        return self._lidar_h

    def frame(self):
        for _ in range(3):
            self.step()
        bgr = vision.webots_to_bgr(self.camera.getImage(), self.width, self.height)
        if bgr is None:
            raise RuntimeError("camera returned no image")
        return bgr

    def save(self, k: int, view: int, bgr) -> None:
        self.capture_count += 1
        self.images.mkdir(exist_ok=True)
        path = self.images / f"{self.capture_count:03d}_object{k}_view{view}.png"
        if not cv.imwrite(str(path), bgr):
            raise RuntimeError(f"could not write {path}")

    def classify(self, tag: str, bgr):
        if not self.cnn.submit(tag, vision.square_crop(bgr)):
            return None, "CNN off"
        for _ in range(int(CNN_TIMEOUT_S * 1000 / self.step_ms)):
            for result in self.cnn.poll():
                if result.tag == tag:
                    return (None, f"CNN error: {result.error}") if result.error else (result, "")
            self.step()
        return None, f"no CNN answer within {CNN_TIMEOUT_S:.0f} s"

    # ----- one object ---------------------------------------------------------
    def look_at(self, k: int, obj) -> dict:
        cx, cy = self.home
        back = math.atan2(cy - obj.y, cx - obj.x)
        # Never stand more than OVERSHOOT_M past the start point: a large object
        # (the truck) would otherwise put the camera beside another object.
        d_max = math.hypot(obj.x - cx, obj.y - cy) + OVERSHOOT_M
        answer = {"x": obj.x, "y": obj.y, "said": "not reached", "status": "NOT REACHED", "views": []}
        cnn_answers, problems, block_checked, frames = [], [], False, []
        for i, scale in enumerate(VIEW_SCALES, start=1):
            d = min(obj.view_distance(), d_max / max(VIEW_SCALES)) * scale
            vx, vy = obj.x + d * math.cos(back), obj.y + d * math.sin(back)
            self.hud(f"Object {k}: driving to view {i}")
            if not (self.drive_to(vx, vy) and self.turn_to(math.atan2(obj.y - vy, obj.x - vx))):
                problems.append(f"view {i} not reached")
                continue
            bgr = self.frame()
            self.save(k, i, bgr)
            if not block_checked:  # first captured view decides: coloured block, or one for the CNN?
                block_checked = True
                blobs = vision.detect_blocks(bgr)
                if blobs and blobs[0].area_fraction >= BLOCK_FILL:
                    colour = blobs[0].colour
                    print(f"[hsv] object {k}: {colour} blob fills {blobs[0].area_fraction:.0%} -> {colour} block")
                    answer.update(said=f"{colour} block", status="BLOCK", label=f"{colour} block")
                    return answer
            result, why = self.classify(f"object{k}_view{i}", bgr)
            if result is None:
                print(f"[cnn] object {k} view {i}: {why}")
                problems.append(f"view {i}: {why}")
                continue
            top3 = ", ".join(f"{label} {p:.0%}" for label, p in result.top3)
            print(f"[cnn] object {k} view {i}/3: {result.label} ({result.confidence:.0%})   top-3: {top3}"
                  f"   [{result.latency_ms:.1f} ms, off the control loop]")
            self.hud(f"Object {k}, view {i}: CNN says {result.label} {result.confidence:.0%}")
            cnn_answers.append((result.label, result.confidence))
            frames.append((result.label, result.confidence, bgr))
            answer["views"].append(f"{result.label} {result.confidence:.0%}")
        if cnn_answers:
            label, mean, sure = vote(cnn_answers)
            # Grad-CAM on the view the CNN was most sure about for the winning class.
            best = max((f for f in frames if f[0] == label), key=lambda f: f[1])
            self.gradcam.submit(f"object{k}", vision.square_crop(best[2]))
            answer.update(label=label, status="SURE" if sure else "UNSURE",
                          said=f"{label} ({mean:.0%})" if sure else f"unsure ({label}? {mean:.0%})")
        elif block_checked:  # reached, but the CNN gave no answer
            answer.update(status="NO ANSWER", said="no CNN answer")
        if problems:
            print(f"[tour] object {k}: " + "; ".join(problems))
        return answer

    # ----- scoring (ground truth used only here) ------------------------------
    def truth_xy(self, name: str) -> tuple[float, float]:
        node = self.sup.getFromDef(name)
        x, y, _ = node.getPosition()
        fwd = BODY_FORWARD_M.get(name, 0.0)
        if fwd:
            m = node.getOrientation()
            x, y = x + fwd * m[0], y + fwd * m[3]
        return x, y

    def score(self) -> None:
        """Compare the robot's answers with what each object really is (one-to-one)."""
        truths = {}
        for n in TRUTH:
            if self.sup.getFromDef(n) is None:
                print(f"[score] WARNING: no node DEF {n} in the world, it cannot be scored")
            else:
                truths[n] = self.truth_xy(n)
        print("=" * 72)
        if not truths:
            print("[score] NO GROUND TRUTH in the world - cannot score the answers")
            return
        # Greedy one-to-one matching: closest pairs first, each real object used once.
        pairs = sorted((math.hypot(tx - a["x"], ty - a["y"]), k, n)
                       for k, a in enumerate(self.answers) for n, (tx, ty) in truths.items())
        match, used = {}, set()
        for dist, k, n in pairs:
            if dist < MATCH_M and k not in match and n not in used:
                match[k] = n
                used.add(n)
        right = 0
        for k, a in enumerate(self.answers):
            real = TRUTH[match[k]] if k in match else "nothing (false detection)"
            if a["status"] in ("NOT REACHED", "NO ANSWER"):
                verdict = a["status"]
            elif a.get("label") == real and a["status"] in ("SURE", "BLOCK"):
                verdict = "RIGHT"
                right += 1
            elif a["status"] == "UNSURE":
                verdict = "UNSURE (right guess)" if a.get("label") == real else "UNSURE (wrong guess)"
            else:
                verdict = "WRONG"
            print(f"  object {k + 1}: robot said {a['said']:28s} really {real:26s} {verdict}")
        missed = [TRUTH[n] for n in truths if n not in used]
        print(f"TOUR SUMMARY: lidar found {len(self.found)} objects; {right} of {len(truths)} real objects named "
              "correctly" + (f"; never found: {', '.join(missed)}" if missed else "; none missed")
              + ("" if self.cnn.enabled else "; CNN WAS OFF"))
        print(f"  images saved in {self.images}")
        print("=" * 72)
        self.hud(f"Done: {right} of {len(truths)} named correctly")

    # ----- tour --------------------------------------------------------------
    def run(self) -> None:
        if not self.cnn.enabled:
            print(f"[init] WARNING: CNN OFF - only blocks can be named. Reason: {self.cnn.description}")
        try:
            self.hud("Scanning with lidar...")
            self.scan()
            for k, obj in enumerate(self.found, start=1):
                print(f"[tour] -> object {k} of {len(self.found)}")
                if not self.drive_to(*self.home):
                    self.answers.append({"x": obj.x, "y": obj.y, "said": "not reached", "status": "NOT REACHED"})
                    continue
                a = self.look_at(k, obj)
                self.answers.append(a)
                print(f"[result] object {k}: {a['said']}")
                self.hud(f"Object {k}: {a['said']}")
            self.drive_to(*self.home)
            for _ in range(int(3000 / self.step_ms)):  # let the last heatmap arrive
                self.step()
        except SystemExit:  # Webots is closing the simulation: nothing more to do
            raise
        except Exception:
            self.wheels(0, 0)
            self.score()
            if self.quit_when_done:
                self.sup.simulationQuit(1)
            raise
        self.wheels(0, 0)
        self.score()
        if self.quit_when_done:
            self.sup.simulationQuit(0)
            self.step()
        while True:
            self.step()


if __name__ == "__main__":
    ObjectFinder(parse_args(sys.argv[1:])).run()
