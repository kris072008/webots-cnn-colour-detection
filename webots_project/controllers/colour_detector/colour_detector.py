"""e-puck controller: search for coloured blocks, approach, photograph, classify.

Behaviour is a finite-state machine (FSM), one method per state:

    SEARCH --wanted block seen--> APPROACH --close enough--> INSPECT --> LEAVE
      |  ^                           |  target lost                       |
      v  |                           v                                    |
      SCAN (360 deg look-around)   SEARCH  <------------------------------+

SEARCH    Drive forward with obstacle avoidance. If no wanted block has been
          seen for a while, switch to SCAN.
SCAN      Spin on the spot for one full turn to look around the arena.
APPROACH  Steer towards the target's centroid with a proportional controller
          (turn harder the further the block is from the image centre).
INSPECT   Stop, save the raw + annotated camera image, log the encounter and
          hand the image to the CNN (which runs on a background thread).
LEAVE     Back away (stopping if the rear sensors see something) and turn.
          That colour is then skipped until every colour has been found.

Press C (with the 3D view focused) to capture and classify the current camera
frame at any time.

Hardware used (e-puck, see README.md): 2 wheel motors (max 6.28 rad/s),
8 IR proximity sensors ps0-ps7, 52x39 RGB camera.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2 as cv
from controller import Keyboard, Robot

import vision
from classifier import INPUT_SIZE, AsyncClassifier

# ---------------------------------------------------------------------------
# Parameters (all tunable in one place)
# ---------------------------------------------------------------------------
TIME_STEP = 32  # ms per control step (matches the world's basicTimeStep)
VISION_PERIOD = 2  # run vision every 2nd step (64 ms ~ 15 fps) to save CPU

# e-puck geometry and limits
MAX_SPEED = 6.28  # rad/s
WHEEL_RADIUS = 0.0205  # m
AXLE_LENGTH = 0.052  # m

# IR proximity sensors (see README for the layout around the robot)
FRONT_RIGHT_SENSORS = (0, 1)  # ps0 front-right, ps1 right-diagonal
FRONT_LEFT_SENSORS = (7, 6)  # ps7 front-left,  ps6 left-diagonal
REAR_SENSORS = (3, 4)  # ps3 back-right,   ps4 back-left
OBSTACLE_THRESHOLD = 80.0  # reading above which something is close (idle noise ~60-70)

# Speeds (fractions of MAX_SPEED)
CRUISE = 0.5 * MAX_SPEED
APPROACH_SPEED = 0.4 * MAX_SPEED
CREEP_SPEED = 0.12 * MAX_SPEED  # when the target is briefly lost
SCAN_SPEED = 0.3 * MAX_SPEED
STEER_GAIN = 0.35 * MAX_SPEED  # P-gain: wheel speed difference per unit offset
AVOID_INNER_WHEEL = -0.3  # inner wheel factor when turning away from an obstacle

# Behaviour thresholds and timings
ARRIVED_AREA = 0.30  # block fills 30 % of the image -> close enough
BUMP_ARRIVAL_AREA = 0.10  # front blocked counts as arrival only if target is this big
LOST_FRAMES = 8  # processed frames without the target before giving up
INSPECT_SECONDS = 1.0
REVERSE_SECONDS = 0.6
TURN_AWAY_SECONDS = 1.4
COLOUR_COOLDOWN_SECONDS = 12.0  # ignore a colour this long after inspecting it
SCAN_AFTER_SECONDS = 8.0  # no wanted block seen this long -> 360 degree scan
SUMMARY_EVERY_SECONDS = 60.0

CONTROLLER_DIR = Path(__file__).resolve().parent
IMAGE_DIR = CONTROLLER_DIR / "images"
MODEL_DIR = CONTROLLER_DIR / "models"
LOG_PATH = IMAGE_DIR / "encounters.csv"


class State(Enum):
    SEARCH = auto()
    SCAN = auto()
    APPROACH = auto()
    INSPECT = auto()
    LEAVE = auto()


@dataclass(frozen=True)
class Encounter:
    colour: str
    sim_time: float
    tag: str | None  # capture tag, None if no image could be taken


class EncounterLog:
    """encounters.csv writer. A disk error disables logging, never the robot."""

    HEADER = ["wall_time", "sim_time_s", "event", "colour", "area_fraction",
              "image", "cnn_label", "cnn_confidence"]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not path.exists() or path.stat().st_size == 0
            self._file = path.open("a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            if is_new:
                self._write(self.HEADER)
        except OSError as exc:
            print(f"[log] cannot open {path}: {exc} - CSV logging disabled")
            self._file = None

    def block_found(self, sim_time: float, det: vision.Detection, image: str) -> None:
        self._row(sim_time, "block_found", det.colour, f"{det.area_fraction:.3f}", image, "", "")

    def prediction(self, sim_time: float, tag: str, label: str, confidence: float) -> None:
        self._row(sim_time, "cnn_prediction", "", "", f"{tag}.png", label, f"{confidence:.3f}")

    def error(self, sim_time: float, tag: str, message: str) -> None:
        self._row(sim_time, "cnn_error", "", "", f"{tag}.png", message, "")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _row(self, sim_time: float, *fields: str) -> None:
        self._write([datetime.now().isoformat(timespec="seconds"), f"{sim_time:.2f}", *fields])

    def _write(self, row: list[str]) -> None:
        if self._file is None:
            return
        try:
            self._writer.writerow(row)
            self._file.flush()  # keep the log complete even if Webots kills us
        except OSError as exc:
            print(f"[log] write to {self.path.name} failed: {exc} - CSV logging disabled")
            self.close()


class ColourDetectorRobot:
    def __init__(self) -> None:
        self.robot = Robot()
        self.left_motor = self._device("left wheel motor")
        self.right_motor = self._device("right wheel motor")
        for motor in (self.left_motor, self.right_motor):
            motor.setPosition(float("inf"))  # velocity control mode
            motor.setVelocity(0.0)

        self.ps = [self._device(f"ps{i}") for i in range(8)]
        for sensor in self.ps:
            sensor.enable(TIME_STEP)

        self.camera = self._device("camera")
        # Only render a frame as often as we process one (pre-filter: fewer
        # frames = less work for the simulator and for us).
        self.camera.enable(TIME_STEP * VISION_PERIOD)
        self.width = self.camera.getWidth()
        self.height = self.camera.getHeight()

        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(TIME_STEP)

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.log = EncounterLog(LOG_PATH)
        self.classifier = AsyncClassifier(MODEL_DIR)
        status = "ON - " if self.classifier.enabled else "OFF - "
        print(f"[init] camera {self.width}x{self.height}, CNN {status}{self.classifier.description}")

        # FSM
        self.state = State.SEARCH
        self.state_started = 0.0
        self.target_colour: str | None = None
        self.frames_without_target = 0
        self.last_wanted_seen = 0.0
        self.cooldown_until: dict[str, float] = {}
        self.found_this_round: set[str] = set()
        self.round_number = 1
        self.state_handlers = {
            State.SEARCH: self._step_search,
            State.SCAN: self._step_scan,
            State.APPROACH: self._step_approach,
            State.INSPECT: self._step_inspect,
            State.LEAVE: self._step_leave,
        }

        # Vision
        self.last_frame = None
        self.last_detections: list[vision.Detection] = []
        self.new_frame = False  # True on steps where a fresh frame was processed
        self.camera_warned = False
        self.vision_ms_total = 0.0
        self.vision_calls = 0

        # Results
        self.capture_count = 0
        self.encounters: list[Encounter] = []
        self.predictions: dict[str, str] = {}
        self.startup_snapshot_done = False
        self.next_summary = SUMMARY_EVERY_SECONDS

    def _device(self, name: str):
        """getDevice that fails with a clear message instead of returning None."""
        dev = self.robot.getDevice(name)
        if dev is None:
            raise RuntimeError(f"device {name!r} not found on this robot (is it an e-puck?)")
        return dev

    # ------------------------------------------------------------------ sensing
    def now(self) -> float:
        return self.robot.getTime()

    def _blocked(self, sensor_ids: tuple[int, ...]) -> bool:
        return max(self.ps[i].getValue() for i in sensor_ids) > OBSTACLE_THRESHOLD

    def obstacle_sides(self) -> tuple[bool, bool]:
        """(left_blocked, right_blocked) from the front/diagonal IR sensors."""
        return self._blocked(FRONT_LEFT_SENSORS), self._blocked(FRONT_RIGHT_SENSORS)

    def update_vision(self, step_count: int) -> None:
        """Grab a frame and detect blocks (only every VISION_PERIOD steps).

        Sets self.new_frame so the FSM can count real frames. A missing or
        bad frame clears the detections, so the robot never steers on stale data.
        """
        self.new_frame = False
        if step_count % VISION_PERIOD:
            return
        try:
            frame = vision.webots_to_bgr(self.camera.getImage(), self.width, self.height)
        except ValueError as exc:
            self._warn_camera(f"bad camera frame skipped (further ones silent): {exc}")
            self.last_detections = []
            return
        if frame is None:
            self.last_detections = []
            if self.now() > 2.0 and self.last_frame is None:
                self._warn_camera("camera has produced no frames after 2 s")
            return
        start = cv.getTickCount()
        self.last_frame = frame
        self.last_detections = vision.detect_blocks(frame)
        self.vision_ms_total += (cv.getTickCount() - start) / cv.getTickFrequency() * 1000
        self.vision_calls += 1
        self.new_frame = True
        if self.wanted_detection() is not None:
            self.last_wanted_seen = self.now()

    def _warn_camera(self, message: str) -> None:
        if not self.camera_warned:
            print(f"[vision] WARNING: {message}")
            self.camera_warned = True

    def wanted_detection(self) -> vision.Detection | None:
        """Largest visible block that is worth approaching.

        Colours already found this round are skipped, so the robot goes
        looking for the others instead of revisiting the nearest block.
        The cooldown also stops it re-approaching the block it just left.
        """
        t = self.now()
        for det in self.last_detections:
            if det.colour in self.found_this_round:
                continue
            if self.cooldown_until.get(det.colour, 0.0) <= t:
                return det
        return None

    def target_detection(self) -> vision.Detection | None:
        return next((d for d in self.last_detections if d.colour == self.target_colour), None)

    # ------------------------------------------------------------------- acting
    def drive(self, left: float, right: float) -> None:
        self.left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
        self.right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))

    def stop(self) -> None:
        self.drive(0.0, 0.0)

    def avoid_obstacle(self, left_blocked: bool, right_blocked: bool) -> None:
        if left_blocked and right_blocked:
            self.drive(-CRUISE, 0.2 * CRUISE)  # back off while turning left
        elif left_blocked:
            self.drive(CRUISE, AVOID_INNER_WHEEL * CRUISE)  # obstacle left -> turn right
        else:
            self.drive(AVOID_INNER_WHEEL * CRUISE, CRUISE)  # obstacle right -> turn left

    @staticmethod
    def full_turn_seconds(wheel_speed: float) -> float:
        """Time to spin 360 degrees in place: body rate = 2 * r * w / L."""
        return 2 * math.pi / (2 * WHEEL_RADIUS * wheel_speed / AXLE_LENGTH)

    # ---------------------------------------------------------------- the FSM
    def change_state(self, new_state: State) -> None:
        if new_state is not self.state:
            target = f" ({self.target_colour})" if self.target_colour else ""
            print(f"[fsm] {self.now():7.2f}s  {self.state.name} -> {new_state.name}{target}")
        self.state = new_state
        self.state_started = self.now()

    def time_in_state(self) -> float:
        return self.now() - self.state_started

    def _start_approach_if_wanted(self) -> bool:
        wanted = self.wanted_detection()
        if wanted is None:
            return False
        self.target_colour = wanted.colour
        self.frames_without_target = 0
        self.change_state(State.APPROACH)
        return True

    def _step_search(self) -> None:
        if self._start_approach_if_wanted():
            return
        left_blocked, right_blocked = self.obstacle_sides()
        if left_blocked or right_blocked:
            self.avoid_obstacle(left_blocked, right_blocked)
        elif self.now() - self.last_wanted_seen > SCAN_AFTER_SECONDS:
            self.change_state(State.SCAN)
        else:
            self.drive(CRUISE, CRUISE)

    def _step_scan(self) -> None:
        if self._start_approach_if_wanted():
            return
        self.drive(-SCAN_SPEED, SCAN_SPEED)
        if self.time_in_state() > self.full_turn_seconds(SCAN_SPEED):
            self.last_wanted_seen = self.now()  # restart the scan timer
            self.change_state(State.SEARCH)

    def _step_approach(self) -> None:
        target = self.target_detection()
        if target is None:
            if self.new_frame:
                self.frames_without_target += 1
            if self.frames_without_target > LOST_FRAMES:
                print(f"[fsm] lost sight of the {self.target_colour} block")
                self.target_colour = None
                self.change_state(State.SEARCH)
            else:
                self.drive(CREEP_SPEED, CREEP_SPEED)
            return
        self.frames_without_target = 0

        left_blocked, right_blocked = self.obstacle_sides()
        front_blocked = left_blocked and right_blocked
        if target.area_fraction >= ARRIVED_AREA or (
                front_blocked and target.area_fraction >= BUMP_ARRIVAL_AREA):
            self.stop()
            self.change_state(State.INSPECT)
            self.record_encounter(target)
            return
        if left_blocked or right_blocked:
            # Something else (wall, other block) is in the way: go round it.
            self.avoid_obstacle(left_blocked, right_blocked)
            return
        offset = target.horizontal_offset(self.width)  # -1 left .. +1 right
        turn = STEER_GAIN * offset
        # Slow down when the block is off to the side so we turn before driving.
        forward = APPROACH_SPEED * (1.0 - 0.5 * min(1.0, abs(offset)))
        self.drive(forward + turn, forward - turn)

    def _step_inspect(self) -> None:
        self.stop()
        if self.time_in_state() >= INSPECT_SECONDS:
            self.cooldown_until[self.target_colour] = self.now() + COLOUR_COOLDOWN_SECONDS
            self.change_state(State.LEAVE)

    def _step_leave(self) -> None:
        t = self.time_in_state()
        if t < REVERSE_SECONDS and not self._blocked(REAR_SENSORS):
            self.drive(-CRUISE, -CRUISE)
        elif t < REVERSE_SECONDS + TURN_AWAY_SECONDS:
            self.drive(CRUISE, -CRUISE)
        else:
            self.target_colour = None
            self.last_wanted_seen = self.now()
            self.change_state(State.SEARCH)

    # ------------------------------------------------------------------ capture
    def capture(self, tag_prefix: str, detection: vision.Detection | None) -> tuple[str, bool] | None:
        """Save raw + annotated frame and send a square crop to the CNN.

        Returns (tag, images_saved), or None if there is no frame yet. A
        failed save is reported but never stops the robot.
        """
        if self.last_frame is None:
            print("[capture] no camera frame available yet")
            return None
        self.capture_count += 1
        tag = f"{tag_prefix}_{self.capture_count:03d}"
        raw_path = IMAGE_DIR / f"{tag}.png"  # PNG: lossless, no JPEG colour artefacts
        try:
            annotated = vision.annotate(self.last_frame, self.last_detections, tag)
            crop = vision.square_crop(self.last_frame, detection.bbox if detection else None)
            cnn_input = cv.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv.INTER_AREA)
            saved = all([
                cv.imwrite(str(raw_path), self.last_frame),
                cv.imwrite(str(IMAGE_DIR / f"{tag}_annotated.png"), annotated),
                cv.imwrite(str(IMAGE_DIR / f"{tag}_cnn_input.png"), cnn_input),
            ])
        except (cv.error, OSError) as exc:
            print(f"[capture] FAILED for {tag}: {type(exc).__name__}: {exc}")
            return tag, False
        print(f"[capture] saved {raw_path.name}" if saved
              else f"[capture] FAILED to write images for {tag} in {IMAGE_DIR}")
        self.classifier.submit(tag, crop)
        return tag, saved

    def record_encounter(self, det: vision.Detection) -> None:
        print(f"[found] {det.colour.upper()} block at {self.now():.1f}s "
              f"(fills {det.area_fraction:.0%} of the image)")
        result = self.capture(f"{det.colour}_block", det)
        tag, saved = result if result else (None, False)
        # The block WAS found (detection succeeded) even if saving the photo failed;
        # the CSV records an empty image path in that case so nothing is hidden.
        self.encounters.append(Encounter(det.colour, self.now(), tag))
        self.log.block_found(self.now(), det, f"{tag}.png" if saved else "")
        self.found_this_round.add(det.colour)
        if self.found_this_round >= set(vision.HSV_RANGES):
            print(f"[found] round {self.round_number} complete: all colours found "
                  f"in {self.now():.1f}s - starting a new round")
            self.print_summary()
            self.found_this_round.clear()
            self.round_number += 1

    def handle_predictions(self) -> None:
        for pred in self.classifier.poll():
            if pred.error:
                print(f"[cnn] {pred.tag}: inference failed - {pred.error}")
                self.log.error(self.now(), pred.tag, pred.error)
                continue
            top = ", ".join(f"{name} {p:.0%}" for name, p in pred.top3)
            print(f"[cnn] {pred.tag}: {pred.label} ({pred.confidence:.0%})  "
                  f"top-3: {top}  [{pred.latency_ms:.1f} ms, off the control loop]")
            self.predictions[pred.tag] = f"{pred.label} ({pred.confidence:.0%})"
            self.log.prediction(self.now(), pred.tag, pred.label, pred.confidence)

    def handle_keyboard(self) -> None:
        key = self.keyboard.getKey()
        while key != -1:
            if (key & Keyboard.KEY) in (ord("C"), ord("c")):  # ignore Shift/Ctrl bits
                self.capture("manual", self.last_detections[0] if self.last_detections else None)
            key = self.keyboard.getKey()

    # ------------------------------------------------------------------ summary
    def print_summary(self) -> None:
        counts = {c: 0 for c in vision.HSV_RANGES}
        for e in self.encounters:
            counts[e.colour] += 1
        avg_vision = self.vision_ms_total / self.vision_calls if self.vision_calls else 0.0
        missing = [c for c, n in counts.items() if n == 0]
        print("=" * 60)
        print(f"SUMMARY at {self.now():.1f}s simulated time (round {self.round_number})")
        print("  blocks found : " + ", ".join(f"{c} x{n}" for c, n in counts.items()))
        print("  not found yet: " + (", ".join(missing) if missing else "none - all colours found"))
        print(f"  captures     : {self.capture_count} (saved in {IMAGE_DIR.name}/)")
        print(f"  vision cost  : {avg_vision:.2f} ms per frame over {self.vision_calls} frames")
        print(f"  CNN          : {'ON' if self.classifier.enabled else 'OFF'} - "
              f"{self.classifier.description}")
        for tag, pred in self.predictions.items():
            print(f"    {tag:<22}: {pred}")
        if self.classifier.dropped_frames or self.classifier.failures:
            print(f"    dropped {self.classifier.dropped_frames} stale frames, "
                  f"{self.classifier.failures} failures")
        print("=" * 60)

    # --------------------------------------------------------------------- main
    def run(self) -> None:
        step_count = 0
        try:
            while self.robot.step(TIME_STEP) != -1:
                step_count += 1
                self.update_vision(step_count)
                if not self.startup_snapshot_done and self.last_frame is not None:
                    # One photo at start-up: in cnn_camera_test.wbt the robot
                    # faces a CIFAR-10 object (a dog) for the CNN camera test.
                    self.startup_snapshot_done = True
                    self.capture("startup", None)
                self.handle_keyboard()
                self.handle_predictions()
                self.state_handlers[self.state]()
                if self.now() >= self.next_summary:
                    self.print_summary()
                    self.next_summary += SUMMARY_EVERY_SECONDS
        finally:
            try:
                self.stop()
                self.handle_predictions()
                self.print_summary()
            finally:
                self.log.close()


if __name__ == "__main__":
    ColourDetectorRobot().run()
