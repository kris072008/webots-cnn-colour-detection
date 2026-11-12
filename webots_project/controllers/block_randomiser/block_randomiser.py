"""Supervisor: place the three colour blocks at random, valid positions.

Runs once at the start of the simulation, then exits. Each block is placed
inside the arena, at least MIN_GAP from every other block and at least
ROBOT_CLEARANCE from the e-puck's start position, so every run is a fresh,
fair search problem.

Pass a seed in the Robot node's controllerArgs (e.g. "42") to repeat a layout
exactly; with no seed, the layout changes every run.
"""

from __future__ import annotations

import math
import random
import sys

from controller import Supervisor

BLOCK_DEFS = ("RED_BLOCK", "GREEN_BLOCK", "BLUE_BLOCK")
# Optional same-coloured non-block objects (worlds/colour_blocks_distractors.wbt).
DISTRACTOR_DEFS = ("DISTRACTOR_1", "DISTRACTOR_2", "DISTRACTOR_3")
ARENA_HALF = 1.5  # 3 x 3 m arena
BLOCK_HALF_DIAGONAL = math.hypot(0.15, 0.15)  # 0.3 x 0.3 m footprint
WALL_MARGIN = BLOCK_HALF_DIAGONAL + 0.1  # keep a gap so the robot can pass
MIN_GAP = 0.7  # centre-to-centre distance between blocks
ROBOT_CLEARANCE = 0.6  # distance from the robot's start position
BLOCK_Z = 0.05  # half the block height: resting on the floor
MAX_TRIES = 1000


def random_layout(rng: random.Random, robot_xy: tuple[float, float],
                  count: int = len(BLOCK_DEFS)) -> list[tuple[float, float]]:
    """Rejection sampling: draw random (x, y) points until each object is inside
    the walls, far enough from the robot and from every object already placed."""
    limit = ARENA_HALF - WALL_MARGIN
    placed: list[tuple[float, float]] = []
    for _ in range(MAX_TRIES):
        p = (rng.uniform(-limit, limit), rng.uniform(-limit, limit))
        if math.dist(p, robot_xy) < ROBOT_CLEARANCE:
            continue
        if any(math.dist(p, q) < MIN_GAP for q in placed):
            continue
        placed.append(p)
        if len(placed) == count:
            return placed
    raise RuntimeError("could not find a valid block layout; reduce MIN_GAP")


def place_objects(supervisor: Supervisor, rng: random.Random,
                  seed: int | None = None) -> dict[str, tuple[float, float]]:
    """Put the blocks (and any distractors in the world) at a random valid
    layout. Returns {DEF name: (x, y)} of everything placed."""
    robot_node = supervisor.getFromDef("EPUCK")
    robot_xy = (0.0, 0.0)
    if robot_node is None:
        print("[randomiser] WARNING: no node with DEF EPUCK - assuming the robot starts at (0, 0)")
    else:
        robot_xy = tuple(robot_node.getField("translation").getSFVec3f()[:2])

    distractors = [d for d in DISTRACTOR_DEFS if supervisor.getFromDef(d) is not None]
    names = list(BLOCK_DEFS) + distractors
    layout = random_layout(rng, robot_xy, len(names))
    placed: dict[str, tuple[float, float]] = {}
    for def_name, (x, y) in zip(names, layout):
        node = supervisor.getFromDef(def_name)
        if node is None:
            print(f"[randomiser] WARNING: no node with DEF {def_name} in the world")
            continue
        z = BLOCK_Z if def_name in BLOCK_DEFS else node.getField("translation").getSFVec3f()[2]
        node.getField("translation").setSFVec3f([x, y, z])
        node.getField("rotation").setSFRotation([0, 0, 1, rng.uniform(0, math.pi)])
        node.resetPhysics()
        placed[def_name] = (x, y)
        print(f"[randomiser] {def_name:<12} -> x={x:+.2f} m, y={y:+.2f} m")
    print(f"[randomiser] placed {len(placed)} of {len(names)} objects (seed={seed})")
    return placed


def main() -> None:
    supervisor = Supervisor()
    seed = None
    if len(sys.argv) > 1 and sys.argv[1].strip():
        try:
            seed = int(sys.argv[1])
        except ValueError:
            print(f"[randomiser] WARNING: controllerArgs {sys.argv[1]!r} is not an integer "
                  "seed - using a random layout")
    rng = random.Random(seed)

    place_objects(supervisor, rng, seed)
    supervisor.step(int(supervisor.getBasicTimeStep()))


if __name__ == "__main__":
    main()
