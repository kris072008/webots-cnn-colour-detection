"""Timing experiment: does CNN inference disturb the 32 ms control loop?

The controller runs normally unless benchmark options are passed in the
e-puck's ``controllerArgs`` field, for example::

    --cnn-mode=sync --cnn-trigger=every-frame --emulate-ms=50 --stop-after=120

Options
-------
--cnn-mode=async|sync        async (default): inference on the background thread.
                             sync: inference inside the control step (the naive way).
--cnn-trigger=event|every-frame
                             event (default): only at a block, key C or start-up.
                             every-frame: classify every processed camera frame.
--emulate-ms=N               add N ms to every inference, to emulate a slow
                             embedded CPU (default 0 = the real model speed).
--stop-after=S               stop after S simulated seconds and write the timing
                             report (default: run until Webots stops the robot).

The controller measures the wall-clock compute time of every control step
(from ``robot.step()`` returning to the next call), and writes a summary to
``images/step_timing.json``. A step whose compute time exceeds the 32 ms step
is a missed deadline: on a real robot its motor commands and obstacle checks
would be late. Webots pauses the simulation while the controller computes, so
in simulation a slow step lowers the real-time rate instead of changing what
the robot does; both numbers are reported.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

Backend = Callable[[np.ndarray], np.ndarray]

CNN_MODES = ("async", "sync")
CNN_TRIGGERS = ("event", "every-frame")


@dataclass(frozen=True)
class Options:
    cnn_mode: str = "async"
    cnn_trigger: str = "event"
    emulate_ms: float = 0.0
    stop_after: float | None = None

    @property
    def is_benchmark(self) -> bool:
        return self != Options()


def parse_options(argv: list[str]) -> Options:
    """Parse ``--key=value`` controller arguments. Raises ValueError on bad input."""
    values: dict[str, str] = {}
    for arg in argv:
        if not arg.strip():
            continue  # Webots passes an empty string when controllerArgs is empty
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"controller argument {arg!r} is not of the form --key=value")
        key, value = arg[2:].split("=", 1)
        if key in values:
            raise ValueError(f"controller argument --{key} given more than once")
        values[key] = value

    unknown = set(values) - {"cnn-mode", "cnn-trigger", "emulate-ms", "stop-after"}
    if unknown:
        raise ValueError(f"unknown controller argument(s): {', '.join(sorted(unknown))}")

    mode = values.get("cnn-mode", "async")
    if mode not in CNN_MODES:
        raise ValueError(f"--cnn-mode must be one of {CNN_MODES}, not {mode!r}")
    trigger = values.get("cnn-trigger", "event")
    if trigger not in CNN_TRIGGERS:
        raise ValueError(f"--cnn-trigger must be one of {CNN_TRIGGERS}, not {trigger!r}")
    emulate_ms = _number(values, "emulate-ms", "0")
    if emulate_ms < 0:
        raise ValueError("--emulate-ms must be >= 0")
    stop_after = _number(values, "stop-after", None)
    if stop_after is not None and stop_after <= 0:
        raise ValueError("--stop-after must be > 0")
    return Options(mode, trigger, emulate_ms, stop_after)


def _number(values: dict[str, str], key: str, default: str | None) -> float | None:
    """A finite float option, or None if absent and default is None."""
    text = values.get(key, default)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"--{key} must be a number, not {text!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"--{key} must be a finite number, not {text!r}")
    return number


class EmulatedLatencyBackend:
    """Wraps a model backend and adds a fixed delay, to emulate a slow CPU.

    The real forward pass still runs, so predictions are unchanged.
    """

    def __init__(self, backend: Backend, extra_ms: float) -> None:
        self._backend = backend
        self._extra_s = extra_ms / 1000.0

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        result = self._backend(batch)
        time.sleep(self._extra_s)
        return result


def step_timing_stats(step_ms: list[float], deadline_ms: float,
                      wall_s: float, sim_s: float) -> dict[str, float | int | None]:
    """Summarise per-step compute times against the control-step deadline."""
    if not step_ms:
        raise ValueError("no control steps were timed")
    t = np.asarray(step_ms, dtype=float)
    missed = int((t > deadline_ms).sum())
    return {
        "steps": int(t.size),
        "deadline_ms": deadline_ms,
        "mean_ms": round(float(t.mean()), 3),
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p99_ms": round(float(np.percentile(t, 99)), 3),
        "max_ms": round(float(t.max()), 3),
        "missed_deadlines": missed,
        "missed_pct": round(100.0 * missed / t.size, 2),
        "sim_seconds": round(sim_s, 2),
        "wall_seconds": round(wall_s, 2),
        # Highest control rate the controller's own computation could sustain
        # (1000 / mean step time); 31.25 Hz is needed for a 32 ms step.
        "sustainable_rate_hz": round(1000.0 / float(t.mean()), 1) if t.mean() > 0 else None,
    }
