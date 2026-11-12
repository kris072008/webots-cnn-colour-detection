"""Unit tests for the inference-timing experiment (no Webots needed)."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "controllers" / "colour_detector"))
import benchmark  # noqa: E402
from classifier import CLASS_NAMES, AsyncClassifier  # noqa: E402


def test_default_options_are_normal_operation():
    opts = benchmark.parse_options([""])  # Webots passes "" for empty controllerArgs
    assert opts == benchmark.Options() and not opts.is_benchmark


def test_parse_all_options():
    opts = benchmark.parse_options(
        ["--cnn-mode=sync", "--cnn-trigger=every-frame", "--emulate-ms=50", "--stop-after=120"])
    assert opts == benchmark.Options("sync", "every-frame", 50.0, 120.0) and opts.is_benchmark


@pytest.mark.parametrize("bad", [
    ["--cnn-mode=fast"], ["--cnn-trigger=always"], ["--emulate-ms=-1"],
    ["--stop-after=0"], ["--colour=red"], ["42"], ["--emulate-ms=abc"],
    ["--emulate-ms=nan"], ["--stop-after=inf"], ["--cnn-mode=sync", "--cnn-mode=async"],
])
def test_bad_options_are_rejected(bad):
    with pytest.raises(ValueError):
        benchmark.parse_options(bad)


def test_step_timing_stats_counts_missed_deadlines():
    stats = benchmark.step_timing_stats([1.0, 2.0, 40.0, 3.0], deadline_ms=32, wall_s=2.0, sim_s=0.128)
    assert stats["steps"] == 4 and stats["missed_deadlines"] == 1 and stats["missed_pct"] == 25.0
    assert stats["max_ms"] == 40.0 and stats["mean_ms"] == 11.5
    assert stats["sustainable_rate_hz"] == pytest.approx(1000 / 11.5, abs=0.1)


def test_step_timing_stats_rejects_empty_input():
    with pytest.raises(ValueError):
        benchmark.step_timing_stats([], 32, 1.0, 1.0)


def test_emulated_latency_adds_delay_and_keeps_output():
    logits = np.arange(10, dtype=np.float32)
    slow = benchmark.EmulatedLatencyBackend(lambda batch: logits, extra_ms=30)
    start = time.perf_counter()
    out = slow(np.zeros((1, 32, 32, 3), np.float32))
    assert (time.perf_counter() - start) >= 0.029 and np.array_equal(out, logits)


def test_classify_now_blocks_and_returns_result_via_poll():
    logits = np.zeros(10, np.float32)
    logits[2] = 5.0  # "bird"
    clf = AsyncClassifier(Path("."), backend=benchmark.EmulatedLatencyBackend(lambda b: logits, 20))
    start = time.perf_counter()
    assert clf.classify_now("s1", np.full((10, 10, 3), 128, np.uint8))
    assert (time.perf_counter() - start) >= 0.019  # ran on this thread
    (pred,) = clf.poll()
    assert pred.tag == "s1" and pred.label == CLASS_NAMES[2] and not pred.error
