"""Plot the inference-timing experiment (report Section 4.7).

Usage: python plot_timing.py <folder with the run JSON files> <output.png>
Also writes summary.csv in that folder.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RUNS = [
    ("A_async_event_real", "A  Deployed: background thread, on events"),
    ("B_sync_frame_real", "B  In the loop, every frame"),
    ("C_async_event_50", "C  Deployed design, 50 ms CNN"),
    ("D_async_frame_50", "D  Background thread, every frame, 50 ms CNN"),
    ("E_sync_event_50", "E  In the loop, on events, 50 ms CNN"),
    ("F_sync_frame_50", "F  In the loop, every frame, 50 ms CNN"),
]
FIELDS = ["steps", "mean_ms", "p99_ms", "max_ms", "missed_deadlines", "missed_pct", "dropped_frames", "blocks_found"]
OK_COLOUR, LATE_COLOUR, DEADLINE_COLOUR = "#2A7F8E", "#C0392B", "#F28C28"


def load(folder: Path) -> list[tuple[str, dict]]:
    rows = []
    for key, label in RUNS:
        with open(folder / f"{key}.json") as fh:
            rows.append((label, json.load(fh)))
    return rows


def write_summary(folder: Path, rows: list[tuple[str, dict]]) -> None:
    with open(folder / "summary.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", "file", "cnn_mode", "cnn_trigger", "emulate_ms"] + FIELDS)
        for (key, _), (label, d) in zip(RUNS, rows):
            o = d["options"]
            writer.writerow([label[:1], f"{key}.json", o["cnn_mode"], o["cnn_trigger"], o["emulate_ms"]]
                            + [d[f] for f in FIELDS])


def plot(rows: list[tuple[str, dict]], out: str) -> None:
    labels = [label for label, _ in rows][::-1]
    data = [d for _, d in rows][::-1]
    y = range(len(rows))
    colours = [LATE_COLOUR if d["missed_deadlines"] else OK_COLOUR for d in data]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]}, dpi=170)

    left.barh([i + 0.2 for i in y], [d["max_ms"] for d in data], height=0.4, color=colours, alpha=0.45)
    left.barh([i - 0.2 for i in y], [d["p99_ms"] for d in data], height=0.4, color=colours)
    left.axvline(32, color=DEADLINE_COLOUR, lw=2, ls="--")
    left.text(34, len(rows) - 0.62, "32 ms deadline", color=DEADLINE_COLOUR, fontsize=9)
    left.set_xscale("log")
    left.set_yticks(list(y))
    left.set_yticklabels(labels, fontsize=8.5)
    left.set_xlabel("Controller compute time per 32 ms step (ms, log scale)")
    legend = [Patch(color="#777777", alpha=0.45, label="worst step"),
              Patch(color="#555555", label="99th percentile step"),
              Patch(color=OK_COLOUR, label="no missed deadline"),
              Patch(color=LATE_COLOUR, label="missed deadlines")]
    left.legend(handles=legend, fontsize=8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, frameon=False)

    right.barh(list(y), [d["missed_pct"] for d in data], color=colours)
    for i, d in zip(y, data):
        right.text(d["missed_pct"] + 1, i, f"{d['missed_deadlines']} of {d['steps']}", va="center", fontsize=8)
    right.set_yticks([])
    right.set_xlim(0, 70)
    right.set_xlabel("Control steps that missed the deadline (%)")

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)


def main() -> None:
    folder, out = Path(sys.argv[1]), sys.argv[2]
    rows = load(folder)
    write_summary(folder, rows)
    plot(rows, out)
    print((folder / "summary.csv").read_text())


if __name__ == "__main__":
    main()
