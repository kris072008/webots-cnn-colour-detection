import json, sys, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = Path(sys.argv[1]); out = sys.argv[2]
runs = [("A_async_event_real", "A  Deployed: background thread, on events"),
        ("B_sync_frame_real", "B  In the loop, every frame"),
        ("C_async_event_50", "C  Deployed design, 50 ms CNN"),
        ("D_async_frame_50", "D  Background thread, every frame, 50 ms CNN"),
        ("E_sync_event_50", "E  In the loop, on events, 50 ms CNN"),
        ("F_sync_frame_50", "F  In the loop, every frame, 50 ms CNN")]
rows = []
with open(T / "summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["config", "file", "cnn_mode", "cnn_trigger", "emulate_ms", "steps", "mean_ms", "p99_ms", "max_ms", "missed_deadlines", "missed_pct", "dropped_frames", "blocks_found"])
    for key, label in runs:
        d = json.load(open(T / f"{key}.json")); o = d["options"]
        rows.append((label, d))
        w.writerow([label[:1], key + ".json", o["cnn_mode"], o["cnn_trigger"], o["emulate_ms"], d["steps"], d["mean_ms"], d["p99_ms"], d["max_ms"], d["missed_deadlines"], d["missed_pct"], d["dropped_frames"], d["blocks_found"]])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]}, dpi=170)
labels = [r[0] for r in rows][::-1]; ds = [r[1] for r in rows][::-1]
y = range(len(rows))
bad = ["#C0392B" if d["missed_deadlines"] else "#2A7F8E" for d in ds]
a1.barh([i + 0.2 for i in y], [d["max_ms"] for d in ds], height=0.4, color=bad, alpha=0.45, label="worst step")
a1.barh([i - 0.2 for i in y], [d["p99_ms"] for d in ds], height=0.4, color=bad, label="99th percentile step")
a1.axvline(32, color="#F28C28", lw=2, ls="--"); a1.text(34, len(rows) - 0.62, "32 ms deadline", color="#F28C28", fontsize=9)
a1.set_xscale("log"); a1.set_yticks(list(y)); a1.set_yticklabels(labels, fontsize=8.5)
a1.set_xlabel("Controller compute time per 32 ms step (ms, log scale)"); from matplotlib.patches import Patch
a1.legend(handles=[Patch(color="#777777", alpha=0.45, label="worst step"), Patch(color="#555555", label="99th percentile step"), Patch(color="#2A7F8E", label="no missed deadline"), Patch(color="#C0392B", label="missed deadlines")], fontsize=8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, frameon=False)
a2.barh(list(y), [d["missed_pct"] for d in ds], color=bad)
for i, d in zip(y, ds):
    a2.text(d["missed_pct"] + 1, i, f"{d['missed_deadlines']} of {d['steps']}", va="center", fontsize=8)
a2.set_yticks([]); a2.set_xlim(0, 70); a2.set_xlabel("Control steps that missed the deadline (%)")
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(out)
print(open(T / "summary.csv").read())
