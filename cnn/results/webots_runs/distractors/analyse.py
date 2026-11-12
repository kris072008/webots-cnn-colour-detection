"""Score the robot in the distractor world, with and without the shape check
(report Section 4.8).

Each run (seed S, mode on|off) has seedS_<mode>_encounters.csv (what the robot
logged: HSV candidates, both CNN labels, and its confirm/reject decisions) and
seedS_<mode>_ground_truth.csv (what the supervisor says the robot was really
facing). This joins them and reports, per mode:
  * distractors wrongly accepted as blocks,
  * real blocks wrongly rejected,
  * the first "round complete" (all three colours believed found): did it
    happen, were all three counted objects real blocks, and when,
  * how often the shape CNN's label was right.
It writes inspections.csv (one row per inspection) and summary.csv.

Usage: python analyse.py
"""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = re.compile(r"seed(\d+)_(on|off)_ground_truth\.csv")


def load_run(seed: str, mode: str) -> list[dict]:
    truth = {r["tag"]: r for r in csv.DictReader(open(HERE / f"seed{seed}_{mode}_ground_truth.csv"))}
    rows = list(csv.DictReader(open(HERE / f"seed{seed}_{mode}_encounters.csv")))
    by_tag: dict[str, dict] = {}
    for r in rows:
        tag = r["image"].replace(".png", "")
        if r["event"] == "cnn_prediction":
            by_tag.setdefault(tag, {})["cifar"] = r["cnn_label"]
        elif r["event"] == "shape_prediction":
            by_tag.setdefault(tag, {})["shape"] = (r["cnn_label"], float(r["cnn_confidence"]))
        elif r["event"] in ("block_confirmed", "rejected_not_block"):
            by_tag.setdefault(tag, {})["decision"] = r["event"]
    out = []
    for r in rows:
        if r["event"] != "block_found":
            continue
        tag = r["image"].replace(".png", "")
        t = truth.get(tag)
        if t is None:
            print(f"WARNING: seed {seed} {mode} {tag}: no ground truth - skipped")
            continue
        info = by_tag.get(tag, {})
        out.append(dict(seed=seed, mode=mode, time=float(r["sim_time_s"]), tag=tag, colour=r["colour"],
                        truth=t["true_object"], cifar=info.get("cifar", ""),
                        shape=info.get("shape", ("", 0.0)), decision=info.get("decision", "")))
    return out


def rounds(run: list[dict]) -> list[tuple[float, bool]]:
    """Each time the robot completed a round (it believed all three colours were
    found): (time, whether all three things it counted really were blocks)."""
    out, counted = [], {}
    for i in run:
        if i["decision"] != "block_confirmed":
            continue
        counted[i["colour"]] = i["truth"] == "block"
        if len(counted) == 3:
            out.append((i["time"], all(counted.values())))
            counted = {}
    return out


def summarise(runs: dict[tuple[str, str], list[dict]]) -> list[list]:
    table = []
    for mode in ("off", "on"):
        keys = sorted(k for k in runs if k[1] == mode)
        if not keys:
            continue
        inspections = [i for k in keys for i in runs[k]]
        decided = [i for i in inspections if i["decision"]]
        accepted = [i for i in decided if i["decision"] == "block_confirmed"]
        wrong_accept = sum(i["truth"] != "block" for i in accepted)
        wrong_reject = sum(i["truth"] == "block" and i["decision"] == "rejected_not_block" for i in decided)
        first = [rounds(runs[k])[0] if rounds(runs[k]) else None for k in keys]
        completed = [f for f in first if f]
        correct = [f for f in completed if f[1]]
        scored = [i for i in inspections if i["truth"] != "nothing" and i["shape"][0]]
        shape_ok = sum(i["shape"][0] == i["truth"] for i in scored)
        table.append([mode, len(keys), len(decided), sum(i["truth"] != "block" for i in decided),
                      wrong_accept, wrong_reject, len(completed), len(correct),
                      round(statistics.mean(f[0] for f in correct), 1) if correct else "",
                      f"{shape_ok}/{len(scored)}"])
    return table


def main() -> None:
    runs = {}
    for p in sorted(HERE.glob("seed*_ground_truth.csv")):
        m = RUN.fullmatch(p.name)
        if m:
            runs[m.groups()] = load_run(*m.groups())
    if not runs:
        raise SystemExit("no runs found: run run_distractor_test.sh first")
    with open(HERE / "inspections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "shape_check", "sim_time_s", "tag", "hsv_colour", "true_object", "cifar_label",
                    "shape_label", "shape_conf", "decision"])
        for key in sorted(runs):
            for i in runs[key]:
                w.writerow([i["seed"], i["mode"], i["time"], i["tag"], i["colour"], i["truth"], i["cifar"],
                            i["shape"][0], f"{i['shape'][1]:.3f}", i["decision"]])
    header = ["shape_check", "runs", "inspections", "of_them_distractors", "distractors_counted_as_blocks",
              "real_blocks_rejected", "runs_completing_round_1", "round_1_all_real_blocks",
              "mean_time_correct_round_1_s", "shape_label_correct"]
    table = summarise(runs)
    with open(HERE / "summary.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([header] + table)
    for row in table:
        print(dict(zip(header, row)))
    for key in sorted(runs):
        for i in runs[key]:
            print(f"  seed {i['seed']} {i['mode']:<3} {i['time']:7.1f} s  HSV {i['colour']:<5} truly {i['truth']:<8} "
                  f"shape {i['shape'][0]:<10} {i['shape'][1]:4.0%}  CIFAR {i['cifar']:<10} {i['decision']}")


if __name__ == "__main__":
    main()
