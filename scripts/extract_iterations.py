#!/usr/bin/env python3
"""Build run/iterations.csv from the per-iteration check files of run 2, session 1.

Reads run/checks/train{N}.json and run/checks/val{N}.json (N = 1..13), computes
minutes since the session start (2026-11-03T16:20:45, local time, the `codex exec`
launch) from the val check timestamp, pulls the `## Iteration N` heading from
run/NOTES.md as a free-text note, and attaches a hard-coded short label of the
primitive the agent added at each iteration.

Columns: iteration,train_acc,val_acc,minutes,primitive,note
Accuracies are percent with one decimal; minutes have one decimal.

Usage (from the repo root):
    python3 scripts/extract_iterations.py            # writes run/iterations.csv
    python3 scripts/extract_iterations.py --print    # CSV to stdout instead
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKS = ROOT / "run" / "checks"
NOTES = ROOT / "run" / "NOTES.md"
OUT = ROOT / "run" / "iterations.csv"

SESSION_START = datetime(2026, 11, 3, 16, 20, 45)  # local time of `codex exec` launch
N_ITER = 13

# Short label of what the agent added at each iteration (it. 9 was reverted).
PRIMITIVES = {
    1: "colour masks, texture, silhouettes, straight lines",
    2: "circle/ring tests, tyre pairs, shape sketches",
    3: "surface context, road, roof lines",
    4: "orientation, polygon panels, ridges",
    5: "stricter object support",
    6: "arched windows, spheres, pump/cassette layouts",
    7: "blade edges, circular texture",
    8: "steeples, crosses, perspective",
    9: "exposure/contrast normalisation (reverted)",
    10: "brass holes, speaker pair, saw silhouette",
    11: "rewrite into one coherent rule set",
    12: "shadow-tolerant tyre arcs",
    13: "canopy sketch, lighter tyre weight",
}


def load_notes_headings(path: Path) -> dict:
    """Map iteration number -> heading text (after 'Iteration N:') from NOTES.md."""
    notes = {}
    if not path.exists():
        return notes
    pat = re.compile(r"^#{1,6}\s*Iteration\s+(\d+)\s*[:\-–—]?\s*(.*?)\s*$", re.IGNORECASE)
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            n = int(m.group(1))
            notes.setdefault(n, m.group(2))  # first matching heading wins
    return notes


def accuracy_pct(path: Path) -> float:
    with path.open(encoding="utf-8") as f:
        return round(100.0 * json.load(f)["accuracy"], 1)


def minutes_since_start(path: Path) -> float:
    with path.open(encoding="utf-8") as f:
        ts = datetime.fromisoformat(json.load(f)["timestamp"])
    return round((ts - SESSION_START).total_seconds() / 60.0, 1)


def build_rows():
    notes = load_notes_headings(NOTES)
    rows = []
    for n in range(1, N_ITER + 1):
        train = CHECKS / f"train{n}.json"
        val = CHECKS / f"val{n}.json"
        rows.append(
            {
                "iteration": n,
                "train_acc": f"{accuracy_pct(train):.1f}",
                "val_acc": f"{accuracy_pct(val):.1f}",
                "minutes": f"{minutes_since_start(val):.1f}",
                "primitive": PRIMITIVES[n],
                "note": notes.get(n, ""),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true", help="write CSV to stdout instead of run/iterations.csv")
    args = ap.parse_args()

    rows = build_rows()
    fields = ["iteration", "train_acc", "val_acc", "minutes", "primitive", "note"]

    if args.print:
        w = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        return

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
