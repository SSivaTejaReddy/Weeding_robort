"""Generate per-epoch CV reports (precision, recall, F1, mAP) from Ultralytics `results.csv`.

Writes a timestamped CSV (and JSON) into `runs/ai_validation/<run>/reports/` for each run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime
import argparse
import math


def compute_f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def process_run(run_dir: Path, out_dir: Path) -> None:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print(f"No results.csv in {run_dir}, skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_csv = out_dir / f"metrics_{timestamp}.csv"
    out_json = out_dir / f"metrics_{timestamp}.json"

    rows = []
    with results_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # try several possible header names used by ultralytics
            try:
                p = float(r.get("metrics/precision(B)") or r.get("metrics/precision") or 0)
                rec = float(r.get("metrics/recall(B)") or r.get("metrics/recall") or 0)
            except ValueError:
                p = 0.0
                rec = 0.0

            # ensure in-range
            p = max(0.0, min(1.0, p))
            rec = max(0.0, min(1.0, rec))
            f1 = compute_f1(p, rec)

            # keep mAP fields if present
            map50 = float(r.get("metrics/mAP50(B)") or r.get("metrics/mAP50") or 0)
            map5095 = float(r.get("metrics/mAP50-95(B)") or r.get("metrics/mAP50-95") or 0)

            epoch = int(r.get("epoch") or r.get("Epoch") or 0)
            time_sec = float(r.get("time") or 0)

            row = {
                "epoch": epoch,
                "time_sec": time_sec,
                "precision": p,
                "recall": rec,
                "f1": f1,
                "mAP50": map50,
                "mAP50-95": map5095,
            }
            rows.append(row)

    # write CSV
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write JSON
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump({"generated_at": timestamp, "rows": rows}, fh, indent=2)

    print(f"Wrote reports: {out_csv} and {out_json}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/ai_validation", help="Root runs dir")
    args = parser.parse_args()

    runs_root = Path(args.runs_dir)
    if not runs_root.exists():
        print(f"Runs directory does not exist: {runs_root}")
        return 1

    for run in sorted(runs_root.iterdir()):
        if not run.is_dir():
            continue
        reports_dir = run / "reports"
        process_run(run, reports_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
