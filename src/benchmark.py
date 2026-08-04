"""Baselines to beat: $/mile heuristics on the time holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import time_split
from metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "train-test.csv"
HOLDOUT_START = "2025-09-01"


def global_rpm_baseline(train: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    rpm = (train["posted_rate"] / train["distance"]).mean()
    return holdout["distance"].to_numpy() * rpm


def equipment_rpm_baseline(train: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    rpm = train.assign(_rpm=train["posted_rate"] / train["distance"]).groupby("equipment")[
        "_rpm"
    ].mean()
    global_rpm = (train["posted_rate"] / train["distance"]).mean()
    mapped = holdout["equipment"].map(rpm).fillna(global_rpm)
    return holdout["distance"].to_numpy() * mapped.to_numpy()


def run_benchmark(data_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_path)
    train, holdout = time_split(raw, HOLDOUT_START)
    y = holdout["posted_rate"].to_numpy()
    print(f"Train rows: {len(train):,} | Holdout rows: {len(holdout):,}")
    print(f"Holdout window: {holdout['date'].min()} → {holdout['date'].max()}")

    results = []
    predictors = [
        ("global_rpm", global_rpm_baseline),
        ("equipment_rpm", equipment_rpm_baseline),
    ]
    for name, fn in predictors:
        pred = np.clip(fn(train, holdout), 1.0, None)
        m = regression_metrics(y, pred)
        m["model"] = name
        results.append(m)
        print(
            f"{name:16s}  MAE={m['MAE']:8.2f}  RMSE={m['RMSE']:8.2f}  "
            f"MAPE={m['MAPE']:6.2f}%  MedAE={m['MedAE']:7.2f}"
        )

    table = pd.DataFrame(results)[["model", "MAE", "RMSE", "MAPE", "MedAE"]]
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "benchmark_metrics.csv", index=False)
    (out_dir / "benchmark_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_dir / 'benchmark_metrics.csv'}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run freight rate benchmarks")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    run_benchmark(args.data)


if __name__ == "__main__":
    main()
