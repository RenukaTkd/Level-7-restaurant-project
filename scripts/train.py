"""End-to-end reproducible pipeline.

    python scripts/train.py            # full run (12 folds)
    python scripts/train.py --quick    # 3 folds, for CI

Outputs (all committed as evidence):
    data/sales_history.csv           seeded history for the pilot restaurant
    models/xgb_*.json, model_meta.json
    reports/backtest_*.csv, metrics.json, model_report.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecast import config  # noqa: E402
from forecast.backtest import residual_quantiles, run_backtest, segment_errors, summarise  # noqa: E402
from forecast.data import load_sales  # noqa: E402
from forecast.model import fit, save  # noqa: E402
from forecast.operations import business_impact  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main(quick: bool = False) -> dict:
    np.random.seed(config.SEED)
    folds = 3 if quick else config.BACKTEST_FOLDS
    config.REPORT_DIR.mkdir(exist_ok=True)

    df = load_sales()
    print(f"Loaded {len(df)} days: {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")

    summary, models, quantiles, impact, segments = {}, {}, {}, None, None
    for target in config.TARGETS:
        print(f"\n=== Backtesting {target} ({folds} folds x {config.HORIZON} days) ===")
        bt = run_backtest(df, target, folds)
        bt.to_csv(config.REPORT_DIR / f"backtest_{target}.csv", index=False)
        res = summarise(bt)
        print(res.round(3).to_string())
        summary[target] = res.round(4).to_dict(orient="index")
        quantiles[target] = residual_quantiles(bt)
        if target == "covers":
            impact = business_impact(bt)
            impact.to_csv(config.REPORT_DIR / "business_impact.csv")
            segments = segment_errors(bt)
            segments.to_csv(config.REPORT_DIR / "segment_errors.csv", index=False)
            print("\nAnnualised plan cost (GBP):\n", impact.to_string())

        models[target], features = fit(df, target)          # final model on all history

    meta = {
        "restaurant": config.RESTAURANT_NAME,
        "targets": config.TARGETS,
        "horizon_days": config.HORIZON,
        "features": features,
        "params": {**config.XGB_PARAMS, "num_boost_round": config.XGB_ROUNDS},
        "seed": config.SEED,
        "train_start": str(df["date"].min().date()),
        "train_end": str(df["date"].max().date()),
        "n_train_days": len(df),
        "backtest_folds": folds,
        "data_sha256": sha256(config.RAW_SALES),
        "residual_quantiles": quantiles,
        "python": platform.python_version(),
    }
    save(models, meta)

    metrics = {"backtest": summary, "business_impact": impact.to_dict(orient="index"),
               "segment_errors": segments.to_dict(orient="records"), "meta": meta}
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    write_report(summary, impact, segments, meta)
    print(f"\nSaved models -> {config.MODEL_DIR}, reports -> {config.REPORT_DIR}")
    return metrics


def write_report(summary: dict, impact: pd.DataFrame, segments: pd.DataFrame, meta: dict) -> None:
    lines = [f"# Model report — {meta['restaurant']}", "",
             f"Training data: {meta['train_start']} → {meta['train_end']} ({meta['n_train_days']} days), "
             f"data hash `{meta['data_sha256']}`, seed {meta['seed']}, "
             f"{meta['backtest_folds']} rolling-origin folds × {meta['horizon_days']}-day horizon.", ""]
    for target, res in summary.items():
        lines += [f"## Baseline comparison — {target}", "",
                  "| Method | MAE | WAPE | RMSE | Bias | WAPE improvement vs naive |",
                  "|---|---|---|---|---|---|"]
        for m, r in res.items():
            lines.append(f"| {m} | {r['MAE']:.2f} | {r['WAPE']:.1%} | {r['RMSE']:.2f} | {r['Bias']:+.2f} "
                         f"| {r['WAPE_improvement_vs_naive_%']:+.1f}% |")
        lines.append("")
    lines += ["## Business impact (annualised, GBP, covers forecast → staffing & prep)", "",
              impact.to_markdown() if hasattr(impact, "to_markdown") and _has_tabulate() else impact.to_string(),
              "", "## Error parity across segments (XGBoost, covers)", "",
              "| Dimension | Group | Days | MAE | WAPE |", "|---|---|---|---|---|"]
    for r in segments.to_dict(orient="records"):
        lines.append(f"| {r['dimension']} | {r['group']} | {r['n_days']} | {r['MAE']:.2f} | {r['WAPE']:.1%} |")
    (config.REPORT_DIR / "model_report.md").write_text("\n".join(lines) + "\n")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
