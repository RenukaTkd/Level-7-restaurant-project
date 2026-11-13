# 3. Reproducibility

| Control | How it is implemented |
|---|---|
| Fixed randomness | `SEED = 42` in `src/forecast/config.py` drives data generation, XGBoost `seed` and `nthread=1` (deterministic tree building) |
| Pinned environment | `requirements.txt` pins pandas, numpy, xgboost and fastapi; `.python-version` = 3.12 |
| Data versioning | `train.py` writes a SHA-256 of the training CSV into `models/model_meta.json` and `/health` |
| Single entry point | `python scripts/train.py` regenerates **every** artefact (data → backtest → model → reports) |
| Config in one place | all hyper-parameters, horizons, staffing ratios and cost assumptions are in `config.py` |
| Model lineage | `model_meta.json` records features, params, training window, xgboost and Python versions, training time |
| Automated checks | GitHub Actions retrains on every push, compares the model hash and runs 11 tests |
| Leakage test | `tests/test_pipeline.py::test_no_leakage_lags_respect_horizon` |

## Evidence
Running `scripts/train.py` twice on the same machine produced byte-identical models:

```
f239423ba21866157e580faa8faa5069e8d8356a  models/xgb_covers.json    (run 1)
f239423ba21866157e580faa8faa5069e8d8356a  models/xgb_covers.json    (run 2)
```

Floating-point results can differ slightly between operating systems or XGBoost builds. CI therefore raises a
*warning* (not a failure) if the hash differs on Linux. The metrics in `reports/metrics.json` stay identical to
2 decimal places.

## How an assessor reproduces the results
```bash
git clone <repo> && cd <repo>
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/train.py && pytest -q
diff <(git show HEAD:reports/model_report.md) reports/model_report.md   # expect no metric changes
```
