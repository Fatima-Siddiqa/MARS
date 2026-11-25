"""Train and evaluate the valuation model on the PSX-wide training set.

Target: log(market cap) minus the market's median log P/E, on the first trading day after the
fiscal-year figures were published (see dataset.py, market_level.py). So exp(prediction + level)
is the market cap that companies with these fundamentals typically had: the model's fair value.

Candidates (simplest first):
  sector_pe_baseline  sector median P/E from training years (P/S for loss-makers) - no fitting
  ridge               linear regression with standardised features + sector one-hot
  gbm                 gradient-boosted trees (scikit-learn HistGradientBoosting)

Evaluation: walk-forward by observation year (train on earlier years only, test on the next),
so no test company-year is ever seen in training. Selection: the simplest candidate within
SIMPLICITY_TOLERANCE of the best pooled test MAE; a fitted model is adopted only if it also
beats the baseline. The final model is refit on all years and saved with its test report.

    python -m app.agents.fundamentals.valuation.train
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.agents.fundamentals.valuation.dataset import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    PROCESSED,
    TARGET,
    TRAINING_CSV,
    build_training_set,
)

log = logging.getLogger(__name__)
MODEL_DIR = PROCESSED / "models"
MODEL_PATH = MODEL_DIR / "valuation_model.joblib"
REPORT_JSON = MODEL_DIR / "valuation_report.json"
REPORT_MD = MODEL_DIR / "valuation_report.md"
OOS_CSV = MODEL_DIR / "oos_predictions.csv"
MODEL_VERSION = "fundamentals-valuation-v1"
SIMPLICITY_TOLERANCE = 0.05
MIN_TRAIN_ROWS = 150
ORDER = ["sector_pe_baseline", "ridge", "gbm"]


class SectorMultipleBaseline:
    """Sector median of log(value / profit) (P/E) or log(value / revenue) for loss-makers."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        d = X.assign(y=y.values)
        pos = d[d["is_loss"] == 0]
        self.pe_ = (pos["y"] - pos["log_profit"]).groupby(pos["sector_code"]).median()
        self.pe_all_ = float((pos["y"] - pos["log_profit"]).median())
        rev = d[d["log_revenue"].notna()]
        self.ps_ = (rev["y"] - rev["log_revenue"]).groupby(rev["sector_code"]).median()
        self.ps_all_ = float((rev["y"] - rev["log_revenue"]).median())
        self.y_all_ = float(d["y"].median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = []
        for r in X.itertuples():
            if r.is_loss == 0 and not np.isnan(r.log_profit):
                out.append(r.log_profit + self.pe_.get(r.sector_code, self.pe_all_))
            elif not np.isnan(r.log_revenue):
                out.append(r.log_revenue + self.ps_.get(r.sector_code, self.ps_all_))
            else:
                out.append(self.y_all_)
        return np.array(out)


def make(name: str):
    if name == "sector_pe_baseline":
        return SectorMultipleBaseline()
    if name == "ridge":
        pre = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    NUMERIC_FEATURES,
                ),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
        )
        return Pipeline([("pre", pre), ("model", RidgeCV(alphas=np.logspace(-2, 3, 20)))])
    if name == "gbm":
        pre = ColumnTransformer(
            [
                ("num", "passthrough", NUMERIC_FEATURES),
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            categorical_features=[len(NUMERIC_FEATURES)],
            random_state=0,
        )
        return Pipeline([("pre", pre), ("model", model)])
    raise ValueError(name)


def _metrics(y, p) -> dict:
    err = np.abs(np.asarray(y) - np.asarray(p))
    return {
        "n": int(len(err)),
        "mae_log": float(err.mean()),
        "median_abs_pct_error": float(np.expm1(np.median(err))),
        "r2": float(r2_score(y, p)) if len(err) > 1 else None,
    }


def walk_forward(ds: pd.DataFrame) -> dict:
    years = sorted(ds["obs_year"].unique())
    folds, pooled = [], {n: ([], []) for n in ORDER}
    residuals = {n: [] for n in ORDER}
    oos = []  # out-of-sample predictions, used to calibrate BUY / HOLD / AVOID
    for y in years:
        train, test = ds[ds["obs_year"] < y], ds[ds["obs_year"] == y]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        fold = {"test_year": int(y), "n_train": len(train), "n_test": len(test)}
        keep = [c for c in ("ticker", "obs_date", "obs_year", "close", TARGET) if c in test]
        frame = test[keep].copy()
        for name in ORDER:
            m = make(name).fit(train[FEATURES], train[TARGET])
            p = m.predict(test[FEATURES])
            fold[name] = _metrics(test[TARGET], p)
            pooled[name][0].extend(test[TARGET])
            pooled[name][1].extend(p)
            residuals[name].extend(np.asarray(test[TARGET]) - p)
            frame[f"pred_{name}"] = p
        folds.append(fold)
        oos.append(frame)
    summary = {n: _metrics(*pooled[n]) for n in ORDER if pooled[n][0]}
    return {
        "folds": folds,
        "pooled": summary,
        "residuals": residuals,
        "oos": pd.concat(oos, ignore_index=True) if oos else pd.DataFrame(),
    }


def choose(ev: dict) -> tuple[str, str]:
    pooled = ev["pooled"]
    if not pooled:
        return "sector_pe_baseline", "not enough data for walk-forward evaluation"
    best = min(pooled.values(), key=lambda m: m["mae_log"])["mae_log"]
    for name in ORDER:
        if name in pooled and pooled[name]["mae_log"] <= best * (1 + SIMPLICITY_TOLERANCE):
            chosen = name
            break
    base = pooled["sector_pe_baseline"]["mae_log"]
    if chosen != "sector_pe_baseline" and pooled[chosen]["mae_log"] >= base:
        return "sector_pe_baseline", "no fitted model beat the baseline on the test years"
    wins = sum(f[chosen]["mae_log"] < f["sector_pe_baseline"]["mae_log"] for f in ev["folds"])
    why = (
        f"lowest test error within {SIMPLICITY_TOLERANCE:.0%} of the best; "
        f"beats the baseline in {wins} of {len(ev['folds'])} test years"
    )
    return chosen, why if chosen != "sector_pe_baseline" else "baseline was best"


def shap_background(model_name: str, X: pd.DataFrame):
    return X.sample(min(200, len(X)), random_state=0)


def train(rebuild_dataset: bool = True) -> dict:
    ds = (
        build_training_set()
        if rebuild_dataset or not TRAINING_CSV.exists()
        else pd.read_csv(TRAINING_CSV, dtype={"sector_code": str})
    )
    ds["sector_code"] = ds["sector_code"].astype(str).str.zfill(4)
    ev = walk_forward(ds)
    chosen, why = choose(ev)
    final = make(chosen).fit(ds[FEATURES], ds[TARGET])
    res = np.asarray(ev["residuals"].get(chosen, []))
    interval = (
        [float(np.quantile(res, 0.1)), float(np.quantile(res, 0.9))] if len(res) >= 50 else None
    )
    report = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "chosen": chosen,
        "why": why,
        "data": {
            "rows": len(ds),
            "companies": int(ds["ticker"].nunique()),
            "obs_years": sorted(int(y) for y in ds["obs_year"].unique()),
            "dropped": ds.attrs.get("dropped", {}),
            "source": "PSX company pages + PSX daily market summaries (official)",
        },
        "target": "log(market cap) - market median log P/E, first trading day after publication",
        "features": FEATURES,
        "walk_forward": ev["folds"],
        "pooled_test": ev["pooled"],
        "residual_interval_80pct_log": interval,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "name": chosen,
            "model": final,
            "features": FEATURES,
            "background": shap_background(chosen, ds[FEATURES]),
            "report": report,
        },
        MODEL_PATH,
    )
    if not ev["oos"].empty:
        oos = ev["oos"].copy()
        oos["pred"] = oos[f"pred_{chosen}"]
        oos["gap_log"] = oos["pred"] - oos[TARGET]  # >0: valued below what fundamentals imply
        oos.to_csv(OOS_CSV, index=False)
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    REPORT_MD.write_text(render_md(report))
    return report


def render_md(r: dict) -> str:
    lines = [
        f"# Fundamentals valuation model ({r['model_version']})",
        "",
        f"Trained {r['trained_at']} on {r['data']['rows']} company-years from "
        f"{r['data']['companies']} PSX companies ({r['data']['source']}).",
        "",
        f"**Chosen: `{r['chosen']}`**: {r['why']}.",
        "",
        "## Pooled walk-forward test results",
        "",
        "| candidate | test rows | MAE (log) | median abs. error | R² |",
        "|---|---|---|---|---|",
    ]
    for n, m in r["pooled_test"].items():
        lines.append(
            f"| {n} | {m['n']} | {m['mae_log']:.3f} | {m['median_abs_pct_error']:.0%} | "
            f"{m['r2']:.2f} |"
        )
    lines += [
        "",
        "## By test year",
        "",
        "| test year | train rows | test rows | " + " | ".join(f"{n} MAE" for n in ORDER) + " |",
        "|---" * (3 + len(ORDER)) + "|",
    ]
    for f in r["walk_forward"]:
        lines.append(
            f"| {f['test_year']} | {f['n_train']} | {f['n_test']} | "
            + " | ".join(f"{f[n]['mae_log']:.3f}" for n in ORDER)
            + " |"
        )
    lines += ["", f"Rows dropped while building the data: {r['data']['dropped']}", ""]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    r = train()
    print(json.dumps({k: r[k] for k in ("chosen", "why", "pooled_test")}, indent=2))


if __name__ == "__main__":
    main()
