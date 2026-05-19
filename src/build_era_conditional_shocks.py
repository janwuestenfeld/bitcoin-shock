"""Era-conditional top-decile shock indicators (within-era full-distribution).

Motivation
----------
- Full-sample cutoffs (paper): mix pre-COVID + post-COVID regimes.
- EWMA-126 rolling cutoffs: over-adapt, smear extreme + non-extreme periods
  within a window.
- Expanding-within-era with prior-era fallback: mechanically biases early-era
  classifications when prior-era cutoff misaligns with new-era vol regime.

This script: **within-era full-distribution top-decile**. For each era and
each shock, the cutoff is the 90th percentile of that shock's raw measures
over the entire era. Each era gets ~10% shock rate by construction. No
within-era extreme/non-extreme mixing; no across-era regime mixing.

What this is for
----------------
The cutoff IS look-ahead in the narrow time-ordered sense (at date t mid-era,
it equals end-of-era top-decile, which the live operator wouldn't have).
But for the OOS R² research question — "does cell-conditional structure
exist by end-of-era regime standards?" — this is the correct cutoff:

1. The OOS R² test uses random K-fold CV (not walk-forward); the test asks
   whether observations within a stable cutoff generalize across folds.
2. The cutoff is a feature (regime classifier), not the outcome.
3. By end-of-era the cutoff stabilizes; today's live cutoff IS the within-era
   full-distribution cutoff for the current era.

For the **live operational dashboard**, an expanding within-era cutoff with
NaN warmup (separate build) is the right tool — but that's not what the
OOS R² test is asking.

Outputs
-------
- data/aux/era_conditional_shocks_panel.parquet
- output/stage3a/results/era_conditional_shock_incidence.json
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "output" / "seed" / "paper1_context" / "panel_with_shocks.parquet"
AUX_DIR = ROOT / "data" / "aux"
AUX_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = AUX_DIR / "era_conditional_shocks_panel.parquet"

RESULTS_DIR = ROOT / "output" / "stage3a" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_INCIDENCE_JSON = RESULTS_DIR / "era_conditional_shock_incidence.json"

QUANTILE = 0.90

SHOCK_SPEC = [
    ("oil_shock",         "wti",         "abs_log_diff"),
    ("dollar_shock",      "usd_broad",   "abs_diff"),
    ("rate_shock",        "y10_fred",    "abs_diff"),
    ("banking_shock",     "stlfsi",      "level"),
    ("gprd_threat_shock", "gprd_threat", "level"),
]

ERAS = [
    ("pre_covid",         "2014-01-02", "2020-02-29"),
    ("post_covid_pre_etf","2020-03-01", "2024-01-09"),
    ("post_etf",          "2024-01-10", "2099-12-31"),
]


def raw_measure(panel: pd.DataFrame, col: str, kind: str) -> pd.Series:
    s = panel[col].astype(float)
    if kind == "abs_log_diff":
        return np.log(s / s.shift(1)).abs()
    if kind == "abs_diff":
        return (s - s.shift(1)).abs()
    if kind == "level":
        return s
    raise ValueError(kind)


def era_of(d: pd.Timestamp) -> str:
    for n, lo, hi in ERAS:
        if pd.Timestamp(lo) <= d <= pd.Timestamp(hi):
            return n
    return "pre_covid"


def main() -> None:
    print(f"[era_cond] reading panel: {PANEL_PATH}", flush=True)
    panel = pd.read_parquet(PANEL_PATH).copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    out = pd.DataFrame(index=panel.index)
    out["era"] = [era_of(d) for d in panel.index]

    incidence = {}
    for shock_name, raw_col, kind in SHOCK_SPEC:
        if raw_col not in panel.columns:
            raise KeyError(raw_col)
        print(f"[era_cond] {shock_name} ({raw_col}, {kind})", flush=True)
        raw = raw_measure(panel, raw_col, kind)

        thresh = pd.Series(np.nan, index=panel.index)
        for era_name, lo, hi in ERAS:
            mask = (panel.index >= pd.Timestamp(lo)) & (panel.index <= pd.Timestamp(hi))
            era_vals = raw[mask].dropna()
            if len(era_vals) < 30:
                continue
            cutoff = float(era_vals.quantile(QUANTILE))
            thresh.loc[mask] = cutoff

        with np.errstate(invalid="ignore"):
            ind = np.where(
                np.isnan(thresh.values) | np.isnan(raw.values),
                np.nan,
                (raw.values > thresh.values).astype(float),
            )
        ind_series = pd.Series(ind, index=panel.index)

        out[f"{shock_name}_era_cond_raw"] = raw.values
        out[f"{shock_name}_era_cond_thresh"] = thresh.values
        out[f"{shock_name}_era_cond"] = ind_series.values

        per_era = {}
        for era_name, _, _ in ERAS:
            mask = out["era"] == era_name
            sub = ind_series[mask]
            fs = panel[shock_name].astype(float)[mask] if shock_name in panel.columns else None
            tser = thresh[mask].dropna()
            cutoff_val = float(tser.iloc[0]) if len(tser) else None
            per_era[era_name] = {
                "n_total": int(mask.sum()),
                "n_available": int(sub.notna().sum()),
                "era_cond_incidence": float(sub.mean(skipna=True)) if sub.notna().any() else None,
                "fullsample_incidence": (
                    float(fs.mean(skipna=True)) if fs is not None and fs.notna().any() else None
                ),
                "within_era_cutoff": cutoff_val,
            }
        incidence[shock_name] = {
            "raw_col": raw_col, "measure_kind": kind,
            "era_cond_overall_incidence": float(ind_series.mean(skipna=True)),
            "fullsample_overall_incidence": (
                float(panel[shock_name].astype(float).mean(skipna=True))
                if shock_name in panel.columns else None
            ),
            "n_available_total": int(ind_series.notna().sum()),
            "by_era": per_era,
        }

    out.to_parquet(OUT_PARQUET)
    print(f"[era_cond] wrote {OUT_PARQUET} "
          f"({OUT_PARQUET.stat().st_size/1024:.1f} KB)", flush=True)

    doc = {
        "_metadata": {
            "script": "code/build_era_conditional_shocks.py",
            "method": "within-era full-distribution top-decile cutoff "
                       "(stable regime classifier; not for live operation)",
            "panel_path": str(PANEL_PATH),
            "quantile": QUANTILE,
            "eras": [{"name": n, "start": s, "end": e} for n, s, e in ERAS],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "seed": 42,
        },
        "shock_incidence": incidence,
    }
    OUT_INCIDENCE_JSON.write_text(json.dumps(doc, indent=2, default=str))
    print(f"[era_cond] wrote {OUT_INCIDENCE_JSON}", flush=True)

    print("\nShock incidence by era (within-era full-distribution cutoffs):", flush=True)
    print(f"{'shock':<22s} {'era':<22s} {'n':>5s} {'n_avail':>7s} "
          f"{'cutoff':>10s} {'era_cond':>10s} {'full_samp':>10s}", flush=True)
    for s, d in incidence.items():
        for era, ed in d["by_era"].items():
            cut = f"{ed['within_era_cutoff']:.4f}" if ed["within_era_cutoff"] is not None else "n/a"
            ec = f"{ed['era_cond_incidence']:.3f}" if ed["era_cond_incidence"] is not None else "n/a"
            fs = f"{ed['fullsample_incidence']:.3f}" if ed["fullsample_incidence"] is not None else "n/a"
            print(f"  {s:<20s} {era:<22s} {ed['n_total']:>5d} {ed['n_available']:>7d} "
                  f"{cut:>10s} {ec:>10s} {fs:>10s}", flush=True)


if __name__ == "__main__":
    main()
