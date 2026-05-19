"""
Build EWMA-126 (halflife=126 calendar days) rolling top-decile shock indicators
for the five retained shock channels.

Motivation
----------
The dashboard's headline cells are post-COVID-basis (cleaner regime), but the
original `panel_with_shocks.parquet` shock indicators were defined using
FULL-SAMPLE (2014-2025) top-decile cutoffs. That creates a methodological
inconsistency: post-COVID days end up firing shock flags at 14-19% (vs the
nominal 10%) because the full-sample cutoff is anchored to pre-COVID
volatility levels. The dashboard's per-shock cells (cell sample sizes,
per-shock means, OOS R²) inherit this mismatch.

Fix: re-define each shock as the top decile of a TRAILING EWMA-weighted
distribution of its raw measure, with halflife = 126 calendar days
(effective sample size ≈ 182 days; today's weight = 1.0; 126 days ago = 0.5;
252 days ago = 0.25). This makes the cutoff regime-aware: each calendar date
fires a shock if its raw measure exceeds today's 90th percentile of the
exponentially-weighted distribution of that measure's recent past.

Halflife rationale
------------------
- Matches the seed's pre-pilot `ewma_halflife: 126` convention (≈ 6 trading
  months, ≈ half a calendar year of weighting mass before decay flattens).
- Effective sample ~182 obs: large enough to estimate a 90th-percentile
  cutoff (50-70 effective tail obs above the cutoff) without overreacting to
  a single noisy week.
- Smooth: no abrupt regime change at a window boundary.

Shock raw measures
------------------
- oil_shock          : |Δ log(WTI)|         from FRED `wti` (DCOILWTICO)
- dollar_shock       : |Δ usd_broad|        from FRED `usd_broad` (DTWEXBGS)  level diff
- rate_shock         : |Δ y10_fred|         from FRED `y10_fred` (DGS10) — yield diff
- banking_shock      :  stlfsi (LEVEL)      from FRED `stlfsi` (STLFSI4)
- gprd_threat_shock  :  gprd_threat (LEVEL) from Caldara-Iacoviello

(The first three are change-based; the last two are level-based — matching
the original `panel_with_shocks.parquet` convention so the cell partition
is comparable.)

EWMA-weighted percentile algorithm
----------------------------------
For each calendar date t and each shock measure series x_1...x_t:
    weights w_k = (1/2) ** ((t - k) / 126)   for k=1..t  (sum = W)
    Sort observations by value ascending.
    Find the smallest value v* such that cumulative-weight(values <= v*) >= 0.9 * W.
    v* is the EWMA-weighted 90th percentile cutoff at date t.
    shock[t] = 1 iff x_t > v*.

Edge case: first 252 trading days. EWMA needs some history to be meaningful;
we set the indicator to NaN for the first 252 panel rows of each shock
series (this is pre-COVID anyway, which the dashboard treats as off-regime).

Inputs
------
- output/seed/paper1_context/panel_with_shocks.parquet (raw measures)

Outputs
-------
- data/aux/ewma_shocks_panel.parquet
    columns per shock: <shock>_ewma (binary 1/0/NaN),
                        <shock>_raw_measure (float),
                        <shock>_ewma_threshold (float, daily cutoff)
- output/stage3a/results/ewma_shock_incidence.json
    incidence rate by era for each shock, full-sample vs EWMA.

Reproducibility
---------------
- np.random.seed(42)
- Input panel path + date range logged in output JSON.
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "output" / "seed" / "paper1_context" / "panel_with_shocks.parquet"
AUX_DIR = ROOT / "data" / "aux"
AUX_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = AUX_DIR / "ewma_shocks_panel.parquet"

RESULTS_DIR = ROOT / "output" / "stage3a" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_INCIDENCE_JSON = RESULTS_DIR / "ewma_shock_incidence.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EWMA_HALFLIFE_DAYS = 126
QUANTILE = 0.90
WARMUP_OBS = 252      # drop first 252 panel rows per shock (NaN flag)

# Shock raw-measure spec: (shock_name, raw_col, measure_kind)
#  measure_kind: "abs_log_diff" | "abs_diff" | "level"
SHOCK_SPEC = [
    ("oil_shock",         "wti",         "abs_log_diff"),
    ("dollar_shock",      "usd_broad",   "abs_diff"),
    ("rate_shock",        "y10_fred",    "abs_diff"),
    ("banking_shock",     "stlfsi",      "level"),
    ("gprd_threat_shock", "gprd_threat", "level"),
]

# Era definitions (mirror dashboard_lookup.py)
ERAS = [
    ("pre_covid",         "2014-01-02", "2020-02-29"),
    ("post_covid_pre_etf","2020-03-01", "2024-01-09"),
    ("post_etf",          "2024-01-10", "2099-12-31"),
]


# ---------------------------------------------------------------------------
# Raw-measure construction
# ---------------------------------------------------------------------------
def raw_measure(panel: pd.DataFrame, col: str, kind: str) -> pd.Series:
    """Compute the raw daily measure for a shock series.

    abs_log_diff:  |log x_t - log x_{t-1}|
    abs_diff:      |x_t - x_{t-1}|
    level:         x_t directly
    """
    s = panel[col].astype(float)
    if kind == "abs_log_diff":
        out = np.log(s / s.shift(1)).abs()
    elif kind == "abs_diff":
        out = (s - s.shift(1)).abs()
    elif kind == "level":
        out = s
    else:
        raise ValueError(f"unknown measure kind: {kind}")
    out.name = "raw"
    return out


# ---------------------------------------------------------------------------
# EWMA-weighted quantile algorithm
# ---------------------------------------------------------------------------
def ewma_rolling_quantile(x: pd.Series, halflife: int,
                           quantile: float, warmup_obs: int) -> pd.Series:
    """For each t, compute the EWMA-weighted `quantile` of x[0..t] using
    exponential weights w_k = 0.5 ** ((t-k)/halflife).

    Returns a Series indexed identically to x. Values before warmup_obs are NaN.

    Implementation: O(N log N) total via two passes.
      Pass 1: precompute decay factor d = 0.5 ** (1/halflife). For each t,
              total_weight(t) = d * total_weight(t-1) + 1, where each weight
              is in "today's time-coordinate" (we re-anchor at t).
      Pass 2: for each t, sort the most-recent W observations (we use all
              available history; tail weights decay fast so we cap at 5
              halflives back = 630 days for efficiency without changing
              the answer to 4 decimals).

    Note: re-sorting up to 630 obs at each t is O(N * W log W) total
    (N ≈ 3000, W ≤ 630). For 5 shocks, total runtime ≈ 30-60 seconds.
    """
    n = len(x)
    decay = 0.5 ** (1.0 / halflife)
    # Cap effective history at 5 halflives (1/32 ≈ 0.03 weight contribution)
    max_lookback = halflife * 5
    out = np.full(n, np.nan)

    x_vals = x.values
    is_nan = np.isnan(x_vals)

    for t in range(n):
        if t < warmup_obs:
            continue
        if is_nan[t]:
            continue
        # Window: from max(0, t - max_lookback + 1) to t (inclusive)
        lo = max(0, t - max_lookback + 1)
        slice_vals = x_vals[lo:t+1]
        # Filter NaN
        mask = ~np.isnan(slice_vals)
        if mask.sum() < 30:
            continue
        valid = slice_vals[mask]
        # Lags relative to t (in calendar-row index, since this is daily data)
        idxs = np.arange(lo, t+1)
        lags = (t - idxs[mask]).astype(float)
        weights = decay ** lags
        # Sort by value
        order = np.argsort(valid)
        vs = valid[order]
        ws = weights[order]
        cum = np.cumsum(ws)
        total = cum[-1]
        target = quantile * total
        # Find smallest v such that cum >= target
        k = int(np.searchsorted(cum, target, side="left"))
        if k >= len(vs):
            k = len(vs) - 1
        out[t] = vs[k]
    return pd.Series(out, index=x.index, name="ewma_q")


# ---------------------------------------------------------------------------
# Era classification helper
# ---------------------------------------------------------------------------
def era_of(date: pd.Timestamp) -> str:
    for name, lo, hi in ERAS:
        if pd.Timestamp(lo) <= date <= pd.Timestamp(hi):
            return name
    return "pre_covid"


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[ewma_shocks] reading panel: {PANEL_PATH}", flush=True)
    panel = pd.read_parquet(PANEL_PATH).copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    out = pd.DataFrame(index=panel.index)
    out["era"] = [era_of(d) for d in panel.index]

    incidence = {}

    for shock_name, raw_col, kind in SHOCK_SPEC:
        if raw_col not in panel.columns:
            raise KeyError(f"panel missing raw column: {raw_col}")
        print(f"[ewma_shocks] computing EWMA cutoffs for {shock_name} "
              f"({raw_col}, {kind})", flush=True)
        raw = raw_measure(panel, raw_col, kind)
        threshold = ewma_rolling_quantile(raw, EWMA_HALFLIFE_DAYS,
                                           QUANTILE, WARMUP_OBS)
        # Indicator: 1 if today's raw > today's threshold; NaN if threshold NaN
        with np.errstate(invalid="ignore"):
            ind = np.where(
                np.isnan(threshold.values) | np.isnan(raw.values),
                np.nan,
                (raw.values > threshold.values).astype(float),
            )
        ind_series = pd.Series(ind, index=panel.index, name=f"{shock_name}_ewma")
        out[f"{shock_name}_raw_measure"] = raw.values
        out[f"{shock_name}_ewma_threshold"] = threshold.values
        out[f"{shock_name}_ewma"] = ind_series.values

        # Compute incidence by era
        ewma_in = pd.DataFrame({
            "era": out["era"].values,
            "ewma": ind_series.values,
            "fullsample": panel[shock_name].astype(float).values,
        }, index=panel.index)
        per_era = {}
        for era_name, _, _ in ERAS:
            sub = ewma_in[ewma_in["era"] == era_name]
            ewma_rate = float(sub["ewma"].mean(skipna=True)) if len(sub) else None
            fs_rate = float(sub["fullsample"].mean(skipna=True)) if len(sub) else None
            n_avail = int(sub["ewma"].notna().sum())
            n_total = int(len(sub))
            per_era[era_name] = {
                "n_total": n_total,
                "n_ewma_available": n_avail,
                "ewma_incidence": ewma_rate,
                "fullsample_incidence": fs_rate,
            }
        # Overall (post-warmup)
        ewma_rate_all = float(ind_series.mean(skipna=True))
        fs_rate_all = float(panel[shock_name].astype(float).mean(skipna=True))
        incidence[shock_name] = {
            "raw_col": raw_col,
            "measure_kind": kind,
            "ewma_overall_incidence": ewma_rate_all,
            "fullsample_overall_incidence": fs_rate_all,
            "n_ewma_available_total": int(ind_series.notna().sum()),
            "by_era": per_era,
        }

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET)
    print(f"[ewma_shocks] wrote: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1024:.1f} KB)",
          flush=True)

    incidence_doc = {
        "_metadata": {
            "script": "code/build_ewma_shocks.py",
            "panel_path": str(PANEL_PATH),
            "panel_n_rows": int(len(panel)),
            "panel_date_range": [str(panel.index.min().date()),
                                  str(panel.index.max().date())],
            "ewma_halflife_days": EWMA_HALFLIFE_DAYS,
            "quantile": QUANTILE,
            "warmup_obs": WARMUP_OBS,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "seed": 42,
        },
        "shock_incidence": incidence,
    }
    OUT_INCIDENCE_JSON.write_text(json.dumps(incidence_doc, indent=2, default=str))
    print(f"[ewma_shocks] wrote: {OUT_INCIDENCE_JSON}", flush=True)

    # Pretty-print summary
    print("\nShock incidence (per-era):", flush=True)
    print(f"{'shock':<22s}  {'era':<22s}  {'n':>5s}  {'n_ewma':>7s}  "
          f"{'ewma_rate':>10s}  {'full_rate':>10s}", flush=True)
    for s, doc in incidence.items():
        for era, era_doc in doc["by_era"].items():
            rate_e = era_doc["ewma_incidence"]
            rate_f = era_doc["fullsample_incidence"]
            print(f"  {s:<20s}  {era:<22s}  {era_doc['n_total']:>5d}  "
                  f"{era_doc['n_ewma_available']:>7d}  "
                  f"{(f'{rate_e:.3f}' if rate_e is not None else 'n/a'):>10s}  "
                  f"{(f'{rate_f:.3f}' if rate_f is not None else 'n/a'):>10s}",
                  flush=True)
        print(f"  {s:<20s}  {'OVERALL':<22s}                  "
              f"{doc['ewma_overall_incidence']:.3f}      "
              f"{doc['fullsample_overall_incidence']:.3f}", flush=True)


if __name__ == "__main__":
    main()
