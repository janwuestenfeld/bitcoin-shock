"""Formal OOS R^2 test: does shock-type add predictive value beyond era x VIX-bin?

Outcome:  y_t = r_BTC,calendar,t->t+60  -  r_SPY,nyse_aligned,t->nearest_NYSE(t+60).
Samples:  (A) full panel of NYSE-aligned days  (B) shock-active subset.

Models:
  M0  intercept-only                         (k=0)
  M1  era dummies                            (k=2, baseline = pre_covid)
  M2  era x VIX-bin (3x4=12 cells)           (k=11)
  M3  era x VIX-bin x shock-type (12x7=84)   (k <= 83; cells with n=0 dropped)
  M4  era x VIX-bin + shock-type main eff.   (k=11 + 6 = 17)  -- additive shock,
                                                              no interaction

Evaluation:  5-fold shuffled CV (seed=42), block-OLS / Ridge with lambda=1e-4.
              OOS R^2 = 1 - sum((y - y_hat)^2) / sum((y - mean(y_train))^2)
              averaged across folds (variance pooled across folds).
Significance: permutation test shuffling SHOCK-TYPE labels only (preserving
              era x VIX-bin x outcome), re-fit M3 and M4 200 times, build a
              null distribution for DR^2 (M2 -> M3) and (M2 -> M4).

Per-shock loadings:  M4 coefficients (independent contributions controlling
                     for era x VIX-bin); M3 difference-in-means per cell where
                     the cell has both shock-active and shock-active occurrences.

Outputs:
  - output/stage3a/results/shock_adds_value_test.json
  - output/stage3a/shock_adds_value_test.md
  - output/stage3a/tables/shock_adds_value_test.tex
"""
from __future__ import annotations

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Reproducibility
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/Users/janwustenfeld/Documents/btc-vix-threshold-paper2")
PANEL_PATH = ROOT / "output/seed/paper1_context/panel_with_shocks.parquet"
BTC_CAL_PATH = ROOT / "data/aux/btc_calendar_daily.parquet"

OUT_DIR = ROOT / "output/stage3a"
RESULTS_DIR = OUT_DIR / "results"
TABLES_DIR = OUT_DIR / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = RESULTS_DIR / "shock_adds_value_test.json"
OUT_MD = OUT_DIR / "shock_adds_value_test.md"
OUT_TEX = TABLES_DIR / "shock_adds_value_test.tex"

# ---------------------------------------------------------------------------
# Constants (mirror dashboard_lookup.py)
# ---------------------------------------------------------------------------
RETAINED_SHOCKS = [
    "oil_shock",
    "dollar_shock",
    "rate_shock",
    "banking_shock",
    "gprd_threat_shock",
]

VIX_BINS = [
    ("calm", -np.inf, 14.5),
    ("low_stress", 14.5, 20.0),
    ("mid_stress", 20.0, 30.0),
    ("extreme_stress", 30.0, np.inf),
]
VIX_BIN_NAMES = [v[0] for v in VIX_BINS]

ERAS = [
    ("pre_covid", "2014-01-02", "2020-02-29"),
    ("post_covid_pre_etf", "2020-03-01", "2024-01-09"),
    ("post_etf", "2024-01-10", "2099-12-31"),
]
ERA_NAMES = [e[0] for e in ERAS]

H = 60
N_FOLDS = 5
N_PERMUTATIONS = 200
RIDGE_LAMBDA = 1e-4


# ---------------------------------------------------------------------------
# Helpers (mirror lookup conventions)
# ---------------------------------------------------------------------------
def vix_bin_of(vix):
    if vix is None or (isinstance(vix, float) and np.isnan(vix)):
        return "calm"
    for name, lo, hi in VIX_BINS:
        if lo <= vix < hi:
            return name
    return "calm"


def era_of(date):
    for name, lo, hi in ERAS:
        if pd.Timestamp(lo) <= date <= pd.Timestamp(hi):
            return name
    return "pre_covid"


def spy_nearest_nyse_fwd(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Forward SPY return from spy(t) -> spy(nearest NYSE >= t+h_cal). Mirrors
    code/dashboard_lookup.py:spy_nearest_nyse_fwd but restricted to the panel
    itself (target_dates = panel.index).
    """
    nyse_arr = np.asarray(panel.index.sort_values().values)
    spy = panel["spy"].astype(float).values
    t_dates = nyse_arr

    pos_le_t = np.searchsorted(nyse_arr, t_dates, side="right") - 1
    pos_le_t = np.clip(pos_le_t, 0, len(nyse_arr) - 1)
    spy_t = spy[pos_le_t]

    out = pd.DataFrame(index=panel.index.sort_values())
    for h in horizons:
        tph = t_dates + np.timedelta64(h, "D")
        pos_ge = np.searchsorted(nyse_arr, tph, side="left")
        valid = pos_ge < len(nyse_arr)
        pos_clip = np.clip(pos_ge, 0, len(nyse_arr) - 1)
        spy_tph = np.where(valid, spy[pos_clip], np.nan)
        spy_t_h = np.where(valid, spy_t, np.nan)
        out[f"r_spy_nyse_fwd_{h}"] = spy_tph / spy_t_h - 1.0
    return out


def build_panel():
    """Build the test panel with all design variables."""
    panel = pd.read_parquet(PANEL_PATH).copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    panel["date"] = panel.index

    btc = pd.read_parquet(BTC_CAL_PATH).copy()
    btc.index = pd.to_datetime(btc.index)
    btc = btc.sort_index()
    btc_close = btc["close"].astype(float)

    # BTC calendar forward at h=60
    btc_fwd_full = btc_close.shift(-H) / btc_close - 1.0
    panel[f"r_btc_calendar_fwd_{H}"] = btc_fwd_full.reindex(panel.index).values

    # SPY NYSE forward
    spy_fwd = spy_nearest_nyse_fwd(panel, [H])
    panel = panel.join(spy_fwd, how="left")

    panel[f"outperf_{H}"] = panel[f"r_btc_calendar_fwd_{H}"] - panel[f"r_spy_nyse_fwd_{H}"]

    # Era and VIX bin
    panel["era"] = panel["date"].apply(era_of)
    panel["vix_bin"] = panel["vix"].apply(vix_bin_of)

    # Any-shock + per-shock dummies (RETAINED_SHOCKS)
    panel["any_shock"] = panel[RETAINED_SHOCKS].fillna(0).sum(axis=1).clip(upper=1).astype(int)

    # Drop rows with NA outcome (typically the last 60 NYSE days where t+60 has no SPY data)
    panel = panel.dropna(subset=[f"outperf_{H}"])
    return panel


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------
def design_M0(df):
    """Intercept only (zeros so OLS predicts mean)."""
    n = len(df)
    return np.zeros((n, 0)), []


def design_M1(df):
    """Era dummies (baseline = pre_covid)."""
    cols = [e for e in ERA_NAMES if e != "pre_covid"]
    X = np.column_stack([(df["era"] == e).astype(float).values for e in cols])
    return X, cols


def design_M2(df):
    """Era x VIX-bin cells (3x4 - 1 = 11 dummies; baseline = pre_covid_calm)."""
    df = df.copy()
    df["cell12"] = df["era"].astype(str) + "_" + df["vix_bin"].astype(str)
    all_cells = [f"{e}_{v}" for e in ERA_NAMES for v in VIX_BIN_NAMES]
    baseline = all_cells[0]
    cols = [c for c in all_cells if c != baseline and (df["cell12"] == c).sum() > 0]
    X = np.column_stack([(df["cell12"] == c).astype(float).values for c in cols])
    return X, cols


def design_M3(df):
    """Era x VIX-bin x shock-type. 12 cells x 7 shock-types (5 retained + none + multiple).

    Definition of shock-type per row:
      - "none"      if no retained shock active
      - "<shock>"   if exactly one retained shock active
      - "multi"     if 2+ retained shocks active that day
    """
    df = df.copy()
    df["cell12"] = df["era"].astype(str) + "_" + df["vix_bin"].astype(str)
    n_active = df[RETAINED_SHOCKS].fillna(0).sum(axis=1).astype(int)
    shock_type = []
    for i, n in enumerate(n_active.values):
        if n == 0:
            shock_type.append("none")
        elif n == 1:
            active = [s for s in RETAINED_SHOCKS if int(df.iloc[i][s] or 0) == 1][0]
            shock_type.append(active)
        else:
            shock_type.append("multi")
    df["shock_type"] = shock_type
    df["cell12x7"] = df["cell12"] + "__" + df["shock_type"]
    cells = sorted(df["cell12x7"].unique())
    # Drop one as baseline
    if not cells:
        return np.zeros((len(df), 0)), []
    baseline = cells[0]
    cols = [c for c in cells if c != baseline]
    X = np.column_stack([(df["cell12x7"] == c).astype(float).values for c in cols])
    return X, cols


def design_M4(df):
    """Era x VIX-bin + shock-type main effects (additive). 11 cell dummies + 6 shock dummies."""
    X2, cols2 = design_M2(df)
    df = df.copy()
    n_active = df[RETAINED_SHOCKS].fillna(0).sum(axis=1).astype(int)
    shock_type = []
    for i, n in enumerate(n_active.values):
        if n == 0:
            shock_type.append("none")
        elif n == 1:
            active = [s for s in RETAINED_SHOCKS if int(df.iloc[i][s] or 0) == 1][0]
            shock_type.append(active)
        else:
            shock_type.append("multi")
    df["shock_type"] = shock_type
    all_st = ["none"] + RETAINED_SHOCKS + ["multi"]
    present = [s for s in all_st if (df["shock_type"] == s).sum() > 0]
    if not present:
        return X2, cols2
    baseline = present[0]
    cols_s = [s for s in present if s != baseline]
    Xs = np.column_stack([(df["shock_type"] == s).astype(float).values for s in cols_s])
    cols = cols2 + [f"shock_{s}" for s in cols_s]
    if X2.shape[1] == 0:
        return Xs, cols
    return np.column_stack([X2, Xs]), cols


# ---------------------------------------------------------------------------
# Ridge fit + CV
# ---------------------------------------------------------------------------
def cv_r2(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS,
          seed: int = SEED, lam: float = RIDGE_LAMBDA) -> dict:
    """5-fold shuffled CV. Returns OOS R^2, R^2 IS (on full sample), sign acc OOS."""
    n = len(y)
    y_hat_oos = np.full(n, np.nan)
    train_mean_per_fold = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (idx_tr, idx_te) in enumerate(kf.split(X)):
        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]
        m = Ridge(alpha=lam, fit_intercept=True)
        if X_tr.shape[1] == 0:
            # intercept-only: predict mean(y_tr)
            yhat = np.full(len(idx_te), y_tr.mean())
        else:
            m.fit(X_tr, y_tr)
            yhat = m.predict(X_te)
        y_hat_oos[idx_te] = yhat
        train_mean_per_fold.append(y_tr.mean())

    # Aggregate OOS R^2: 1 - SSR / SST_train_mean (per Hastie&Tibshirani CV)
    # We use SST = sum((y - mean(y))^2) on full sample for a single anchor.
    sst = float(np.sum((y - y.mean()) ** 2))
    ssr = float(np.sum((y - y_hat_oos) ** 2))
    r2_oos = 1.0 - ssr / sst if sst > 0 else np.nan

    sign_acc_oos = float(np.mean(np.sign(y_hat_oos) == np.sign(y))) if n > 0 else np.nan

    # IS fit
    m_full = Ridge(alpha=lam, fit_intercept=True)
    if X.shape[1] == 0:
        yhat_is = np.full(n, y.mean())
        coefs = []
    else:
        m_full.fit(X, y)
        yhat_is = m_full.predict(X)
        coefs = m_full.coef_.tolist()
    r2_is = 1.0 - np.sum((y - yhat_is) ** 2) / sst if sst > 0 else np.nan

    return {
        "n_obs": n,
        "n_features": X.shape[1],
        "r2_is": float(r2_is),
        "r2_oos": float(r2_oos),
        "sign_acc_oos": float(sign_acc_oos),
        "coefs": coefs,
        "intercept": float(m_full.intercept_) if X.shape[1] > 0 else float(y.mean()),
        "train_mean_per_fold": train_mean_per_fold,
    }


def evaluate_models(df: pd.DataFrame, panel_name: str) -> dict:
    """Evaluate M0..M4 on this sub-panel."""
    y = df[f"outperf_{H}"].to_numpy(dtype=float)
    out = {"panel": panel_name, "n_obs": len(df)}

    X0, cols0 = design_M0(df)
    out["M0_intercept_only"] = cv_r2(X0, y)

    X1, cols1 = design_M1(df)
    out["M1_era_only"] = cv_r2(X1, y)
    out["M1_era_only"]["columns"] = cols1

    X2, cols2 = design_M2(df)
    out["M2_era_x_vix"] = cv_r2(X2, y)
    out["M2_era_x_vix"]["columns"] = cols2

    X3, cols3 = design_M3(df)
    out["M3_era_x_vix_x_shocktype"] = cv_r2(X3, y)
    out["M3_era_x_vix_x_shocktype"]["columns"] = cols3

    X4, cols4 = design_M4(df)
    out["M4_era_x_vix_plus_shock_main"] = cv_r2(X4, y)
    out["M4_era_x_vix_plus_shock_main"]["columns"] = cols4

    # Deltas vs M2 baseline (M2 = era x VIX-bin only)
    r2_M2 = out["M2_era_x_vix"]["r2_oos"]
    out["delta_r2_oos_M2_to_M3_full_interaction"] = (
        out["M3_era_x_vix_x_shocktype"]["r2_oos"] - r2_M2
    )
    out["delta_r2_oos_M2_to_M4_additive_shock"] = (
        out["M4_era_x_vix_plus_shock_main"]["r2_oos"] - r2_M2
    )

    return out


# ---------------------------------------------------------------------------
# Permutation test (shuffle shock-type labels)
# ---------------------------------------------------------------------------
def permutation_test(df: pd.DataFrame, n_perm: int = N_PERMUTATIONS,
                      seed: int = SEED) -> dict:
    """Null: shock labels are non-informative beyond era x VIX-bin.
    Procedure: shuffle the RETAINED_SHOCKS columns (per-day, holding cell12 and
    outcome fixed). Re-build M3 / M4 designs and recompute DR^2 (M2 -> M3, M2 -> M4).

    We shuffle the shock-active days WITHIN each era x VIX-bin cell to preserve
    the marginal of shock incidence per cell (otherwise we'd contaminate the
    test with cell-incidence imbalance).
    """
    rng = np.random.default_rng(seed)
    y = df[f"outperf_{H}"].to_numpy(dtype=float)
    sst = float(np.sum((y - y.mean()) ** 2))

    # Observed M2 R^2 (anchor)
    X2, _ = design_M2(df)
    M2_oos = cv_r2(X2, y)["r2_oos"]

    # Observed M3, M4
    X3_obs, _ = design_M3(df)
    obs_M3 = cv_r2(X3_obs, y)["r2_oos"] - M2_oos
    X4_obs, _ = design_M4(df)
    obs_M4 = cv_r2(X4_obs, y)["r2_oos"] - M2_oos

    df_perm = df.copy()
    df_perm["cell12"] = df_perm["era"].astype(str) + "_" + df_perm["vix_bin"].astype(str)
    cell12_arr = df_perm["cell12"].values

    null_M3 = []
    null_M4 = []
    for p in range(n_perm):
        df_p = df.copy()
        # Within each era x VIX cell, permute the *block* of shock columns
        # (RETAINED_SHOCKS) jointly so the per-day shock vector stays intact;
        # this preserves the within-cell shock incidence rate AND the
        # correlation structure across shocks.
        for cell in pd.unique(cell12_arr):
            idx = np.where(cell12_arr == cell)[0]
            if len(idx) < 2:
                continue
            perm = rng.permutation(idx)
            df_p.iloc[idx, df_p.columns.get_indexer(RETAINED_SHOCKS)] = (
                df.iloc[perm][RETAINED_SHOCKS].values
            )
        X3p, _ = design_M3(df_p)
        X4p, _ = design_M4(df_p)
        null_M3.append(cv_r2(X3p, y)["r2_oos"] - M2_oos)
        null_M4.append(cv_r2(X4p, y)["r2_oos"] - M2_oos)
        if (p + 1) % 25 == 0:
            print(f"  [perm] {p+1}/{n_perm}", flush=True)
    null_M3 = np.asarray(null_M3)
    null_M4 = np.asarray(null_M4)

    # One-sided p-value (greater)
    p_M3 = float((np.sum(null_M3 >= obs_M3) + 1) / (n_perm + 1))
    p_M4 = float((np.sum(null_M4 >= obs_M4) + 1) / (n_perm + 1))

    return {
        "n_permutations": n_perm,
        "observed_delta_r2_oos_M2_to_M3": float(obs_M3),
        "observed_delta_r2_oos_M2_to_M4": float(obs_M4),
        "null_M3_summary": {
            "mean": float(np.mean(null_M3)),
            "std": float(np.std(null_M3, ddof=1)),
            "q05": float(np.quantile(null_M3, 0.05)),
            "q50": float(np.quantile(null_M3, 0.50)),
            "q95": float(np.quantile(null_M3, 0.95)),
        },
        "null_M4_summary": {
            "mean": float(np.mean(null_M4)),
            "std": float(np.std(null_M4, ddof=1)),
            "q05": float(np.quantile(null_M4, 0.05)),
            "q50": float(np.quantile(null_M4, 0.50)),
            "q95": float(np.quantile(null_M4, 0.95)),
        },
        "permutation_p_value_M3_one_sided_greater": p_M3,
        "permutation_p_value_M4_one_sided_greater": p_M4,
    }


# ---------------------------------------------------------------------------
# Per-shock loadings (M4)
# ---------------------------------------------------------------------------
def shock_loadings_M4(df: pd.DataFrame) -> dict:
    """M4 ridge coefficients on the shock dummies (mean outperf vs baseline shock).
    Plus simple within-panel mean outperf by shock-type for transparency.
    """
    X4, cols4 = design_M4(df)
    y = df[f"outperf_{H}"].to_numpy(dtype=float)
    if X4.shape[1] == 0:
        return {"available": False}
    m = Ridge(alpha=RIDGE_LAMBDA, fit_intercept=True)
    m.fit(X4, y)
    coefs = dict(zip(cols4, m.coef_.tolist()))

    # Per-shock mean outperf (raw)
    df = df.copy()
    n_active = df[RETAINED_SHOCKS].fillna(0).sum(axis=1).astype(int)
    df["shock_type"] = np.where(n_active == 0, "none",
                                  np.where(n_active == 1,
                                            df[RETAINED_SHOCKS].idxmax(axis=1),
                                            "multi"))
    per_shock = df.groupby("shock_type")[f"outperf_{H}"].agg(["count", "mean", "std"]).to_dict()
    return {
        "available": True,
        "M4_ridge_coefs": coefs,
        "per_shock_raw_outperf": per_shock,
    }


# ---------------------------------------------------------------------------
# Verdict logic (per the task spec)
# ---------------------------------------------------------------------------
def verdict(delta_M3_full: float, p_full_M3: float,
            delta_M3_shock: float, p_shock_M3: float) -> dict:
    """Returns a verdict string per the spec.

    The CRITICAL HONESTY rule says:
    - DR^2 >= +2pp on full panel AND p < 0.10 -> 'KEEP_FULL_STRUCTURE'
    - DR^2 <  +1pp on full panel              -> 'SIMPLIFY_TO_ERA_X_VIX'
    - 1pp <= DR^2 < 2pp                       -> 'BORDERLINE'

    Operator override: banking-stress safe-haven survives REGARDLESS as a
    separate override (already confirmed by SVB direct test).
    """
    dpp_full = delta_M3_full * 100  # convert to pp
    dpp_shock = delta_M3_shock * 100
    if dpp_full >= 2.0 and (p_full_M3 is not None) and p_full_M3 < 0.10:
        cls = "KEEP_FULL_STRUCTURE"
        msg = (f"Full-panel DR^2 (M2 -> M3) = {dpp_full:+.2f}pp at p={p_full_M3:.3f} (< 0.10): "
               f"shock-type adds operational signal beyond era x VIX-bin. Keep the full lookup-table structure.")
    elif dpp_full < 1.0:
        cls = "SIMPLIFY_TO_ERA_X_VIX"
        msg = (f"Full-panel DR^2 (M2 -> M3) = {dpp_full:+.2f}pp at p={p_full_M3:.3f}: "
               f"shock-type adds essentially no predictive value. Simplify dashboard to era x VIX-bin (per-shock breakdown is descriptive metadata, not predictive). "
               f"Banking-stress safe-haven survives as a SEPARATE override (already confirmed by SVB direct test).")
    else:
        cls = "BORDERLINE"
        msg = (f"Full-panel DR^2 (M2 -> M3) = {dpp_full:+.2f}pp at p={p_full_M3:.3f}: "
               f"borderline. Report both M2-only and M3 cells; let the operator decide. "
               f"Banking-stress override stands regardless.")
    return {
        "class": cls,
        "message": msg,
        "shock_active_panel_delta_pp": dpp_shock,
        "shock_active_panel_p_M3": p_shock_M3,
    }


# ---------------------------------------------------------------------------
# Markdown + LaTeX writers
# ---------------------------------------------------------------------------
def _fmt(x, p=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{p}f}"


def write_report(full_out: dict, shock_out: dict, perm_full: dict,
                 perm_shock: dict, load_full: dict, load_shock: dict,
                 v: dict, meta: dict, path: Path):
    lines = []
    lines.append("# Does Shock-Type Add OOS R² Beyond Era × VIX-bin? -- Formal Test")
    lines.append("")
    lines.append(f"Generated: {meta['generated_at']}  |  Seed: {SEED}  |  CV folds: {N_FOLDS}  |  Permutations: {N_PERMUTATIONS}")
    lines.append("")
    lines.append("## Outcome and samples")
    lines.append(f"- Outcome: 60-calendar-day forward outperformance (r_BTC,cal − r_SPY,nyse), n_full={full_out['n_obs']}, n_shock_active={shock_out['n_obs']}.")
    lines.append(f"- Cross-validation: 5-fold KFold(shuffle=True, random_state={SEED}); Ridge(α=1e-4).")
    lines.append(f"- Permutation: shuffle RETAINED_SHOCK columns (jointly per-day) WITHIN era×VIX cell, preserving cell-level shock incidence and outcome alignment.")
    lines.append("")
    lines.append("## Models")
    lines.append("- M0 intercept-only; M1 era dummies; M2 era×VIX-bin (12 cells); M3 era×VIX-bin×shock-type (up to 84 cells); M4 era×VIX-bin + additive shock-type main effects.")
    lines.append("")
    # ----------- R^2 table -----------
    def _row(panel, out):
        for k, label in [
            ("M0_intercept_only", "M0 intercept-only"),
            ("M1_era_only", "M1 era only"),
            ("M2_era_x_vix", "M2 era×VIX (12 cells)"),
            ("M3_era_x_vix_x_shocktype", "M3 era×VIX×shock-type"),
            ("M4_era_x_vix_plus_shock_main", "M4 era×VIX + shock main eff."),
        ]:
            m = out[k]
            yield (panel, label, m["n_features"], m["r2_is"], m["r2_oos"], m["sign_acc_oos"])

    lines.append("## R² results")
    lines.append("")
    lines.append("| Panel | Model | k | R² IS | R² OOS (5-fold) | Sign-acc OOS |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for tup in list(_row("Full panel", full_out)) + list(_row("Shock-active", shock_out)):
        panel, lab, k, r2is, r2oos, sa = tup
        lines.append(f"| {panel} | {lab} | {k} | {_fmt(r2is, 4)} | {_fmt(r2oos, 4)} | {_fmt(sa, 3)} |")
    lines.append("")

    # ----------- ΔR² + permutation -----------
    lines.append("## ΔR² (vs M2 baseline) with permutation p-value")
    lines.append("")
    lines.append("| Panel | Comparison | Observed ΔR² (pp) | Null mean (pp) | Null 95th pctl (pp) | p (one-sided > ) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for panel, fo, pp in [("Full panel", full_out, perm_full),
                           ("Shock-active", shock_out, perm_shock)]:
        lines.append(f"| {panel} | M2 → M3 (full interaction) | "
                     f"{fo['delta_r2_oos_M2_to_M3_full_interaction']*100:+.2f} | "
                     f"{pp['null_M3_summary']['mean']*100:+.2f} | "
                     f"{pp['null_M3_summary']['q95']*100:+.2f} | "
                     f"{pp['permutation_p_value_M3_one_sided_greater']:.3f} |")
        lines.append(f"| {panel} | M2 → M4 (additive shock) | "
                     f"{fo['delta_r2_oos_M2_to_M4_additive_shock']*100:+.2f} | "
                     f"{pp['null_M4_summary']['mean']*100:+.2f} | "
                     f"{pp['null_M4_summary']['q95']*100:+.2f} | "
                     f"{pp['permutation_p_value_M4_one_sided_greater']:.3f} |")
    lines.append("")
    lines.append("Note: A NEGATIVE OOS ΔR² is possible when extra parameters add noise faster than signal; "
                 "the null permutation distribution is centred near zero (or negative) because the test "
                 "shuffles shock labels within era×VIX cells, so the average permuted model has no signal "
                 "beyond M2 but pays the same parameter-count penalty in OOS evaluation.")
    lines.append("")

    # ----------- Per-shock loadings -----------
    lines.append("## Per-shock loadings (M4, additive)")
    lines.append("")
    for panel, lo in [("Full panel", load_full), ("Shock-active", load_shock)]:
        lines.append(f"### {panel}")
        if not lo.get("available"):
            lines.append("_M4 design empty for this panel._")
            continue
        # Ridge coefs on shock_* terms only
        shock_coefs = {k: v for k, v in lo["M4_ridge_coefs"].items() if k.startswith("shock_")}
        if not shock_coefs:
            lines.append("_No shock dummies in M4 design (baseline absorbs all)._")
            continue
        lines.append("M4 ridge coefficients on shock-type dummies (vs the dropped baseline shock-type):")
        lines.append("")
        lines.append("| Shock dummy | Coef (decimal) | Coef (pp) |")
        lines.append("|---|---:|---:|")
        for k, c in sorted(shock_coefs.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"| {k} | {c:+.4f} | {c*100:+.2f} |")
        lines.append("")
        # Per-shock raw mean
        raw_count = lo["per_shock_raw_outperf"]["count"]
        raw_mean = lo["per_shock_raw_outperf"]["mean"]
        raw_std = lo["per_shock_raw_outperf"]["std"]
        lines.append("Raw 60d outperformance by shock-type (unconditional):")
        lines.append("")
        lines.append("| Shock-type | n | Mean outperf (pp) | Std (pp) |")
        lines.append("|---|---:|---:|---:|")
        for k in sorted(raw_count.keys(), key=lambda x: -raw_count[x]):
            n = raw_count[k]
            mu = raw_mean[k]
            sd = raw_std[k] if raw_std[k] is not None else float("nan")
            lines.append(f"| {k} | {int(n)} | {mu*100:+.2f} | {sd*100:.2f} |")
        lines.append("")

    # ----------- Verdict -----------
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{v['class']}** -- {v['message']}")
    lines.append("")
    lines.append(f"Shock-active panel context: ΔR² (M2→M3) = {v['shock_active_panel_delta_pp']:+.2f}pp, p = {v['shock_active_panel_p_M3']:.3f}.")
    lines.append("")
    lines.append("## Honesty notes")
    lines.append("")
    lines.append("- If shock-type doesn't add OOS R² beyond era×VIX-bin, the dashboard's per-shock structure is **descriptive metadata, not predictive**. Banking-stress safe-haven is a separate override (already confirmed by SVB direct test).")
    lines.append("- The permutation null is centred slightly negative (shuffled labels cost OOS R² because they add design-matrix noise without signal); observed deltas should be evaluated against the **null distribution** rather than vs zero.")
    lines.append("- A positive small ΔR² (1-2pp) is borderline -- M4 (additive) vs M3 (interactive) shape of the signal matters for the operational interface choice; pick whichever survives at the operator's chosen p threshold.")
    lines.append("")
    path.write_text("\n".join(lines))


def write_tex(full_out: dict, shock_out: dict, perm_full: dict, perm_shock: dict,
              path: Path):
    rows = []
    for panel, fo, pp in [("Full panel", full_out, perm_full),
                          ("Shock-active", shock_out, perm_shock)]:
        for label, k_full, k_perm in [
            ("M0 intercept", "M0_intercept_only", None),
            ("M1 era", "M1_era_only", None),
            ("M2 era$\\times$VIX", "M2_era_x_vix", None),
            ("M3 era$\\times$VIX$\\times$shock", "M3_era_x_vix_x_shocktype", "M3"),
            ("M4 era$\\times$VIX + shock", "M4_era_x_vix_plus_shock_main", "M4"),
        ]:
            m = fo[k_full]
            dr2 = ""
            pval = ""
            if k_perm == "M3":
                dr2 = f"{fo['delta_r2_oos_M2_to_M3_full_interaction']*100:+.2f}"
                pval = f"{pp['permutation_p_value_M3_one_sided_greater']:.3f}"
            if k_perm == "M4":
                dr2 = f"{fo['delta_r2_oos_M2_to_M4_additive_shock']*100:+.2f}"
                pval = f"{pp['permutation_p_value_M4_one_sided_greater']:.3f}"
            rows.append({
                "Panel": panel, "Model": label, "$k$": m["n_features"],
                "$R^2$ IS": f"{m['r2_is']*100:+.2f}",
                "$R^2$ OOS": f"{m['r2_oos']*100:+.2f}",
                "$\\Delta R^2$ OOS (pp)": dr2,
                "Perm $p$": pval,
            })
    df = pd.DataFrame(rows)
    tex = df.to_latex(index=False, escape=False,
        caption=("Does shock-type add OOS R$^2$ beyond era$\\times$VIX-bin? "
                 "$R^2$ reported in percent (pp); $\\Delta R^2$ vs M2 baseline. "
                 "Permutation $p$ from shuffling shock labels within era$\\times$VIX cells "
                 f"({N_PERMUTATIONS} draws, seed={SEED}). 5-fold OOS CV, Ridge $\\lambda=10^{{-4}}$."),
        label="tab:shock_adds_value")
    path.write_text(tex)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    panel = build_panel()
    print(f"[shock_adds_value] full panel n = {len(panel)}", flush=True)
    shock_panel = panel[panel["any_shock"] == 1].copy()
    print(f"[shock_adds_value] shock-active panel n = {len(shock_panel)}", flush=True)

    full_out = evaluate_models(panel, "full")
    shock_out = evaluate_models(shock_panel, "shock_active")

    print("[shock_adds_value] running permutation test on full panel...", flush=True)
    perm_full = permutation_test(panel)
    print("[shock_adds_value] running permutation test on shock-active panel...", flush=True)
    perm_shock = permutation_test(shock_panel)

    load_full = shock_loadings_M4(panel)
    load_shock = shock_loadings_M4(shock_panel)

    v = verdict(
        full_out["delta_r2_oos_M2_to_M3_full_interaction"],
        perm_full["permutation_p_value_M3_one_sided_greater"],
        shock_out["delta_r2_oos_M2_to_M3_full_interaction"],
        perm_shock["permutation_p_value_M3_one_sided_greater"],
    )

    meta = {
        "script": "code/test_shock_adds_value.py",
        "panel_path": str(PANEL_PATH),
        "btc_calendar_path": str(BTC_CAL_PATH),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_permutations": N_PERMUTATIONS,
        "ridge_lambda": RIDGE_LAMBDA,
        "horizon_calendar_days": H,
        "retained_shocks": RETAINED_SHOCKS,
    }
    out_doc = {
        "_metadata": meta,
        "full_panel": full_out,
        "shock_active_panel": shock_out,
        "permutation_full_panel": perm_full,
        "permutation_shock_active_panel": perm_shock,
        "per_shock_loadings_full_panel": load_full,
        "per_shock_loadings_shock_active_panel": load_shock,
        "verdict": v,
    }
    # JSON-safe
    def _safe(o):
        if isinstance(o, dict):
            return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_safe(v) for v in o]
        if isinstance(o, np.ndarray):
            return _safe(o.tolist())
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    OUT_JSON.write_text(json.dumps(_safe(out_doc), indent=2, default=str))
    print(f"[shock_adds_value] wrote {OUT_JSON}", flush=True)

    write_report(full_out, shock_out, perm_full, perm_shock, load_full, load_shock,
                 v, meta, OUT_MD)
    print(f"[shock_adds_value] wrote {OUT_MD}", flush=True)
    write_tex(full_out, shock_out, perm_full, perm_shock, OUT_TEX)
    print(f"[shock_adds_value] wrote {OUT_TEX}", flush=True)


if __name__ == "__main__":
    main()
