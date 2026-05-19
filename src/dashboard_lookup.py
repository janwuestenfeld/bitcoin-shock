"""
Dashboard lookup table: per-shock x VIX-bin x era forward-return distributions
on a CALENDAR-DAY basis.

CALENDAR-DAY OPERATIONAL DASHBOARD (not NYSE-trading-day).
  - BTC trades 24/7/365 -> forward returns measured at calendar t+h (h in days)
  - SPY/VIX trade NYSE only -> for backward state at t and forward SPY at t+h,
    use the nearest NYSE day (<= t for backward, >= t+h for forward).
  - This re-build replaces the prior trading-day version and aligns the
    dashboard with how operational users (and the BlackRock chart) read
    "+10 days" or "+60 days" from an event date.

Builds:
  - Calendar-day BTC forward returns for h in {5, 20, 60, 90} calendar days
    using data/aux/btc_calendar_daily.parquet.
  - SPY forward returns at the nearest NYSE day >= (t + h calendar) using the
    panel's spy column.
  - Outperformance = r_BTC_calendar_fwd(t->t+h) - r_SPY_NYSE_fwd(t->nearest NYSE >= t+h).

For each (shock, vix_bin, era, horizon) cell we record:
  n, distribution moments (mean/std/q05/q25/q50/q75/q95/share_positive) of:
      r_btc_fwd_h (calendar), r_spy_fwd_h (nearest NYSE), outperf_btc_minus_spy

Sparse cells (n < 20) flagged and fall back to (shock, era) then (shock)
aggregates.

Headline basis: post-COVID (cleaner regime, per the paper). Pre-COVID cells are
computed but tagged 'different_regime_use_with_caution'.

Reproducible: numpy seed=42, inputs logged.

Outputs:
  output/stage3a/results/dashboard_lookup_table.json    (the lookup table)
  data/aux/btc_calendar_daily.parquet                   (BTC calendar series)

Horizon menu:
  h=5, 20, 60 are the original BlackRock-style operational horizons. h=90 was
  added 2026-05-19 to provide a window-sensitivity diagnostic next to h=60:
  BTC outperformance near 60d is fragile (e.g., the Russia-Ukraine 60-vs-90d
  flip from +8.7pp to large underperformance once the 60->90 window catches
  the May 2022 LUNA/Terra collapse), so reporting h=90 next to h=60 disclosures
  the window-fragility.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Reproducibility
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "output" / "seed" / "paper1_context" / "panel_with_shocks.parquet"
BTC_CAL_PATH = ROOT / "data" / "aux" / "btc_calendar_daily.parquet"
OUT_DIR = ROOT / "output" / "stage3a" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "dashboard_lookup_table.json"
OUT_PATH_EWMA = OUT_DIR / "dashboard_lookup_table_ewma.json"
EWMA_SHOCKS_PATH = ROOT / "data" / "aux" / "ewma_shocks_panel.parquet"
# _EWMA_PATCH_APPLIED_v1

# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------
SHOCK_KEYS = [
    "oil_shock",
    "dollar_shock",
    "rate_shock",
    "banking_shock",          # STLFSI4-based banking stress
    "gprd_threat_shock",
    "none",                   # no retained shock active
    "any",                    # any of the five
]
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

ERAS = [
    ("pre_covid", "2014-01-02", "2020-02-29"),
    ("post_covid_pre_etf", "2020-03-01", "2024-01-09"),
    ("post_etf", "2024-01-10", "2099-12-31"),
]

HORIZONS = [5, 20, 60, 90]   # CALENDAR days

# Headline framework constants (from the paper)
TAU_UNIVERSAL_POSTCOVID = 14.5
DBETA_INDEP_HANSEN_POSTCOVID = {
    "oil_shock": 0.74,
    "dollar_shock": 0.51,
    "rate_shock": 0.48,
    "gprd_threat_shock": 0.38,
    "banking_shock": -0.33,   # safe-haven (independent Hansen)
}
N_MIN = 20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def vix_bin_of(vix: float) -> str:
    if vix is None or (isinstance(vix, float) and np.isnan(vix)):
        return "calm"
    for name, lo, hi in VIX_BINS:
        if lo <= vix < hi:
            return name
    return "calm"


def era_of(date: pd.Timestamp) -> str:
    for name, lo, hi in ERAS:
        if pd.Timestamp(lo) <= date <= pd.Timestamp(hi):
            return name
    return "pre_covid"


def _summ(arr: np.ndarray) -> dict:
    """Return summary dict for a 1-D array (filtered NaN)."""
    x = arr[~np.isnan(arr)]
    n = int(x.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None, "std": None,
            "q05": None, "q25": None, "q50": None, "q75": None, "q95": None,
            "share_positive": None,
        }
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if n > 1 else 0.0,
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "share_positive": float(np.mean(x > 0)),
    }


# ---------------------------------------------------------------------------
# BTC calendar-day series management
# ---------------------------------------------------------------------------
def fetch_coinmetrics_btc(start: str, end: str) -> pd.DataFrame:
    """Pull BTC daily PriceUSD from the CoinMetrics community API."""
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    rows = []
    next_page_token = None
    page = 0
    while True:
        params = {
            "assets": "btc",
            "metrics": "PriceUSD",
            "frequency": "1d",
            "start_time": start,
            "end_time": end,
            "page_size": 10000,
        }
        if next_page_token:
            params["next_page_token"] = next_page_token
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        page += 1
        chunk = d.get("data", [])
        rows.extend(chunk)
        next_page_token = d.get("next_page_token")
        if not next_page_token or not chunk:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.normalize().dt.tz_localize(None)
    df["close"] = pd.to_numeric(df["PriceUSD"], errors="coerce")
    df = df[["date", "close"]].drop_duplicates(subset="date").sort_values("date")
    return df.set_index("date")


def ensure_btc_calendar() -> pd.DataFrame:
    """Load or rebuild the BTC calendar-day series."""
    if BTC_CAL_PATH.exists():
        cal = pd.read_parquet(BTC_CAL_PATH)
        cal.index = pd.to_datetime(cal.index)
        return cal
    # Rebuild
    BTC_CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.today().normalize()
    print(f"[dashboard_lookup] fetching BTC calendar series: 2014-01-02 -> {end.date()}", flush=True)
    cm = fetch_coinmetrics_btc("2014-01-01", str(end.date()))
    all_days = pd.date_range(pd.Timestamp("2014-01-02"), end, freq="D")
    close = cm["close"].reindex(all_days)
    source = pd.Series(["coinmetrics"] * len(close), index=close.index)
    source[close.isna()] = "missing"
    close_ff = close.ffill(limit=2)
    log_ret = np.log(close_ff / close_ff.shift(1))
    out = pd.DataFrame({
        "close": close_ff.values,
        "log_return": log_ret.values,
        "close_raw": close.values,
        "source": source.values,
    }, index=close_ff.index)
    out.index.name = "date"
    out.to_parquet(BTC_CAL_PATH)
    return out


# ---------------------------------------------------------------------------
# Calendar-day forward-return computation
# ---------------------------------------------------------------------------
def btc_calendar_fwd_returns(btc_cal: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Compute BTC simple forward returns at calendar h-day horizons.

    Returns a DataFrame indexed by every calendar date in btc_cal.index, with
    one column per horizon: r_btc_calendar_fwd_h = close(t+h)/close(t) - 1.
    NaN if t+h is outside btc_cal.index or close(t)/close(t+h) is NaN.
    """
    close = btc_cal["close"].astype(float)
    # Build a fast lookup by index
    out = pd.DataFrame(index=close.index)
    for h in horizons:
        out[f"r_btc_calendar_fwd_{h}"] = close.shift(-h) / close - 1.0
    return out


def spy_nearest_nyse_fwd(panel: pd.DataFrame, target_dates: pd.DatetimeIndex,
                          horizons: list[int]) -> pd.DataFrame:
    """For each target calendar date t and horizon h, compute:
      SPY forward return = spy(NYSE_nearest_ge(t+h)) / spy(NYSE_nearest_le(t)) - 1
    Both sides snap to the panel's NYSE-day SPY series.

    Returns DataFrame indexed by target_dates with columns:
      r_spy_nyse_fwd_h
      nyse_t_used_h     (date used for SPY(t))
      nyse_tph_used_h   (date used for SPY(t+h))
      nyse_gap_t_days_h (calendar gap between target t and nyse_t_used)
      nyse_gap_tph_days_h (calendar gap between target t+h and nyse_tph_used)
    """
    nyse_idx = panel.index.sort_values()
    spy = panel["spy"].astype(float)
    # Pre-build sorted numpy arrays for fast searchsorted
    nyse_arr = np.asarray(nyse_idx.values)  # datetime64

    out = pd.DataFrame(index=target_dates)
    # Backward-snap: NYSE_nearest_le(t)
    t_dates = np.asarray(target_dates.values)
    pos_le_t = np.searchsorted(nyse_arr, t_dates, side="right") - 1  # idx of largest nyse <= t
    pos_le_t = np.clip(pos_le_t, 0, len(nyse_arr) - 1)
    nyse_le_t = nyse_arr[pos_le_t]
    valid_le_t = (nyse_le_t <= t_dates) & (pos_le_t >= 0)

    spy_t_vals = np.where(valid_le_t, spy.values[pos_le_t], np.nan)

    for h in horizons:
        tph = t_dates + np.timedelta64(h, "D")
        # Forward-snap: NYSE_nearest_ge(t+h)
        pos_ge_tph = np.searchsorted(nyse_arr, tph, side="left")  # idx of smallest nyse >= t+h
        valid_ge_tph = pos_ge_tph < len(nyse_arr)
        pos_ge_tph_clip = np.clip(pos_ge_tph, 0, len(nyse_arr) - 1)
        nyse_ge_tph = nyse_arr[pos_ge_tph_clip]
        spy_tph_vals = np.where(valid_ge_tph & valid_le_t,
                                 spy.values[pos_ge_tph_clip], np.nan)
        spy_t_vals_h = np.where(valid_ge_tph & valid_le_t, spy_t_vals, np.nan)

        ret = spy_tph_vals / spy_t_vals_h - 1.0
        gap_t = (t_dates - nyse_le_t).astype("timedelta64[D]").astype(int)
        gap_tph = (nyse_ge_tph - tph).astype("timedelta64[D]").astype(int)

        out[f"r_spy_nyse_fwd_{h}"] = ret
        out[f"nyse_t_used_{h}"] = pd.to_datetime(nyse_le_t)
        out[f"nyse_tph_used_{h}"] = np.where(valid_ge_tph,
                                              pd.to_datetime(nyse_ge_tph),
                                              pd.NaT)
        out[f"nyse_gap_t_days_{h}"] = gap_t
        out[f"nyse_gap_tph_days_{h}"] = np.where(valid_ge_tph, gap_tph, np.nan)

    return out


# ---------------------------------------------------------------------------
# Build panel with calendar-day forward returns merged in
# ---------------------------------------------------------------------------
def prepare_panel(shock_basis: str = "fullsample") -> pd.DataFrame:
    """Build a working dataframe for cell-conditional lookups.

    The rows in this dataframe are still NYSE-trading-days (the days where we
    observe VIX/SPY/shock indicators -- the conditioning variables). For each
    such NYSE day t we ATTACH:
      - r_btc_calendar_fwd_h: BTC return over h calendar days
        from CoinMetrics (yes, available because BTC trades 24/7).
      - r_spy_nyse_fwd_h: SPY return from t to the nearest NYSE day >= (t+h calendar).
      - outperf_h = r_btc_calendar_fwd_h - r_spy_nyse_fwd_h.

    This is the operational shape: conditioning on the NYSE state at t, predict
    the calendar-day BTC trajectory (and SPY-baseline) going forward h days.
    """
    df = pd.read_parquet(PANEL_PATH).copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df["date"] = df.index

    # BTC calendar series
    btc_cal = ensure_btc_calendar()
    btc_fwd = btc_calendar_fwd_returns(btc_cal, HORIZONS)
    # Subset to the panel's NYSE days
    btc_fwd_at_nyse = btc_fwd.reindex(df.index)

    # SPY nearest-NYSE forward returns
    spy_fwd = spy_nearest_nyse_fwd(df, df.index, HORIZONS)

    # Era and VIX bin
    df["era"] = df["date"].apply(era_of)
    df["vix_bin"] = df["vix"].apply(vix_bin_of)

    # Attach forward returns
    for h in HORIZONS:
        df[f"r_btc_calendar_fwd_{h}"] = btc_fwd_at_nyse[f"r_btc_calendar_fwd_{h}"].values
        df[f"r_spy_nyse_fwd_{h}"] = spy_fwd[f"r_spy_nyse_fwd_{h}"].values
        df[f"outperf_calendar_fwd_{h}"] = (
            df[f"r_btc_calendar_fwd_{h}"] - df[f"r_spy_nyse_fwd_{h}"]
        )
        df[f"nyse_gap_tph_days_{h}"] = spy_fwd[f"nyse_gap_tph_days_{h}"].values
        df[f"beta_60_calendar_drift_{h}"] = (
            df["beta_60"].shift(-h) - df["beta_60"]  # h trading-rows ahead -- approximate
        )

    # Optionally swap fullsample shock indicators for EWMA-126 rolling-cutoff indicators
    if shock_basis == "ewma126":
        if not EWMA_SHOCKS_PATH.exists():
            raise FileNotFoundError(
                f"EWMA shocks parquet missing: {EWMA_SHOCKS_PATH}. "
                f"Run: python code/build_ewma_shocks.py"
            )
        ewma = pd.read_parquet(EWMA_SHOCKS_PATH).copy()
        ewma.index = pd.to_datetime(ewma.index)
        for s in RETAINED_SHOCKS:
            col = f"{s}_ewma"
            if col not in ewma.columns:
                raise KeyError(f"EWMA panel missing column: {col}")
            # Convert to nullable Int (preserve NaN -> dropped from any/none sums)
            vals = ewma[col].reindex(df.index)
            df[s] = vals.astype("Float64")          # keep NaN where EWMA undefined
    elif shock_basis != "fullsample":
        raise ValueError(f"unknown shock_basis: {shock_basis!r}")

    # 'any' shock indicator
    df["any_shock"] = df[RETAINED_SHOCKS].fillna(0).sum(axis=1).clip(upper=1).astype(int)
    df["no_shock"] = (df["any_shock"] == 0).astype(int)
    df["shock_basis"] = shock_basis
    return df


# ---------------------------------------------------------------------------
# Cell membership and record
# ---------------------------------------------------------------------------
def shock_mask(df: pd.DataFrame, shock: str) -> pd.Series:
    if shock == "any":
        return df["any_shock"] == 1
    if shock == "none":
        return df["no_shock"] == 1
    if shock in RETAINED_SHOCKS:
        # Coerce to float (handles nullable Int64 and Float64 from EWMA path)
        s = pd.to_numeric(df[shock], errors="coerce").fillna(0).astype(float)
        return s == 1.0
    raise ValueError(f"Unknown shock: {shock}")


def era_mask(df: pd.DataFrame, era: str) -> pd.Series:
    return df["era"] == era


def vix_mask(df: pd.DataFrame, vbin: str) -> pd.Series:
    return df["vix_bin"] == vbin


def cell_record(df: pd.DataFrame, horizon: int) -> dict:
    """Compute distribution moments for a sub-panel at a given horizon."""
    n = int(len(df))
    if n == 0:
        return {
            "n": 0,
            "r_btc_calendar_fwd": _summ(np.array([])),
            "r_spy_nyse_fwd": _summ(np.array([])),
            "outperf_calendar_fwd": _summ(np.array([])),
            "beta_60_drift_t_to_tph": {"n": 0, "mean": None, "std": None},
            "nyse_gap_tph_days_mean": None,
        }
    btc_fwd = df[f"r_btc_calendar_fwd_{horizon}"].to_numpy(dtype=float)
    spy_fwd = df[f"r_spy_nyse_fwd_{horizon}"].to_numpy(dtype=float)
    out = df[f"outperf_calendar_fwd_{horizon}"].to_numpy(dtype=float)
    bdrift = df[f"beta_60_calendar_drift_{horizon}"].to_numpy(dtype=float)
    gap = df[f"nyse_gap_tph_days_{horizon}"].to_numpy(dtype=float)

    return {
        "n": n,
        "r_btc_calendar_fwd": _summ(btc_fwd),
        "r_spy_nyse_fwd": _summ(spy_fwd),
        "outperf_calendar_fwd": _summ(out),
        "beta_60_drift_t_to_tph": {
            "n": int(np.sum(~np.isnan(bdrift))),
            "mean": float(np.nanmean(bdrift)) if np.any(~np.isnan(bdrift)) else None,
            "std": float(np.nanstd(bdrift, ddof=1)) if np.sum(~np.isnan(bdrift)) > 1 else None,
        },
        "nyse_gap_tph_days_mean": (float(np.nanmean(gap)) if np.any(~np.isnan(gap)) else None),
    }


# ---------------------------------------------------------------------------
# Build the full lookup table
# ---------------------------------------------------------------------------
def build_table(shock_basis: str = "fullsample") -> dict:
    df = prepare_panel(shock_basis=shock_basis)

    metadata = {
        "script": "code/dashboard_lookup.py",
        "panel_path": str(PANEL_PATH),
        "btc_calendar_path": str(BTC_CAL_PATH),
        "panel_n_obs": int(len(df)),
        "panel_date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "btc_calendar_n_days": int(len(ensure_btc_calendar())),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "shock_dim": SHOCK_KEYS,
        "vix_bins": [{"name": n, "lo": (None if not np.isfinite(lo) else lo),
                       "hi": (None if not np.isfinite(hi) else hi)}
                      for n, lo, hi in VIX_BINS],
        "era_dim": [{"name": n, "start": s, "end": e} for n, s, e in ERAS],
        "horizons_calendar_days": HORIZONS,
        "sparse_threshold_n_min": N_MIN,
        "tau_universal_postcovid": TAU_UNIVERSAL_POSTCOVID,
        "delta_beta_indep_hansen_postcovid": DBETA_INDEP_HANSEN_POSTCOVID,
        "headline_basis": "post_covid_pre_etf + post_etf (pre_covid reported but flagged different_regime_use_with_caution)",
        "seed": 42,
        "shock_basis": shock_basis,
        "ewma_shocks_path": (str(EWMA_SHOCKS_PATH) if shock_basis == "ewma126" else None),
        "convention_notes": [
            "Conditioning rows = NYSE-trading-day observations (panel index).",
            "BTC forward returns: simple returns over CALENDAR h days, "
            "from CoinMetrics 24/7 series.",
            "SPY forward returns: from spy(t) to spy(nearest NYSE day >= t+h calendar). "
            "If t+h falls on NYSE, gap = 0; if on a weekend or holiday, gap = 1 to 3 days typically.",
            "outperformance = r_btc_calendar_fwd - r_spy_nyse_fwd  (so SPY return spans a "
            "slightly longer window than h calendar days on average; this is the operational "
            "comparison the dashboard reports).",
            "h=90 added 2026-05-19 as a window-sensitivity diagnostic next to h=60: "
            "BTC near-60d outperformance is window-fragile (e.g., Russia-Ukraine flips "
            "from +8.7pp at 60d to large underperformance at 90d once May 2022 LUNA "
            "collapse enters the window).",
        ],
    }

    cells = {}
    fallback_shock_era = {}
    fallback_shock = {}

    # 3-way cells
    for shock in SHOCK_KEYS:
        smask = shock_mask(df, shock)
        for vbin_name, _, _ in VIX_BINS:
            vmask = vix_mask(df, vbin_name)
            for era_name, _, _ in ERAS:
                emask = era_mask(df, era_name)
                sub = df[smask & vmask & emask]
                key = f"{shock}__{vbin_name}__{era_name}"
                cell_entry = {
                    "shock": shock,
                    "vix_bin": vbin_name,
                    "era": era_name,
                    "n_days": int(len(sub)),
                    "insufficient_data": bool(len(sub) < N_MIN),
                    "regime_warning": ("different_regime_use_with_caution"
                                       if era_name == "pre_covid" else None),
                    "by_horizon_calendar": {},
                }
                for h in HORIZONS:
                    cell_entry["by_horizon_calendar"][str(h)] = cell_record(sub, h)
                if cell_entry["insufficient_data"]:
                    cell_entry["usable_or_fallback"] = "fallback_shock_era"
                    cell_entry["fallback_target"] = f"{shock}__ALLVIX__{era_name}"
                else:
                    cell_entry["usable_or_fallback"] = "primary"
                    cell_entry["fallback_target"] = None
                cells[key] = cell_entry

    # (shock, era) fallbacks
    for shock in SHOCK_KEYS:
        smask = shock_mask(df, shock)
        for era_name, _, _ in ERAS:
            emask = era_mask(df, era_name)
            sub = df[smask & emask]
            key = f"{shock}__ALLVIX__{era_name}"
            fb_entry = {
                "shock": shock,
                "vix_bin": "ALLVIX",
                "era": era_name,
                "n_days": int(len(sub)),
                "insufficient_data": bool(len(sub) < N_MIN),
                "regime_warning": ("different_regime_use_with_caution"
                                   if era_name == "pre_covid" else None),
                "by_horizon_calendar": {},
            }
            for h in HORIZONS:
                fb_entry["by_horizon_calendar"][str(h)] = cell_record(sub, h)
            if fb_entry["insufficient_data"]:
                fb_entry["usable_or_fallback"] = "fallback_shock"
                fb_entry["fallback_target"] = f"{shock}__ALLVIX__ALLERA"
            else:
                fb_entry["usable_or_fallback"] = "primary"
                fb_entry["fallback_target"] = None
            fallback_shock_era[key] = fb_entry

    # (shock) fallbacks
    for shock in SHOCK_KEYS:
        smask = shock_mask(df, shock)
        sub = df[smask]
        key = f"{shock}__ALLVIX__ALLERA"
        fb_entry = {
            "shock": shock,
            "vix_bin": "ALLVIX",
            "era": "ALLERA",
            "n_days": int(len(sub)),
            "insufficient_data": bool(len(sub) < N_MIN),
            "regime_warning": None,
            "by_horizon_calendar": {},
        }
        for h in HORIZONS:
            fb_entry["by_horizon_calendar"][str(h)] = cell_record(sub, h)
        fb_entry["usable_or_fallback"] = (
            "primary" if not fb_entry["insufficient_data"] else "no_fallback_available"
        )
        fb_entry["fallback_target"] = None
        fallback_shock[key] = fb_entry

    return {
        "_metadata": metadata,
        "cells_primary": cells,
        "cells_fallback_shock_era": fallback_shock_era,
        "cells_fallback_shock": fallback_shock,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build dashboard lookup table.")
    p.add_argument("--shock-basis", default="fullsample",
                   choices=["fullsample", "ewma126"],
                   help="fullsample = original top-decile shock cutoffs (2014-2025); "
                        "ewma126 = EWMA-halflife-126-day rolling top-decile cutoffs")
    args = p.parse_args()
    shock_basis = args.shock_basis
    out_path = OUT_PATH_EWMA if shock_basis == "ewma126" else OUT_PATH
    print(f"[dashboard_lookup] reading panel: {PANEL_PATH}", flush=True)
    print(f"[dashboard_lookup] shock_basis: {shock_basis}", flush=True)
    print(f"[dashboard_lookup] reading BTC calendar: {BTC_CAL_PATH}", flush=True)
    if not BTC_CAL_PATH.exists():
        print("[dashboard_lookup] BTC calendar series missing -- will fetch from CoinMetrics", flush=True)
    table = build_table(shock_basis=shock_basis)
    n_primary = len(table["cells_primary"])
    n_sparse = sum(1 for c in table["cells_primary"].values() if c["insufficient_data"])
    n_fb_se = len(table["cells_fallback_shock_era"])
    n_fb_s = len(table["cells_fallback_shock"])
    print(f"[dashboard_lookup] primary cells: {n_primary} "
          f"(sparse: {n_sparse}, well-powered: {n_primary - n_sparse})", flush=True)
    print(f"[dashboard_lookup] (shock,era) fallbacks: {n_fb_se}", flush=True)
    print(f"[dashboard_lookup] (shock) fallbacks: {n_fb_s}", flush=True)
    out_path.write_text(json.dumps(table, indent=2, default=str))
    print(f"[dashboard_lookup] wrote: {out_path} ({out_path.stat().st_size/1024:.1f} KB)",
          flush=True)


if __name__ == "__main__":
    main()
