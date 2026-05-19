"""
Dashboard forecaster (CALENDAR-DAY basis).

Operational dashboard for BTC behavioural regime + forward return forecast,
keyed on CALENDAR dates (BTC trades 24/7; VIX/SPY/shock state snaps to the
nearest NYSE day at and before the query date).

CLI:
    python code/dashboard_forecast.py --date 2026-05-19           # today
    python code/dashboard_forecast.py --date 2025-12-25           # Christmas (NYSE closed)
    python code/dashboard_forecast.py --date 2024-01-10           # post-ETF era start
    python code/dashboard_forecast.py --date 2023-03-09           # SVB
    python code/dashboard_forecast.py --date 2022-02-21           # Russia (Presidents' Day)

Importable:
    from code.dashboard_forecast import forecast
    out = forecast(date="2026-05-19")

The forecaster:
  1. Resolves date (any calendar date OK).
  2. Backward-snaps to most-recent NYSE day for state (VIX, beta_60, shock
     indicators) and reports vix_staleness_days / shock_staleness_days.
  3. Determines era from the calendar date itself.
  4. Looks up (shock, vix_bin, era) cells in the calendar-day lookup table.
  5. Falls back to (shock, era), then (shock) if n < 20.
  6. Combines across active shocks:
       beta prediction:
         If banking_shock active -> SAFE-HAVEN override Delta-beta = -0.33
         (independent Hansen post-COVID; SVB validation confirmed).
         Else: max Delta-beta over active shocks.
       direction prediction: sqrt(n)-weighted cell-mean over active shocks.
  7. Reports top-5 historical analogues (same era, same shock-profile, nearest VIX)
     with their realized calendar-day forward returns.
  8. Notes BlackRock-event matches when cell membership matches one of the 6 events.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "output" / "seed" / "paper1_context" / "panel_with_shocks.parquet"
BTC_CAL_PATH = ROOT / "data" / "aux" / "btc_calendar_daily.parquet"
LOOKUP_PATH = ROOT / "output" / "stage3a" / "results" / "dashboard_lookup_table.json"
BLACKROCK_CAL_PATH = (
    ROOT / "output" / "stage3a" / "results" / "blackrock_6event_calendar_validation.json"
)

RETAINED_SHOCKS = [
    "oil_shock", "dollar_shock", "rate_shock",
    "banking_shock", "gprd_threat_shock",
]
SHOCK_ALIASES = {
    "oil": "oil_shock", "wti": "oil_shock",
    "dollar": "dollar_shock", "usd": "dollar_shock", "uup": "dollar_shock",
    "rate": "rate_shock", "rates": "rate_shock", "y10": "rate_shock",
    "banking": "banking_shock", "bank": "banking_shock",
    "stlfsi": "banking_shock", "stress": "banking_shock",
    "gpr": "gprd_threat_shock", "gpr_threat": "gprd_threat_shock",
    "geopolitical": "gprd_threat_shock",
}

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
HORIZONS_CAL = [5, 20, 60, 90]   # CALENDAR days (h=90 added for window-sensitivity)
TAU = 14.5

# Independent-Hansen post-COVID Delta-beta_k
DBETA_K = {
    "oil_shock": 0.74,
    "dollar_shock": 0.51,
    "rate_shock": 0.48,
    "gprd_threat_shock": 0.38,
    "banking_shock": -0.33,
}

N_MIN = 20

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
_panel_cache: pd.DataFrame | None = None
_btc_cal_cache: pd.DataFrame | None = None
_lookup_cache: dict | None = None
_blackrock_cache: dict | None = None


def load_panel() -> pd.DataFrame:
    global _panel_cache
    if _panel_cache is not None:
        return _panel_cache
    df = pd.read_parquet(PANEL_PATH).copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df["date"] = df.index
    df["era"] = df["date"].apply(era_of)
    df["vix_bin"] = df["vix"].apply(vix_bin_of)
    df["any_shock"] = df[RETAINED_SHOCKS].fillna(0).sum(axis=1).clip(upper=1).astype(int)
    df["no_shock"] = (df["any_shock"] == 0).astype(int)
    _panel_cache = df
    return df


def load_btc_calendar() -> pd.DataFrame:
    global _btc_cal_cache
    if _btc_cal_cache is not None:
        return _btc_cal_cache
    df = pd.read_parquet(BTC_CAL_PATH).copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    _btc_cal_cache = df
    return df


def load_lookup() -> dict:
    global _lookup_cache
    if _lookup_cache is not None:
        return _lookup_cache
    _lookup_cache = json.loads(LOOKUP_PATH.read_text())
    return _lookup_cache


def load_blackrock_calendar() -> dict:
    global _blackrock_cache
    if _blackrock_cache is not None:
        return _blackrock_cache
    try:
        _blackrock_cache = json.loads(BLACKROCK_CAL_PATH.read_text())
    except FileNotFoundError:
        _blackrock_cache = {}
    return _blackrock_cache


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
    date = pd.Timestamp(date)
    for name, lo, hi in ERAS:
        if pd.Timestamp(lo) <= date <= pd.Timestamp(hi):
            return name
    return "pre_covid"


def normalize_shock(s: str) -> str:
    s2 = s.strip().lower()
    if s2 in RETAINED_SHOCKS:
        return s2
    return SHOCK_ALIASES.get(s2, s2)


def normalize_shocks(items: Iterable[str]) -> list[str]:
    out = []
    for s in items:
        n = normalize_shock(s)
        if n in RETAINED_SHOCKS and n not in out:
            out.append(n)
    return out


def resolve_date(date_arg: str) -> pd.Timestamp:
    if date_arg in {"today", None, ""}:
        return pd.Timestamp(datetime.utcnow().date())
    return pd.Timestamp(date_arg)


def backward_nyse_snap(date: pd.Timestamp, panel: pd.DataFrame
                       ) -> tuple[pd.Timestamp, pd.Series, int]:
    """Snap to nearest NYSE day <= date.

    Returns (snapped_date, panel_row, staleness_calendar_days).
    Staleness is the # of calendar days between snapped_date and date.
    If date is after panel end, returns the panel's last NYSE day and
    staleness from that to date.
    """
    if date in panel.index:
        return date, panel.loc[date], 0
    prior = panel.loc[:date]
    if len(prior) == 0:
        # date is before panel start -- return first row
        first = panel.index.min()
        return first, panel.iloc[0], int((first - date).days)
    snapped = prior.index.max()
    return snapped, panel.loc[snapped], int((date - snapped).days)


def forward_nyse_snap(date: pd.Timestamp, panel: pd.DataFrame
                      ) -> tuple[pd.Timestamp | None, pd.Series | None, int | None]:
    """Snap to nearest NYSE day >= date.  Returns (snapped, row, gap) or None."""
    after = panel.loc[date:]
    if len(after) == 0:
        return None, None, None
    snapped = after.index.min()
    return snapped, panel.loc[snapped], int((snapped - date).days)


# ---------------------------------------------------------------------------
# Cell lookup with fallback
# ---------------------------------------------------------------------------
def cell_lookup(shock: str, vbin: str, era: str, lookup: dict
                ) -> tuple[dict, str]:
    """Return (cell_record, source_tag).  Tags: 'primary', 'fallback_shock_era',
    'fallback_shock', 'no_data'."""
    key = f"{shock}__{vbin}__{era}"
    primary = lookup["cells_primary"].get(key)
    if primary and not primary["insufficient_data"]:
        return primary, "primary"
    se_key = f"{shock}__ALLVIX__{era}"
    se = lookup["cells_fallback_shock_era"].get(se_key)
    if se and not se["insufficient_data"]:
        return se, "fallback_shock_era"
    s_key = f"{shock}__ALLVIX__ALLERA"
    s = lookup["cells_fallback_shock"].get(s_key)
    if s and not s["insufficient_data"]:
        return s, "fallback_shock"
    return primary or {}, "no_data"


# ---------------------------------------------------------------------------
# Historical analogues (calendar-day forward returns)
# ---------------------------------------------------------------------------
def historical_analogues(target_vix: float, active_shocks: list[str], era: str,
                         panel: pd.DataFrame, btc_cal: pd.DataFrame,
                         top_k: int = 5,
                         exclude_date: pd.Timestamp | None = None) -> list[dict]:
    """Top-K most similar past NYSE days by:
      same era, same active-shock profile (set equality), nearest VIX.
    Realized forward returns reported on CALENDAR-DAY basis using BTC 24/7 series.
    """
    sub = panel[panel["era"] == era].copy()
    def shocks_on_row(r):
        return [s for s in RETAINED_SHOCKS if int(r.get(s, 0) or 0) == 1]
    sub["row_shocks"] = sub.apply(shocks_on_row, axis=1)
    target_set = set(active_shocks)
    if len(target_set) == 0:
        match = sub[sub["row_shocks"].apply(lambda x: len(x) == 0)]
        match_kind = "exact_none"
    else:
        match = sub[sub["row_shocks"].apply(lambda x: set(x) == target_set)]
        match_kind = "exact_shock_profile"
        if len(match) < top_k:
            relax = sub[sub["row_shocks"].apply(
                lambda x: len(set(x) & target_set) > 0)]
            if len(relax) > len(match):
                match = relax
                match_kind = "relaxed_subset_overlap"
    if exclude_date is not None:
        match = match[match.index != exclude_date]
    if len(match) == 0:
        return []
    match = match.copy()
    match["vix_distance"] = (match["vix"] - target_vix).abs()
    match = match.sort_values("vix_distance").head(top_k)

    btc_close = btc_cal["close"].astype(float)
    out = []
    for date_, row in match.iterrows():
        # Calendar-day BTC fwd returns
        btc_t = float(btc_close.loc[date_]) if date_ in btc_close.index else None
        realized = {}
        for h in HORIZONS_CAL:
            tph = date_ + pd.Timedelta(days=h)
            if btc_t is None or tph not in btc_close.index:
                # Try fallback: most recent close <= tph
                if btc_t is None:
                    realized[f"btc_fwd_{h}d_cal"] = None
                    continue
                if tph <= btc_close.index.max():
                    btc_tph = float(btc_close.loc[:tph].iloc[-1])
                else:
                    realized[f"btc_fwd_{h}d_cal"] = None
                    continue
            else:
                btc_tph = float(btc_close.loc[tph])
            realized[f"btc_fwd_{h}d_cal"] = btc_tph / btc_t - 1.0
        out.append({
            "date": str(date_.date()),
            "vix": float(row["vix"]),
            "vix_distance": float(row["vix_distance"]),
            "active_shocks": list(row["row_shocks"]),
            "era": row["era"],
            "match_kind": match_kind,
            "realized_calendar": realized,
        })
    return out


# ---------------------------------------------------------------------------
# Combine across multiple active shocks
# ---------------------------------------------------------------------------
def combine_beta_prediction(active_shocks: list[str], current_beta: float | None) -> dict:
    components = {s: DBETA_K[s] for s in active_shocks if s in DBETA_K}
    if not components:
        delta_beta = 0.0
        driver = "no_retained_shocks_active"
    elif "banking_shock" in components:
        delta_beta = DBETA_K["banking_shock"]
        driver = "banking_safe_haven_override_independent_hansen_postcovid"
    else:
        driver_key = max(components, key=components.get)
        delta_beta = components[driver_key]
        driver = f"max_dbeta_amongst_active__{driver_key}"
    predicted_beta = (
        float(current_beta) + float(delta_beta)
        if current_beta is not None and not (isinstance(current_beta, float) and np.isnan(current_beta))
        else None
    )
    return {
        "delta_beta_components_postcovid_indep_hansen": components,
        "combined_delta_beta": float(delta_beta),
        "combined_driver": driver,
        "current_beta_60": (None if current_beta is None or (isinstance(current_beta, float) and np.isnan(current_beta)) else float(current_beta)),
        "predicted_beta_at_horizon": predicted_beta,
    }


def combine_direction_prediction(per_shock_forecast: dict, horizon: int) -> dict:
    """sqrt(n)-weighted cell-mean across active-shock cells."""
    means_btc, means_spy, means_out, weights = [], [], [], []
    contributors = []
    for shock, sf in per_shock_forecast.items():
        h_rec = sf["by_horizon"].get(str(horizon))
        if not h_rec or h_rec["r_btc_calendar_fwd"]["n"] == 0:
            continue
        n = h_rec["r_btc_calendar_fwd"]["n"]
        w = float(np.sqrt(n))
        if h_rec["r_btc_calendar_fwd"]["mean"] is None:
            continue
        means_btc.append(h_rec["r_btc_calendar_fwd"]["mean"])
        means_spy.append(h_rec["r_spy_nyse_fwd"]["mean"])
        means_out.append(h_rec["outperf_calendar_fwd"]["mean"])
        weights.append(w)
        contributors.append({"shock": shock, "n": n, "weight": w,
                             "mean_btc_fwd_cal": h_rec["r_btc_calendar_fwd"]["mean"]})
    if not weights:
        return {
            "expected_r_btc": None,
            "expected_r_spy": None,
            "expected_outperformance": None,
            "weighting": "no_contributors",
            "contributors": [],
        }
    weights = np.array(weights)
    weights = weights / weights.sum()
    return {
        "expected_r_btc": float(np.sum(weights * np.array(means_btc))),
        "expected_r_spy": float(np.sum(weights * np.array(means_spy))),
        "expected_outperformance": float(np.sum(weights * np.array(means_out))),
        "weighting": "sqrt_n",
        "contributors": contributors,
    }


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
def confidence_label(min_n: int) -> str:
    if min_n >= 100:
        return "high"
    if min_n >= 50:
        return "medium"
    if min_n >= 20:
        return "low"
    return "very_low"


# ---------------------------------------------------------------------------
# BlackRock event comparison (calendar-day version)
# ---------------------------------------------------------------------------
def blackrock_match(active_shocks: list[str], vbin: str, era: str) -> dict | None:
    br = load_blackrock_calendar()
    events = br.get("events", [])
    if not events:
        return None
    target_set = set(active_shocks)
    best = None
    best_score = -1
    for ev in events:
        ev_shocks = set(ev.get("active_shocks_day_of_state", []))
        ev_era = ev.get("era")
        if ev_era != era:
            continue
        if not ev_shocks and not target_set:
            score = 1.0
        elif not ev_shocks or not target_set:
            score = 0.0
        else:
            score = len(ev_shocks & target_set) / len(ev_shocks | target_set)
        if score > best_score:
            best_score = score
            best = {
                "event": ev.get("event_name"),
                "event_date_calendar": ev.get("event_date_calendar"),
                "event_vix": ev.get("vix_at_state"),
                "event_active_shocks": list(ev_shocks),
                "event_era": ev_era,
                "realized_10d_pct": {
                    "btc": (ev["per_horizon_calendar"]["10"]["btc_fwd"]*100
                            if ev["per_horizon_calendar"]["10"]["btc_fwd"] is not None else None),
                    "spy": (ev["per_horizon_calendar"]["10"]["spy_fwd"]*100
                            if ev["per_horizon_calendar"]["10"]["spy_fwd"] is not None else None),
                    "outperf": (ev["per_horizon_calendar"]["10"]["outperf"]*100
                                if ev["per_horizon_calendar"]["10"]["outperf"] is not None else None),
                },
                "realized_60d_pct": {
                    "btc": (ev["per_horizon_calendar"]["60"]["btc_fwd"]*100
                            if ev["per_horizon_calendar"]["60"]["btc_fwd"] is not None else None),
                    "spy": (ev["per_horizon_calendar"]["60"]["spy_fwd"]*100
                            if ev["per_horizon_calendar"]["60"]["spy_fwd"] is not None else None),
                    "outperf": (ev["per_horizon_calendar"]["60"]["outperf"]*100
                                if ev["per_horizon_calendar"]["60"]["outperf"] is not None else None),
                },
                "shock_profile_jaccard": score,
            }
    if best is None or best_score < 0.34:
        return None
    return best


# ---------------------------------------------------------------------------
# Main forecast
# ---------------------------------------------------------------------------
def forecast(date: str = "today",
             vix: float | None = None,
             active_shocks: list[str] | None = None,
             era: str | None = None,
             horizons: list[int] | None = None) -> dict:
    panel = load_panel()
    btc_cal = load_btc_calendar()
    lookup = load_lookup()

    horizons = horizons or HORIZONS_CAL
    target_date = resolve_date(date)

    # Backward snap for state
    nyse_t, row_t, vix_stale = backward_nyse_snap(target_date, panel)
    # State variables
    vix_value = float(vix) if vix is not None else float(row_t["vix"])
    vix_staleness_days = 0 if vix is not None else vix_stale
    # Active shocks
    if active_shocks is None:
        active_shocks_out = [s for s in RETAINED_SHOCKS if int(row_t.get(s, 0) or 0) == 1]
        shock_staleness_days = vix_stale
    else:
        active_shocks_out = normalize_shocks(active_shocks)
        shock_staleness_days = 0
    # Era
    era_value = era or era_of(target_date)
    # VIX bin
    vbin = vix_bin_of(vix_value)
    regime = "calm" if vix_value < TAU else "stress"
    # current beta_60
    current_beta = float(row_t["beta_60"]) if not pd.isna(row_t["beta_60"]) else None

    # Per-shock cell lookup
    per_shock = {}
    cell_membership = []
    iter_shocks = active_shocks_out if active_shocks_out else ["none"]
    for shock in iter_shocks:
        cell, source = cell_lookup(shock, vbin, era_value, lookup)
        cell_membership.append({
            "shock": shock, "vix_bin": vbin, "era": era_value,
            "cell_source": source,
            "n_in_cell": (cell.get("n_days", 0) if source != "no_data" else 0),
        })
        per_shock[shock] = {
            "cell_source": source,
            "n_days": cell.get("n_days", 0),
            "insufficient_data": cell.get("insufficient_data", True) if source == "no_data" else False,
            "by_horizon": cell.get("by_horizon_calendar", {}),
        }

    # Also report the "any" aggregate (any-shock-active) and "none" aggregate
    extra_agg = {}
    for s in ["any", "none"]:
        cell, source = cell_lookup(s, vbin, era_value, lookup)
        extra_agg[s] = {
            "cell_source": source,
            "n_days": cell.get("n_days", 0),
            "by_horizon": cell.get("by_horizon_calendar", {}),
        }

    # Per-horizon combined forecast
    forecast_per_horizon = {}
    for h in horizons:
        comb = combine_direction_prediction(per_shock, h)
        # Quantiles: pool quantiles of contributing shocks
        q05_list, q95_list, q50_list, ns = [], [], [], []
        for shock, sf in per_shock.items():
            rec = sf["by_horizon"].get(str(h), {})
            r = rec.get("r_btc_calendar_fwd", {})
            if r.get("n", 0) > 0 and r.get("q05") is not None:
                q05_list.append(r["q05"])
                q95_list.append(r["q95"])
                q50_list.append(r["q50"])
                ns.append(r["n"])
        if q05_list:
            quantiles = {
                "q05": float(min(q05_list)),
                "q50": float(np.median(q50_list)),
                "q95": float(max(q95_list)),
            }
            n_analogues = int(max(ns))
        else:
            quantiles = {"q05": None, "q50": None, "q95": None}
            n_analogues = 0
        fallback_used = any(p["cell_source"] != "primary" for p in per_shock.values())
        forecast_per_horizon[str(h)] = {
            "horizon_calendar_days": h,
            "expected_r_btc": comb["expected_r_btc"],
            "expected_r_spy": comb["expected_r_spy"],
            "expected_outperformance": comb["expected_outperformance"],
            "prediction_quantiles_btc_fwd_cal": quantiles,
            "n_analogues_in_cell": n_analogues,
            "fallback_used": fallback_used,
            "weighting_scheme": comb["weighting"],
            "contributors": comb["contributors"],
        }

    # Combined beta prediction
    combined_beta = combine_beta_prediction(active_shocks_out, current_beta)

    min_n = min((p["n_days"] for p in per_shock.values()), default=0)
    conf = confidence_label(min_n)

    # Historical analogues (calendar-day realized)
    analogues = historical_analogues(
        target_vix=vix_value,
        active_shocks=active_shocks_out,
        era=era_value,
        panel=panel, btc_cal=btc_cal,
        top_k=5,
        exclude_date=nyse_t,
    )

    # BlackRock match
    br_match = blackrock_match(active_shocks_out, vbin, era_value)

    # Regime warning for pre-COVID
    regime_warning = None
    if era_value == "pre_covid":
        regime_warning = ("Pre-COVID era: BTC was structurally uncorrelated "
                          "with equities; predictions based on this era are "
                          "off-regime; use with caution.")

    # Notes
    notes = []
    if vix_staleness_days > 0:
        notes.append(f"VIX state pulled from nearest NYSE day {nyse_t.date()} "
                     f"(staleness = {vix_staleness_days} calendar days; query date "
                     f"{target_date.date()} is non-NYSE).")
    if shock_staleness_days > 0 and active_shocks is None:
        notes.append(f"Shock indicators pulled from same NYSE state day; "
                     f"staleness = {shock_staleness_days} calendar days.")
    if regime_warning:
        notes.append(regime_warning)
    if "banking_shock" in active_shocks_out:
        notes.append("Banking-shock active: dashboard headline uses the "
                     "independent-Hansen post-COVID Delta-beta = -0.33 "
                     "(safe-haven), per the SVB validation. The hierarchical "
                     "posterior +0.60 (which predicts amplification) is "
                     "reported in the paper's robustness; the SVB realized "
                     "outcome (+17pp BTC outperformance at 60d) confirmed the "
                     "safe-haven direction.")
    if conf == "very_low":
        notes.append("Cell sample size <20 even after fallback; treat "
                     "directional verdict as indicative only.")

    return {
        "inputs": {
            "date_requested": str(target_date.date()),
            "nyse_state_date_used": str(nyse_t.date()),
            "vix_staleness_days": vix_staleness_days,
            "shock_staleness_days": shock_staleness_days,
            "vix": vix_value,
            "active_shocks": active_shocks_out,
            "era": era_value,
            "current_beta_60": (None if current_beta is None else float(current_beta)),
        },
        "regime": regime,
        "tau_universal_postcovid": TAU,
        "vix_bin": vbin,
        "cell_membership": cell_membership,
        "aggregate_cells_any_none": extra_agg,
        "horizons_calendar_days": horizons,
        "forecast_per_horizon": forecast_per_horizon,
        "combined_beta_prediction": combined_beta,
        "historical_analogues_calendar": analogues,
        "confidence": conf,
        "blackrock_comparison": br_match,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# CLI rendering
# ---------------------------------------------------------------------------
def _print_human(out: dict) -> None:
    print("=" * 70)
    inp = out["inputs"]
    print(f"DASHBOARD FORECAST  (calendar-day basis)")
    print(f"  Date requested      : {inp['date_requested']}")
    print(f"  NYSE state used     : {inp['nyse_state_date_used']}"
          + (f"  (VIX staleness: {inp['vix_staleness_days']} cal days)" if inp['vix_staleness_days'] > 0 else ""))
    print(f"  VIX                 : {inp['vix']:.2f}    bin={out['vix_bin']}  regime={out['regime']}  (tau={out['tau_universal_postcovid']})")
    print(f"  Era                 : {inp['era']}")
    print(f"  Active shocks       : {inp['active_shocks'] or 'none'}")
    cb = out["combined_beta_prediction"]
    print(f"  Current beta_60     : {cb['current_beta_60']}")
    print(f"  Combined Delta-beta : {cb['combined_delta_beta']:+.3f}  (driver: {cb['combined_driver']})")
    if cb["predicted_beta_at_horizon"] is not None:
        print(f"  Predicted beta at h : {cb['predicted_beta_at_horizon']:+.3f}")
    print(f"  Confidence          : {out['confidence']}")

    print("\nCELL MEMBERSHIP:")
    for cm in out["cell_membership"]:
        print(f"  {cm['shock']:>20s}  bin={cm['vix_bin']:<14s}  era={cm['era']:<22s}"
              f"  source={cm['cell_source']:<22s}  n={cm['n_in_cell']}")

    print("\nFORECAST BY HORIZON (CALENDAR days; decimals):")
    for h_str, f in out["forecast_per_horizon"].items():
        ebtc = f["expected_r_btc"]; espy = f["expected_r_spy"]; eout = f["expected_outperformance"]
        q = f["prediction_quantiles_btc_fwd_cal"]
        def fmt(x): return f"{x:+.4f}" if x is not None else "  n/a"
        print(f"  h={h_str:>3}d (cal): E[r_BTC]={fmt(ebtc)}  E[r_SPY]={fmt(espy)}  "
              f"E[BTC-SPY]={fmt(eout)}  q05/q50/q95={q['q05']}/{q['q50']}/{q['q95']}  "
              f"n={f['n_analogues_in_cell']}  fallback={f['fallback_used']}")

    print("\nHISTORICAL ANALOGUES (top 5; calendar-day forward returns):")
    if not out["historical_analogues_calendar"]:
        print("  (none)")
    for a in out["historical_analogues_calendar"]:
        r = a["realized_calendar"]
        def fmt(x): return f"{x:+.3f}" if x is not None else " n/a"
        print(f"  {a['date']}  VIX={a['vix']:5.2f}  shocks={a['active_shocks']}  "
              f"5d/20d/60d = {fmt(r.get('btc_fwd_5d_cal'))}/{fmt(r.get('btc_fwd_20d_cal'))}/{fmt(r.get('btc_fwd_60d_cal'))}  "
              f"({a['match_kind']})")

    if out["blackrock_comparison"]:
        br = out["blackrock_comparison"]
        print("\nBLACKROCK ANALOGUE EVENT MATCH (calendar-day basis):")
        print(f"  Event       : {br['event']}  ({br['event_date_calendar']})")
        print(f"  Event VIX   : {br['event_vix']}")
        print(f"  Event shocks: {br['event_active_shocks']}")
        print(f"  Jaccard     : {br['shock_profile_jaccard']:.2f}")
        r10 = br["realized_10d_pct"]; r60 = br["realized_60d_pct"]
        def fp(x): return f"{x:+.2f}%" if x is not None else "  n/a"
        print(f"  Realized 10d: BTC={fp(r10['btc'])}  SPY={fp(r10['spy'])}  OUT={fp(r10['outperf'])}")
        print(f"  Realized 60d: BTC={fp(r60['btc'])}  SPY={fp(r60['spy'])}  OUT={fp(r60['outperf'])}")

    if out["notes"]:
        print("\nNOTES:")
        for n in out["notes"]:
            print(f"  - {n}")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="BTC dashboard forecaster (calendar-day basis)."
    )
    p.add_argument("--date", default="today",
                   help="YYYY-MM-DD or 'today' (any calendar date; weekends/holidays OK)")
    p.add_argument("--vix", type=float, default=None,
                   help="Override VIX (else pulled from nearest NYSE day <= date)")
    p.add_argument("--shocks", default=None,
                   help="Comma-separated active shocks (e.g. 'oil,rate'); "
                        "if absent, inferred from nearest NYSE day <= date.")
    p.add_argument("--era", default=None,
                   choices=[None, "pre_covid", "post_covid_pre_etf", "post_etf"],
                   help="Override era (else auto from date)")
    p.add_argument("--horizons", default=None,
                   help="Comma-separated calendar-day horizons (default 5,20,60)")
    p.add_argument("--json", action="store_true",
                   help="Print full JSON instead of human-readable summary")
    args = p.parse_args(argv)

    shocks_list = None
    if args.shocks:
        shocks_list = [s for s in args.shocks.split(",") if s.strip()]
    horizons = None
    if args.horizons:
        horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    out = forecast(date=args.date, vix=args.vix,
                   active_shocks=shocks_list, era=args.era, horizons=horizons)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        _print_human(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
