"""Generate the dashboard's current-state JSON for the live page to fetch.

Reads:
  - data/panel_with_shocks.parquet                    (VIX/BTC/SPY/oil/dollar/rates/STLFSI/GPR)
  - output/dashboard_lookup_table.json                (84-cell base + fallbacks)
  - output/shock_adds_value_test_walkforward.json     (walk-forward R² + verdict)
  - output/shock_adds_value_test.json                 (full-sample comparator)
  - output/shock_adds_value_test_ewma.json            (EWMA comparator)
  - output/shock_adds_value_test_era_cond.json        (era-cond comparator)

Writes:
  - data/dashboard_output.json (consumed by index.html via fetch())

Designed to be run daily (cron-style) after panel refresh.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data/panel_with_shocks.parquet"
LOOKUP = ROOT / "output/dashboard_lookup_table.json"
WF_JSON = ROOT / "output/shock_adds_value_test_walkforward.json"
FS_JSON = ROOT / "output/shock_adds_value_test.json"
EW_JSON = ROOT / "output/shock_adds_value_test_ewma.json"
EC_JSON = ROOT / "output/shock_adds_value_test_era_cond.json"
OUT = ROOT / "data/dashboard_output.json"

SHOCKS = ["oil_shock", "dollar_shock", "rate_shock", "banking_shock", "gprd_threat_shock"]
ERAS = [("pre_covid","2014-01-02","2020-02-29"),
        ("post_covid_pre_etf","2020-03-01","2024-01-09"),
        ("post_etf","2024-01-10","2099-12-31")]
VIX_BINS = [("calm",-np.inf,14.5),("low_stress",14.5,20.0),
            ("mid_stress",20.0,30.0),("extreme_stress",30.0,np.inf)]
HORIZONS = [5, 20, 60, 90]


def era_of(d):
    for n,lo,hi in ERAS:
        if pd.Timestamp(lo) <= d <= pd.Timestamp(hi): return n
    return "pre_covid"

def vix_bin_of(v):
    if pd.isna(v): return "calm"
    for n,lo,hi in VIX_BINS:
        if lo <= v < hi: return n
    return "calm"


def _cell_horizon(cell: dict, h: int) -> dict:
    """Pull (share_positive, mean_outperf, n) for cell at horizon h."""
    if not cell:
        return {"share_positive": None, "mean_outperf": None, "n": 0}
    hd = cell.get("by_horizon_calendar", {}).get(str(h), {})
    if not hd:
        return {"share_positive": None, "mean_outperf": None, "n": int(cell.get("n_days", 0))}
    op = hd.get("outperf_calendar_fwd", {})
    return {
        "share_positive": op.get("share_positive"),
        "mean_outperf": op.get("mean"),
        "n": int(hd.get("n", 0)),
    }


def main():
    panel = pd.read_parquet(PANEL).copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    # ---- Current state: last row of the panel ----
    last = panel.iloc[-1]
    last_date = panel.index[-1]
    current = {
        "current_date": str(last_date.date()),
        "panel_end_date": str(last_date.date()),
        "current_vix": float(last["vix"]),
        "vix_bin": vix_bin_of(last["vix"]),
        "current_era": era_of(last_date),
        "active_shocks": {s: bool(int(last[s])) if s in last.index and pd.notna(last[s]) else False
                          for s in SHOCKS},
    }
    active = [s for s, on in current["active_shocks"].items() if on]

    # ---- Cell lookup with multi-tier fallback ----
    lookup = json.loads(LOOKUP.read_text())
    primary = lookup["cells_primary"]
    fb_se = lookup["cells_fallback_shock_era"]
    fb_s  = lookup["cells_fallback_shock"]

    era = current["current_era"]
    vb = current["vix_bin"]

    # Pick the dominant single shock for the headline lookup.
    # If multiple fire, we lookup each and report all; if none, use "none".
    if not active:
        primary_shock = "none"
    elif len(active) == 1:
        primary_shock = active[0]
    else:
        # For multi-shock, lookup per-shock and pick the most-negative-leaning
        # cell (most-discriminating from "none" baseline) as the headline.
        # Surface all shock-specific lookups in `per_shock` for transparency.
        primary_shock = active[0]  # placeholder; chart shows all

    headline_key = f"{primary_shock}__{vb}__{era}"
    headline_cell = primary.get(headline_key)
    fallback_se = fb_se.get(f"{primary_shock}__ALLVIX__{era}")
    fallback_s  = fb_s.get(f"{primary_shock}__ALLVIX__ALLERA")
    used = headline_cell or fallback_se or fallback_s or {}
    used_tier = "primary" if headline_cell else ("fallback_shock_era" if fallback_se
                else ("fallback_shock" if fallback_s else "none"))

    current["primary_shock"] = primary_shock
    current["cell_key"] = headline_key
    current["cell_n"] = int(used.get("n_days", used.get("n", 0)))
    current["cell_tier_used"] = used_tier
    current["cell_regime_warning"] = used.get("regime_warning")

    current["horizons"] = {str(h): _cell_horizon(used, h) for h in HORIZONS}

    # Per-shock breakdown (for multi-shock days and for the cell-heatmap section)
    per_shock = {}
    for s in active:
        k = f"{s}__{vb}__{era}"
        c = primary.get(k) or fb_se.get(f"{s}__ALLVIX__{era}") or fb_s.get(f"{s}__ALLVIX__ALLERA") or {}
        per_shock[s] = {
            "key": k,
            "n": int(c.get("n_days", c.get("n", 0))),
            "h60": _cell_horizon(c, 60),
        }
    current["per_shock_lookup"] = per_shock

    # ---- Methodology progression ----
    def _load(p):
        try: return json.loads(p.read_text())
        except Exception: return None
    fs, ew, ec, wf = _load(FS_JSON), _load(EW_JSON), _load(EC_JSON), _load(WF_JSON)

    def _dr(d, k="delta_r2_oos_M2_to_M3_full_interaction"):
        return float(d["full_panel"][k]) if d else None
    def _r2(d, mod): return float(d["full_panel"][mod]["r2_oos"]) if d else None
    def _pv(d):
        return float(d["permutation_full_panel"]["permutation_p_value_M3_one_sided_greater"]) \
            if d else None
    def _bank(d):
        if not d: return None
        coefs = d.get("per_shock_loadings_full_panel", {}).get("M4_ridge_coefs", {})
        return float(coefs.get("shock_banking_shock", 0)) if coefs else None
    def _verd(d): return d.get("verdict", {}).get("class") if d else None
    def _n(d): return int(d["full_panel"]["n_obs"]) if d else None

    current["methodology"] = {
        "full_sample": {"label":"Full-sample (paper)", "n_obs": _n(fs),
            "dr2_m3_pp": (_dr(fs) or 0)*100, "p_value": _pv(fs),
            "banking_m4_pp": (_bank(fs) or 0)*100, "verdict": _verd(fs)},
        "ewma_126": {"label":"EWMA-126 rolling", "n_obs": _n(ew),
            "dr2_m3_pp": (_dr(ew) or 0)*100, "p_value": _pv(ew),
            "banking_m4_pp": (_bank(ew) or 0)*100, "verdict": _verd(ew)},
        "era_conditional": {"label":"Era-conditional (look-ahead)", "n_obs": _n(ec),
            "dr2_m3_pp": (_dr(ec) or 0)*100, "p_value": _pv(ec),
            "banking_m4_pp": (_bank(ec) or 0)*100, "verdict": _verd(ec)},
        "walkforward": {"label":"Walk-forward + q99 fallback", "n_obs": _n(wf),
            "dr2_m3_pp": (_dr(wf) or 0)*100, "p_value": _pv(wf),
            "banking_m4_pp": (_bank(wf) or 0)*100, "verdict": _verd(wf),
            "m2_r2_pp": (_r2(wf, "M2_era_x_vix") or 0)*100,
            "m3_r2_pp": (_r2(wf, "M3_era_x_vix_x_shocktype") or 0)*100},
    }

    # ---- All cells table (for the heatmap) — primary cells only ----
    table = []
    for key, c in primary.items():
        h60 = _cell_horizon(c, 60)
        table.append({
            "key": key,
            "shock": c.get("shock"),
            "vix_bin": c.get("vix_bin"),
            "era": c.get("era"),
            "n": int(c.get("n_days", 0)),
            "share_positive_60d": h60["share_positive"],
            "mean_outperf_60d": h60["mean_outperf"],
            "regime_warning": c.get("regime_warning"),
        })
    current["cells"] = table

    # ---- VIX history (last ~2 years for time-series chart) ----
    hist = panel[["vix"]].dropna().reset_index()
    hist.columns = ["date", "vix"]
    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    current["vix_history"] = hist.tail(750).to_dict(orient="records")

    current["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    current["panel_n_rows"] = int(len(panel))
    current["data_sources"] = {
        "VIX": "FRED VIXCLS",
        "BTC": "CoinMetrics community API",
        "SPY": "yfinance",
        "Oil": "FRED DCOILWTICO",
        "Dollar": "FRED DTWEXBGS",
        "Rates": "FRED DGS10",
        "Banking": "FRED STLFSI4",
        "GPR-threat": "Caldara-Iacoviello (matteoiacoviello.com)",
    }

    OUT.write_text(json.dumps(current, indent=2, default=str))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
