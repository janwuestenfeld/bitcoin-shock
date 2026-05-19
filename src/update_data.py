"""Generate the dashboard's current-state JSON.

Provides:
  - Top-line state (date, VIX, era, active shocks)
  - Per-shock state cards: raw value, walk-forward cutoff, distance from cutoff,
    active boolean, recent history (last ~2 years) for sparkline charts
  - Shock-vs-VIX co-movement (correlation + scatter data)
  - Today's cell lookup + horizon distribution
  - Closest BlackRock-style historical analogue
  - Methodology summary (4-cutoff comparison)
  - 84-cell historical reference table

Reads:
  - data/panel_with_shocks.parquet
  - data/era_conditional_walkforward_shocks_panel.parquet
  - output/dashboard_lookup_table.json
  - output/shock_adds_value_test_*.json
  - output/blackrock_6event_calendar_validation.json

Writes:
  - data/dashboard_output.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data/panel_with_shocks.parquet"
WF_PANEL = ROOT / "data/era_conditional_walkforward_shocks_panel.parquet"
LOOKUP = ROOT / "output/dashboard_lookup_table.json"
BLACKROCK = ROOT / "output/blackrock_6event_calendar_validation.json"
WF_JSON = ROOT / "output/shock_adds_value_test_walkforward.json"
FS_JSON = ROOT / "output/shock_adds_value_test.json"
EW_JSON = ROOT / "output/shock_adds_value_test_ewma.json"
EC_JSON = ROOT / "output/shock_adds_value_test_era_cond.json"
OUT = ROOT / "data/dashboard_output.json"

SHOCKS = ["oil_shock", "dollar_shock", "rate_shock", "banking_shock", "gprd_threat_shock"]
SHOCK_DISPLAY = {
    "oil_shock": {"label":"Oil", "raw_desc":"|Δlog WTI|",
        "source":"FRED DCOILWTICO", "story":"Big oil moves stress macro vol"},
    "dollar_shock": {"label":"Dollar", "raw_desc":"|Δ USD-broad|",
        "source":"FRED DTWEXBGS", "story":"Dollar shocks transmit through global liquidity"},
    "rate_shock": {"label":"Rates", "raw_desc":"|Δ 10Y yield|",
        "source":"FRED DGS10", "story":"Rate moves drive duration repricing"},
    "banking_shock": {"label":"Banking", "raw_desc":"STLFSI4 level",
        "source":"FRED STLFSI4", "story":"Banking stress can trigger safe-haven flows"},
    "gprd_threat_shock": {"label":"GPR-Threat", "raw_desc":"GPR-threat index",
        "source":"Caldara-Iacoviello", "story":"Geopolitical risk reshapes risk-asset correlations"},
}
ERAS = [("pre_covid","2014-01-02","2020-02-29"),
        ("post_covid_pre_etf","2020-03-01","2024-01-09"),
        ("post_etf","2024-01-10","2099-12-31")]
VIX_BINS = [("calm",-np.inf,14.5),("low_stress",14.5,20.0),
            ("mid_stress",20.0,30.0),("extreme_stress",30.0,np.inf)]
HORIZONS = [5, 20, 60, 90]
HISTORY_DAYS = 750  # ~2 years for sparkline charts


def era_of(d):
    for n,lo,hi in ERAS:
        if pd.Timestamp(lo) <= d <= pd.Timestamp(hi): return n
    return "pre_covid"

def vix_bin_of(v):
    if pd.isna(v): return "calm"
    for n,lo,hi in VIX_BINS:
        if lo <= v < hi: return n
    return "calm"


def _cell_horizon(cell, h):
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

    wf = pd.read_parquet(WF_PANEL).copy()
    wf.index = pd.to_datetime(wf.index)
    wf = wf.sort_index()

    # Last common row
    common_idx = panel.index.intersection(wf.index)
    last_date = common_idx.max()
    last_panel = panel.loc[last_date]
    last_wf = wf.loc[last_date]

    out = {
        "current_date": str(last_date.date()),
        "current_vix": float(last_panel["vix"]),
        "vix_bin": vix_bin_of(last_panel["vix"]),
        "current_era": era_of(last_date),
    }

    # ===== Per-shock current state =====
    per_shock = {}
    for s in SHOCKS:
        raw = float(last_wf.get(f"{s}_wf_raw", np.nan)) if pd.notna(last_wf.get(f"{s}_wf_raw")) else None
        thr = float(last_wf.get(f"{s}_wf_thresh", np.nan)) if pd.notna(last_wf.get(f"{s}_wf_thresh")) else None
        active = bool(last_wf.get(f"{s}_wf", 0) == 1) if pd.notna(last_wf.get(f"{s}_wf")) else False
        # Distance from cutoff: (raw - thr) / thr in %, only if thr defined and non-zero
        if raw is not None and thr is not None and thr != 0:
            pct_of_cut = float(raw / thr * 100)
            dist_pp = float((raw / thr - 1) * 100)
        else:
            pct_of_cut = None; dist_pp = None
        per_shock[s] = {
            "display": SHOCK_DISPLAY[s],
            "raw_current": raw, "cutoff_current": thr,
            "pct_of_cutoff": pct_of_cut, "distance_from_cutoff_pp": dist_pp,
            "active": active,
        }
    out["active_shocks"] = {s: per_shock[s]["active"] for s in SHOCKS}
    out["per_shock_state"] = per_shock

    # ===== Per-shock history (raw + cutoff, last 2 years) =====
    hist_window = wf.tail(HISTORY_DAYS).copy()
    hist_window["date"] = hist_window.index.strftime("%Y-%m-%d")
    # Also pull VIX over the same window
    vix_window = panel.loc[panel.index.isin(hist_window.index), "vix"]
    hist_records = {}
    for s in SHOCKS:
        cols = [f"{s}_wf_raw", f"{s}_wf_thresh", f"{s}_wf"]
        sub = hist_window[["date"] + cols].rename(columns={
            f"{s}_wf_raw":"raw", f"{s}_wf_thresh":"cutoff", f"{s}_wf":"active"
        }).copy()
        sub["active"] = sub["active"].fillna(0).astype(int)
        # Round to keep JSON small
        sub["raw"] = sub["raw"].round(6)
        sub["cutoff"] = sub["cutoff"].round(6)
        hist_records[s] = sub.to_dict(orient="records")
    out["shock_history"] = hist_records

    # VIX history (same window)
    vix_hist = pd.DataFrame({
        "date": hist_window["date"].values,
        "vix": panel.loc[hist_window.index, "vix"].round(3).values,
    })
    out["vix_history"] = vix_hist.to_dict(orient="records")

    # ===== Shock-vs-VIX co-movement (correlation + scatter, last 2 years) =====
    comovement = {}
    for s in SHOCKS:
        raw_s = hist_window[f"{s}_wf_raw"]
        vix_s = panel.loc[hist_window.index, "vix"]
        common = pd.concat([raw_s, vix_s], axis=1).dropna()
        common.columns = ["raw", "vix"]
        if len(common) >= 30:
            corr = float(common.corr().iloc[0,1])
        else:
            corr = None
        comovement[s] = {
            "correlation_with_vix": corr,
            "n_observations": int(len(common)),
        }
    out["comovement"] = comovement

    # ===== Cell lookup =====
    lookup = json.loads(LOOKUP.read_text())
    primary = lookup["cells_primary"]
    fb_se = lookup["cells_fallback_shock_era"]
    fb_s = lookup["cells_fallback_shock"]
    active = [s for s, on in out["active_shocks"].items() if on]
    if not active:
        primary_shock = "none"
    elif len(active) == 1:
        primary_shock = active[0]
    else:
        primary_shock = active[0]
    era = out["current_era"]
    vb = out["vix_bin"]
    headline_key = f"{primary_shock}__{vb}__{era}"
    headline_cell = primary.get(headline_key)
    used = headline_cell or fb_se.get(f"{primary_shock}__ALLVIX__{era}") \
        or fb_s.get(f"{primary_shock}__ALLVIX__ALLERA") or {}
    used_tier = "primary" if headline_cell else (
        "fallback_shock_era" if fb_se.get(f"{primary_shock}__ALLVIX__{era}")
        else ("fallback_shock" if fb_s.get(f"{primary_shock}__ALLVIX__ALLERA") else "none"))

    out["primary_shock"] = primary_shock
    out["cell_key"] = headline_key
    out["cell_n"] = int(used.get("n_days", used.get("n", 0)))
    out["cell_tier_used"] = used_tier
    out["cell_regime_warning"] = used.get("regime_warning")
    out["horizons"] = {str(h): _cell_horizon(used, h) for h in HORIZONS}

    # Per-shock cell lookups (for multi-shock days)
    per_shock_cells = {}
    for s in active:
        k = f"{s}__{vb}__{era}"
        c = primary.get(k) or fb_se.get(f"{s}__ALLVIX__{era}") or fb_s.get(f"{s}__ALLVIX__ALLERA") or {}
        per_shock_cells[s] = {
            "key": k, "n": int(c.get("n_days", c.get("n", 0))),
            "h60": _cell_horizon(c, 60),
        }
    out["per_shock_lookup"] = per_shock_cells

    # ===== Closest BlackRock-style historical analogue =====
    try:
        br = json.loads(BLACKROCK.read_text())
        events = br.get("events", []) if isinstance(br, dict) else []
        # Find events that match today's regime (same era + vix_bin + at least one shared shock)
        matches = []
        for e in events:
            if e.get("era") == era and e.get("vix_bin") == vb:
                ev_shocks = set(e.get("active_shocks", []))
                today_shocks = set(active)
                overlap = ev_shocks & today_shocks
                if overlap or (not ev_shocks and not today_shocks):
                    matches.append({**e, "shock_overlap": list(overlap)})
        out["closest_analogues"] = matches[:3]
    except Exception:
        out["closest_analogues"] = []

    # ===== Methodology summary =====
    def _load(p):
        try: return json.loads(p.read_text())
        except Exception: return None
    fs, ew, ec, wfj = _load(FS_JSON), _load(EW_JSON), _load(EC_JSON), _load(WF_JSON)

    def _dr(d): return float(d["full_panel"]["delta_r2_oos_M2_to_M3_full_interaction"]) if d else None
    def _r2(d, mod): return float(d["full_panel"][mod]["r2_oos"]) if d else None
    def _pv(d):
        return float(d["permutation_full_panel"]["permutation_p_value_M3_one_sided_greater"]) if d else None
    def _bank(d):
        if not d: return None
        coefs = d.get("per_shock_loadings_full_panel", {}).get("M4_ridge_coefs", {})
        return float(coefs.get("shock_banking_shock", 0)) if coefs else None
    def _verd(d): return d.get("verdict", {}).get("class") if d else None
    def _n(d): return int(d["full_panel"]["n_obs"]) if d else None

    out["methodology"] = {
        "full_sample": {"label":"Full-sample (paper)", "n_obs": _n(fs),
            "dr2_m3_pp": (_dr(fs) or 0)*100, "p_value": _pv(fs),
            "banking_m4_pp": (_bank(fs) or 0)*100, "verdict": _verd(fs)},
        "ewma_126": {"label":"EWMA-126 rolling", "n_obs": _n(ew),
            "dr2_m3_pp": (_dr(ew) or 0)*100, "p_value": _pv(ew),
            "banking_m4_pp": (_bank(ew) or 0)*100, "verdict": _verd(ew)},
        "era_conditional": {"label":"Era-conditional (look-ahead)", "n_obs": _n(ec),
            "dr2_m3_pp": (_dr(ec) or 0)*100, "p_value": _pv(ec),
            "banking_m4_pp": (_bank(ec) or 0)*100, "verdict": _verd(ec)},
        "walkforward": {"label":"Walk-forward + q99 fallback", "n_obs": _n(wfj),
            "dr2_m3_pp": (_dr(wfj) or 0)*100, "p_value": _pv(wfj),
            "banking_m4_pp": (_bank(wfj) or 0)*100, "verdict": _verd(wfj),
            "m2_r2_pp": (_r2(wfj, "M2_era_x_vix") or 0)*100,
            "m3_r2_pp": (_r2(wfj, "M3_era_x_vix_x_shocktype") or 0)*100},
    }

    # ===== 84-cell historical reference =====
    cells = []
    for key, c in primary.items():
        h60 = _cell_horizon(c, 60)
        cells.append({
            "key": key, "shock": c.get("shock"), "vix_bin": c.get("vix_bin"), "era": c.get("era"),
            "n": int(c.get("n_days", 0)),
            "share_positive_60d": h60["share_positive"],
            "mean_outperf_60d": h60["mean_outperf"],
            "regime_warning": c.get("regime_warning"),
        })
    out["cells"] = cells

    out["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["panel_n_rows"] = int(len(panel))

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
