"""Generate the dashboard's current-state JSON.

Practitioner-relevant payload:
  - Top-line state + day-over-day diff ("what changed today")
  - Per-shock state cards with proximity (z-score), last-fired, sparkline series
  - β_60(BTC, SPY) current + trajectory (backward-rolling; no look-ahead)
  - Today's cell + horizon strip + confidence chip
  - "What would flip the cell" — auto-computed boundary deltas
  - BlackRock 6-event validation strip (cell at event + realized 60d)
  - 12-cell era × VIX-bin aggregate (current cell highlighted)
  - 84-cell full reference (collapsed in UI)
  - Methodology snapshot (collapsed in UI)
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
BTC_CAL = ROOT / "data/btc_calendar_daily.parquet"
LOOKUP = ROOT / "output/dashboard_lookup_table.json"
BLACKROCK = ROOT / "output/blackrock_6event_calendar_validation.json"
WF_JSON = ROOT / "output/shock_adds_value_test_walkforward.json"
FS_JSON = ROOT / "output/shock_adds_value_test.json"
EW_JSON = ROOT / "output/shock_adds_value_test_ewma.json"
EC_JSON = ROOT / "output/shock_adds_value_test_era_cond.json"
OUT = ROOT / "data/dashboard_output.json"

SHOCKS = ["oil_shock", "dollar_shock", "rate_shock", "banking_shock", "gprd_threat_shock"]
SHOCK_DISPLAY = {
    "oil_shock": {"label":"Oil", "raw_desc":"|Δlog WTI|", "source":"FRED DCOILWTICO",
        "kind":"unsigned"},
    "dollar_shock": {"label":"Dollar", "raw_desc":"|Δ USD-broad|", "source":"FRED DTWEXBGS",
        "kind":"unsigned"},
    "rate_shock": {"label":"Rates", "raw_desc":"|Δ 10Y yield|", "source":"FRED DGS10",
        "kind":"unsigned"},
    "banking_shock": {"label":"Banking", "raw_desc":"STLFSI4 level", "source":"FRED STLFSI4",
        "kind":"signed"},
    "gprd_threat_shock": {"label":"GPR-Threat", "raw_desc":"GPR-threat index",
        "source":"Caldara-Iacoviello", "kind":"unsigned"},
}
ERAS = [("pre_covid","2014-01-02","2020-02-29"),
        ("post_covid_pre_etf","2020-03-01","2024-01-09"),
        ("post_etf","2024-01-10","2099-12-31")]
VIX_BINS = [("calm",-np.inf,14.5),("low_stress",14.5,20.0),
            ("mid_stress",20.0,30.0),("extreme_stress",30.0,np.inf)]
VIX_BOUNDARIES = [14.5, 20.0, 30.0]
HORIZONS = [5, 20, 60, 90]
HISTORY_DAYS = 750
BETA_WINDOW = 60


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


def compute_beta_60(panel: pd.DataFrame) -> pd.Series:
    """60-day backward-rolling OLS β of BTC log-returns on SPY log-returns.
    Backward-rolling by construction — no look-ahead."""
    if "btc" in panel.columns:
        btc = panel["btc"].astype(float)
    elif "btc_close" in panel.columns:
        btc = panel["btc_close"].astype(float)
    else:
        # Try CoinMetrics calendar parquet as fallback (will be sparser on NYSE days)
        btc_cal = pd.read_parquet(BTC_CAL)
        btc_cal.index = pd.to_datetime(btc_cal.index)
        btc = btc_cal["close"].reindex(panel.index, method="ffill")
    spy = panel["spy"].astype(float)
    rb = np.log(btc / btc.shift(1))
    rs = np.log(spy / spy.shift(1))
    cov = rb.rolling(BETA_WINDOW).cov(rs)
    var = rs.rolling(BETA_WINDOW).var()
    beta = cov / var
    return beta


def confidence_for_n(n: int) -> str:
    if n < 5: return "no_call"
    if n < 20: return "low"
    if n < 50: return "medium"
    return "high"


def main():
    panel = pd.read_parquet(PANEL).copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    wf = pd.read_parquet(WF_PANEL).copy()
    wf.index = pd.to_datetime(wf.index)
    wf = wf.sort_index()

    last_date = panel.index.intersection(wf.index).max()
    last_panel = panel.loc[last_date]
    last_wf = wf.loc[last_date]
    # Previous day (for day-over-day diff)
    prior_dates = panel.index[panel.index < last_date]
    prior_date = prior_dates[-1] if len(prior_dates) else None
    prior_panel = panel.loc[prior_date] if prior_date is not None else None
    prior_wf = wf.loc[prior_date] if prior_date is not None and prior_date in wf.index else None

    out = {
        "current_date": str(last_date.date()),
        "current_vix": float(last_panel["vix"]),
        "vix_bin": vix_bin_of(last_panel["vix"]),
        "current_era": era_of(last_date),
    }

    # ===== β_60 series =====
    beta = compute_beta_60(panel)
    beta_today = float(beta.iloc[-1]) if pd.notna(beta.iloc[-1]) else None
    # Change over last 5 trading days
    beta_5d_ago = float(beta.iloc[-6]) if len(beta) >= 6 and pd.notna(beta.iloc[-6]) else None
    beta_delta_5d = (beta_today - beta_5d_ago) if (beta_today is not None and beta_5d_ago is not None) else None
    out["beta_60"] = {
        "today": beta_today, "five_day_ago": beta_5d_ago, "delta_5d": beta_delta_5d,
        "window_days": BETA_WINDOW,
        "note": "60-day backward-rolling OLS β(BTC, SPY) — no look-ahead by construction"
    }
    beta_hist = beta.dropna().tail(HISTORY_DAYS)
    out["beta_history"] = [{"date": d.strftime("%Y-%m-%d"), "beta": round(float(v), 4)}
                           for d, v in beta_hist.items()]

    # ===== Per-shock state + z-score (for signed measures) + last-fired =====
    per_shock = {}
    for s in SHOCKS:
        disp = SHOCK_DISPLAY[s]
        raw = float(last_wf.get(f"{s}_wf_raw", np.nan)) if pd.notna(last_wf.get(f"{s}_wf_raw")) else None
        thr = float(last_wf.get(f"{s}_wf_thresh", np.nan)) if pd.notna(last_wf.get(f"{s}_wf_thresh")) else None
        active = bool(last_wf.get(f"{s}_wf", 0) == 1) if pd.notna(last_wf.get(f"{s}_wf")) else False
        prior_active = bool(prior_wf.get(f"{s}_wf", 0) == 1) if (prior_wf is not None and pd.notna(prior_wf.get(f"{s}_wf"))) else None

        # Compute z-score for signed measures (banking), pct-of-cutoff for unsigned
        kind = disp["kind"]
        proximity = None
        proximity_label = ""
        if raw is not None and thr is not None:
            if kind == "signed":
                # z-score using trailing 252d std of the raw series
                raw_hist = wf[f"{s}_wf_raw"].dropna().tail(252)
                std = float(raw_hist.std()) if len(raw_hist) >= 30 else None
                if std and std > 0:
                    proximity = float((raw - thr) / std)  # σ above/below
                    proximity_label = f"{proximity:+.2f}σ vs threshold"
            else:
                if thr != 0:
                    proximity = float(raw / thr)  # ratio: 1.0 == at threshold
                    proximity_label = f"{proximity*100:.0f}% of cutoff"

        # Last-fired
        fires = wf[f"{s}_wf"].fillna(0)
        last_fire_idx = fires[fires == 1].index
        if len(last_fire_idx):
            last_fire = last_fire_idx[-1]
            days_since_last = int((last_date - last_fire).days)
            last_fire_str = last_fire.strftime("%Y-%m-%d")
        else:
            days_since_last = None; last_fire_str = None

        # 20-day binary trend (kept for label) AND 60-day prior comparison (load-bearing)
        trend_20d = None
        if raw is not None and thr is not None:
            rh = wf[f"{s}_wf_raw"].dropna().tail(40)
            if len(rh) >= 40:
                trend_20d = "rising" if rh.tail(20).mean() > rh.head(20).mean() else "falling"

        # 60-day prior: raw value 60 trading days ago + change
        raw_60d_ago = None; raw_change_60d_pct = None; raw_change_60d_abs = None
        raw_series = wf[f"{s}_wf_raw"].dropna()
        if len(raw_series) >= 61 and raw is not None:
            r60 = float(raw_series.iloc[-61])  # value 60 obs before today
            raw_60d_ago = r60
            raw_change_60d_abs = raw - r60
            if kind == "unsigned" and r60 != 0:
                raw_change_60d_pct = (raw / r60 - 1.0) * 100
            elif kind == "signed":
                # For signed measures (banking), report the absolute delta in std-units
                std60 = float(raw_series.tail(252).std()) if len(raw_series.tail(252)) >= 30 else None
                if std60 and std60 > 0:
                    raw_change_60d_pct = raw_change_60d_abs / std60  # σ change

        # Cutoff change over 60 days (how much has the cutoff itself shifted?)
        cutoff_60d_ago = None; cutoff_change_60d = None
        thr_series = wf[f"{s}_wf_thresh"].dropna()
        if len(thr_series) >= 61 and thr is not None:
            c60 = float(thr_series.iloc[-61])
            cutoff_60d_ago = c60
            if c60 != 0:
                cutoff_change_60d = (thr / c60 - 1.0) * 100 if kind == "unsigned" else (thr - c60)

        per_shock[s] = {
            "display": disp,
            "raw_current": raw, "cutoff_current": thr,
            "raw_60d_ago": raw_60d_ago,
            "raw_change_60d_pct": raw_change_60d_pct,
            "raw_change_60d_abs": raw_change_60d_abs,
            "cutoff_60d_ago": cutoff_60d_ago,
            "cutoff_change_60d_pct": cutoff_change_60d,
            "active": active, "prior_active": prior_active,
            "proximity": proximity, "proximity_label": proximity_label,
            "days_since_last_fire": days_since_last, "last_fire_date": last_fire_str,
            "trend_20d": trend_20d,
        }
    out["active_shocks"] = {s: per_shock[s]["active"] for s in SHOCKS}
    out["per_shock_state"] = per_shock

    # ===== Per-shock 2-year history =====
    hist_window = wf.tail(HISTORY_DAYS).copy()
    hist_window["date"] = hist_window.index.strftime("%Y-%m-%d")
    hist_records = {}
    for s in SHOCKS:
        cols = [f"{s}_wf_raw", f"{s}_wf_thresh", f"{s}_wf"]
        sub = hist_window[["date"] + cols].rename(columns={
            f"{s}_wf_raw":"raw", f"{s}_wf_thresh":"cutoff", f"{s}_wf":"active"
        }).copy()
        sub["active"] = sub["active"].fillna(0).astype(int)
        sub["raw"] = sub["raw"].round(6)
        sub["cutoff"] = sub["cutoff"].round(6)
        hist_records[s] = sub.to_dict(orient="records")
    out["shock_history"] = hist_records

    # ===== VIX history (90 days for the trajectory chart) =====
    vix_window = panel.loc[panel.index >= (last_date - pd.Timedelta(days=110)), "vix"].dropna()
    out["vix_history_90d"] = [
        {"date": d.strftime("%Y-%m-%d"), "vix": round(float(v), 2)}
        for d, v in vix_window.items()
    ]
    # Longer history for the 2-year section
    vix_long = panel.loc[panel.index >= (last_date - pd.Timedelta(days=730)), "vix"].dropna()
    out["vix_history_long"] = [
        {"date": d.strftime("%Y-%m-%d"), "vix": round(float(v), 2)}
        for d, v in vix_long.items()
    ]

    # ===== Day-over-day "what changed today" =====
    changed = []
    if prior_panel is not None:
        vix_diff = out["current_vix"] - float(prior_panel["vix"])
        prior_bin = vix_bin_of(float(prior_panel["vix"]))
        if abs(vix_diff) >= 0.05:
            changed.append({
                "kind": "vix",
                "text": f"VIX {out['current_vix']:.2f} ({vix_diff:+.2f} d/d){', regime shift to '+out['vix_bin'] if prior_bin != out['vix_bin'] else ', '+out['vix_bin']+' regime unchanged'}"
            })
    if prior_wf is not None:
        for s in SHOCKS:
            now = per_shock[s]["active"]
            prior = per_shock[s]["prior_active"]
            if prior is not None and now != prior:
                action = "fired today (was inactive yesterday)" if now else "turned off today (was active yesterday)"
                changed.append({"kind": "shock", "shock": s, "text": f"{per_shock[s]['display']['label']} {action}"})
    if beta_delta_5d is not None and abs(beta_delta_5d) >= 0.05:
        changed.append({
            "kind": "beta",
            "text": f"β₆₀ now {beta_today:.2f} ({beta_delta_5d:+.2f} vs 5d ago)"
        })
    out["what_changed"] = changed

    # ===== Cell lookup =====
    lookup = json.loads(LOOKUP.read_text())
    primary = lookup["cells_primary"]
    fb_se = lookup["cells_fallback_shock_era"]
    fb_s = lookup["cells_fallback_shock"]
    active = [s for s, on in out["active_shocks"].items() if on]
    primary_shock = "none" if not active else (active[0] if len(active) == 1 else active[0])
    era = out["current_era"]
    vb = out["vix_bin"]
    headline_key = f"{primary_shock}__{vb}__{era}"
    headline_cell = primary.get(headline_key)
    used = headline_cell or fb_se.get(f"{primary_shock}__ALLVIX__{era}") \
        or fb_s.get(f"{primary_shock}__ALLVIX__ALLERA") or {}
    used_tier = ("primary" if headline_cell else
        ("fallback_shock_era" if fb_se.get(f"{primary_shock}__ALLVIX__{era}") else
         ("fallback_shock" if fb_s.get(f"{primary_shock}__ALLVIX__ALLERA") else "none")))

    out["primary_shock"] = primary_shock
    out["cell_key"] = headline_key
    out["cell_n"] = int(used.get("n_days", used.get("n", 0)))
    out["cell_tier_used"] = used_tier
    out["cell_regime_warning"] = used.get("regime_warning")
    out["horizons"] = {str(h): _cell_horizon(used, h) for h in HORIZONS}
    out["confidence"] = confidence_for_n(out["cell_n"])

    # ===== What would flip the cell — auto-computed boundary deltas =====
    def _lookup_with_fallback(shock, vix_bin, era_name):
        k1 = f"{shock}__{vix_bin}__{era_name}"
        c = primary.get(k1)
        if c and _cell_horizon(c, 60)["n"] >= 5:
            return c, "primary"
        c2 = fb_se.get(f"{shock}__ALLVIX__{era_name}")
        if c2:
            return c2, "fallback_shock_era"
        c3 = fb_s.get(f"{shock}__ALLVIX__ALLERA")
        if c3:
            return c3, "fallback_shock"
        return c or {}, "none"

    flips = []
    # VIX boundary deltas
    for b in VIX_BOUNDARIES:
        delta = b - out["current_vix"]
        new_bin = vix_bin_of(b + 0.01)
        if new_bin != vb and abs(delta) < 15:
            new_cell, new_tier = _lookup_with_fallback(primary_shock, new_bin, era)
            new_sp = _cell_horizon(new_cell, 60).get("share_positive")
            direction = "rises" if delta > 0 else "falls"
            flips.append({
                "trigger": f"VIX {direction} by {abs(delta):.1f} to {b:.1f}",
                "effect": f"regime shifts to {new_bin}",
                "new_share_positive": new_sp,
                "new_n": new_cell.get("n_days", 0),
                "tier": new_tier,
            })
    # Shock activation flips
    for s in SHOCKS:
        st = per_shock[s]
        if st["active"]:
            new_active = [x for x in active if x != s]
            new_primary_shock = "none" if not new_active else new_active[0]
            new_cell, new_tier = _lookup_with_fallback(new_primary_shock, vb, era)
            flips.append({
                "trigger": f"{st['display']['label']} turns off",
                "effect": f"cell shifts to {new_primary_shock}__{vb}__{era}",
                "new_share_positive": _cell_horizon(new_cell, 60).get("share_positive"),
                "new_n": new_cell.get("n_days", 0),
                "tier": new_tier,
            })
        else:
            if st["proximity"] is not None and st["display"]["kind"] == "unsigned":
                gap_pct = (1.0 - st["proximity"]) * 100
                if gap_pct < 40:
                    new_active_list = sorted(active + [s])
                    new_primary_shock = new_active_list[0]
                    new_cell, new_tier = _lookup_with_fallback(new_primary_shock, vb, era)
                    flips.append({
                        "trigger": f"{st['display']['label']} fires (raw needs +{gap_pct:.0f}% to reach cutoff)",
                        "effect": f"cell shifts to {new_primary_shock}__{vb}__{era}",
                        "new_share_positive": _cell_horizon(new_cell, 60).get("share_positive"),
                        "new_n": new_cell.get("n_days", 0),
                        "tier": new_tier,
                    })
    # Banking-specific: STLFSI4 rises to cutoff → safe-haven cell activates
    bk_st = per_shock["banking_shock"]
    if not bk_st["active"] and bk_st["proximity"] is not None and bk_st["display"]["kind"] == "signed":
        sigma_gap = -bk_st["proximity"] if bk_st["proximity"] < 0 else 0
        if sigma_gap > 0:
            new_cell, new_tier = _lookup_with_fallback("banking_shock", vb, era)
            flips.append({
                "trigger": f"Banking shock fires (STLFSI4 needs +{sigma_gap:.1f}σ to reach threshold)",
                "effect": f"safe-haven cell activates (banking_shock × {vb} × {era})",
                "new_share_positive": _cell_horizon(new_cell, 60).get("share_positive"),
                "new_n": new_cell.get("n_days", 0),
                "kind": "safe_haven_override",
                "tier": new_tier,
            })
    out["what_would_flip"] = flips[:6]

    # ===== BlackRock 6-event validation =====
    try:
        br = json.loads(BLACKROCK.read_text())
        events = []
        for e in br.get("events", []):
            h60 = e["per_horizon_calendar"].get("60", {})
            events.append({
                "name": e["event_name"],
                "date": e["event_date_calendar"],
                "era": e["era"],
                "regime": e["regime"],
                "vix": e["vix_at_state"],
                "beta": e["beta_60_at_state"],
                "active_shocks": e["active_shocks_day_of_state"],
                "btc_fwd_60d": h60.get("btc_fwd"),
                "spy_fwd_60d": h60.get("spy_fwd"),
                "outperf_60d": h60.get("outperf"),
            })
        out["blackrock_events"] = events
    except Exception:
        out["blackrock_events"] = []

    # ===== 12-cell era × VIX-bin aggregate (current cell highlighted) =====
    # Aggregate across shock-types: average share-positive 60d weighted by n
    grid_12 = {}
    for era_n, _, _ in ERAS:
        for vb_n, _, _ in VIX_BINS:
            key = f"{era_n}__{vb_n}"
            grid_12[key] = {"era": era_n, "vix_bin": vb_n, "weighted_share": None,
                            "total_n": 0, "is_current": (era_n == era and vb_n == vb)}
    for c_key, c in primary.items():
        if c.get("shock") == "any": continue  # avoid double-counting
        if c.get("shock") == "none": continue
        h60 = _cell_horizon(c, 60)
        n = h60["n"]; sp = h60["share_positive"]
        if not n or sp is None: continue
        k = f"{c['era']}__{c['vix_bin']}"
        cell = grid_12[k]
        prior_n = cell["total_n"]
        if prior_n == 0:
            cell["weighted_share"] = sp
        else:
            cell["weighted_share"] = (cell["weighted_share"] * prior_n + sp * n) / (prior_n + n)
        cell["total_n"] = prior_n + n
    out["grid_12"] = list(grid_12.values())

    # ===== 84-cell full table =====
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

    # ===== Methodology (collapsed in UI) =====
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

    out["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["panel_n_rows"] = int(len(panel))

    # Recursively coerce NaN/Inf → None so browser JSON.parse() accepts the file.
    # Python's default json.dumps writes literal NaN/Infinity, which is valid Python
    # but invalid per the JSON spec; browsers will throw.
    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_clean(x) for x in o]
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o): return None
            return o
        if isinstance(o, np.floating):
            v = float(o)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        return o

    OUT.write_text(json.dumps(_clean(out), indent=2, default=str, allow_nan=False))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
