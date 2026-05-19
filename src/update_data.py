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
    """Extract (share_positive, mean_outperf, n) at horizon h from a cell dict.

    Handles two shapes:
      1. Lookup-table cell: stats nested under by_horizon_calendar[h]['outperf_calendar_fwd']
      2. Momentum-conditioned cell: stats at top level (share_positive, mean_outperf, n_days)
    """
    if not cell:
        return {"share_positive": None, "mean_outperf": None, "n": 0}
    # Shape 2: momentum-conditioned cell with top-level stats
    if "share_positive" in cell and "by_horizon_calendar" not in cell:
        return {
            "share_positive": cell.get("share_positive"),
            "mean_outperf": cell.get("mean_outperf"),
            "n": int(cell.get("n_days", cell.get("n", 0))),
        }
    # Shape 1: lookup-table cell
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


def classify_regime(vix_bin, era, active_shocks, btc_prior_60d):
    """Deterministic 5-regime classifier from synthesis of agent outputs.

    Decision tree (order matters):
      1. Banking firing AND BTC prior ≤ -10%        → REVERSAL_PRIMED
      2. BTC prior ≥ 0 AND (oil|dollar|rates|banking firing) → TREND_KILL
      3. Calm VIX AND post-ETF AND BTC prior ≥ 0 AND no trend-killer
                                                     → HIGH_BETA_BULL
      4. GPR-threat firing alone (no other shock)   → DRAG
      5. else                                        → BASELINE
    """
    trend_killers = {"oil_shock", "dollar_shock", "rate_shock", "banking_shock"}
    active = set(s for s, on in active_shocks.items() if on)
    bp = btc_prior_60d if btc_prior_60d is not None else 0.0

    if "banking_shock" in active and bp <= -0.10:
        return {
            "label": "REVERSAL_PRIMED", "display": "Reversal Primed",
            "trigger": "Banking shock firing + BTC down ≥10% prior 60d",
            "intuition": "Banking-stress is the only documented reversal catalyst. Historically catalyzes BTC bounces from drawdowns (+7.8pp Δ vs no-shock baseline). Highest-conviction long in the framework.",
            "forward_60d": "+5 to +10pp outperformance vs SPY (bimodal — buy-the-catalyst, not buy-and-hold)",
            "class": "bull",
        }
    if bp >= 0 and (active & trend_killers):
        killers_active = sorted(active & trend_killers)
        return {
            "label": "TREND_KILL", "display": "Trend Kill",
            "trigger": f"BTC up + tightening shock firing ({', '.join([s.replace('_shock','') for s in killers_active])})",
            "intuition": "Oil/dollar/rates/banking shocks kill BTC uptrends (−11 to −17pp Δ when BTC prior up). Tightening pressure neutralizes the uptrend; sometimes reverses it.",
            "forward_60d": "−10 to −15pp underperformance vs SPY (reduce or hedge)",
            "class": "bear",
        }
    if vix_bin == "calm" and era == "post_etf" and bp >= 0 and not (active & trend_killers):
        return {
            "label": "HIGH_BETA_BULL", "display": "High-Beta Bull",
            "trigger": "Calm VIX (<14.5) + post-ETF + BTC up + no trend-killer firing",
            "intuition": "The paper's signature finding: post-ETF calm-regime β rises +1.02. BTC trades as leveraged SPY with no macro headwinds. Institutional flow dominates the correlation channel.",
            "forward_60d": "+8 to +14pp outperformance vs SPY (highest-conviction long)",
            "class": "bull",
        }
    if "gprd_threat_shock" in active and len(active - {"gprd_threat_shock"}) == 0:
        return {
            "label": "DRAG", "display": "Drag",
            "trigger": "GPR-threat firing alone (no banking/oil/dollar/rates)",
            "intuition": "Geopolitical-threat shocks impose a symmetric −5pp drag regardless of BTC direction. Uncertainty premium tax, not a regime-shifter. Smallest of the five shock effects.",
            "forward_60d": "−3 to −7pp underperformance (neutral-to-mild underweight)",
            "class": "bear",
        }
    return {
        "label": "BASELINE", "display": "Baseline",
        "trigger": "No decisive shock-trigger combination",
        "intuition": "BTC tracks its usual β to SPY with no regime-specific edge. The framework's default state.",
        "forward_60d": "±2pp (no actionable signal)",
        "class": "neutral",
    }


def human_cell_label(shock, vix_bin, era):
    """Translate cell_key to human-readable label."""
    shock_lbl = {
        "none": "No shocks", "any": "Any shock",
        "oil_shock": "Oil shock", "dollar_shock": "Dollar shock",
        "rate_shock": "Rates shock", "banking_shock": "Banking shock",
        "gprd_threat_shock": "GPR-Threat shock",
    }.get(shock, shock)
    vix_lbl = {"calm": "Calm VIX", "low_stress": "Low-stress VIX",
               "mid_stress": "Mid-stress VIX", "extreme_stress": "Extreme-stress VIX"}.get(vix_bin, vix_bin)
    era_lbl = {"pre_covid": "Pre-COVID", "post_covid_pre_etf": "Post-COVID / Pre-ETF",
               "post_etf": "Post-ETF"}.get(era, era)
    return f"{shock_lbl} × {vix_lbl} × {era_lbl}"


SHOCK_MECHANISM = {
    "oil_shock": {"channel": "Real-rate / inflation pulse",
        "story": "Large WTI moves stress macro vol and signal growth/inflation surprise. Kills BTC uptrends; neutral in drawdowns."},
    "dollar_shock": {"channel": "Global liquidity drain",
        "story": "Strong dollar (top-decile move) drains global liquidity; disproportionately punishes BTC uptrends."},
    "rate_shock": {"channel": "Discount-rate transmission",
        "story": "Top-decile 10Y yield moves reprice duration risk. Mirrors oil/dollar — kills uptrends, neutral in downtrends."},
    "banking_shock": {"channel": "Safe-haven flight to non-sovereign asset",
        "story": "STLFSI4 stress historically catalyzes BTC reversals (SVB 2023 pattern). Only shock with bidirectional asymmetric effect; the framework's reversal channel."},
    "gprd_threat_shock": {"channel": "Geopolitical uncertainty premium",
        "story": "Caldara-Iacoviello GPR-threat spikes impose a symmetric −5pp drag regardless of BTC direction. Smallest of the five effects but consistent."},
}


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

    # ===== BTC prior 60d (fourth regime dimension; backward-rolling, no look-ahead) =====
    btc_cal = pd.read_parquet(BTC_CAL).copy()
    btc_cal.index = pd.to_datetime(btc_cal.index)
    btc_close = btc_cal["close"].astype(float)
    btc_prior_60d_series = btc_close / btc_close.shift(60) - 1.0
    btc_prior_today = float(btc_prior_60d_series.reindex([last_date], method="ffill").iloc[0])
    btc_prior_direction = "up" if btc_prior_today > 0 else "down"

    out = {
        "current_date": str(last_date.date()),
        "current_vix": float(last_panel["vix"]),
        "vix_bin": vix_bin_of(last_panel["vix"]),
        "current_era": era_of(last_date),
        "btc_prior_60d": btc_prior_today,
        "btc_prior_direction": btc_prior_direction,
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

    # ===== Day-over-day "what changed today" (with consequences, not just events) =====
    changed = []
    if prior_panel is not None:
        vix_diff = out["current_vix"] - float(prior_panel["vix"])
        prior_bin = vix_bin_of(float(prior_panel["vix"]))
        if abs(vix_diff) >= 0.05:
            shift_txt = (f"regime shifted from {prior_bin} to {out['vix_bin']}" if prior_bin != out['vix_bin']
                        else f"{out['vix_bin']} regime unchanged")
            changed.append({
                "kind": "vix",
                "text": f"VIX {out['current_vix']:.2f} ({vix_diff:+.2f} d/d), {shift_txt}"
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
    out["cell_human_label"] = human_cell_label(primary_shock, vb, era)
    out["cell_n"] = int(used.get("n_days", used.get("n", 0)))
    out["cell_tier_used"] = used_tier
    out["cell_regime_warning"] = used.get("regime_warning")
    out["horizons"] = {str(h): _cell_horizon(used, h) for h in HORIZONS}
    out["confidence"] = confidence_for_n(out["cell_n"])

    # ===== Regime classification (deterministic 5-label taxonomy) =====
    out["regime"] = classify_regime(vb, era, out["active_shocks"], btc_prior_today)

    # ===== Shock mechanism dictionary for prose display =====
    out["shock_mechanism"] = SHOCK_MECHANISM

    # ===== Momentum-conditioned cell stats (compute from raw panel) =====
    # For today's cell (era × vix-bin × primary_shock), split historical analogues by BTC prior 60d direction.
    # Get the historical sample matching today's cell
    panel_with_prior = panel.copy()
    panel_with_prior["btc_prior_60d"] = btc_prior_60d_series.reindex(panel.index, method="ffill")
    panel_with_prior["era"] = [era_of(d) for d in panel.index]
    panel_with_prior["vix_bin"] = panel_with_prior["vix"].apply(vix_bin_of)
    # Forward outperformance (BTC calendar - SPY NYSE) at h=60
    spy_arr = panel["spy"].astype(float).values
    nyse_arr = np.asarray(panel.index.values)
    tph_60 = nyse_arr + np.timedelta64(60, "D")
    pos_ge = np.searchsorted(nyse_arr, tph_60, side="left")
    valid = pos_ge < len(nyse_arr)
    pos_clip = np.clip(pos_ge, 0, len(nyse_arr) - 1)
    spy_fwd_60 = np.where(valid, spy_arr[pos_clip] / spy_arr - 1.0, np.nan)
    btc_fwd_60 = (btc_close.shift(-60) / btc_close - 1.0).reindex(panel.index, method="ffill")
    panel_with_prior["outperf_60"] = btc_fwd_60.values - spy_fwd_60
    # Add shock indicators from walk-forward panel
    for s in SHOCKS:
        panel_with_prior[s] = wf[f"{s}_wf"].reindex(panel.index).fillna(0).astype(int)

    def _cell_subset(era_v, vix_v, shock_v):
        """Return subset of panel matching cell + valid forward outperf + valid prior."""
        if shock_v == "none":
            mask = (panel_with_prior["era"] == era_v) & (panel_with_prior["vix_bin"] == vix_v)
            for s in SHOCKS:
                mask &= (panel_with_prior[s] == 0)
        elif shock_v == "any":
            mask = (panel_with_prior["era"] == era_v) & (panel_with_prior["vix_bin"] == vix_v)
            shock_mask = pd.Series(False, index=panel_with_prior.index)
            for s in SHOCKS:
                shock_mask |= (panel_with_prior[s] == 1)
            mask &= shock_mask
        else:
            mask = (panel_with_prior["era"] == era_v) & (panel_with_prior["vix_bin"] == vix_v) & (panel_with_prior[shock_v] == 1)
        return panel_with_prior[mask & panel_with_prior["outperf_60"].notna() & panel_with_prior["btc_prior_60d"].notna()]

    def _split_stats(subset):
        up = subset[subset["btc_prior_60d"] > 0]
        dn = subset[subset["btc_prior_60d"] <= 0]
        def stats(s):
            if not len(s): return {"n": 0, "share_positive": None, "mean_outperf": None}
            return {"n": int(len(s)), "share_positive": float((s["outperf_60"] > 0).mean()),
                    "mean_outperf": float(s["outperf_60"].mean())}
        return {"prior_up": stats(up), "prior_down": stats(dn), "aggregate": stats(subset)}

    cell_subset = _cell_subset(era, vb, primary_shock)
    out["momentum_conditioned"] = _split_stats(cell_subset)
    # The "matched" sub-cell is the one matching today's BTC prior direction
    matched_key = f"prior_{btc_prior_direction}"
    out["momentum_conditioned"]["matched"] = out["momentum_conditioned"][matched_key]
    out["momentum_conditioned"]["matched_direction"] = btc_prior_direction
    out["momentum_conditioned_confidence"] = confidence_for_n(out["momentum_conditioned"]["matched"]["n"])

    # ===== Effect-size matrix: per-shock Δ vs no-shock baseline, split by BTC prior =====
    # For each shock, full-sample mean outperf for (shock_fire, prior↑) vs (no_shock, prior↑), and same for prior↓
    es_matrix = []
    for s in SHOCKS:
        # Shock fires
        fire_mask = (panel_with_prior[s] == 1)
        fire_up = panel_with_prior[fire_mask & (panel_with_prior["btc_prior_60d"] > 0) & panel_with_prior["outperf_60"].notna()]
        fire_dn = panel_with_prior[fire_mask & (panel_with_prior["btc_prior_60d"] <= 0) & panel_with_prior["outperf_60"].notna()]
        # No shock (this specific shock not firing)
        no_mask = (panel_with_prior[s] == 0)
        no_up = panel_with_prior[no_mask & (panel_with_prior["btc_prior_60d"] > 0) & panel_with_prior["outperf_60"].notna()]
        no_dn = panel_with_prior[no_mask & (panel_with_prior["btc_prior_60d"] <= 0) & panel_with_prior["outperf_60"].notna()]
        def mu(s): return float(s["outperf_60"].mean()) if len(s) else None
        delta_up = (mu(fire_up) - mu(no_up)) * 100 if (mu(fire_up) is not None and mu(no_up) is not None) else None
        delta_dn = (mu(fire_dn) - mu(no_dn)) * 100 if (mu(fire_dn) is not None and mu(no_dn) is not None) else None
        asymmetry = abs(delta_up - delta_dn) if (delta_up is not None and delta_dn is not None) else None
        # Classify behavior
        kind = None
        if delta_up is not None and delta_dn is not None:
            if abs(delta_up) > 5 and abs(delta_dn) > 5 and np.sign(delta_up) != np.sign(delta_dn):
                kind = "reversal"  # shock changes direction
            elif abs(delta_up) > 5 and abs(delta_dn) < 3:
                kind = "trend_killer"  # only kills uptrends
            elif abs(delta_up) < 5 and abs(delta_dn) > 5 and delta_dn > 0:
                kind = "catalyst"  # only catalyzes from down
            elif abs(asymmetry or 0) < 5:
                kind = "drag"  # symmetric drag
            else:
                kind = "mixed"
        es_matrix.append({
            "shock": s,
            "label": SHOCK_DISPLAY[s]["label"],
            "prior_up_delta_pp": delta_up,
            "prior_down_delta_pp": delta_dn,
            "asymmetry_pp": asymmetry,
            "kind": kind,
            "n_fire_up": int(len(fire_up)), "n_fire_dn": int(len(fire_dn)),
        })
    out["effect_size_matrix"] = es_matrix

    # ===== Build the momentum-conditioned panel ONCE, reuse everywhere =====
    # This is the panel with BTC prior 60d + forward-60d outperf + shock flags joined.
    # Used by: what_would_flip lookup, momentum_conditioned cell stats, grid_12 view,
    # 84-cell sub-cell stats. Build once for efficiency.
    panel_with_prior = panel.copy()
    panel_with_prior["btc_prior_60d"] = btc_prior_60d_series.reindex(panel.index, method="ffill")
    panel_with_prior["era"] = [era_of(d) for d in panel.index]
    panel_with_prior["vix_bin"] = panel_with_prior["vix"].apply(vix_bin_of)
    spy_arr2 = panel["spy"].astype(float).values
    nyse_arr2 = np.asarray(panel.index.values)
    tph_60_2 = nyse_arr2 + np.timedelta64(60, "D")
    pos_ge_2 = np.searchsorted(nyse_arr2, tph_60_2, side="left")
    valid2 = pos_ge_2 < len(nyse_arr2)
    pos_clip_2 = np.clip(pos_ge_2, 0, len(nyse_arr2) - 1)
    spy_fwd_60_2 = np.where(valid2, spy_arr2[pos_clip_2] / spy_arr2 - 1.0, np.nan)
    btc_fwd_60_2 = (btc_close.shift(-60) / btc_close - 1.0).reindex(panel.index, method="ffill")
    panel_with_prior["outperf_60"] = btc_fwd_60_2.values - spy_fwd_60_2
    for s in SHOCKS:
        panel_with_prior[s] = wf[f"{s}_wf"].reindex(panel.index).fillna(0).astype(int)

    def _cell_subset_by_shock_vix_era(shock_v, vb_v, era_v):
        """Subset of panel_with_prior matching the (shock, vix_bin, era) cell."""
        if shock_v == "none":
            mask = (panel_with_prior["era"] == era_v) & (panel_with_prior["vix_bin"] == vb_v)
            for s in SHOCKS:
                mask &= (panel_with_prior[s] == 0)
        elif shock_v == "any":
            mask = (panel_with_prior["era"] == era_v) & (panel_with_prior["vix_bin"] == vb_v)
            shock_mask = pd.Series(False, index=panel_with_prior.index)
            for s in SHOCKS:
                shock_mask |= (panel_with_prior[s] == 1)
            mask &= shock_mask
        else:
            mask = ((panel_with_prior["era"] == era_v)
                    & (panel_with_prior["vix_bin"] == vb_v)
                    & (panel_with_prior[shock_v] == 1))
        return panel_with_prior[mask & panel_with_prior["outperf_60"].notna()
                                & panel_with_prior["btc_prior_60d"].notna()]

    def _lookup_with_fallback(shock, vix_bin, era_name, prior_dir=None):
        """Momentum-conditioned cell lookup with 3-tier fallback.
        prior_dir: 'up' | 'down' | None (unconditional)."""
        if prior_dir is not None:
            sub = _cell_subset_by_shock_vix_era(shock, vix_bin, era_name)
            if prior_dir == "up":
                sub = sub[sub["btc_prior_60d"] > 0]
            elif prior_dir == "down":
                sub = sub[sub["btc_prior_60d"] <= 0]
            if len(sub) >= 5:
                return ({
                    "n_days": int(len(sub)),
                    "share_positive": float((sub["outperf_60"] > 0).mean()),
                    "mean_outperf": float(sub["outperf_60"].mean()),
                }, "primary_momentum_conditioned")
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
            new_cell, new_tier = _lookup_with_fallback(primary_shock, new_bin, era, prior_dir=btc_prior_direction)
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
            new_cell, new_tier = _lookup_with_fallback(new_primary_shock, vb, era, prior_dir=btc_prior_direction)
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
                    new_cell, new_tier = _lookup_with_fallback(new_primary_shock, vb, era, prior_dir=btc_prior_direction)
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
            new_cell, new_tier = _lookup_with_fallback("banking_shock", vb, era, prior_dir=btc_prior_direction)
            flips.append({
                "trigger": f"Banking shock fires (STLFSI4 needs +{sigma_gap:.1f}σ to reach threshold)",
                "effect": f"safe-haven cell activates (banking_shock × {vb} × {era})",
                "new_share_positive": _cell_horizon(new_cell, 60).get("share_positive"),
                "new_n": new_cell.get("n_days", 0),
                "kind": "safe_haven_override",
                "tier": new_tier,
            })
    out["what_would_flip"] = flips[:6]

    # ===== BlackRock 6-event validation — augmented with BTC prior 60d at event =====
    try:
        br = json.loads(BLACKROCK.read_text())
        events = []
        for e in br.get("events", []):
            h60 = e["per_horizon_calendar"].get("60", {})
            event_date = pd.Timestamp(e["event_date_calendar"])
            # BTC prior 60d at the event date
            ev_btc_prior = None
            try:
                ev_btc_prior = float(btc_prior_60d_series.reindex([event_date], method="ffill").iloc[0])
            except Exception:
                pass
            ev_btc_dir = "up" if (ev_btc_prior or 0) > 0 else "down"
            events.append({
                "name": e["event_name"],
                "date": e["event_date_calendar"],
                "era": e["era"],
                "regime": e["regime"],
                "vix": e["vix_at_state"],
                "beta": e["beta_60_at_state"],
                "active_shocks": e["active_shocks_day_of_state"],
                "btc_prior_60d": ev_btc_prior,
                "btc_prior_direction": ev_btc_dir,
                "btc_fwd_60d": h60.get("btc_fwd"),
                "spy_fwd_60d": h60.get("spy_fwd"),
                "outperf_60d": h60.get("outperf"),
            })
        out["blackrock_events"] = events
    except Exception:
        out["blackrock_events"] = []

    # ===== 12-cell era × VIX-bin aggregate, momentum-conditioned =====
    # Build three views: aggregate (prior-direction-agnostic), prior_up, prior_down.
    # Each cell shows weighted share-positive 60d across shock types within the cell.
    def _build_grid_view(prior_filter):
        """prior_filter: None (aggregate), 'up' (BTC prior > 0), 'down' (BTC prior <= 0)"""
        grid = {}
        for era_n, _, _ in ERAS:
            for vb_n, _, _ in VIX_BINS:
                key = f"{era_n}__{vb_n}"
                grid[key] = {"era": era_n, "vix_bin": vb_n,
                             "weighted_share": None, "total_n": 0,
                             "is_current": (era_n == era and vb_n == vb)}
        # Iterate raw panel rows for accurate momentum-conditioned aggregation
        for ts, row in panel_with_prior.iterrows():
            if pd.isna(row.get("outperf_60")): continue
            bp = row.get("btc_prior_60d")
            if pd.isna(bp): continue
            if prior_filter == "up" and bp <= 0: continue
            if prior_filter == "down" and bp > 0: continue
            k = f"{row['era']}__{row['vix_bin']}"
            if k not in grid: continue
            cell = grid[k]
            # Running mean of share_positive (treat each obs as 1)
            cell["total_n"] += 1
            sp = 1.0 if row["outperf_60"] > 0 else 0.0
            if cell["weighted_share"] is None:
                cell["weighted_share"] = sp
            else:
                n = cell["total_n"]
                cell["weighted_share"] = ((cell["weighted_share"] * (n - 1)) + sp) / n
        return list(grid.values())

    out["grid_12_aggregate"] = _build_grid_view(None)
    out["grid_12_prior_up"] = _build_grid_view("up")
    out["grid_12_prior_down"] = _build_grid_view("down")
    # Default view = today's direction
    out["grid_12"] = out[f"grid_12_prior_{btc_prior_direction}"]

    # ===== 84-cell full table, momentum-conditioned =====
    # For each primary cell, compute aggregate + prior_up + prior_down sub-stats from raw panel
    def _cell_subset_by_shock_vix_era(shock_v, vb_v, era_v):
        if shock_v == "none":
            mask = (panel_with_prior["era"] == era_v) & (panel_with_prior["vix_bin"] == vb_v)
            for s in SHOCKS:
                mask &= (panel_with_prior[s] == 0)
        elif shock_v == "any":
            mask = (panel_with_prior["era"] == era_v) & (panel_with_prior["vix_bin"] == vb_v)
            shock_mask = pd.Series(False, index=panel_with_prior.index)
            for s in SHOCKS:
                shock_mask |= (panel_with_prior[s] == 1)
            mask &= shock_mask
        else:
            mask = ((panel_with_prior["era"] == era_v)
                    & (panel_with_prior["vix_bin"] == vb_v)
                    & (panel_with_prior[shock_v] == 1))
        return panel_with_prior[mask & panel_with_prior["outperf_60"].notna()
                                & panel_with_prior["btc_prior_60d"].notna()]

    cells = []
    for key, c in primary.items():
        h60 = _cell_horizon(c, 60)
        sub = _cell_subset_by_shock_vix_era(c.get("shock"), c.get("vix_bin"), c.get("era"))
        up = sub[sub["btc_prior_60d"] > 0]
        dn = sub[sub["btc_prior_60d"] <= 0]
        def _ss(s):
            if not len(s): return {"n": 0, "share_positive": None, "mean_outperf": None}
            return {"n": int(len(s)),
                    "share_positive": float((s["outperf_60"] > 0).mean()),
                    "mean_outperf": float(s["outperf_60"].mean())}
        cells.append({
            "key": key, "shock": c.get("shock"), "vix_bin": c.get("vix_bin"), "era": c.get("era"),
            "n": int(c.get("n_days", 0)),
            "share_positive_60d": h60["share_positive"],
            "mean_outperf_60d": h60["mean_outperf"],
            "regime_warning": c.get("regime_warning"),
            "prior_up": _ss(up),
            "prior_down": _ss(dn),
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
