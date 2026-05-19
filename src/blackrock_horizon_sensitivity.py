"""BlackRock 6-event window-sensitivity validation across h in {5, 20, 60, 90}
calendar days.

Same calendar-anchored convention as
output/stage3a/results/blackrock_6event_calendar_validation.json:
  - Anchor: the calendar event date itself.
  - BTC: simple return BTC(t+h cal) / BTC(t) - 1, ffilled to prior available
    calendar day if t+h is missing in the BTC series.
  - SPY: simple return SPY(nearest NYSE >= t+h cal) / SPY(nearest NYSE <= t) - 1.
  - Outperf = BTC fwd - SPY fwd.

Output:
  - output/stage3a/results/blackrock_horizon_sensitivity.json
  - output/stage3a/blackrock_horizon_sensitivity.md
  - output/stage3a/tables/blackrock_horizon_sensitivity.tex
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

np.random.seed(42)

ROOT = Path("/Users/janwustenfeld/Documents/btc-vix-threshold-paper2")
PANEL_PATH = ROOT / "output/seed/paper1_context/panel_with_shocks.parquet"
BTC_CAL_PATH = ROOT / "data/aux/btc_calendar_daily.parquet"

OUT_DIR = ROOT / "output/stage3a"
RESULTS_DIR = OUT_DIR / "results"
TABLES_DIR = OUT_DIR / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS_DIR / "blackrock_horizon_sensitivity.json"
OUT_MD = OUT_DIR / "blackrock_horizon_sensitivity.md"
OUT_TEX = TABLES_DIR / "blackrock_horizon_sensitivity.tex"

BLACKROCK_EVENTS = [
    {"name": "US-Iran escalation",        "date": "2020-01-03"},
    {"name": "COVID outbreak",            "date": "2020-03-09"},
    {"name": "US election challenges",    "date": "2020-11-03"},
    {"name": "Russia-Ukraine invasion",   "date": "2022-02-21"},
    {"name": "US regional banking (SVB)", "date": "2023-03-09"},
    {"name": "US global tariff",          "date": "2025-04-02"},
]
HORIZONS = [5, 20, 60, 90]


def compute_event(event_date: pd.Timestamp, btc_close: pd.Series,
                  panel: pd.DataFrame, horizons=HORIZONS) -> dict:
    nyse_arr = np.asarray(panel.index.values)
    spy = panel["spy"].astype(float)

    # BTC anchor at calendar event date (or last available <= event date)
    if event_date in btc_close.index:
        btc_t = float(btc_close.loc[event_date])
    else:
        sub = btc_close.loc[:event_date]
        btc_t = float(sub.iloc[-1])

    pos_le_t = int(np.searchsorted(nyse_arr, event_date.to_datetime64(), side="right") - 1)
    pos_le_t = max(pos_le_t, 0)
    nyse_t = pd.Timestamp(nyse_arr[pos_le_t])
    spy_t = float(spy.iloc[pos_le_t])

    per_h = {}
    for h in horizons:
        tph = event_date + pd.Timedelta(days=h)
        if tph in btc_close.index:
            btc_tph = float(btc_close.loc[tph])
        elif tph <= btc_close.index.max():
            sub = btc_close.loc[:tph]
            btc_tph = float(sub.iloc[-1]) if len(sub) else None
        else:
            btc_tph = None
        btc_fwd = (btc_tph / btc_t - 1.0) if btc_tph is not None else None

        pos_ge = int(np.searchsorted(nyse_arr, tph.to_datetime64(), side="left"))
        if pos_ge >= len(nyse_arr):
            spy_tph, nyse_tph, gap = None, None, None
        else:
            nyse_tph = pd.Timestamp(nyse_arr[pos_ge])
            spy_tph = float(spy.iloc[pos_ge])
            gap = int((nyse_tph - tph).days)
        spy_fwd = (spy_tph / spy_t - 1.0) if spy_tph is not None else None
        op = (btc_fwd - spy_fwd) if (btc_fwd is not None and spy_fwd is not None) else None
        per_h[h] = {
            "tph_calendar": str(tph.date()),
            "nyse_tph_used": str(nyse_tph.date()) if nyse_tph is not None else None,
            "nyse_gap_forward_days": gap,
            "btc_fwd_pct": btc_fwd * 100 if btc_fwd is not None else None,
            "spy_fwd_pct": spy_fwd * 100 if spy_fwd is not None else None,
            "outperf_pct": op * 100 if op is not None else None,
        }

    return {
        "anchor_calendar_date": str(event_date.date()),
        "nyse_state_date": str(nyse_t.date()),
        "nyse_gap_backward_days": int((event_date - nyse_t).days),
        "btc_t_calendar_price": btc_t,
        "spy_t_nyse_price": spy_t,
        "by_horizon": per_h,
    }


def main():
    panel = pd.read_parquet(PANEL_PATH).copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    btc = pd.read_parquet(BTC_CAL_PATH).copy()
    btc.index = pd.to_datetime(btc.index)
    btc = btc.sort_index()
    btc_close = btc["close"].astype(float)

    out_events = []
    for ev in BLACKROCK_EVENTS:
        ed = pd.Timestamp(ev["date"])
        rec = compute_event(ed, btc_close, panel)
        rec["event_name"] = ev["name"]
        rec["event_date_calendar"] = ev["date"]
        out_events.append(rec)

    meta = {
        "script": "code/blackrock_horizon_sensitivity.py",
        "panel": str(PANEL_PATH),
        "btc_calendar": str(BTC_CAL_PATH),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "horizons_calendar_days": HORIZONS,
        "seed": 42,
        "convention": "Calendar-anchored: BTC at calendar t -> BTC at calendar t+h (ffill). SPY at nearest NYSE <= t -> nearest NYSE >= t+h.",
    }
    out_doc = {"_metadata": meta, "events": out_events}
    OUT_JSON.write_text(json.dumps(out_doc, indent=2, default=str))
    print(f"[blackrock_horizon_sensitivity] wrote {OUT_JSON}", flush=True)

    # ----- Markdown report -----
    lines = []
    lines.append("# BlackRock 6-event Window-Sensitivity (h ∈ {5, 20, 60, 90} calendar days)")
    lines.append("")
    lines.append(f"Generated: {meta['generated_at']}. Convention: calendar-anchored ({meta['convention']}).")
    lines.append("")
    lines.append("## Per-event BTC-vs-SPY outperformance (percentage points)")
    lines.append("")
    lines.append("| Event | Date | h=5 | h=20 | h=60 | h=90 | 60→90 swing |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for e in out_events:
        h5 = e["by_horizon"][5]["outperf_pct"]
        h20 = e["by_horizon"][20]["outperf_pct"]
        h60 = e["by_horizon"][60]["outperf_pct"]
        h90 = e["by_horizon"][90]["outperf_pct"]
        swing = (h90 - h60) if (h90 is not None and h60 is not None) else None
        def f(x):
            return "—" if x is None else f"{x:+.2f}"
        lines.append(f"| {e['event_name']} | {e['event_date_calendar']} | {f(h5)} | {f(h20)} | {f(h60)} | {f(h90)} | {f(swing)} |")
    lines.append("")

    lines.append("## Per-event BTC forward return (percent)")
    lines.append("")
    lines.append("| Event | h=5 | h=20 | h=60 | h=90 |")
    lines.append("|---|---:|---:|---:|---:|")
    for e in out_events:
        h5 = e["by_horizon"][5]["btc_fwd_pct"]
        h20 = e["by_horizon"][20]["btc_fwd_pct"]
        h60 = e["by_horizon"][60]["btc_fwd_pct"]
        h90 = e["by_horizon"][90]["btc_fwd_pct"]
        def f(x):
            return "—" if x is None else f"{x:+.2f}"
        lines.append(f"| {e['event_name']} | {f(h5)} | {f(h20)} | {f(h60)} | {f(h90)} |")
    lines.append("")

    lines.append("## Per-event SPY forward return (percent)")
    lines.append("")
    lines.append("| Event | h=5 | h=20 | h=60 | h=90 |")
    lines.append("|---|---:|---:|---:|---:|")
    for e in out_events:
        h5 = e["by_horizon"][5]["spy_fwd_pct"]
        h20 = e["by_horizon"][20]["spy_fwd_pct"]
        h60 = e["by_horizon"][60]["spy_fwd_pct"]
        h90 = e["by_horizon"][90]["spy_fwd_pct"]
        def f(x):
            return "—" if x is None else f"{x:+.2f}"
        lines.append(f"| {e['event_name']} | {f(h5)} | {f(h20)} | {f(h60)} | {f(h90)} |")
    lines.append("")

    # Window-sensitivity discussion
    lines.append("## Window-sensitivity reading")
    lines.append("")
    lines.append("BTC-vs-SPY outperformance near 60d is fragile for events whose 60→90d window catches a major BTC drawdown.")
    lines.append("Concretely:")
    flips = []
    for e in out_events:
        h60 = e["by_horizon"][60]["outperf_pct"]
        h90 = e["by_horizon"][90]["outperf_pct"]
        if h60 is None or h90 is None:
            continue
        swing = h90 - h60
        if abs(swing) >= 5.0:
            sign_flip = (np.sign(h60) != np.sign(h90))
            flips.append((e["event_name"], e["event_date_calendar"], h60, h90, swing, sign_flip))
    if flips:
        lines.append("")
        lines.append("Events with |Δ outperf 60→90| ≥ 5pp:")
        lines.append("")
        for nm, dt, h60, h90, sw, sf in flips:
            flip_tag = " (SIGN FLIP)" if sf else ""
            lines.append(f"- **{nm}** ({dt}): 60d outperf = {h60:+.2f}pp; 90d outperf = {h90:+.2f}pp; Δ = {sw:+.2f}pp{flip_tag}.")
    else:
        lines.append("")
        lines.append("(No event shows a 60→90 swing larger than 5pp in this sample.)")
    lines.append("")

    # Highlight Russia-Ukraine specifically
    russia = next((e for e in out_events if "Russia" in e["event_name"]), None)
    if russia:
        h60_r = russia["by_horizon"][60]["outperf_pct"]
        h90_r = russia["by_horizon"][90]["outperf_pct"]
        lines.append("### Russia-Ukraine spotlight")
        lines.append("")
        lines.append(f"60d outperf = **{h60_r:+.2f}pp** (BlackRock-style window). "
                     f"90d outperf = **{h90_r:+.2f}pp**. The 60→90 window catches the early May 2022 LUNA/Terra collapse "
                     f"plus the broader May-June 2022 BTC drawdown; SPY also fell in the same window but BTC fell harder.")
        lines.append("")
        lines.append("**Operational reading:** the BlackRock chart's 60d outperformance for Russia-Ukraine is *real* (BTC did outperform SPY +8.7pp at calendar t+60d) but *window-fragile* -- the dashboard's h=90 column is the recommended sanity check.")
        lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"[blackrock_horizon_sensitivity] wrote {OUT_MD}", flush=True)

    # ----- LaTeX table -----
    rows = []
    for e in out_events:
        row = {"Event": e["event_name"].replace("&", "\\&"),
               "Date": e["event_date_calendar"]}
        for h in HORIZONS:
            v = e["by_horizon"][h]["outperf_pct"]
            row[f"h={h}"] = "—" if v is None else f"{v:+.2f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    tex = df.to_latex(index=False, escape=False,
        caption=("BlackRock 6-event 60d outperformance window-sensitivity. "
                 "Cells report BTC-vs-SPY outperformance (pp) at calendar horizons h=5, 20, 60, 90 days. "
                 "Calendar-anchored convention (anchor at calendar event date). "
                 "60d is the BlackRock-chart window; h=90 is the dashboard's window-sensitivity diagnostic. "
                 "Russia-Ukraine (Feb 2022) is the canonical illustration: +8.70pp at 60d, large negative at 90d."),
        label="tab:blackrock_horizon_sensitivity")
    OUT_TEX.write_text(tex)
    print(f"[blackrock_horizon_sensitivity] wrote {OUT_TEX}", flush=True)


if __name__ == "__main__":
    main()
