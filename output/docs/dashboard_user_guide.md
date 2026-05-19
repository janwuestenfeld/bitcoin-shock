# BTC Regime Dashboard — Operational User Guide

*One-page reference for the framework built from Paper 2 (BTC-VIX threshold)
on a post-COVID-restricted basis (2020-03 onward).*

---

## What the dashboard does

Given today's `(VIX, active shock-flags, calendar date)`, output a **directional
bias** for BTC's β with SPY and forward returns at h = 5 / 20 / 60 / 90 calendar
days. Not a point forecast — a probabilistic bias call with confidence band.

**Input convention:** any calendar date works (weekends, US holidays OK). VIX
and shock state snap backward to the most recent NYSE day; staleness is
reported. BTC return windows use BTC's 24/7 calendar series; SPY return windows
use the nearest NYSE day ≥ end date.

---

## The framework's three operational signals

### 1. Cell-conditional directional bias (the headline)

For each `(shock-type × VIX-bin × era)` cell, the dashboard reports the
historical share of 60d forward outperformance (BTC return − SPY return > 0):

| Bias class | Cell share-positive | Examples |
|-----------|--------------------:|----------|
| **Strongly bullish** | ≥ 65% | Banking+extreme-stress (71%); Dollar+extreme-stress (67%); Rate+low-stress (67%) |
| **Moderately bullish** | 55-65% | Dollar+low-stress; GPR+low-stress-post-ETF |
| **Neutral / no edge** | 45-55% | Oil+extreme; Most calm-VIX cells in post-ETF era |
| **Moderately bearish** | 35-45% | Most mid-stress non-banking cells |
| **Strongly bearish** | ≤ 35% | Banking+mid-stress (27%); GPR+mid-stress-pre-ETF (27%, Russia-Ukraine cell); Oil+mid-stress (31%) |

Share-positive ranges 27% to 71% across cells — that's the framework's
discriminating power. Use it for **bias-tilt sizing**, not for return prediction.

### 2. Banking-stress safe-haven override

When `banking_shock` is active (STLFSI4 top decile), apply the safe-haven
override regardless of VIX level: predict Δβ = −0.33 (BTC decouples from SPY).
Confirmed by SVB direct test: realized +17pp outperformance at calendar 60d.
This is the only shock with a positive M4 ridge loading in the formal OOS test
— it's the single load-bearing per-shock structural signal.

### 3. Horizon-sensitivity disclosure

Report all four horizons (h = 5, 20, 60, 90). Most BlackRock-style events are
robust between 60d and 90d. The one exception is Russia-Ukraine-style events
where a major BTC-specific tail event (LUNA collapse) lands inside the 60→90
window — then the sign can flip. Always display both 60d and 90d so the user
can see window-fragility.

---

## How to read a prediction

```
Input: 2026-05-19, VIX = 17.3, gpr_threat active, post-ETF era

Output:
  Regime: stress (VIX > 14.5)
  Cell: gpr_threat_shock × low_stress × post_etf  (n=45 analogues)
  Share-positive at 60d: 60% → MODERATELY BULLISH bias
  Cell mean outperformance: +2.3pp (95% CI: −15pp to +20pp)
  Banking override: not active
  Horizon-sensitivity: h=60 +2.3pp, h=90 similar (no major-tail-event risk in
  this analogue distribution)

  Confidence: medium (n=45 in primary cell; no fallback needed)
  Staleness: shock indicators 140 days old (panel ends 2025-12-30 — REFRESH
  before live use)
```

---

## What to use this for

- **Position-tilt sizing:** scale long-BTC vs short-BTC exposure based on cell
  bias-tilt. A 71%-positive cell warrants higher long allocation than a
  50%-positive cell.
- **Risk-off vs risk-on calls:** stress regime + non-banking shock → BTC tracks
  SPY downward; reduce risk. Stress regime + banking shock → safe-haven; can
  hold or add.
- **Event-window sizing:** when entering a known shock event (FOMC, GPR
  flashpoint, banking-stress signal), look up the cell and adjust horizon and
  size accordingly.
- **Counter-narrative discipline:** when BlackRock-style "BTC outperformed in
  event X" claims surface, look up the cell's share-positive and check whether
  the claim is a typical draw from a bullish cell or an upper-tail draw from a
  bearish cell. The latter doesn't generalize.

## What NOT to use this for

- **Point return prediction.** OOS R² is ~6% on the full panel, near zero on
  shock-active subsamples. The framework is a bias-detector, not an
  alpha-extractor. Point estimates have ±15-30pp CI at h=60; don't trade them
  as forecasts.
- **Same-day β jumps.** β_60 is a 60-day rolling object that updates slowly.
  "Today's shock → tomorrow's β" isn't what this estimates. Use cell membership
  as a regime classifier instead.
- **Pre-COVID regime carry-over.** Pre-COVID BTC was structurally uncorrelated
  with equities — different asset class. The dashboard correctly flags
  pre-COVID cells with `regime_warning: different_regime_use_with_caution`. Do
  not extrapolate post-COVID predictions to pre-COVID conditions, or assume
  pre-COVID behavior will reassert under a future institutional retreat.

---

## Calibrated probabilistic expectations

A bearish-cell prediction (e.g., 27% share-positive at 60d) means **73% of
historical analogues underperformed**. Roughly 1-in-4 such predictions will
land on the wrong side of the cell mean even when the framework is performing
exactly as designed. This is *probabilistic prediction working correctly*, not
framework failure.

Concrete example: Russia-Ukraine (2022-02-21) sat in a 27%-positive cell with
mean −7.7pp at h=60. Realized outperformance was +8.7pp — a 27%-tail draw.
The framework predicted "short-bias with size-adjusted conviction"; the realized
outcome was in the 27% of cases where short-bias is wrong. That's not a miss;
that's calibrated uncertainty operating.

For sizing: think of cell share-positive as a **win-rate**, not a guarantee. A
71%-positive cell justifies higher conviction than 55%; both are useful biases;
neither is a sure thing.

---

## Maintenance: refresh requirements

The dashboard's panel ends 2025-12-30. Before live operational use, run a
refresh script that pulls:

- **VIX** (FRED `VIXCLS`) through today's date
- **STLFSI4, DCOILWTICO, DTWEXBGS, DGS10, GPR-Threats** through today
- **BTC daily close** from CoinMetrics community API through today
- **NFCI** weekly through latest available print
- **SPY daily close** from yfinance through today

Then re-run `code/dashboard_lookup.py` to extend the lookup table and the
`code/dashboard_forecast.py` CLI will pick up the fresh data automatically.

Without refresh, today's prediction relies on shock indicators 140+ days stale
— display the staleness flag prominently and treat the prediction as
diagnostic-only until refreshed.

---

## Key files

- `code/dashboard_forecast.py` — CLI entry point. Run with `--date YYYY-MM-DD`.
- `code/dashboard_lookup.py` — builds the lookup table from the panel.
- `output/stage3a/results/dashboard_lookup_table.json` — 84-cell base table at h=5/20/60/90.
- `output/stage3a/results/dashboard_lookup_table_with_trend.json` — augmented with pre-shock trend sub-cells (diagnostic).
- `output/stage3a/dashboard_design.md` — full design + cell definitions + worked examples.
- `output/stage3a/blackrock_horizon_sensitivity.md` — BlackRock 6-event window-sensitivity (5/6 robust 60→90; Russia-Ukraine fragile via LUNA).
- `output/stage3a/post_covid_restriction.md` — post-COVID-only restriction (the dashboard's empirical basis vs the paper's full-panel basis).
- `output/stage3a/shock_adds_value_test.md` — formal OOS R² test validating per-shock structure (+2.94pp at p=0.005).

---

*Built post-pipeline from Paper 2 (BTC-VIX threshold) outputs. The paper itself
ships at field tier (JFQA target); the dashboard is the operational
deliverable on the cleaner post-COVID basis. Last update: 2026-05-19.*
