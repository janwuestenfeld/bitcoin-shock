# HANDOFF — Bitcoin Conditional Shock Regime Classifier

**Last updated:** 2026-05-19
**For:** the next Claude session (or any new collaborator) picking up this project

This document captures the full project state, methodology, code inventory, and
open items so a fresh agent can resume without losing context.

---

## What this project is

A **walk-forward, look-ahead-free regime classifier** that predicts whether BTC
will outperform SPY over the next 60 calendar days, conditional on:
- VIX bin (calm <14.5 / low-stress 14.5–20 / mid-stress 20–30 / extreme ≥30)
- Era (pre-COVID / post-COVID-pre-ETF / post-ETF, with breakpoints 2020-03-01 and 2024-01-10)
- Five macro-shock indicators (oil, dollar, rates, banking-stress, GPR-threat)
- BTC's prior 60-day return (momentum baseline)

Lives at `github.com/janwuestenfeld/bitcoin-shock`. Repo deploys as a GitHub
Pages site (Plotly + vanilla JS, single-page dashboard). Daily refresh via
GitHub Actions cron.

**Core thesis:** the conventional narrative "BTC = digital safe-haven →
all macro shocks help BTC" is empirically wrong. Macro shocks behave
fundamentally differently:
- **Banking-stress** = bidirectional reversal catalyst (changes BTC trajectory)
- **Oil / Dollar / Rates** = trend killers (terminate BTC uptrends, neutral in downtrends)
- **GPR-threat** = uniform symmetric drag (small −5pp regardless of direction)

The dashboard's value is showing **when** a shock carries information
and **what kind**, conditional on the regime.

---

## Empirical headline

| Test | Result |
|---|---|
| ΔR² M2→M3 (walk-forward) | **+3.32pp** at p=0.005 |
| ΔR² beyond BTC momentum | **+1.97pp** OOS R² |
| Banking M4 coefficient (post-ETF, BTC down) | +17.3pp marginal contribution |
| Banking M4 coefficient (post-ETF, BTC up) | +8.7pp marginal contribution |
| GPR effect-size asymmetry | 1.3pp (symmetric drag) |
| Banking effect-size asymmetry | 24.4pp (largest — reversal) |

---

## Methodology (the four-stage iteration that landed here)

Through a sequence of user-driven methodological pressure tests, the framework converged on the current spec:

### 1. Full-sample cutoffs (paper basis)
- Top-decile of |Δlog WTI|, |Δ USD-broad|, etc., over the entire 2014-2025 panel.
- **Bias:** mixes pre-COVID and post-COVID regimes; post-COVID days fire at 14-21% (vs nominal 10%) because the cutoff is anchored to lower pre-COVID volatility.
- Headline ΔR² M2→M3 = +2.94pp, but inflated by cross-era cutoff mixing.

### 2. EWMA-126 rolling cutoffs
- Top-decile of an exponentially-weighted moving distribution, halflife 126 trading days (~6 calendar months).
- **Bias:** smears extreme and non-extreme periods *within* a window. ΔR² collapses to +0.94pp.
- Banking loses its unique positive M4 coefficient (was +0.52pp, now −1.82pp).
- Verdict: BORDERLINE.

### 3. Era-conditional with within-era full-distribution cutoff
- Top-decile within each era's full distribution.
- **Honest about a subtle look-ahead** (uses end-of-era cutoff retroactively), but reasonable for the OOS R² test (random K-fold CV, cutoff is a feature not outcome).
- ΔR² = +0.76pp (further attenuated).
- The user correctly flagged this as look-ahead-tainted.

### 4. Walk-forward + q99 tail fallback ← **CURRENT SPEC**
- Within-era expanding cutoff: at each date t, compute the q90 of within-era observations from era-start through t-1.
- Warmup phase (first 60 within-era obs): use prior-era q99 (tail-quantile) as fallback. This catches obvious tail events (e.g., COVID crash, which exceeds prior-era q99 across oil/dollar/banking) without introducing the prior-era-q90 mechanical bias.
- **Zero look-ahead.** Operator can compute today's cutoff with only data through today.
- ΔR² M2→M3 = **+3.32pp at p=0.005**, KEEP_FULL_STRUCTURE verdict.
- Banking-shock M4 coefficient flipped sign vs the COVID-excluded version: from +1.76pp to −1.30pp. The "banking safe-haven" channel is **post-ETF cell-conditional, not unconditional**.

### Why walk-forward delivers a STRONGER signal than look-ahead variants
The user identified the epistemological reason: a shock is a deviation from the **information available at the time**. Future observations cannot influence contemporaneous investor decisions, so they shouldn't be used to label which days were shocks. Walk-forward classifications match the information set that drove actual investor behavior; look-ahead classifications mislabel days, weakening the empirical fit.

### Momentum-confound test
User asked: "is the shock signal just BTC momentum dressed up?" The OOS R² test with BTC prior 60d as a covariate:
- BTC prior 60d alone → +2.05pp R²
- + era × VIX-bin → +4.46pp (momentum + macro regime, joint)
- + shock × era × VIX-bin full interaction → +6.43pp (**+1.97pp incremental**)
- + momentum × shock interactions → +7.16pp (+0.73pp more)

**The +1.97pp incremental is the framework's true edge** beyond what BTC momentum + macro regime alone explain. ~60% of the original walk-forward +3.32pp ΔR² survives the momentum control.

### Trajectory test (causal framing)
User asked: "does the shock change BTC's trajectory, or just amplify what was already happening?" Test compares forward outcomes by `(shock_fire, BTC prior direction)` vs `(no_shock, BTC prior direction)`:
- **Banking + BTC down** = trajectory-reversing (Δ +7.8pp full sample, +17.3pp post-ETF; the SVB pattern)
- **Banking + BTC up** = trend-killing (Δ −16.6pp full sample)
- **Oil/dollar/rates + BTC up** = trend killers (Δ −11 to −17pp; neutral when BTC down)
- **GPR** = symmetric drag (Δ −5 to −6pp regardless of direction)

This taxonomy directly drives the 5-regime classification.

---

## 5-Regime taxonomy (deterministic decision tree)

Order matters in the tree below — first match wins:

1. **REVERSAL_PRIMED** — Banking shock firing AND BTC prior 60d ≤ −10%
   - Intuition: Banking-stress is the only documented reversal catalyst. Catalyzes BTC bounces from drawdowns. The SVB pattern.
   - Expected forward 60d: +5 to +10pp (bimodal — buy-the-catalyst, not buy-and-hold)
   - Class: bull

2. **TREND_KILL** — BTC prior 60d ≥ 0 AND any of {oil, dollar, rates, banking} firing
   - Intuition: Tightening shocks neutralize/reverse BTC uptrends (−11 to −17pp Δ).
   - Expected forward 60d: −10 to −15pp underperformance
   - Class: bear

3. **HIGH_BETA_BULL** — Calm VIX (<14.5) AND post-ETF era AND BTC prior 60d ≥ 0 AND no trend-killer firing
   - Intuition: Paper's signature finding — post-ETF calm-regime β rises +1.02. BTC trades as leveraged SPY with no headwinds.
   - Expected forward 60d: +8 to +14pp outperformance (highest-conviction long)
   - Class: bull

4. **DRAG** — GPR-threat firing alone (no other shock active)
   - Intuition: Symmetric −5pp drag. Uncertainty premium tax, not a regime-shifter.
   - Expected forward 60d: −3 to −7pp (neutral-to-mild underweight)
   - Class: bear

5. **BASELINE** — Everything else
   - Intuition: No actionable signal; default state.
   - Expected forward 60d: ±2pp
   - Class: neutral

Computed in `src/update_data.py::classify_regime()`.

---

## Repo layout (current)

```
bitcoin-shock/
├── HANDOFF.md                          ← this file
├── README.md                           ← public-facing intro
├── LICENSE                             ← MIT
├── .gitignore
├── requirements.txt                    ← Python deps for refresh script
├── index.html                          ← the dashboard (Plotly + vanilla JS)
├── .github/workflows/refresh.yml       ← daily 06:05 UTC cron (matches PRESS)
├── data/
│   ├── panel_with_shocks.parquet       ← VIX/BTC/SPY/macro/GPR daily NYSE panel (extends nightly)
│   ├── btc_calendar_daily.parquet      ← 24/7 BTC close series (CoinMetrics)
│   ├── era_conditional_walkforward_shocks_panel.parquet ← walk-forward shock flags + cutoffs
│   ├── ewma_shocks_panel.parquet       ← EWMA-126 comparator (kept for methodology)
│   ├── era_conditional_shocks_panel.parquet ← era-cond look-ahead comparator (kept for methodology)
│   └── dashboard_output.json           ← current-state JSON the page fetches
├── src/
│   ├── refresh_data.py                 ← daily refresh: FRED + CoinMetrics + yfinance + GPR
│   ├── build_era_conditional_shocks_walkforward.py ← THE walk-forward shock builder
│   ├── build_era_conditional_shocks.py ← era-cond comparator (kept)
│   ├── build_ewma_shocks.py            ← EWMA-126 comparator (kept)
│   ├── dashboard_lookup.py             ← builds 84-cell lookup table from the panel
│   ├── dashboard_forecast.py           ← CLI: forecast for a given date (standalone)
│   ├── update_data.py                  ← generates data/dashboard_output.json from panels + lookup
│   ├── empirical_blackrock_validation.py ← BlackRock 6-event historical case-study generator
│   ├── blackrock_horizon_sensitivity.py ← h=5/20/60/90 robustness on BlackRock events
│   ├── test_shock_adds_value.py        ← OOS R² test, full-sample basis (methodology comparator)
│   ├── test_shock_adds_value_ewma.py   ← EWMA basis (methodology comparator)
│   ├── test_shock_adds_value_era_cond.py ← era-cond basis (methodology comparator)
│   └── test_shock_adds_value_walkforward.py ← walk-forward basis (THE headline result)
└── output/
    ├── dashboard_lookup_table.json     ← 84-cell historical analogue table
    ├── shock_adds_value_test_walkforward.json ← headline OOS R² result + permutation null
    ├── shock_adds_value_test.json      ← full-sample comparator
    ├── shock_adds_value_test_ewma.json ← EWMA comparator
    ├── shock_adds_value_test_era_cond.json ← era-cond comparator
    ├── era_conditional_walkforward_shock_incidence.json ← per-era shock-fire rates
    ├── blackrock_6event_calendar_validation.json ← 6-event facts (regime + realized 60d)
    ├── blackrock_horizon_sensitivity.json
    └── docs/                           ← human-readable methodology write-ups
```

---

## Python script inventory (one-liner per script)

### Data refresh (production)
- **`src/refresh_data.py`** — daily pipeline orchestrator. Pulls FRED (VIX/WTI/USD-broad/10Y/STLFSI4 via CSV endpoints, no API key) + CoinMetrics community-api (BTC daily close) + yfinance (SPY) + Caldara-Iacoviello XLS (GPR-threat). Forward-fills lagged FRED series so |Δ|-based shocks don't NaN-out. Extends the panel parquet, re-runs build_walkforward + dashboard_lookup + update_data.

### Walk-forward shock construction
- **`src/build_era_conditional_shocks_walkforward.py`** — THE shock builder. For each shock at each date t: if within-era count ≥ 60 use within-era q90 (expanding); else use prior-era q99 (tail fallback). Pre-COVID era's first 60 days are NaN (no prior era).

### Dashboard data flow
- **`src/dashboard_lookup.py`** — builds the 84-cell historical analogue table. For each (era × VIX-bin × shock-type) cell, computes h=5/20/60/90 share-positive, mean outperformance, and n. Includes 2 fallback tiers (era-aggregated, fully-aggregated).
- **`src/update_data.py`** — generates `data/dashboard_output.json`. The most important script for the UI: computes BTC prior 60d, β₆₀, regime classification via `classify_regime()`, per-shock state (raw, cutoff, z-score, last-fired, 20d trend, 60d-prior trend), what-would-flip boundary deltas, BlackRock 6-event facts, 12-cell era×VIX aggregate, methodology comparison stats. Handles NaN→null coercion for browser JSON.parse compatibility.

### Methodology comparators (kept for transparency)
- **`src/build_era_conditional_shocks.py`** — within-era full-distribution cutoff (look-ahead variant, kept as comparator).
- **`src/build_ewma_shocks.py`** — EWMA-126 rolling cutoff (kept as comparator).
- **`src/test_shock_adds_value*.py`** — 4 variants of the formal OOS R² test (ridge + 5-fold CV + 200-permutation null), one per cutoff methodology. The walkforward version is the headline; the others document the convergence story.

### Validation
- **`src/empirical_blackrock_validation.py`** — runs the 6 BlackRock-narrative events (Iran 2020, COVID 2020, Election 2020, Russia-Ukraine 2022, SVB 2023, Tariff 2025) through the framework and records realized 60d outperformance.
- **`src/blackrock_horizon_sensitivity.py`** — h=5/20/60/90 sensitivity check on the 6-event sample.

### CLI utility
- **`src/dashboard_forecast.py`** — command-line forecast for an arbitrary historical date. Useful for ad-hoc queries: `python src/dashboard_forecast.py --date 2023-03-09`.

---

## Daily refresh pipeline

Cron schedule: **06:05 UTC daily** (matching PRESS model refresh slot).

GitHub Actions workflow (`.github/workflows/refresh.yml`):
1. Checkout repo
2. Set up Python 3.11
3. Install requirements.txt
4. Run `python src/refresh_data.py` — refreshes panel + walk-forward shocks + lookup + dashboard JSON
5. If any data files changed, commit with message `data: YYYY-MM-DD auto-refresh` and push

Two manual GitHub steps required ONCE after first push:
1. Settings → Actions → General → "Allow all actions and reusable workflows"
2. Settings → Actions → General → Workflow permissions → "Read and write permissions"

---

## Open items / what's next

### Production
- [ ] Push to GitHub: `cd /tmp/claude-501/dashboard_repo && git push`
- [ ] Enable Actions + write permissions (one-time, see above)
- [ ] Verify cron fires successfully (first run will be ~06:05 UTC tomorrow, or trigger manually via Actions tab "Run workflow")
- [ ] Verify live URL renders the new structure (header / intro / decision tree / actionability / mechanism / shock cards / what-would-flip / effect-size / β / worked examples / BlackRock / VIX / heatmap / methodology)

### Methodology — potential extensions
- [ ] Backtest equity curve: cumulative "trade the regime call" vs always-long BTC vs always-long SPY. Was deferred from the original 3-agent synthesis. Most credibility-relevant addition still on the table.
- [ ] Calibration table: of historical BULLISH calls, what % landed positive? Of BEARISH, what % landed negative? Quick to compute from the existing per-cell historical data.
- [ ] Cross-asset replication tabs (EEM, HYG, GLD). Code exists in original empirical_cross_asset.py — would need adaptation to the dashboard repo layout.
- [ ] Walk-forward Hansen τ recomputation (currently the dashboard uses pre-specified VIX bins at 14.5/20/30, not the paper's estimated τ ≈ 16.35). Would add a rolling-Hansen τ(t) display.

### Dashboard UX
- [ ] Glossary popover with `data-gloss` hover tooltips on 12 load-bearing terms (proposed by onboarding agent, not yet implemented).
- [ ] Guided-tour overlay (6-step coachmark sequence) — opt-in, low priority.
- [ ] Visitor-state toggle (first_visit / returning / expert) to dim/hide onboarding for repeat users.
- [ ] β₆₀ today value (1.93 as of last regen) looks anomalously high — sanity check whether the β computation in update_data.py is correctly aligned (BTC calendar vs SPY NYSE).
- [ ] Today's matched sub-cell (n=2) correctly reports NO_CALL but the aggregate cell (n=7) shows 40% sp+ as context — verify this dual-display reads cleanly to a practitioner.

### Data quality
- [ ] Caldara-Iacoviello GPR-threat is monthly-updated (sometimes weekly); the XLS endpoint occasionally 404s. Refresh script handles gracefully (forward-fills if unavailable). Consider a backup source if matteoiacoviello.com goes down for extended periods.
- [ ] STLFSI4 is weekly — forward-filled onto daily NYSE. Watch for any weekly-to-daily artifact in the banking shock cutoff during transition periods.

---

## How to resume (for a fresh Claude session)

1. **Read this HANDOFF.md and the README.md** — that's the bulk of context.
2. **Check git log** to see the commit history. Current head should be `6e1bf30 Accessibility rewrite`.
3. **Inspect dashboard locally** (sandbox network may block, but `python3 -m http.server 8765` from the repo root + browse to `localhost:8765/` works on the user's machine).
4. **For ad-hoc queries about the data**, the `data/dashboard_output.json` is the canonical current state and `output/shock_adds_value_test_walkforward.json` is the canonical R² result.
5. **For methodology questions**, the four `test_shock_adds_value*.py` scripts are the source-of-truth implementations.
6. **For UI changes**, edit `index.html` + the `JS` block at bottom. The data flows from `update_data.py` → `dashboard_output.json` → `fetch()` in the page.

### Commands cheat-sheet

```bash
cd /tmp/claude-501/dashboard_repo  # or wherever the repo lives

# Regenerate dashboard JSON from current panel
python src/update_data.py

# Full refresh: pull fresh data, rebuild everything
python src/refresh_data.py

# Preview locally
python3 -m http.server 8765
# → http://localhost:8765/

# Push to GitHub
git push
```

---

## Methodology references

- **Hansen (1996, 2000)** — rolling-window threshold regression used in Paper 1 to identify τ.
- **Caldara & Iacoviello (2022)** — "Measuring Geopolitical Risk". The GPR-threat index source.
- **Walk-forward identification** — standard time-series prediction methodology; no look-ahead in feature construction.
- **Q99 tail fallback** — bespoke construction for this framework, motivated by the user's observation that strict no-look-ahead with NaN warmup throws out economically important regime-transition events (e.g., COVID crash).

---

*This handoff was written 2026-05-19 after a single session that took the dashboard from a 2025-12-30 paper-derived snapshot to a thesis-first practitioner tool with auto-refresh, momentum-conditioned regime classification, and worked-example explanatory scaffolding.*
