# Bitcoin Conditional Shock Regime Classifier

**A walk-forward, look-ahead-free regime classifier for BTC vs SPY relative performance.**

Given today's `(VIX, active macro-shock indicators, era)`, the classifier maps the regime into
one of 84 historical cells (3 eras × 4 VIX bins × 7 shock-types) and reports the cell's
historical share of 60-day BTC outperformance over SPY, with horizon distribution at h = 5 / 20 / 60 / 90
calendar days.

## Live site

[shock.wuestenfeld.eu](https://shock.wuestenfeld.eu) *(deployed from this repo)*

## Core idea

Bitcoin's beta with SPY is regime-conditional: it shifts with VIX threshold (Hansen 1996/2000),
spot-ETF launch (Jan 2024), and the type of macro shock active that day (oil / dollar / rate /
banking-stress / GPR-threat, all top-decile binary). The classifier identifies which historical
analogues today's regime most resembles, then reports their forward-60d BTC-vs-SPY outcome
distribution.

The classifier is a **probabilistic regime classifier, not a point forecaster.** A cell that's
71% share-positive at h=60 means 71% of historical analogues had BTC outperform SPY over the
following 60 calendar days — not that today specifically will.

## Methodology convergence (walk-forward + q99 fallback)

The shock-cutoff methodology iterated through four variants to converge on no-look-ahead identification:

| Cutoff method | M3 ΔR² OOS | Banking M4 | Verdict |
|---|---:|---:|---|
| Full-sample (paper) | +2.94pp | +0.52pp | KEEP (inflated by cross-era mixing) |
| EWMA-126 rolling | +0.94pp | −1.82pp | BORDERLINE (smears extreme + non-extreme within window) |
| Era-conditional (look-ahead) | +0.76pp | +0.17pp | BORDERLINE (under-states signal via wrong-time-frame classification) |
| **Walk-forward + q99 tail fallback** | **+3.32pp (p=0.005)** | **−1.30pp** | **KEEP_FULL_STRUCTURE** |

The walk-forward result is stronger than the look-ahead-permitting variants because shock labels
under walk-forward are aligned with the information set contemporaneous investors actually had:
future observations cannot influence past investor behavior, so they should not influence the
classifier's shock identifications either.

**Banking-shock M4 coefficient flips sign** between COVID-excluded (+1.76pp) and COVID-included
(−1.30pp) — revealing the banking safe-haven channel is **post-ETF cell-conditional, not unconditional**.

## What this is and isn't

**This is:**
- A regime classifier with a defensible cell-conditional probability framework
- A walk-forward, no-look-ahead identification — operator-implementable in real time
- An honest representation of the dashboard's edge: +3.32pp ΔR² OOS over an era × VIX-bin baseline

**This isn't:**
- A point-price forecaster
- A same-day β predictor (β is a 60-day rolling object)
- Effective at era boundaries (~60-day warmup; classifier correctly stays silent during regime transitions)

## Repo layout

```
.
├── index.html                              # Live dashboard (PRESS-styled, Plotly)
├── data/
│   ├── dashboard_output.json               # Current-state JSON the page fetches
│   ├── panel_with_shocks.parquet           # Input data (VIX/BTC/SPY/oil/dollar/rates/STLFSI/GPR)
│   └── btc_calendar_daily.parquet          # 24/7 BTC close (CoinMetrics)
├── src/
│   ├── update_data.py                      # Daily refresh: panel → dashboard_output.json
│   ├── build_era_conditional_shocks_walkforward.py  # Walk-forward + q99 tail fallback
│   ├── build_era_conditional_shocks.py     # Era-conditional (look-ahead, comparator)
│   ├── build_ewma_shocks.py                # EWMA-126 (comparator)
│   ├── dashboard_lookup.py                 # Build 84-cell lookup table
│   ├── dashboard_forecast.py               # CLI: forecast for a given date
│   ├── test_shock_adds_value.py            # Formal OOS R² test, full-sample basis
│   ├── test_shock_adds_value_ewma.py       # EWMA basis
│   ├── test_shock_adds_value_era_cond.py   # Era-cond basis
│   ├── test_shock_adds_value_walkforward.py # Walk-forward basis (the headline)
│   ├── empirical_blackrock_validation.py   # 6-event validation
│   └── blackrock_horizon_sensitivity.py    # h=5/20/60/90 robustness
└── output/
    ├── dashboard_lookup_table.json
    ├── shock_adds_value_test_walkforward.json   # Headline result
    ├── shock_adds_value_test.json               # Full-sample comparator
    ├── shock_adds_value_test_ewma.json          # EWMA comparator
    ├── shock_adds_value_test_era_cond.json      # Era-cond comparator
    ├── era_conditional_walkforward_shock_incidence.json
    ├── blackrock_6event_calendar_validation.json
    ├── blackrock_horizon_sensitivity.json
    └── docs/
        ├── dashboard_user_guide.md
        ├── post_covid_restriction.md
        ├── blackrock_6event_calendar_validation.md
        ├── blackrock_horizon_sensitivity.md
        └── shock_adds_value_test_walkforward.md
```

## Updating the dashboard

```bash
# 1. Refresh panel data (FRED / CoinMetrics / yfinance / Caldara-Iacoviello)
#    [Set up by deploying infrastructure; not included in this repo]

# 2. Regenerate dashboard_output.json from the latest panel
python src/update_data.py

# 3. Commit & push to deploy
git add data/dashboard_output.json data/panel_with_shocks.parquet
git commit -m "data: $(date +%Y-%m-%d) refresh"
git push
```

## Reproducibility

All five shock indicators are derivable from public sources:
- VIX, WTI, USD-broad, 10Y, STLFSI4 from [FRED](https://fred.stlouisfed.org/)
- BTC daily close from [CoinMetrics community API](https://docs.coinmetrics.io/api/v4/)
- SPY daily close from yfinance
- GPR-threat from [Caldara-Iacoviello](https://www.matteoiacoviello.com/gpr.htm)

The methodology iteration (build & test scripts in `src/`) reproduces the four-variant comparison.

## Citation

If you use this classifier:

> Wuestenfeld, J. (2026). *Bitcoin Conditional Shock Regime Classifier*.
> https://github.com/janwuestenfeld/bitcoin-shock

## License

MIT
