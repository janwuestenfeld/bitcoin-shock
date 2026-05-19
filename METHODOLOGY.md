# METHODOLOGY

Technical reference for the walk-forward shock-cutoff methodology, the
84-cell regime decomposition, the 5-label regime taxonomy, the OOS R²
test framework, and the Python implementation. See `HANDOFF.md` for
project-level context and `README.md` for the public-facing summary.

---

## Data

### Sources

| Series | Source | Endpoint | Frequency |
|---|---|---|---|
| VIX | FRED VIXCLS | `fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS` | daily NYSE |
| WTI | FRED DCOILWTICO | `id=DCOILWTICO` | daily |
| USD broad | FRED DTWEXBGS | `id=DTWEXBGS` | daily |
| 10Y yield | FRED DGS10 | `id=DGS10` | daily |
| STLFSI4 | FRED STLFSI4 | `id=STLFSI4` | weekly |
| BTC daily close | CoinMetrics community-api | `community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics=PriceUSD&frequency=1d` | 24/7 daily |
| SPY daily close | yfinance | `yf.download('SPY', auto_adjust=False)['Close']` | daily NYSE |
| GPR-threat | Caldara-Iacoviello | `matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls` column `GPRD_THREAT` | daily |

### Panel layout

`data/panel_with_shocks.parquet` — daily NYSE-trading-day index, columns:
- `vix`, `wti`, `usd_broad`, `y10_fred`, `stlfsi`, `gprd_threat` (raw level data)
- `spy`, `btc` (price levels)
- Legacy columns from paper-era (Mkt_RF, SMB, HML, gld, uup, etc.) — kept for compatibility, not used by walk-forward methodology
- Full-sample shock flags (`oil_shock`, `dollar_shock`, ...) — paper-era basis, NOT used by current dashboard

`data/era_conditional_walkforward_shocks_panel.parquet` — same index, columns per shock:
- `{shock}_wf_raw` — raw measure (|Δlog WTI|, level for STLFSI4, etc.)
- `{shock}_wf_thresh` — walk-forward cutoff at that date
- `{shock}_wf` — binary indicator (1 if raw > thresh, 0 otherwise, NaN during pre-COVID warmup)

---

## Eras

```python
ERAS = [
    ("pre_covid",         "2014-01-02", "2020-02-29"),
    ("post_covid_pre_etf","2020-03-01", "2024-01-09"),
    ("post_etf",          "2024-01-10", "2099-12-31"),
]
```

- Post-COVID break: 2020-03-01 (WHO pandemic declaration / first FOMC emergency cut window)
- Post-ETF break: 2024-01-10 (spot BTC ETF approvals)

---

## VIX bins

Pre-specified, not estimated. Hansen-identified τ ≈ 16.35 from Paper 1 informs but doesn't drive the bin boundaries.

```python
VIX_BINS = [
    ("calm",          -inf,  14.5),
    ("low_stress",    14.5,  20.0),
    ("mid_stress",    20.0,  30.0),
    ("extreme_stress", 30.0,  inf),
]
```

---

## Shock definitions

Each shock is binary: 1 if today's raw measure exceeds the walk-forward q90 cutoff, 0 otherwise.

| Shock | Raw measure | Kind |
|---|---|---|
| `oil_shock` | `|log(wti_t / wti_{t-1})|` | unsigned, magnitude-based |
| `dollar_shock` | `|usd_broad_t − usd_broad_{t-1}|` | unsigned |
| `rate_shock` | `|y10_fred_t − y10_fred_{t-1}|` | unsigned |
| `banking_shock` | `stlfsi_t` (raw level) | signed |
| `gprd_threat_shock` | `gprd_threat_t` (raw level) | unsigned |

---

## Walk-forward + q99 tail fallback

The current spec for shock-cutoff construction. Implemented in `src/build_era_conditional_shocks_walkforward.py`.

For each shock series, for each date t:
1. Identify era_t.
2. Let n_in_era = count of observations in era_t with date < t (strict no look-ahead).
3. If n_in_era ≥ 60 (WARMUP_OBS):
   - cutoff = q90 of within-era observations through t−1
4. Else (warmup):
   - If a prior era exists:
     - cutoff = q99 of the prior era's closed-distribution observations
       (the prior era is fully known by era boundary, so its q99 is in F_t)
   - Else (pre-COVID, no prior era):
     - cutoff = NaN (skip classification)
5. shock_indicator[t] = 1 if raw[t] > cutoff else 0 (NaN if cutoff is NaN)

The prior-era q99 fallback catches obvious tail events (e.g., COVID crash, which exceeds pre-COVID q99 across oil/dollar/banking by wide margins) without introducing the prior-era-q90 mechanical bias that would over-classify normal early-era days as shocks.

---

## 84-cell decomposition

`src/dashboard_lookup.py` builds the cell table. For each `(shock, vix_bin, era)` cell:
- n = count of historical days matching the cell
- For each horizon h ∈ {5, 20, 60, 90} calendar days:
  - `r_btc_calendar_fwd`: BTC return from t to t+h (using 24/7 BTC calendar)
  - `r_spy_nyse_fwd`: SPY return from t to nearest NYSE date ≥ t+h
  - `outperf_calendar_fwd`: BTC fwd − SPY fwd
  - share_positive, mean, std, q05/25/50/75/95 of each distribution

Three lookup tiers (in priority order):
1. `cells_primary` — keyed `{shock}__{vix_bin}__{era}` (84 cells)
2. `cells_fallback_shock_era` — `{shock}__ALLVIX__{era}` (when primary has n<5)
3. `cells_fallback_shock` — `{shock}__ALLVIX__ALLERA` (last resort)

`shock` dimension: `none`, `any`, `oil_shock`, `dollar_shock`, `rate_shock`, `banking_shock`, `gprd_threat_shock` (7 levels). Combined with 4 VIX bins × 3 eras = 84 cells.

---

## OOS R² test framework

`src/test_shock_adds_value_walkforward.py` (and the three comparator variants).

Methodology:
- Outcome y = BTC vs SPY outperformance at h=60 calendar days
- Five nested models with increasing feature complexity:
  - M0: intercept only
  - M1: era dummies (3 levels − baseline)
  - M2: era × VIX-bin cells (12 levels − baseline)
  - M3: era × VIX-bin × shock-type cells (full interaction, up to 84 cells)
  - M4: era × VIX-bin cells + additive shock-type main effects (parsimonious version of M3)
- Estimator: Ridge regression (α = 1e-4) on standardized features
- Validation: 5-fold KFold(shuffle=True, random_state=42), OOS R² = 1 − SSR_oos / SST
- Significance: permutation null with B=200 shuffles of shock labels within era × VIX cells

Headline ΔR² M2→M3 (walk-forward basis, full panel n=2,818):
- **+3.32pp at p=0.005, KEEP_FULL_STRUCTURE verdict**

### Momentum-confound test

`/tmp/claude/test_momentum_vs_shock.py` (ad-hoc script, can be re-run):
- Adds BTC prior 60d return as an additional covariate
- Tests whether shock signal survives controlling for BTC momentum

Results:
| Model | k | R² OOS | Δ |
|---|---:|---:|---:|
| BTC prior 60d alone | 1 | +2.05pp | (baseline) |
| Era × VIX-bin alone | 11 | +3.20pp | (separate baseline) |
| Prior + era × VIX-bin (additive) | 12 | +4.46pp | joint |
| + shock × era × VIX-bin full interaction | 60 | +6.43pp | **+1.97pp** beyond prior + macro |
| + prior × shock interactions | 65 | +7.16pp | +0.73pp more |

**The +1.97pp is the framework's true edge** beyond what BTC momentum + macro regime alone predict.

### Trajectory / causal test

Per-shock comparison of `(shock_fire, prior_dir)` vs `(no_shock, prior_dir)` — measures the shock's marginal contribution beyond what would have happened anyway:

| Shock | Prior↑ Δ | Prior↓ Δ | Asymmetry | Kind |
|---|---:|---:|---:|---|
| Banking | −16.6pp | +7.8pp | 24.4pp | reversal (bidirectional) |
| Oil | −17.6pp | +0.1pp | 17.6pp | trend killer (one-sided) |
| Dollar | −11.6pp | −0.4pp | 11.2pp | trend killer |
| Rates | −11.1pp | −2.4pp | 8.7pp | trend killer |
| GPR-Threat | −5.0pp | −6.3pp | 1.3pp | symmetric drag |

This taxonomy directly drives the 5-regime classification (`classify_regime()` in `update_data.py`).

---

## 5-regime classifier (decision tree)

Implemented in `src/update_data.py::classify_regime(vix_bin, era, active_shocks, btc_prior_60d)`.

```python
def classify_regime(vix_bin, era, active_shocks, btc_prior_60d):
    trend_killers = {"oil_shock", "dollar_shock", "rate_shock", "banking_shock"}
    active = set(s for s, on in active_shocks.items() if on)
    bp = btc_prior_60d or 0.0

    # 1. Banking + BTC down ≥10% — reversal setup
    if "banking_shock" in active and bp <= -0.10:
        return REVERSAL_PRIMED  # +5 to +10pp, bull

    # 2. BTC up + any trend-killer — uptrend in danger
    if bp >= 0 and (active & trend_killers):
        return TREND_KILL  # -10 to -15pp, bear

    # 3. Calm + post-ETF + BTC up + no trend-killer — the paper's signature
    if vix_bin == "calm" and era == "post_etf" and bp >= 0 and not (active & trend_killers):
        return HIGH_BETA_BULL  # +8 to +14pp, bull

    # 4. GPR firing alone — drag
    if "gprd_threat_shock" in active and len(active - {"gprd_threat_shock"}) == 0:
        return DRAG  # -3 to -7pp, bear

    # 5. Everything else — no signal
    return BASELINE  # ±2pp, neutral
```

Order matters: REVERSAL_PRIMED checks first (highest-edge), then TREND_KILL, then HIGH_BETA_BULL, then DRAG, then BASELINE.

---

## BTC β₆₀ computation

In `src/update_data.py::compute_beta_60()`:

```python
btc = panel["btc"].astype(float)  # (or BTC calendar series as fallback)
spy = panel["spy"].astype(float)
rb = np.log(btc / btc.shift(1))   # BTC log returns
rs = np.log(spy / spy.shift(1))   # SPY log returns
cov = rb.rolling(60).cov(rs)      # rolling 60-day covariance
var = rs.rolling(60).var()        # rolling 60-day variance
beta = cov / var
```

Backward-rolling by construction — uses returns from t−60 to t, no look-ahead. Reported on the dashboard with a 5-trading-day delta.

---

## Look-ahead audit (every step certified)

| Component | Look-ahead-free? | Why |
|---|---|---|
| Shock cutoffs | ✓ | Walk-forward expanding within-era q90, prior-era q99 tail fallback during warmup |
| Shock classification | ✓ | Strict `raw[t] > cutoff[t]` where cutoff uses only data through t−1 |
| BTC β₆₀ | ✓ | Backward 60d rolling OLS |
| BTC prior 60d | ✓ | Backward 60d return |
| VIX bins | ✓ | Pre-specified constants (14.5, 20, 30) |
| Era boundaries | ✓ | Exogenous events (COVID, ETF launch) |
| Cell labels | ✓ | All inputs above are walk-forward |
| Forward outperformance | n/a | This is the outcome, computed only for the historical sample |
| OOS R² test | ✓ | 5-fold random CV — feature is the cell label (walk-forward), outcome is held out |
| Lookup table | acceptable | Uses end-of-era historical data, but this is a fixed reference table; live operator's "today's analogue" is computed against a snapshot table that won't shift |

---

## Reproducibility

```bash
# Full pipeline (panel + walk-forward + lookup + dashboard JSON)
python src/refresh_data.py

# Just the walk-forward shocks (uses existing panel)
python src/build_era_conditional_shocks_walkforward.py

# Just the lookup table
python src/dashboard_lookup.py

# Just regenerate the dashboard JSON
python src/update_data.py

# Re-run the OOS R² test
python src/test_shock_adds_value_walkforward.py
```

All scripts use `np.random.seed(42)` and `KFold(random_state=42)` where stochasticity is involved.

---

## Known limitations (honest disclosure)

1. **Policy regime shifts are exogenous to the framework.** The COVID 2020 case study shows TREND-KILL was a structurally bearish regime that delivered +18pp realized — because the Fed's $3tn balance-sheet expansion re-rated BTC as a liquidity asset. The framework is blind to central-bank reaction functions.

2. **At era boundaries, the first ~60 within-era days use prior-era q99 fallback.** This is operationally honest (an operator at era-start genuinely has no within-era cutoff yet) but means classifications during early-COVID and early-post-ETF are wider than the steady-state.

3. **Sample sizes vary widely by cell.** Most-populated cells have n≥100 (calm pre-COVID); least-populated have n=2-7 (rare shock combinations in short eras). The dashboard's `confidence` chip (HIGH/MEDIUM/LOW/NO_CALL) tracks this.

4. **The 84-cell decomposition is a snapshot.** Cells that grow over time will see their share-positive estimates drift; the dashboard's daily refresh updates these, but a true walk-forward backtest would require rebuilding the lookup table at every historical date.

5. **BTC prior 60d is correlated with several inputs.** The +1.97pp momentum-controlled R² is a partial answer, not a complete decomposition. A full causal-identification study would need to address residual correlations between shock-firing days and BTC's own price trajectory.

6. **GPR-threat is the noisiest input.** The Caldara-Iacoviello daily series is text-mining-based and has higher Q4-Q1 seasonal patterns. The framework's −5pp drag for GPR shocks is statistically real but the smallest of the five effects.

---

*Last updated 2026-05-19. Maintained alongside HANDOFF.md.*
