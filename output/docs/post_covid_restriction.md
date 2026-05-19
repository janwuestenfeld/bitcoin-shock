# Post-COVID Restriction — Robustness Check

**Status:** Post-pipeline diagnostic. NOT incorporated into the paper. The user will decide whether to integrate.

## Motivation and design

The paper's headline Block 1 verdict (`Δβ_calm = +1.02` at the 2024-01-10 ETF break) and Block 2 verdicts (`σ_β² ≈ 0`; common `Δβ_universal ≈ +0.79`; shock-specific `τ_k`) are estimated on the full 2014-03-31 — 2025-12-30 panel (n = 2,957 daily obs after dropping NA on `beta_60` and `vix`). The pre-2020 era is structurally different from the post-COVID era: BTC's 60-day rolling β-on-SPY runs near zero before COVID and rises persistently positive afterward. The full-panel design therefore averages over two heterogeneous regimes, which mechanically inflates the post-ETF `Δβ_calm` by anchoring the pre-period to a near-zero-β baseline that pre-dates BTC's behavioral integration into risk-asset trading.

**Restriction.** Re-run the headline tests on `date >= 2020-03-01` only (n = 1,467; post-COVID era of positive equity correlation; common regime). Compare to the full-panel headline numbers.

**Test settings.** Block 1 bootstrap CIs at B=500 (Hansen) and B=500 (Chow). Block 2 hierarchical Bayes on a reduced budget: STLFSI4 banking proxy + Prior A only, 6000 warmup × 4 chains × 1500 stored samples (thin = 4). Seed 20260519. All else identical to paper headline.

---

## 1. Block 1 Chow at 2024-01-10 — BTC

### Full-sample Hansen (within each panel)

| Quantity | Full panel | Post-COVID |
|---|---|---|
| n | 2,957 | 1,467 |
| date range | 2014-03-31 to 2025-12-30 | 2020-03-02 to 2025-12-30 |
| τ̂ (95% CI) | 16.35 [14.70, 19.00] | 14.54 [13.88, 25.74] |
| β_calm | 0.297 | 0.789 |
| β_stress | 0.955 | 1.260 |
| Δβ = β_stress − β_calm (95% CI) | 0.658 [0.410, 0.916] | 0.471 [−0.250, 0.873] |

### Chow break at 2024-01-10

| Quantity | Full panel | Post-COVID |
|---|---|---|
| n_pre / n_post | 2,462 / 495 | 972 / 495 |
| τ_pre | 16.32 | 14.91 |
| τ_post | 14.61 | 14.61 |
| β_calm_pre | 0.109 | 0.314 |
| β_calm_post | 1.131 | 1.131 |
| **Δβ_calm (95% CI)** | **+1.022 [+0.237, +1.674]** | **+0.817 [−0.201, +1.400]** |
| β_stress_pre | 0.874 | 1.231 |
| β_stress_post | 1.354 | 1.354 |
| Δβ_stress (95% CI) | +0.480 [−0.216, +0.890] | +0.123 [−0.480, +0.493] |
| Δτ (95% CI) | −1.71 [−5.88, +7.50] | −0.30 [−7.08, +8.13] |

### Robustness: Chow at fixed Paper-1 anchor τ = 19.47

| Quantity | Post-COVID (anchor τ = 19.47) |
|---|---|
| pre β_calm | 0.959 (n=365) |
| post β_calm | 1.298 (n=396) |
| **Δβ_calm @ τ=19.47 (95% CI)** | **+0.340 [−0.050, +0.769]** |
| pre β_stress | 1.246 (n=607) |
| post β_stress | 1.288 (n=99) |
| Δβ_stress @ τ=19.47 (95% CI) | +0.042 [−0.355, +0.525] |

### Read

**Verdict materially weakens — the post-ETF calm-state shift loses statistical significance on the post-COVID baseline.**

- The headline `Δβ_calm = +1.02 [+0.24, +1.67]` shrinks to `+0.82 [−0.20, +1.40]`; the 95% CI now **includes zero**.
- The robustness Chow at the fixed Paper-1 anchor `τ = 19.47` shrinks even more: `+0.34 [−0.05, +0.77]` — CI also includes zero.
- The shrinkage is driven by the pre-ETF baseline: under the full panel, `β_calm_pre = +0.11` (which averages 2014–2024 calm days, including the near-zero-β 2014–2019 period); under post-COVID, `β_calm_pre = +0.31`. The post-ETF level is unchanged (`+1.13`) because that period is the same in both panels.
- Conclusion: a non-trivial share of the headline `+1.02` magnitude is mechanical — it reflects the gap between a contaminated 2014–2024 baseline and the post-ETF level, not a structural ETF-driven calm-state shift estimated on the same regime. The point estimate post-restriction (+0.82, or +0.34 at the anchor) is still positive and economically interesting, but the inferential strength of the headline claim drops sharply.

---

## 2. Block 2 Hierarchical Bayes — STLFSI4 + Prior A

### Headline posteriors

| Quantity | Full panel (STLFSI4_A) | Post-COVID (STLFSI4_A) | Change |
|---|---|---|---|
| τ_universal mean (sd) | 18.24 (1.38) | 14.55 (0.38) | −3.7 VIX pts |
| Δβ_universal mean (sd) | +0.792 (0.051) | +0.608 (0.103) | −23% |
| σ_τ² posterior mean | **10.60** | **0.46** | **−96%** |
| σ_β² posterior mean | **0.0096** | **0.0449** | **+367%** |
| **P(σ_β² > 0.1)** | **0.010** | **0.112** | **+11×** |
| **P(σ_τ² < 1)** | **0.000** | **0.889** | **flipped** |
| max R-hat (primary params) | (sampler v3 PASS) | 1.081 (≤ 1.10 ✓; > 1.05) | borderline |

### Per-shock τ_k and Δβ_k posterior means

| Shock k | τ_k full | τ_k post-COVID | Δβ_k full | Δβ_k post-COVID |
|---|---|---|---|---|
| oil        | 17.94 | 14.73 | +0.799 | +0.734 |
| dollar     | 14.92 | 14.34 | +0.806 | +0.551 |
| rate       | 17.86 | 14.64 | +0.747 | +0.583 |
| banking    | 23.88 | 14.54 | +0.800 | +0.597 |
| gpr_threat | 16.16 | 14.45 | +0.813 | +0.515 |

### Per-shock independent Hansen on post-COVID panel (no shrinkage)

| Shock k | n | τ̂ | Δβ |
|---|---|---|---|
| oil        | 209 | 16.12 | +0.74 |
| dollar     | 186 | 14.94 | +0.51 |
| rate       | 276 | 17.88 | +0.48 |
| gpr_threat | 215 | 20.31 | +0.38 |
| banking    | 139 | 29.57 | **−0.33** (sign-flips) |

### Read

**Both Q2 verdicts reverse on the post-COVID restriction.**

1. **`σ_β² ≈ 0` verdict reverses materially.** P(σ_β² > 0.1) jumps from 0.010 (full panel) to **0.112** (post-COVID) — an 11× increase that puts mass in the "heterogeneous response magnitudes" region. σ_β² posterior mean increases by 367% (0.0096 → 0.0449). The per-shock Δβ_k means visibly spread on the post-COVID panel (range +0.515 to +0.734, vs. +0.747 to +0.813 on the full panel), and Δβ_universal sd doubles (0.051 → 0.103). The "common response magnitude across shocks" claim is partly an artifact of pre-COVID heterogeneity averaging out — once that regime is removed, the per-shock Δβ_k start to look genuinely different.

2. **`shock-specific τ_k` verdict reverses.** P(σ_τ² < 1) goes from 0.000 (full panel: τ_k clearly different across shocks) to **0.889** (post-COVID: τ_k cluster tightly at ~14.5). σ_τ² posterior mean collapses from 10.60 → 0.46. On the post-COVID panel, the hierarchical model now finds τ_k essentially common, not shock-specific.

3. **Banking shock sign-flips on independent Hansen.** Post-COVID independent Hansen on banking-shock days gives Δβ = **−0.33** (vs. +0.80 on the full panel) — i.e. on banking-stress days post-COVID, BTC's β is *lower* in the stress regime than in the calm regime. The banking τ̂ also climbs from 23.88 to 29.57. The full-panel banking pattern (large Δβ at high τ) does not replicate on post-COVID data.

4. **τ_universal drifts down by ~3.7 VIX pts** (18.24 → 14.55), consistent with the post-COVID era having a structurally lower threshold than the pooled estimate.

5. **R-hat note.** The post-COVID fit reaches max R-hat = 1.081 on a reduced sampler budget (6000+1500). It passes the 1.10 conventional threshold but exceeds the paper's headline 1.05 target — worth flagging if these results are integrated.

The structural reading: under the post-COVID regime, the data look more like "all shocks trigger a *common* threshold (~14.5 VIX) with *somewhat heterogeneous* Δβ magnitudes (range +0.4 to +0.7, with banking actually negative)" — the opposite of the paper's headline characterization.

---

## 3. Cross-asset Block 1 Chow at 2024-01-10 — post-COVID restriction

### Full panel (paper headline) vs post-COVID restriction

| Asset | Δβ_calm full panel (95% CI) | Δβ_calm post-COVID (95% CI) |
|---|---|---|
| BTC | **+1.022** [+0.237, +1.674] | **+0.817** [−0.201, +1.400] |
| EEM | −0.261 [−0.469, −0.165] | −0.082 [−0.236, +0.082] |
| HYG | −0.028 [−0.115, +0.022] | −0.029 [−0.103, +0.028] |
| GLD | +0.491 [+0.154, +0.685] | +0.345 [−0.195, +0.488] |

### β_calm pre vs post (post-COVID restriction)

| Asset | β_calm_pre | β_calm_post | τ_pre | τ_post |
|---|---|---|---|---|
| BTC | +0.314 | +1.131 | 14.91 | 14.61 |
| EEM | +0.911 | +0.829 | 14.91 | 13.33 |
| HYG | +0.336 | +0.307 | 23.36 | 14.08 |
| GLD | −0.044 | +0.300 | 14.91 | 14.08 |

### Cross-asset full-sample Hansen on post-COVID panel

| Asset | τ̂ (95% CI) | Δβ = β_stress − β_calm (95% CI) | n |
|---|---|---|---|
| BTC | 14.54 [13.88, 25.74] | +0.471 [−0.250, +0.873] | 1,467 |
| EEM | 14.54 [13.51, 32.00] | −0.085 [−0.159, +0.098] | 1,467 |
| HYG | 22.14 [14.46, 28.91] | +0.064 [−0.056, +0.116] | 1,467 |
| GLD | 17.22 [13.40, 29.71] | +0.038 [−0.183, +0.180] | 1,467 |

### Read

**BTC-specificity of the calm-state shift survives in the point estimate, but with weaker inference.**

- On the post-COVID panel, BTC's `Δβ_calm = +0.82` remains noticeably larger in magnitude than EEM (−0.08), HYG (−0.03), and GLD (+0.34). The sign and ranking are preserved: BTC is the only asset with a positive, economically large calm-state shift around 2024-01-10.
- The other assets also shrink toward zero on the post-COVID restriction:
  - **EEM**: −0.26 [−0.47, −0.17] → −0.08 [−0.24, +0.08] (CI now includes zero)
  - **GLD**: +0.49 [+0.15, +0.69] → +0.35 [−0.20, +0.49] (CI now includes zero)
  - **HYG**: essentially unchanged (~−0.03)
- So the "BTC stands alone" headline survives, but **none of the four assets has a Δβ_calm 95% CI that excludes zero on the post-COVID panel.** The paper's cross-asset narrative is correct in *rank* (BTC > GLD > HYG > EEM) but the underlying significance of the comparisons is materially weaker than the full-panel table suggests.
- The pre-ETF baseline gap for BTC (β_calm_pre = +0.31 post-COVID vs. +0.11 full panel) accounts for the +0.20 shrinkage; for the other assets, the baselines barely move because their β-on-SPY profiles are already stable across 2014-2024.

---

## 4. Overall honest read

Of 5 headline verdicts checked, **2/5 survive** the post-COVID restriction:

| Verdict | Full-panel value | Post-COVID value | Survives? |
|---|---|---|---|
| Q1: Δβ_calm > 0 with CI excluding zero | +1.02 [+0.24, +1.67] | +0.82 [−0.20, +1.40] | **✗ — CI now includes zero** |
| Q1: Δβ_calm magnitude ≥ +0.5 | +1.02 | +0.82 | ✓ point estimate still > 0.5 |
| Q2: σ_β² ≈ 0 (P(σ_β² > 0.1) ≤ 0.10) | P = 0.010 | P = 0.112 | **✗ — verdict reverses** |
| Q2: σ_τ² > 1 (shock-specific τ_k) | P(σ_τ² < 1) = 0.000 | P(σ_τ² < 1) = 0.889 | **✗ — verdict flips** |
| Cross-asset BTC-specificity (BTC ranks first) | BTC +1.02, others ≤ +0.49 | BTC +0.82, others ≤ +0.35 | ✓ ranking preserved |

**Diagnostic interpretation.** The pre-2020 period is a different regime for BTC — near-zero equity β, pre-institutionalization, smaller market cap, distinct investor base. Restricting to post-COVID removes that heterogeneity from the baseline and tests whether the paper's headline findings are structural to the current regime or are partly mechanical artifacts of averaging over two regimes. The above table is the answer.

**Where this leaves the paper, in one paragraph.** Three of the paper's four prominent claims (the ETF Δβ_calm significance with full-panel CI excluding zero; the σ_β² ≈ 0 "common-magnitude" verdict; and the σ_τ² > 0 "shock-specific-threshold" verdict) are partly artifacts of averaging across the pre- and post-2020 regimes. Two findings survive intact: the post-ETF calm-state shift remains economically sized (~+0.34 to +0.82 depending on specification) and BTC remains the asset where this shift is largest across the four-asset comparison. The paper as currently framed overstates the inferential strength of the headline. A more defensible framing would either (a) report post-COVID as the headline panel and full-panel as a sample-sensitivity robustness check (the opposite of the current structure), or (b) keep the full-panel headline but explicitly partition the contribution between "pre-2020 baseline divergence" and "post-2020 ETF-driven shift" and re-cast the σ_β²/σ_τ² verdicts as full-panel-only.

---

## Code and reproducibility

- Driver: `code/empirical_post_covid_check.py`
- Raw results: `output/stage3a/results/post_covid_restriction.json`
- Reused infrastructure: `code/emp_core.py`, `code/emp_block1.py`, `code/emp_block2.py`, `code/empirical_cross_asset.py`
- Seed: `20260519`. Pre-COVID cutoff: `2020-03-01`. ETF break: `2024-01-10`. Paper-1 anchor τ: `19.47`.
- Block 1 bootstrap: B=500. Block 2 MCMC: 4 chains × 6,000 warmup × 1,500 stored samples (thin=4). Max R-hat = 1.081 (passes 1.10; exceeds the paper's headline 1.05 target).
- Run log: `code/run_log_post_covid.txt`.
