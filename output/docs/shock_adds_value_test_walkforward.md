# Does Shock-Type Add OOS R² Beyond Era × VIX-bin? — EWMA-Basis Re-Run

Generated: 2026-05-19T14:36:58.583550Z  |  Seed: 42  |  CV folds: 5  |  Permutations: 200

## What changed vs the original run

- Shock flags substituted from `data/aux/ewma_shocks_panel.parquet` (rolling EWMA-126 cutoff) in place of the full-sample shock columns in `panel_with_shocks.parquet`.
- All other design choices identical: same M0–M4 design, 5-fold KFold(shuffle=True, random_state=42), Ridge α=1e-4, permutation shuffles RETAINED_SHOCKS jointly within era×VIX cells (B=200).
- Full panel n=2818; shock-active sub-panel n=1136 (EWMA basis has fewer shock-active days than the full-sample basis).

## R² results — full-sample vs EWMA basis (side-by-side)

| Panel | Model | k (full-sample) | R² OOS, full-sample | k (EWMA) | R² OOS, EWMA |
|---|---|---:|---:|---:|---:|
| Full | M0 intercept-only | 0 | -0.0002 | 0 | -0.0004 |
| Full | M1 era only | 2 | 0.0024 | 2 | 0.0047 |
| Full | M2 era×VIX (12 cells) | 11 | 0.0335 | 11 | 0.0323 |
| Full | M3 era×VIX×shock-type | 75 | 0.0629 | 72 | 0.0654 |
| Full | M4 era×VIX + shock main eff. | 17 | 0.0502 | 17 | 0.0461 |
| Shock-active | M0 intercept-only | 0 | -0.0020 | 0 | -0.0021 |
| Shock-active | M1 era only | 2 | -0.0042 | 2 | -0.0018 |
| Shock-active | M2 era×VIX (12 cells) | 11 | -0.0025 | 11 | -0.0011 |
| Shock-active | M3 era×VIX×shock-type | 63 | -0.0191 | 61 | 0.0376 |
| Shock-active | M4 era×VIX + shock main eff. | 16 | 0.0092 | 16 | -0.0034 |

## EWMA basis — detailed R² (full and shock-active panels)

| Panel | Model | k | R² IS | R² OOS (5-fold) | Sign-acc OOS |
|---|---|---:|---:|---:|---:|
| Full panel | M0 intercept-only | 0 | 0.0000 | -0.0004 | 0.544 |
| Full panel | M1 era only | 2 | 0.0065 | 0.0047 | 0.544 |
| Full panel | M2 era×VIX (12 cells) | 11 | 0.0373 | 0.0323 | 0.549 |
| Full panel | M3 era×VIX×shock-type | 72 | 0.0931 | 0.0654 | 0.609 |
| Full panel | M4 era×VIX + shock main eff. | 17 | 0.0527 | 0.0461 | 0.551 |
| Shock-active | M0 intercept-only | 0 | 0.0000 | -0.0021 | 0.509 |
| Shock-active | M1 era only | 2 | 0.0025 | -0.0018 | 0.509 |
| Shock-active | M2 era×VIX (12 cells) | 11 | 0.0165 | -0.0011 | 0.511 |
| Shock-active | M3 era×VIX×shock-type | 61 | 0.1148 | 0.0376 | 0.590 |
| Shock-active | M4 era×VIX + shock main eff. | 16 | 0.0276 | -0.0034 | 0.527 |

## ΔR² (vs M2 baseline) with permutation p-value — EWMA basis

| Panel | Comparison | Observed ΔR² (pp) | Null mean (pp) | Null 95th pctl (pp) | p (one-sided > ) |
|---|---|---:|---:|---:|---:|
| Full panel | M2 → M3 (full interaction) | +3.32 | -1.96 | -0.98 | 0.005 |
| Full panel | M2 → M4 (additive shock) | +1.38 | -0.24 | +0.07 | 0.005 |
| Shock-active | M2 → M3 (full interaction) | +3.87 | -4.53 | -2.55 | 0.005 |
| Shock-active | M2 → M4 (additive shock) | -0.22 | -0.62 | +0.07 | 0.194 |

## ΔR² head-to-head: full-sample basis vs EWMA basis

| Panel | Comparison | ΔR² (pp), full-sample | p, full-sample | ΔR² (pp), EWMA | p, EWMA |
|---|---|---:|---:|---:|---:|
| Full panel | M2 → M3 | +2.94 | 0.005 | +3.32 | 0.005 |
| Full panel | M2 → M4 | +1.67 | 0.005 | +1.38 | 0.005 |
| Shock-active | M2 → M3 | -1.66 | 0.010 | +3.87 | 0.005 |
| Shock-active | M2 → M4 | +1.17 | 0.005 | -0.22 | 0.194 |

## Per-shock loadings (M4, additive) — EWMA basis

### Full panel
M4 ridge coefficients on shock-type dummies (vs the dropped baseline shock-type):

| Shock dummy | Coef (decimal) | Coef (pp) |
|---|---:|---:|
| shock_banking_shock | -0.0130 | -1.30 |
| shock_gprd_threat_shock | -0.0687 | -6.87 |
| shock_rate_shock | -0.0867 | -8.67 |
| shock_dollar_shock | -0.0908 | -9.08 |
| shock_multi | -0.1128 | -11.28 |
| shock_oil_shock | -0.1227 | -12.27 |

Raw 60d outperformance by shock-type (unconditional, EWMA basis):

| Shock-type | n | Mean outperf (pp) | Std (pp) |
|---|---:|---:|---:|
| none | 1682 | +14.14 | 39.20 |
| multi | 339 | +1.68 | 22.09 |
| gprd_threat_shock | 293 | +7.86 | 34.42 |
| banking_shock | 149 | +9.22 | 26.26 |
| dollar_shock | 145 | +5.26 | 30.09 |
| rate_shock | 131 | +5.37 | 32.06 |
| oil_shock | 79 | +2.26 | 24.04 |

### Shock-active
M4 ridge coefficients on shock-type dummies (vs the dropped baseline shock-type):

| Shock dummy | Coef (decimal) | Coef (pp) |
|---|---:|---:|
| shock_banking_shock | +0.0770 | +7.70 |
| shock_gprd_threat_shock | +0.0640 | +6.40 |
| shock_rate_shock | +0.0440 | +4.40 |
| shock_dollar_shock | +0.0368 | +3.68 |
| shock_multi | -0.0012 | -0.12 |

Raw 60d outperformance by shock-type (unconditional, EWMA basis):

| Shock-type | n | Mean outperf (pp) | Std (pp) |
|---|---:|---:|---:|
| multi | 339 | +1.68 | 22.09 |
| gprd_threat_shock | 293 | +7.86 | 34.42 |
| banking_shock | 149 | +9.22 | 26.26 |
| dollar_shock | 145 | +5.26 | 30.09 |
| rate_shock | 131 | +5.37 | 32.06 |
| oil_shock | 79 | +2.26 | 24.04 |

## Banking-shock check (was the only positive loading in the original run)

- **Full panel**: 0 shock dummies have positive M4 coefficients (vs baseline), 6 negative.
  - `shock_banking_shock` coef = -1.30pp. Uniquely positive among shock dummies? **False**.
- **Shock-active**: 4 shock dummies have positive M4 coefficients (vs baseline), 1 negative.
  - `shock_banking_shock` coef = +7.70pp. Uniquely positive among shock dummies? **False**.

## Verdict

**KEEP_FULL_STRUCTURE** — Full-panel DR^2 (M2 -> M3) = +3.32pp at p=0.005 (< 0.10): shock-type adds operational signal beyond era x VIX-bin under the EWMA basis. Previous full-sample signal was real, not a cutoff artifact. Dashboard should ship on the EWMA basis.

Shock-active panel context: ΔR² (M2→M3) = +3.87pp, p = 0.005.

## Honest assessment — EWMA-cleaned signal vs full-sample signal

- Full-panel ΔR² (M2→M3): full-sample basis = +2.94pp (p=0.005); EWMA basis = +3.32pp (p=0.005).
- **Finding**: The EWMA basis confirms the shock-type signal is real (not a cutoff artifact). The dashboard should ship on the EWMA basis with the same M3 full-interaction structure.

- The permutation null is centred slightly negative (shuffled labels cost OOS R² because they add design-matrix noise without signal); observed deltas should be evaluated against the null distribution rather than vs zero.
- If `shock_banking_shock` loses its unique positive loading under EWMA, the safe-haven channel that previously appeared singular may have been cutoff-defined rather than structurally identified. Surface this as a finding in its own right.
