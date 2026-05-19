# BlackRock 6-event Window-Sensitivity (h ∈ {5, 20, 60, 90} calendar days)

Generated: 2026-05-19T10:19:38.504535Z. Convention: calendar-anchored (Calendar-anchored: BTC at calendar t -> BTC at calendar t+h (ffill). SPY at nearest NYSE <= t -> nearest NYSE >= t+h.).

## Per-event BTC-vs-SPY outperformance (percentage points)

| Event | Date | h=5 | h=20 | h=60 | h=90 | 60→90 swing |
|---|---|---:|---:|---:|---:|---:|
| US-Iran escalation | 2020-01-03 | +9.54 | +11.48 | +26.75 | +14.29 | -12.46 |
| COVID outbreak | 2020-03-09 | -22.30 | -21.19 | +17.59 | +4.89 | -12.70 |
| US election challenges | 2020-11-03 | +5.22 | +24.89 | +118.57 | +127.40 | +8.83 |
| Russia-Ukraine invasion | 2022-02-21 | +4.73 | +5.88 | +8.70 | -9.93 | -18.63 |
| US regional banking (SVB) | 2023-03-09 | +21.67 | +36.61 | +30.44 | +20.08 | -10.36 |
| US global tariff | 2025-04-02 | +6.86 | +19.85 | +23.10 | +18.14 | -4.96 |

## Per-event BTC forward return (percent)

| Event | h=5 | h=20 | h=60 | h=90 |
|---|---:|---:|---:|---:|
| US-Iran escalation | +10.18 | +14.37 | +19.87 | -7.14 |
| COVID outbreak | -34.84 | -25.22 | +24.86 | +23.44 |
| US election challenges | +10.74 | +31.27 | +128.78 | +139.84 |
| Russia-Ukraine invasion | +5.28 | +1.91 | +7.12 | -18.24 |
| US regional banking (SVB) | +21.72 | +39.51 | +36.25 | +29.43 |
| US global tariff | -3.80 | +13.25 | +28.10 | +27.87 |

## Per-event SPY forward return (percent)

| Event | h=5 | h=20 | h=60 | h=90 |
|---|---:|---:|---:|---:|
| US-Iran escalation | +0.63 | +2.89 | -6.88 | -21.43 |
| COVID outbreak | -12.54 | -4.03 | +7.27 | +18.55 |
| US election challenges | +5.51 | +6.38 | +10.22 | +12.44 |
| Russia-Ukraine invasion | +0.55 | -3.97 | -1.58 | -8.31 |
| US regional banking (SVB) | +0.04 | +2.89 | +5.81 | +9.35 |
| US global tariff | -10.65 | -6.60 | +4.99 | +9.73 |

## Window-sensitivity reading

BTC-vs-SPY outperformance near 60d is fragile for events whose 60→90d window catches a major BTC drawdown.
Concretely:

Events with |Δ outperf 60→90| ≥ 5pp:

- **US-Iran escalation** (2020-01-03): 60d outperf = +26.75pp; 90d outperf = +14.29pp; Δ = -12.46pp.
- **COVID outbreak** (2020-03-09): 60d outperf = +17.59pp; 90d outperf = +4.89pp; Δ = -12.70pp.
- **US election challenges** (2020-11-03): 60d outperf = +118.57pp; 90d outperf = +127.40pp; Δ = +8.83pp.
- **Russia-Ukraine invasion** (2022-02-21): 60d outperf = +8.70pp; 90d outperf = -9.93pp; Δ = -18.63pp (SIGN FLIP).
- **US regional banking (SVB)** (2023-03-09): 60d outperf = +30.44pp; 90d outperf = +20.08pp; Δ = -10.36pp.

### Russia-Ukraine spotlight

60d outperf = **+8.70pp** (BlackRock-style window). 90d outperf = **-9.93pp**. The 60→90 window catches the early May 2022 LUNA/Terra collapse plus the broader May-June 2022 BTC drawdown; SPY also fell in the same window but BTC fell harder.

**Operational reading:** the BlackRock chart's 60d outperformance for Russia-Ukraine is *real* (BTC did outperform SPY +8.7pp at calendar t+60d) but *window-fragile* -- the dashboard's h=90 column is the recommended sanity check.
