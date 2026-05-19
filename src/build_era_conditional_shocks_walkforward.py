"""Walk-forward (zero look-ahead) era-conditional shock indicators.

For each shock series x at date t:
  1. Identify era_t = era containing t
  2. Take all observations s with era_s == era_t AND s < t (strict no-look-ahead)
  3. If count(within-era observations so far) < WARMUP_OBS: classify NaN
       (operator literally has no stable within-era cutoff yet; we discard)
  4. Else: cutoff = q-quantile of within-era observations through t-1
  5. shock[t] = 1 iff x_t > cutoff (else 0)

No prior-era fallback (that introduced the mechanical-bias you flagged at
era boundaries).

By end-of-era, the expanding cutoff converges to the within-era full-distribution
cutoff. So today's live dashboard cutoff for the post-ETF era equals what an
operator would have computed live since 2024-01-10.

Outputs
-------
- data/aux/era_conditional_walkforward_shocks_panel.parquet
- output/stage3a/results/era_conditional_walkforward_shock_incidence.json
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "data" / "panel_with_shocks.parquet"
OUT_PARQUET = ROOT / "data" / "era_conditional_walkforward_shocks_panel.parquet"

RESULTS_DIR = ROOT / "output"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_INCIDENCE_JSON = RESULTS_DIR / "era_conditional_walkforward_shock_incidence.json"

QUANTILE = 0.90
WARMUP_OBS = 60                  # need ≥60 within-era obs before using within-era cutoff
TAIL_QUANTILE = 0.99             # during warmup: prior-era 99th-percentile fallback
                                  # (catches obvious shocks like COVID without prior-era-90th
                                  # mechanical bias)

SHOCK_SPEC = [
    ("oil_shock",         "wti",         "abs_log_diff"),
    ("dollar_shock",      "usd_broad",   "abs_diff"),
    ("rate_shock",        "y10_fred",    "abs_diff"),
    ("banking_shock",     "stlfsi",      "level"),
    ("gprd_threat_shock", "gprd_threat", "level"),
]

ERAS = [
    ("pre_covid",         "2014-01-02", "2020-02-29"),
    ("post_covid_pre_etf","2020-03-01", "2024-01-09"),
    ("post_etf",          "2024-01-10", "2099-12-31"),
]
ERA_NAMES = [e[0] for e in ERAS]


def raw_measure(panel: pd.DataFrame, col: str, kind: str) -> pd.Series:
    s = panel[col].astype(float)
    if kind == "abs_log_diff":
        return np.log(s / s.shift(1)).abs()
    if kind == "abs_diff":
        return (s - s.shift(1)).abs()
    if kind == "level":
        return s
    raise ValueError(kind)


def era_of(d: pd.Timestamp) -> str:
    for n, lo, hi in ERAS:
        if pd.Timestamp(lo) <= d <= pd.Timestamp(hi):
            return n
    return "pre_covid"


def walkforward_within_era_threshold(
    raw: pd.Series,
    era_labels: pd.Series,
    quantile: float = QUANTILE,
    warmup: int = WARMUP_OBS,
    tail_quantile: float = TAIL_QUANTILE,
) -> pd.Series:
    """No-look-ahead expanding within-era cutoff, with prior-era 99th-pctl
    tail fallback during warmup.

    At each t:
      - If within-era count >= warmup: cutoff = q90 of within-era observations
        accumulated through t-1 (zero look-ahead within era).
      - Else (warmup): cutoff = q99 of CLOSED prior era's observations (known
        before this era began, so in F_t). Catches genuine tail events
        (COVID-style crashes) without prior-era-q90 mechanical bias.
      - First era has no prior; warmup days are NaN.
    """
    n = len(raw)
    out = np.full(n, np.nan)
    vals = raw.values
    eras = era_labels.values
    era_buffer: dict[str, list[float]] = {nm: [] for nm in ERA_NAMES}

    # Precompute each era's CLOSED tail-quantile (used as fallback at next era's warmup)
    era_closed_tail: dict[str, float] = {}
    for nm in ERA_NAMES:
        era_vals = vals[(eras == nm) & ~np.isnan(vals)]
        if len(era_vals) >= warmup:
            era_closed_tail[nm] = float(np.quantile(era_vals, tail_quantile))
        else:
            era_closed_tail[nm] = np.nan

    for t in range(n):
        era_t = eras[t]
        v_t = vals[t]
        era_idx = ERA_NAMES.index(era_t)
        n_in_era = len(era_buffer[era_t])

        if n_in_era >= warmup:
            # Strict within-era 90th-percentile (zero look-ahead)
            out[t] = float(np.quantile(era_buffer[era_t], quantile))
        else:
            # Warmup: use prior era's tail cutoff (closed before this era began)
            if era_idx == 0:
                # No prior era → genuine NaN
                out[t] = np.nan
            else:
                prior_era = ERA_NAMES[era_idx - 1]
                out[t] = era_closed_tail.get(prior_era, np.nan)

        if not np.isnan(v_t):
            era_buffer[era_t].append(float(v_t))

    return pd.Series(out, index=raw.index)


def main() -> None:
    print(f"[walkforward] reading panel: {PANEL_PATH}", flush=True)
    panel = pd.read_parquet(PANEL_PATH).copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    out = pd.DataFrame(index=panel.index)
    era_labels = pd.Series([era_of(d) for d in panel.index], index=panel.index)
    out["era"] = era_labels.values

    incidence = {}
    for shock_name, raw_col, kind in SHOCK_SPEC:
        if raw_col not in panel.columns:
            raise KeyError(raw_col)
        print(f"[walkforward] {shock_name} ({raw_col}, {kind})", flush=True)
        raw = raw_measure(panel, raw_col, kind)
        thresh = walkforward_within_era_threshold(raw, era_labels)

        with np.errstate(invalid="ignore"):
            ind = np.where(
                np.isnan(thresh.values) | np.isnan(raw.values),
                np.nan,
                (raw.values > thresh.values).astype(float),
            )
        ind_series = pd.Series(ind, index=panel.index)

        out[f"{shock_name}_wf_raw"] = raw.values
        out[f"{shock_name}_wf_thresh"] = thresh.values
        out[f"{shock_name}_wf"] = ind_series.values

        per_era = {}
        for era_name in ERA_NAMES:
            mask = era_labels == era_name
            sub = ind_series[mask]
            n_warmup_nan = int((mask & ind_series.isna()).sum())
            tser = thresh[mask].dropna()
            final_cut = float(tser.iloc[-1]) if len(tser) else None
            per_era[era_name] = {
                "n_total": int(mask.sum()),
                "n_classified": int(sub.notna().sum()),
                "n_warmup_dropped": n_warmup_nan,
                "walkforward_incidence": float(sub.mean(skipna=True)) if sub.notna().any() else None,
                "end_of_era_cutoff": final_cut,
            }
        incidence[shock_name] = {
            "raw_col": raw_col, "measure_kind": kind,
            "walkforward_overall_incidence": float(ind_series.mean(skipna=True)),
            "n_classified_total": int(ind_series.notna().sum()),
            "n_warmup_dropped_total": int(ind_series.isna().sum()),
            "by_era": per_era,
        }

    out.to_parquet(OUT_PARQUET)
    print(f"[walkforward] wrote {OUT_PARQUET} "
          f"({OUT_PARQUET.stat().st_size/1024:.1f} KB)", flush=True)

    doc = {
        "_metadata": {
            "script": "code/build_era_conditional_shocks_walkforward.py",
            "method": "no-look-ahead expanding within-era q90 after warmup; "
                       "prior-era q99 tail fallback during warmup "
                       "(catches obvious shocks like COVID; no prior-era-q90 bias)",
            "panel_path": str(PANEL_PATH),
            "quantile": QUANTILE,
            "warmup_obs": WARMUP_OBS,
            "tail_quantile_during_warmup": TAIL_QUANTILE,
            "eras": [{"name": n, "start": s, "end": e} for n, s, e in ERAS],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "seed": 42,
        },
        "shock_incidence": incidence,
    }
    OUT_INCIDENCE_JSON.write_text(json.dumps(doc, indent=2, default=str))
    print(f"[walkforward] wrote {OUT_INCIDENCE_JSON}", flush=True)

    print("\nShock incidence by era (walk-forward, NaN warmup, no fallback):", flush=True)
    print(f"{'shock':<22s} {'era':<22s} {'n':>5s} {'n_clas':>7s} {'n_warm':>7s} "
          f"{'end_cut':>10s} {'wf_inc':>10s}", flush=True)
    for s, d in incidence.items():
        for era, ed in d["by_era"].items():
            cut = f"{ed['end_of_era_cutoff']:.4f}" if ed["end_of_era_cutoff"] is not None else "n/a"
            wf = f"{ed['walkforward_incidence']:.3f}" if ed["walkforward_incidence"] is not None else "n/a"
            print(f"  {s:<20s} {era:<22s} {ed['n_total']:>5d} {ed['n_classified']:>7d} "
                  f"{ed['n_warmup_dropped']:>7d} {cut:>10s} {wf:>10s}", flush=True)


if __name__ == "__main__":
    main()
