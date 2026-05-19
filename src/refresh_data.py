"""Daily data refresh: pull latest VIX/BTC/SPY/macro/GPR, extend panel,
rebuild walk-forward shocks, rebuild lookup, regenerate dashboard JSON.

Sources:
  - VIX, WTI (DCOILWTICO), USD broad (DTWEXBGS), 10Y (DGS10), STLFSI4 → FRED
  - BTC daily close → CoinMetrics community API
  - SPY daily close → yfinance
  - GPR-threat (gprd_threat) → Caldara-Iacoviello daily CSV

Designed for GitHub Actions: runs nightly, commits the updated parquets +
JSON if anything new arrived. Fails soft on individual source errors so
one bad source doesn't kill the entire refresh.

Usage:
  python src/refresh_data.py
"""
from __future__ import annotations

import io
import subprocess
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "data" / "panel_with_shocks.parquet"
BTC_CAL_PATH = ROOT / "data" / "btc_calendar_daily.parquet"

# FRED series IDs we pull
FRED_SERIES = {
    "vix": "VIXCLS",
    "wti": "DCOILWTICO",
    "usd_broad": "DTWEXBGS",
    "y10_fred": "DGS10",
    "stlfsi": "STLFSI4",
}

# GPR-threat source (Caldara-Iacoviello)
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def log(msg):
    print(f"[refresh] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Source pulls
# ----------------------------------------------------------------------------
def fetch_fred(series_id: str) -> pd.Series:
    """FRED CSV endpoint (no API key required)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    val_col = series_id
    if val_col not in df.columns:
        val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    series = pd.to_numeric(df[val_col], errors="coerce")
    series.name = series_id
    return series.dropna()


def fetch_btc_coinmetrics(start_date: str) -> pd.Series:
    """BTC daily close from CoinMetrics Community API (free, no key required).

    Note: the correct endpoint is `community-api.coinmetrics.io` (with the
    `community-api` subdomain). `api.coinmetrics.io` requires authentication.
    """
    url = (
        "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        f"?assets=btc&metrics=PriceUSD&start_time={start_date}"
        "&frequency=1d&page_size=10000"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise RuntimeError("CoinMetrics returned no data")
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df["PriceUSD"] = pd.to_numeric(df["PriceUSD"], errors="coerce")
    s = df.set_index("time")["PriceUSD"].sort_index().dropna()
    s.name = "btc"
    return s


def fetch_spy_yfinance(start_date: str) -> pd.Series:
    """SPY daily close from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed (pip install yfinance)")
    df = yf.download("SPY", start=start_date, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("yfinance returned no SPY data")
    # Handle MultiIndex columns (newer yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        col = df["Close"]["SPY"] if ("Close", "SPY") in df.columns else df["Close"].iloc[:, 0]
    else:
        col = df["Close"]
    s = col.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "spy"
    return s.dropna()


def fetch_gpr_threat() -> pd.Series:
    """Caldara-Iacoviello GPR-threat daily index.

    The XLS has columns: DAY (int YYYYMMDD), N10D, GPRD, GPRD_ACT,
    GPRD_THREAT, date (proper datetime), ... We use the `date` column
    and the `GPRD_THREAT` value column.
    """
    try:
        r = requests.get(GPR_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content))
    except Exception as e:
        raise RuntimeError(f"GPR-threat download failed: {e}")
    df.columns = [c.strip() for c in df.columns]
    if "GPRD_THREAT" not in df.columns:
        raise RuntimeError(f"GPRD_THREAT column missing. Available: {list(df.columns)}")
    if "date" in df.columns:
        date_col = "date"
        df[date_col] = pd.to_datetime(df[date_col])
    elif "DAY" in df.columns:
        # DAY is integer YYYYMMDD
        date_col = "DAY"
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d")
    else:
        raise RuntimeError(f"No date column. Available: {list(df.columns)}")
    s = df.set_index(date_col)["GPRD_THREAT"].sort_index().dropna()
    s.name = "gprd_threat"
    return s


# ----------------------------------------------------------------------------
# Panel reconstruction
# ----------------------------------------------------------------------------
def rebuild_panel(panel_existing: pd.DataFrame) -> pd.DataFrame:
    """Extend panel with whatever fresh data is available. Fails soft per source."""
    log(f"Existing panel: {len(panel_existing)} rows, ends {panel_existing.index.max().date()}")
    start = (panel_existing.index.max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    # Pull each source; collect successes
    new_data = {}
    for name, fred_id in FRED_SERIES.items():
        try:
            s = fetch_fred(fred_id)
            new_data[name] = s
            log(f"FRED {name} ({fred_id}): {len(s)} obs, latest {s.index.max().date()}")
        except Exception as e:
            log(f"  FAILED FRED {name}: {e}")

    try:
        new_data["btc"] = fetch_btc_coinmetrics(start)
        log(f"BTC (CoinMetrics): {len(new_data['btc'])} new obs, latest {new_data['btc'].index.max().date()}")
    except Exception as e:
        log(f"  FAILED BTC: {e}")

    try:
        new_data["spy"] = fetch_spy_yfinance(start)
        log(f"SPY (yfinance): {len(new_data['spy'])} new obs, latest {new_data['spy'].index.max().date()}")
    except Exception as e:
        log(f"  FAILED SPY: {e}")

    try:
        new_data["gprd_threat"] = fetch_gpr_threat()
        log(f"GPR-threat: {len(new_data['gprd_threat'])} obs, latest {new_data['gprd_threat'].index.max().date()}")
    except Exception as e:
        log(f"  FAILED GPR-threat (will keep stale GPR data): {e}")

    # Build NYSE trading-day index up to today, union of FRED series
    if not new_data:
        log("No fresh sources succeeded; panel unchanged")
        return panel_existing

    # Use existing panel's column order. For each new column, forward-fill/reindex
    # onto the panel's NYSE-trading-day index extended through latest VIX date.
    latest_vix_date = new_data.get("vix", pd.Series()).index.max() if "vix" in new_data else None
    if latest_vix_date is None or latest_vix_date <= panel_existing.index.max():
        log("No new VIX data — keeping panel as-is")
        return panel_existing

    # New NYSE days = VIX days after the panel's last date, up to latest
    vix_new = new_data["vix"]
    new_nyse_days = vix_new.index[(vix_new.index > panel_existing.index.max()) & (vix_new.index <= latest_vix_date)]
    if not len(new_nyse_days):
        log("No new NYSE days to append")
        return panel_existing
    log(f"Appending {len(new_nyse_days)} new NYSE days: {new_nyse_days[0].date()} → {new_nyse_days[-1].date()}")

    # Construct new rows
    new_rows = pd.DataFrame(index=new_nyse_days)
    for col in panel_existing.columns:
        new_rows[col] = np.nan
    # Map series to columns
    for col, src in [("vix", "vix"), ("wti", "wti"), ("usd_broad", "usd_broad"),
                     ("y10_fred", "y10_fred"), ("stlfsi", "stlfsi"),
                     ("btc", "btc"), ("spy", "spy"), ("gprd_threat", "gprd_threat")]:
        if src in new_data and col in panel_existing.columns:
            s = new_data[src]
            # For weekly series (STLFSI4): forward-fill onto daily NYSE
            if src == "stlfsi":
                s_daily = s.reindex(new_nyse_days, method="ffill")
            elif src == "gprd_threat":
                s_daily = s.reindex(new_nyse_days, method="ffill")
            else:
                s_daily = s.reindex(new_nyse_days)
            new_rows[col] = s_daily.values

    # Forward-fill from existing panel's last row for any column that's still NaN
    # (e.g., GPR-threat if source failed today)
    last_existing = panel_existing.iloc[-1]
    for col in new_rows.columns:
        if new_rows[col].isna().all() and col in last_existing.index and pd.notna(last_existing[col]):
            new_rows[col] = last_existing[col]
            log(f"  {col}: forward-filled from last existing value (source failed)")

    # Preserve any non-data columns (like full-sample shock flags) — set to 0 for new days
    # The walk-forward build_era_conditional_shocks_walkforward.py will compute its own shock flags
    extended = pd.concat([panel_existing, new_rows])
    extended = extended[~extended.index.duplicated(keep="last")].sort_index()
    log(f"Extended panel: {len(extended)} rows, now ends {extended.index.max().date()}")
    return extended


def rebuild_btc_calendar(btc_cal_existing: pd.DataFrame) -> pd.DataFrame:
    """Extend the 24/7 BTC calendar series."""
    start = (btc_cal_existing.index.max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        s = fetch_btc_coinmetrics(start)
    except Exception as e:
        log(f"  FAILED BTC calendar refresh: {e}")
        return btc_cal_existing
    s.name = "close"
    df = s.to_frame()
    # Merge
    combined = pd.concat([btc_cal_existing, df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    log(f"BTC calendar: {len(combined)} rows, ends {combined.index.max().date()}")
    return combined


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def main():
    log(f"Starting refresh at {datetime.now(timezone.utc).isoformat()}")

    # 1. Refresh panel
    if not PANEL_PATH.exists():
        log(f"ERROR: panel parquet missing at {PANEL_PATH}")
        sys.exit(1)
    panel = pd.read_parquet(PANEL_PATH)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    panel_new = rebuild_panel(panel)
    if len(panel_new) > len(panel):
        panel_new.to_parquet(PANEL_PATH)
        log(f"WROTE panel: +{len(panel_new) - len(panel)} new rows")
    else:
        log("panel unchanged")

    # 2. Refresh BTC calendar
    if BTC_CAL_PATH.exists():
        btc_cal = pd.read_parquet(BTC_CAL_PATH)
        btc_cal.index = pd.to_datetime(btc_cal.index)
        btc_cal = btc_cal.sort_index()
        btc_cal_new = rebuild_btc_calendar(btc_cal)
        if len(btc_cal_new) > len(btc_cal):
            btc_cal_new.to_parquet(BTC_CAL_PATH)
            log(f"WROTE BTC calendar: +{len(btc_cal_new) - len(btc_cal)} new rows")
        else:
            log("BTC calendar unchanged")

    # 3. Rebuild walk-forward shock panel
    log("Rebuilding walk-forward shocks...")
    r = subprocess.run([sys.executable, str(ROOT / "src" / "build_era_conditional_shocks_walkforward.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  build_era_conditional_shocks_walkforward FAILED:\n{r.stderr}")
        sys.exit(1)
    log("  walk-forward shocks rebuilt")

    # 4. Rebuild lookup table
    log("Rebuilding 84-cell lookup table...")
    r = subprocess.run([sys.executable, str(ROOT / "src" / "dashboard_lookup.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  dashboard_lookup FAILED:\n{r.stderr}")
        sys.exit(1)
    log("  lookup rebuilt")

    # 5. Regenerate dashboard JSON
    log("Regenerating dashboard JSON...")
    r = subprocess.run([sys.executable, str(ROOT / "src" / "update_data.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  update_data FAILED:\n{r.stderr}")
        sys.exit(1)
    log("  dashboard JSON regenerated")

    log("Refresh complete")


if __name__ == "__main__":
    main()
