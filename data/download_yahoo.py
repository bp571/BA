"""Lädt OHLCV-Daten via yfinance für die Tickerliste in config/energy_assets.yaml.

- Speichert pro Ticker eine CSV in data/raw_yahoo/{symbol}.csv
- Behält nur Assets mit Historie ab MIN_START (Default 2010-01-01)
- Schreibt finalen, gefilterten YAML nach config/energy_assets_filtered.yaml
- Schreibt Audit-Report nach data/yahoo_download_report.csv

Usage:
    python data/download_yahoo.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import yfinance as yf

CONFIG_PATH = Path("config/energy_assets.yaml")
OUT_YAML = Path("config/energy_assets_filtered.yaml")
RAW_DIR = Path("data/raw_yahoo")
REPORT = Path("data/yahoo_download_report.csv")
MIN_START = "2010-01-01"
END_DATE: Optional[str] = None
TOLERANCE_DAYS = 7


def _safe_filename(symbol: str) -> str:
    # Yahoo-Suffixe wie .HK, .TO, .NS etc. enthalten Punkte — fuer Dateinamen escapen
    return symbol.replace("/", "_")


def load_config() -> list[dict]:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return data.get("energy_assets", [])


def download_one(symbol: str) -> dict:
    """Download max history fuer ein Symbol und gib Status zurueck."""
    record = {
        "symbol": symbol,
        "n_rows": 0,
        "start": None,
        "end": None,
        "keep": False,
        "error": None,
    }
    try:
        df = yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=False)
        if df is None or df.empty:
            record["error"] = "no_data"
            return record

        # Timezone-naive Index (Yahoo liefert tz-aware fuer manche Boersen)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Auf MIN_START filtern (falls Asset frueher gelistet)
        df_filt = df[df.index >= pd.Timestamp(MIN_START)]
        if df_filt.empty:
            record.update(start=str(df.index[0].date()), end=str(df.index[-1].date()),
                          n_rows=len(df), error="no_data_after_min_start")
            return record

        start_dt = df_filt.index[0]
        # 2010-Filter: erstes Datum darf max. TOLERANCE_DAYS nach MIN_START liegen
        delta = (start_dt - pd.Timestamp(MIN_START)).days
        keep = delta <= TOLERANCE_DAYS

        record.update(
            n_rows=len(df_filt),
            start=str(start_dt.date()),
            end=str(df_filt.index[-1].date()),
            keep=keep,
        )

        if keep:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            out_path = RAW_DIR / f"{_safe_filename(symbol)}.csv"
            df_filt.to_csv(out_path)
    except Exception as e:
        record["error"] = str(e)[:120]
    return record


def main() -> None:
    assets = load_config()
    print(f"Lade {len(assets)} Ticker via yfinance (min_start={MIN_START}, tol={TOLERANCE_DAYS}d)...")

    rows = []
    for i, asset in enumerate(assets, 1):
        symbol = asset["symbol"]
        rec = download_one(symbol)
        rec["name"] = asset.get("name", "")
        flag = "KEEP" if rec["keep"] else ("SHORT" if rec["n_rows"] > 0 else "FAIL")
        print(f"  [{i:>3}/{len(assets)}] {symbol:<14} {flag:<5} start={rec['start']} n={rec['n_rows']} err={rec['error']}")
        rows.append(rec)
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT, index=False)

    # Gefiltertes YAML schreiben (nur 'keep' = True)
    kept = [{"symbol": r["symbol"], "name": r["name"]} for r in rows if r["keep"]]
    OUT_YAML.write_text(
        yaml.safe_dump({"energy_assets": kept}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    n_keep = int(df["keep"].sum())
    n_short = int(((df["n_rows"] > 0) & (~df["keep"])).sum())
    n_fail = int((df["n_rows"] == 0).sum())
    print("\n--- Zusammenfassung ---")
    print(f"  KEEP  (Historie >= {MIN_START}): {n_keep}/{len(df)}")
    print(f"  SHORT (zu kurze Historie):       {n_short}/{len(df)}")
    print(f"  FAIL  (keine Daten):             {n_fail}/{len(df)}")
    print(f"  CSVs:           {RAW_DIR}")
    print(f"  Gefiltertes YAML: {OUT_YAML}")
    print(f"  Report:         {REPORT}")


if __name__ == "__main__":
    main()
