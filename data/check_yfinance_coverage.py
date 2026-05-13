"""Prüft Yahoo Finance Verfügbarkeit & Historie für eine Tickerliste.

Nutzt yfinance: lädt 5 Tage ab `min_start` herunter — wenn etwas zurückkommt,
existiert der Ticker und hat Historie ≥ min_start.

Usage:
    python data/check_yfinance_coverage.py --input config/energy_assets.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def check_ticker(ticker: str, min_start: str = "2010-01-01") -> dict:
    """Prüft Existenz & Historie via yfinance History-Download."""
    result = {
        "ticker": ticker,
        "available": False,
        "start_date": None,
        "end_date": None,
        "history_ok": False,
        "n_rows": 0,
        "error": None,
    }
    try:
        t = yf.Ticker(ticker)
        # Komplette Historie laden, max period — günstiger Indikator
        hist = t.history(start=min_start, auto_adjust=False, actions=False)
        if hist is None or hist.empty:
            # Zweiter Versuch: vielleicht existiert Ticker, aber nicht ab min_start
            hist_all = t.history(period="max", auto_adjust=False, actions=False)
            if hist_all is None or hist_all.empty:
                result["error"] = "no_data"
                return result
            result.update(
                available=True,
                start_date=str(hist_all.index[0].date()),
                end_date=str(hist_all.index[-1].date()),
                n_rows=len(hist_all),
                history_ok=False,
            )
            return result
        result.update(
            available=True,
            start_date=str(hist.index[0].date()),
            end_date=str(hist.index[-1].date()),
            n_rows=len(hist),
            history_ok=(hist.index[0].strftime("%Y-%m-%d") <= min_start[:10] or
                        (hist.index[0] - pd.Timestamp(min_start, tz=hist.index[0].tz)).days <= 7),
        )
    except Exception as e:
        result["error"] = str(e)[:100]
    return result


def load_tickers(input_path: Optional[str], tickers_arg: Optional[str]) -> list[str]:
    if tickers_arg:
        return [t.strip() for t in tickers_arg.split(",") if t.strip()]
    if input_path:
        path = Path(input_path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            col = "ticker" if "ticker" in df.columns else df.columns[0]
            return [str(t).strip() for t in df[col].dropna().tolist()]
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            data = yaml.safe_load(path.read_text())

            def _find(d):
                if isinstance(d, list):
                    return [x["symbol"] for x in d if isinstance(x, dict) and "symbol" in x] or None
                if isinstance(d, dict):
                    for v in d.values():
                        r = _find(v)
                        if r:
                            return r
                return None
            return [str(s).strip() for s in (_find(data) or [])]
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    raise ValueError("--input oder --tickers erforderlich")


def main() -> None:
    parser = argparse.ArgumentParser(description="Yahoo Finance Coverage Check")
    parser.add_argument("--input", help="YAML/CSV/TXT mit Tickern.")
    parser.add_argument("--tickers", help="Komma-separierte Liste.")
    parser.add_argument("--output", default="data/yfinance_coverage_report.csv")
    parser.add_argument("--min-start", default="2010-01-01")
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    tickers = load_tickers(args.input, args.tickers)
    print(f"Prüfe {len(tickers)} Ticker gegen Yahoo Finance (min_start={args.min_start})...")

    rows = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(tickers, 1):
        res = check_ticker(t, args.min_start)
        flag = "OK" if res["history_ok"] else ("AVAIL" if res["available"] else "MISS")
        print(f"  [{i:>3}/{len(tickers)}] {t:<12} {flag:<6} start={res['start_date']} n={res['n_rows']} err={res['error']}")
        rows.append(res)
        pd.DataFrame(rows).to_csv(out, index=False)
        time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    n_ok = int(df["history_ok"].sum())
    n_avail = int(df["available"].sum())
    print("\n--- Zusammenfassung ---")
    print(f"  Verfügbar:              {n_avail}/{len(df)}")
    print(f"  Historie >= {args.min_start}: {n_ok}/{len(df)}")
    print(f"  Report:                 {out}")


if __name__ == "__main__":
    main()
