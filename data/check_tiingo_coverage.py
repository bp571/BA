"""Prüft Tiingo-Verfügbarkeit & Historie für eine Tickerliste.

Nutzt den Metadata-Endpoint (`/tiingo/daily/{ticker}`) — ein einziger leichter
Request pro Ticker, ohne komplette Preisdaten zu laden.

Usage:
    python data/check_tiingo_coverage.py --input tickers.txt --output coverage.csv
    python data/check_tiingo_coverage.py --tickers XOM,CVX,BP,SHEL
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv


def check_ticker(ticker: str, api_key: str, min_start: str = "2010-01-01") -> dict:
    """Fragt Tiingo-Metadaten für einen Ticker ab.

    Args:
        ticker: Tiingo-Ticker (US oder ADR).
        api_key: Tiingo API Key.
        min_start: Frühestes gewünschtes Startdatum für Historien-Check.

    Returns:
        Dict mit Feldern: ticker, available, start_date, end_date,
        history_ok, name, exchange, error.
    """
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}"
    headers = {"Content-Type": "application/json", "Authorization": f"Token {api_key}"}

    result = {
        "ticker": ticker,
        "available": False,
        "start_date": None,
        "end_date": None,
        "history_ok": False,
        "name": None,
        "exchange": None,
        "error": None,
    }

    try:
        # Retry on 429 mit exponentiellem Backoff (Tiingo Free: 50/h)
        for attempt in range(8):
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 60))
                wait = min(max(retry_after, 60 * (2 ** attempt)), 3600)
                print(f"    [429] {ticker}: warte {wait}s (Versuch {attempt+1}/8)")
                time.sleep(wait)
                continue
            break
        if r.status_code == 404:
            result["error"] = "not_found"
            return result
        if r.status_code == 429:
            result["error"] = "rate_limited"
            return result
        r.raise_for_status()
        meta = r.json()

        start = meta.get("startDate")
        end = meta.get("endDate")
        result.update(
            available=True,
            start_date=start,
            end_date=end,
            name=meta.get("name"),
            exchange=meta.get("exchangeCode"),
            history_ok=(start is not None and start <= min_start),
        )
    except requests.HTTPError as e:
        result["error"] = f"http_{e.response.status_code}"
    except Exception as e:
        result["error"] = str(e)[:80]

    return result


def load_tickers(input_path: Optional[str], tickers_arg: Optional[str]) -> list[str]:
    if tickers_arg:
        return [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    if input_path:
        path = Path(input_path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            col = "ticker" if "ticker" in df.columns else df.columns[0]
            return [str(t).strip().upper() for t in df[col].dropna().tolist()]
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            data = yaml.safe_load(path.read_text())
            # Sucht rekursiv die erste Liste von Dicts mit 'symbol'-Feld
            def _find(d):
                if isinstance(d, list):
                    return [x["symbol"] for x in d if isinstance(x, dict) and "symbol" in x] or None
                if isinstance(d, dict):
                    for v in d.values():
                        r = _find(v)
                        if r:
                            return r
                return None
            syms = _find(data) or []
            return [str(s).strip().upper() for s in syms]
        return [
            line.strip().upper()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    raise ValueError("Entweder --input oder --tickers angeben.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiingo Coverage Check")
    parser.add_argument("--input", help="Pfad zu .txt (eine Zeile pro Ticker) oder .csv (Spalte 'ticker').")
    parser.add_argument("--tickers", help="Komma-separierte Tickerliste, z.B. 'XOM,CVX,BP'.")
    parser.add_argument("--output", default="data/tiingo_coverage_report.csv", help="Output-CSV.")
    parser.add_argument("--min-start", default="2010-01-01", help="Frühestes gewünschtes Startdatum.")
    parser.add_argument("--sleep", type=float, default=0.3, help="Pause zwischen Requests (s).")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        raise SystemExit("TIINGO_API_KEY fehlt in .env")

    tickers = load_tickers(args.input, args.tickers)

    # Resume: bestehende Ergebnisse aus Output-CSV übernehmen, Erfolge skippen
    out = Path(args.output)
    existing: dict[str, dict] = {}
    if out.exists():
        prev = pd.read_csv(out)
        for _, row in prev.iterrows():
            existing[row["ticker"]] = row.to_dict()
        skip = {t for t, r in existing.items() if r.get("available") is True or r.get("error") == "not_found"}
        print(f"Resume: {len(skip)} Ticker bereits geprüft, überspringe.")
    else:
        skip = set()

    todo = [t for t in tickers if t not in skip]
    print(f"Prüfe {len(todo)}/{len(tickers)} Ticker gegen Tiingo (min_start={args.min_start})...")

    rows = [existing[t] for t in tickers if t in existing]
    for i, t in enumerate(todo, 1):
        res = check_ticker(t, api_key, args.min_start)
        flag = "OK" if res["history_ok"] else ("AVAIL" if res["available"] else "MISS")
        print(f"  [{i:>3}/{len(todo)}] {t:<10} {flag:<6} start={res['start_date']} err={res['error']}")
        rows.append(res)
        # Inkrementell speichern, damit Abbrüche nicht alles verlieren
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_ok = int(df["history_ok"].sum())
    n_avail = int(df["available"].sum())
    print("\n--- Zusammenfassung ---")
    print(f"  Verfügbar (irgendeine Historie): {n_avail}/{len(df)}")
    print(f"  Historie >= {args.min_start}:       {n_ok}/{len(df)}")
    print(f"  Report:                          {out}")


if __name__ == "__main__":
    main()
