"""DataFactory: laedt OHLCV-Daten via yfinance und cached sie lokal.

Interface:
    DataFactory(config_path).get_tickers() -> list[str]
    DataFactory(config_path).load_or_download(ticker) -> pd.DataFrame
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import yfinance as yf


class DataFactory:
    """Loader fuer Energy-Asset OHLCV-Daten (yfinance + lokales CSV-Cache)."""

    def __init__(self, config_path: str = "config/energy_assets_filtered.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = Path("data/raw_yahoo")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.start_date: str = (
            self.config.get("settings", {}).get("start_date", "2010-01-01")
        )

    @staticmethod
    def _safe_filename(symbol: str) -> str:
        """Yahoo-Suffixe (z.B. .HK, .TO) erlauben — nur Pfadtrenner entfernen."""
        return symbol.replace("/", "_")

    def get_tickers(self) -> list[str]:
        """Extrahiert Ticker aus YAML.

        Unterstuetzt mehrere Schemata:
          - {'energy_assets':  [{symbol, name}, ...]}        (Standard Train/Filtered)
          - {'holdout_assets': [{symbol, name}, ...]}        (Holdout-Liste)
          - {'portfolio': {'tickers': [{symbol, ...}, ...]}} (alt)
        """
        if "energy_assets" in self.config:
            raw = self.config["energy_assets"]
        elif "holdout_assets" in self.config:
            raw = self.config["holdout_assets"]
        else:
            raw = self.config.get("portfolio", {}).get("tickers", [])
        return [t["symbol"] if isinstance(t, dict) else t for t in raw]

    def _download_yahoo(self, ticker: str) -> pd.DataFrame:
        """Laedt komplette Historie ab start_date von Yahoo Finance."""
        try:
            print(f"Lade {ticker} von Yahoo...")
            df = yf.Ticker(ticker).history(
                start=self.start_date, auto_adjust=False, actions=False
            )
            if df is None or df.empty:
                print(f"  Keine Daten fuer {ticker}.")
                return pd.DataFrame()

            # Timezone-naive Index (manche Boersen liefern tz-aware)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index.name = "datetime"

            # Adj Close kann fehlen, wenn auto_adjust ueberschreibt -> nachziehen
            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]

            cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            df = df[[c for c in cols if c in df.columns]]

            time.sleep(0.15)
            return df
        except Exception as e:
            print(f"  Fehler beim Download von {ticker}: {e}")
            return pd.DataFrame()

    def load_or_download(self, ticker: str) -> pd.DataFrame:
        """Liest lokalen CSV-Cache, ansonsten Yahoo-Download + Speichern."""
        path = self.raw_dir / f"{self._safe_filename(ticker)}.csv"

        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index.name = "datetime"
            return df

        df = self._download_yahoo(ticker)
        if not df.empty:
            df.to_csv(path)
            print(f"  {ticker} gespeichert.")
        return df

    def load_aligned_panel(
        self,
        tickers: Optional[list[str]] = None,
        freq: str = "B",
    ) -> dict[str, pd.DataFrame]:
        """Laedt alle Assets und richtet sie auf gemeinsamen Handelskalender aus.

        Loesung der Trading-Kalender-Heterogenitaet (verschiedene Boersen,
        unterschiedliche Feiertage): Reindex auf gemeinsamen Business-Day-Range
        und lineare Interpolation fehlender Werte (= Mittelwert aus letztem
        und naechstem verfuegbaren Wert; Pandas: method='linear').

        Args:
            tickers: Optionale Tickerliste. Default: alle aus YAML.
            freq: Frequenz fuer Reindex. 'B' = Business Day (Mo-Fr).

        Returns:
            Dict {ticker: aligned DataFrame}.
        """
        if tickers is None:
            tickers = self.get_tickers()

        raw: dict[str, pd.DataFrame] = {}
        for t in tickers:
            df = self.load_or_download(t)
            if not df.empty:
                raw[t] = df

        if not raw:
            return {}

        # Gemeinsamer Datumsbereich: min start .. max end (Business Days)
        start = min(df.index.min() for df in raw.values())
        end = max(df.index.max() for df in raw.values())
        common_index = pd.date_range(start=start, end=end, freq=freq)

        aligned: dict[str, pd.DataFrame] = {}
        for t, df in raw.items():
            # Reindex auf gemeinsamen Kalender, lineare Interpolation der Gaps,
            # Raender (vor erstem / nach letztem realen Wert) bleiben NaN.
            reindexed = df.reindex(common_index)
            reindexed = reindexed.interpolate(method="linear", limit_area="inside")
            reindexed.index.name = "datetime"
            aligned[t] = reindexed

        return aligned
