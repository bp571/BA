import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path

class DataFactory:
    def __init__(self, config_path="config/assets.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.raw_dir = Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def get_energy_tickers(self):
        """Extrahiert die Ticker-Liste aus der YAML."""
        return self.config.get('energy', {}).get('tickers', [])

    def load_or_download(self, ticker):
        """Prüft lokale CSV, sonst Download von yfinance."""
        path = self.raw_dir / f"{ticker}.csv"
        
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        
        print(f"Download: {ticker}...")
        # auto_adjust=True sorgt für saubere OHLC-Daten ohne Multi-Index
        df = yf.download(ticker, start="2020-01-01", auto_adjust=True)
        
        if not df.empty:
            # Falls yfinance trotzdem einen Multi-Index liefert (Ticker-Name in Header)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.index.name = 'datetime'
            df.to_csv(path)
            
        return df