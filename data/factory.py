import os
import time
import pandas as pd
import yaml
import requests
from pathlib import Path
from dotenv import load_dotenv

class DataFactory:
    def __init__(self, config_path="config/assets.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        load_dotenv()
        # 1. Variable auf Tiingo ändern
        self.api_key = os.getenv("TIINGO_API_KEY") 
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY fehlt in der .env-Datei!")
        
        # 2. Verzeichnisnamen anpassen
        self.raw_dir = Path("data/raw_tiingo")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def get_tickers(self):
        """Extrahiert Ticker aus der neuen 'portfolio'-Kategorie."""
        # 3. Hier 'energy' durch 'portfolio' ersetzen
        tickers = self.config.get('portfolio', {}).get('tickers', [])
        return [t['symbol'] if isinstance(t, dict) else t for t in tickers]

    def _download_tiingo(self, ticker):
        """Lädt OHLC-Daten von Tiingo für Kronos herunter."""
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        }
        start_dt = self.config.get('settings', {}).get('start_date', '2010-01-01')
        params = {
            'startDate': start_dt,  # Startdatum für die Historie
            'resampleFreq': 'daily'
        }
        
        try:
            print(f"Lade {ticker} von Tiingo...")
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"  ⚠️ Keine Daten für {ticker} erhalten.")
                return pd.DataFrame()
            
            # Tiingo liefert eine Liste von Dicts, die direkt in DataFrame geladen werden kann
            df = pd.DataFrame(data)
            
            # Spalten für Kronos/Transformer standardisieren
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'date': 'datetime'}) 
            # Setze sie NICHT als Index, oder benutze reset_index() vor dem Speichern
            df.set_index('datetime', inplace=True)
            
            # Mapping auf deine gewohnten Spaltennamen
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'adjClose': 'Adj Close',
                'volume': 'Volume'
            })
            
            # Nur relevante Spalten behalten
            cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            df = df[cols]
            
            # Kurze Pause für API-Stabilität (Tiingo Free Tier ist großzügiger als AV)
            time.sleep(1) 
            return df
            
        except Exception as e:
            print(f"  ⚠️ Fehler beim Download von {ticker}: {e}")
            return pd.DataFrame()

    def load_or_download(self, ticker):
        """Lädt lokales CSV oder triggert Tiingo-Download."""
        path = self.raw_dir / f"{ticker}.csv"
        
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        
        df = self._download_tiingo(ticker)
        
        if not df.empty:
            df.to_csv(path)
            print(f"  ✅ {ticker} gespeichert.")
            
        return df