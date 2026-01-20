import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_PARAMS = {
    'context_hours': 80,
    'forecast_hours': 12,
    'data_path': str(PROJECT_ROOT / "data" / "processed" / "gold_2025_processed.csv"), 
    'start_date': '2025-01-10',
    'steps': 300
}

def load_and_prepare_data(data_path, price_column='close'):
    """
    Lädt Daten, entfernt Zeitzonen und stellt sicher, dass alle 
    für Kronos notwendigen OHLC-Spalten vorhanden sind.
    """
    df = pd.read_csv(data_path)
    
    # Zeitzonen entfernen (wie besprochen)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Kronos benötigt zwingend diese 4 Spalten:
    ohlc_cols = ['open', 'high', 'low', 'close']
    
    # Falls die Spalten fehlen (z.B. bei EURUSD), füllen wir sie mit dem verfügbaren Preis
    for col in ohlc_cols:
        if col not in df.columns:
            # Falls 'open' nicht da ist, nimm 'close' oder was auch immer da ist
            df[col] = df[price_column] if price_column in df.columns else df.iloc[:, 1]
                
    return df, ohlc_cols, 'close'

def get_data_periods(df, start_date, step, context_hours, forecast_hours):
    """Extrahiert Fenster ohne Data Leakage via Zeitstempel."""
    current_cutoff = pd.to_datetime(start_date) + timedelta(days=step)
    
    context_start = current_cutoff - timedelta(hours=context_hours)
    context_data = df[(df['datetime'] >= context_start) & (df['datetime'] < current_cutoff)]
    
    target_end = current_cutoff + timedelta(hours=forecast_hours)
    target_data = df[(df['datetime'] >= current_cutoff) & (df['datetime'] < target_end)]
    
    if len(context_data) < 10 or len(target_data) == 0:
        return None, None
        
    return context_data, target_data