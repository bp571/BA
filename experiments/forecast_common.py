import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_PARAMS = {           # 1h   / daily
    'context_steps': 80,     # 80   / 40
    'forecast_steps': 24,    # 12   / 12
    'stride_steps': 24,       # 12   / 12
    'data_path': str(PROJECT_ROOT / "data" / "processed" / "dax_1h.csv"), 
    'start_date': '2020-03-01', 
    'steps': 1450
}

def load_and_prepare_data(data_path, price_column='close'):
    """Lädt Daten, entfernt Zeitzonen und bereitet OHLC vor."""
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    ohlc_cols = ['open', 'high', 'low', 'close']
    for col in ohlc_cols:
        if col not in df.columns:
            df[col] = df[price_column] if price_column in df.columns else df.iloc[:, 1]
                
    return df, ohlc_cols, 'close'

def get_data_periods(df, start_date, step, context_steps, forecast_steps, stride_steps=None):
    """
    Findet den Start-Index basierend auf dem Datum und nutzt dann 
    Zeitschritte für die Fenster-Extraktion.
    """
    stride = stride_steps if stride_steps is not None else DEFAULT_PARAMS['stride_steps']
    
    # 1. Finde den Index des Start-Datums (einmalig pro Experiment sinnvoll, 
    # hier zur Sicherheit in jedem Schritt berechnet)
    start_dt = pd.to_datetime(start_date)
    # Finde den ersten Index, der >= start_date ist
    start_indices = df.index[df['datetime'] >= start_dt]
    
    if len(start_indices) == 0:
        return None, None
    
    base_index = start_indices[0]
    
    # 2. Berechne den aktuellen Trennpunkt (Cutoff) via Index
    # Fix: Prevent overlap between windows by adding forecast_steps to stride
    cutoff_idx = base_index + (step * (stride + forecast_steps))
    
    # 3. Fenster-Extraktion via iloc (garantiert exakte Tensor-Größen)
    context_data = df.iloc[cutoff_idx - context_steps : cutoff_idx]
    target_data = df.iloc[cutoff_idx : cutoff_idx + forecast_steps]
    
    # Validierung
    if len(context_data) < context_steps or len(target_data) < forecast_steps:
        return None, None
        
    return context_data, target_data

def save_results_json(results, filename="forecast_results.json"):
    """Speichert Ergebnisse als JSON."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)