import pandas as pd
import torch
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

# Dynamische Pfad-Logik: Wurzelverzeichnis finden (3 Ebenen hoch)
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import der flachen Funktion aus dem core-Paket und der common-Logik
try:
    from core.model_loader import load_kronos_predictor 
    from forecast_common import DEFAULT_PARAMS, load_and_prepare_data, get_data_periods
    
    # NEU: Import aus dem übergeordneten 'experiments' Ordner
    # Da 'experiments' im sys.path via project_root erreichbar ist:
    from experiments.metrics import calculate_all_metrics 
except ImportError as e:
    print(f"Fehler beim Import: {e}")
    sys.exit(1)

def run_rolling_forecast(data_path=None, start_date=None, steps=None, context_hours=None, forecast_hours=None):
    params = {
        'data_path': data_path or DEFAULT_PARAMS['data_path'],
        'start_date': start_date or DEFAULT_PARAMS['start_date'],
        'steps': steps or DEFAULT_PARAMS['steps'],
        'context_hours': context_hours or DEFAULT_PARAMS['context_hours'],
        'forecast_hours': forecast_hours or DEFAULT_PARAMS['forecast_hours']
    }
    
    # Predictor einmalig laden
    print("Lade Kronos Predictor...")
    predictor = load_kronos_predictor()
    
    # Daten laden
    df, use_cols, price_column = load_and_prepare_data(params['data_path'])
    
    all_actuals = []
    all_predictions = []
    results = []
    
    print(f"Starte Rolling Forecast: {params['steps']} Schritte")

    for i in tqdm(range(params['steps']), desc="Kronos Loop"):
        context_data, target_data = get_data_periods(
            df, params['start_date'], i, params['context_hours'], params['forecast_hours']
        )
        
        if context_data is None or target_data is None:
            continue
        
        try:
            # Sicherstellen, dass wir genügend target_data haben
            if len(target_data) == 0:
                continue
                
            # Vorhersage nur für verfügbare target_data Länge generieren
            actual_len = len(target_data)
            
            pred_df = predictor.predict(
                df=context_data[use_cols],
                x_timestamp=context_data['datetime'],
                y_timestamp=target_data['datetime'],
                pred_len=actual_len
            )
            
            # Nur die Werte speichern, für die wir auch echte Vergleichsdaten haben
            actual_values = target_data[price_column].tolist()
            predicted_values = pred_df[price_column].tolist()
            
            # Sicherstellen, dass beide Arrays gleiche Länge haben
            min_len = min(len(actual_values), len(predicted_values))
            actual_values = actual_values[:min_len]
            predicted_values = predicted_values[:min_len]
            
            if min_len > 0:  # Nur hinzufügen wenn Daten vorhanden
                all_actuals.extend(actual_values)
                all_predictions.extend(predicted_values)
                
                results.append({
                    "date": target_data['datetime'].iloc[0],
                    "actual": actual_values,
                    "predicted": predicted_values
                })
            
        except Exception as e:
            print(f"Fehler in Schritt {i + 1}: {e}")
            continue

    # --- Metrik-Ausgabe am Ende ---
    if all_actuals:
        # Konvertierung in Arrays für die metrics.py
        y_true = np.array(all_actuals)
        y_pred = np.array(all_predictions)
        
        results = calculate_all_metrics(y_true, y_pred)
        
        print("\n" + "="*50)
        print(f" FINALE ERGEBNISSE - BACHELORARBEIT ")
        print("="*50)
        print(f"RMSE: {results.get('RMSE', 0):.6f}")
        print(f"MAE:  {results.get('MAE', 0):.6f}")
        print(f"Directional Accuracy: {results.get('Directional_Accuracy', 0):.2f}%")
        print(f"IC Return: {results.get('IC_Return', 0):.4f}")
        
        if results.get('Is_Lagging'):
            print("\n[!] WARNUNG: Modell zeigt Lagging-Effekt (IC_L1 > IC_R)")
        print("="*50)
    
    return all_predictions

if __name__ == "__main__":
    run_rolling_forecast()