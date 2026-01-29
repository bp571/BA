import pandas as pd
import torch
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime

# Dynamische Pfad-Logik: Wurzelverzeichnis finden (3 Ebenen hoch)
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from core.model_loader import load_kronos_predictor
    from forecast_common import DEFAULT_PARAMS, load_and_prepare_data, get_data_periods, save_results_json
    
    from experiments.metrics import calculate_all_metrics, calculate_ic_statistics
except ImportError as e:
    print(f"Fehler beim Import: {e}")
    sys.exit(1)

def run_rolling_forecast():
    params = DEFAULT_PARAMS
    predictor = load_kronos_predictor()
    df, use_cols, price_column = load_and_prepare_data(params['data_path'])
    
    all_actuals = []
    all_predictions = []
    rolling_ic_values = []

    print(f"Starte Rolling Forecast: {params['steps']} Schritte")

    for i in tqdm(range(params['steps']), desc="Kronos Loop"):
        context_data, target_data = get_data_periods(
            df, params['start_date'], i, 
            params['context_steps'], params['forecast_steps'],
            stride_steps=params.get('stride_steps')
        )
        
        if context_data is None or target_data is None or len(target_data) == 0:
            continue
        
        try:
            pred_df = predictor.predict(
                df=context_data[use_cols],
                x_timestamp=context_data['datetime'],
                y_timestamp=target_data['datetime'],
                pred_len=len(target_data)
            )
            
            act = target_data[price_column].values
            pre = pred_df[price_column].values
            
            # Sammle Metriken für dieses Fenster
            window_metrics = calculate_all_metrics(act, pre)
            if 'IC_Return' in window_metrics:
                rolling_ic_values.append(window_metrics['IC_Return'])

            all_actuals.extend(act.tolist())
            all_predictions.extend(pre.tolist())
            
        except Exception as e:
            continue

    # --- Ausgabe mit DirAcc ---
    if rolling_ic_values and all_actuals:
        y_true = np.array(all_actuals)
        y_pred = np.array(all_predictions)
        
        # Globale Metriken berechnen (für DirAcc)
        global_metrics = calculate_all_metrics(y_true, y_pred)
        ic_stats = calculate_ic_statistics(rolling_ic_values)
        
        mean_ic = ic_stats.get('IC_Mean', 0)
        ci95 = ic_stats.get('IC_CI95_Mean', [0, 0])
        dir_acc = global_metrics.get('Directional_Accuracy', 0)
        ic_std = np.std(rolling_ic_values)

        print("\n" + "="*50)
        print(f"ERGEBNISSE: {params['data_path'].split('/')[-1]}")
        print("-" * 50)
        print(f"Anzahl Fenster (n):     {len(rolling_ic_values)}")
        print(f"Directional Accuracy:   {dir_acc:.2f}%")
        print(f"IC Mean:       {mean_ic:.4f}")
        print(f"IC Std:        {ic_std:.4f}")
        print(f"95% Konfidenzintervall: [{ci95[0]:.4f}, {ci95[1]:.4f}]")
        
        # Ergebnisse als JSON speichern
        results = {
            'dataset': params['data_path'].split('/')[-1],
            'timestamp': datetime.now().isoformat(),
            'n_windows': len(rolling_ic_values),
            'directional_accuracy': dir_acc,
            'ic_mean': mean_ic,
            'ic_std': ic_std,
            'ic_ci95': ci95,
            'actual_values': all_actuals,
            'predicted_values': all_predictions,
            'params': params
        }
        csv_name = Path(params['data_path']).stem
        save_results_json(results, f"kronos_forecast_{csv_name}.json")

    return all_predictions

if __name__ == "__main__":
    run_rolling_forecast()