"""
Multi-Asset Kronos Rolling Forecast
Kronos Rolling Forecast für verschiedene Assets mit Metrik-Vergleich.
"""

import pandas as pd
import torch
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'models' / 'Kronos'))
sys.path.append(str(project_root / 'experiments'))

from model import Kronos, KronosTokenizer, KronosPredictor
from forecast_common import load_and_prepare_data, get_data_periods
from metrics import calculate_all_metrics

# Asset Konfiguration
ASSETS = {
    'energy': {
        'name': 'Strom',
        'path': 'data/processed/energy_2020-2025.csv',
        'price_col': 'close',
        'start_date': '2023-01-01'
    },
    'apple': {
        'name': 'Apple',
        'path': 'data/processed/apple_2025.csv',
        'price_col': 'close',
        'start_date': '2025-07-01'
    },
    'gold': {
        'name': 'Gold',
        'path': 'data/processed/gold_2025_processed.csv',
        'price_col': 'close',
        'start_date': '2025-07-01'
    }
}

PARAMS = {
    'context_hours': 400,
    'forecast_hours': 24,
    'steps': 30
}


def run_asset_forecast(predictor, asset_key, config):
    """Führt Rolling Forecast für ein Asset durch"""
    print(f"Processing {config['name']}...")
    
    df, use_cols, price_col = load_and_prepare_data(config['path'], config['price_col'])
    results = []
    
    for i in tqdm(range(PARAMS['steps']), desc=asset_key):
        context_data, target_data = get_data_periods(
            df, config['start_date'], i,
            PARAMS['context_hours'], PARAMS['forecast_hours']
        )
        
        if context_data is None or target_data is None:
            continue
            
        try:
            pred_df = predictor.predict(
                df=context_data[use_cols],
                x_timestamp=context_data['datetime'],
                y_timestamp=target_data['datetime'],
                pred_len=PARAMS['forecast_hours'],
                T=1.0,
                top_p=0.9,
                sample_count=1
            )
            
            results.append({
                "actual": target_data[price_col].tolist(),
                "predicted": pred_df[price_col].tolist()
            })
            
        except Exception as e:
            continue
    
    print(f"{config['name']}: {len(results)}/{PARAMS['steps']} successful")
    return results


def print_comparison(all_results):
    """Zeigt Metrik-Vergleich zwischen Assets"""
    print("\n" + "="*50)
    print("KRONOS MULTI-ASSET COMPARISON")
    print("="*50)
    
    asset_metrics = {}
    for asset_key, results in all_results.items():
        if results:
            # Flatten results
            all_actual = []
            all_pred = []
            for r in results:
                all_actual.extend(r['actual'])
                all_pred.extend(r['predicted'])
            
            # Calculate metrics using existing metrics.py
            metrics = calculate_all_metrics(np.array(all_actual), np.array(all_pred))
            asset_metrics[asset_key] = metrics
    
    # Results table
    print(f"\n{'Asset':<10} {'MAE':<8} {'RMSE':<8} {'MAPE%':<8} {'IC':<8} {'RankIC':<8} {'DirAcc%':<8}")
    print("-" * 68)
    
    for asset_key, metrics in asset_metrics.items():
        name = ASSETS[asset_key]['name']
        rank_ic = metrics.get('RankIC', 0.0)
        print(f"{name:<10} {metrics['MAE']:<8.3f} {metrics['RMSE']:<8.3f} "
              f"{metrics['MAPE']:<8.1f} {metrics['IC']:<8.3f} {rank_ic:<8.3f} {metrics['Directional_Accuracy']:<8.1f}")
    
    print(f"\nDurchschnittliche Metriken über alle Forecast-Perioden:")
    print(f"Anzahl forecast-Tage pro Asset:")
    for asset_key, metrics in asset_metrics.items():
        name = ASSETS[asset_key]['name']
        days = metrics.get('Count', 0) // 24  # Assuming 24 hour forecasts
        print(f"  {name}: {metrics.get('forecast_days', 0)} Tage")
    
    print("="*50)


def main(assets=None, steps=None):
    """Hauptfunktion"""
    if steps:
        PARAMS['steps'] = steps
    
    selected_assets = assets if assets else list(ASSETS.keys())
    
    # Check data availability
    available_assets = []
    for asset in selected_assets:
        if Path(ASSETS[asset]['path']).exists():
            available_assets.append(asset)
        else:
            print(f"Data not found for {asset}")
    
    if not available_assets:
        print("No data files found")
        return
    
    # Initialize Kronos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    
    # Run forecasts
    all_results = {}
    for asset in available_assets:
        results = run_asset_forecast(predictor, asset, ASSETS[asset])
        all_results[asset] = results
    
    # Show comparison
    print_comparison(all_results)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Kronos Multi-Asset Forecast')
    parser.add_argument('--assets', nargs='+', choices=list(ASSETS.keys()),
                        help='Assets to forecast')
    parser.add_argument('--steps', type=int, help='Forecast steps')
    
    args = parser.parse_args()
    main(args.assets, args.steps)