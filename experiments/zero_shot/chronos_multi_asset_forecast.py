"""
Multi-Asset Chronos Rolling Forecast
Chronos Rolling Forecast für verschiedene Assets mit Metrik-Vergleich.
Direkt vergleichbar mit kronos_multi_asset_forecast.py
"""

import pandas as pd
import torch
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'core'))
sys.path.append(str(project_root / 'experiments'))

from model_loader import ChronosLoader
from forecast_common import load_and_prepare_data, get_data_periods
from metrics import calculate_all_metrics, calculate_baseline_comparison, calculate_persistence_baseline
import matplotlib.pyplot as plt
from pathlib import Path

# Asset Konfiguration - IDENTISCH zu Kronos Version
ASSETS = {
    'energy': {
        'name': 'Strom',
        'path': 'data/processed/energy_2020-2025.csv',
        'price_col': 'close',
        'start_date': '2025-10-01'
    },
    'apple': {
        'name': 'Apple',
        'path': 'data/processed/apple_2025.csv',
        'price_col': 'close',
        'start_date': '2025-10-01'
    },
    'gold': {
        'name': 'Gold',
        'path': 'data/processed/gold_2025_processed.csv',
        'price_col': 'close',
        'start_date': '2025-10-01'
    }
}

# Parameter - IDENTISCH zu Kronos Version
PARAMS = {
    'context_hours': 400,
    'forecast_hours': 24,
    'steps': 300
}


def prepare_chronos_data(context_data, use_cols, price_col):
    """
    Bereitet Daten für Chronos vor (benötigt id und target columns)
    
    Args:
        context_data: DataFrame mit Context-Daten
        use_cols: Liste der zu verwendenden Spalten
        price_col: Haupt-Preisspalte
    
    Returns:
        DataFrame für Chronos-Vorhersage
    """
    chronos_data = context_data.copy()
    
    # Chronos benötigt 'id' und 'target' Spalten
    chronos_data['id'] = 'asset_price'
    chronos_data['target'] = chronos_data[price_col]
    
    # Stelle sicher, dass Zeitstempel regulär sind (für Chronos wichtig)
    chronos_data = chronos_data.set_index('datetime')
    chronos_data = chronos_data.asfreq('h', method='ffill')  # Stündliche Frequenz mit Forward Fill
    chronos_data = chronos_data.reset_index()
    
    return chronos_data


def run_asset_forecast(pipeline, asset_key, config):
    """Führt Rolling Forecast für ein Asset durch - MIT PERSISTENCE BASELINE"""
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
            # Bereite Daten für Chronos vor
            chronos_context = prepare_chronos_data(context_data, use_cols, price_col)
            
            # Chronos Vorhersage
            pred_df = pipeline.predict_df(
                chronos_context,
                prediction_length=PARAMS['forecast_hours'],
                quantile_levels=[0.5],  # Nur Median für Vergleichbarkeit mit Kronos
                id_column="id",
                timestamp_column="datetime",
                target="target"
            )
            
            # Extrahiere Vorhersagen (Median = 0.5 Quantil)
            predicted_values = pred_df['0.5'].values
            
            # Extrahiere Context-Preise für Persistence Baseline
            context_prices = context_data[price_col].values
            
            results.append({
                "actual": target_data[price_col].tolist(),
                "predicted": predicted_values.tolist(),
                "context_prices": context_prices.tolist()  # Für Baseline
            })
            
        except Exception as e:
            print(f"Error in step {i}: {e}")
            continue
    
    print(f"{config['name']}: {len(results)}/{PARAMS['steps']} successful")
    return results


def plot_first_window_comparison(all_results, save_dir="evaluation/plots"):
    """
    Plottet das erste Forecast-Fenster für jedes Asset: Actual vs Chronos vs Baseline
    Zur optischen Prüfung auf Time-Lag-Verhalten
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for asset_key, results in all_results.items():
        if not results:
            continue
            
        # Nimm nur das erste Forecast-Fenster
        first_result = results[0]
        actual = np.array(first_result['actual'])
        chronos_pred = np.array(first_result['predicted'])
        
        # Erstelle Naive Baseline für erstes Fenster
        context_prices = np.array(first_result['context_prices'])
        baseline_pred = calculate_persistence_baseline(context_prices, len(actual))
        
        # Create time axis (24 hours)
        hours = np.arange(len(actual))
        
        # Plot erstellen
        plt.figure(figsize=(12, 6))
        plt.plot(hours, actual, 'b-', label='Actual Price', linewidth=2)
        plt.plot(hours, chronos_pred, 'r--', label='Chronos Prediction', linewidth=2)
        plt.plot(hours, baseline_pred, 'g:', label='Naive Baseline (Last Value)', linewidth=2)
        
        plt.title(f'{ASSETS[asset_key]["name"]} - First 24h Forecast Window\nTime-Lag Analysis (Chronos)',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Hours', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Berechne Key Metrics für Annotation
        mae_chronos = np.mean(np.abs(actual - chronos_pred))
        mae_baseline = np.mean(np.abs(actual - baseline_pred))
        
        # Korrelation für Time-Lag Detection
        correlation_chronos = np.corrcoef(actual, chronos_pred)[0, 1]
        
        # Time-Lag Test: Korrelation zwischen actual[1:] und prediction[:-1]
        if len(actual) > 1:
            lag_correlation = np.corrcoef(actual[1:], chronos_pred[:-1])[0, 1]
            lag_text = f"Time-Lag Corr: {lag_correlation:.3f}"
        else:
            lag_text = "Time-Lag: N/A"
        
        # Add metrics text box
        textstr = f'MAE Chronos: {mae_chronos:.3f}\nMAE Baseline: {mae_baseline:.3f}\n'
        textstr += f'Chronos Corr: {correlation_chronos:.3f}\n{lag_text}'
        
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save plot
        filename = f'chronos_{asset_key}_first_window_timelag_analysis.png'
        plt.tight_layout()
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved time-lag analysis plot: {filename}")


def print_comparison(all_results):
    """Zeigt Metrik-Vergleich zwischen Assets - CHRONOS vs PERSISTENCE BASELINE"""
    print("\n" + "="*95)
    print("CHRONOS vs PERSISTENCE BASELINE (SCALED & RETURN-BASED)")
    print("="*95)
    
    asset_comparisons = {}
    
    for asset_key, results in all_results.items():
        if results:
            all_actual = []
            all_pred_chronos = []
            
            for r in results:
                all_actual.extend(r['actual'])
                all_pred_chronos.extend(r['predicted'])
            
            # 1. Baseline berechnen
            baseline_predictions = []
            for r in results:
                cp = np.array(r['context_prices'])
                # Nutze die Länge der tatsächlichen Werte für den Baseline-Horizont
                h = len(r['actual'])
                b_step = calculate_persistence_baseline(cp, h)
                baseline_predictions.extend(b_step.tolist())
            
            # 2. Metriken berechnen (Nutzt deine neue calculate_all_metrics)
            c_mets = calculate_all_metrics(np.array(all_actual), np.array(all_pred_chronos))
            b_mets = calculate_all_metrics(np.array(all_actual), np.array(baseline_predictions))
            
            asset_comparisons[asset_key] = {'c': c_mets, 'b': b_mets}

    # Header: IC_R = Return IC, IC_L1 = Lag-Check (t vs t-1)
    print(f"\n{'Asset':<8} {'Model':<10} {'wMAPE%':<8} {'MASE':<8} {'IC_R':<8} {'IC_L1':<8} {'DirAcc%':<8}")
    print("-" * 95)
    
    for asset_key, comp in asset_comparisons.items():
        name = ASSETS[asset_key]['name']
        
        # Helfer um Keys sicher auszulesen (verhindert KeyError)
        for m_name, m_dict in [("Chronos", comp['c']), ("Baseline", comp['b'])]:
            # Hier liegt die Lösung: Wir nutzen .get() mit Default-Werten
            wmape_val = m_dict.get('wMAPE', 0.0)
            mase_val  = m_dict.get('MASE', 0.0)
            ic_ret    = m_dict.get('IC_Return', 0.0) # Vorher 'IC' -> jetzt 'IC_Return'
            ic_lag    = m_dict.get('IC_Lag_1', 0.0)
            dir_acc   = m_dict.get('Directional_Accuracy', 0.0)
            
            label = name if m_name == "Chronos" else ""
            print(f"{label:<8} {m_name:<10} {wmape_val:<8.1f} "
                  f"{mase_val:<8.3f} {ic_ret:<8.3f} {ic_lag:<8.3f} {dir_acc:<8.1f}")
        
        # Wissenschaftlicher Hinweis für deine BA
        if comp['c'].get('Is_Lagging', False):
            print(f"  [!] Info: {name} zeigt Lagging-Effekt (IC_L1 > IC_R)")
        print("-" * 95)


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
            print(f"Data not found for {asset}: {ASSETS[asset]['path']}")
    
    if not available_assets:
        print("No data files found")
        return
    
    # Initialize Chronos
    print("Loading Chronos-2 pipeline...")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosLoader.load("amazon/chronos-2", device_map=device_map)
    print(f"✓ Chronos loaded on {device_map}")
    
    # Run forecasts
    all_results = {}
    for asset in available_assets:
        results = run_asset_forecast(pipeline, asset, ASSETS[asset])
        all_results[asset] = results
    
    # Show comparison
    print_comparison(all_results)
    
    # Create time-lag analysis plots for first window of each asset
    plot_first_window_comparison(all_results)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chronos Multi-Asset Forecast')
    parser.add_argument('--assets', nargs='+', choices=list(ASSETS.keys()),
                        help='Assets to forecast')
    parser.add_argument('--steps', type=int, help='Forecast steps')
    
    args = parser.parse_args()
    main(args.assets, args.steps)