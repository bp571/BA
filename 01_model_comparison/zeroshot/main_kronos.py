"""
Multi-Asset Forecasting mit Batch-Processing

Optimierte Version, die Kronos' predict_batch() Fähigkeit nutzt
für drastisch schnellere Verarbeitung mehrerer Assets.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from data.factory import DataFactory
from core.model_loader import load_kronos_predictor
from core.reproducibility import set_all_seeds
from experiments.runner import run_rolling_benchmark_multi_asset
from tqdm import tqdm


def main(config_path="config/assets.yaml", seed=13, context=80, forecast=12):
    set_all_seeds(seed=seed)
    start_time = time.time()

    # 1. Initialisierung
    factory = DataFactory(config_path=config_path)
    predictor = load_kronos_predictor()

    results_dir = Path("01_model_comparison/results/kronos") / f"seed_{seed}"
    results_dir.mkdir(exist_ok=True, parents=True)

    base_params = {
        'context_steps': context,
        'forecast_steps': forecast,
        'stride_steps': forecast,
        'steps': 120
    }
    
    # 2. Assets laden
    tickers = factory.get_tickers()
    if not tickers:
        print("Keine Ticker in assets.yaml gefunden!")
        return
    
    asset_data = {}
    skipped_tickers = []
    
    for ticker in tqdm(tickers, desc="Lade Assets"):
        try:
            df = factory.load_or_download(ticker)
            if df.empty:
                skipped_tickers.append(ticker)
                continue
            
            # Test Set: 2021 - heute
            test_start = pd.Timestamp('2021-01-01', tz='UTC')
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[df.index >= test_start]
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df[df['datetime'] >= test_start]
            
            if df.empty:
                skipped_tickers.append(ticker)
                continue
            
            if 'datetime' not in df.columns:
                df = df.reset_index().rename(columns={df.index.name: 'datetime', 'date': 'datetime'})
            
            # Sicherstellen, dass die Spalte existiert, bevor sie gespeichert wird
            asset_data[ticker] = df


            n_total = len(df)
            c = base_params['context_steps']
            f = base_params['forecast_steps']
            s = base_params['stride_steps']
            
            max_steps = (n_total - c - f) // s + 1
            
            if max_steps <= 0:
                skipped_tickers.append(ticker)
                continue
            
            asset_data[ticker] = df
            
        except Exception as e:
            print(f"  ⚠️ Fehler beim Vorbereiten von {ticker}: {e}")
            skipped_tickers.append(ticker)
    
    if not asset_data:
        print("Keine gültigen Assets geladen!")
        return
    
    # 3. Batch Processing
    BATCH_SIZE = 48
    
    all_results = run_rolling_benchmark_multi_asset(
        predictor=predictor,
        asset_data_dict=asset_data,
        params=base_params,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    
    # 4. Ergebnisse speichern
    final_summary = {}
    for ticker, result in all_results.items():
        if result:
            final_summary[ticker] = result['metrics']
            
            output_path = results_dir / f"result_{ticker}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
    
    final_path = results_dir / "final_energy_study.json"
    with open(final_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': 'Kronos',
            'data_source': 'tiingo',
            'config_path': config_path,
            'random_seed': seed,
            'params': base_params,
            'batch_size': BATCH_SIZE,
            'processing_time_seconds': time.time() - start_time,
            'n_assets_processed': len(all_results),
            'n_assets_total': len(tickers),
            'summary': final_summary
        }, f, indent=4)
    
    total_duration = time.time() - start_time
    print(f"Benchmark abgeschlossen in {total_duration:.1f}s. Ergebnisse in {results_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--config", type=str, default="config/assets.yaml")
    parser.add_argument("--context", type=int, default=80)
    parser.add_argument("--forecast", type=int, default=12)
    args = parser.parse_args()
    main(config_path=args.config, seed=args.seed, context=args.context, forecast=args.forecast)
