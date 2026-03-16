"""
Multi-Asset Forecasting mit Batch-Processing

Optimierte Version, die Kronos' predict_batch() Fähigkeit nutzt
für drastisch schnellere Verarbeitung mehrerer Assets.

Performance-Ziel: 30-40x Beschleunigung gegenüber sequentieller Verarbeitung
(von ~70 Minuten auf ~2-5 Minuten)
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

from data.factory import DataFactory
from core.model_loader import load_kronos_predictor
from core.reproducibility import set_all_seeds
from experiments.runner import run_rolling_benchmark_multi_asset
from tqdm import tqdm


def main():
    set_all_seeds(seed=13)
    start_time = time.time()
    
    # 1. Initialisierung
    factory = DataFactory()
    predictor = load_kronos_predictor()
    results_dir = Path("results_kronos")
    results_dir.mkdir(exist_ok=True)
    
    base_params = {
        'context_steps':80,
        'forecast_steps': 12,
        'stride_steps': 12,
        'steps': 120
    }
    
    # 2. Assets laden
    tickers = factory.get_energy_tickers()
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
            'random_seed': 13,
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
    main()
