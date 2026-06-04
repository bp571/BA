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


def main(config_path="config/energy_assets_filtered.yaml", seed=13, context=80, forecast=12,
         test_start="2021-01-01", test_end=None, results_subdir=None):
    set_all_seeds(seed=seed)
    start_time = time.time()

    # 1. Initialisierung
    factory = DataFactory(config_path=config_path)
    predictor = load_kronos_predictor()

    subdir = results_subdir if results_subdir else f"seed_{seed}"
    results_dir = Path("01_model_comparison/results/kronos") / subdir
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
            
            # Test-Fenster mit Context-Buffer: behalte `context` Bars VOR test_start,
            # sodass das erste Forecast-Fenster exakt bei test_start beginnt. Identisch
            # zur Fine-Tuned-Eval (main_kronos_finetuned.py) → faire Fold-Vergleichbarkeit.
            test_start_ts = pd.Timestamp(test_start)
            test_end_ts = pd.Timestamp(test_end) if test_end else None
            ctx_buffer = base_params['context_steps']
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index.searchsorted(test_start_ts, side='left')
                lo = max(0, idx - ctx_buffer)
                df = df.iloc[lo:]
                if test_end_ts is not None:
                    df = df[df.index <= test_end_ts]
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime').reset_index(drop=True)
                idx = df['datetime'].searchsorted(test_start_ts, side='left')
                lo = max(0, idx - ctx_buffer)
                df = df.iloc[lo:]
                if test_end_ts is not None:
                    df = df[df['datetime'] <= test_end_ts]

            if df.empty:
                skipped_tickers.append(ticker)
                continue

            if 'datetime' not in df.columns:
                df = df.reset_index().rename(columns={df.index.name: 'datetime', 'date': 'datetime'})

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
    BATCH_SIZE = 1024
    
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
            'data_source': 'yahoo',
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
    parser.add_argument("--config", type=str, default="config/energy_assets_filtered.yaml")
    parser.add_argument("--context", type=int, default=80)
    parser.add_argument("--forecast", type=int, default=12)
    parser.add_argument("--test-start", type=str, default="2021-01-01")
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--results-subdir", type=str, default=None)
    args = parser.parse_args()
    main(config_path=args.config, seed=args.seed, context=args.context, forecast=args.forecast,
         test_start=args.test_start, test_end=args.test_end, results_subdir=args.results_subdir)
