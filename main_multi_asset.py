"""
Multi-Asset Forecasting mit Batch-Processing

Optimierte Version von main.py, die Kronos' predict_batch() Capability nutzt
für dramatisch schnellere Verarbeitung mehrerer Assets.

Performance-Ziel: 30-40x Speedup gegenüber sequentieller Verarbeitung
(von ~70 Minuten auf ~2-5 Minuten)
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

from data.factory import DataFactory
from core.model_loader import load_kronos_predictor
from experiments.runner import run_rolling_benchmark_multi_asset
from tqdm import tqdm


def main():
    print("=" * 70)
    print("🚀 MULTI-ASSET BATCH FORECASTING PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Setup
    print("\n📦 Loading components...")
    factory = DataFactory()
    predictor = load_kronos_predictor()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Basis-Parameter (identisch zu main.py)
    base_params = {
        'context_steps': 80,
        'forecast_steps': 12,
        'stride_steps': 12,
        'steps': 120  # maximal möglich mit Daten seit 1.1.20
    }
    
    print(f"\n⚙️  Parameters:")
    for key, value in base_params.items():
        print(f"   - {key}: {value}")
    
    # 2. Assets laden
    print("\n📥 Loading asset data...")
    tickers = factory.get_energy_tickers()
    if not tickers:
        print("❌ Keine Ticker in assets.yaml gefunden!")
        return
    
    print(f"   Found {len(tickers)} tickers")
    
    # Lade ALLE Asset-Daten in einem Durchgang
    asset_data = {}
    skipped_tickers = []
    
    for ticker in tqdm(tickers, desc="Loading data"):
        try:
            df = factory.load_or_download(ticker)
            if df.empty:
                skipped_tickers.append(ticker)
                continue
            
            # Dynamische Berechnung der maximal möglichen Steps
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
            print(f"\n⚠️  {ticker}: Failed to load - {e}")
            skipped_tickers.append(ticker)
    
    if skipped_tickers:
        print(f"\n⚠️  Skipped {len(skipped_tickers)} tickers: {', '.join(skipped_tickers)}")
    
    if not asset_data:
        print("❌ No valid assets loaded!")
        return
    
    print(f"\n✅ Successfully loaded {len(asset_data)} assets")
    
    # 3. BATCH PROCESSING - Der Hauptunterschied zu main.py!
    print("\n" + "=" * 70)
    print("🔥 STARTING BATCH PREDICTION (This is where the magic happens!)")
    print("=" * 70)
    
    batch_start = time.time()
    
    # Batch-Size kann angepasst werden basierend auf verfügbarem GPU-Memory
    # Für die meisten GPUs (8-16GB) ist 32-64 ein guter Start
    BATCH_SIZE = 48  
    
    all_results = run_rolling_benchmark_multi_asset(
        predictor=predictor,
        asset_data_dict=asset_data,
        params=base_params,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    
    batch_duration = time.time() - batch_start
    print(f"\n⏱️  Batch processing completed in {batch_duration:.1f} seconds")
    
    # 4. Ergebnisse speichern (identisch zu main.py)
    print("\n💾 Saving results...")
    
    final_summary = {}
    for ticker, result in all_results.items():
        if result:
            final_summary[ticker] = result['metrics']
            
            # Speichere detaillierte Ergebnisse pro Asset
            output_path = results_dir / f"result_{ticker}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
    
    # Gesamtergebnis speichern
    final_path = results_dir / "final_energy_study.json"
    with open(final_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'params': base_params,
            'batch_size': BATCH_SIZE,
            'processing_time_seconds': batch_duration,
            'n_assets_processed': len(all_results),
            'n_assets_total': len(tickers),
            'summary': final_summary
        }, f, indent=4)
    
    # 5. Performance Summary
    total_duration = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED!")
    print("=" * 70)
    print(f"\n📊 Performance Summary:")
    print(f"   - Total time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"   - Batch processing: {batch_duration:.1f}s ({batch_duration/60:.1f} min)")
    print(f"   - Assets processed: {len(all_results)}/{len(tickers)}")
    print(f"   - Batch size: {BATCH_SIZE}")
    
    if all_results:
        total_windows = sum(r['metrics'].get('N_Windows', 0) for r in all_results.values())
        total_predictions = sum(r['metrics'].get('N_Predictions', 0) for r in all_results.values())
        avg_time_per_window = batch_duration / total_windows if total_windows > 0 else 0
        
        print(f"   - Total windows: {total_windows}")
        print(f"   - Total predictions: {total_predictions}")
        print(f"   - Avg time/window: {avg_time_per_window:.3f}s")
        print(f"   - Windows/second: {total_windows/batch_duration:.1f}")
    
    print(f"\n📁 Results saved to: {results_dir}")
    print(f"   - Final summary: {final_path}")
    print(f"   - Individual results: result_{{ticker}}.json")
    
    print("\n" + "=" * 70)
    
    # Optionaler Vergleich mit sequentieller Version
    print("\n💡 Performance Comparison:")
    estimated_sequential_time = total_windows * 2.5  # ~2.5s pro Window sequential
    speedup = estimated_sequential_time / batch_duration if batch_duration > 0 else 0
    print(f"   - Estimated sequential time: {estimated_sequential_time:.0f}s ({estimated_sequential_time/60:.1f} min)")
    print(f"   - Actual batch time: {batch_duration:.1f}s ({batch_duration/60:.1f} min)")
    print(f"   - Speedup factor: ~{speedup:.1f}x faster! 🚀")
    print("=" * 70)


if __name__ == "__main__":
    main()
