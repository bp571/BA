import os
import json
from datetime import datetime
from pathlib import Path

from data.factory import DataFactory
from core.model_loader import load_chronos_predictor
from core.reproducibility import set_all_seeds
from experiments.runner import run_rolling_benchmark
from tqdm import tqdm

def main():
    import time
    set_all_seeds(seed=13)
    start_time = time.time()
    
    # 1. Initialisierung
    factory = DataFactory()
    predictor = load_chronos_predictor()
    results_dir = Path("results_chronos")
    results_dir.mkdir(exist_ok=True)
    
    base_params = {
        'context_steps': 80,
        'forecast_steps': 12,
        'stride_steps': 12,
        'steps': 120
    }
    
    # 2. Assets laden
    tickers = factory.get_energy_tickers()
    if not tickers:
        print("Keine Ticker in assets.yaml gefunden!")
        return

    all_results = {}

    # 3. Assets verarbeiten
    for ticker in tqdm(tickers, desc="Verarbeite Assets"):
        try:
            df = factory.load_or_download(ticker)
            if df.empty:
                continue
            
            n_total = len(df)
            c = base_params['context_steps']
            f = base_params['forecast_steps']
            s = base_params['stride_steps']
            
            max_steps = (n_total - c - f) // s + 1
            
            current_params = base_params.copy()
            current_params['steps'] = max(0, min(base_params['steps'], max_steps))
            
            if current_params['steps'] == 0:
                continue

            result = run_rolling_benchmark(predictor, df, ticker, current_params)
            
            if result:
                all_results[ticker] = result['metrics']
                
                output_path = results_dir / f"result_{ticker}.json"
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=4)
                    
        except Exception as e:
            pass

    # 4. Ergebnisse speichern
    final_path = results_dir / "final_energy_study.json"
    with open(final_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': 'Chronos',
            'random_seed': 13,
            'params': base_params,
            'processing_time_seconds': time.time() - start_time,
            'n_assets_processed': len(all_results),
            'n_assets_total': len(tickers),
            'summary': all_results
        }, f, indent=4)
    
    total_duration = time.time() - start_time
    print(f"Benchmark abgeschlossen in {total_duration:.1f}s. Ergebnisse in {results_dir}")

if __name__ == "__main__":
    main()
