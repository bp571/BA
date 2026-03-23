import os
import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.factory import DataFactory
from core.model_loader import load_kronos_predictor
from core.reproducibility import set_all_seeds
from experiments.runner import run_rolling_benchmark_multi_asset
from tqdm import tqdm

def main(config_path="config/assets.yaml", seed=13, adapter_path=None):
    import time
    set_all_seeds(seed=seed)
    start_time = time.time()
    
    factory = DataFactory(config_path=config_path)
    
    if adapter_path is None:
        adapter_path = Path("models/kronos-lora-finetuned/final")
    else:
        adapter_path = Path(adapter_path)
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    
    print(f"Loading fine-tuned Kronos model from: {adapter_path}")
    predictor = load_kronos_predictor(adapter_path=str(adapter_path))
    
    results_dir = Path("results_kronos_finetuned") / f"seed_{seed}"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    base_params = {
        'context_steps': 80,
        'forecast_steps': 12,
        'stride_steps': 12,
        'steps': 120
    }
    
    tickers = factory.get_tickers()
    if not tickers:
        print("No tickers found in assets.yaml!")
        return
    
    asset_data = {}
    skipped_tickers = []
    
    for ticker in tqdm(tickers, desc="Loading assets"):
        try:
            df = factory.load_or_download(ticker)
            if df.empty:
                skipped_tickers.append(ticker)
                continue
            
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
        print("No valid assets loaded!")
        return
    
    BATCH_SIZE = 48
    
    all_results = run_rolling_benchmark_multi_asset(
        predictor=predictor,
        asset_data_dict=asset_data,
        params=base_params,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    
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
            'model': 'Kronos-FineTuned',
            'model_base': 'NeoQuasar/Kronos-base',
            'adapter_path': str(adapter_path),
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
    print(f"Benchmark completed in {total_duration:.1f}s. Results in {results_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--config", type=str, default="config/assets.yaml")
    parser.add_argument("--adapter-path", type=str, default=None)
    args = parser.parse_args()
    main(config_path=args.config, seed=args.seed, adapter_path=args.adapter_path)
