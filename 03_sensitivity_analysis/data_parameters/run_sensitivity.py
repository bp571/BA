import sys
from pathlib import Path
import json
import time
import numpy as np
from SALib.sample import sobol as sobol_sample
import yaml
from tqdm import tqdm
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from data.factory import DataFactory
from core.model_loader import load_kronos_predictor
from core.reproducibility import set_all_seeds
from experiments.runner import run_rolling_benchmark_multi_asset

def load_config(config_path="03_sensitivity_analysis/data_parameters/config/parameter_space.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def generate_grid_samples(config):
    from itertools import product
    
    param_names = list(config['parameter_space'].keys())
    
    if 'univariate_values' in config:
        grid_values = {name: config['univariate_values'][name] for name in param_names}
    else:
        n_points = config['sampling'].get('grid_points', 5)
        grid_values = {}
        for name in param_names:
            pmin = config['parameter_space'][name]['min']
            pmax = config['parameter_space'][name]['max']
            grid_values[name] = np.linspace(pmin, pmax, n_points, dtype=int).tolist()
    
    param_configs = []
    for combo in product(*[grid_values[name] for name in param_names]):
        param_configs.append({name: value for name, value in zip(param_names, combo)})
    
    return param_configs

def generate_parameter_samples(config, method='sobol', n_override=None):
    if method == 'grid':
        return generate_grid_samples(config)
    
    param_names = list(config['parameter_space'].keys())
    
    # Für Integer-Gleichverteilung: Grenzen um 0.5 erweitern
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': [
            [config['parameter_space'][p]['min'] - 0.5, 
             config['parameter_space'][p]['max'] + 0.5]
            for p in param_names
        ]
    }
    
    n_samples = n_override if n_override else config['sampling']['n_samples']
    seed = config['sampling']['seed']
    
    # Exakte Sobol-Sequenz generieren (N * (D + 2) Samples)
    samples = sobol_sample.sample(problem, n_samples, calc_second_order=False, seed=seed)
    
    Path("03_sensitivity_analysis/data_parameters/results/sobol_results/raw").mkdir(parents=True, exist_ok=True)
    np.save("03_sensitivity_analysis/data_parameters/results/sobol_results/raw/sobol_X.npy", samples)
    
    param_configs = []
    for sample in samples:
        # Runden für die Ausführung im Modell
        param_configs.append({name: int(np.round(value)) for name, value in zip(param_names, sample)})
    
    return param_configs

def prepare_asset_data(config_path, seed):
    factory = DataFactory(config_path=config_path)
    tickers = factory.get_tickers()
    
    asset_data = {}
    for ticker in tickers:
        try:
            df = factory.load_or_download(ticker)
            if df.empty:
                continue
            
            test_start = pd.Timestamp('2021-01-01', tz='UTC')
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[df.index >= test_start]
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df[df['datetime'] >= test_start]
            
            if df.empty:
                continue
            
            if 'datetime' not in df.columns:
                df = df.reset_index().rename(columns={df.index.name: 'datetime', 'date': 'datetime'})
            
            asset_data[ticker] = df
        except Exception as e:
            print(f"  Fehler bei {ticker}: {e}")
            continue
    
    return asset_data

def run_experiment(params, experiment_id, asset_data, predictor, batch_size, output_dir, max_windows=None):
    context = params['context_steps']
    forecast = params['forecast_steps']
    stride = forecast
    
    min_data_length = min(len(df) for df in asset_data.values())
    max_steps = (min_data_length - context - forecast) // stride + 1
    
    if max_steps <= 0:
        return None
    
    # Limit number of windows for faster testing
    if max_windows is not None:
        max_steps = min(max_steps, max_windows)
    
    run_params = {
        'context_steps': context,
        'forecast_steps': forecast,
        'stride_steps': stride,
        'steps': max_steps
    }
    
    try:
        results = run_rolling_benchmark_multi_asset(
            predictor=predictor,
            asset_data_dict=asset_data,
            params=run_params,
            batch_size=batch_size,
            verbose=False
        )
        
        output_file = output_dir / f"exp_{experiment_id:04d}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'experiment_id': experiment_id,
                'parameters': {**params, 'stride_steps': stride},
                'max_steps': max_steps,
                'n_assets': len(results),
                'results': {
                    ticker: {
                        'metrics': result['metrics'],
                        'n_predictions': len(result['raw_values']['actual'])
                    }
                    for ticker, result in results.items()
                }
            }, f, indent=2)
        
        return results
    except Exception as e:
        print(f"  Experiment {experiment_id} failed: {e}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parameter Sensitivity Analysis for Kronos')
    parser.add_argument('--method', type=str, default='sobol',
                       choices=['sobol', 'grid'],
                       help='Sampling method')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--batch_size', type=int, default=48,
                       help='Batch size')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing results')
    parser.add_argument('--max-windows', type=int, default=None,
                       help='Max rolling windows per asset (for speed)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (5 windows, 8 samples)')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Override n_samples for Sobol/hybrid methods')
    
    args = parser.parse_args()
    
    max_windows = 5 if args.quick else args.max_windows
    n_samples_override = 8 if args.quick else args.n_samples
    
    print("=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS - KRONOS")
    print("=" * 80)
    print(f"Method: {args.method}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    if max_windows:
        print(f"Max windows per asset: {max_windows}")
    if n_samples_override:
        print(f"N samples (Sobol): {n_samples_override}")
    if args.quick:
        print("⚡ QUICK MODE")
    print("=" * 80)
    print()
    
    config = load_config()
    param_samples = generate_parameter_samples(config, method=args.method, n_override=n_samples_override)
    
    print(f"Generated {len(param_samples)} parameter configurations")
    print()
    
    output_dir = Path(f"03_sensitivity_analysis/data_parameters/results/raw_{args.method}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    existing_ids = set()
    if args.resume:
        for f in output_dir.glob("exp_*.json"):
            exp_id = int(f.stem.split('_')[1])
            existing_ids.add(exp_id)
        print(f"Resuming: {len(existing_ids)} experiments already completed")
    
    asset_data = prepare_asset_data("03_sensitivity_analysis/data_parameters/config/assets_sensitivity.yaml", args.seed)
    print(f"Loaded {len(asset_data)} assets")
    print()
    
    set_all_seeds(seed=args.seed)
    predictor = load_kronos_predictor()
    
    print("Running experiments...")
    start_time = time.time()
    
    results = []
    for i, params in enumerate(tqdm(param_samples, desc="Progress")):
        if args.resume and i in existing_ids:
            results.append(None)
            continue
        
        result = run_experiment(
            params=params,
            experiment_id=i,
            asset_data=asset_data,
            predictor=predictor,
            batch_size=args.batch_size,
            output_dir=output_dir,
            max_windows=max_windows
        )
        results.append(result)
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 80)
    print("COMPLETED")
    print("=" * 80)
    print(f"Total experiments: {len(param_samples)}")
    print(f"Successful: {sum(1 for r in results if r is not None)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(param_samples):.1f}s per experiment)")
    print(f"\nResults: {output_dir}")
    print(f"Next: python 03_sensitivity_analysis/data_parameters/analyze_sensitivity.py --results-dir {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
