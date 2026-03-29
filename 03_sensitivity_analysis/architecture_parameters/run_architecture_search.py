import sys
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '02_finetuning' / 'models' / 'Kronos'))

from model.kronos import Kronos, KronosTokenizer, KronosPredictor
from data.factory import DataFactory
from experiments.runner import run_rolling_benchmark_multi_asset
from core.reproducibility import set_all_seeds


def sample_architecture_params():
    d_model_choices = [16, 32, 64, 128]
    num_heads_choices = [2, 4, 8]
    num_layers_choices = [1, 2, 3, 4]
    dropout_choices = [0.1, 0.2, 0.3]
    
    while True:
        d_model = random.choice(d_model_choices)
        num_heads = random.choice(num_heads_choices)
        
        if d_model % num_heads == 0:
            break
    
    return {
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': random.choice(num_layers_choices),
        'dropout': random.choice(dropout_choices)
    }


def create_kronos_with_params(params, device):
    s1_bits = 8
    s2_bits = 8
    ff_dim = params['d_model'] * 4
    
    tokenizer = KronosTokenizer(
        d_in=6,
        d_model=params['d_model'],
        n_heads=params['num_heads'],
        ff_dim=ff_dim,
        n_enc_layers=2,
        n_dec_layers=2,
        ffn_dropout_p=params['dropout'],
        attn_dropout_p=params['dropout'],
        resid_dropout_p=params['dropout'],
        s1_bits=s1_bits,
        s2_bits=s2_bits,
        beta=0.25,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        group_size=8
    )
    
    model = Kronos(
        s1_bits=s1_bits,
        s2_bits=s2_bits,
        n_layers=params['num_layers'],
        d_model=params['d_model'],
        n_heads=params['num_heads'],
        ff_dim=ff_dim,
        ffn_dropout_p=params['dropout'],
        attn_dropout_p=params['dropout'],
        resid_dropout_p=params['dropout'],
        token_dropout_p=params['dropout'],
        learn_te=False
    )
    
    tokenizer.to(device)
    model.to(device).eval()
    
    return KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=128)


def load_assets(seed):
    factory = DataFactory(config_path="config/assets.yaml")
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
            print(f"  Error loading {ticker}: {e}")
            continue
    
    return asset_data


def run_experiment(params, asset_data, device, context=80, forecast=12):
    try:
        predictor = create_kronos_with_params(params, device)
        
        stride = forecast
        min_data_length = min(len(df) for df in asset_data.values())
        max_steps = min((min_data_length - context - forecast) // stride + 1, 10)
        
        if max_steps <= 0:
            return None
        
        run_params = {
            'context_steps': context,
            'forecast_steps': forecast,
            'stride_steps': stride,
            'steps': max_steps
        }
        
        results = run_rolling_benchmark_multi_asset(
            predictor=predictor,
            asset_data_dict=asset_data,
            params=run_params,
            batch_size=16,
            verbose=False
        )
        
        mae_list = []
        rankic_list = []
        
        for result in results.values():
            metrics = result.get('metrics', {})
            mae_list.append(metrics.get('MAE_indicative', np.nan))
            rankic_list.append(metrics.get('RankIC_TimeSeries_Mean', np.nan))
        
        return {
            'mae': np.nanmean(mae_list),
            'rankic': np.nanmean(rankic_list)
        }
    except Exception as e:
        print(f"  Experiment failed: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--context', type=int, default=120)
    parser.add_argument('--forecast', type=int, default=6)
    args = parser.parse_args()
    
    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Architecture Sensitivity Analysis")
    print(f"Samples: {args.n_samples}")
    print(f"Device: {device}")
    print()
    
    asset_data = load_assets(args.seed)
    print(f"Loaded {len(asset_data)} assets\n")
    
    output_dir = Path("03_sensitivity_analysis/architecture_parameters/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i in tqdm(range(args.n_samples), desc="Random Search"):
        params = sample_architecture_params()
        metrics = run_experiment(params, asset_data, device, args.context, args.forecast)
        
        if metrics:
            results.append({
                'd_model': params['d_model'],
                'num_heads': params['num_heads'],
                'num_layers': params['num_layers'],
                'dropout': params['dropout'],
                'mae': metrics['mae'],
                'rankic': metrics['rankic']
            })
    
    df = pd.DataFrame(results)
    csv_path = output_dir / f"architecture_search_{args.n_samples}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nCompleted: {len(results)}/{args.n_samples} successful")
    print(f"Results saved: {csv_path}")
    print(f"\nNext: python 03_sensitivity_analysis/architecture_parameters/analyze_rf.py")


if __name__ == "__main__":
    main()
