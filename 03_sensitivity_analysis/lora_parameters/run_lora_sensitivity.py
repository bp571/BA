"""
LoRA Hyperparameter Sensitivity Analysis for Kronos

Loads pretrained NeoQuasar/Kronos-base weights and varies LoRA hyperparameters
(lora_r, lora_alpha, lora_dropout, learning_rate, use_ffn) to identify which
have the largest effect on forecasting performance.

Results are saved as CSV, compatible with analyze_rf.py for importance ranking.
"""

import sys
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from gluonts.dataset.arrow import ArrowFile

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '02_finetuning' / 'models' / 'Kronos'))

from model.kronos import Kronos, KronosTokenizer, KronosPredictor
from data.factory import DataFactory
from experiments.runner import run_rolling_benchmark_multi_asset
from core.reproducibility import set_all_seeds


class KronosDataset(Dataset):
    def __init__(self, arrow_path, context_length=512):
        self.context_length = context_length
        self.entries = list(ArrowFile(arrow_path))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        target = self.entries[idx]['target']
        if len(target) > self.context_length:
            start_idx = np.random.randint(0, len(target) - self.context_length)
            target = target[start_idx:start_idx + self.context_length]
        timestamps = np.arange(len(target))
        time_features = np.stack([
            np.zeros(len(target)),
            np.zeros(len(target)),
            timestamps % 7,
            timestamps % 31 + 1,
            (timestamps // 30) % 12 + 1,
        ], axis=-1)
        return (
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(time_features, dtype=torch.float32),
        )


def sample_lora_params():
    return {
        'lora_r':       random.choice([4, 8, 16, 32]),
        'lora_alpha':   random.choice([8, 16, 32, 64]),
        'lora_dropout': random.choice([0.0, 0.05, 0.1, 0.2]),
        'learning_rate': random.choice([5e-5, 1e-4, 3e-4]),
        'use_ffn':      random.choice([0, 1]),  # 0: attn-only, 1: attn + FFN
    }


def get_target_modules(model, use_ffn):
    n_layers = len(model.transformer)
    modules = []
    for i in range(n_layers):
        modules += [
            f"transformer.{i}.self_attn.q_proj",
            f"transformer.{i}.self_attn.k_proj",
            f"transformer.{i}.self_attn.v_proj",
            f"transformer.{i}.self_attn.out_proj",
        ]
        if use_ffn:
            modules += [
                f"transformer.{i}.ffn.w1",
                f"transformer.{i}.ffn.w2",
                f"transformer.{i}.ffn.w3",
            ]
    return modules


def apply_lora_and_train(pretrained_model, tokenizer, params, dataloader, device, max_steps):
    model = copy.deepcopy(pretrained_model)

    lora_config = LoraConfig(
        r=params['lora_r'],
        lora_alpha=params['lora_alpha'],
        target_modules=get_target_modules(model, params['use_ffn']),
        lora_dropout=params['lora_dropout'],
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, lora_config)
    model.to(device).train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=0.01,
    )

    global_step = 0
    optimizer.zero_grad()

    while global_step < max_steps:
        for batch_x, batch_x_stamp in dataloader:
            batch_x = batch_x.to(device)
            batch_x_stamp = batch_x_stamp.to(device)

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            s1_logits, s2_logits = model(
                token_seq_0[:, :-1], token_seq_1[:, :-1], batch_x_stamp[:, :-1, :]
            )
            loss, _, _ = model.base_model.head.compute_loss(
                s1_logits, s2_logits, token_seq_0[:, 1:], token_seq_1[:, 1:]
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step >= max_steps:
                break

    model.eval()
    return model


def load_asset_data():
    factory = DataFactory(config_path="config/assets.yaml")
    asset_data = {}

    for ticker in factory.get_tickers():
        try:
            df = factory.load_or_download(ticker)
            if df.empty:
                continue

            val_start = pd.Timestamp('2019-01-01', tz='UTC')
            val_end   = pd.Timestamp('2021-01-01', tz='UTC')

            if isinstance(df.index, pd.DatetimeIndex):
                df = df[(df.index >= val_start) & (df.index < val_end)]
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df[(df['datetime'] >= val_start) & (df['datetime'] < val_end)]

            if df.empty:
                continue

            if 'datetime' not in df.columns:
                df = df.reset_index().rename(columns={df.index.name or 'date': 'datetime'})

            asset_data[ticker] = df
        except Exception as e:
            print(f"  Error loading {ticker}: {e}")

    return asset_data


def run_evaluation(lora_model, tokenizer, asset_data, device, context, forecast):
    predictor = KronosPredictor(
        model=lora_model,
        tokenizer=tokenizer,
        device=device,
        max_context=512,
    )

    stride = forecast
    min_len = min(len(df) for df in asset_data.values())
    max_steps = min((min_len - context - forecast) // stride + 1, 10)

    if max_steps <= 0:
        return None

    results = run_rolling_benchmark_multi_asset(
        predictor=predictor,
        asset_data_dict=asset_data,
        params={
            'context_steps': context,
            'forecast_steps': forecast,
            'stride_steps':  stride,
            'steps':         max_steps,
        },
        batch_size=16,
        verbose=False,
    )

    mae_vals    = [r.get('metrics', {}).get('MAE_indicative', np.nan)           for r in results.values()]
    rankic_vals = [r.get('metrics', {}).get('RankIC_TimeSeries_Mean', np.nan)   for r in results.values()]
    return {
        'mae':    np.nanmean(mae_vals),
        'rankic': np.nanmean(rankic_vals),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples',   type=int,   default=60)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--context',     type=int,   default=80)
    parser.add_argument('--forecast',    type=int,   default=12)
    parser.add_argument('--train-steps', type=int,   default=150)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("LoRA Hyperparameter Sensitivity Analysis")
    print(f"  Samples:     {args.n_samples}")
    print(f"  Train steps: {args.train_steps}")
    print(f"  Device:      {device}")
    print()

    cache_dir = str(project_root / '02_finetuning' / 'models' / 'model_cache')
    data_path = project_root / 'data' / 'processed' / 'train_data_kronos.arrow'

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    print("Loading pretrained Kronos (once)...")
    tokenizer = KronosTokenizer.from_pretrained(
        "NeoQuasar/Kronos-Tokenizer-base", cache_dir=cache_dir
    )
    tokenizer.eval().to(device)

    pretrained_model = Kronos.from_pretrained("NeoQuasar/Kronos-base", cache_dir=cache_dir)
    pretrained_model.to(device).eval()

    print("Loading training data...")
    dataloader = DataLoader(
        KronosDataset(data_path, context_length=512),
        batch_size=4,
        shuffle=True,
        drop_last=True,
    )

    print("Loading validation asset data (2019-2020)...")
    asset_data = load_asset_data()
    print(f"Loaded {len(asset_data)} assets\n")

    output_dir = Path("03_sensitivity_analysis/lora_parameters/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for i in tqdm(range(args.n_samples), desc="LoRA Search"):
        params = sample_lora_params()
        try:
            lora_model = apply_lora_and_train(
                pretrained_model, tokenizer, params, dataloader, device, args.train_steps
            )
            metrics = run_evaluation(lora_model, tokenizer, asset_data, device, args.context, args.forecast)

            if metrics:
                records.append({**params, **metrics})

            del lora_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Sample {i} failed: {e}")

    df = pd.DataFrame(records)
    csv_path = output_dir / f"lora_search_{args.n_samples}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nCompleted: {len(records)}/{args.n_samples} successful")
    print(f"Results:   {csv_path}")
    print(f"\nNext: python 03_sensitivity_analysis/lora_parameters/analyze_lora_rf.py")


if __name__ == "__main__":
    main()
