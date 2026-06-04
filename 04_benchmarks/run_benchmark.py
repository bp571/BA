"""Einheitlicher Benchmark-Runner fuer statistische / klassische Baselines.

Setup identisch zu 02_finetuning/evaluation/main_kronos_finetuned.py
(context=80, forecast=18, test_start=2021-01-01, Context-Buffer-Slicing),
damit Ergebnisse 1:1 mit dem fine-tuned Kronos-Modell vergleichbar sind.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.reproducibility import set_all_seeds  # noqa: E402
from data.factory import DataFactory  # noqa: E402
from experiments.runner import run_rolling_benchmark_multi_asset  # noqa: E402


def build_predictor(model: str, xgb_model_path: str):
    if model == "naive":
        from core.baseline_predictors import NaivePredictor
        return NaivePredictor(use_drift=True)
    if model == "arima":
        from core.baseline_predictors import ARIMAPredictor
        return ARIMAPredictor(order=(1, 1, 1))
    if model == "xgboost":
        from core.xgboost_wrapper import XGBoostPredictor
        return XGBoostPredictor(model_path=xgb_model_path)
    raise ValueError(f"Unbekanntes Modell: {model}")


def main(
    model: str,
    config_path: str,
    seed: int,
    context: int,
    forecast: int,
    test_start: str,
    test_end: str | None,
    results_root: str,
    xgb_model_path: str,
    batch_size: int,
) -> None:
    set_all_seeds(seed=seed)
    t0 = time.time()

    factory = DataFactory(config_path=config_path)
    predictor = build_predictor(model, xgb_model_path)

    results_dir = Path(results_root) / model / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "context_steps": context,
        "forecast_steps": forecast,
        "stride_steps": forecast,
        "steps": 120,
    }

    tickers = factory.get_tickers()
    if not tickers:
        print("Keine Ticker gefunden.")
        return

    asset_data = {}
    skipped = []
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end) if test_end else None
    ctx_buffer = base_params["context_steps"]

    for ticker in tqdm(tickers, desc="Lade Assets"):
        try:
            df = factory.load_or_download(ticker)
            if df.empty:
                skipped.append(ticker); continue

            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index.searchsorted(test_start_ts, side="left")
                lo = max(0, idx - ctx_buffer)
                df = df.iloc[lo:]
                if test_end_ts is not None:
                    df = df[df.index <= test_end_ts]
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime").reset_index(drop=True)
                idx = df["datetime"].searchsorted(test_start_ts, side="left")
                lo = max(0, idx - ctx_buffer)
                df = df.iloc[lo:]
                if test_end_ts is not None:
                    df = df[df["datetime"] <= test_end_ts]

            if df.empty:
                skipped.append(ticker); continue
            if "datetime" not in df.columns:
                df = df.reset_index().rename(columns={df.index.name: "datetime", "date": "datetime"})

            n_total = len(df)
            c, f, s = base_params["context_steps"], base_params["forecast_steps"], base_params["stride_steps"]
            if (n_total - c - f) // s + 1 <= 0:
                skipped.append(ticker); continue
            asset_data[ticker] = df
        except Exception as e:
            print(f"  skip {ticker}: {e}")
            skipped.append(ticker)

    if not asset_data:
        print("Keine gueltigen Assets.")
        return

    all_results = run_rolling_benchmark_multi_asset(
        predictor=predictor,
        asset_data_dict=asset_data,
        params=base_params,
        batch_size=batch_size,
        verbose=True,
    )

    final_summary = {}
    for ticker, result in all_results.items():
        if result:
            final_summary[ticker] = result["metrics"]
            with open(results_dir / f"result_{ticker}.json", "w") as f:
                json.dump(result, f, indent=4, default=str)

    with open(results_dir / "final_energy_study.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "data_source": "yfinance",
            "config_path": config_path,
            "random_seed": seed,
            "params": base_params,
            "batch_size": batch_size,
            "test_start": test_start,
            "test_end": test_end,
            "processing_time_seconds": time.time() - t0,
            "n_assets_processed": len(all_results),
            "n_assets_total": len(tickers),
            "summary": final_summary,
        }, f, indent=4, default=str)

    print(f"[{model}] Fertig in {time.time()-t0:.1f}s. Ergebnisse: {results_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["naive", "arima", "xgboost"], required=True)
    p.add_argument("--config", default="config/energy_assets_train.yaml")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--context", type=int, default=80)
    p.add_argument("--forecast", type=int, default=18)
    p.add_argument("--test-start", default="2021-01-01")
    p.add_argument("--test-end", default=None)
    p.add_argument("--results-root", default="04_benchmarks/results")
    p.add_argument("--xgb-model-path", default="04_benchmarks/models/xgb_global.pkl")
    p.add_argument("--batch-size", type=int, default=48)
    a = p.parse_args()
    main(a.model, a.config, a.seed, a.context, a.forecast, a.test_start, a.test_end,
         a.results_root, a.xgb_model_path, a.batch_size)
