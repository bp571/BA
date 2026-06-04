import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from core.model_loader import load_chronos_predictor
from core.reproducibility import set_all_seeds
from data.factory import DataFactory
from experiments.runner import run_rolling_benchmark_multi_asset


def main(
    config_path: str = "config/energy_assets_holdout.yaml",
    seed: int = 13,
    context: int = 80,
    forecast: int = 18,
    test_start: str = "2021-01-01",
    test_end: str | None = None,
    results_subdir: str | None = None,
) -> None:
    set_all_seeds(seed=seed)
    t0 = time.time()

    factory = DataFactory(config_path=config_path)
    predictor = load_chronos_predictor()

    subdir = results_subdir if results_subdir else f"seed_{seed}"
    results_dir = Path("01_model_comparison/results/chronos") / subdir
    results_dir.mkdir(exist_ok=True, parents=True)

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

    # Context-buffer slicing: gleiche Logik wie run_benchmark.py
    asset_data: dict = {}
    skipped = []
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end) if test_end else None
    ctx_buffer = context

    for ticker in tqdm(tickers, desc="Lade Assets"):
        try:
            df = factory.load_or_download(ticker)
            if df.empty:
                skipped.append(ticker)
                continue

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
                skipped.append(ticker)
                continue
            if "datetime" not in df.columns:
                df = df.reset_index().rename(
                    columns={df.index.name: "datetime", "date": "datetime"}
                )

            n_total = len(df)
            c, f, s = base_params["context_steps"], base_params["forecast_steps"], base_params["stride_steps"]
            if (n_total - c - f) // s + 1 <= 0:
                skipped.append(ticker)
                continue
            asset_data[ticker] = df
        except Exception as e:
            print(f"  skip {ticker}: {e}")
            skipped.append(ticker)

    if not asset_data:
        print("Keine gültigen Assets.")
        return

    all_results = run_rolling_benchmark_multi_asset(
        predictor=predictor,
        asset_data_dict=asset_data,
        params=base_params,
        batch_size=48,
        verbose=True,
    )

    final_summary = {}
    for ticker, result in all_results.items():
        if result:
            final_summary[ticker] = result["metrics"]
            with open(results_dir / f"result_{ticker}.json", "w") as fh:
                json.dump(result, fh, indent=4, default=str)

    with open(results_dir / "final_energy_study.json", "w") as fh:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "Chronos",
            "data_source": "yfinance",
            "config_path": config_path,
            "random_seed": seed,
            "params": base_params,
            "test_start": test_start,
            "processing_time_seconds": time.time() - t0,
            "n_assets_processed": len(all_results),
            "n_assets_total": len(tickers),
            "summary": final_summary,
        }, fh, indent=4, default=str)

    print(f"Fertig in {time.time()-t0:.1f}s. Ergebnisse: {results_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--config", type=str, default="config/energy_assets_holdout.yaml")
    p.add_argument("--context", type=int, default=80)
    p.add_argument("--forecast", type=int, default=18)
    p.add_argument("--test-start", type=str, default="2021-01-01")
    p.add_argument("--test-end", type=str, default=None)
    p.add_argument("--results-subdir", type=str, default=None)
    a = p.parse_args()
    main(a.config, a.seed, a.context, a.forecast, a.test_start, a.test_end, a.results_subdir)
