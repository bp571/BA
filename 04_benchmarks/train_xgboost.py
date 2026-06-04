"""Trainiert globales XGBoost-Returns-Modell auf Daten vor test_start.

Aggregiert (X, y) ueber alle Assets, trainiert ein einziges Modell und
speichert es als Pickle fuer XGBoostPredictor.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.xgboost_wrapper import build_training_matrix  # noqa: E402
from data.factory import DataFactory  # noqa: E402


def main(
    config_path: str = "config/energy_assets_train.yaml",
    test_start: str = "2021-01-01",
    out_path: str = "04_benchmarks/models/xgb_global.pkl",
    n_estimators: int = 400,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    seed: int = 13,
) -> None:
    from xgboost import XGBRegressor

    factory = DataFactory(config_path=config_path)
    tickers = factory.get_tickers()
    test_start_ts = pd.Timestamp(test_start)

    X_all, y_all = [], []
    skipped = []
    for ticker in tqdm(tickers, desc="Build training matrix"):
        try:
            df = factory.load_or_download(ticker)
            if df.empty:
                skipped.append(ticker); continue
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[df.index < test_start_ts]
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df[df["datetime"] < test_start_ts]
            if df.empty or len(df) < 50:
                skipped.append(ticker); continue
            close = df["close"].values.astype(np.float64) if "close" in df.columns else df["Close"].values.astype(np.float64)
            close = close[np.isfinite(close) & (close > 0)]
            X, y = build_training_matrix(close)
            if len(X) > 0:
                X_all.append(X); y_all.append(y)
        except Exception as e:
            print(f"  skip {ticker}: {e}")
            skipped.append(ticker)

    if not X_all:
        raise RuntimeError("Keine Trainingsdaten gesammelt.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f"Trainings-Samples: {len(X):,} ueber {len(tickers) - len(skipped)} Assets")

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        objective="reg:squarederror",
        tree_method="hist",
    )
    model.fit(X, y)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"Modell gespeichert: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/energy_assets_train.yaml")
    p.add_argument("--test-start", default="2021-01-01")
    p.add_argument("--out", default="04_benchmarks/models/xgb_global.pkl")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    main(a.config, a.test_start, a.out, a.n_estimators, a.max_depth, a.learning_rate, a.seed)
