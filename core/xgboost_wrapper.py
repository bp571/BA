"""XGBoost-Predictor mit Chronos/Kronos-kompatibler API.

Globales (asset-agnostisches) Modell auf Log-Return-Features.
Recursive Multi-Step-Forecast: nach jedem 1-Step-Predict werden
Features fortgeschrieben und das Modell erneut aufgerufen.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import pickle

import numpy as np
import pandas as pd


LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
ROLL_WINDOWS = (5, 10)
N_FEATURES = len(LAGS) + 2 * len(ROLL_WINDOWS)  # lags + roll_mean + roll_std


def build_feature_row(returns: np.ndarray) -> np.ndarray:
    """Berechnet Feature-Vektor aus letzten Log-Returns.

    returns: 1D-Array, mind. max(LAGS, max(ROLL_WINDOWS)) Werte.
    Reihenfolge: lag_1..lag_K, roll_mean_w1, roll_mean_w2, roll_std_w1, roll_std_w2.
    """
    feats = []
    for lag in LAGS:
        feats.append(returns[-lag])
    for w in ROLL_WINDOWS:
        feats.append(float(np.mean(returns[-w:])))
    for w in ROLL_WINDOWS:
        feats.append(float(np.std(returns[-w:], ddof=0)))
    return np.asarray(feats, dtype=np.float64)


def build_training_matrix(close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Aus einer Close-Serie -> (X, y) fuer 1-Step-Return-Forecast."""
    rets = np.diff(np.log(close))
    min_hist = max(max(LAGS), max(ROLL_WINDOWS))
    if len(rets) <= min_hist:
        return np.empty((0, N_FEATURES)), np.empty((0,))
    X, y = [], []
    for t in range(min_hist, len(rets)):
        X.append(build_feature_row(rets[:t]))
        y.append(rets[t])
    return np.asarray(X), np.asarray(y)


class XGBoostPredictor:
    """Globales XGBoost-Modell auf Log-Return-Features (asset-agnostisch).

    Recursive Forecast:
        1. Berechne Features aus Context-Log-Returns.
        2. Predict r_hat -> Close_hat = Close_last * exp(r_hat).
        3. Append r_hat an Return-Serie, recompute Features, wiederhole.
    """

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost-Modell nicht gefunden: {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def _forecast_close(self, context_close: np.ndarray, pred_len: int) -> np.ndarray:
        rets = np.diff(np.log(context_close)).tolist()
        last_close = float(context_close[-1])
        min_hist = max(max(LAGS), max(ROLL_WINDOWS))
        if len(rets) < min_hist:
            # Fallback: Random Walk konstant (sollte bei context=80 nie passieren)
            return np.full(pred_len, last_close)

        preds_close = []
        cur_close = last_close
        for _ in range(pred_len):
            feats = build_feature_row(np.asarray(rets))
            r_hat = float(self.model.predict(feats.reshape(1, -1))[0])
            cur_close = cur_close * float(np.exp(r_hat))
            preds_close.append(cur_close)
            rets.append(r_hat)
        return np.asarray(preds_close)

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.DatetimeIndex,
        y_timestamp: pd.DatetimeIndex,
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = False,
    ) -> pd.DataFrame:
        close_pred = self._forecast_close(df["close"].values, pred_len)
        out = pd.DataFrame({
            "open": close_pred,
            "high": close_pred,
            "low": close_pred,
            "close": close_pred,
        })
        out["datetime"] = y_timestamp[:pred_len].values
        return out

    def predict_batch(
        self,
        df_list: List[pd.DataFrame],
        x_timestamp_list: List[pd.DatetimeIndex],
        y_timestamp_list: List[pd.DatetimeIndex],
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = False,
    ) -> List[pd.DataFrame]:
        return [
            self.predict(df, x_ts, y_ts, pred_len)
            for df, x_ts, y_ts in zip(df_list, x_timestamp_list, y_timestamp_list)
        ]
