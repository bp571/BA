"""Statistische Baseline-Predictors mit Chronos/Kronos-kompatibler API.

Enthält:
    NaivePredictor: Random Walk mit Drift (pred[h] = last + h * mean_diff).
        Bewusst NICHT konstanter RW: Time-Series-RankIC waere sonst NaN
        (Spearman einer konstanten Serie ist undefiniert).
    ARIMAPredictor: SARIMAX per-Window-Fit auf Context-Close-Werte.
"""

from __future__ import annotations

from typing import List
import warnings

import numpy as np
import pandas as pd


def _ohlc_from_close(close: np.ndarray) -> dict:
    return {"open": close, "high": close, "low": close, "close": close}


class NaivePredictor:
    """Random Walk mit Drift auf Close-Preisen.

    drift = mean(diff(close_context))
    pred[h] = last_close + (h+1) * drift,  h = 0..pred_len-1
    OHLC werden alle gleich gesetzt (close), da BatchWindowPredictor nur
    auf 'close' zugreift.
    """

    def __init__(self, use_drift: bool = True) -> None:
        self.use_drift = use_drift

    def _forecast_close(self, context_close: np.ndarray, pred_len: int) -> np.ndarray:
        last = float(context_close[-1])
        if self.use_drift and len(context_close) > 1:
            drift = float(np.mean(np.diff(context_close)))
        else:
            drift = 0.0
        steps = np.arange(1, pred_len + 1, dtype=float)
        return last + drift * steps

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
        out = pd.DataFrame(_ohlc_from_close(close_pred))
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


class ARIMAPredictor:
    """ARIMA per-Window-Fit auf Context-Close-Werten.

    Default (1,1,1) - schnell und robust. Bei Fit-Fehler Fallback auf
    Naive-Drift, damit der Rolling-Loop nicht abbricht.
    """

    def __init__(self, order: tuple = (1, 1, 1)) -> None:
        self.order = order
        self._fallback = NaivePredictor(use_drift=True)

    def _forecast_close(self, context_close: np.ndarray, pred_len: int) -> np.ndarray:
        try:
            from statsmodels.tsa.arima.model import ARIMA  # local import: optional dep
        except ImportError as e:
            raise RuntimeError("statsmodels nicht installiert: pip install statsmodels") from e

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ARIMA(context_close, order=self.order)
                fit = model.fit(method_kwargs={"warn_convergence": False})
                fc = np.asarray(fit.forecast(steps=pred_len))
                if not np.all(np.isfinite(fc)):
                    raise ValueError("ARIMA forecast non-finite")
                return fc
            except Exception:
                return self._fallback._forecast_close(context_close, pred_len)

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
        out = pd.DataFrame(_ohlc_from_close(close_pred))
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
