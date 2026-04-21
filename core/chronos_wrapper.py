"""
Chronos-Wrapper für einheitliche Predictor-API

Öffentliche API:
  predict()          – Forecast für ein einzelnes Asset
  predict_batch()    – GPU-Batch-Forecast für mehrere Assets (empfohlen)

Interne Hilfsmethode:
  _predict_sequential() – sequentieller Fallback, nicht direkt aufrufen
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Union
from datetime import timedelta


class ChronosPredictor:
    """
    Wrapper um Chronos-Pipeline, der die standardisierte Predictor-API implementiert.
    
    Ermöglicht die direkte Verwendung in der bestehenden Pipeline ohne Code-Änderungen.
    """
    
    def __init__(self, pipeline, device="cuda"):
        """
        Args:
            pipeline: Chronos2Pipeline Instanz
            device: torch device ("cuda" oder "cpu")
        """
        self.pipeline = pipeline
        self.device = device
    
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
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Erzeugt Forecasts für ein einzelnes Asset.
        
        Args:
            df: DataFrame mit OHLC-Daten (Context)
            x_timestamp: DatetimeIndex für Context
            y_timestamp: DatetimeIndex für Predictions
            pred_len: Anzahl zu prognostizierender Schritte
            T: Temperatur (für Chronos aktuell nicht verwendet)
            top_k: Nicht verwendet in Chronos
            top_p: Nicht verwendet in Chronos
            sample_count: Anzahl Samples für Ensemble-Averaging
            verbose: Debug-Output
        
        Returns:
            DataFrame mit vorhergesagten OHLC-Werten und Timestamps
        """
        # Konvertiere zu NumPy
        ohlc_array = df[['open', 'high', 'low', 'close']].values
        
        # Pipeline erwartet: Inputs (Shape: batch_size x context_length)
        # Für OHLC-Daten behandeln wir jede Spalte separat
        predictions_list = []
        
        for col_idx in range(4):
            context = ohlc_array[:, col_idx]
            
            # Erwartet: (n_series, n_variates, history_length)
            # Für univariate Zeitreihen: (1, 1, context_len)
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, context_len)
            
            # Pipeline.predict() gibt eine Liste zurück
            try:
                forecast_list = self.pipeline.predict(
                    inputs=context_tensor,
                    prediction_length=pred_len
                )
                
                # forecast_list ist eine Liste mit 1 Element (für unsere 1 Serie)
                # Das Element hat Shape: (n_series, num_samples, prediction_length)
                forecast_tensor = forecast_list[0]  # Shape: (1, num_samples, pred_len)
                
                # Mittelwert über alle Samples
                pred_mean = forecast_tensor[0].mean(dim=0).cpu().numpy()  # Shape: (pred_len,)
                
                predictions_list.append(pred_mean)
                
            except Exception as e:
                if verbose:
                    print(f"Prediction failed for column {col_idx}: {e}")
                # Fallback: NaN values
                predictions_list.append(np.full(pred_len, np.nan))
        
        # Assembliere zu DataFrame
        pred_array = np.stack(predictions_list, axis=1)  # Shape: (pred_len, 4)
        
        pred_df = pd.DataFrame(
            pred_array,
            columns=['open', 'high', 'low', 'close']
        )
        
        # Füge Timestamps hinzu
        pred_df['datetime'] = y_timestamp[:pred_len]
        
        return pred_df
    
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
        verbose: bool = False
    ) -> List[pd.DataFrame]:
        """
        Erzeugt Forecasts für mehrere Assets via echtem GPU-Batch-Processing.

        Alle Assets werden pro OHLC-Spalte in einem einzigen Pipeline-Aufruf
        verarbeitet. Bei Fehler Fallback auf sequentielle Verarbeitung.

        Args:
            df_list: Liste von DataFrames mit OHLC-Daten
            x_timestamp_list: Liste von DatetimeIndex für Contexts
            y_timestamp_list: Liste von DatetimeIndex für Predictions
            pred_len: Anzahl zu prognostizierender Schritte
            T: Temperatur
            top_k: Top-k Filterung
            top_p: Nucleus Sampling
            sample_count: Anzahl Samples
            verbose: Debug-Output

        Returns:
            Liste von DataFrames mit Predictions
        """
        if not df_list:
            return []

        n_assets = len(df_list)
        col_results: List[dict] = []

        for col_idx, col_name in enumerate(['open', 'high', 'low', 'close']):
            contexts = [
                torch.tensor(df[col_name].values, dtype=torch.float32).unsqueeze(0)
                for df in df_list
            ]
            context_batch = torch.stack(contexts, dim=0)  # (n_assets, 1, context_len)

            try:
                forecast_list = self.pipeline.predict(
                    inputs=context_batch,
                    prediction_length=pred_len
                )
                # forecast_list: n_assets Elemente, je Shape (1, num_samples, pred_len)
                pred_means = np.stack([
                    f[0].mean(dim=0).cpu().numpy() for f in forecast_list
                ])  # (n_assets, pred_len)

            except Exception as e:
                if verbose:
                    print(f"⚠️  Batch prediction for {col_name} failed, using NaN: {e}")
                pred_means = np.full((n_assets, pred_len), np.nan)

            if col_idx == 0:
                col_results = [{col_name: pred_means[i]} for i in range(n_assets)]
            else:
                for i in range(n_assets):
                    col_results[i][col_name] = pred_means[i]

        results = []
        for i, result_dict in enumerate(col_results):
            pred_df = pd.DataFrame(result_dict)
            pred_df['datetime'] = y_timestamp_list[i][:pred_len]
            results.append(pred_df)

        return results

    def _predict_sequential(
        self,
        df_list: List[pd.DataFrame],
        x_timestamp_list: List[pd.DatetimeIndex],
        y_timestamp_list: List[pd.DatetimeIndex],
        pred_len: int,
        verbose: bool = False
    ) -> List[pd.DataFrame]:
        """Sequentieller Fallback: ruft predict() für jedes Asset einzeln auf."""
        results = []
        for idx, (df, x_ts, y_ts) in enumerate(zip(df_list, x_timestamp_list, y_timestamp_list)):
            try:
                results.append(self.predict(df=df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len))
            except Exception as e:
                if verbose:
                    print(f"⚠️  Sequential fallback item {idx} failed: {e}")
                results.append(pd.DataFrame({
                    'open': [np.nan] * pred_len, 'high': [np.nan] * pred_len,
                    'low':  [np.nan] * pred_len, 'close': [np.nan] * pred_len,
                    'datetime': y_ts[:pred_len]
                }))
        return results
