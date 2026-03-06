"""
Chronos Wrapper für API-Kompatibilität mit KronosPredictor

Dieser Wrapper ermöglicht die Verwendung von Chronos mit derselben
API wie Kronos, sodass die bestehende Pipeline ohne Änderungen
mit beiden Modellen arbeiten kann.
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Union
from datetime import timedelta


class ChronosPredictor:
    """
    Wrapper um Chronos-Pipeline, der die gleiche API wie KronosPredictor bietet.
    
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
            df: DataFrame mit OHLC-Daten (context)
            x_timestamp: DatetimeIndex für Context
            y_timestamp: DatetimeIndex für Predictions
            pred_len: Anzahl zu prognostizierender Schritte
            T: Temperature (für Chronos wird dies als limit_prediction_length verwendet)
            top_k: Nicht verwendet in Chronos
            top_p: Nicht verwendet in Chronos
            sample_count: Anzahl Samples für Ensemble-Averaging
            verbose: Debug-Output
        
        Returns:
            DataFrame mit predicted OHLC-Werten und Timestamps
        """
        # Konvertiere zu numpy für Chronos
        ohlc_array = df[['open', 'high', 'low', 'close']].values
        
        # Chronos2Pipeline erwartet: inputs (shape: batch_size x context_length)
        # Für OHLC-Daten behandeln wir jede Spalte separat
        predictions_list = []
        
        for col_idx in range(4):
            context = ohlc_array[:, col_idx]
            
            # Chronos2 erwartet: (n_series, n_variates, history_length)
            # Für univariate Zeitreihen: (1, 1, context_len)
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, context_len)
            
            # Chronos2Pipeline.predict() gibt eine Liste zurück
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
        Erzeugt Forecasts für mehrere Assets in einem Batch.
        
        Chronos unterstützt nativen Batch-Processing, was die Verarbeitung beschleunigt.
        
        Args:
            df_list: Liste von DataFrames mit OHLC-Daten
            x_timestamp_list: Liste von DatetimeIndex für Contexts
            y_timestamp_list: Liste von DatetimeIndex für Predictions
            pred_len: Anzahl zu prognostizierender Schritte
            T: Temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
            sample_count: Anzahl Samples
            verbose: Debug-Output
        
        Returns:
            Liste von DataFrames mit Predictions
        """
        if not df_list:
            return []
        
        # Für echtes Batch-Processing: Alle Zeitreihen in einen Batch packen
        # Aber da wir 4 Spalten (OHLC) haben, machen wir es pro Spalte
        
        results = []
        
        for idx, (df, x_timestamp, y_timestamp) in enumerate(zip(df_list, x_timestamp_list, y_timestamp_list)):
            try:
                pred_df = self.predict(
                    df=df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=T,
                    top_k=top_k,
                    top_p=top_p,
                    sample_count=sample_count,
                    verbose=verbose
                )
                results.append(pred_df)
            except Exception as e:
                if verbose:
                    print(f"⚠️  Batch item {idx} failed: {e}")
                # Erstelle leeres Ergebnis mit korrekter Struktur
                empty_df = pd.DataFrame({
                    'open': [np.nan] * pred_len,
                    'high': [np.nan] * pred_len,
                    'low': [np.nan] * pred_len,
                    'close': [np.nan] * pred_len,
                    'datetime': y_timestamp[:pred_len]
                })
                results.append(empty_df)
        
        return results
    
    def predict_batch_optimized(
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
        Optimierte Batch-Version, die alle Assets gleichzeitig verarbeitet.
        
        Nutzt Chronos' native Batch-Capability für maximale Performance.
        """
        if not df_list:
            return []
        
        n_assets = len(df_list)
        results = []
        
        # Verarbeite jede OHLC-Spalte separat im Batch
        for col_name in ['open', 'high', 'low', 'close']:
            col_idx = ['open', 'high', 'low', 'close'].index(col_name)
            
            # Sammle alle Zeitreihen für diese Spalte
            contexts = []
            for df in df_list:
                context = df[col_name].values
                # Chronos2 erwartet: (n_series, n_variates, history_length)
                # Füge variate dimension hinzu: (1, context_len)
                contexts.append(torch.tensor(context, dtype=torch.float32).unsqueeze(0))
            
            # Stack zu Batch: (n_assets, 1, context_len)
            context_batch = torch.stack(contexts, dim=0)
            
            # Batch Prediction
            try:
                forecast_list = self.pipeline.predict(
                    inputs=context_batch,  # Shape: (n_assets, 1, context_len)
                    prediction_length=pred_len
                )
                
                # forecast_list ist eine Liste mit n_assets Elementen
                # Jedes Element hat Shape: (1, num_samples, pred_len)
                pred_means_list = []
                for forecast_tensor in forecast_list:
                    # forecast_tensor Shape: (1, num_samples, pred_len)
                    pred_mean = forecast_tensor[0].mean(dim=0).cpu().numpy()  # Shape: (pred_len,)
                    pred_means_list.append(pred_mean)
                
                pred_means = np.stack(pred_means_list, axis=0)  # Shape: (n_assets, pred_len)
                
                # Speichere diese Spalte für jedes Asset
                if col_idx == 0:  # Erste Spalte: Initialisiere results
                    for i in range(n_assets):
                        results.append({col_name: pred_means[i]})
                else:  # Weitere Spalten: Füge hinzu
                    for i in range(n_assets):
                        results[i][col_name] = pred_means[i]
                        
            except Exception as e:
                if verbose:
                    print(f"⚠️  Batch prediction for {col_name} failed: {e}")
                # Fallback zu NaN
                if col_idx == 0:
                    for i in range(n_assets):
                        results.append({col_name: np.full(pred_len, np.nan)})
                else:
                    for i in range(n_assets):
                        results[i][col_name] = np.full(pred_len, np.nan)
        
        # Konvertiere zu DataFrames
        final_results = []
        for i, result_dict in enumerate(results):
            pred_df = pd.DataFrame(result_dict)
            pred_df['datetime'] = y_timestamp_list[i][:pred_len]
            final_results.append(pred_df)
        
        return final_results
