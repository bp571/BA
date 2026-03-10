"""
Batch Window Predictor für Multi-Asset Forecasting

Dieser Wrapper ermöglicht die parallele Verarbeitung von Rolling Windows
aus mehreren Assets durch die Batch-Prediction Capability der Forecast-Modelle.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class BatchWindowPredictor:
    """
    Wrapper für batch-basierte Rolling Window Predictions.
    
    Sammelt Windows von verschiedenen Assets und führt sie parallel
    durch die predict_batch() Methode des Predictors aus.
    Funktioniert mit jedem Predictor der die predict() und predict_batch() API implementiert.
    """
    
    def __init__(self, predictor, verbose=False):
        """
        Args:
            predictor: Predictor Instanz mit predict() und predict_batch() API
            verbose: Wenn True, zeigt Batch-Processing Details an
        """
        self.predictor = predictor
        self.verbose = verbose
    
    def predict_windows_batch(
        self, 
        windows_data: List[Dict],
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1
    ) -> Dict[Tuple[str, int], pd.DataFrame]:
        """
        Führt Batch-Predictions für mehrere Rolling Windows durch.
        
        Args:
            windows_data: Liste von Window-Definitionen, jedes Dict enthält:
                - 'ticker': str - Asset-Identifikator
                - 'window_id': int - Rolling Window Index
                - 'context_data': DataFrame mit OHLC-Daten
                - 'context_datetime': DatetimeIndex für Context
                - 'target_datetime': DatetimeIndex für Prediction
                - 'forecast_steps': int - Anzahl zu prognostizierender Schritte
            T: Temperatur für Sampling
            top_k: Top-k Filterung
            top_p: Nucleus Sampling Schwellenwert
            sample_count: Anzahl Samples für Averaging
        
        Returns:
            Dict mit (ticker, window_id) als Key und Predictions-DataFrame als Value
        """
        if not windows_data:
            return {}
        
        # Gruppiere Windows nach (context_steps, forecast_steps) für Batch-Kompatibilität
        groups = self._group_compatible_windows(windows_data)
        
        results = {}
        total_groups = len(groups)
        
        for group_idx, ((context_len, forecast_len), group_windows) in enumerate(groups.items()):
            if self.verbose:
                print(f"Processing batch group {group_idx + 1}/{total_groups}: "
                      f"context={context_len}, forecast={forecast_len}, "
                      f"n_windows={len(group_windows)}")
            
            # Batch-Prediction für diese Gruppe
            try:
                batch_results = self._predict_batch_group(
                    group_windows, 
                    forecast_len,
                    T=T,
                    top_k=top_k,
                    top_p=top_p,
                    sample_count=sample_count
                )
                results.update(batch_results)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Batch group failed: {e}")
                # Fallback zu einzelnen Predictions
                for window in group_windows:
                    try:
                        single_result = self._predict_single_window(
                            window,
                            T=T,
                            top_k=top_k,
                            top_p=top_p,
                            sample_count=sample_count
                        )
                        key = (window['ticker'], window['window_id'])
                        results[key] = single_result
                    except Exception as single_e:
                        if self.verbose:
                            print(f"⚠️  Window {key} failed: {single_e}")
                        continue
        
        return results
    
    def _group_compatible_windows(
        self,
        windows_data: List[Dict]
    ) -> Dict[Tuple[int, int], List[Dict]]:
        """
        Gruppiert Windows nach kompatiblen Dimensionen für Batch-Processing.
        
        predict_batch() erfordert gleiche Längen für alle Inputs.
        
        Returns:
            Dict mit (context_length, forecast_length) als Key
        """
        groups = defaultdict(list)
        
        for window in windows_data:
            context_len = len(window['context_data'])
            forecast_len = window['forecast_steps']
            key = (context_len, forecast_len)
            groups[key].append(window)
        
        return dict(groups)
    
    def _predict_batch_group(
        self,
        group_windows: List[Dict],
        forecast_len: int,
        T: float,
        top_k: int,
        top_p: float,
        sample_count: int
    ) -> Dict[Tuple[str, int], pd.DataFrame]:
        """
        Führt Batch-Prediction für eine Gruppe kompatibler Windows durch.
        """
        # Bereite Batch-Inputs vor
        df_list = []
        x_timestamp_list = []
        y_timestamp_list = []
        
        for window in group_windows:
            df_list.append(window['context_data'][['open', 'high', 'low', 'close']])
            x_timestamp_list.append(window['context_datetime'])
            y_timestamp_list.append(window['target_datetime'])
        
        # Batch-Prediction
        pred_dfs = self.predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_timestamp_list,
            y_timestamp_list=y_timestamp_list,
            pred_len=forecast_len,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False  # Detaillierter Output nur auf höherer Ebene
        )
        
        # Mappe Ergebnisse zurück zu (ticker, window_id)
        results = {}
        for i, window in enumerate(group_windows):
            key = (window['ticker'], window['window_id'])
            results[key] = pred_dfs[i]
        
        return results
    
    def _predict_single_window(
        self,
        window: Dict,
        T: float,
        top_k: int,
        top_p: float,
        sample_count: int
    ) -> pd.DataFrame:
        """
        Fallback für einzelne Window-Prediction, wenn Batch fehlschlägt.
        """
        pred_df = self.predictor.predict(
            df=window['context_data'][['open', 'high', 'low', 'close']],
            x_timestamp=window['context_datetime'],
            y_timestamp=window['target_datetime'],
            pred_len=window['forecast_steps'],
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False
        )
        return pred_df


def create_window_definition(
    ticker: str,
    window_id: int,
    context_data: pd.DataFrame,
    context_datetime: pd.DatetimeIndex,
    target_datetime: pd.DatetimeIndex,
    forecast_steps: int
) -> Dict:
    """
    Hilfsfunktion zum Erstellen einer Window-Definition für BatchWindowPredictor.
    
    Args:
        ticker: Asset-Identifikator
        window_id: Rolling Window Index
        context_data: DataFrame mit historischen OHLC-Daten
        context_datetime: DatetimeIndex für Context
        target_datetime: DatetimeIndex für Predictions
        forecast_steps: Anzahl Forecast-Schritte
    
    Returns:
        Dict mit Window-Definition
    """
    return {
        'ticker': ticker,
        'window_id': window_id,
        'context_data': context_data,
        'context_datetime': context_datetime,
        'target_datetime': target_datetime,
        'forecast_steps': forecast_steps
    }
