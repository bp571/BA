import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Union, Tuple, List

# --- BASIS-METRIKEN ---

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None) -> float:
    """MASE: < 1 bedeutet besser als der Naive Forecast (Random Walk)."""
    mae_forecast = mae(y_true, y_pred)
    naive_data = y_train if y_train is not None else y_true
    if len(naive_data) <= 1: return np.inf
    mae_naive = np.mean(np.abs(np.diff(naive_data)))
    return mae_forecast / mae_naive if mae_naive != 0 else np.inf

# --- FINANZ-SPEZIFISCHE METRIKEN (IC & RETURNS) ---

def calculate_log_returns(y: np.ndarray) -> np.ndarray:
    y_safe = np.where(y <= 0, 1e-8, y) 
    return np.diff(np.log(y_safe))

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson Korrelation (IC) auf Log-Returns."""
    if len(y_true) < 2: return 0.0
    ic, _ = pearsonr(y_true, y_pred)
    return ic

# --- STATISTIK FÜR DIE THESIS (KONFIDENZINTERVALLE) ---

def calculate_ic_statistics(ic_values: List[float]) -> dict:
    """
    Berechnet den Mittelwert des IC und das 95% Konfidenzintervall.
    Wichtig für den Beweis, dass Ergebnisse nicht zufällig sind.
    """
    if not ic_values: return {}
    ic_array = np.array(ic_values)
    mean_ic = np.mean(ic_array)
    std_ic = np.std(ic_array)
    n = len(ic_array)
    
    # Standardfehler und 95% Konfidenzintervall (z = 1.96)
    se = std_ic / np.sqrt(n) if n > 0 else 0
    ci95_lower = mean_ic - 1.96 * se
    ci95_upper = mean_ic + 1.96 * se

    return {
        'IC_Mean': mean_ic,
        'IC_Std': std_ic,
        'IC_CI95': (ci95_lower, ci95_upper),
        'Count': n
    }

# --- HAUPTFUNKTION FÜR DEN RUNNER ---

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None) -> dict:
    """Berechnet das kompakte Set an Metriken für einen Ticker."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Metriken auf Preis-Ebene
    results = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MASE': mase(y_true, y_pred, y_train)
    }
    
    # Metriken auf Return-Ebene (IC)
    ret_true = calculate_log_returns(y_true)
    ret_pred = calculate_log_returns(y_pred)
    results['IC_Return'] = information_coefficient(ret_true, ret_pred)
    
    # Lag-Check (Beweis gegen triviales Hinterherlaufen)
    if len(ret_true) > 1:
        ic_lag, _ = pearsonr(ret_true[:-1], ret_pred[1:])
        results['Is_Lagging'] = bool(ic_lag > results['IC_Return'])
    
    return results

def calculate_asset_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Berechnet das 95% Konfidenzintervall der Vorhersagefehler für ein einzelnes Asset.
    Zeigt an, wie stark die Vorhersage pro Asset typischerweise schwankt.
    """
    errors = y_true - y_pred
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # 95% Intervall (1.96 * Standardabweichung der Fehler)
    ci95_margin = 1.96 * std_error
    
    return {
        'Error_Mean': mean_error,
        'Error_Std': std_error,
        'Error_CI95_Lower': mean_error - ci95_margin,
        'Error_CI95_Upper': mean_error + ci95_margin
    }