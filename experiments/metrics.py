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
    """Calculate log returns with robust handling of negative/zero prices."""
    # Use relative minimum based on data range instead of absolute tiny value
    min_price = np.maximum(1e-6, np.nanmin(y[y > 0]) * 0.001) if np.any(y > 0) else 1e-6
    y_safe = np.where(y <= 0, min_price, y)
    return np.diff(np.log(y_safe))

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson Korrelation (IC) auf Log-Returns."""
    if len(y_true) < 2: return 0.0
    ic, _ = pearsonr(y_true, y_pred)
    return ic

def rank_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman Rang-Korrelation (RankIC) auf Log-Returns."""
    if len(y_true) < 2: return 0.0
    ric, _ = spearmanr(y_true, y_pred)
    return ric

# --- STATISTIK FÜR DIE THESIS (KONFIDENZINTERVALLE) ---

def calculate_ic_statistics(ic_values: List[float], prefix: str = "IC") -> dict:
    """
    Berechnet den Mittelwert und das 95% Konfidenzintervall für eine Liste von Werten.
    Verwendet t-Verteilung für kleine Stichproben (n < 30) für statistische Solidität.
    """
    if not ic_values: return {}
    ic_array = np.array(ic_values)
    mean_val = np.mean(ic_array)
    std_val = np.std(ic_array, ddof=1)  # Sample std with Bessel's correction
    n = len(ic_array)
    
    if n == 1:
        ci95_lower = ci95_upper = mean_val
    else:
        se = std_val / np.sqrt(n)
        # Use t-distribution for small samples, z-distribution for large samples
        if n < 30:
            from scipy.stats import t
            t_critical = t.ppf(0.975, df=n-1)  # 95% CI, two-tailed
        else:
            t_critical = 1.96  # z-score for large samples
        
        ci95_lower = mean_val - t_critical * se
        ci95_upper = mean_val + t_critical * se

    return {
        f'{prefix}_Mean': mean_val,
        f'{prefix}_Std': std_val,
        f'{prefix}_CI95': (ci95_lower, ci95_upper),
        f'{prefix}_Count': n
    }

# --- HAUPTFUNKTION FÜR DEN RUNNER ---

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy: percentage of correct directional predictions.
    Returns percentage (0-100%).
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    
    # Calculate actual and predicted directions (up/down)
    actual_directions = np.diff(y_true) > 0  # True for up, False for down
    predicted_directions = np.diff(y_pred) > 0
    
    # Calculate accuracy
    correct_predictions = actual_directions == predicted_directions
    accuracy = np.mean(correct_predictions) * 100.0  # Convert to percentage
    
    return accuracy

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None) -> dict:
    """Berechnet das kompakte Set an Metriken für einen Ticker."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Metriken auf Preis-Ebene
    results = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MASE': mase(y_true, y_pred, y_train),
        'Directional_Accuracy': directional_accuracy(y_true, y_pred)
    }
    
    # Metriken auf Return-Ebene (IC & RankIC)
    ret_true = calculate_log_returns(y_true)
    ret_pred = calculate_log_returns(y_pred)
    results['IC_Return'] = information_coefficient(ret_true, ret_pred)
    results['RankIC_Return'] = rank_information_coefficient(ret_true, ret_pred)
    
    # Corrected Lag-Check: Check if predictions are just lagged versions of actuals
    if len(ret_true) > 1:
        # Correct comparison: shifted actual vs current prediction
        ic_lag, _ = pearsonr(ret_true[:-1], ret_pred[1:])  # Fixed: correct time alignment
        results['Is_Lagging'] = bool(abs(ic_lag) > abs(results['IC_Return']))
    else:
        results['Is_Lagging'] = False
    
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