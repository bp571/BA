import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Union, Tuple, List

# --- BASISMETRIKEN ---

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None) -> float:
    """MASE: < 1 bedeutet besser als die naive Prognose (Random Walk)."""
    mae_forecast = mae(y_true, y_pred)
    naive_data = y_train if y_train is not None else y_true
    if len(naive_data) <= 1: return np.inf
    mae_naive = np.mean(np.abs(np.diff(naive_data)))
    return mae_forecast / mae_naive if mae_naive != 0 else np.inf

# --- FINANZSPEZIFISCHE METRIKEN (IC & RETURNS) ---

def calculate_log_returns(y: np.ndarray, anchor: float = None) -> np.ndarray:
    if anchor is not None:
        y = np.insert(y, 0, anchor)
    return np.log(y[1:] / y[:-1])

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson-Korrelation (IC) auf Log-Returns.
    
    WICHTIG: Dies ist ein TIME-SERIES IC - misst die Korrelation zwischen vorhergesagten
    und tatsächlichen Returns INNERHALB eines Assets über den Forecast-Horizont.
    Dies ist NICHT das Cross-Sectional IC aus der Finanzforschung.
    """
    if len(y_true) < 2: return 0.0
    ic, _ = pearsonr(y_true, y_pred)
    return ic

def rank_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman-Rangkorrelation (RankIC) auf Log-Returns.
    
    WICHTIG: Dies ist ein TIME-SERIES RankIC - misst die Rangkorrelation zwischen vorhergesagten
    und tatsächlichen Returns INNERHALB eines Assets über den Forecast-Horizont.
    Dies ist NICHT das Cross-Sectional RankIC, welches die Korrelation ÜBER Assets hinweg
    zu einem bestimmten Zeitpunkt misst (siehe evaluate_results.py für Cross-Sectional RankIC).
    """
    if len(y_true) < 2: return 0.0
    ric, _ = spearmanr(y_true, y_pred)
    return ric

# --- STATISTIK FÜR DIE THESIS (KONFIDENZINTERVALLE) ---

def calculate_ic_statistics(ic_values: List[float], prefix: str = "IC") -> dict:
    """
    Berechnet den Mittelwert und das 95% Konfidenzintervall für eine Liste von Werten.
    Korrigiert für Autokorrelation in überlappenden Rolling Windows.
    """
    if not ic_values: return {}
    ic_array = np.array(ic_values)
    mean_val = np.mean(ic_array)
    std_val = np.std(ic_array, ddof=1)  # Stichproben-Std mit Bessel-Korrektur
    n = len(ic_array)
    
    if n == 1:
        ci95_lower = ci95_upper = mean_val
    else:
        # Berechne effektive Stichprobengröße unter Berücksichtigung der Autokorrelation
        if n > 2:
            # Einfache Lag-1 Autokorrelationskorrektur
            autocorr = np.corrcoef(ic_array[:-1], ic_array[1:])[0,1] if n > 2 else 0
            autocorr = max(-0.99, min(0.99, autocorr))  # Clippen zur Vermeidung von Divisionsproblemen
            n_eff = n * (1 - autocorr) / (1 + autocorr)  # Effektive Stichprobengröße
            n_eff = max(1, min(n, n_eff))  # Begrenzung zwischen 1 und n
        else:
            n_eff = n
            
        se = std_val / np.sqrt(n_eff)
        # Verwende t-Verteilung mit effektiven Freiheitsgraden
        if n_eff < 30:
            from scipy.stats import t
            t_critical = t.ppf(0.975, df=max(1, int(n_eff-1)))  # 95% KI, zweiseitig
        else:
            t_critical = 1.96  # z-Score für große Stichproben
        
        ci95_lower = mean_val - t_critical * se
        ci95_upper = mean_val + t_critical * se

    return {
        f'{prefix}_Mean': mean_val,
        f'{prefix}_Std': std_val,
        f'{prefix}_CI95': (ci95_lower, ci95_upper),
        f'{prefix}_Count': n,
        f'{prefix}_Effective_N': n_eff if n > 1 else n
    }

# --- HAUPTFUNKTION FÜR DEN RUNNER ---

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Berechnet die Richtungsgenauigkeit: Prozentsatz korrekter Richtungsvorhersagen.
    Gibt Prozentsatz zurück (0-100%).
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    
    # Berechne tatsächliche und vorhergesagte Richtungen (aufwärts/abwärts)
    actual_directions = np.diff(y_true) > 0  # True für aufwärts, False für abwärts
    predicted_directions = np.diff(y_pred) > 0
    
    # Berechne Genauigkeit
    correct_predictions = actual_directions == predicted_directions
    accuracy = np.mean(correct_predictions) * 100.0  # Konvertiere zu Prozent
    
    return accuracy

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None) -> dict:
    """
    Berechnet Metriken für ein einzelnes Rolling Window (Time-Series).
    
    WICHTIG: Die IC/RankIC hier sind TIME-SERIES Metriken (innerhalb eines Assets).
    Für Cross-Sectional IC (über Assets hinweg) siehe evaluate_results.py.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 1. Metriken auf Preisebene
    results = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MASE': mase(y_true, y_pred, y_train) if y_train is not None and len(y_train) > 1 else np.nan,
        'Directional_Accuracy': directional_accuracy(y_true, y_pred)
    }
    
    # 2. Ankerpunkt für Renditen bestimmen (der letzte bekannte Preis P_t)
    anchor = y_train[-1] if y_train is not None and len(y_train) > 0 else None
    
    # 3. TIME-SERIES Metriken auf Return-Ebene mit Anker berechnen
    # Das berechnet: [log(P_t+1 / P_t), log(P_t+2 / P_t+1), ...]
    ret_true = calculate_log_returns(y_true, anchor=anchor)
    ret_pred = calculate_log_returns(y_pred, anchor=anchor)
    
    # Umbenannte Keys für Klarheit: Dies sind TIME-SERIES-Korrelationen
    results['IC_TimeSeries'] = information_coefficient(ret_true, ret_pred)
    results['RankIC_TimeSeries'] = rank_information_coefficient(ret_true, ret_pred)
    
    # 4. Lag-Check für Look-Ahead-Bias-Erkennung
    if len(ret_true) > 1:
        # Prüfe, ob Vorhersage für t+1 mit dem echten Return von t korreliert (Lagging)
        ic_lag, _ = pearsonr(ret_true[:-1], ret_pred[1:])
        results['Is_Lagging'] = bool(abs(ic_lag) > abs(results['IC_TimeSeries']))
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