import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Union, Tuple


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Berechnet Mean Absolute Error (MAE)
    
    Args:
        y_true: Tatsächliche Werte
        y_pred: Vorhergesagte Werte
    
    Returns:
        MAE Wert
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Berechnet Root Mean Square Error (RMSE)
    
    Args:
        y_true: Tatsächliche Werte
        y_pred: Vorhergesagte Werte
    
    Returns:
        RMSE Wert
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Berechnet Mean Absolute Percentage Error (MAPE)
    
    Args:
        y_true: Tatsächliche Werte
        y_pred: Vorhergesagte Werte
    
    Returns:
        MAPE Wert in Prozent
    """
    # Vermeidet Division durch Null
    mask = y_true != 0
    if not mask.any():
        return np.inf
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Berechnet Information Coefficient (IC) - Korrelation zwischen predicted und actual values
    
    Args:
        y_true: Tatsächliche Werte
        y_pred: Vorhergesagte Werte
    
    Returns:
        Tuple (IC, p-value)
    """
    try:
        ic, p_value = pearsonr(y_true, y_pred)
        return ic, p_value
    except:
        return 0.0, 1.0

def rank_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Berechnet Spearman-Rangkorrelation (Rank IC)
    """
    try:
        # spearmanr berechnet die Korrelation der Ränge
        ric, p_value = spearmanr(y_true, y_pred)
        return ric, p_value
    except:
        return 0.0, 1.0

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Berechnet Directional Accuracy - Prozentsatz korrekter Richtungsvorhersagen
    
    Args:
        y_true: Tatsächliche Werte
        y_pred: Vorhergesagte Werte
    
    Returns:
        Directional Accuracy in Prozent
    """
    if len(y_true) <= 1:
        return 0.0
    
    # Berechne Änderungen
    true_changes = np.diff(y_true)
    pred_changes = np.diff(y_pred)
    
    # Prüfe, ob Richtung korrekt vorhergesagt wurde
    correct_direction = np.sign(true_changes) == np.sign(pred_changes)
    
    return np.mean(correct_direction) * 100


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Berechnet alle Metriken für gegebene Vorhersagen
    
    Args:
        y_true: Tatsächliche Werte
        y_pred: Vorhergesagte Werte
    
    Returns:
        Dictionary mit allen Metriken
    """
    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'IC': np.nan,
            'IC_pvalue': np.nan,
            'Directional_Accuracy': np.nan,
            'Count': 0
        }
    
    ic, ic_pvalue = information_coefficient(y_true_clean, y_pred_clean)
    ric, p_value = rank_information_coefficient(y_true_clean, y_pred_clean)
    
    return {
        'MAE': mae(y_true_clean, y_pred_clean),
        'RMSE': rmse(y_true_clean, y_pred_clean),
        'MAPE': mape(y_true_clean, y_pred_clean),
        'IC': ic,
        'IC_pvalue': ic_pvalue,
        'RankIC': ric,
        'RankIC_pvalue': p_value,
        'Directional_Accuracy': directional_accuracy(y_true_clean, y_pred_clean),
        'Count': len(y_true_clean)
    }


def load_and_calculate_metrics_from_csv(csv_path: str) -> dict:
    """
    Lädt CSV mit Vorhersagen und berechnet Metriken
    
    Args:
        csv_path: Pfad zur CSV-Datei mit Spalten 'actual_value' und 'predicted_value'
    
    Returns:
        Dictionary mit allen Metriken
    """
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['actual_value', 'predicted_value']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV muss Spalten {required_columns} enthalten")
        
        return calculate_all_metrics(df['actual_value'].values, df['predicted_value'].values)
    
    except Exception as e:
        print(f"Fehler beim Laden von {csv_path}: {e}")
        return {
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'IC': np.nan,
            'IC_pvalue': np.nan,
            'Directional_Accuracy': np.nan,
            'Count': 0,
            'Error': str(e)
        }