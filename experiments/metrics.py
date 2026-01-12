import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Union, Tuple

def calculate_log_returns(y: np.ndarray) -> np.ndarray:
    """
    Berechnet Log-Returns: ln(p_t / p_{t-1}).
    Erstes Element wird entfernt, da kein Vorwert existiert.
    """
    # Verhindert Fehler bei Werten <= 0 (wichtig für Energie-Daten)
    y_safe = np.where(y <= 0, 1e-8, y) 
    return np.diff(np.log(y_safe))




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


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Berechnet weighted Mean Absolute Percentage Error (wMAPE)
    
    VORTEIL: Skaleninvariant und robust gegen Werte nahe Null
    FORMEL: wMAPE = sum(|actual - predicted|) / sum(|actual|) * 100
    
    Args:
        y_true: Tatsächliche Werte
        y_pred: Vorhergesagte Werte
    
    Returns:
        wMAPE Wert in Prozent
    """
    # Berechne Summen für gewichteten Ansatz
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    
    # Vermeide Division durch Null
    if denominator == 0:
        return np.inf
    
    return (numerator / denominator) * 100


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None,
         seasonal_period: int = 1) -> float:
    """
    Berechnet Mean Absolute Scaled Error (MASE)
    
    VORTEIL: Skaleninvariant, vergleichbar zwischen verschiedenen Assets
    FORMEL: MASE = MAE / MAE_naive
    
    Args:
        y_true: Tatsächliche Werte
        y_pred: Vorhergesagte Werte
        y_train: Trainingsdaten für Naive Baseline (optional)
        seasonal_period: Seasonal lag für naive forecast (default=1 für random walk)
    
    Returns:
        MASE Wert (< 1 = besser als naive, > 1 = schlechter als naive)
    """
    # Berechne MAE der Vorhersage
    mae_forecast = mae(y_true, y_pred)
    
    # Bestimme Daten für naive Baseline
    if y_train is not None and len(y_train) > seasonal_period:
        # Nutze Trainingsdaten für naive forecast
        naive_data = y_train
    else:
        # Fallback: Nutze y_true für in-sample naive forecast
        naive_data = y_true
    
    if len(naive_data) <= seasonal_period:
        return np.inf
    
    # Berechne MAE der naiven Vorhersage (Seasonal naive)
    naive_forecast = naive_data[:-seasonal_period]
    naive_actual = naive_data[seasonal_period:]
    
    if len(naive_forecast) == 0:
        return np.inf
        
    mae_naive = np.mean(np.abs(naive_actual - naive_forecast))
    
    # Vermeide Division durch Null
    if mae_naive == 0:
        return 0.0 if mae_forecast == 0 else np.inf
    
    return mae_forecast / mae_naive


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


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None) -> dict:
    """
    Berechnet alle Metriken inkl. Log-Return IC und Lag-Check zur Detektion von 
    'Hinterherlaufen' (Lagging).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return {'Status': 'Error: Not enough data'}

    # 1. IC auf Log-Returns (Wissenschaftlicher Standard)
    ret_true = calculate_log_returns(y_true_clean)
    ret_pred = calculate_log_returns(y_pred_clean)
    
    ic, ic_pvalue = information_coefficient(ret_true, ret_pred)
    ric, p_value = rank_information_coefficient(ret_true, ret_pred)

    # 2. NEU: Lag-Check (Beweis gegen 'Hinterherlaufen')
    # Wir prüfen, ob die Vorhersage (heute) höher mit der Realität (gestern) korreliert
    ic_lag = np.nan
    is_lagging = False
    if len(ret_true) > 1:
        # Pearson-Korrelation: Predicted Return[t] vs Actual Return[t-1]
        ic_lag, _ = pearsonr(ret_true[:-1], ret_pred[1:])
        is_lagging = ic_lag > ic  # True, wenn das Modell dem Markt nur folgt

    # 3. Skaleninvariante Fehler-Metriken
    wmape_value = wmape(y_true_clean, y_pred_clean)
    
    if y_train is not None:
        y_train_clean = np.array(y_train)[~np.isnan(np.array(y_train))]
        mase_value = mase(y_true_clean, y_pred_clean, y_train_clean)
    else:
        mase_value = mase(y_true_clean, y_pred_clean)
    
    # 4. Zusammenführung aller Ergebnisse
    return {
        'MAE': mae(y_true_clean, y_pred_clean),
        'RMSE': rmse(y_true_clean, y_pred_clean),
        'wMAPE': wmape_value,
        'MASE': mase_value,
        'IC_Return': ic,           # Kernmetrik für deine Thesis
        'RankIC_Return': ric,      # Robuste Variante
        'IC_Lag_1': ic_lag,        # Der Lag-Wert
        'Is_Lagging': is_lagging,  # Warnflagge (Sollte False sein)
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


def calculate_persistence_baseline(context_data: np.ndarray, forecast_hours: int) -> np.ndarray:
    """
    Erstellt eine Persistence Forecast (Naive Baseline)
    
    LOGIK: Hält den letzten Wert aus den Context-Daten für den gesamten
    Forecast-Horizont konstant (Last-Value-Carried-Forward)
    
    Args:
        context_data: Array der Context-Werte (z.B. letzte 400h)
        forecast_hours: Anzahl Stunden für die Vorhersage (z.B. 24h)
    
    Returns:
        Array mit konstanten Vorhersagewerten (Länge = forecast_hours)
    
    Example:
        context = [100, 101, 99, 102, 98]  # Letzte 5 Preise
        forecast = calculate_persistence_baseline(context, 3)
        # Returns: [98, 98, 98]  # Letzter Wert (98) für 3 Stunden konstant
    """
    if len(context_data) == 0:
        raise ValueError("Context data cannot be empty")
    
    # Letzter verfügbarer Wert aus Context
    last_value = context_data[-1]
    
    # Repliziere diesen Wert für gesamten Forecast-Horizont
    persistence_forecast = np.full(forecast_hours, last_value)
    
    return persistence_forecast


def calculate_baseline_comparison(y_true: np.ndarray, y_pred_model: np.ndarray,
                                 context_data: np.ndarray, forecast_hours: int) -> dict:
    """
    Vergleicht Model-Vorhersagen mit Persistence Baseline
    
    Args:
        y_true: Tatsächliche Werte
        y_pred_model: Model-Vorhersagen (z.B. Kronos)
        context_data: Context-Daten für Persistence Baseline
        forecast_hours: Forecast-Horizont
    
    Returns:
        Dictionary mit Metriken für Model und Baseline
    """
    # Erstelle Persistence Baseline
    y_pred_baseline = calculate_persistence_baseline(context_data, forecast_hours)
    
    # Berechne Metriken für beide
    model_metrics = calculate_all_metrics(y_true, y_pred_model)
    baseline_metrics = calculate_all_metrics(y_true, y_pred_baseline)
    
    return {
        'model': model_metrics,
        'baseline': baseline_metrics,
        'baseline_predictions': y_pred_baseline.tolist()
    }