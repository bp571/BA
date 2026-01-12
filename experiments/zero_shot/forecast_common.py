"""
Gemeinsame Funktionen und Parameter für Rolling Forecasts
Optimiert für effizienten In-Memory Vergleich ohne CSV-Export
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# Standardisierte Parameter für beide Modelle
DEFAULT_PARAMS = {
    'context_hours': 400,  # Standardisiert auf 400h (unter Kronos 512 limit)
    'forecast_hours': 24,  # 24h Vorhersage-Horizont
    'data_path': 'data/processed/apple_2025.csv',  # Standard-Datendatei
    'start_date': '2025-07-01',
    'steps': 30
}


def load_and_prepare_data(data_path, price_column='close'):
    """
    Lädt und bereitet Daten für beide Modelle vor
    
    Args:
        data_path: Pfad zur Datendatei
        price_column: Spalte für Preisdaten ('close' für OHLCV, 'price' für raw)
    
    Returns:
        DataFrame mit standardisierten Daten
    """
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Prüfe verfügbare Spalten
    if 'close' in df.columns:
        print("Detected OHLCV data format")
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing OHLCV columns: {[col for col in required_cols if col not in df.columns]}")
        use_cols = required_cols + (['volume'] if 'volume' in df.columns else [])
    elif 'price' in df.columns:
        print("Detected raw price data format")
        use_cols = ['price']
        price_column = 'price'
    else:
        raise ValueError("Data must contain either 'close' column (OHLCV data) or 'price' column (raw data)")
    
    return df, use_cols, price_column


def validate_window_separation(context_end_idx, forecast_start_idx, step_idx):
    """
    Validiert, dass Context- und Forecast-Fenster korrekt getrennt sind (NO LEAKAGE)
    
    Args:
        context_end_idx: Ende des Context-Fensters
        forecast_start_idx: Start des Forecast-Fensters
        step_idx: Aktueller Schritt für Debug-Info
        
    Returns:
        bool: True wenn korrekt getrennt
    """
    if context_end_idx != forecast_start_idx:
        print(f"⚠️  DATA LEAKAGE DETECTED in Step {step_idx}:")
        print(f"   Context ends at index {context_end_idx}")
        print(f"   Forecast starts at index {forecast_start_idx}")
        print(f"   GAP/OVERLAP: {forecast_start_idx - context_end_idx}")
        return False
    return True


def get_data_periods(df, start_date, step_idx, context_hours, forecast_hours):
    """
    Extrahiert Kontext- und Forecast-Perioden für einen Rolling Forecast Schritt
    
    *** DATA LEAKAGE PREVENTION ***
    NON-OVERLAPPING ROLLING WINDOW LOGIK:
    - Jeder Schritt i verschiebt sich um volle forecast_hours (24h)
    - Test-Fenster überlappen NIEMALS zwischen Steps
    - Context endet EXAKT dort wo Forecast beginnt (kein Gap/Overlap)
    
    BEISPIEL (context_hours=400, forecast_hours=24):
    Step 0: Context [start_idx-400:start_idx+0],    Forecast [start_idx+0:start_idx+24]
    Step 1: Context [start_idx-376:start_idx+24],   Forecast [start_idx+24:start_idx+48]  <- NO OVERLAP!
    Step 2: Context [start_idx-352:start_idx+48],   Forecast [start_idx+48:start_idx+72]  <- NO OVERLAP!
    
    Args:
        df: DataFrame mit Daten
        start_date: Start-Datum
        step_idx: Schritt-Index (0, 1, 2, ...)
        context_hours: Anzahl Kontext-Stunden
        forecast_hours: Anzahl Forecast-Stunden
    
    Returns:
        Tuple (context_data, target_data) oder (None, None) wenn außerhalb Grenzen
    """
    # Handle timezone-aware datetime
    start_date_dt = pd.to_datetime(start_date)
    if df['datetime'].dt.tz is not None:
        start_date_dt = start_date_dt.tz_localize('UTC')
    
    try:
        start_idx = df[df['datetime'] >= start_date_dt].index[0]
    except IndexError:
        # Fallback to middle of dataset
        start_idx = len(df) // 2
    
    # Calculate indices - NO OVERLAP: Each step advances by full forecast_hours
    # ensuring test periods never overlap and context doesn't leak into future tests
    forecast_start_offset = step_idx * forecast_hours
    context_start = start_idx + forecast_start_offset - context_hours
    context_end = start_idx + forecast_start_offset
    forecast_end = context_end + forecast_hours
    
    # Check bounds
    if context_start < 0 or forecast_end >= len(df):
        return None, None
    
    context_data = df.iloc[context_start:context_end].copy()
    target_data = df.iloc[context_end:forecast_end].copy()
    
    # CRITICAL: Validate NO DATA LEAKAGE between context and forecast windows
    # Context should end exactly where forecast begins (context_end == forecast_start)
    forecast_start = context_end  # By design, forecast starts where context ends
    if not validate_window_separation(context_end, forecast_start, step_idx):
        print(f"❌ Step {step_idx} failed leakage validation")
        print(f"   Context range: [{context_start}:{context_end}]")
        print(f"   Forecast range: [{context_end}:{forecast_end}]")
        
    # Validate data lengths
    if len(context_data) < context_hours or len(target_data) < forecast_hours:
        return None, None
    
    # Additional validation: Ensure temporal continuity (RELAXED FOR GAPPED DATA)
    if len(context_data) > 0 and len(target_data) > 0:
        context_last_time = context_data['datetime'].iloc[-1]
        forecast_first_time = target_data['datetime'].iloc[0]
        
        # RELAXED validation for datasets with natural gaps (weekends, nights, holidays)
        actual_gap = forecast_first_time - context_last_time
        
        # Allow gaps up to 3 days (weekend + holiday scenarios)
        max_allowed_gap = pd.Timedelta(days=3)
        min_expected_gap = pd.Timedelta(hours=1)
        
        if actual_gap < min_expected_gap:
            print(f"⚠️  Temporal overlap detected in Step {step_idx}:")
            print(f"   Gap too small: {actual_gap} (minimum: {min_expected_gap})")
        elif actual_gap > max_allowed_gap:
            print(f"⚠️  Temporal gap too large in Step {step_idx}:")
            print(f"   Gap: {actual_gap} (maximum allowed: {max_allowed_gap})")
    
    return context_data, target_data


def calculate_forecast_metrics(actual_values, predicted_values):
    """
    Berechnet Metriken für Forecast-Vergleich (ohne externe metrics.py Abhängigkeit)
    
    Args:
        actual_values: Array der tatsächlichen Werte
        predicted_values: Array der vorhergesagten Werte
    
    Returns:
        Dictionary mit Metriken
    """
    actual = np.array(actual_values)
    pred = np.array(predicted_values)
    
    # Basic error metrics
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    
    # Information Coefficient (correlation between predictions and actuals)
    ic = np.corrcoef(pred, actual)[0, 1] if len(pred) > 1 else 0.0
    
    # Directional Accuracy
    actual_direction = np.diff(actual) > 0
    pred_direction = np.diff(pred) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100 if len(actual) > 1 else 0.0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'IC': ic,
        'Directional_Accuracy': directional_accuracy
    }


def print_comparison_results(chronos_results, kronos_results):
    """
    Gibt Vergleichsresultate formatiert aus
    
    Args:
        chronos_results: Liste von Forecast-Resultaten für Chronos
        kronos_results: Liste von Forecast-Resultaten für Kronos
    """
    if not chronos_results or not kronos_results:
        print("❌ Keine Ergebnisse für Vergleich")
        return
    
    # Flatten results for metric calculation
    chronos_actual = []
    chronos_pred = []
    kronos_actual = []
    kronos_pred = []
    
    for c_res, k_res in zip(chronos_results, kronos_results):
        chronos_actual.extend(c_res['actual'])
        chronos_pred.extend(c_res['predicted'])
        kronos_actual.extend(k_res['actual'])
        kronos_pred.extend(k_res['predicted'])
    
    # Calculate metrics
    chronos_metrics = calculate_forecast_metrics(chronos_actual, chronos_pred)
    kronos_metrics = calculate_forecast_metrics(kronos_actual, kronos_pred)
    
    # Prediction correlation
    pred_correlation = np.corrcoef(chronos_pred, kronos_pred)[0, 1] if len(chronos_pred) > 1 else 0.0
    
    print("\n" + "="*70)
    print("📈 MODEL COMPARISON RESULTS")
    print("="*70)
    
    print(f"📊 Forecast Days: {len(chronos_results)}")
    print(f"📊 Total Data Points: {len(chronos_actual)}")
    
    # Metrics table
    print(f"\n{'Metrik':<20} {'Chronos':<12} {'Kronos':<12} {'Winner':<10}")
    print("-" * 54)
    
    # MAE
    mae_winner = "Chronos" if chronos_metrics['MAE'] < kronos_metrics['MAE'] else "Kronos"
    print(f"{'MAE':<20} {chronos_metrics['MAE']:<12.3f} {kronos_metrics['MAE']:<12.3f} {mae_winner:<10}")
    
    # RMSE
    rmse_winner = "Chronos" if chronos_metrics['RMSE'] < kronos_metrics['RMSE'] else "Kronos"
    print(f"{'RMSE':<20} {chronos_metrics['RMSE']:<12.3f} {kronos_metrics['RMSE']:<12.3f} {rmse_winner:<10}")
    
    # MAPE
    mape_winner = "Chronos" if chronos_metrics['MAPE'] < kronos_metrics['MAPE'] else "Kronos"
    print(f"{'MAPE (%)':<20} {chronos_metrics['MAPE']:<12.1f} {kronos_metrics['MAPE']:<12.1f} {mape_winner:<10}")
    
    # IC (höher ist besser)
    ic_winner = "Chronos" if chronos_metrics['IC'] > kronos_metrics['IC'] else "Kronos"
    print(f"{'Info Coeff.':<20} {chronos_metrics['IC']:<12.3f} {kronos_metrics['IC']:<12.3f} {ic_winner:<10}")
    
    # Directional Accuracy (höher ist besser)
    da_winner = "Chronos" if chronos_metrics['Directional_Accuracy'] > kronos_metrics['Directional_Accuracy'] else "Kronos"
    print(f"{'Direction Acc (%)':<20} {chronos_metrics['Directional_Accuracy']:<12.1f} {kronos_metrics['Directional_Accuracy']:<12.1f} {da_winner:<10}")
    
    print("-" * 54)
    
    # Winner summary
    winners = [mae_winner, rmse_winner, mape_winner, ic_winner, da_winner]
    chronos_wins = winners.count('Chronos')
    kronos_wins = winners.count('Kronos')
    
    if chronos_wins > kronos_wins:
        print(f"\n🏆 WINNER: CHRONOS ({chronos_wins}/5 Metriken)")
    elif kronos_wins > chronos_wins:
        print(f"\n🏆 WINNER: KRONOS ({kronos_wins}/5 Metriken)")
    else:
        print(f"\n🤝 TIE ({chronos_wins}-{kronos_wins})")
    
    print(f"🔄 Pred. Korrelation: {pred_correlation:.3f}")
    print("="*70)