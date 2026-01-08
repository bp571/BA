import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import glob
from tqdm import tqdm
import time

# Add the experiments directory to the Python path for metrics import
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'experiments'))

from metrics import calculate_all_metrics, load_and_calculate_metrics_from_csv


def find_prediction_csvs(start_date: str, steps: int = 30) -> tuple:
    """
    Sucht nach bestehenden CSV-Dateien mit Vorhersageergebnissen
    
    Args:
        start_date: Start-Datum im Format 'YYYY-MM-DD'
        steps: Anzahl der Vorhersagetage
    
    Returns:
        Tuple (chronos_csv_path, kronos_csv_path) oder (None, None) wenn nicht gefunden
    """
    # Berechne erwartetes End-Datum
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = start_dt + timedelta(days=steps-1)
    end_date = end_dt.strftime('%Y-%m-%d')
    
    # Suche nach den spezifischen CSV-Dateien
    chronos_pattern = f"experiments/zero_shot/chronos_predictions_{start_date}_{end_date}.csv"
    kronos_pattern = f"experiments/zero_shot/kronos_predictions_{start_date}_{end_date}.csv"
    
    chronos_csv = chronos_pattern if os.path.exists(chronos_pattern) else None
    kronos_csv = kronos_pattern if os.path.exists(kronos_pattern) else None
    
    # Fallback: Suche nach ähnlichen Dateien
    if not chronos_csv:
        chronos_files = glob.glob(f"experiments/zero_shot/chronos_predictions_{start_date}_*.csv")
        chronos_csv = chronos_files[0] if chronos_files else None
        
    if not kronos_csv:
        kronos_files = glob.glob(f"experiments/zero_shot/kronos_predictions_{start_date}_*.csv")
        kronos_csv = kronos_files[0] if kronos_files else None
    
    return chronos_csv, kronos_csv


def run_rolling_forecast_script(script_name: str, model_name: str, start_date: str, steps: int) -> bool:
    """
    Führt ein Rolling Forecast Skript aus
    
    Args:
        script_name: Name des Python-Skripts
        model_name: Name des Modells (für Ausgabe)
        start_date: Start-Datum für das Forecast
        steps: Anzahl der Forecast-Tage
    
    Returns:
        True wenn erfolgreich, False sonst
    """
    script_path = f"experiments/zero_shot/{script_name}"
    
    if not os.path.exists(script_path):
        print(f"❌ {script_path} nicht gefunden")
        return False
    
    print(f"🚀 {model_name} Forecast läuft...")
    
    try:
        # Führe das Skript mit Parametern aus
        result = subprocess.run([
            sys.executable, script_path,
            '--start-date', start_date,
            '--steps', str(steps)
        ], capture_output=False, text=True, cwd=str(project_root))
        
        if result.returncode == 0:
            print(f"✅ {model_name} abgeschlossen")
            return True
        else:
            print(f"❌ {model_name} fehlgeschlagen")
            return False
            
    except Exception as e:
        print(f"❌ Fehler bei {model_name}: {e}")
        return False


def load_and_merge_predictions(chronos_csv: str, kronos_csv: str) -> pd.DataFrame:
    """
    Lädt beide CSV-Dateien und führt sie für den Vergleich zusammen
    
    Args:
        chronos_csv: Pfad zur Chronos CSV-Datei
        kronos_csv: Pfad zur Kronos CSV-Datei
    
    Returns:
        DataFrame mit beiden Vorhersagen zum Vergleich
    """
    try:
        chronos_df = pd.read_csv(chronos_csv)
        kronos_df = pd.read_csv(kronos_csv)
        
        # Erstelle gemeinsame Zeitstempel für Vergleich
        chronos_df['timestamp'] = chronos_df['date'] + '_' + chronos_df['hour'].astype(str)
        kronos_df['timestamp'] = kronos_df['date'] + '_' + kronos_df['hour'].astype(str)
        
        # Führe DataFrames zusammen
        merged_df = chronos_df.merge(
            kronos_df,
            on='timestamp',
            suffixes=('_chronos', '_kronos')
        )
        
        return merged_df
        
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")
        return pd.DataFrame()


def compare_models(merged_df: pd.DataFrame) -> dict:
    """
    Vergleicht beide Modelle anhand der zusammengeführten Daten
    
    Args:
        merged_df: DataFrame mit beiden Vorhersagen
    
    Returns:
        Dictionary mit Vergleichsresultaten
    """
    if merged_df.empty:
        return {}
    
    # Berechne Metriken für beide Modelle
    chronos_metrics = calculate_all_metrics(
        merged_df['actual_value_chronos'].values,
        merged_df['predicted_value_chronos'].values
    )
    
    kronos_metrics = calculate_all_metrics(
        merged_df['actual_value_kronos'].values,
        merged_df['predicted_value_kronos'].values
    )
    
    # Vergleiche Vorhersagen direkt miteinander
    chronos_vs_kronos_ic = np.corrcoef(
        merged_df['predicted_value_chronos'].values,
        merged_df['predicted_value_kronos'].values
    )[0, 1]
    
    return {
        'chronos_metrics': chronos_metrics,
        'kronos_metrics': kronos_metrics,
        'prediction_correlation': chronos_vs_kronos_ic,
        'sample_size': len(merged_df)
    }


def print_comparison_results(results: dict):
    """
    Gibt die Vergleichsresultate formatiert aus
    """
    if not results:
        print("❌ Keine Ergebnisse")
        return
    
    print("\n" + "="*70)
    print("📈 MODEL COMPARISON RESULTS")
    print("="*70)
    
    chronos = results['chronos_metrics']
    kronos = results['kronos_metrics']
    
    print(f"📊 Datenpunkte: {results['sample_size']}")
    
    # Kompakte Metriken-Tabelle
    print(f"\n{'Metrik':<20} {'Chronos':<12} {'Kronos':<12} {'Winner':<10}")
    print("-" * 54)
    
    # MAE
    mae_winner = "Chronos" if chronos['MAE'] < kronos['MAE'] else "Kronos"
    print(f"{'MAE':<20} {chronos['MAE']:<12.3f} {kronos['MAE']:<12.3f} {mae_winner:<10}")
    
    # RMSE
    rmse_winner = "Chronos" if chronos['RMSE'] < kronos['RMSE'] else "Kronos"
    print(f"{'RMSE':<20} {chronos['RMSE']:<12.3f} {kronos['RMSE']:<12.3f} {rmse_winner:<10}")
    
    # MAPE
    mape_winner = "Chronos" if chronos['MAPE'] < kronos['MAPE'] else "Kronos"
    print(f"{'MAPE (%)':<20} {chronos['MAPE']:<12.1f} {kronos['MAPE']:<12.1f} {mape_winner:<10}")
    
    # IC (höher ist besser)
    ic_winner = "Chronos" if chronos['IC'] > kronos['IC'] else "Kronos"
    print(f"{'Info Coeff.':<20} {chronos['IC']:<12.3f} {kronos['IC']:<12.3f} {ic_winner:<10}")
    
    # Directional Accuracy (höher ist besser)
    da_winner = "Chronos" if chronos['Directional_Accuracy'] > kronos['Directional_Accuracy'] else "Kronos"
    print(f"{'Direction Acc (%)':<20} {chronos['Directional_Accuracy']:<12.1f} {kronos['Directional_Accuracy']:<12.1f} {da_winner:<10}")
    
    print("-" * 54)
    
    # Winner Summary
    winners = {
        'MAE': mae_winner,
        'RMSE': rmse_winner,
        'MAPE': mape_winner,
        'IC': ic_winner,
        'DA': da_winner
    }
    
    chronos_wins = sum(1 for w in winners.values() if w == 'Chronos')
    kronos_wins = sum(1 for w in winners.values() if w == 'Kronos')
    
    if chronos_wins > kronos_wins:
        print(f"\n🏆 WINNER: CHRONOS ({chronos_wins}/5 Metriken)")
    elif kronos_wins > chronos_wins:
        print(f"\n🏆 WINNER: KRONOS ({kronos_wins}/5 Metriken)")
    else:
        print(f"\n🤝 TIE ({chronos_wins}-{kronos_wins})")
    
    print(f"🔄 Pred. Korrelation: {results['prediction_correlation']:.3f}")
    print("="*70)


def main(start_date: str = '2024-01-01', steps: int = 30, force_rerun: bool = False):
    """
    Hauptfunktion für den Rolling Forecast Vergleich
    
    Args:
        start_date: Start-Datum für Rolling Forecasts
        steps: Anzahl der Vorhersagetage
        force_rerun: Erzwingt Neuausführung auch wenn CSV-Dateien existieren
    """
    # 1. Prüfe ob CSV-Ergebnisse bereits vorliegen
    chronos_csv, kronos_csv = find_prediction_csvs(start_date, steps)
    
    if force_rerun or not chronos_csv or not kronos_csv:
        # Erstelle eine Liste der auszuführenden Forecasts
        forecasts_to_run = []
        if force_rerun or not chronos_csv:
            forecasts_to_run.append(("chronos_rolling_forecast.py", "Chronos"))
        if force_rerun or not kronos_csv:
            forecasts_to_run.append(("kronos_rolling_forecast.py", "Kronos"))
        
        # Führe alle Forecasts mit Fortschrittsbalken aus
        for script_name, model_name in tqdm(forecasts_to_run, desc="Models", unit="model"):
            success = run_rolling_forecast_script(script_name, model_name, start_date, steps)
            if not success:
                print(f"❌ Abbruch: {model_name} fehlgeschlagen")
                return
        
        # Suche erneut nach CSV-Dateien
        chronos_csv, kronos_csv = find_prediction_csvs(start_date, steps)
    
    if not chronos_csv or not kronos_csv:
        print("❌ CSV-Dateien fehlen")
        return
    
    # 2. Lade und vergleiche die Ergebnisse
    merged_df = load_and_merge_predictions(chronos_csv, kronos_csv)
    
    if merged_df.empty:
        print("❌ Daten konnten nicht geladen werden")
        return
    
    # 3. Berechne Metriken und zeige Vergleich
    results = compare_models(merged_df)
    
    print_comparison_results(results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rolling Forecast Model Comparison')
    parser.add_argument('--start-date', default='2024-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--steps', type=int, default=30,
                        help='Forecast days')
    parser.add_argument('--force-rerun', action='store_true',
                        help='Force rerun')
    
    args = parser.parse_args()
    
    main(
        start_date=args.start_date,
        steps=args.steps,
        force_rerun=args.force_rerun
    )