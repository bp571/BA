import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
from pathlib import Path

# Add the core directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'core'))

from model_loader import ChronosLoader


def save_predictions_to_csv(results, output_path):
    """
    Speichert Rolling Forecast Rohdaten als CSV für spätere Metrik-Berechnungen
    
    Args:
        results: List von Dictionaries mit Rolling Forecast Ergebnissen
        output_path: Pfad zur CSV-Ausgabedatei
    """
    rows = []
    for day_result in results:
        date = day_result["date"]
        actual_values = day_result["actual"]
        predicted_values = day_result["predicted"]
        
        for hour in range(24):
            if hour < len(actual_values) and hour < len(predicted_values):
                rows.append({
                    'date': date,
                    'hour': hour,
                    'actual_value': actual_values[hour],
                    'predicted_value': predicted_values[hour]
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"CSV predictions saved to: {output_path}")


def prepare_chronos_data(df_context):
    """Prepare context data in Chronos format (univariate time series)"""
    # For Chronos, we use only the price column as target
    chronos_df = df_context[['datetime', 'price']].copy()
    chronos_df['id'] = 'energy_price'
    chronos_df = chronos_df.rename(columns={'price': 'target'})
    
    # Ensure regular frequency (hourly)
    chronos_df = chronos_df.set_index('datetime')
    chronos_df = chronos_df.asfreq('h', method='ffill')  # Fill gaps with forward fill
    chronos_df = chronos_df.reset_index()
    
    return chronos_df


def run_rolling_forecast(raw_data_path, start_date, steps=31):
    # 1. Load Chronos pipeline
    pipeline = ChronosLoader.load("amazon/chronos-2", device_map="cpu")
    
    # 2. Load raw data (like in chronos_zero_shot_single.py)
    df = pd.read_csv(raw_data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Resample to hourly data to ensure regular frequency for Chronos
    df_hourly = df.set_index('datetime').resample('H').mean()
    df_hourly = df_hourly.dropna().reset_index()
    
    results = []
    # Find start index based on datetime
    start_date_dt = pd.to_datetime(start_date)
    try:
        start_idx = df_hourly[df_hourly['datetime'] >= start_date_dt].index[0]
    except:
        start_idx = len(df_hourly) // 2  # Fallback to middle of dataset

    print(f"Starting Chronos Rolling Forecast from {start_date} for {steps} days...")
    print(f"Using hourly price data with {len(df_hourly)} records")

    for i in tqdm(range(steps)):
        # Index-based selection (similar to Kronos)
        context_start_idx = start_idx + (i * 24) - 168  # 168 hours (7 days) back
        context_end_idx = start_idx + (i * 24)  # Current point
        forecast_end_idx = context_end_idx + 24  # 24 hours ahead
        
        try:
            # Check bounds
            if context_start_idx < 0 or forecast_end_idx >= len(df_hourly):
                continue
                
            # Extract data periods
            context_data = df_hourly.iloc[context_start_idx:context_end_idx].copy()
            target_data = df_hourly.iloc[context_end_idx:forecast_end_idx].copy()
            
            if len(context_data) < 168 or len(target_data) < 24:
                continue

            # Prepare data in Chronos format
            chronos_context = prepare_chronos_data(context_data)
            
            # Make prediction using Chronos
            pred_df = pipeline.predict_df(
                chronos_context,
                prediction_length=24,  # 24 hours ahead
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="datetime",
                target="target"
            )
            
            # Extract median predictions (0.5 quantile)
            pred_values = pred_df['0.5'].values
            actual_values = target_data['price'].values
            
            current_date = target_data['datetime'].iloc[0].strftime("%Y-%m-%d")
            results.append({
                "date": current_date,
                "actual": actual_values.tolist(),
                "predicted": pred_values.tolist()
            })

        except Exception as e:
            print(f"Error processing day {i+1}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Save results as CSV for metric calculations
    # Generate CSV filename with date range
    if results:
        start_date_str = results[0]["date"]
        end_date_str = results[-1]["date"]
        csv_output_path = f"experiments/zero_shot/chronos_predictions_{start_date_str}_{end_date_str}.csv"
    else:
        csv_output_path = f"experiments/zero_shot/chronos_predictions_{start_date}.csv"
    
    save_predictions_to_csv(results, csv_output_path)
    print(f"Processed {len(results)}/{steps} days successfully")
    
    return results


if __name__ == "__main__":
    run_rolling_forecast(
        raw_data_path='data/raw/smard_energy_data_2020_2025_combined.csv',
        start_date='2024-01-01',
        steps=30
    )