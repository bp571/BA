import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
from pathlib import Path

# Add the core directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'core'))

from model_loader import KronosLoader


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


def run_rolling_forecast(data_path, start_date, steps=31):
    # 1. Load predictor using KronosLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = KronosLoader.get_predictor(device=device)
    
    # 2. Load data
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Ensure required columns exist
    if 'amount' not in df.columns:
        df['amount'] = df['volume'] * df['avg_price']
    
    results = []
    # Find start index based on datetime
    start_date_dt = pd.to_datetime(start_date)
    try:
        start_idx = df[df['datetime'] >= start_date_dt].index[0]
    except:
        start_idx = len(df) // 2  # Fallback to middle of dataset

    print(f"Starting Rolling Forecast from {start_date} for {steps} days...")

    for i in tqdm(range(steps)):
        # Index-based selection (like in the working script)
        context_start_idx = start_idx + (i * 24) - 168  # 168 hours back
        context_end_idx = start_idx + (i * 24)  # Current point
        forecast_end_idx = context_end_idx + 24  # 24 hours ahead
        
        try:
            # Check bounds
            if context_start_idx < 0 or forecast_end_idx >= len(df):
                continue
                
            # Extract data periods
            context_data = df.iloc[context_start_idx:context_end_idx]
            target_data = df.iloc[context_end_idx:forecast_end_idx]
            
            if len(context_data) < 168 or len(target_data) < 24:
                continue

            # Prepare timestamps
            input_timestamps = pd.to_datetime(context_data['datetime'])
            future_timestamps = pd.to_datetime(target_data['datetime'])
            
            # Make prediction
            predictions_df = predictor.predict(
                df=context_data[['open', 'high', 'low', 'close', 'volume', 'amount']],
                x_timestamp=input_timestamps,
                y_timestamp=future_timestamps,
                pred_len=24,
                T=1.0,
                top_k=0,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
            
            # Extract prediction and actual values
            pred_values = predictions_df['close'].values
            actual_values = target_data['close'].values
            
            current_date = target_data['datetime'].iloc[0].strftime("%Y-%m-%d")
            results.append({
                "date": current_date,
                "actual": actual_values.tolist(),
                "predicted": pred_values.tolist()
            })

        except Exception as e:
            print(f"Error processing day {i+1}: {e}")

    # 3. Save results as CSV for metric calculations
    # Generate CSV filename with date range
    if results:
        start_date_str = results[0]["date"]
        end_date_str = results[-1]["date"]
        csv_output_path = f"experiments/zero_shot/kronos_predictions_{start_date_str}_{end_date_str}.csv"
    else:
        csv_output_path = f"experiments/zero_shot/kronos_predictions_{start_date}.csv"
    
    save_predictions_to_csv(results, csv_output_path)
    print(f"Processed {len(results)}/{steps} days successfully")
    
    return results


if __name__ == "__main__":
    run_rolling_forecast(
        data_path='data/processed/smard_energy_data_2020_2025_combined_hourly_candles.csv',
        start_date='2024-01-01',
        steps=30
    )