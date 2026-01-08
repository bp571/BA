import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
from pathlib import Path

# Add the core directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'core'))

from model_loader import ChronosLoader


def calculate_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    # Handle negative prices in MAPE calculation by using absolute values
    mape = np.mean(np.abs((actual - pred) / np.where(np.abs(actual) < 1e-5, 1e-5, np.abs(actual)))) * 100
    rmse = np.sqrt(np.mean((actual - pred)**2))
    return {"mae": float(mae), "mape": float(mape), "rmse": float(rmse)}


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
            
            # Calculate metrics 
            day_metrics = calculate_metrics(actual_values, pred_values)
            
            current_date = target_data['datetime'].iloc[0].strftime("%Y-%m-%d")
            results.append({
                "date": current_date,
                "actual": actual_values.tolist(),
                "predicted": pred_values.tolist(),
                "metrics": day_metrics,
                "quantiles": {
                    "q10": pred_df['0.1'].values.tolist(),
                    "q50": pred_df['0.5'].values.tolist(), 
                    "q90": pred_df['0.9'].values.tolist()
                }
            })

        except Exception as e:
            print(f"Error processing day {i+1}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Save results
    output_path = "experiments/zero_shot/results_chronos_rolling.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {output_path}")
    print(f"Processed {len(results)}/{steps} days successfully")
    
    # Calculate and print overall metrics
    if results:
        all_actual = []
        all_predicted = []
        for result in results:
            all_actual.extend(result['actual'])
            all_predicted.extend(result['predicted'])
        
        overall_metrics = calculate_metrics(np.array(all_actual), np.array(all_predicted))
        print(f"\nOverall Performance:")
        print(f"  MAE: {overall_metrics['mae']:.3f} €/MWh")
        print(f"  RMSE: {overall_metrics['rmse']:.3f} €/MWh")
        print(f"  MAPE: {overall_metrics['mape']:.3f}%")
    
    return results


if __name__ == "__main__":
    run_rolling_forecast(
        raw_data_path='data/raw/smard_energy_data_2020_2025_combined.csv',
        start_date='2024-01-01',
        steps=30
    )