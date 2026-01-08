#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys

# Add core directory to path for model_loader
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'core'))

from model_loader import ChronosLoader

# Paths
project_root = Path(__file__).parent.parent.parent
DATA_PATH = project_root / 'data' / 'raw' / 'smard_energy_data_2020_2025_combined.csv'
RESULTS_DIR = project_root / 'experiments' / 'zero_shot'

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("CHRONOS ZERO-SHOT 24H PREDICTION")
    print("=" * 60)
    
    print("Loading Chronos-2 pipeline...")
    pipeline = ChronosLoader.load("amazon/chronos-2", device_map="cpu")
    
    print("Loading raw energy data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Data loaded: {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Resample to hourly data to ensure regular frequency
    print("Resampling to hourly frequency for Chronos compatibility...")
    df = df.set_index('datetime')
    df_hourly = df.resample('H').mean()  # Resample to hourly, taking mean of 4x 15-minute values
    df_hourly = df_hourly.dropna().reset_index()  # Remove any NaN values and reset index
    
    print(f"After resampling: {len(df_hourly)} hourly records from {df_hourly['datetime'].min()} to {df_hourly['datetime'].max()}")
    
    # Split data: use last 24 hours as test set, rest as context
    test_df = df_hourly.iloc[-24:].copy()  # Last 24 hours for testing
    context_df = df_hourly.iloc[:-24].copy()  # All data before test period
    
    # Prepare data for Chronos (needs id column)
    context_df['id'] = 'energy_price'
    context_df = context_df.rename(columns={'price': 'target'})
    
    # Ensure the timestamps are regular (no gaps)
    context_df = context_df.set_index('datetime')
    context_df = context_df.asfreq('h', method='ffill')  # Fill any gaps with forward fill
    context_df = context_df.reset_index()
    
    # Get actual values and timestamps for comparison
    actual_values = test_df['price'].values
    test_timestamps = test_df['datetime'].tolist()
    
    print(f"Context period: {context_df['datetime'].min()} to {context_df['datetime'].max()}")
    print(f"Test period: {test_timestamps[0]} to {test_timestamps[-1]}")
    
    print("Running zero-shot forecast for test period...")
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=24,  # Predict 24 steps ahead
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="datetime",
        target="target"
    )
    
    print(f"✓ Forecast completed: {len(pred_df)} predictions")
    
    # Extract prediction values for all 24 hours
    predictions_median = pred_df['0.5'].values
    predictions_q10 = pred_df['0.1'].values
    predictions_q90 = pred_df['0.9'].values
    
    # Calculate performance metrics
    def calculate_metrics(actual, predicted):
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        mape = np.mean(np.abs((predicted - actual) / (actual + 1e-8))) * 100
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
    
    metrics = calculate_metrics(actual_values, predictions_median)
    
    # Create timestamped predictions with actual values
    predictions_list = []
    for i in range(24):
        predictions_list.append({
            'timestamp': test_timestamps[i].isoformat(),
            'hour': i + 1,
            'actual_price': float(actual_values[i]),
            'price_median': float(predictions_median[i]),
            'price_q10': float(predictions_q10[i]),
            'price_q90': float(predictions_q90[i]),
            'confidence_interval': {
                'lower_bound': float(predictions_q10[i]),
                'upper_bound': float(predictions_q90[i]),
                'range': float(predictions_q90[i] - predictions_q10[i])
            },
            'errors': {
                'absolute_error': float(abs(predictions_median[i] - actual_values[i])),
                'relative_error_pct': float(((predictions_median[i] - actual_values[i]) / actual_values[i]) * 100)
            }
        })
    
    # Prepare results for JSON export with standardized structure
    results = {
        'model_info': {
            'name': 'chronos-2',
            'type': 'zero_shot',
            'prediction_type': 'probabilistic',
            'data_input': 'univariate_price_series'
        },
        'data_source': {
            'file': 'smard_energy_data_2020_2025_combined.csv',
            'type': 'raw_price_data',
            'context_length': len(context_df),
            'context_period': {
                'start': context_df['datetime'].min().isoformat(),
                'end': context_df['datetime'].max().isoformat()
            }
        },
        'forecast_info': {
            'period': {
                'start': test_timestamps[0].isoformat(),
                'end': test_timestamps[-1].isoformat(),
                'hours': 24
            },
            'prediction_length': 24,
            'forecast_timestamp': datetime.now().isoformat()
        },
        'performance_metrics': {
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'mape': float(metrics['mape'])
        },
        'predictions': predictions_list,
        'model_specific': {
            'quantile_levels': [0.1, 0.5, 0.9],
            'prediction_method': 'quantile_regression'
        },
        'summary_statistics': {
            'price_statistics': {
                'actual_price_range': {
                    'min': float(actual_values.min()),
                    'max': float(actual_values.max()),
                    'mean': float(actual_values.mean())
                },
                'predicted_price_range': {
                    'min': float(predictions_median.min()),
                    'max': float(predictions_median.max()),
                    'mean': float(predictions_median.mean())
                },
                'uncertainty_metrics': {
                    'avg_confidence_interval_range': float(np.mean(predictions_q90 - predictions_q10)),
                    'overall_uncertainty_range': {
                        'min': float(predictions_q10.min()),
                        'max': float(predictions_q90.max())
                    }
                }
            }
        }
    }
    
    print(f"\n✓ 24-hour prediction completed successfully!")
    print(f"Test period: {test_timestamps[0]} to {test_timestamps[-1]}")
    print(f"\nPerformance Metrics:")
    print(f"  MAE: {metrics['mae']:.3f} €/MWh")
    print(f"  RMSE: {metrics['rmse']:.3f} €/MWh")
    print(f"  MAPE: {metrics['mape']:.3f}%")
    print(f"\nPrice summary (€/MWh):")
    print(f"  Actual range: {actual_values.min():.3f} - {actual_values.max():.3f}")
    print(f"  Predicted range: {predictions_median.min():.3f} - {predictions_median.max():.3f}")
    print(f"  Average confidence range: {np.mean(predictions_q90 - predictions_q10):.3f}")
    
    # Save results to JSON
    results_path = RESULTS_DIR / 'chronos_zero_shot_24h_prediction.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # Also output JSON to console
    print("\nJSON Output:")
    print(json.dumps(results, indent=2))
    
    print("\n" + "=" * 60)
    print("CHRONOS 24H PREDICTION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
