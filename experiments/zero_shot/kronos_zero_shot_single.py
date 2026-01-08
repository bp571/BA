#!/usr/bin/env python3
"""
Kronos Zero-Shot Single Prediction Script for Energy Price Forecasting

This script makes a single prediction for the next time step using the pre-trained Kronos model
and outputs results as JSON.

Author: Energy Forecasting System
Date: 2026-01-06
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the core directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'core'))

from model_loader import KronosLoader

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = project_root / 'data' / 'processed' / 'smard_energy_data_2020_2025_combined_hourly_candles.csv'
OUTPUT_DIR = project_root / 'evaluation' / 'plots'
RESULTS_DIR = project_root / 'experiments' / 'zero_shot'

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Test parameters
SEQUENCE_LENGTH = 400  # Lookbook window (as in README example)
FORECAST_LENGTH = 24   # 24 hours ahead prediction
SAMPLE_COUNT = 1       # Number of samples for uncertainty estimation


# Model loading now handled by KronosLoader


def load_and_prepare_energy_data():
    """Load energy data and prepare it in the format Kronos expects"""
    print("Loading energy data...")
    
    # Load the processed energy data
    df = pd.read_csv(DATA_PATH)
    
    # Convert datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for negative prices
    negative_count = (df['close'] < 0).sum()
    print(f"Found {negative_count} negative price periods ({negative_count/len(df)*100:.1f}%)")
    
    # Create amount column (volume * average price)
    if 'amount' not in df.columns:
        df['amount'] = df['volume'] * df['avg_price']
    
    # Ensure we have required columns: open, high, low, close, volume, amount
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in required_cols:
        if col not in df.columns:
            if col == 'amount':
                df[col] = df['volume'] * df[['open', 'high', 'low', 'close']].mean(axis=1)
            else:
                raise ValueError(f"Required column '{col}' not found in data")
    
    print(f"✓ Data prepared with shape: {df.shape}")
    return df


def add_negative_price_offset(df, offset_value=50.0):
    """
    Add a positive offset to handle negative prices while preserving relative patterns
    
    Args:
        df: DataFrame with price columns
        offset_value: Positive value to add to all price columns
    
    Returns:
        df_offset: DataFrame with offset applied
        offset_value: The offset value used (for later removal from predictions)
    """
    print(f"Applying offset of {offset_value} to handle negative prices...")
    
    df_offset = df.copy()
    price_cols = ['open', 'high', 'low', 'close']
    
    # Store original ranges for verification
    original_ranges = {}
    for col in price_cols:
        original_ranges[col] = (df[col].min(), df[col].max())
    
    # Apply offset to price columns
    for col in price_cols:
        df_offset[col] = df[col] + offset_value
    
    print("Original ranges:")
    for col, (min_val, max_val) in original_ranges.items():
        print(f"  {col}: [{min_val:.2f}, {max_val:.2f}]")
    
    print("Offset ranges:")
    for col in price_cols:
        print(f"  {col}: [{df_offset[col].min():.2f}, {df_offset[col].max():.2f}]")
    
    # Verify no negative values remain
    negative_remaining = sum((df_offset[col] < 0).sum() for col in price_cols)
    if negative_remaining > 0:
        print(f"WARNING: {negative_remaining} negative values still remain!")
    else:
        print("✓ All negative values handled successfully")
    
    return df_offset, offset_value


def run_24h_prediction(predictor, df):
    """
    Run zero-shot forecasting for the next 24 hours using KronosPredictor
    
    Returns:
        predictions: numpy array with 24 predictions
        context_info: dict with context information
    """
    
    # Use the last SEQUENCE_LENGTH points as context
    total_length = len(df)
    
    if total_length < SEQUENCE_LENGTH:
        raise ValueError(f"Dataset too small: need at least {SEQUENCE_LENGTH} points, got {total_length}")
    
    # Extract context data (last SEQUENCE_LENGTH points)
    context_data = df.iloc[-SEQUENCE_LENGTH:].copy()
    
    # Get the last datetime and calculate next 24 timestamps (assuming hourly data)
    last_datetime = pd.to_datetime(context_data['datetime'].iloc[-1])
    future_timestamps = pd.Series([last_datetime + pd.Timedelta(hours=i+1) for i in range(FORECAST_LENGTH)])
    
    # Prepare input timestamps
    input_timestamps = pd.to_datetime(context_data['datetime'])
    
    print(f"Context period: {input_timestamps.iloc[0]} to {input_timestamps.iloc[-1]}")
    print(f"Forecasting period: {future_timestamps.iloc[0]} to {future_timestamps.iloc[-1]}")
    
    # Generate predictions using the official KronosPredictor API
    try:
        predictions_df = predictor.predict(
            df=context_data[['open', 'high', 'low', 'close', 'volume', 'amount']],
            x_timestamp=input_timestamps,
            y_timestamp=future_timestamps,
            pred_len=FORECAST_LENGTH,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=SAMPLE_COUNT,
            verbose=True
        )
        
        # Extract predictions (24 rows)
        prediction_values = predictions_df[['open', 'high', 'low', 'close', 'volume', 'amount']].values
        
        return prediction_values, {
            'last_datetime': last_datetime,
            'future_timestamps': future_timestamps,
            'context_length': SEQUENCE_LENGTH,
            'context_start': input_timestamps.iloc[0],
            'context_end': input_timestamps.iloc[-1]
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_24h_prediction_with_context(predictor, context_data, test_timestamps):
    """
    Run zero-shot forecasting for 24 hours with specific context and test timestamps
    
    Returns:
        predictions: numpy array with 24 predictions
        context_info: dict with context information
    """
    
    # Prepare input timestamps and future timestamps
    input_timestamps = pd.to_datetime(context_data['datetime'])
    future_timestamps = pd.Series(test_timestamps)
    
    print(f"Context period: {input_timestamps.iloc[0]} to {input_timestamps.iloc[-1]}")
    print(f"Test period: {future_timestamps.iloc[0]} to {future_timestamps.iloc[-1]}")
    
    # Generate predictions using the official KronosPredictor API
    try:
        predictions_df = predictor.predict(
            df=context_data[['open', 'high', 'low', 'close', 'volume', 'amount']],
            x_timestamp=input_timestamps,
            y_timestamp=future_timestamps,
            pred_len=FORECAST_LENGTH,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=SAMPLE_COUNT,
            verbose=True
        )
        
        # Extract predictions (24 rows)
        prediction_values = predictions_df[['open', 'high', 'low', 'close', 'volume', 'amount']].values
        
        return prediction_values, {
            'context_length': len(context_data),
            'context_start': input_timestamps.iloc[0],
            'context_end': input_timestamps.iloc[-1],
            'test_timestamps': future_timestamps
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main execution function"""
    print("=" * 60)
    print("KRONOS ZERO-SHOT 24H PREDICTION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Sequence length: {SEQUENCE_LENGTH} hours")
    print(f"Forecast length: {FORECAST_LENGTH} hours")
    print()
    
    try:
        # 1. Load predictor using KronosLoader
        predictor = KronosLoader.get_predictor(device=DEVICE)
        
        # 2. Load and prepare data
        df = load_and_prepare_energy_data()
        
        # 4. Handle negative prices with offset
        df_offset, price_offset = add_negative_price_offset(df, offset_value=100.0)
        
        # 5. Split data for actual comparison: use last 24 hours as test set
        print("Splitting data for actual comparison...")
        
        test_df = df_offset.iloc[-24:].copy()  # Last 24 hours for testing
        context_df = df_offset.iloc[-400-24:-24].copy()  # 400 hours before test period
        
        # Get actual values and timestamps for comparison
        actual_prices = test_df[['open', 'high', 'low', 'close']].values
        actual_volumes = test_df['volume'].values
        actual_amounts = test_df['amount'].values
        test_timestamps = test_df['datetime'].tolist()
        
        print("Running zero-shot prediction for test period...")
        
        prediction_values, context_info = run_24h_prediction_with_context(predictor, context_df, test_timestamps)
        
        if prediction_values is not None:
            # Remove offset from price predictions (first 4 columns: OHLC)
            predicted_prices = prediction_values[:, :4] - price_offset
            predicted_volumes = prediction_values[:, 4]
            predicted_amounts = prediction_values[:, 5]
            
            # Remove offset from actual prices for comparison
            actual_prices_clean = actual_prices - price_offset
            
            # Calculate performance metrics
            def calculate_metrics_ohlc(actual, predicted):
                metrics = {}
                price_cols = ['open', 'high', 'low', 'close']
                for i, col in enumerate(price_cols):
                    mae = np.mean(np.abs(predicted[:, i] - actual[:, i]))
                    rmse = np.sqrt(np.mean((predicted[:, i] - actual[:, i]) ** 2))
                    mape = np.mean(np.abs((predicted[:, i] - actual[:, i]) / (actual[:, i] + 1e-8))) * 100
                    metrics[col] = {'mae': mae, 'rmse': rmse, 'mape': mape}
                return metrics
            
            metrics = calculate_metrics_ohlc(actual_prices_clean, predicted_prices)
            
            # Create timestamped predictions with actual values
            predictions_list = []
            for i in range(FORECAST_LENGTH):
                predictions_list.append({
                    'timestamp': test_timestamps[i].isoformat(),
                    'hour': i + 1,
                    'actual_prices': {
                        'open': float(actual_prices_clean[i, 0]),
                        'high': float(actual_prices_clean[i, 1]),
                        'low': float(actual_prices_clean[i, 2]),
                        'close': float(actual_prices_clean[i, 3])
                    },
                    'predicted_prices': {
                        'open': float(predicted_prices[i, 0]),
                        'high': float(predicted_prices[i, 1]),
                        'low': float(predicted_prices[i, 2]),
                        'close': float(predicted_prices[i, 3])
                    },
                    'actual_volume': float(actual_volumes[i]),
                    'predicted_volume': float(predicted_volumes[i]),
                    'actual_amount': float(actual_amounts[i]),
                    'predicted_amount': float(predicted_amounts[i]),
                    'price_errors': {
                        'open_ae': float(abs(predicted_prices[i, 0] - actual_prices_clean[i, 0])),
                        'high_ae': float(abs(predicted_prices[i, 1] - actual_prices_clean[i, 1])),
                        'low_ae': float(abs(predicted_prices[i, 2] - actual_prices_clean[i, 2])),
                        'close_ae': float(abs(predicted_prices[i, 3] - actual_prices_clean[i, 3])),
                        'close_re_pct': float(((predicted_prices[i, 3] - actual_prices_clean[i, 3]) / actual_prices_clean[i, 3]) * 100)
                    }
                })
            
            # Prepare results for JSON export with standardized structure
            results = {
                'model_info': {
                    'name': 'kronos',
                    'type': 'zero_shot',
                    'prediction_type': 'deterministic',
                    'data_input': 'multivariate_ohlc_series'
                },
                'data_source': {
                    'file': 'smard_energy_data_2020_2025_combined_hourly_candles.csv',
                    'type': 'processed_ohlc_data',
                    'context_length': context_info['context_length'],
                    'context_period': {
                        'start': context_info['context_start'].isoformat(),
                        'end': context_info['context_end'].isoformat()
                    }
                },
                'forecast_info': {
                    'period': {
                        'start': test_timestamps[0].isoformat(),
                        'end': test_timestamps[-1].isoformat(),
                        'hours': FORECAST_LENGTH
                    },
                    'prediction_length': FORECAST_LENGTH,
                    'forecast_timestamp': datetime.now().isoformat()
                },
                'performance_metrics': {
                    'open': {'mae': float(metrics['open']['mae']), 'rmse': float(metrics['open']['rmse']), 'mape': float(metrics['open']['mape'])},
                    'high': {'mae': float(metrics['high']['mae']), 'rmse': float(metrics['high']['rmse']), 'mape': float(metrics['high']['mape'])},
                    'low': {'mae': float(metrics['low']['mae']), 'rmse': float(metrics['low']['rmse']), 'mape': float(metrics['low']['mape'])},
                    'close': {'mae': float(metrics['close']['mae']), 'rmse': float(metrics['close']['rmse']), 'mape': float(metrics['close']['mape'])}
                },
                'predictions': predictions_list,
                'model_specific': {
                    'price_offset_applied': price_offset,
                    'prediction_method': 'autoregressive_generation',
                    'device': DEVICE,
                    'sequence_length': SEQUENCE_LENGTH
                },
                'summary_statistics': {
                    'price_statistics': {
                        'actual_price_ranges': {
                            'open': {'min': float(actual_prices_clean[:, 0].min()), 'max': float(actual_prices_clean[:, 0].max())},
                            'high': {'min': float(actual_prices_clean[:, 1].min()), 'max': float(actual_prices_clean[:, 1].max())},
                            'low': {'min': float(actual_prices_clean[:, 2].min()), 'max': float(actual_prices_clean[:, 2].max())},
                            'close': {'min': float(actual_prices_clean[:, 3].min()), 'max': float(actual_prices_clean[:, 3].max())}
                        },
                        'predicted_price_ranges': {
                            'open': {'min': float(predicted_prices[:, 0].min()), 'max': float(predicted_prices[:, 0].max())},
                            'high': {'min': float(predicted_prices[:, 1].min()), 'max': float(predicted_prices[:, 1].max())},
                            'low': {'min': float(predicted_prices[:, 2].min()), 'max': float(predicted_prices[:, 2].max())},
                            'close': {'min': float(predicted_prices[:, 3].min()), 'max': float(predicted_prices[:, 3].max())}
                        },
                        'avg_actual_prices': {
                            'open': float(actual_prices_clean[:, 0].mean()),
                            'high': float(actual_prices_clean[:, 1].mean()),
                            'low': float(actual_prices_clean[:, 2].mean()),
                            'close': float(actual_prices_clean[:, 3].mean())
                        },
                        'avg_predicted_prices': {
                            'open': float(predicted_prices[:, 0].mean()),
                            'high': float(predicted_prices[:, 1].mean()),
                            'low': float(predicted_prices[:, 2].mean()),
                            'close': float(predicted_prices[:, 3].mean())
                        }
                    },
                    'volume_statistics': {
                        'actual_total_volume': float(actual_volumes.sum()),
                        'predicted_total_volume': float(predicted_volumes.sum()),
                        'avg_actual_volume': float(actual_volumes.mean()),
                        'avg_predicted_volume': float(predicted_volumes.mean())
                    }
                }
            }
            
            # Display results summary
            print(f"\n✓ 24-hour prediction completed successfully!")
            print(f"Test period: {test_timestamps[0]} to {test_timestamps[-1]}")
            print(f"\nPerformance Metrics (Close Price):")
            print(f"  MAE: {metrics['close']['mae']:.3f} €/MWh")
            print(f"  RMSE: {metrics['close']['rmse']:.3f} €/MWh")
            print(f"  MAPE: {metrics['close']['mape']:.3f}%")
            print(f"\nPrice summary (€/MWh):")
            print(f"  Actual close range: {actual_prices_clean[:, 3].min():.3f} - {actual_prices_clean[:, 3].max():.3f}")
            print(f"  Predicted close range: {predicted_prices[:, 3].min():.3f} - {predicted_prices[:, 3].max():.3f}")
            print(f"  Actual close average: {actual_prices_clean[:, 3].mean():.3f}")
            print(f"  Predicted close average: {predicted_prices[:, 3].mean():.3f}")
            
            # Save results to JSON
            results_path = RESULTS_DIR / 'kronos_zero_shot_24h_prediction.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ Results saved to: {results_path}")
            
            # Also output JSON to console
            print("\nJSON Output:")
            print(json.dumps(results, indent=2))
            
        else:
            print("Failed to generate prediction")
            return 1
            
        print("\n" + "=" * 60)
        print("KRONOS 24H PREDICTION COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)