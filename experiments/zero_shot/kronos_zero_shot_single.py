#!/usr/bin/env python3
"""
Kronos Zero-Shot Single Prediction Script for Financial Time Series Forecasting

This script makes 24-hour predictions using the pre-trained Kronos model and creates
visualization plots. Supports energy, gold, and Apple stock price data.

Author: Financial Forecasting System
Date: 2026-01-09
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Add the core directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'core'))

from model_loader import KronosLoader

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = project_root / 'evaluation' / 'plots'
RESULTS_DIR = project_root / 'experiments' / 'zero_shot'

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Test parameters
SEQUENCE_LENGTH = 400  # Lookbook window (as in README example)
FORECAST_LENGTH = 24   # 24 hours ahead prediction
SAMPLE_COUNT = 1       # Number of samples for uncertainty estimation

# Available data configurations
DATA_CONFIGS = {
    'energy': {
        'path': project_root / 'data' / 'processed' / 'energy_2020-2025.csv',
        'description': 'Energy price data (2020-2025)',
        'has_negative_prices': True
    },
    'gold': {
        'path': project_root / 'data' / 'processed' / 'gold_2025_processed.csv',
        'description': 'Gold price data (2025)',
        'has_negative_prices': False
    },
    'apple': {
        'path': project_root / 'data' / 'processed' / 'apple_2025.csv',
        'description': 'Apple stock price data (2025)',
        'has_negative_prices': False
    }
}


# Model loading now handled by KronosLoader


def load_and_prepare_data(data_config_key):
    """Load financial data and prepare it in the format Kronos expects
    
    Args:
        data_config_key: Key from DATA_CONFIGS ('energy' or 'gold')
    
    Returns:
        DataFrame with standardized columns
    """
    if data_config_key not in DATA_CONFIGS:
        raise ValueError(f"Unknown data config: {data_config_key}. Available: {list(DATA_CONFIGS.keys())}")
    
    config = DATA_CONFIGS[data_config_key]
    data_path = config['path']
    
    print(f"Loading {config['description']}...")
    print(f"Data path: {data_path}")
    
    # Load the processed data
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Convert datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for negative prices (mainly for energy data)
    if config['has_negative_prices']:
        negative_count = (df['close'] < 0).sum()
        print(f"Found {negative_count} negative price periods ({negative_count/len(df)*100:.1f}%)")
    
    # Create amount column if missing (volume * average price)
    if 'amount' not in df.columns:
        if 'avg_price' in df.columns:
            # Energy data has avg_price column
            df['amount'] = df['volume'] * df['avg_price']
            print("✓ Created amount column using volume * avg_price")
        else:
            # Gold data: use average of OHLC
            avg_price = df[['open', 'high', 'low', 'close']].mean(axis=1)
            df['amount'] = df['volume'] * avg_price
            print("✓ Created amount column using volume * average(OHLC)")
    
    # Ensure we have required columns: open, high, low, close, volume, amount
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✓ Data prepared with shape: {df.shape}")
    print(f"✓ Required columns present: {required_cols}")
    return df, config


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


def create_prediction_plot(test_timestamps, actual_prices_clean, predicted_prices, metrics, data_type, unit):
    """
    Create a comprehensive prediction plot with actual vs predicted prices
    
    Args:
        test_timestamps: List of datetime objects
        actual_prices_clean: Array of actual OHLC prices
        predicted_prices: Array of predicted OHLC prices
        metrics: Dictionary with performance metrics
        data_type: Type of data ('energy' or 'gold')
        unit: Price unit string
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.2)
    
    # Convert timestamps to datetime objects if needed
    timestamps = [pd.to_datetime(ts) if isinstance(ts, str) else ts for ts in test_timestamps]
    
    # Main price comparison plot
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot actual prices
    ax1.plot(timestamps, actual_prices_clean[:, 3], 'b-', linewidth=2, label='Actual Close', alpha=0.8)
    ax1.plot(timestamps, actual_prices_clean[:, 0], 'b--', linewidth=1, label='Actual Open', alpha=0.6)
    
    # Plot predicted prices
    ax1.plot(timestamps, predicted_prices[:, 3], 'r-', linewidth=2, label='Predicted Close', alpha=0.8)
    ax1.plot(timestamps, predicted_prices[:, 0], 'r--', linewidth=1, label='Predicted Open', alpha=0.6)
    
    # Fill area between high and low for actual prices
    ax1.fill_between(timestamps, actual_prices_clean[:, 2], actual_prices_clean[:, 1],
                     alpha=0.2, color='blue', label='Actual High-Low Range')
    
    # Fill area between high and low for predicted prices
    ax1.fill_between(timestamps, predicted_prices[:, 2], predicted_prices[:, 1],
                     alpha=0.2, color='red', label='Predicted High-Low Range')
    
    ax1.set_title(f'{data_type.title()} Price Prediction - 24H Forecast', fontsize=16, fontweight='bold')
    ax1.set_ylabel(f'Price ({unit})', fontsize=12)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d.%m'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    
    # Error analysis plot
    ax2 = fig.add_subplot(gs[1, 0])
    errors = predicted_prices[:, 3] - actual_prices_clean[:, 3]  # Close price errors
    ax2.bar(range(len(errors)), errors, color=['red' if e > 0 else 'blue' for e in errors], alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Prediction Errors (Close Price)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'Error ({unit})', fontsize=10)
    ax2.set_xlabel('Hour', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Metrics display
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    metrics_text = f"""
Performance Metrics (Close Price):
  • MAE: {metrics['close']['mae']:.3f} {unit}
  • RMSE: {metrics['close']['rmse']:.3f} {unit}
  • MAPE: {metrics['close']['mape']:.1f}%

Price Statistics:
  • Actual Range: {actual_prices_clean[:, 3].min():.2f} - {actual_prices_clean[:, 3].max():.2f}
  • Predicted Range: {predicted_prices[:, 3].min():.2f} - {predicted_prices[:, 3].max():.2f}
  • Actual Avg: {actual_prices_clean[:, 3].mean():.2f}
  • Predicted Avg: {predicted_prices[:, 3].mean():.2f}
"""
    
    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Scatter plot: Actual vs Predicted
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(actual_prices_clean[:, 3], predicted_prices[:, 3], alpha=0.7, color='purple')
    
    # Perfect prediction line
    min_price = min(actual_prices_clean[:, 3].min(), predicted_prices[:, 3].min())
    max_price = max(actual_prices_clean[:, 3].max(), predicted_prices[:, 3].max())
    ax4.plot([min_price, max_price], [min_price, max_price], 'k--', alpha=0.5, label='Perfect Prediction')
    
    ax4.set_xlabel(f'Actual Price ({unit})', fontsize=10)
    ax4.set_ylabel(f'Predicted Price ({unit})', fontsize=10)
    ax4.set_title('Actual vs Predicted (Close)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Residual analysis
    ax5 = fig.add_subplot(gs[2, 1])
    residuals = predicted_prices[:, 3] - actual_prices_clean[:, 3]
    ax5.plot(timestamps, residuals, 'go-', alpha=0.7, markersize=4)
    ax5.axhline(y=0, color='red', linestyle='-', linewidth=1)
    ax5.fill_between(timestamps, residuals, alpha=0.3, color='green')
    ax5.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    ax5.set_ylabel(f'Residual ({unit})', fontsize=10)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'Kronos {data_type.title()} Prediction Analysis - 24H Forecast',
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save the plot
    plot_filename = f'kronos_{data_type}_prediction_24h.png'
    plot_path = OUTPUT_DIR / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()
    
    return plot_path


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Kronos Zero-Shot 24H Financial Time Series Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available data configurations:
{chr(10).join(f'  {key}: {config["description"]}' for key, config in DATA_CONFIGS.items())}

Examples:
  # Predict with energy data (default, last 24 hours)
  python kronos_zero_shot_single.py --data energy
  
  # Predict with Apple data
  python kronos_zero_shot_single.py --data apple
  
  # Predict for a specific date range (Apple stock from March 1-2, 2025)
  python kronos_zero_shot_single.py --data apple --start-date "2025-03-01" --end-date "2025-03-02"
  
  # Predict with specific datetime (24 hours starting from June 15, 2025 at 15:30)
  python kronos_zero_shot_single.py --data apple --start-date "2025-06-15 15:30:00"
        """
    )
    
    parser.add_argument('--data',
                       choices=list(DATA_CONFIGS.keys()),
                       default='energy',
                       help='Data type to use for prediction')
    parser.add_argument('--offset',
                       type=float,
                       default=100.0,
                       help='Price offset for handling negative prices (only applied to energy data)')
    parser.add_argument('--start-date',
                       type=str,
                       default=None,
                       help='Start date for prediction period (YYYY-MM-DD HH:MM:SS or YYYY-MM-DD). If not specified, uses last 24 hours.')
    parser.add_argument('--end-date',
                       type=str,
                       default=None,
                       help='End date for prediction period (YYYY-MM-DD HH:MM:SS or YYYY-MM-DD). If not specified, uses 24 hours after start-date.')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("KRONOS ZERO-SHOT 24H PREDICTION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data type: {args.data}")
    print(f"Sequence length: {SEQUENCE_LENGTH} hours")
    print(f"Forecast length: {FORECAST_LENGTH} hours")
    print()
    
    try:
        # 1. Load predictor using KronosLoader with detailed logging
        print("🔧 Loading Kronos model and tokenizer...")
        print(f"   Cache directory: {project_root / 'models' / 'model_cache'}")
        print(f"   Target device: {DEVICE}")
        
        predictor = KronosLoader.get_predictor(device=DEVICE)
        
        # Debug: Check if predictor loaded correctly
        print("✓ Kronos predictor loaded successfully")
        print(f"   Model type: {type(predictor.model).__name__}")
        print(f"   Tokenizer type: {type(predictor.tokenizer).__name__}")
        print(f"   Predictor device: {predictor.device}")
        
        # Check model parameters
        total_params = sum(p.numel() for p in predictor.model.parameters())
        trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Check if model is on correct device
        model_devices = {name: str(param.device) for name, param in predictor.model.named_parameters()}
        unique_devices = set(model_devices.values())
        print(f"   Model parameters on devices: {unique_devices}")
        
        # 2. Load and prepare data
        df, config = load_and_prepare_data(args.data)
        
        # 3. Handle negative prices with offset (only for energy data)
        price_offset = 0.0
        if config['has_negative_prices']:
            df_offset, price_offset = add_negative_price_offset(df, offset_value=args.offset)
        else:
            df_offset = df.copy()
            print("✓ No negative price handling needed for this data type")
        
        # 4. Split data for actual comparison
        print("Splitting data for actual comparison...")
        
        # Handle custom date range if provided
        if args.start_date:
            try:
                # Parse start date
                if len(args.start_date.split(' ')) == 1:
                    # Only date provided, assume start of trading day
                    start_date = pd.to_datetime(args.start_date + " 14:30:00")
                else:
                    start_date = pd.to_datetime(args.start_date)
                
                # Parse end date or calculate from start date
                if args.end_date:
                    if len(args.end_date.split(' ')) == 1:
                        # Only date provided, assume end of trading day
                        end_date = pd.to_datetime(args.end_date + " 20:30:00")
                    else:
                        end_date = pd.to_datetime(args.end_date)
                else:
                    # Use 24 hours from start date
                    end_date = start_date + pd.Timedelta(hours=24)
                
                print(f"Using custom date range: {start_date} to {end_date}")
                
                # Find indices for the specified date range
                mask = (df_offset['datetime'] >= start_date) & (df_offset['datetime'] <= end_date)
                test_indices = df_offset.index[mask].tolist()
                
                if len(test_indices) == 0:
                    raise ValueError(f"No data found for the specified date range: {start_date} to {end_date}")
                elif len(test_indices) < 24:
                    print(f"Warning: Only {len(test_indices)} data points found in the specified range (expected 24)")
                
                # Get test data
                test_df = df_offset.iloc[test_indices].copy()
                
                # Get context data (400 hours before the test period)
                context_start_idx = max(0, test_indices[0] - 400)
                context_end_idx = test_indices[0]
                
                if context_end_idx - context_start_idx < 400:
                    print(f"Warning: Only {context_end_idx - context_start_idx} context points available (expected 400)")
                
                context_df = df_offset.iloc[context_start_idx:context_end_idx].copy()
                
            except Exception as e:
                print(f"Error parsing dates: {e}")
                print("Using default behavior (last 24 hours)")
                test_df = df_offset.iloc[-24:].copy()
                context_df = df_offset.iloc[-400-24:-24].copy()
        else:
            # Default behavior: use last 24 hours as test set
            test_df = df_offset.iloc[-24:].copy()  # Last 24 hours for testing
            context_df = df_offset.iloc[-400-24:-24].copy()  # 400 hours before test period
        
        # Ensure we have enough context data
        if len(context_df) < SEQUENCE_LENGTH:
            print(f"Warning: Context data has only {len(context_df)} points, but model needs {SEQUENCE_LENGTH}")
            # Use all available data as context
            context_df = df_offset.iloc[:test_df.index[0]].copy()
            if len(context_df) < SEQUENCE_LENGTH:
                # Take the last SEQUENCE_LENGTH points from available data
                context_df = df_offset.iloc[max(0, test_df.index[0] - SEQUENCE_LENGTH):test_df.index[0]].copy()
        
        print(f"Context data: {len(context_df)} points from {context_df['datetime'].iloc[0]} to {context_df['datetime'].iloc[-1]}")
        print(f"Test data: {len(test_df)} points from {test_df['datetime'].iloc[0]} to {test_df['datetime'].iloc[-1]}")
        
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
            
            # Determine data type and units for display
            data_type = args.data
            if data_type == 'energy':
                unit = '€/MWh'
            elif data_type == 'gold':
                unit = 'USD/oz'
            elif data_type == 'apple':
                unit = 'USD'
            else:
                unit = 'units'
            
            # Display results summary
            print(f"\n✓ 24-hour prediction completed successfully!")
            print(f"Test period: {test_timestamps[0]} to {test_timestamps[-1]}")
            print(f"\nPerformance Metrics (Close Price):")
            print(f"  MAE: {metrics['close']['mae']:.3f} {unit}")
            print(f"  RMSE: {metrics['close']['rmse']:.3f} {unit}")
            print(f"  MAPE: {metrics['close']['mape']:.3f}%")
            print(f"\nPrice summary ({unit}):")
            print(f"  Actual close range: {actual_prices_clean[:, 3].min():.3f} - {actual_prices_clean[:, 3].max():.3f}")
            print(f"  Predicted close range: {predicted_prices[:, 3].min():.3f} - {predicted_prices[:, 3].max():.3f}")
            print(f"  Actual close average: {actual_prices_clean[:, 3].mean():.3f}")
            print(f"  Predicted close average: {predicted_prices[:, 3].mean():.3f}")
            
            # Create and display prediction plot
            print(f"\n📊 Creating prediction visualization...")
            plot_path = create_prediction_plot(
                test_timestamps=test_timestamps,
                actual_prices_clean=actual_prices_clean,
                predicted_prices=predicted_prices,
                metrics=metrics,
                data_type=data_type,
                unit=unit
            )
            
        else:
            print("Failed to generate prediction")
            return 1
            
        print("\n" + "=" * 60)
        print(f"KRONOS 24H {data_type.upper()} PREDICTION COMPLETED")
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