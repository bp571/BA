import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add the core directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'core'))

from model_loader import ChronosLoader
from forecast_common import DEFAULT_PARAMS, load_and_prepare_data, get_data_periods

# Add the experiments directory to the Python path for metrics
sys.path.append(str(project_root / 'experiments'))
from metrics import calculate_all_metrics


def prepare_chronos_data(df_context, price_column='close'):
    """Prepare context data in Chronos format (univariate time series)"""
    # For Chronos, we use only the datetime and price column as target
    chronos_df = df_context[['datetime', price_column]].copy()
    chronos_df['id'] = 'price_series'
    chronos_df = chronos_df.rename(columns={price_column: 'target'})
    
    # Ensure regular frequency (hourly)
    chronos_df = chronos_df.set_index('datetime')
    chronos_df = chronos_df.asfreq('h', method='ffill')  # Fill gaps with forward fill
    chronos_df = chronos_df.reset_index()
    
    return chronos_df


def run_rolling_forecast(data_path=None, start_date=None, steps=None, context_hours=None, forecast_hours=None):
    """
    Optimiertes Rolling Forecast mit Chronos (In-Memory, ohne CSV)
    Nutzt standardisierte Parameter aus forecast_common
    """
    # Use standardized parameters
    params = {
        'data_path': data_path or DEFAULT_PARAMS['data_path'],
        'start_date': start_date or DEFAULT_PARAMS['start_date'],
        'steps': steps or DEFAULT_PARAMS['steps'],
        'context_hours': context_hours or DEFAULT_PARAMS['context_hours'],
        'forecast_hours': forecast_hours or DEFAULT_PARAMS['forecast_hours']
    }
    
    # 1. Load Chronos pipeline
    pipeline = ChronosLoader.load("amazon/chronos-2", device_map="cpu")
    
    # 2. Load and prepare data using common function
    df, use_cols, price_column = load_and_prepare_data(params['data_path'], price_column='close')
    
    # 3. Run rolling forecast
    results = []
    
    print(f"Starting Chronos Rolling Forecast from {params['start_date']} for {params['steps']} days")
    print(f"Context: {params['context_hours']}h, Forecast: {params['forecast_hours']}h")
    print(f"Using price column: {price_column}")

    for i in tqdm(range(params['steps']), desc="Chronos Forecast"):
        # Get data periods using common function
        context_data, target_data = get_data_periods(
            df, params['start_date'], i, params['context_hours'], params['forecast_hours']
        )
        
        if context_data is None or target_data is None:
            continue
        
        try:
            # Prepare data in Chronos format
            chronos_context = prepare_chronos_data(context_data, price_column)
            
            # Make prediction using Chronos
            pred_df = pipeline.predict_df(
                chronos_context,
                prediction_length=params['forecast_hours'],
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="datetime",
                target="target"
            )
            
            # Extract median predictions (0.5 quantile)
            pred_values = pred_df['0.5'].values
            actual_values = target_data[price_column].values
            
            # Store results in standardized format
            results.append({
                "date": target_data['datetime'].iloc[0].strftime("%Y-%m-%d"),
                "actual": actual_values.tolist(),
                "predicted": pred_values.tolist()
            })

        except Exception as e:
            print(f"Error processing day {i+1}: {e}")

    print(f"Chronos completed: {len(results)}/{params['steps']} days")
    
    # Calculate and display enhanced metrics
    if results:
        print_enhanced_metrics(results)
    
    return results


def print_enhanced_metrics(results):
    """Zeigt erweiterte Metriken wie im Multi-Asset-Script"""
    print("\n" + "="*70)
    print("ENHANCED METRICS (SCALED & RETURN-BASED)")
    print("="*70)
    
    # Flatten all results
    all_actual = []
    all_predicted = []
    
    for r in results:
        all_actual.extend(r['actual'])
        all_predicted.extend(r['predicted'])
    
    # Calculate metrics using the enhanced metrics function
    metrics = calculate_all_metrics(np.array(all_actual), np.array(all_predicted))
    
    print(f"\n{'Metric':<20} {'Value':<12}")
    print("-" * 32)
    print(f"{'wMAPE (%)':<20} {metrics.get('wMAPE', 0.0):<12.1f}")
    print(f"{'MASE':<20} {metrics.get('MASE', 0.0):<12.3f}")
    print(f"{'IC_Return':<20} {metrics.get('IC_Return', 0.0):<12.3f}")
    print(f"{'IC_Lag_1':<20} {metrics.get('IC_Lag_1', 0.0):<12.3f}")
    print(f"{'Dir_Accuracy (%)':<20} {metrics.get('Directional_Accuracy', 0.0):<12.1f}")
    print("-" * 32)
    
    # Scientific insight
    if metrics.get('Is_Lagging', False):
        print("  [!] Info: Model shows lagging effect (IC_L1 > IC_R)")

    print(f"Total data points: {metrics.get('Count', 0)}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chronos Rolling Forecast - Optimized')
    parser.add_argument('--data-path', help='Path to data file (CSV format)')
    parser.add_argument('--start-date', help='Start date for rolling forecast (YYYY-MM-DD)')
    parser.add_argument('--steps', type=int, help='Number of forecast days')
    parser.add_argument('--context-hours', type=int, help='Context length in hours')
    parser.add_argument('--forecast-hours', type=int, help='Forecast horizon in hours')
    
    args = parser.parse_args()
    
    results = run_rolling_forecast(
        data_path=args.data_path,
        start_date=args.start_date,
        steps=args.steps,
        context_hours=args.context_hours,
        forecast_hours=args.forecast_hours
    )
    
    print(f"Generated {len(results)} forecast results")