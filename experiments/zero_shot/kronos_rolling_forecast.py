import pandas as pd
import torch
import sys
from pathlib import Path
from tqdm import tqdm

# Add the models/Kronos directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'models' / 'Kronos'))

from model import Kronos, KronosTokenizer, KronosPredictor
from forecast_common import DEFAULT_PARAMS, load_and_prepare_data, get_data_periods


def run_rolling_forecast(data_path=None, start_date=None, steps=None, context_hours=None, forecast_hours=None):
    """
    Optimiertes Rolling Forecast mit Kronos (In-Memory, ohne CSV)
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
    
    # 1. Load tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    
    # 2. Load and prepare data using common function
    df, use_cols, price_column = load_and_prepare_data(params['data_path'])
    
    # 3. Run rolling forecast
    results = []
    
    print(f"Starting Kronos Rolling Forecast from {params['start_date']} for {params['steps']} days")
    print(f"Context: {params['context_hours']}h, Forecast: {params['forecast_hours']}h")
    print(f"Using columns: {use_cols}")
    
    for i in tqdm(range(params['steps']), desc="Kronos Forecast"):
        # Get data periods using common function
        context_data, target_data = get_data_periods(
            df, params['start_date'], i, params['context_hours'], params['forecast_hours']
        )
        
        if context_data is None or target_data is None:
            continue
        
        try:
            # Make prediction
            pred_df = predictor.predict(
                df=context_data[use_cols],
                x_timestamp=context_data['datetime'],
                y_timestamp=target_data['datetime'],
                pred_len=params['forecast_hours'],
                T=1.0,
                top_p=0.9,
                sample_count=1
            )
            
            # Store results in standardized format
            results.append({
                "date": target_data['datetime'].iloc[0].strftime("%Y-%m-%d"),
                "actual": target_data[price_column].tolist(),
                "predicted": pred_df[price_column].tolist()
            })
            
        except Exception as e:
            print(f"Error on day {i+1}: {e}")
    
    print(f"Kronos completed: {len(results)}/{params['steps']} days")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Kronos Rolling Forecast - Optimized')
    parser.add_argument('--data-path', help='Path to CSV data file')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--steps', type=int, help='Number of forecast days')
    parser.add_argument('--context-hours', type=int, help='Context length in hours (must be < 512)')
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