"""
Optimierter Rolling Forecast Vergleich (Chronos vs Kronos)
- Direkte Funktionsaufrufe ohne subprocess
- In-Memory Vergleich ohne CSV-Export
- Standardisierte Parameter für beide Modelle
"""
import sys
from pathlib import Path
from tqdm import tqdm

# Add local modules to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'experiments' / 'zero_shot'))

from forecast_common import DEFAULT_PARAMS, print_comparison_results
from chronos_rolling_forecast import run_rolling_forecast as chronos_forecast
from kronos_rolling_forecast import run_rolling_forecast as kronos_forecast


def compare_forecasts(data_path=None, start_date=None, steps=None, context_hours=None, forecast_hours=None, run_both=True, run_chronos=True, run_kronos=True):
    """
    Führt optimierten Vergleich beider Rolling Forecasts durch
    
    Args:
        data_path: Pfad zur Eingabedatei (verwendet DEFAULT_PARAMS wenn None)
        start_date: Start-Datum (verwendet DEFAULT_PARAMS wenn None)
        steps: Anzahl Forecast-Tage (verwendet DEFAULT_PARAMS wenn None)
        context_hours: Kontext-Länge in Stunden (verwendet DEFAULT_PARAMS wenn None)
        forecast_hours: Forecast-Horizont in Stunden (verwendet DEFAULT_PARAMS wenn None)
        run_both: Wenn True, führe beide Modelle aus
        run_chronos: Wenn True und run_both=False, führe nur Chronos aus
        run_kronos: Wenn True und run_both=False, führe nur Kronos aus
    
    Returns:
        Dictionary mit Ergebnissen beider Modelle
    """
    
    # Use parameters or defaults
    params = {
        'data_path': data_path or DEFAULT_PARAMS['data_path'],
        'start_date': start_date or DEFAULT_PARAMS['start_date'], 
        'steps': steps or DEFAULT_PARAMS['steps'],
        'context_hours': context_hours or DEFAULT_PARAMS['context_hours'],
        'forecast_hours': forecast_hours or DEFAULT_PARAMS['forecast_hours']
    }
    
    print("🔧 OPTIMIZED FORECAST COMPARISON")
    print("="*50)
    print(f"📅 Start Date: {params['start_date']}")
    print(f"📊 Forecast Days: {params['steps']}")
    print(f"⏱️ Context: {params['context_hours']}h, Forecast: {params['forecast_hours']}h")
    print(f"📂 Data File: {params['data_path']}")
    print("="*50)
    
    results = {}
    
    # Determine which models to run
    models_to_run = []
    if run_both:
        models_to_run = [('Chronos', chronos_forecast), ('Kronos', kronos_forecast)]
    else:
        if run_chronos:
            models_to_run.append(('Chronos', chronos_forecast))
        if run_kronos:
            models_to_run.append(('Kronos', kronos_forecast))
    
    if not models_to_run:
        print("❌ Keine Modelle ausgewählt")
        return {}
    
    # Run selected models
    for model_name, forecast_func in tqdm(models_to_run, desc="Running Models", unit="model"):
        print(f"\n🚀 Starting {model_name}...")
        
        try:
            model_results = forecast_func(
                data_path=params['data_path'],
                start_date=params['start_date'],
                steps=params['steps'],
                context_hours=params['context_hours'],
                forecast_hours=params['forecast_hours']
            )
            results[model_name.lower()] = model_results
            print(f"✅ {model_name} completed: {len(model_results)} days")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[model_name.lower()] = []
    
    # Compare results if both models were run and successful
    if len(results) == 2 and all(results.values()):
        chronos_results = results.get('chronos', [])
        kronos_results = results.get('kronos', [])
        
        if chronos_results and kronos_results:
            print_comparison_results(chronos_results, kronos_results)
        else:
            print("❌ Keine Ergebnisse für Vergleich verfügbar")
    elif len(results) == 1:
        model_name = list(results.keys())[0].title()
        result_count = len(list(results.values())[0])
        print(f"\n✅ {model_name} einzeln ausgeführt: {result_count} days")
    else:
        print("❌ Keine erfolgreichen Ergebnisse")
    
    return results


def list_available_data_files():
    """List available data files in data/processed directory"""
    import os
    processed_dir = "data/processed"
    if os.path.exists(processed_dir):
        files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        return files
    return []


if __name__ == "__main__":
    import argparse
    import os
    
    # Check available data files
    available_files = list_available_data_files()
    
    parser = argparse.ArgumentParser(
        description='Optimized Rolling Forecast Comparison (Chronos vs Kronos)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Optimierte Version ohne CSV-Export und subprocess.
Beide Modelle verwenden jetzt identische standardisierte Parameter.

Available data files in data/processed/:
{chr(10).join(f'  - {file}' for file in available_files) if available_files else '  No CSV files found'}

Examples:
  # Compare both models with default settings
  python compare_rolling_forecasts.py
  
  # Compare with specific data file
  python compare_rolling_forecasts.py --data-path data/processed/energy_2020-2025.csv
  
  # Compare with custom parameters
  python compare_rolling_forecasts.py --context-hours 168 --forecast-hours 12 --steps 10
  
  # Run only Chronos
  python compare_rolling_forecasts.py --chronos-only
  
  # Run only Kronos  
  python compare_rolling_forecasts.py --kronos-only
        """
    )
    
    parser.add_argument('--data-path', 
                        help=f'Path to data file (default: {DEFAULT_PARAMS["data_path"]})')
    parser.add_argument('--start-date', default=DEFAULT_PARAMS['start_date'],
                        help=f'Start date (default: {DEFAULT_PARAMS["start_date"]})')
    parser.add_argument('--steps', type=int, default=DEFAULT_PARAMS['steps'],
                        help=f'Number of forecast days (default: {DEFAULT_PARAMS["steps"]})')
    parser.add_argument('--context-hours', type=int, default=DEFAULT_PARAMS['context_hours'],
                        help=f'Context length in hours (default: {DEFAULT_PARAMS["context_hours"]})')
    parser.add_argument('--forecast-hours', type=int, default=DEFAULT_PARAMS['forecast_hours'],
                        help=f'Forecast horizon in hours (default: {DEFAULT_PARAMS["forecast_hours"]})')
    parser.add_argument('--chronos-only', action='store_true',
                        help='Run only Chronos model')
    parser.add_argument('--kronos-only', action='store_true',
                        help='Run only Kronos model')
    parser.add_argument('--list-files', action='store_true',
                        help='List available data files and exit')
    
    args = parser.parse_args()
    
    # Handle list files option
    if args.list_files:
        print("Available data files in data/processed/:")
        for file in available_files:
            print(f"  - data/processed/{file}")
        exit(0)
    
    # Validate data path if provided
    if args.data_path and not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        print("\nAvailable files:")
        for file in available_files:
            print(f"  - data/processed/{file}")
        exit(1)
    
    # Determine run mode
    run_both = not (args.chronos_only or args.kronos_only)
    run_chronos = run_both or args.chronos_only
    run_kronos = run_both or args.kronos_only
    
    # Run comparison
    results = compare_forecasts(
        data_path=args.data_path,
        start_date=args.start_date,
        steps=args.steps,
        context_hours=args.context_hours,
        forecast_hours=args.forecast_hours,
        run_both=run_both,
        run_chronos=run_chronos,
        run_kronos=run_kronos
    )
    
    print(f"\n🏁 Comparison completed with {sum(len(r) for r in results.values())} total forecast days")