import os
import json
from datetime import datetime
from pathlib import Path

from data.factory import DataFactory
from core.model_loader import load_chronos_predictor
from experiments.runner import run_rolling_benchmark
from tqdm import tqdm

def main():
    # 1. Setup
    factory = DataFactory()
    predictor = load_chronos_predictor()
    results_dir = Path("results_chronos")
    results_dir.mkdir(exist_ok=True)
    
    # Deine fixen Basis-Parameter
    base_params = {
        'context_steps': 80,
        'forecast_steps': 12,
        'stride_steps': 12,
        'steps': 120  # maximal möglich mit Daten seit 1.1.20
    }
    
    # 2. Assets laden
    tickers = factory.get_energy_tickers()
    if not tickers:
        print("Keine Ticker in assets.yaml gefunden!")
        return

    all_results = {}

    # 3. Loop über alle Energie-Assets
    for ticker in tqdm(tickers, desc="Processing assets"):
        try:
            # Daten laden
            df = factory.load_or_download(ticker)
            if df.empty: 
                continue
            
            # --- Dynamische Berechnung der maximal möglichen Steps ---
            n_total = len(df)
            c = base_params['context_steps']
            f = base_params['forecast_steps']
            s = base_params['stride_steps']
            
            # Formel: Stellt sicher, dass wir nicht über den Anfang des DFs hinausgehen
            max_steps = (n_total - c - f) // s + 1
            
            # Kopie der Parameter für diesen Ticker erstellen
            current_params = base_params.copy()
            current_params['steps'] = max(0, min(base_params['steps'], max_steps))
            
            if current_params['steps'] == 0:
                print(f"\nSkipping {ticker}: Not enough data for one window.")
                continue
            # ---------------------------------------------------------

            # Benchmark ausführen mit angepassten Steps
            result = run_rolling_benchmark(predictor, df, ticker, current_params)
            
            if result:
                all_results[ticker] = result['metrics']
                
                output_path = results_dir / f"result_{ticker}.json"
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=4)
                    
        except Exception as e:
            print(f"Fehler bei {ticker}: {e}")

    # 4. Gesamtergebnis speichern
    final_path = results_dir / "final_energy_study.json"
    with open(final_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': 'Chronos',
            'params': base_params,
            'summary': all_results
        }, f, indent=4)
    
    print(f"Benchmark abgeschlossen. Ergebnisse in {results_dir}")

if __name__ == "__main__":
    main()
