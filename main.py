import json
from datetime import datetime
from pathlib import Path

from data.factory import DataFactory
from core.model_loader import load_kronos_predictor 
from experiments.runner import run_rolling_benchmark

def main():
    # 1. Setup
    factory = DataFactory()
    predictor = load_kronos_predictor() 
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Konfiguration für den Rolling Forecast 
    params = {
        'context_steps': 80,
        'forecast_steps': 12,
        'stride_steps': 12,
        'steps': 100
    }
    
    # 2. Assets laden
    tickers = factory.get_energy_tickers()
    if not tickers:
        print("Keine Ticker in assets.yaml gefunden!")
        return

    all_results = {}

    # 3. Loop über alle 20+ Energie-Assets
    for ticker in tickers:
        try:
            # Daten laden (lokal oder yfinance)
            df = factory.load_or_download(ticker)
            if df.empty: continue
            
            # Benchmark ausführen 
            result = run_rolling_benchmark(predictor, df, ticker, params)
            
            if result:
                all_results[ticker] = result['metrics']
                
                # Zwischenspeichern für jedes Asset (Sicherheit)
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
            'params': params,
            'summary': all_results
        }, f, indent=4)
    
    print(f"Benchmark abgeschlossen. Ergebnisse in {results_dir}")

if __name__ == "__main__":
    main()