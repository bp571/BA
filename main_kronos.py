"""
Multi-Asset Forecasting mit Batch-Processing

Optimierte Version, die Kronos' predict_batch() Fähigkeit nutzt
für drastisch schnellere Verarbeitung mehrerer Assets.

Performance-Ziel: 30-40x Beschleunigung gegenüber sequentieller Verarbeitung
(von ~70 Minuten auf ~2-5 Minuten)
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

from data.factory import DataFactory
from core.model_loader import load_kronos_predictor
from core.reproducibility import set_all_seeds
from experiments.runner import run_rolling_benchmark_multi_asset
from tqdm import tqdm


def main():
    # Setze Random Seeds für Reproduzierbarkeit
    set_all_seeds(seed=42)
    
    print("=" * 70)
    print("🚀 MULTI-ASSET BATCH FORECASTING PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Initialisierung
    print("\n📦 Lade Komponenten...")
    factory = DataFactory()
    predictor = load_kronos_predictor()
    results_dir = Path("results_kronos")
    results_dir.mkdir(exist_ok=True)
    
    # Basis-Parameter (identisch zu main.py)
    base_params = {
        'context_steps': 80,
        'forecast_steps': 12,
        'stride_steps': 12,
        'steps': 120  # maximal möglich mit Daten seit 1.1.20
    }
    
    print(f"\n⚙️  Parameter:")
    for key, value in base_params.items():
        print(f"   - {key}: {value}")
    
    # 2. Assets laden
    print("\n📥 Lade Asset-Daten...")
    tickers = factory.get_energy_tickers()
    if not tickers:
        print("❌ Keine Ticker in assets.yaml gefunden!")
        return
    
    print(f"   {len(tickers)} Tickers gefunden")
    
    # Lade alle Asset-Daten in einem Durchgang
    asset_data = {}
    skipped_tickers = []
    
    for ticker in tqdm(tickers, desc="Lade Daten"):
        try:
            df = factory.load_or_download(ticker)
            if df.empty:
                skipped_tickers.append(ticker)
                continue
            
            # Dynamische Berechnung der maximal möglichen Steps
            n_total = len(df)
            c = base_params['context_steps']
            f = base_params['forecast_steps']
            s = base_params['stride_steps']
            
            max_steps = (n_total - c - f) // s + 1
            
            if max_steps <= 0:
                skipped_tickers.append(ticker)
                continue
            
            asset_data[ticker] = df
            
        except Exception as e:
            print(f"\n⚠️  {ticker}: Laden fehlgeschlagen - {e}")
            skipped_tickers.append(ticker)
    
    if skipped_tickers:
        print(f"\n⚠️  {len(skipped_tickers)} Tickers übersprungen: {', '.join(skipped_tickers)}")
    
    if not asset_data:
        print("❌ Keine gültigen Assets geladen!")
        return
    
    print(f"\n✅ {len(asset_data)} Assets erfolgreich geladen")
    
    # 3. BATCH PROCESSING - Der entscheidende Unterschied!
    print("\n" + "=" * 70)
    print("🔥 STARTE BATCH PREDICTION (Hier geschieht die Magie!)")
    print("=" * 70)
    
    batch_start = time.time()
    
    # Batch-Größe kann basierend auf verfügbarem GPU-Speicher angepasst werden
    # Für die meisten GPUs (8-16GB) ist 32-64 ein guter Startwert
    BATCH_SIZE = 48  
    
    all_results = run_rolling_benchmark_multi_asset(
        predictor=predictor,
        asset_data_dict=asset_data,
        params=base_params,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    
    batch_duration = time.time() - batch_start
    print(f"\n⏱️  Batch-Verarbeitung abgeschlossen in {batch_duration:.1f} Sekunden")
    
    # 4. Ergebnisse speichern
    print("\n💾 Speichere Ergebnisse...")
    
    final_summary = {}
    for ticker, result in all_results.items():
        if result:
            final_summary[ticker] = result['metrics']
            
            # Speichere detaillierte Ergebnisse pro Asset
            output_path = results_dir / f"result_{ticker}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
    
    # Gesamtergebnis speichern
    final_path = results_dir / "final_energy_study.json"
    with open(final_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'random_seed': 42,
            'params': base_params,
            'batch_size': BATCH_SIZE,
            'processing_time_seconds': batch_duration,
            'n_assets_processed': len(all_results),
            'n_assets_total': len(tickers),
            'summary': final_summary
        }, f, indent=4)
    
    # 5. Performance-Zusammenfassung
    total_duration = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE ABGESCHLOSSEN!")
    print("=" * 70)
    print(f"\n📊 Performance-Zusammenfassung:")
    print(f"   - Gesamtzeit: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"   - Batch-Verarbeitung: {batch_duration:.1f}s ({batch_duration/60:.1f} min)")
    print(f"   - Assets verarbeitet: {len(all_results)}/{len(tickers)}")
    print(f"   - Batch-Größe: {BATCH_SIZE}")
    
    if all_results:
        total_windows = sum(r['metrics'].get('N_Windows', 0) for r in all_results.values())
        total_predictions = sum(r['metrics'].get('N_Predictions', 0) for r in all_results.values())
        avg_time_per_window = batch_duration / total_windows if total_windows > 0 else 0
        
        print(f"   - Gesamt Windows: {total_windows}")
        print(f"   - Gesamt Vorhersagen: {total_predictions}")
        print(f"   - Durchschn. Zeit/Window: {avg_time_per_window:.3f}s")
        print(f"   - Windows/Sekunde: {total_windows/batch_duration:.1f}")
    
    print(f"\n📁 Ergebnisse gespeichert in: {results_dir}")
    print(f"   - Finale Zusammenfassung: {final_path}")
    print(f"   - Einzelergebnisse: result_{{ticker}}.json")
    
    print("\n" + "=" * 70)
    
    # Optionaler Vergleich mit sequentieller Version
    print("\n💡 Performance-Vergleich:")
    estimated_sequential_time = total_windows * 2.5  # ~2.5s pro Window sequentiell
    speedup = estimated_sequential_time / batch_duration if batch_duration > 0 else 0
    print(f"   - Geschätzte sequentielle Zeit: {estimated_sequential_time:.0f}s ({estimated_sequential_time/60:.1f} min)")
    print(f"   - Tatsächliche Batch-Zeit: {batch_duration:.1f}s ({batch_duration/60:.1f} min)")
    print(f"   - Beschleunigungsfaktor: ~{speedup:.1f}x schneller! 🚀")
    print("=" * 70)


if __name__ == "__main__":
    main()
