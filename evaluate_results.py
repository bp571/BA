import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def evaluate_study(file_path="final_energy_study.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    rows = []

    for ticker, metrics in summary.items():
        rows.append({
            'Ticker': ticker,
            'IC': metrics['IC_Mean'],
            'IC_Lower': metrics['IC_CI95'][0],
            'IC_Upper': metrics['IC_CI95'][1],
            'Is_Lagging': metrics['Is_Lagging'],
            'MASE': metrics['MASE']
        })

    df = pd.DataFrame(rows).sort_values('IC', ascending=True)

    # 1. Statistiken berechnen
    mean_ic = df['IC'].mean()
    print(f"--- Analyse abgeschlossen ---")
    print(f"Durchschnittlicher IC über alle Assets: {mean_ic:.4f}")
    print(f"Anzahl Assets mit positivem IC: {(df['IC'] > 0).sum()} von {len(df)}")
    print(f"Anzahl Lagging-Modelle: {df['Is_Lagging'].sum()}")

    # 2. Visualisierung (IC mit Fehlerbalken)
    plt.figure(figsize=(12, 8))
    
    # Fehlerbalken berechnen (Abstand vom Mean)
    y_err = [df['IC'] - df['IC_Lower'], df['IC_Upper'] - df['IC']]
    
    colors = ['red' if x else 'skyblue' for x in df['Is_Lagging']]
    
    bars = plt.barh(df['Ticker'], df['IC'], xerr=y_err, color=colors, capsize=5, label='IC (Mean)')
    
    plt.axvline(0, color='black', linestyle='-', linewidth=1)
    plt.axvline(mean_ic, color='green', linestyle='--', label=f'Gesamt-Schnitt ({mean_ic:.3f})')
    
    plt.title(f"Information Coefficient (IC) pro Asset inkl. 95% Konfidenzintervall\n"
              f"Context: {data['params']['context_steps']}, Forecast: {data['params']['forecast_steps']}", fontsize=14)
    plt.xlabel("IC (Richtungsvorhersage-Güte)", fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ic_evaluation_energy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    evaluate_study("results/final_energy_study.json")