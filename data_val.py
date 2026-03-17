import matplotlib.pyplot as plt
from data.factory import DataFactory
import pandas as pd

def check_and_plot_assets():
    # Initialisierung der Factory (nutzt deine config/assets.yaml)
    factory = DataFactory()
    tickers = factory.get_tickers()
    
    print(f"🔍 Prüfe {len(tickers)} Assets auf Datenqualität...\n")
    
    # Grid-Layout berechnen
    cols = 3
    rows = (len(tickers) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Lade {ticker}...")
        df = factory.load_or_download(ticker)
        
        if df.empty:
            print(f"  ⚠️  KEINE DATEN für {ticker} gefunden!")
            continue
            
        data_len = len(df)
        # Check: 512 (Context) + 64 (Buffer/Min-Split)
        status = "✅ OK" if data_len > 1000 else "❌ ZU KURZ (Gefahr von 0-Train)"
        print(f"  📊 Länge: {data_len} Tage | {status}")

        # Plotten
        ax = axes[i]
        ax.plot(df.index, df['Close'], label='Close')
        ax.set_title(f"{ticker} (N={data_len})")
        ax.grid(True, alpha=0.3)
        
        if data_len <= 600:
            ax.set_facecolor('#ffe6e6') # Roter Hintergrund bei kurzen Daten

    # Leere subplots entfernen
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_and_plot_assets()