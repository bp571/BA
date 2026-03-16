import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from data.factory import DataFactory


def prepare_chronos_data(
    train_output_path="data/processed/train_data.arrow",
    val_output_path="data/processed/val_data.arrow",
    val_split=0.2,
    min_length=64
):
    """
    Konvertiert Finanzdaten aus der DataFactory in das Chronos-kompatible Arrow-Format.
    
    Args:
        train_output_path: Pfad für Trainingsdaten
        val_output_path: Pfad für Validierungsdaten
        val_split: Anteil der Validierungsdaten (0.0-1.0)
        min_length: Minimale Länge der Zeitreihen (filtert zu kurze Reihen)
    """
    factory = DataFactory()
    tickers = factory.get_energy_tickers()
    
    train_dataset = []
    val_dataset = []
    
    skipped_count = 0
    processed_count = 0
    
    print(f"🔄 Verarbeite {len(tickers)} Tickers...")
    print(f"   Minimale Länge: {min_length}")
    print(f"   Validation Split: {val_split * 100:.1f}%\n")
    
    for ticker in tickers:
        print(f"   Verarbeite {ticker}...", end=" ")
        
        try:
            df = factory.load_or_download(ticker)
            
            if df.empty:
                print("⚠️  Leer, überspringe")
                skipped_count += 1
                continue
            
            # Prüfe Mindestlänge
            if len(df) < min_length:
                print(f"⚠️  Zu kurz ({len(df)} < {min_length}), überspringe")
                skipped_count += 1
                continue
            
            # Prüfe auf NaN-Werte und entferne sie
            df_clean = df.dropna(subset=["Close"])
            if len(df_clean) < min_length:
                print(f"⚠️  Zu viele NaNs ({len(df_clean)} < {min_length}), überspringe")
                skipped_count += 1
                continue
            
            # Konvertiere Index zu pandas Timestamp (falls nicht bereits)
            if not isinstance(df_clean.index, pd.DatetimeIndex):
                print(f"⚠️  Index ist kein DatetimeIndex, überspringe")
                skipped_count += 1
                continue
            
            # Erstelle Start-Timestamp als pandas Timestamp
            start = pd.Timestamp(df_clean.index[0])
            
            # Extrahiere Target als numpy array
            target = df_clean["Close"].values.astype(np.float32)
            
            # Split in Train/Val
            if val_split > 0:
                split_idx = int(len(target) * (1 - val_split))
                
                # Training-Teil
                train_entry = {
                    "start": start,
                    "target": target[:split_idx],
                    "item_id": f"{ticker}_train"
                }
                train_dataset.append(train_entry)
                
                # Validation-Teil
                val_start = start + pd.Timedelta(days=split_idx)  # Approximation
                val_entry = {
                    "start": val_start,
                    "target": target[split_idx:],
                    "item_id": f"{ticker}_val"
                }
                val_dataset.append(val_entry)
            else:
                # Nur Training (kein Split)
                entry = {
                    "start": start,
                    "target": target,
                    "item_id": ticker
                }
                train_dataset.append(entry)
            
            print(f"✅ {len(target)} Datenpunkte")
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Fehler: {e}")
            skipped_count += 1
            continue
    
    # Statistiken
    print(f"\n📊 Zusammenfassung:")
    print(f"   Verarbeitet: {processed_count}")
    print(f"   Übersprungen: {skipped_count}")
    print(f"   Training Zeitreihen: {len(train_dataset)}")
    if val_dataset:
        print(f"   Validation Zeitreihen: {len(val_dataset)}")
    
    if not train_dataset:
        raise ValueError("❌ Keine Trainingsdaten erstellt! Prüfe die Datenquelle.")
    
    # Speichern im Arrow-Format
    print(f"\n💾 Speichere Daten...")
    
    # Training
    train_path = Path(train_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        ArrowWriter(compression="lz4").write_to_file(
            train_dataset,
            path=train_path
        )
        print(f"   ✅ Training: {train_path}")
    except Exception as e:
        print(f"   ❌ Fehler beim Speichern von Training-Daten: {e}")
        raise
    
    # Validation (falls vorhanden)
    if val_dataset:
        val_path = Path(val_output_path)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            ArrowWriter(compression="lz4").write_to_file(
                val_dataset,
                path=val_path
            )
            print(f"   ✅ Validation: {val_path}")
        except Exception as e:
            print(f"   ❌ Fehler beim Speichern von Validation-Daten: {e}")
            raise
    
    print(f"\n✅ Datenvorbereitung abgeschlossen!")
    
    return {
        "train_path": train_path,
        "val_path": val_path if val_dataset else None,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset)
    }


if __name__ == "__main__":
    prepare_chronos_data()