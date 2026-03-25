import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from data.factory import DataFactory


def prepare_kronos_data(
    train_output_path="data/processed/train_data_kronos.arrow",
    val_output_path="data/processed/val_data_kronos.arrow",
    train_end="2018-12-31",
    val_start="2019-01-01",
    val_end="2020-12-31",
    min_length=64
):
    """
    Konvertiert Finanzdaten für Kronos (OHLCV + Amount).
    
    Kronos benötigt: [open, high, low, close, volume, amount]
    """
    train_end = pd.Timestamp(train_end, tz='UTC')
    val_start = pd.Timestamp(val_start, tz='UTC')
    val_end = pd.Timestamp(val_end, tz='UTC')
    factory = DataFactory()
    tickers = factory.get_tickers()
    
    train_dataset = []
    val_dataset = []
    
    skipped_count = 0
    processed_count = 0
    
    print(f"🔄 Verarbeite {len(tickers)} Tickers für Kronos...")
    print(f"   Training: 2010 - {train_end.date()}")
    print(f"   Validation: {val_start.date()} - {val_end.date()}")
    print(f"   Format: OHLCV + Amount\n")
    
    for ticker in tickers:
        print(f"   {ticker}...", end=" ")
        
        try:
            df = factory.load_or_download(ticker)
            
            if df.empty:
                print("⚠️  Leer")
                skipped_count += 1
                continue
            
            if len(df) < min_length:
                print(f"⚠️  Zu kurz ({len(df)})")
                skipped_count += 1
                continue
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"⚠️  Fehlende Spalten: {missing}")
                skipped_count += 1
                continue
            
            df_clean = df.dropna(subset=required_cols)
            if len(df_clean) < min_length:
                print(f"⚠️  Zu viele NaNs")
                skipped_count += 1
                continue
            
            if not isinstance(df_clean.index, pd.DatetimeIndex):
                print(f"⚠️  Kein DatetimeIndex")
                skipped_count += 1
                continue
            
            # Add Volume if missing
            if 'Volume' not in df_clean.columns:
                df_clean['Volume'] = 0.0
            
            # Calculate Amount (Volume * Average Price)
            df_clean['Amount'] = df_clean['Volume'] * df_clean[['Open', 'High', 'Low', 'Close']].mean(axis=1)
            
            # Training Set
            train_df = df_clean[df_clean.index <= train_end]
            if len(train_df) >= min_length:
                # Stack all features: [open, high, low, close, volume, amount]
                target = np.stack([
                    train_df['Open'].values,
                    train_df['High'].values,
                    train_df['Low'].values,
                    train_df['Close'].values,
                    train_df['Volume'].values,
                    train_df['Amount'].values
                ], axis=-1).astype(np.float32)
                
                train_entry = {
                    "start": pd.Timestamp(train_df.index[0]),
                    "target": target,
                    "item_id": f"{ticker}_train"
                }
                train_dataset.append(train_entry)
            
            # Validation Set
            val_df = df_clean[(df_clean.index >= val_start) & (df_clean.index <= val_end)]
            if len(val_df) >= min_length:
                target = np.stack([
                    val_df['Open'].values,
                    val_df['High'].values,
                    val_df['Low'].values,
                    val_df['Close'].values,
                    val_df['Volume'].values,
                    val_df['Amount'].values
                ], axis=-1).astype(np.float32)
                
                val_entry = {
                    "start": pd.Timestamp(val_df.index[0]),
                    "target": target,
                    "item_id": f"{ticker}_val"
                }
                val_dataset.append(val_entry)
            
            print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}")
            processed_count += 1
            
        except Exception as e:
            print(f"❌ {e}")
            skipped_count += 1
            continue
    
    print(f"\n📊 Zusammenfassung:")
    print(f"   Verarbeitet: {processed_count}")
    print(f"   Übersprungen: {skipped_count}")
    print(f"   Training: {len(train_dataset)} Zeitreihen")
    print(f"   Validation: {len(val_dataset)} Zeitreihen")
    
    if not train_dataset:
        raise ValueError("❌ Keine Trainingsdaten erstellt!")
    
    print(f"\n💾 Speichere Daten...")
    
    train_path = Path(train_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    ArrowWriter(compression="lz4").write_to_file(train_dataset, path=train_path)
    print(f"   ✅ Training: {train_path}")
    
    if val_dataset:
        val_path = Path(val_output_path)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        ArrowWriter(compression="lz4").write_to_file(val_dataset, path=val_path)
        print(f"   ✅ Validation: {val_path}")
    
    print(f"\n✅ Kronos-Datenvorbereitung abgeschlossen!")
    
    return {
        "train_path": train_path,
        "val_path": val_path if val_dataset else None,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset)
    }


if __name__ == "__main__":
    prepare_kronos_data()
