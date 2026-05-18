import sys
from pathlib import Path
from typing import Optional, List
import yaml

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from data.factory import DataFactory


def _load_symbols_from_yaml(config_path: str, top_level_key: str) -> List[str]:
    """Loads symbols from a YAML config file."""
    path = Path(config_path)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [asset["symbol"] for asset in data.get(top_level_key, [])]

def prepare_kronos_data(
    train_output_path="data/processed/train_data_kronos.arrow",
    val_output_path="data/processed/val_data_kronos.arrow",
    train_end="2018-12-31",
    val_start="2019-01-01",
    val_end="2020-12-31",
    min_length=64,
    config_path: str = "config/energy_assets_train.yaml",
    holdout_config_path: Optional[str] = None, # Neuer Parameter für Asset-basierten Holdout
    max_data_date: Optional[str] = None,
):
    """
    Konvertiert Finanzdaten für Kronos (OHLCV + Amount).

    Kronos benötigt: [open, high, low, close, volume, amount]

    Args:
        train_output_path (str): Pfad zur Speicherung der Trainingsdaten im Arrow-Format.
        val_output_path (str): Pfad zur Speicherung der Validierungsdaten im Arrow-Format.
        train_end (str): Enddatum für den Trainingsdatensatz.
        val_start (str): Startdatum für den Validierungsdatensatz.
        val_end (str): Enddatum für den Validierungsdatensatz.
        min_length (int): Minimale Länge einer Zeitreihe, um berücksichtigt zu werden.
        config_path (str): Pfad zur YAML-Konfigurationsdatei der Assets.
        holdout_config_path (Optional[str]): Optionaler Pfad zu einer YAML-Konfigurationsdatei,
                                              die Assets enthält, die vom Training/Validierung ausgeschlossen werden sollen.
        max_data_date (Optional[str]): Optionales Enddatum für alle Daten, die verarbeitet werden.
                                        Daten nach diesem Datum werden ignoriert. Dies ist nützlich,
                                        um einen finalen Holdout-Zeitraum von den Trainings- und
                                        Validierungsdaten abzugrenzen.
    """
    # tz-naive: DataFactory liefert tz-naive DatetimeIndex
    train_end_ts = pd.Timestamp(train_end)
    val_start_ts = pd.Timestamp(val_start)
    val_end_ts = pd.Timestamp(val_end)
    max_data_date_ts = pd.Timestamp(max_data_date) if max_data_date else None
    factory = DataFactory(config_path=config_path)
    
    all_tickers = factory.get_tickers()
    tickers_to_process = set(all_tickers)

    if holdout_config_path:
        holdout_symbols = _load_symbols_from_yaml(holdout_config_path, top_level_key="holdout_assets")
        if holdout_symbols:
            print(f"Excluding {len(holdout_symbols)} holdout assets from {holdout_config_path}")
            tickers_to_process = tickers_to_process - set(holdout_symbols)
            
    tickers = sorted(list(tickers_to_process))
    
    train_dataset = []
    val_dataset = []
    
    skipped_count = 0
    processed_count = 0
    
    print(f"🔄 Verarbeite {len(tickers)} Tickers für Kronos...")
    print(f"   Training:   bis {train_end_ts.date()}")
    print(f"   Validation: {val_start_ts.date()} - {val_end_ts.date()}")
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

            # Apply overall max_data_date filter if provided (for holdout)
            if max_data_date_ts:
                df_clean = df_clean[df_clean.index <= max_data_date_ts]
                if df_clean.empty:
                    print(f"⚠️  Keine Daten nach {max_data_date_ts.date()}")
                    skipped_count += 1
                    continue
                if len(df_clean) < min_length:
                    print(f"⚠️  Zu kurz nach {max_data_date_ts.date()} ({len(df_clean)})")
                    skipped_count += 1
                    continue
            
            # Add Volume if missing
            if 'Volume' not in df_clean.columns:
                df_clean['Volume'] = 0.0
            
            # Calculate Amount (Volume * Average Price)
            df_clean['Amount'] = df_clean['Volume'] * df_clean[['Open', 'High', 'Low', 'Close']].mean(axis=1)
            
            # Training Set
            # Training Set (constrained by train_end_ts and potentially max_data_date_ts)
            train_df = df_clean[df_clean.index <= train_end_ts]
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
                    "start": train_df.index[0],
                    "target": target,
                    "item_id": f"{ticker}_train"
                }
                train_dataset.append(train_entry)
            
            # Validation Set (constrained by val_start_ts, val_end_ts and potentially max_data_date_ts)
            val_df = df_clean[(df_clean.index >= val_start_ts) & (df_clean.index <= val_end_ts)]
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
                    "start": val_df.index[0],
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/energy_assets_train.yaml")
    args = parser.parse_args()
    prepare_kronos_data(config_path=args.config)
