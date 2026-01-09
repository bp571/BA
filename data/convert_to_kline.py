import pandas as pd
import os
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.asset_manager import AssetManager

def convert_asset_to_hourly_candles(asset_name, start_date, end_date, output_file=None, asset_manager=None):
    """
    Fetch and convert asset data to hourly candlestick format using AssetManager
    
    Args:
        asset_name (str): Name of the asset (e.g., 'gold', 'silver', 'energy_price')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_file (str, optional): Path for the output file. Falls None, wird automatisch generiert.
        asset_manager (AssetManager, optional): AssetManager instance
    
    Returns:
        pd.DataFrame: DataFrame with hourly candlestick data
    """
    
    # Initialize AssetManager if not provided
    if asset_manager is None:
        try:
            asset_manager = AssetManager()
        except Exception as e:
            print(f"ERROR: Failed to initialize AssetManager: {e}")
            return None
    
    try:
        # Fetch standardized data from AssetManager
        print(f"Fetching {asset_name} data from {start_date} to {end_date}...")
        df = asset_manager.get_asset_data(asset_name, start_date, end_date, interval='1h')
        
        if df.empty:
            print(f"ERROR: No data returned for {asset_name}")
            return None
            
        print(f"Retrieved {len(df)} hourly records")
        
        # Data is already in hourly candlestick format from AssetManager
        # Just ensure proper ordering and add any missing calculations
        hourly_df = df.copy()
        
        # Ensure datetime is properly formatted
        hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
        
        # Add average price if not present
        if 'avg_price' not in hourly_df.columns:
            hourly_df['avg_price'] = (hourly_df['open'] + hourly_df['high'] +
                                     hourly_df['low'] + hourly_df['close']) / 4
        
        # Round values for consistency
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'avg_price']
        for col in numeric_columns:
            if col in hourly_df.columns:
                hourly_df[col] = hourly_df[col].round(6)
        
        # Sort by datetime
        hourly_df = hourly_df.sort_values('datetime').reset_index(drop=True)
        
        # Generate output filename if not provided
        if output_file is None:
            start_str = start_date.replace('-', '')
            end_str = end_date.replace('-', '')
            output_file = f"data/processed/{asset_name}_hourly_candles_{start_str}_{end_str}.csv"
        
        # Save to CSV
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            hourly_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
        except Exception as e:
            print(f"ERROR beim Speichern: {e}")
            return hourly_df
        
        # Print summary
        asset_info = asset_manager.get_asset_info(asset_name)
        print(f"Asset: {asset_info['name']} ({asset_info['symbol']})")
        print(f"Provider: {asset_info['provider']}")
        print(f"Records: {len(hourly_df)}")
        if 'close' in hourly_df.columns:
            print(f"Price range: {hourly_df['close'].min():.2f} - {hourly_df['close'].max():.2f} {asset_info.get('currency', 'USD')}")
        
        return hourly_df
        
    except Exception as e:
        print(f"ERROR processing {asset_name}: {e}")
        return None

def convert_15min_to_hourly_candles(input_file, output_file=None, volume_source=None):
    """
    Konvertiert 15-Minuten-Energiepreisdaten in stündliche Candlestick-Daten.
    
    Args:
        input_file (str): Pfad zur CSV-Datei mit 15-Minuten-Daten
        output_file (str, optional): Pfad für die Ausgabedatei. Falls None, wird automatisch generiert.
        volume_source (str, optional): Quelle für Volume-Berechnung: 'load', 'residual_load' oder None
    
    Returns:
        pd.DataFrame: DataFrame mit stündlichen Candlestick-Daten
    """
    
    # CSV-Datei laden
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"ERROR: Datei {input_file} nicht gefunden!")
        return None
    except Exception as e:
        print(f"ERROR beim Laden der Datei: {e}")
        return None
    
    # Datenvalidierung
    required_columns = ['price', 'datetime']
    if volume_source:
        required_columns.append(volume_source)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Fehlende Spalten: {missing_columns}")
        return None
    
    # Datetime-Spalte konvertieren
    if df['datetime'].dtype == 'object':
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except Exception as e:
            print(f"ERROR bei Datetime-Konvertierung: {e}")
            return None
    
    # Timestamp-Spalte erstellen falls nicht vorhanden
    if 'timestamp' not in df.columns:
        df['timestamp'] = (df['datetime'].astype('int64') // 10**6).astype('int64')  # Convert to milliseconds
    
    # Nach Datum sortieren
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Stunden-Index erstellen (abgerundet auf volle Stunden)
    df['hour'] = df['datetime'].dt.floor('h')
    
    hourly_candles = []
    grouped_hours = list(df.groupby('hour'))
    total_hours = len(grouped_hours)
    
    for i, (hour, group) in enumerate(grouped_hours, 1):
        # Fortschrittsanzeige
        current_date = hour.strftime('%Y-%m-%d %H:00')
        print(f"\rProcessing {i}/{total_hours}: {current_date}", end="", flush=True)
        
        # Sortiere nach Zeitstempel für korrekte Open/Close-Werte
        group = group.sort_values('datetime')
        
        # OHLC berechnen
        open_price = group.iloc[0]['price']   # Erster Wert der Stunde
        high_price = group['price'].max()     # Höchster Wert der Stunde
        low_price = group['price'].min()      # Niedrigster Wert der Stunde
        close_price = group.iloc[-1]['price'] # Letzter Wert der Stunde
        
        # Volume berechnen basierend auf volume_source
        if volume_source and volume_source in group.columns:
            # Summe der gewählten Spalte für die Stunde
            volume = round(group[volume_source].sum(), 2)
        else:
            # Standard: Anzahl der 15-Min-Intervalle (normalerweise 4)
            volume = len(group)
        
        # Durchschnittspreis für zusätzliche Info
        avg_price = group['price'].mean()
        
        hourly_candles.append({
            'datetime': hour,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'avg_price': round(avg_price, 2)
        })
    
    print()  # Neue Zeile nach Fortschrittsanzeige
    
    # DataFrame erstellen
    hourly_df = pd.DataFrame(hourly_candles)
    
    # Ausgabedatei bestimmen
    if output_file is None:
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join("data/processed", f"{input_name}_hourly_candles.csv")
    
    # In CSV speichern
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        hourly_df.to_csv(output_file, index=False)
        print(f"Gespeichert: {output_file}")
    except Exception as e:
        print(f"ERROR beim Speichern: {e}")
        return hourly_df
    
    return hourly_df

def main():
    """Hauptfunktion für Kommandozeilenausführung mit Asset-Manager Support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert data to hourly candlesticks - supports both legacy CSV files and new Asset Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New Asset Manager approach (recommended)
  python convert_to_kline.py --asset gold --start-date 2024-01-01 --end-date 2024-12-31
  python convert_to_kline.py --asset silver --start-date 2023-01-01 --end-date 2023-12-31
  
  # Legacy CSV file approach (backward compatibility)
  python convert_to_kline.py data/raw/smard_energy_data_2020_2025_combined.csv
  
  # List available assets
  python convert_to_kline.py --list-assets
        """
    )
    
    # Asset Manager options
    parser.add_argument('--asset',
                       help='Asset name (e.g., gold, silver, oil, energy_price)')
    parser.add_argument('--start-date',
                       help='Start date in YYYY-MM-DD format (required with --asset)')
    parser.add_argument('--end-date',
                       help='End date in YYYY-MM-DD format (required with --asset)')
    parser.add_argument('--list-assets', action='store_true',
                       help='List all available assets and exit')
    
    # Legacy options
    parser.add_argument('input_file', nargs='?',
                       default="data/raw/smard_energy_data_2020_2025_combined.csv",
                       help='Legacy: Eingabe-CSV-Datei für direkte Konvertierung')
    parser.add_argument('-o', '--output',
                       help='Ausgabe-CSV-Datei (optional)')
    parser.add_argument('--volume-source', choices=['load', 'residual_load'],
                       help='Legacy: Spalte für Volume-Berechnung: load oder residual_load')
    
    args = parser.parse_args()
    
    # Handle list assets request
    if args.list_assets:
        try:
            asset_manager = AssetManager()
            assets = asset_manager.get_available_assets()
            categories = asset_manager.list_categories()
            
            print("Available Assets:")
            print("=" * 50)
            
            for category_name, category_info in categories.items():
                print(f"\n{category_info['description']}:")
                for asset_name in category_info['assets']:
                    if asset_name in assets:
                        asset_info = assets[asset_name]
                        print(f"  {asset_name:15} - {asset_info['name']} ({asset_info['symbol']})")
                        print(f"  {'':15}   Provider: {asset_info['provider']}")
            
            # Assets not in categories
            categorized_assets = set()
            for cat_info in categories.values():
                categorized_assets.update(cat_info['assets'])
            
            other_assets = set(assets.keys()) - categorized_assets
            if other_assets:
                print(f"\nOther Assets:")
                for asset_name in sorted(other_assets):
                    asset_info = assets[asset_name]
                    print(f"  {asset_name:15} - {asset_info['name']} ({asset_info['symbol']})")
                    
            return 0
            
        except Exception as e:
            print(f"ERROR: Failed to list assets: {e}")
            return 1
    
    # New Asset Manager approach
    if args.asset:
        if not args.start_date or not args.end_date:
            print("ERROR: --start-date and --end-date are required when using --asset")
            return 1
        
        try:
            hourly_candles = convert_asset_to_hourly_candles(
                args.asset,
                args.start_date,
                args.end_date,
                args.output
            )
            return 0 if hourly_candles is not None else 1
            
        except Exception as e:
            print(f"ERROR: {e}")
            return 1
    
    # Legacy CSV file approach
    else:
        print("Using legacy CSV file conversion...")
        hourly_candles = convert_15min_to_hourly_candles(
            args.input_file,
            args.output,
            args.volume_source
        )
        return 0 if hourly_candles is not None else 1

if __name__ == "__main__":
    sys.exit(main())