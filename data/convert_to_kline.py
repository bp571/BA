import pandas as pd
import os
from datetime import datetime
import sys

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
    """Hauptfunktion für Kommandozeilenausführung"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Konvertiere 15-Min-Daten zu stündlichen Candlesticks')
    parser.add_argument('input_file', nargs='?', 
                       default="data/raw/smard_energy_data_2020_2025_combined.csv",
                       help='Eingabe-CSV-Datei')
    parser.add_argument('-o', '--output', 
                       help='Ausgabe-CSV-Datei (optional)')
    parser.add_argument('--volume-source', choices=['load', 'residual_load'],
                       help='Spalte für Volume-Berechnung: load oder residual_load')
    
    args = parser.parse_args()
    
    # Konvertierung durchführen
    hourly_candles = convert_15min_to_hourly_candles(
        args.input_file, 
        args.output,
        args.volume_source
    )
    
    return 0 if hourly_candles is not None else 1

if __name__ == "__main__":
    sys.exit(main())