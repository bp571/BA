import requests
import pandas as pd
import os
import sys
from datetime import datetime

def download_single_smard_dataset(filter_id, region="DE-LU", resolution="quarterhour", dataset_name="", test_mode=False):
    """
    Lädt einen einzelnen SMARD-Datensatz für den gewünschten Zeitraum herunter.
    
    Args:
        filter_id (str): SMARD Filter-ID für den gewünschten Datentyp
        region (str): Marktgebiet (Standard: DE-LU)
        resolution (str): Zeitauflösung (Standard: quarterhour)
        dataset_name (str): Name des Datensatzes für Ausgaben
        test_mode (bool): Wenn True, wird nur eine Woche im Januar 2024 geladen
    
    Returns:
        pandas.DataFrame: DataFrame mit Spalten ['timestamp', 'value'] oder None bei Fehler
    """
    BASE_URL = "https://www.smard.de/app/chart_data"
    
    print(f"Lade {dataset_name}...")
    
    # Schritt 1: Timestamp-Index von SMARD-API abrufen
    index_url = f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    response = requests.get(index_url)
    if response.status_code != 200:
        print(f"ERROR: Index für {dataset_name} konnte nicht abgerufen werden (HTTP {response.status_code})")
        return None

    timestamps = response.json()["timestamps"]
    
    # Schritt 2: Timestamps für gewünschten Zeitraum filtern
    if test_mode:
        # Test-Modus: nur erste Woche Januar 2024
        search_start = int(datetime(2023, 12, 30).timestamp() * 1000)
        end_period = int(datetime(2024, 1, 7, 23, 59).timestamp() * 1000)
    else:
        # Vollständiger Zeitraum 2020-2025
        search_start = int(datetime(2019, 12, 15).timestamp() * 1000)
        end_period = int(datetime(2025, 12, 31, 23, 59).timestamp() * 1000)
    
    relevant_timestamps = [ts for ts in timestamps if search_start <= ts <= end_period]
    
    # Schritt 3: Daten für jeden Timestamp-Block herunterladen
    all_data = []
    total = len(relevant_timestamps)
    
    for i, ts in enumerate(relevant_timestamps, 1):
        current_date = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d')
        print(f"\r  Progress: {i}/{total}", end="", flush=True)
        
        data_url = f"{BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{ts}.json"
        data_res = requests.get(data_url)
        if data_res.status_code == 200:
            all_data.extend(data_res.json()["series"])
    
    print()  # Neue Zeile nach Fortschrittsanzeige
    
    # Schritt 4: Daten verarbeiten
    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "value"])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        
        # Daten exakt auf Zielzeitraum zuschneiden
        if test_mode:
            df = df[(df['datetime'] >= '2024-01-01 00:00:00') &
                    (df['datetime'] <= '2024-01-07 23:45:00')]
        else:
            df = df[(df['datetime'] >= '2020-01-01 00:00:00') &
                    (df['datetime'] <= '2025-12-31 23:45:00')]
        
        # Daten bereinigen: nach Datum sortieren und Duplikate entfernen
        df = df.sort_values('datetime').drop_duplicates('datetime')
        
        print(f" ✓")
        return df[['timestamp', 'datetime', 'value']]
    else:
        print(f" ✗ Keine Daten gefunden")
        return None

def download_and_merge_energy_data(test_mode=False):
    """
    Lädt drei Energiedatensätze von SMARD.de herunter und führt sie zusammen:
    - Preise (ID: 4169)
    - Netzlast (ID: 410)
    - Residuallast (ID: 4359)
    
    Speichert das Ergebnis als CSV mit Spalten: datetime, price, load, residual_load
    
    Args:
        test_mode (bool): Wenn True, wird nur eine Woche im Januar 2024 geladen
    """
    RAW_DIR = "data/raw"
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Definition der gewünschten Datensätze
    datasets = {
        "4169": {"name": "Day-Ahead Preise", "column": "price"},
        "410": {"name": "Netzlast", "column": "load"},
        "4359": {"name": "Residuallast", "column": "residual_load"}
    }
    
    
    # Alle Datensätze herunterladen
    dataframes = {}
    for filter_id, info in datasets.items():
        df = download_single_smard_dataset(filter_id, dataset_name=info["name"], test_mode=test_mode)
        if df is not None:
            dataframes[filter_id] = df
    
    # Datensätze zusammenführen
    if len(dataframes) == 0:
        print("✗ ERROR: Keine Datensätze erfolgreich heruntergeladen.")
        return
    
    print("Zusammenführung...")
    
    # Starte mit dem ersten verfügbaren Datensatz
    first_key = list(dataframes.keys())[0]
    merged_df = dataframes[first_key][['timestamp', 'datetime']].copy()
    merged_df[datasets[first_key]["column"]] = dataframes[first_key]["value"]
    
    # Füge weitere Datensätze hinzu
    for filter_id in list(dataframes.keys())[1:]:
        df = dataframes[filter_id]
        column_name = datasets[filter_id]["column"]
        
        # Inner Join über timestamp für vollständige Datenabdeckung
        temp_df = df[['timestamp', 'value']].rename(columns={'value': column_name})
        merged_df = merged_df.merge(temp_df, on='timestamp', how='inner')
    
    if len(merged_df) == 0:
        print("✗ ERROR: Keine übereinstimmenden Zeitstempel gefunden.")
        return
    
    # Finale Spaltenauswahl und Sortierung
    final_columns = ['datetime'] + [datasets[fid]["column"] for fid in dataframes.keys()]
    merged_df = merged_df[final_columns].sort_values('datetime')
    
    # Speichern
    suffix = "test" if test_mode else "2020_2025"
    output_file = os.path.join(RAW_DIR, f"smard_energy_data_{suffix}_combined.csv")
    merged_df.to_csv(output_file, index=False)
    
    print(f"✓ Erfolgreich gespeichert: {output_file}")
    print(f"  {len(merged_df)} Datensätze ({merged_df['datetime'].min()} - {merged_df['datetime'].max()})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SMARD Energiedaten Download')
    parser.add_argument('--full', action='store_true',
                       help='Vollständiger Download 2020-2025 (Standard: Test-Modus)')
    args = parser.parse_args()
    
    # Standardmäßig Test-Modus, außer --full wird angegeben
    test_mode = not args.full
    download_and_merge_energy_data(test_mode=test_mode)