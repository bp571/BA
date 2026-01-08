# Überarbeitete Implementierung: CSV-Rohdaten für Kronos Rolling Forecast

## Ziel
Erstelle eine CSV-Datei mit nur den essentiellen Rohdaten (actual/predicted) als Zwischenergebnis für flexible Metrik-Berechnungen.

## CSV-Schema (nur Rohdaten)

```csv
date,hour,actual_value,predicted_value
2025-01-01,0,2.16,-15.19
2025-01-01,1,1.60,20.80
2025-01-01,2,0.00,5.63
...
```

## Implementierungsplan

### 1. Neue Funktion: `save_predictions_to_csv()`
```python
def save_predictions_to_csv(results, output_path):
    """
    Speichert Rolling Forecast Rohdaten als CSV für spätere Metrik-Berechnungen
    
    Args:
        results: List von Dictionaries mit Rolling Forecast Ergebnissen
        output_path: Pfad zur CSV-Ausgabedatei
    """
    import pandas as pd
    
    rows = []
    for day_result in results:
        date = day_result["date"]
        actual_values = day_result["actual"]
        predicted_values = day_result["predicted"]
        
        for hour in range(24):
            if hour < len(actual_values) and hour < len(predicted_values):
                rows.append({
                    'date': date,
                    'hour': hour,
                    'actual_value': actual_values[hour],
                    'predicted_value': predicted_values[hour]
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.2f')
```

### 2. Integration in run_rolling_forecast()
- Ersetze JSON-Output durch CSV-Output
- Neue Dateinamen-Konvention: `kronos_predictions_YYYY-MM-DD_YYYY-MM-DD.csv`

### 3. Vorteile
- **Performance**: Prediction läuft nur einmal
- **Flexibilität**: Beliebige Metriken später berechenbar  
- **Kompakt**: Nur 4 Spalten statt komplexer JSON-Struktur
- **Standard**: CSV ist universell kompatibel

## Separate Metrik-Berechnung
Später kann eine separate Funktion/Skript die Metriken aus der CSV berechnen:
```python
df = pd.read_csv('kronos_predictions_2025-01-01_2025-05-22.csv')
mae = (df['actual_value'] - df['predicted_value']).abs().mean()
mape = ((df['actual_value'] - df['predicted_value']).abs() / df['actual_value']).mean() * 100
# etc.