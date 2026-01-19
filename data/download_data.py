import yfinance as yf
import os

# Pfad des aktuellen Skripts ermitteln
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "EURUSD.csv")

# Daten laden
df = yf.download("EURUSD=X", start="2020-01-01", end="2025-12-31", interval="1d")

# Im Skript-Pfad speichern
df.to_csv(file_path)

