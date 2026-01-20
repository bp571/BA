import yfinance as yf
import os

# Pfad des aktuellen Skripts ermitteln
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "gold.csv")

# Daten laden
df = yf.download("GC=F", start="2024-01-01", end="2026-01-19", interval="1h")

# Im Skript-Pfad speichern
df.to_csv(file_path)

