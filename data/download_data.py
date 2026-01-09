import yfinance as yf
import os

# Pfad des aktuellen Skripts ermitteln
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "apple_2025.csv")

# Daten laden
df = yf.download("AAPL", start="2025-01-01", end="2025-12-31", interval="1h")

# Im Skript-Pfad speichern
df.to_csv(file_path)

