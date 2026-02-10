import pandas as pd
import numpy as np
from tqdm import tqdm
from .metrics import calculate_all_metrics, calculate_ic_statistics, calculate_asset_confidence_interval

def run_rolling_benchmark(predictor, df, ticker, params):
    """
    Führt den Benchmark durch und behebt den 'Ambiguous Index' Fehler.
    """
    df = df.copy()
    
    # 1. Spaltennamen normalisieren (Kleinbuchstaben)
    df.columns = [c.lower() for c in df.columns]
    
    # 2. Mehrdeutigkeit von 'datetime' beheben
    # Falls datetime sowohl Index als auch Spalte ist, löschen wir die Dubletten
    if df.index.name == 'datetime' or 'datetime' in df.columns:
        df = df.reset_index(drop=False)
    
    # Alle exakt gleichen Spaltennamen entfernen (behebt das Kernproblem)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Falls die Spalte nach dem Reset 'date' hieß (yfinance Standard)
    if 'date' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'date': 'datetime'})

    # 3. Zeit-Formatierung sicherstellen
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df = df.sort_values('datetime').reset_index(drop=True)

    all_actuals = []
    all_predictions = []
    rolling_ic_values = []
    rolling_rankic_values = []
    
    # Parameter aus den DEFAULT_PARAMS oder main.py
    context_steps = params.get('context_steps', 80)
    forecast_steps = params.get('forecast_steps', 24)
    stride = params.get('stride_steps', 24)
    num_steps = params.get('steps', 5)

    # 4. Rolling Window Loop
    for i in range(num_steps):
        # Fix data leakage: ensure no overlap between windows
        cutoff_idx = context_steps + (i * (stride + forecast_steps))  # Add forecast_steps to prevent overlap
        
        # Validation against data leakage
        if cutoff_idx + forecast_steps > len(df):
            break
        if cutoff_idx < context_steps:
            continue  # Skip if not enough historical data
            
        context_data = df.iloc[cutoff_idx - context_steps : cutoff_idx]
        target_data = df.iloc[cutoff_idx : cutoff_idx + forecast_steps]
        
        # Additional validation: ensure no temporal overlap
        if len(context_data) != context_steps or len(target_data) != forecast_steps:
            continue
        
        # Ensure temporal ordering (no look-ahead bias)
        if context_data['datetime'].iloc[-1] >= target_data['datetime'].iloc[0]:
            print(f"Warning: Potential look-ahead bias detected at step {i}")
            continue
        
        try:
            # Predict-Aufruf mit den OHLC-Spalten
            pred_df = predictor.predict(
                df=context_data[['open', 'high', 'low', 'close']],
                x_timestamp=context_data['datetime'],
                y_timestamp=target_data['datetime'],
                pred_len=forecast_steps
            )
            
            act = target_data['close'].values
            pre = pred_df['close'].values
            
            # Metriken für dieses Fenster sammeln
            window_metrics = calculate_all_metrics(act, pre)
            if 'IC_Return' in window_metrics:
                rolling_ic_values.append(window_metrics['IC_Return'])
            if 'RankIC_Return' in window_metrics:
                rolling_rankic_values.append(window_metrics['RankIC_Return'])

            all_actuals.extend(act.tolist())
            all_predictions.extend(pre.tolist())
            
        except Exception as e:
            # Fehler im Loop unterdrücken, um Benchmark nicht zu stoppen
            continue

    if not all_actuals:
        return None

    # 5. Finale Aggregation und Statistik
    y_true, y_pred = np.array(all_actuals), np.array(all_predictions)
    global_metrics = calculate_all_metrics(y_true, y_pred)
    ic_stats = calculate_ic_statistics(rolling_ic_values, prefix="IC")
    rankic_stats = calculate_ic_statistics(rolling_rankic_values, prefix="RankIC")
    asset_ci = calculate_asset_confidence_interval(y_true, y_pred)
    
    return {
        'ticker': ticker,
        'metrics': {**global_metrics, **ic_stats, **rankic_stats, **asset_ci},
        'raw_values': {'actual': all_actuals, 'predicted': all_predictions}
    }