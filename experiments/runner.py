import pandas as pd
import numpy as np
from tqdm import tqdm
from .metrics import calculate_all_metrics, calculate_ic_statistics, calculate_asset_confidence_interval

def run_rolling_benchmark(predictor, df, ticker, params):
    """
    Führt Rolling Window Benchmark durch.
    
    Berechnet TIME-SERIES Metriken pro Window (Korrelation innerhalb eines Assets).
    Für Cross-Sectional RankIC (über Assets hinweg) siehe evaluate_results.py.
    """
    df = df.copy()
    
    # 1. Spaltennamen normalisieren
    df.columns = [c.lower() for c in df.columns]
    
    if df.index.name == 'datetime' or 'datetime' in df.columns:
        df = df.reset_index(drop=False)
    
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'date' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'date': 'datetime'})

    # 2. Zeit-Formatierung
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 3. DATENQUALITÄTS-CHECKS
    # Check für Duplikate
    if df['datetime'].duplicated().any():
        n_dupes = df['datetime'].duplicated().sum()
        print(f"⚠️  {ticker}: {n_dupes} duplicate timestamps found - removing duplicates")
        df = df.drop_duplicates(subset='datetime', keep='first')
    
    # Check für Gaps (fehlende Handelstage)
    time_diffs = df['datetime'].diff()
    large_gaps = time_diffs > pd.Timedelta(days=5)
    if large_gaps.any():
        n_gaps = large_gaps.sum()
        print(f"⚠️  {ticker}: {n_gaps} gaps > 5 days detected in time series")
    
    # Check für Outliers in Returns (nur Warnung, kein Ausschluss)
    if len(df) > 1:
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        outliers = np.abs(returns - returns.mean()) > 3 * returns.std()
        if outliers.any():
            n_outliers = outliers.sum()
            print(f"⚠️  {ticker}: {n_outliers} outlier returns (>3 std) detected")

    all_actuals = []
    all_predictions = []
    all_dates = []
    all_anchors = []  # Store last context price for each forecast window
    rolling_ic_values = []
    rolling_rankic_values = []
    
    context_steps = params.get('context_steps', 80)
    forecast_steps = params.get('forecast_steps', 24)
    stride = params.get('stride_steps', 24)
    num_steps = params.get('steps', 5)

    # 4. Rolling Window Loop
    for i in range(num_steps):
        cutoff_idx = context_steps + (i * stride)
        
        if cutoff_idx + forecast_steps > len(df):
            break
            
        context_data = df.iloc[cutoff_idx - context_steps : cutoff_idx]
        target_data = df.iloc[cutoff_idx : cutoff_idx + forecast_steps]
        
        if len(context_data) != context_steps or len(target_data) != forecast_steps:
            continue
        
        try:
            pred_df = predictor.predict(
                df=context_data[['open', 'high', 'low', 'close']],
                x_timestamp=context_data['datetime'],
                y_timestamp=target_data['datetime'],
                pred_len=forecast_steps
            )
            
            act = target_data['close'].values
            pre = pred_df['close'].values
            
            # FIX 5: Übergebe vollständigen Context für MASE + Anchor für Returns
            # y_train wird für zwei Zwecke genutzt:
            # 1. MASE Baseline (braucht mehrere Werte für naive Forecast)
            # 2. Anchor für Return-Berechnung (letzter Wert)
            y_train_context = context_data['close'].values
            window_metrics = calculate_all_metrics(act, pre, y_train=y_train_context)
            
            # Aktualisierte Key-Namen (Time-Series Metriken)
            if 'IC_TimeSeries' in window_metrics:
                rolling_ic_values.append(window_metrics['IC_TimeSeries'])
            if 'RankIC_TimeSeries' in window_metrics:
                rolling_rankic_values.append(window_metrics['RankIC_TimeSeries'])

            # Store anchor price (last context price) for each forecast day
            anchor_price = context_data['close'].iloc[-1]
            all_actuals.extend(act.tolist())
            all_predictions.extend(pre.tolist())
            all_dates.extend(target_data['datetime'].dt.strftime('%Y-%m-%d').tolist())
            all_anchors.extend([anchor_price] * len(act))
            
        except Exception as e:
            # Log exceptions for debugging
            print(f"⚠️  {ticker}: Window {i} failed: {str(e)}")
            continue

    if not all_actuals:
        return None

    # 5. Finale Aggregation über Rolling Windows
    # WICHTIG: Wir berechnen KEINE globalen Metriken auf den concatenierten Predictions,
    # da diese aus diskonnektierten Rolling Windows stammen und keine kontinuierliche
    # Zeitreihe bilden. Stattdessen aggregieren wir nur die Window-basierten Metriken.
    
    y_true, y_pred = np.array(all_actuals), np.array(all_predictions)
    
    # TIME-SERIES Metriken: Aggregiere nur über die Rolling Window Statistiken
    ic_stats = calculate_ic_statistics(rolling_ic_values, prefix="IC_TimeSeries")
    rankic_stats = calculate_ic_statistics(rolling_rankic_values, prefix="RankIC_TimeSeries")
    
    # Zusätzlich: Einfache beschreibende Statistiken auf Preisebene
    # Diese sind nur Orientierungshilfen, nicht für wissenschaftliche Interpretation
    mae_val = np.mean(np.abs(y_true - y_pred))
    rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    basic_metrics = {
        'MAE_indicative': mae_val,  # Nur indikativ, da über disconnected Windows
        'RMSE_indicative': rmse_val,
        'N_Windows': len(rolling_ic_values),
        'N_Predictions': len(y_true)
    }
    
    return {
        'ticker': ticker,
        'metrics': {**basic_metrics, **ic_stats, **rankic_stats},
        'raw_values': {
            'actual': all_actuals,
            'predicted': all_predictions,
            'dates': all_dates,
            'anchors': all_anchors  # Last context price for each forecast day
        }
    }