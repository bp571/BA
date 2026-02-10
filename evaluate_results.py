import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def evaluate_study(file_path="results/final_energy_study.json", show_both_metrics=True):
    path = Path(file_path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = data['summary']
    params = data.get('params', {})
    n_steps = params.get('steps', 5)  # Die gewünschte 5
    f_steps = params.get('forecast_steps', 12)
    results_dir = path.parent
    
    # 1. Daten laden und in Fenster (Steps) strukturieren
    actuals_list, preds_list = [], []
    for ticker in summary.keys():
        res_file = results_dir / f"result_{ticker}.json"
        if res_file.exists():
            with open(res_file, 'r') as f:
                res = json.load(f)
                actuals_list.append(pd.Series(res['raw_values']['actual'], name=ticker))
                preds_list.append(pd.Series(res['raw_values']['predicted'], name=ticker))

    # Fix: Use log_returns for consistency with metrics.py (additive, better statistical properties)
    def calculate_log_returns_df(df):
        return df.apply(lambda col: col.pct_change().apply(lambda x: np.log(1 + x) if x > -0.999 else np.nan)).dropna()
    
    df_act = calculate_log_returns_df(pd.concat(actuals_list, axis=1))
    df_pre = calculate_log_returns_df(pd.concat(preds_list, axis=1))

    # 2. Cross-Sectional IC und RankIC Berechnung (Portfolio Management Goldstandard)
    from scipy.stats import spearmanr, pearsonr
    
    # Fix: Ensure temporal synchronization - align dataframes by index
    min_len = min(len(df_act), len(df_pre))
    df_act = df_act.iloc[:min_len]
    df_pre = df_pre.iloc[:min_len]
    
    # Ensure exact temporal alignment
    df_act.index = range(len(df_act))
    df_pre.index = range(len(df_pre))
    
    window_ics, window_rank_ics = [], []
    for i in range(n_steps):
        start = i * f_steps
        end = (i + 1) * f_steps
        
        step_ics, step_rank_ics = [], []
        for t in range(start, min(end, len(df_act))):
            actual_t = df_act.iloc[t].dropna()
            pred_t = df_pre.iloc[t].dropna()
            
            # Critical: Only use assets present in BOTH dataframes at time t
            common_assets = actual_t.index.intersection(pred_t.index)
            if len(common_assets) > 2:  # Need minimum 3 assets for meaningful correlation
                actual_vals = actual_t[common_assets]
                pred_vals = pred_t[common_assets]
                
                # Additional validation: remove infinite/extreme values
                mask = np.isfinite(actual_vals) & np.isfinite(pred_vals)
                if mask.sum() > 2:
                    actual_clean = actual_vals[mask]
                    pred_clean = pred_vals[mask]
                    
                    # IC (Pearson) - measures linear relationship
                    ic_t, _ = pearsonr(pred_clean, actual_clean)  # Note: pred first, actual second for consistency
                    if not np.isnan(ic_t):
                        step_ics.append(ic_t)
                    
                    # RankIC (Spearman) - measures monotonic relationship, robust to outliers
                    rank_ic_t, _ = spearmanr(pred_clean, actual_clean)
                    if not np.isnan(rank_ic_t):
                        step_rank_ics.append(rank_ic_t)
        
        if step_ics:
            window_ics.append(np.mean(step_ics))
        if step_rank_ics:
            window_rank_ics.append(np.mean(step_rank_ics))

    global_ic = np.mean(window_ics) if window_ics else 0.0
    global_rank_ic = np.mean(window_rank_ics) if window_rank_ics else 0.0

    # 3. Asset-Daten für Plot (IC)
    rows = []
    for ticker, m in summary.items():
        ci = m.get('IC_CI95', [0, 0])
        rows.append({
            'Ticker': ticker,
            'Mean': m.get('IC_Mean', 0),
            'Low': ci[0], 'High': ci[1],
            'Lag': m.get('Is_Lagging', False)
        })
    df_plot = pd.DataFrame(rows).sort_values('Mean')

    # Calculate RankIC confidence intervals from cross-sectional window values
    if window_rank_ics:
        rankic_std = np.std(window_rank_ics, ddof=1)
        rankic_se = rankic_std / np.sqrt(len(window_rank_ics))
        rankic_ci_lower = global_rank_ic - 1.96 * rankic_se
        rankic_ci_upper = global_rank_ic + 1.96 * rankic_se
    else:
        rankic_ci_lower = rankic_ci_upper = 0
    
    # 4. Output & Visualisierung
    print(f"\n--- Bachelorarbeit: Evaluation (n={n_steps} Fenster) ---")
    print(f"Globaler Cross-Sectional IC: {global_ic:.4f}")
    print(f"Globaler Cross-Sectional RankIC: {global_rank_ic:.4f} (CI95: [{rankic_ci_lower:.4f}, {rankic_ci_upper:.4f}])")

    plt.figure(figsize=(10, 7))
    y_err = [df_plot['Mean'] - df_plot['Low'], df_plot['High'] - df_plot['Mean']]
    plt.barh(df_plot['Ticker'], df_plot['Mean'], xerr=y_err, 
             color=['#e74c3c' if x else '#3498db' for x in df_plot['Lag']], capsize=4)
    
    plt.axvline(0, color='black', lw=1)
    plt.title(f"Performance (Steps: {n_steps} | Forecast: {f_steps})\nGlobal IC: {global_ic:.4f} | Global RankIC: {global_rank_ic:.4f}")
    plt.xlabel("IC mit 95% Konfidenzintervall")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_study("results/final_energy_study.json")