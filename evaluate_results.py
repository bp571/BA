import json
import pandas as pd
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

    df_act = pd.concat(actuals_list, axis=1).pct_change().dropna()
    df_pre = pd.concat(preds_list, axis=1).pct_change().dropna()

    # 2. Cross-Sectional IC und RankIC Berechnung
    from scipy.stats import spearmanr, pearsonr
    import numpy as np
    
    window_ics, window_rank_ics = [], []
    for i in range(n_steps):
        start = i * f_steps
        end = (i + 1) * f_steps
        
        step_ics, step_rank_ics = [], []
        for t in range(start, min(end, len(df_act))):
            actual_t = df_act.iloc[t].dropna()
            pred_t = df_pre.iloc[t].dropna()
            
            common_assets = actual_t.index.intersection(pred_t.index)
            if len(common_assets) > 2:
                actual_vals = actual_t[common_assets]
                pred_vals = pred_t[common_assets]
                
                # IC (Pearson)
                ic_t, _ = pearsonr(pred_vals, actual_vals)
                if not np.isnan(ic_t):
                    step_ics.append(ic_t)
                
                # RankIC (Spearman)
                actual_ranks = actual_vals.rank()
                pred_ranks = pred_vals.rank()
                rank_ic_t, _ = spearmanr(pred_ranks, actual_ranks)
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

    # 4. Output & Visualisierung
    print(f"\n--- Bachelorarbeit: Evaluation (n={n_steps} Fenster) ---")
    print(f"Globaler Cross-Sectional IC: {global_ic:.4f}")
    print(f"Globaler Cross-Sectional RankIC: {global_rank_ic:.4f}")

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