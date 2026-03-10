import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, t as t_dist
import sys
sys.path.append(str(Path(__file__).parent))
from experiments.metrics import calculate_ic_statistics

def generate_latex_table(model_name, ic_stats, df_ic_ts, t_stat, p_value_overall,
                         mean_pearson_ic, std_pearson_ic, avg_assets, min_assets,
                         max_assets, output_path):
    """
    Generiert eine LaTeX-Tabelle mit den Evaluierungsergebnissen.
    
    Args:
        model_name: Name des evaluierten Modells
        ic_stats: Dictionary mit IC-Statistiken
        df_ic_ts: DataFrame mit Zeitreihen der IC-Werte
        t_stat: t-Statistik
        p_value_overall: Gesamter p-Wert
        mean_pearson_ic: Durchschnittlicher Pearson IC
        std_pearson_ic: Standardabweichung Pearson IC
        avg_assets: Durchschnittliche Anzahl Assets pro Tag
        min_assets: Minimale Anzahl Assets
        max_assets: Maximale Anzahl Assets
        output_path: Pfad für die Ausgabedatei
    """
    mean_ic = ic_stats['RankIC_CrossSectional_Mean']
    ci_lower, ci_upper = ic_stats['RankIC_CrossSectional_CI95']
    std_ic = ic_stats['RankIC_CrossSectional_Std']
    n_days = ic_stats['RankIC_CrossSectional_Count']
    n_eff = ic_stats['RankIC_CrossSectional_Effective_N']
    
    positive_days = (df_ic_ts['RankIC'] > 0).sum()
    positive_pct = positive_days / n_days * 100
    
    # LaTeX-Tabelle erstellen
    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Cross-Sectional Rank IC Analyse}
\label{tab:rankic_results}
\begin{tabular}{lr}
\toprule
\textbf{Metrik} & \textbf{Wert} \\
\midrule
Modell & """ + model_name + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Rank IC Statistiken}} \\
Mean Rank IC & """ + f"{mean_ic:.4f}" + r""" \\
95\% Konfidenzintervall & """ + f"[{ci_lower:.4f}, {ci_upper:.4f}]" + r""" \\
Standardabweichung & """ + f"{std_ic:.4f}" + r""" \\
Anzahl Tage & """ + f"{n_days}" + r""" \\
Effektive N (angepasst) & """ + f"{n_eff:.1f}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Statistische Signifikanz}} \\
t-Statistik & """ + f"{t_stat:.2f}" + r""" \\
p-Wert (H$_0$: IC=0) & """ + f"{p_value_overall:.4f}" + r""" \\
Signifikanz ($\alpha=0.05$) & """ + ("Ja" if p_value_overall < 0.05 else "Nein") + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Richtungsanalyse}} \\
Positive IC Tage & """ + f"{positive_days}/{n_days} ({positive_pct:.1f}\\%)" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Pearson IC}} \\
Mean IC (Pearson) & """ + f"{mean_pearson_ic:.4f}" + r""" \\
Std IC (Pearson) & """ + f"{std_pearson_ic:.4f}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Cross-Section Größe}} \\
Durchschnitt Assets/Tag & """ + f"{avg_assets:.1f}" + r""" \\
Min/Max Assets & """ + f"{min_assets}/{max_assets}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    # LaTeX-Datei schreiben
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"\n✓ LaTeX-Tabelle gespeichert: {output_path}")
    print(f"  Fügen Sie diese in Ihre Arbeit ein mit: \\input{{{output_path}}}")

def evaluate_study(results_dir="results"):
    """
    Evaluiert die Ergebnisse einer Studie.
    
    Args:
        results_dir: Verzeichnis mit den Ergebnissen (z.B. "results" oder "results_chronos")
    """
    results_path = Path(results_dir)
    file_path = results_path / "final_energy_study.json"
    
    if not file_path.exists():
        print(f"❌ ERROR: File not found: {file_path}")
        print(f"   Make sure to run the pipeline first!")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model_name = data.get('model', 'Unknown')
    print(f"\n📊 Evaluating results for model: {model_name}")
    print(f"📁 Results directory: {results_dir}")
    
    summary = data.get('summary', {})
    
    # 1. Daten mit Zeitstempel laden
    actual_dfs = []
    pred_dfs = []
    anchor_dfs = []

    for ticker in summary.keys():
        res_file = results_path / f"result_{ticker}.json"
        if res_file.exists():
            with open(res_file, 'r') as f:
                res = json.load(f)
                rv = res['raw_values']
                
                # Erstelle DataFrames mit Datum als Index
                dates = pd.to_datetime(rv['dates'])
                
                actual_dfs.append(pd.Series(rv['actual'], index=dates, name=ticker))
                pred_dfs.append(pd.Series(rv['predicted'], index=dates, name=ticker))
                
                # Load anchors if available (last context price for each forecast)
                if 'anchors' in rv:
                    anchor_dfs.append(pd.Series(rv['anchors'], index=dates, name=ticker))

    # Alle Assets in zwei große DataFrames mergen (automatisches Alignment über Datum)
    df_act = pd.concat(actual_dfs, axis=1).sort_index()
    df_pre = pd.concat(pred_dfs, axis=1).sort_index()
    
    # 2. Transformation in Log-Returns RELATIV zum Anchor
    # Bei Multi-Step Forecasts basieren alle Predictions auf dem gleichen Context-Ende
    if anchor_dfs:
        df_anc = pd.concat(anchor_dfs, axis=1).sort_index()
        # Returns relativ zum letzten Context-Preis (Anchor)
        df_act_ret = np.log(df_act / df_anc)
        df_pre_ret = np.log(df_pre / df_anc)
        print(f"\n✓ Using anchor-based returns (relative to last context price)")
    else:
        # Fallback: Tag-zu-Tag Returns (alte Methode, aber inkonsistent bei Multi-Step)
        print(f"\n⚠️  WARNING: No anchors found, using day-to-day returns (may be incorrect for multi-step forecasts)")
        df_act_ret = np.log(df_act / df_act.shift(1))
        df_pre_ret = np.log(df_pre / df_pre.shift(1))
    
    df_act_ret = df_act_ret.dropna(how='all')
    df_pre_ret = df_pre_ret.dropna(how='all')

    # Nur Zeilen nutzen, die in beiden DFs vorkommen
    common_idx = df_act_ret.index.intersection(df_pre_ret.index)
    df_act_ret = df_act_ret.loc[common_idx]
    df_pre_ret = df_pre_ret.loc[common_idx]

    # 3. Cross-Sectional RankIC Berechnung
    # WICHTIG: Dies ist der echte Cross-Sectional RankIC aus der Finanzforschung
    # (Korrelation ÜBER Assets hinweg zu einem Zeitpunkt)
    results_per_day = []
    assets_per_day = []
    
    for t in df_act_ret.index:
        a_t = df_act_ret.loc[t]
        p_t = df_pre_ret.loc[t]
        
        mask = a_t.notna() & p_t.notna()
        n_assets = mask.sum()
        
        if n_assets >= 10:  # Mindestens 10 Assets für stabilen IC
            ric, p_val_rank = spearmanr(p_t[mask], a_t[mask])
            ic, p_val_pearson = pearsonr(p_t[mask], a_t[mask])
            results_per_day.append({
                'date': t,
                'RankIC': ric,
                'IC': ic,
                'p_value': p_val_rank,
                'n_assets': n_assets
            })
            assets_per_day.append(n_assets)

    if len(results_per_day) == 0:
        print("\n❌ ERROR: No valid Cross-Sectional RankIC values calculated")
        print(f"   Total dates in data: {len(df_act_ret)}")
        print(f"   Reason: Not enough assets (need >=10) with valid data on any date")
        print(f"\n📊 Data availability per date:")
        for t in df_act_ret.index[:10]:  # Show first 10 dates
            a_t = df_act_ret.loc[t]
            p_t = df_pre_ret.loc[t]
            mask = a_t.notna() & p_t.notna()
            n_assets = mask.sum()
            print(f"   {t.date()}: {n_assets} assets")
        return
    
    df_ic_ts = pd.DataFrame(results_per_day).set_index('date')

    # FIX 4: Konfidenzintervalle und statistische Tests
    print("\n" + "="*60)
    print("CROSS-SECTIONAL RANKIC ANALYSIS")
    print("="*60)
    
    # Aggregierte Statistiken mit Konfidenzintervallen
    ic_stats = calculate_ic_statistics(df_ic_ts['RankIC'].values.tolist(), prefix="RankIC_CrossSectional")
    
    mean_ic = ic_stats['RankIC_CrossSectional_Mean']
    ci_lower, ci_upper = ic_stats['RankIC_CrossSectional_CI95']
    n_days = ic_stats['RankIC_CrossSectional_Count']
    n_eff = ic_stats['RankIC_CrossSectional_Effective_N']
    
    # T-Test: H0: RankIC = 0
    std_ic = ic_stats['RankIC_CrossSectional_Std']
    if n_eff > 1:
        t_stat = mean_ic / (std_ic / np.sqrt(n_eff))
        p_value_overall = 2 * (1 - t_dist.cdf(abs(t_stat), df=max(1, n_eff-1)))
    else:
        t_stat = np.nan
        p_value_overall = np.nan
    
    # Ausgabe
    print(f"\n📊 Summary Statistics:")
    print(f"   Mean RankIC:          {mean_ic:.4f}")
    print(f"   95% CI:               [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   Standard Deviation:   {std_ic:.4f}")
    print(f"   Number of Days:       {n_days}")
    print(f"   Effective N (adj):    {n_eff:.1f}")
    print(f"   t-statistic:          {t_stat:.2f}")
    print(f"   p-value (H0: IC=0):   {p_value_overall:.4f}")
    
    significance = "✅ SIGNIFICANT" if p_value_overall < 0.05 else "⚠️  NOT SIGNIFICANT"
    print(f"   Result:               {significance}")
    
    print(f"\n📈 Directional Analysis:")
    positive_days = (df_ic_ts['RankIC'] > 0).sum()
    print(f"   Positive IC Days:     {positive_days}/{n_days} ({positive_days/n_days*100:.1f}%)")
    
    # IC (Pearson) Statistics
    ic_values = df_ic_ts['IC'].values
    mean_pearson_ic = np.mean(ic_values)
    std_pearson_ic = np.std(ic_values, ddof=1)
    print(f"\n📊 IC (Pearson):")
    print(f"   Mean IC:              {mean_pearson_ic:.4f}")
    print(f"   Std IC:               {std_pearson_ic:.4f}")
    
    avg_assets = np.mean(assets_per_day)
    min_assets = np.min(assets_per_day)
    max_assets = np.max(assets_per_day)
    print(f"\n🎯 Cross-Section Size:")
    print(f"   Average Assets/Day:   {avg_assets:.1f}")
    print(f"   Min/Max Assets:       {min_assets}/{max_assets}")
    
    if avg_assets < 10:
        print(f"   ⚠️  WARNING: Small cross-section (<10 assets) leads to high RankIC variance")

    # 5. Visualisierung mit Konfidenzintervallen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Rolling RankIC mit CI
    rolling_ic = df_ic_ts['RankIC'].rolling(window=20, min_periods=5).mean()
    ax1.plot(df_ic_ts.index, rolling_ic, label='20-Day Rolling Mean', linewidth=2)
    ax1.axhline(mean_ic, color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {mean_ic:.3f}')
    ax1.axhline(ci_lower, color='red', linestyle=':', alpha=0.5, label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    ax1.axhline(ci_upper, color='red', linestyle=':', alpha=0.5)
    ax1.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax1.fill_between(df_ic_ts.index, ci_lower, ci_upper, alpha=0.1, color='red')
    ax1.set_title("Cross-Sectional Rank IC Over Time (20-Day Rolling)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Rank IC")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tägliche RankIC Werte (Scatter)
    colors = ['green' if x > 0 else 'red' for x in df_ic_ts['RankIC']]
    ax2.scatter(df_ic_ts.index, df_ic_ts['RankIC'], c=colors, alpha=0.5, s=20)
    ax2.axhline(mean_ic, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_ic:.3f}')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.fill_between(df_ic_ts.index, ci_lower, ci_upper, alpha=0.15, color='blue')
    ax2.set_title("Daily Cross-Sectional Rank IC", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rank IC")
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 6. LaTeX-Tabelle generieren
    latex_output_path = results_path / "results_table.tex"
    generate_latex_table(
        model_name=model_name,
        ic_stats=ic_stats,
        df_ic_ts=df_ic_ts,
        t_stat=t_stat,
        p_value_overall=p_value_overall,
        mean_pearson_ic=mean_pearson_ic,
        std_pearson_ic=std_pearson_ic,
        avg_assets=avg_assets,
        min_assets=min_assets,
        max_assets=max_assets,
        output_path=latex_output_path
    )
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate forecasting results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing the results (default: results)')
    
    args = parser.parse_args()
    evaluate_study(results_dir=args.results_dir)