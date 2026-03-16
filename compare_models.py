"""
Wissenschaftlich korrekter Modellvergleich: Zero-Shot vs. Fine-Tuned

Implementiert statistische Tests gemäß Best Practices für Zeitreihen-Forecasting:
- Paired t-test für IC-Metriken über Assets
- Konfidenzintervalle mit Autokorrelationskorrektur
- Diebold-Mariano Test für Forecast-Genauigkeit
- Cross-Sectional RankIC Vergleich
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel, ttest_ind, spearmanr, t as t_dist
from typing import Dict, Tuple
import sys

sys.path.append(str(Path(__file__).parent))
from experiments.metrics import calculate_ic_statistics


def load_results(results_dir: str) -> Dict:
    """Lade Ergebnisse aus einem Results-Verzeichnis."""
    results_path = Path(results_dir)
    file_path = results_path / "final_energy_study.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Ergebnisse nicht gefunden: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Lade Detail-Ergebnisse pro Asset
    summary = data.get('summary', {})
    asset_results = {}
    
    for ticker in summary.keys():
        res_file = results_path / f"result_{ticker}.json"
        if res_file.exists():
            with open(res_file, 'r') as f:
                asset_results[ticker] = json.load(f)
    
    data['asset_results'] = asset_results
    return data


def paired_t_test(metric_1: np.ndarray, metric_2: np.ndarray, metric_name: str = "") -> Dict:
    """
    Paired t-test für zwei Modelle über Assets hinweg.
    
    H0: Kein Unterschied zwischen den Modellen
    H1: Modell 2 ist besser als Modell 1
    """
    if len(metric_1) != len(metric_2):
        raise ValueError("Metriken müssen gleiche Länge haben")
    
    if len(metric_1) < 2:
        return {
            'mean_diff': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'significant': False
        }
    
    # Paired t-test
    t_stat, p_value = ttest_rel(metric_2, metric_1, alternative='greater')
    
    diff = metric_2 - metric_1
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    # Konfidenzintervall für den Unterschied
    n = len(diff)
    se = std_diff / np.sqrt(n)
    t_crit = t_dist.ppf(0.975, df=n-1)  # 95% CI, zweiseitig
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_95': (ci_lower, ci_upper),
        'significant': p_value < 0.05,
        'n_assets': n
    }


def diebold_mariano_test(errors_1: np.ndarray, errors_2: np.ndarray) -> Dict:
    """
    Diebold-Mariano Test für Forecast-Genauigkeit.
    
    Testet, ob die Forecasts von Modell 2 signifikant besser sind als Modell 1.
    Berücksichtigt Autokorrelation in den Forecast-Fehlern.
    """
    if len(errors_1) != len(errors_2):
        raise ValueError("Fehler-Arrays müssen gleiche Länge haben")
    
    # Quadratische Fehler
    loss_1 = errors_1 ** 2
    loss_2 = errors_2 ** 2
    
    # Loss-Differenzen
    d = loss_1 - loss_2  # Positiv: Modell 2 ist besser
    
    mean_d = np.mean(d)
    
    # Varianz unter Berücksichtigung von Autokorrelation
    n = len(d)
    if n < 3:
        return {'dm_statistic': np.nan, 'p_value': np.nan, 'significant': False}
    
    # Berechne Autokovarianz-robuste Varianz (Newey-West)
    gamma_0 = np.var(d, ddof=1)
    
    # Lag-1 Autokovarianz
    if n > 1:
        gamma_1 = np.mean((d[:-1] - mean_d) * (d[1:] - mean_d))
    else:
        gamma_1 = 0
    
    # Newey-West Varianz-Schätzer (mit Lag 1)
    var_d = (gamma_0 + 2 * gamma_1) / n
    
    if var_d <= 0:
        var_d = gamma_0 / n  # Fallback
    
    # DM-Statistik
    dm_stat = mean_d / np.sqrt(var_d)
    
    # p-value (einseitig, H1: Modell 2 ist besser)
    p_value = 1 - t_dist.cdf(dm_stat, df=n-1)
    
    return {
        'dm_statistic': dm_stat,
        'mean_loss_diff': mean_d,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def compare_cross_sectional_ic(results_1: Dict, results_2: Dict, results_dir_1: str, results_dir_2: str) -> Dict:
    """
    Vergleicht Cross-Sectional RankIC zwischen zwei Modellen.
    Berechnet die täglichen IC-Werte und führt einen paired t-test durch.
    """
    # Lade raw values für beide Modelle
    asset_results_1 = results_1['asset_results']
    asset_results_2 = results_2['asset_results']
    
    # Gemeinsame Assets
    common_tickers = set(asset_results_1.keys()) & set(asset_results_2.keys())
    
    if not common_tickers:
        return {'error': 'Keine gemeinsamen Assets gefunden'}
    
    # Erstelle DataFrames mit Predictions und Actuals
    actual_dfs_1 = []
    pred_dfs_1 = []
    anchor_dfs_1 = []
    
    actual_dfs_2 = []
    pred_dfs_2 = []
    anchor_dfs_2 = []
    
    for ticker in common_tickers:
        rv1 = asset_results_1[ticker]['raw_values']
        rv2 = asset_results_2[ticker]['raw_values']
        
        dates1 = pd.to_datetime(rv1['dates'])
        dates2 = pd.to_datetime(rv2['dates'])
        
        actual_dfs_1.append(pd.Series(rv1['actual'], index=dates1, name=ticker))
        pred_dfs_1.append(pd.Series(rv1['predicted'], index=dates1, name=ticker))
        if 'anchors' in rv1:
            anchor_dfs_1.append(pd.Series(rv1['anchors'], index=dates1, name=ticker))
        
        actual_dfs_2.append(pd.Series(rv2['actual'], index=dates2, name=ticker))
        pred_dfs_2.append(pd.Series(rv2['predicted'], index=dates2, name=ticker))
        if 'anchors' in rv2:
            anchor_dfs_2.append(pd.Series(rv2['anchors'], index=dates2, name=ticker))
    
    # Merge zu DataFrames
    df_act_1 = pd.concat(actual_dfs_1, axis=1).sort_index()
    df_pre_1 = pd.concat(pred_dfs_1, axis=1).sort_index()
    
    df_act_2 = pd.concat(actual_dfs_2, axis=1).sort_index()
    df_pre_2 = pd.concat(pred_dfs_2, axis=1).sort_index()
    
    # Returns berechnen (relativ zu Anchors wenn verfügbar)
    if anchor_dfs_1 and anchor_dfs_2:
        df_anc_1 = pd.concat(anchor_dfs_1, axis=1).sort_index()
        df_anc_2 = pd.concat(anchor_dfs_2, axis=1).sort_index()
        
        df_act_ret_1 = np.log(df_act_1 / df_anc_1)
        df_pre_ret_1 = np.log(df_pre_1 / df_anc_1)
        
        df_act_ret_2 = np.log(df_act_2 / df_anc_2)
        df_pre_ret_2 = np.log(df_pre_2 / df_anc_2)
    else:
        df_act_ret_1 = np.log(df_act_1 / df_act_1.shift(1))
        df_pre_ret_1 = np.log(df_pre_1 / df_pre_1.shift(1))
        
        df_act_ret_2 = np.log(df_act_2 / df_act_2.shift(1))
        df_pre_ret_2 = np.log(df_pre_2 / df_pre_2.shift(1))
    
    df_act_ret_1 = df_act_ret_1.dropna(how='all')
    df_pre_ret_1 = df_pre_ret_1.dropna(how='all')
    df_act_ret_2 = df_act_ret_2.dropna(how='all')
    df_pre_ret_2 = df_pre_ret_2.dropna(how='all')
    
    # Finde gemeinsame Zeitpunkte
    common_dates = df_act_ret_1.index.intersection(df_act_ret_2.index)
    
    if len(common_dates) == 0:
        return {'error': 'Keine gemeinsamen Zeitpunkte gefunden'}
    
    # Berechne tägliche RankIC für beide Modelle
    rankic_1_list = []
    rankic_2_list = []
    valid_dates = []
    
    for t in common_dates:
        a_t = df_act_ret_1.loc[t]
        p1_t = df_pre_ret_1.loc[t]
        p2_t = df_pre_ret_2.loc[t]
        
        # Mask für gültige Werte
        mask = a_t.notna() & p1_t.notna() & p2_t.notna()
        n_valid = mask.sum()
        
        if n_valid >= 10:  # Mindestens 10 Assets
            ric_1, _ = spearmanr(p1_t[mask], a_t[mask])
            ric_2, _ = spearmanr(p2_t[mask], a_t[mask])
            
            rankic_1_list.append(ric_1)
            rankic_2_list.append(ric_2)
            valid_dates.append(t)
    
    if len(rankic_1_list) < 2:
        return {'error': 'Nicht genug gemeinsame Datenpunkte für Vergleich'}
    
    rankic_1 = np.array(rankic_1_list)
    rankic_2 = np.array(rankic_2_list)
    
    # Paired t-test
    test_result = paired_t_test(rankic_1, rankic_2, "RankIC")
    
    # Statistiken für beide Modelle
    stats_1 = calculate_ic_statistics(rankic_1_list, prefix="Model1_RankIC")
    stats_2 = calculate_ic_statistics(rankic_2_list, prefix="Model2_RankIC")
    
    return {
        'model_1_stats': stats_1,
        'model_2_stats': stats_2,
        'paired_test': test_result,
        'n_dates': len(valid_dates),
        'rankic_timeseries_1': rankic_1_list,
        'rankic_timeseries_2': rankic_2_list,
        'dates': valid_dates
    }


def compare_models(results_dir_1: str, results_dir_2: str, model_1_name: str = "Zero-Shot", 
                   model_2_name: str = "Fine-Tuned"):
    """
    Hauptfunktion für wissenschaftlich korrekten Modellvergleich.
    
    Args:
        results_dir_1: Verzeichnis mit Baseline-Ergebnissen (Zero-Shot)
        results_dir_2: Verzeichnis mit Fine-Tuned-Ergebnissen
        model_1_name: Name des ersten Modells
        model_2_name: Name des zweiten Modells
    """
    print("\n" + "="*80)
    print(f"WISSENSCHAFTLICHER MODELLVERGLEICH: {model_1_name} vs. {model_2_name}")
    print("="*80)
    
    # 1. Lade Ergebnisse
    print(f"\n📂 Lade Ergebnisse...")
    try:
        results_1 = load_results(results_dir_1)
        results_2 = load_results(results_dir_2)
    except FileNotFoundError as e:
        print(f"\n❌ FEHLER: {e}")
        print("\n💡 Tipp: Führen Sie zuerst die Evaluationen aus:")
        print(f"   - python main_chronos.py  (für {model_1_name})")
        print(f"   - python main_chronos_finetuned.py  (für {model_2_name})")
        return
    
    print(f"   ✅ {model_1_name}: {results_1['n_assets_processed']} Assets")
    print(f"   ✅ {model_2_name}: {results_2['n_assets_processed']} Assets")
    
    # 2. Gemeinsame Assets identifizieren
    summary_1 = results_1.get('summary', {})
    summary_2 = results_2.get('summary', {})
    
    common_tickers = set(summary_1.keys()) & set(summary_2.keys())
    
    if not common_tickers:
        print("\n❌ FEHLER: Keine gemeinsamen Assets gefunden!")
        return
    
    print(f"\n🎯 Gemeinsame Assets: {len(common_tickers)}")
    
    # 3. Extrahiere Time-Series IC-Metriken pro Asset
    ic_mean_1 = []
    ic_mean_2 = []
    rankic_mean_1 = []
    rankic_mean_2 = []
    mae_1 = []
    mae_2 = []
    
    for ticker in common_tickers:
        metrics_1 = summary_1[ticker]
        metrics_2 = summary_2[ticker]
        
        # IC Time-Series
        if 'IC_TimeSeries_Mean' in metrics_1 and 'IC_TimeSeries_Mean' in metrics_2:
            ic_mean_1.append(metrics_1['IC_TimeSeries_Mean'])
            ic_mean_2.append(metrics_2['IC_TimeSeries_Mean'])
        
        # RankIC Time-Series
        if 'RankIC_TimeSeries_Mean' in metrics_1 and 'RankIC_TimeSeries_Mean' in metrics_2:
            rankic_mean_1.append(metrics_1['RankIC_TimeSeries_Mean'])
            rankic_mean_2.append(metrics_2['RankIC_TimeSeries_Mean'])
        
        # MAE
        if 'MAE_indicative' in metrics_1 and 'MAE_indicative' in metrics_2:
            mae_1.append(metrics_1['MAE_indicative'])
            mae_2.append(metrics_2['MAE_indicative'])
    
    # 4. Statistische Tests: Time-Series Metriken
    print("\n" + "="*80)
    print("TEIL 1: TIME-SERIES METRIKEN (innerhalb Assets)")
    print("="*80)
    
    if rankic_mean_1 and rankic_mean_2:
        print("\n📊 RankIC (Time-Series) - Paired t-test:")
        rankic_test = paired_t_test(np.array(rankic_mean_1), np.array(rankic_mean_2), "RankIC_TS")
        
        print(f"   {model_1_name} Mean:     {np.mean(rankic_mean_1):.4f}")
        print(f"   {model_2_name} Mean:     {np.mean(rankic_mean_2):.4f}")
        print(f"   Differenz:              {rankic_test['mean_diff']:.4f}")
        print(f"   95% CI:                 [{rankic_test['ci_95'][0]:.4f}, {rankic_test['ci_95'][1]:.4f}]")
        print(f"   t-Statistik:            {rankic_test['t_statistic']:.3f}")
        print(f"   p-Wert:                 {rankic_test['p_value']:.4f}")
        
        if rankic_test['significant']:
            print(f"   ✅ SIGNIFIKANT besser (p < 0.05)")
        else:
            print(f"   ⚠️  NICHT signifikant (p >= 0.05)")
    
    # 5. Diebold-Mariano Test für Forecast-Genauigkeit
    print("\n📊 Forecast-Genauigkeit - Diebold-Mariano Test:")
    
    # Sammle alle Fehler
    all_errors_1 = []
    all_errors_2 = []
    
    for ticker in common_tickers:
        res_1 = results_1['asset_results'][ticker]
        res_2 = results_2['asset_results'][ticker]
        
        actuals_1 = np.array(res_1['raw_values']['actual'])
        preds_1 = np.array(res_1['raw_values']['predicted'])
        
        actuals_2 = np.array(res_2['raw_values']['actual'])
        preds_2 = np.array(res_2['raw_values']['predicted'])
        
        # Gleiche Länge verwenden
        min_len = min(len(actuals_1), len(actuals_2))
        
        all_errors_1.extend((actuals_1[:min_len] - preds_1[:min_len]).tolist())
        all_errors_2.extend((actuals_2[:min_len] - preds_2[:min_len]).tolist())
    
    if all_errors_1 and all_errors_2:
        dm_result = diebold_mariano_test(np.array(all_errors_1), np.array(all_errors_2))
        
        mae_1_val = np.mean(np.abs(all_errors_1))
        mae_2_val = np.mean(np.abs(all_errors_2))
        
        print(f"   {model_1_name} MAE:      {mae_1_val:.4f}")
        print(f"   {model_2_name} MAE:      {mae_2_val:.4f}")
        print(f"   DM-Statistik:           {dm_result['dm_statistic']:.3f}")
        print(f"   p-Wert:                 {dm_result['p_value']:.4f}")
        
        if dm_result['significant']:
            print(f"   ✅ SIGNIFIKANT besser (p < 0.05)")
        else:
            print(f"   ⚠️  NICHT signifikant (p >= 0.05)")
    
    # 6. Cross-Sectional IC Vergleich
    print("\n" + "="*80)
    print("TEIL 2: CROSS-SECTIONAL RANKIC (über Assets)")
    print("="*80)
    
    cs_comparison = compare_cross_sectional_ic(results_1, results_2, results_dir_1, results_dir_2)
    
    if 'error' in cs_comparison:
        print(f"\n❌ {cs_comparison['error']}")
    else:
        stats_1 = cs_comparison['model_1_stats']
        stats_2 = cs_comparison['model_2_stats']
        paired = cs_comparison['paired_test']
        
        print(f"\n📊 Cross-Sectional RankIC Vergleich ({cs_comparison['n_dates']} Tage):")
        print(f"\n   {model_1_name}:")
        print(f"      Mean RankIC:         {stats_1['Model1_RankIC_Mean']:.4f}")
        print(f"      95% CI:              [{stats_1['Model1_RankIC_CI95'][0]:.4f}, {stats_1['Model1_RankIC_CI95'][1]:.4f}]")
        
        print(f"\n   {model_2_name}:")
        print(f"      Mean RankIC:         {stats_2['Model2_RankIC_Mean']:.4f}")
        print(f"      95% CI:              [{stats_2['Model2_RankIC_CI95'][0]:.4f}, {stats_2['Model2_RankIC_CI95'][1]:.4f}]")
        
        print(f"\n   Paired t-test:")
        print(f"      Differenz:           {paired['mean_diff']:.4f}")
        print(f"      95% CI:              [{paired['ci_95'][0]:.4f}, {paired['ci_95'][1]:.4f}]")
        print(f"      t-Statistik:         {paired['t_statistic']:.3f}")
        print(f"      p-Wert:              {paired['p_value']:.4f}")
        
        if paired['significant']:
            print(f"      ✅ SIGNIFIKANT besser (p < 0.05)")
        else:
            print(f"      ⚠️  NICHT signifikant (p >= 0.05)")
        
        # 7. Visualisierung
        visualize_comparison(cs_comparison, model_1_name, model_2_name)
    
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG")
    print("="*80)
    print(f"\nDer Vergleich wurde mit {len(common_tickers)} gemeinsamen Assets durchgeführt.")
    print(f"Alle Tests verwenden einen Signifikanzlevel von α = 0.05.")
    print("\n💡 Interpretation:")
    print("   - Time-Series Metriken: Prognosequalität INNERHALB eines Assets")
    print("   - Cross-Sectional RankIC: Ranking-Fähigkeit ÜBER Assets hinweg")
    print("   - Diebold-Mariano: Forecast-Genauigkeit unter Autokorrelation")
    print("\n")


def visualize_comparison(cs_comparison: Dict, model_1_name: str, model_2_name: str):
    """Visualisiert den Cross-Sectional RankIC Vergleich."""
    if 'error' in cs_comparison:
        return
    
    rankic_1 = cs_comparison['rankic_timeseries_1']
    rankic_2 = cs_comparison['rankic_timeseries_2']
    dates = cs_comparison['dates']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Zeitreihen beider Modelle
    ax1 = axes[0]
    ax1.plot(dates, rankic_1, label=f'{model_1_name}', alpha=0.6, linewidth=1.5, color='blue')
    ax1.plot(dates, rankic_2, label=f'{model_2_name}', alpha=0.6, linewidth=1.5, color='red')
    
    # Mittelwerte
    mean_1 = np.mean(rankic_1)
    mean_2 = np.mean(rankic_2)
    ax1.axhline(mean_1, color='blue', linestyle='--', alpha=0.5, label=f'{model_1_name} Mean: {mean_1:.3f}')
    ax1.axhline(mean_2, color='red', linestyle='--', alpha=0.5, label=f'{model_2_name} Mean: {mean_2:.3f}')
    ax1.axhline(0, color='black', linewidth=1, alpha=0.3)
    
    ax1.set_title("Cross-Sectional RankIC Comparison Over Time", fontsize=12, fontweight='bold')
    ax1.set_ylabel("RankIC")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Differenz
    ax2 = axes[1]
    diff = np.array(rankic_2) - np.array(rankic_1)
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax2.bar(dates, diff, color=colors, alpha=0.6, width=1.0)
    ax2.axhline(0, color='black', linewidth=1)
    
    mean_diff = np.mean(diff)
    ax2.axhline(mean_diff, color='purple', linestyle='--', linewidth=2, 
                label=f'Mean Difference: {mean_diff:.3f}')
    
    ax2.set_title(f"RankIC Difference ({model_2_name} - {model_1_name})", 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("RankIC Difference")
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_rankic.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualisierung gespeichert: model_comparison_rankic.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Wissenschaftlicher Modellvergleich')
    parser.add_argument('--baseline', type=str, default='results_chronos',
                       help='Verzeichnis mit Baseline-Ergebnissen (Standard: results_chronos)')
    parser.add_argument('--finetuned', type=str, default='results_chronos_finetuned',
                       help='Verzeichnis mit Fine-Tuned-Ergebnissen (Standard: results_chronos_finetuned)')
    parser.add_argument('--baseline-name', type=str, default='Zero-Shot',
                       help='Name des Baseline-Modells')
    parser.add_argument('--finetuned-name', type=str, default='Fine-Tuned',
                       help='Name des Fine-Tuned-Modells')
    
    args = parser.parse_args()
    
    compare_models(
        results_dir_1=args.baseline,
        results_dir_2=args.finetuned,
        model_1_name=args.baseline_name,
        model_2_name=args.finetuned_name
    )
