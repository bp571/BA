"""
Wissenschaftlich korrekter Modellvergleich zwischen zwei Modellen

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

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from experiments.metrics import calculate_ic_statistics


def find_seed_dirs(base_dir):
    """Findet alle seed_X Unterverzeichnisse."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    seed_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    return seed_dirs


def load_results_single(results_path: Path) -> Dict:
    """Lade Ergebnisse aus einem einzelnen Results-Verzeichnis."""
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


def load_results(results_dir: str) -> Dict:
    """
    Lade Ergebnisse aus einem Results-Verzeichnis.
    Unterstützt automatisch Multi-Seed Aggregation.
    """
    results_path = Path(results_dir)
    seed_dirs = find_seed_dirs(results_dir)
    
    # Falls keine Seeds gefunden: alte Struktur nutzen
    if not seed_dirs:
        return load_results_single(results_path)
    
    # Multi-Seed: Aggregiere alle Seeds
    print(f"   Gefunden: {len(seed_dirs)} Seeds, aggregiere...")
    
    all_data = []
    for seed_dir in seed_dirs:
        try:
            data = load_results_single(seed_dir)
            all_data.append(data)
        except FileNotFoundError:
            print(f"   Ueberspringe {seed_dir.name}: Ergebnisse nicht vollstaendig")
    
    if not all_data:
        raise FileNotFoundError(f"Keine gültigen Seed-Ergebnisse in {results_dir}")
    
    # Aggregiere: Nimm erste als Basis, erweitere asset_results
    aggregated = all_data[0].copy()
    aggregated['n_seeds'] = len(all_data)
    aggregated['seeds_aggregated'] = [d.get('random_seed', 'unknown') for d in all_data]
    
    # Kombiniere asset_results von allen Seeds
    combined_asset_results = {}
    all_tickers = set()
    for data in all_data:
        all_tickers.update(data['asset_results'].keys())
    
    for ticker in all_tickers:
        # Sammle raw_values von allen Seeds für dieses Asset
        combined_actuals = []
        combined_predicted = []
        combined_dates = []
        combined_anchors = []
        
        for data in all_data:
            if ticker in data['asset_results']:
                rv = data['asset_results'][ticker]['raw_values']
                combined_actuals.extend(rv['actual'])
                combined_predicted.extend(rv['predicted'])
                combined_dates.extend(rv['dates'])
                if 'anchors' in rv:
                    combined_anchors.extend(rv['anchors'])
        
        if combined_actuals:
            combined_asset_results[ticker] = {
                'ticker': ticker,
                'raw_values': {
                    'actual': combined_actuals,
                    'predicted': combined_predicted,
                    'dates': combined_dates,
                    'anchors': combined_anchors if combined_anchors else None
                }
            }
    
    aggregated['asset_results'] = combined_asset_results
    aggregated['n_assets_processed'] = len(combined_asset_results)
    
    # Update summary mit aggregierten Metriken (Durchschnitt über Seeds)
    aggregated_summary = {}
    for ticker in all_tickers:
        ticker_metrics = []
        for data in all_data:
            if ticker in data.get('summary', {}):
                ticker_metrics.append(data['summary'][ticker])
        
        if ticker_metrics:
            # Durchschnitt aller Metriken über Seeds
            avg_metrics = {}
            for key in ticker_metrics[0].keys():
                values = [m[key] for m in ticker_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
            aggregated_summary[ticker] = avg_metrics
    
    aggregated['summary'] = aggregated_summary
    
    return aggregated


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
    
    # Bereite Daten vor: Aggregiere nach (date, ticker)
    data_1 = []
    data_2 = []
    
    for ticker in common_tickers:
        rv1 = asset_results_1[ticker]['raw_values']
        rv2 = asset_results_2[ticker]['raw_values']
        
        for i, date in enumerate(rv1['dates']):
            data_1.append({
                'date': pd.to_datetime(date),
                'ticker': ticker,
                'actual': rv1['actual'][i],
                'predicted': rv1['predicted'][i],
                'anchor': rv1['anchors'][i] if 'anchors' in rv1 and rv1['anchors'] else None
            })
        
        for i, date in enumerate(rv2['dates']):
            data_2.append({
                'date': pd.to_datetime(date),
                'ticker': ticker,
                'actual': rv2['actual'][i],
                'predicted': rv2['predicted'][i],
                'anchor': rv2['anchors'][i] if 'anchors' in rv2 and rv2['anchors'] else None
            })
    
    df1 = pd.DataFrame(data_1)
    df2 = pd.DataFrame(data_2)
    
    # Gruppiere nach (date, ticker) und mittele (für Multi-Seed)
    df1 = df1.groupby(['date', 'ticker']).mean().reset_index()
    df2 = df2.groupby(['date', 'ticker']).mean().reset_index()
    
    # Berechne Returns
    if df1['anchor'].notna().any() and df2['anchor'].notna().any():
        df1['actual_ret'] = np.log(df1['actual'] / df1['anchor'])
        df1['pred_ret'] = np.log(df1['predicted'] / df1['anchor'])
        df2['actual_ret'] = np.log(df2['actual'] / df2['anchor'])
        df2['pred_ret'] = np.log(df2['predicted'] / df2['anchor'])
    else:
        # Fallback: ohne Anchor
        df1 = df1.sort_values(['ticker', 'date'])
        df2 = df2.sort_values(['ticker', 'date'])
        df1['actual_ret'] = df1.groupby('ticker')['actual'].transform(lambda x: np.log(x / x.shift(1)))
        df1['pred_ret'] = df1.groupby('ticker')['predicted'].transform(lambda x: np.log(x / x.shift(1)))
        df2['actual_ret'] = df2.groupby('ticker')['actual'].transform(lambda x: np.log(x / x.shift(1)))
        df2['pred_ret'] = df2.groupby('ticker')['predicted'].transform(lambda x: np.log(x / x.shift(1)))
    
    # Finde gemeinsame Datums
    common_dates = sorted(set(df1['date']).intersection(set(df2['date'])))
    
    if len(common_dates) == 0:
        return {'error': 'Keine gemeinsamen Zeitpunkte gefunden'}
    
    # Berechne tägliche RankIC
    rankic_1_list = []
    rankic_2_list = []
    valid_dates = []
    
    for t in common_dates:
        day_data_1 = df1[df1['date'] == t].copy()
        day_data_2 = df2[df2['date'] == t].copy()
        
        # Merge auf gemeinsame Ticker
        merged = pd.merge(day_data_1[['ticker', 'actual_ret', 'pred_ret']],
                         day_data_2[['ticker', 'actual_ret', 'pred_ret']],
                         on='ticker', suffixes=('_1', '_2'))
        
        # Nur Zeilen mit gültigen Werten
        merged = merged.dropna()
        
        if len(merged) >= 10:  # Mindestens 10 Assets
            ric_1, _ = spearmanr(merged['pred_ret_1'], merged['actual_ret_1'])
            ric_2, _ = spearmanr(merged['pred_ret_2'], merged['actual_ret_2'])
            
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


def compare_models(results_dir_1: str, results_dir_2: str, model_1_name: str = "Model-1",
                   model_2_name: str = "Model-2"):
    """
    Hauptfunktion für wissenschaftlich korrekten Modellvergleich.
    
    Args:
        results_dir_1: Verzeichnis mit Ergebnissen des ersten Modells
        results_dir_2: Verzeichnis mit Ergebnissen des zweiten Modells
        model_1_name: Name des ersten Modells
        model_2_name: Name des zweiten Modells
    """
    print("\n" + "="*80)
    print(f"WISSENSCHAFTLICHER MODELLVERGLEICH: {model_1_name} vs. {model_2_name}")
    print("="*80)
    
    # 1. Lade Ergebnisse
    print(f"\nLade Ergebnisse...")
    try:
        results_1 = load_results(results_dir_1)
        results_2 = load_results(results_dir_2)
    except FileNotFoundError as e:
        print(f"\nFEHLER: {e}")
        print("\nTipp: Fuehren Sie zuerst die Evaluationen aus:")
        return

    # Seeds Info
    seeds_info_1 = f" ({results_1['n_seeds']} Seeds)" if 'n_seeds' in results_1 else ""
    seeds_info_2 = f" ({results_2['n_seeds']} Seeds)" if 'n_seeds' in results_2 else ""

    print(f"   OK {model_1_name}: {results_1['n_assets_processed']} Assets{seeds_info_1}")
    print(f"   OK {model_2_name}: {results_2['n_assets_processed']} Assets{seeds_info_2}")
    
    if 'seeds_aggregated' in results_1:
        print(f"      Seeds {model_1_name}: {results_1['seeds_aggregated']}")
    if 'seeds_aggregated' in results_2:
        print(f"      Seeds {model_2_name}: {results_2['seeds_aggregated']}")
    
    # 2. Gemeinsame Assets identifizieren
    summary_1 = results_1.get('summary', {})
    summary_2 = results_2.get('summary', {})
    
    common_tickers = set(summary_1.keys()) & set(summary_2.keys())
    
    if not common_tickers:
        print("\nFEHLER: Keine gemeinsamen Assets gefunden!")
        return
    
    print(f"\nGemeinsame Assets: {len(common_tickers)}")
    
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
        print("\nRankIC (Time-Series) - Paired t-test:")
        rankic_test = paired_t_test(np.array(rankic_mean_1), np.array(rankic_mean_2), "RankIC_TS")
        
        print(f"   {model_1_name} Mean:     {np.mean(rankic_mean_1):.4f}")
        print(f"   {model_2_name} Mean:     {np.mean(rankic_mean_2):.4f}")
        print(f"   Differenz:              {rankic_test['mean_diff']:.4f}")
        print(f"   95% CI:                 [{rankic_test['ci_95'][0]:.4f}, {rankic_test['ci_95'][1]:.4f}]")
        print(f"   t-Statistik:            {rankic_test['t_statistic']:.3f}")
        print(f"   p-Wert:                 {rankic_test['p_value']:.4f}")
        
        if rankic_test['significant']:
            print(f"   SIGNIFIKANT besser (p < 0.05)")
        else:
            print(f"   NICHT signifikant (p >= 0.05)")

    # 5. Diebold-Mariano Test fuer Forecast-Genauigkeit
    print("\nForecast-Genauigkeit - Diebold-Mariano Test:")
    
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
            print(f"   SIGNIFIKANT besser (p < 0.05)")
        else:
            print(f"   NICHT signifikant (p >= 0.05)")
    
    # 6. Cross-Sectional IC Vergleich
    print("\n" + "="*80)
    print("TEIL 2: CROSS-SECTIONAL RANKIC (über Assets)")
    print("="*80)
    
    cs_comparison = compare_cross_sectional_ic(results_1, results_2, results_dir_1, results_dir_2)
    
    if 'error' in cs_comparison:
        print(f"\nFEHLER: {cs_comparison['error']}")
    else:
        stats_1 = cs_comparison['model_1_stats']
        stats_2 = cs_comparison['model_2_stats']
        paired = cs_comparison['paired_test']
        
        print(f"\nCross-Sectional RankIC Vergleich ({cs_comparison['n_dates']} Tage):")
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
            print(f"      SIGNIFIKANT besser (p < 0.05)")
        else:
            print(f"      NICHT signifikant (p >= 0.05)")
        
        # 7. Visualisierung
        visualize_comparison(cs_comparison, model_1_name, model_2_name)


def visualize_comparison(cs_comparison: Dict, model_1_name: str, model_2_name: str):
    """Visualisiert Mean RankIC mit 95% CI für beide Modelle."""
    if 'error' in cs_comparison:
        return
    
    stats_1 = cs_comparison['model_1_stats']
    stats_2 = cs_comparison['model_2_stats']
    
    mean_1 = stats_1['Model1_RankIC_Mean']
    ci_1 = stats_1['Model1_RankIC_CI95']
    
    mean_2 = stats_2['Model2_RankIC_Mean']
    ci_2 = stats_2['Model2_RankIC_CI95']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = [model_1_name, model_2_name]
    means = [mean_1, mean_2]
    errors_lower = [mean_1 - ci_1[0], mean_2 - ci_2[0]]
    errors_upper = [ci_1[1] - mean_1, ci_2[1] - mean_2]
    
    x_pos = np.arange(len(models))
    colors = ['blue', 'red']
    
    ax.errorbar(x_pos, means, yerr=[errors_lower, errors_upper], fmt='o',
                markersize=10, color='black', ecolor='black', capsize=5, capthick=2, linewidth=2)
    
    for i, color in enumerate(colors):
        ax.plot(x_pos[i], means[i], 'o', markersize=10, color=color, alpha=0.8)
    
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Mean RankIC", fontsize=12, fontweight='bold')
    ax.set_title("Cross-Sectional RankIC Comparison (with 95% CI)",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (m, ci) in enumerate([(mean_1, ci_1), (mean_2, ci_2)]):
        ax.text(i, m + (ci[1] - m) + 0.01, f'{m:.4f}\n[{ci[0]:.4f}, {ci[1]:.4f}]',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison_rankic.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualisierung gespeichert: model_comparison_rankic.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Wissenschaftlicher Modellvergleich')
    parser.add_argument('--baseline', type=str, default='01_model_comparison/results/chronos',
                       help='Verzeichnis mit Ergebnissen des ersten Modells (Standard: 01_model_comparison/results/chronos)')
    parser.add_argument('--comparison', type=str, default='01_model_comparison/results/kronos',
                       help='Verzeichnis mit Ergebnissen des zweiten Modells (Standard: 01_model_comparison/results/kronos)')
    parser.add_argument('--baseline-name', type=str, default='Chronos',
                       help='Name des ersten Modells')
    parser.add_argument('--comparison-name', type=str, default='Kronos',
                       help='Name des zweiten Modells')
    
    args = parser.parse_args()
    
    compare_models(
        results_dir_1=args.baseline,
        results_dir_2=args.comparison,
        model_1_name=args.baseline_name,
        model_2_name=args.comparison_name
    )
