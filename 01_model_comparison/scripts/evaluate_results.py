import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, t as t_dist
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.metrics import calculate_ic_statistics


def evaluate_multi_seed(results_dir="results_chronos"):
    """
    Evaluiert Ergebnisse über mehrere Seeds.
    Lädt alle Seeds, aggregiert und gibt finale Statistiken aus.
    """
    base_path = Path(results_dir)
    
    if not base_path.exists():
        print(f"❌ FEHLER: Verzeichnis nicht gefunden: {results_dir}")
        return
    
    # Finde alle seed_X Verzeichnisse
    seed_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    
    if not seed_dirs:
        print(f"❌ Keine Seed-Verzeichnisse gefunden in {results_dir}/")
        print(f"   Erwarte Unterverzeichnisse: seed_13/, seed_42/, etc.")
        return
    
    print(f"\n{'='*80}")
    print(f"MULTI-SEED EVALUATION: {results_dir}")
    print(f"{'='*80}")
    print(f"Gefundene Seeds: {len(seed_dirs)}")

    # Sammle alle IC-Werte von allen Seeds
    all_ic_values = []
    
    for seed_dir in seed_dirs:
        file_path = seed_dir / "final_energy_study.json"
        if not file_path.exists():
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summary = data.get('summary', {})
        
        # Lade raw values
        actual_dfs = []
        pred_dfs = []
        anchor_dfs = []
        
        for ticker in summary.keys():
            res_file = seed_dir / f"result_{ticker}.json"
            if res_file.exists():
                with open(res_file, 'r') as f:
                    res = json.load(f)
                    rv = res['raw_values']
                    
                    dates = pd.to_datetime(rv['dates'])
                    actual_dfs.append(pd.Series(rv['actual'], index=dates, name=ticker))
                    pred_dfs.append(pd.Series(rv['predicted'], index=dates, name=ticker))
                    
                    if 'anchors' in rv:
                        anchor_dfs.append(pd.Series(rv['anchors'], index=dates, name=ticker))
        
        if not actual_dfs:
            continue
        
        # Merge zu DataFrames
        df_act = pd.concat(actual_dfs, axis=1).sort_index()
        df_pre = pd.concat(pred_dfs, axis=1).sort_index()
        
        # Returns berechnen
        if anchor_dfs:
            df_anc = pd.concat(anchor_dfs, axis=1).sort_index()
            df_act_ret = np.log(df_act / df_anc)
            df_pre_ret = np.log(df_pre / df_anc)
        else:
            df_act_ret = np.log(df_act / df_act.shift(1))
            df_pre_ret = np.log(df_pre / df_pre.shift(1))
        
        df_act_ret = df_act_ret.dropna(how='all')
        df_pre_ret = df_pre_ret.dropna(how='all')
        
        common_idx = df_act_ret.index.intersection(df_pre_ret.index)
        df_act_ret = df_act_ret.loc[common_idx]
        df_pre_ret = df_pre_ret.loc[common_idx]
        
        # Cross-Sectional RankIC für diesen Seed
        for t in df_act_ret.index:
            a_t = df_act_ret.loc[t]
            p_t = df_pre_ret.loc[t]
            
            mask = a_t.notna() & p_t.notna()
            n_assets = mask.sum()
            
            if n_assets >= 10:
                ric, _ = spearmanr(p_t[mask], a_t[mask])
                all_ic_values.append(ric)
    
    if not all_ic_values:
        print("\n❌ FEHLER: Keine gültigen IC-Werte über alle Seeds")
        return
    
    # Aggregierte Statistiken über alle Seeds
    ic_stats = calculate_ic_statistics(all_ic_values, prefix="RankIC")
    
    mean_ic = ic_stats['RankIC_Mean']
    ci_lower, ci_upper = ic_stats['RankIC_CI95']
    n_total = ic_stats['RankIC_Count']
    n_eff = ic_stats['RankIC_Effective_N']
    std_ic = ic_stats['RankIC_Std']
    
    # T-Test
    if n_eff > 1:
        t_stat = mean_ic / (std_ic / np.sqrt(n_eff))
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=max(1, n_eff-1)))
    else:
        t_stat = np.nan
        p_value = np.nan
    
    print(f"\n📊 Final Statistics:")
    print(f"   Seeds:                {len(seed_dirs)}")
    print(f"   Total Days:           {n_total}")
    print(f"   Effective N:          {n_eff}")
    print(f"   Mean RankIC:          {mean_ic:.4f}")
    print(f"   95% CI:               [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   Standard Deviation:   {std_ic:.4f}")
    print(f"   t-statistic:          {t_stat:.2f}")
    print(f"   p-value (H0: IC=0):   {p_value:.4f}")
    
    significance = "✅ SIGNIFICANT" if p_value < 0.05 else "⚠️  NOT SIGNIFICANT"
    print(f"   Result:               {significance}")
    
    positive_days = sum(1 for ic in all_ic_values if ic > 0)
    print(f"\n📈 Directional Analysis:")
    print(f"   Positive IC Days:     {positive_days}/{n_total} ({positive_days/n_total*100:.1f}%)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Seed Evaluation')
    parser.add_argument('--results-dir', type=str, default='results_chronos',
                       help='Results directory (default: results_chronos)')
    
    args = parser.parse_args()
    evaluate_multi_seed(results_dir=args.results_dir)
