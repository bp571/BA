import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, t as t_dist
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.metrics import calculate_ic_statistics


def evaluate_multi_seed(results_dir="01_model_comparison/results/kronos"):
    """
    Evaluiert Ergebnisse über mehrere Seeds per Ensemble-Methode.

    Predictions werden pro (Ticker, Datum) über alle Seeds gemittelt (Ensemble),
    dann wird der Cross-Sectional RankIC einmalig pro Tag berechnet.
    N = eindeutige Testtage, keine Pseudo-Replikation durch Seed-Pooling.
    """
    base_path = Path(results_dir)

    if not base_path.exists():
        print(f"❌ FEHLER: Verzeichnis nicht gefunden: {results_dir}")
        return

    seed_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("seed_")])

    if not seed_dirs:
        print(f"❌ Keine Seed-Verzeichnisse gefunden in {results_dir}/")
        print(f"   Erwarte Unterverzeichnisse: seed_13/, seed_42/, etc.")
        return

    print(f"\n{'='*80}")
    print(f"MULTI-SEED EVALUATION (Ensemble): {results_dir}")
    print(f"{'='*80}")
    print(f"Gefundene Seeds: {len(seed_dirs)}")

    # Phase 1: Rohdaten aller Seeds sammeln
    # Actuals + Anchors sind datenbasiert (seed-unabhängig) → einmalig speichern
    # Predictions variieren je Seed → als Liste sammeln und dann mitteln
    ticker_data: dict = {}

    for seed_dir in seed_dirs:
        file_path = seed_dir / "final_energy_study.json"
        if not file_path.exists():
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        summary = data.get('summary', {})

        for ticker in summary.keys():
            res_file = seed_dir / f"result_{ticker}.json"
            if not res_file.exists():
                continue
            with open(res_file, 'r') as f:
                res = json.load(f)

            rv = res['raw_values']
            dates = pd.to_datetime(rv['dates'])

            pred_series = pd.Series(rv['predicted'], index=dates, name=ticker)

            if ticker not in ticker_data:
                ticker_data[ticker] = {
                    'actual': pd.Series(rv['actual'], index=dates, name=ticker),
                    'pred_list': [pred_series],
                    'anchor': pd.Series(rv['anchors'], index=dates, name=ticker) if 'anchors' in rv else None,
                }
            else:
                ticker_data[ticker]['pred_list'].append(pred_series)

    if not ticker_data:
        print("\n❌ FEHLER: Keine Ticker-Daten geladen.")
        return

    # Phase 2: Ensemble-DataFrames bauen
    actual_dfs = []
    pred_dfs = []
    anchor_dfs = []

    for ticker, td in ticker_data.items():
        actual_dfs.append(td['actual'])
        # Ensemble: Mittelwert der Predictions über alle Seeds pro Datum
        pred_ensemble = pd.concat(td['pred_list'], axis=1).mean(axis=1)
        pred_ensemble.name = ticker
        pred_dfs.append(pred_ensemble)
        if td['anchor'] is not None:
            anchor_dfs.append(td['anchor'])

    df_act = pd.concat(actual_dfs, axis=1).sort_index()
    df_pre = pd.concat(pred_dfs, axis=1).sort_index()

    # Phase 3: Log-Returns berechnen
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
    df_act_cs = df_act.loc[df_act.index.intersection(common_idx)]
    df_pre_cs = df_pre.loc[df_pre.index.intersection(common_idx)]

    # Phase 4: Cross-Sectional RankIC pro Tag auf Ensemble-Predictions
    all_ic_values = []
    all_mae_values = []

    for t in df_act_ret.index:
        a_t = df_act_ret.loc[t]
        p_t = df_pre_ret.loc[t]

        mask = a_t.notna() & p_t.notna()
        n_assets = mask.sum()

        if n_assets >= 10:
            ric, _ = spearmanr(p_t[mask], a_t[mask])
            all_ic_values.append(ric)

            if t in df_act_cs.index and t in df_pre_cs.index:
                act_p = df_act_cs.loc[t]
                pre_p = df_pre_cs.loc[t]
                price_mask = act_p.notna() & pre_p.notna() & mask
                if price_mask.sum() >= 10:
                    all_mae_values.append(
                        np.mean(np.abs(act_p[price_mask].values - pre_p[price_mask].values))
                    )

    if not all_ic_values:
        print("\n❌ FEHLER: Keine gültigen IC-Werte berechnet.")
        return

    # Phase 5: Statistiken
    ic_stats = calculate_ic_statistics(all_ic_values, prefix="RankIC")

    mean_ic = ic_stats['RankIC_Mean']
    ci_lower, ci_upper = ic_stats['RankIC_CI95']
    n_total = ic_stats['RankIC_Count']
    n_eff = ic_stats['RankIC_Effective_N']
    std_ic = ic_stats['RankIC_Std']

    if n_eff > 1:
        t_stat = mean_ic / (std_ic / np.sqrt(n_eff))
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=max(1, n_eff - 1)))
    else:
        t_stat = np.nan
        p_value = np.nan

    mean_mae = np.mean(all_mae_values) if all_mae_values else np.nan
    std_mae = np.std(all_mae_values, ddof=1) if len(all_mae_values) > 1 else np.nan

    print(f"\nFinal Statistics (Ensemble ueber {len(seed_dirs)} Seeds):")
    print(f"   Seeds:                {len(seed_dirs)}")
    print(f"   Total Days:           {n_total}")
    print(f"   Effective N:          {n_eff:.1f}")
    print(f"   Mean RankIC:          {mean_ic:.4f}")
    print(f"   95% CI:               [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   Standard Deviation:   {std_ic:.4f}")
    print(f"   t-statistic:          {t_stat:.2f}")
    print(f"   p-value (H0: IC=0):   {p_value:.4f}")

    significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
    print(f"   Result:               {significance}")
    print(f"\n   Mean MAE (price):     {mean_mae:.4f}")
    print(f"   MAE Std:              {std_mae:.4f}")

    positive_days = sum(1 for ic in all_ic_values if ic > 0)
    print(f"\nDirectional Analysis:")
    print(f"   Positive IC Days:     {positive_days}/{n_total} ({positive_days/n_total*100:.1f}%)")

    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Seed Evaluation (Ensemble)')
    parser.add_argument('--results-dir', type=str, default='01_model_comparison/results/kronos',
                       help='Results directory')

    args = parser.parse_args()
    evaluate_multi_seed(results_dir=args.results_dir)
