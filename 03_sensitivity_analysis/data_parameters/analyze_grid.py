import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_grid_results(results_dir="03_sensitivity_analysis/data_parameters/results/grid_results/raw"):
    results_dir = Path(results_dir)
    
    all_results = []
    
    for json_file in sorted(results_dir.glob("exp_*.json")):
        with open(json_file) as f:
            data = json.load(f)
        
        exp_id = data['experiment_id']
        params = data['parameters']
        
        mae_list = []
        ic_list = []
        rankic_list = []
        
        for asset, result in data['results'].items():
            metrics = result['metrics']
            mae_list.append(metrics.get('MAE_indicative', np.nan))
            ic_list.append(metrics.get('IC_TimeSeries_Mean', np.nan))
            rankic_list.append(metrics.get('RankIC_TimeSeries_Mean', np.nan))
        
        all_results.append({
            'exp_id': exp_id,
            'context_steps': params['context_steps'],
            'forecast_steps': params['forecast_steps'],
            'mae_mean': np.nanmean(mae_list),
            'ic_mean': np.nanmean(ic_list),
            'rankic_mean': np.nanmean(rankic_list),
            'n_assets': len(mae_list)
        })
    
    df = pd.DataFrame(all_results)
    return df


def create_heatmaps(df, output_dir="03_sensitivity_analysis/data_parameters/results/grid_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric in ['mae_mean', 'ic_mean', 'rankic_mean']:
        pivot = df.pivot(
            index='forecast_steps',
            columns='context_steps',
            values=metric
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r' if metric == 'mae_mean' else 'RdYlGn')
        plt.title(f'Grid Search: {metric.upper().replace("_", " ")}')
        plt.xlabel('Context Steps')
        plt.ylabel('Forecast Steps')
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / f'heatmap_{metric}.png'}")


def print_summary(df):
    print("=" * 80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    print(f"Total experiments: {len(df)}")
    print(f"Context steps range: {df['context_steps'].min()} - {df['context_steps'].max()}")
    print(f"Forecast steps range: {df['forecast_steps'].min()} - {df['forecast_steps'].max()}")
    print()
    
    print("MAE Statistics:")
    print("-" * 80)
    print(f"  Mean: {df['mae_mean'].mean():.3f}")
    print(f"  Std:  {df['mae_mean'].std():.3f}")
    print(f"  Min:  {df['mae_mean'].min():.3f} at context={df.loc[df['mae_mean'].idxmin(), 'context_steps']:.0f}, forecast={df.loc[df['mae_mean'].idxmin(), 'forecast_steps']:.0f}")
    print(f"  Max:  {df['mae_mean'].max():.3f} at context={df.loc[df['mae_mean'].idxmax(), 'context_steps']:.0f}, forecast={df.loc[df['mae_mean'].idxmax(), 'forecast_steps']:.0f}")
    print()
    
    print("IC Statistics:")
    print("-" * 80)
    print(f"  Mean: {df['ic_mean'].mean():.4f}")
    print(f"  Std:  {df['ic_mean'].std():.4f}")
    print(f"  Best: {df['ic_mean'].max():.4f} at context={df.loc[df['ic_mean'].idxmax(), 'context_steps']:.0f}, forecast={df.loc[df['ic_mean'].idxmax(), 'forecast_steps']:.0f}")
    print()
    
    print("RankIC Statistics:")
    print("-" * 80)
    print(f"  Mean: {df['rankic_mean'].mean():.4f}")
    print(f"  Std:  {df['rankic_mean'].std():.4f}")
    print(f"  Best: {df['rankic_mean'].max():.4f} at context={df.loc[df['rankic_mean'].idxmax(), 'context_steps']:.0f}, forecast={df.loc[df['rankic_mean'].idxmax(), 'forecast_steps']:.0f}")
    print()
    
    print("Top 5 Configurations (by RankIC):")
    print("-" * 80)
    top5 = df.nlargest(5, 'rankic_mean')[['context_steps', 'forecast_steps', 'mae_mean', 'ic_mean', 'rankic_mean']]
    print(top5.to_string(index=False))
    print()
    
    print("Top 5 Configurations (by MAE - lower is better):")
    print("-" * 80)
    top5_mae = df.nsmallest(5, 'mae_mean')[['context_steps', 'forecast_steps', 'mae_mean', 'ic_mean', 'rankic_mean']]
    print(top5_mae.to_string(index=False))
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Grid Search Results')
    parser.add_argument('--results-dir', type=str, 
                       default='03_sensitivity_analysis/data_parameters/results/grid_results/raw')
    parser.add_argument('--output-dir', type=str,
                       default='03_sensitivity_analysis/data_parameters/results/grid_results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip heatmap generation')
    args = parser.parse_args()
    
    df = load_grid_results(args.results_dir)
    
    print_summary(df)
    
    csv_path = Path(args.output_dir) / "grid_search_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Full results saved: {csv_path}")
    print()
    
    if not args.no_plots:
        create_heatmaps(df, args.output_dir)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
