import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.analyze import sobol
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

sns.set_style("whitegrid")

def load_config():
    with open("sensitivity_analysis/config/parameter_space.yaml") as f:
        return yaml.safe_load(f)

def load_results(results_dir="sensitivity_analysis/results/raw"):
    results_dir = Path(results_dir)
    X = np.load(results_dir / "sobol_X.npy")
    N_samples = len(X)
    
    # Dictionary für alle Zielmetriken initialisieren
    Y = {
        'IC_Mean': np.zeros(N_samples),
        'RankIC_Mean': np.zeros(N_samples),
        'MAE_Mean': np.zeros(N_samples)
    }
    
    for i in range(N_samples):
        result_file = results_dir / f"exp_{i:04d}.json"
        
        ic_val, rank_ic_val, mae_val = np.nan, np.nan, np.nan
        
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            
            res = data.get('results', {})
            
            # Extraktion der Metriken aus den Asset-Ergebnissen
            ic_list = [r['metrics'].get('IC_TimeSeries_Mean', np.nan) 
                       for r in res.values() if 'metrics' in r]
            rank_ic_list = [r['metrics'].get('RankIC_TimeSeries_Mean', np.nan) 
                            for r in res.values() if 'metrics' in r]
            mae_list = [r['metrics'].get('MAE_indicative', np.nan) 
                        for r in res.values() if 'metrics' in r]
            
            if ic_list: ic_val = np.nanmean(ic_list)
            if rank_ic_list: rank_ic_val = np.nanmean(rank_ic_list)
            if mae_list: mae_val = np.nanmean(mae_list)

            # Innerhalb der load_results Schleife nach der Mittelung über Assets:

            # 1. IC Bestrafung (Ziel: Maximierung)
            # Wenn NaN, dann 0.0 (keine Korrelation)
            y_ic = ic_val if not np.isnan(ic_val) else 0.0

            # 2. RankIC Bestrafung (Ziel: Maximierung)
            y_rank_ic = rank_ic_val if not np.isnan(rank_ic_val) else 0.0

            # 3. MAE Bestrafung (Ziel: Minimierung)
            # Wenn NaN, dann ein massiver Fehlerwert
            y_mae = mae_val if not np.isnan(mae_val) else 999.0 

            # Zuweisung an die Arrays
            Y['IC_Mean'][i] = y_ic
            Y['RankIC_Mean'][i] = y_rank_ic
            Y['MAE_Mean'][i] = y_mae
                    
    return X, Y

def compute_sobol_indices(X, Y, config):
    param_names = list(config['parameter_space'].keys())
    
    # Problem-Definition für SALib (muss exakt zum Sampling passen)
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': [
            [config['parameter_space'][p]['min'], 
             config['parameter_space'][p]['max']]
            for p in param_names
        ]
    }
    
    sobol_results = {}
    
    for metric_name, y_values in Y.items():
        # Da load_results bereits Penalty-Werte (0.0 oder 999.0) gesetzt hat,
        # übergeben wir y_values direkt ohne weitere Transformation.
        try:
            # calc_second_order=False, da dies beim Sampling so definiert wurde
            Si = sobol.analyze(problem, y_values, calc_second_order=False)
            
            # Ergebnisse für die Serialisierung aufbereiten
            sobol_results[metric_name] = {
                'S1': Si['S1'].tolist(),        # First-order indices
                'ST': Si['ST'].tolist(),        # Total-order indices
                'S1_conf': Si['S1_conf'].tolist(),
                'ST_conf': Si['ST_conf'].tolist(),
                'names': param_names
            }
        except Exception as e:
            print(f"Fehler bei der Analyse der Metrik {metric_name}: {e}")
            
    return sobol_results

def print_report(sobol_results):
    print("\n" + "=" * 80)
    print("SOBOL SENSITIVITY ANALYSIS REPORT")
    print("=" * 80)
    
    for metric_name, results in sobol_results.items():
        print(f"\nMetric: {metric_name}")
        print("-" * 80)
        print(f"{'Parameter':<20} {'S1 (Main)':<15} {'ST (Total)':<15} {'Interaction':<15}")
        print("-" * 80)
        
        for i, param in enumerate(results['names']):
            s1 = results['S1'][i]
            st = results['ST'][i]
            interaction = st - s1
            
            print(f"{param:<20} {s1:>10.4f}     {st:>10.4f}     {interaction:>10.4f}")
        
        print()
        
        max_s1_idx = np.argmax(results['S1'])
        max_st_idx = np.argmax(results['ST'])
        
        print(f"Most influential (main): {results['names'][max_s1_idx]} (S1={results['S1'][max_s1_idx]:.4f})")
        print(f"Most influential (total): {results['names'][max_st_idx]} (ST={results['ST'][max_st_idx]:.4f})")
    
    print("\n" + "=" * 80)

def plot_sobol_indices(sobol_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric, results in sobol_results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        S1 = np.array(results['S1'])
        ST = np.array(results['ST'])
        params = results['names']
        
        x = np.arange(len(params))
        width = 0.6
        
        bars1 = ax1.bar(x, S1, width, color='steelblue', alpha=0.8)
        ax1.set_ylabel('First-order Sensitivity (S1)', fontsize=11)
        ax1.set_title('Main Effects', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(params, rotation=45, ha='right')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars1, S1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        bars2 = ax2.bar(x, ST, width, color='darkorange', alpha=0.8)
        ax2.set_ylabel('Total Sensitivity (ST)', fontsize=11)
        ax2.set_title('Total Effects', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(params, rotation=45, ha='right')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars2, ST):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'Sobol Sensitivity: {metric}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'sobol_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_parameter_response(X, Y, param_names, output_dir):
    output_dir = Path(output_dir)
    
    for metric_name, y_values in Y.items():
        valid_idx = ~np.isnan(y_values)
        X_valid = X[valid_idx]
        y_valid = y_values[valid_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for i, param in enumerate(param_names):
            ax = axes[i]
            
            ax.scatter(X_valid[:, i], y_valid, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
            ax.set_xlabel(param, fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'{metric_name} vs {param}', fontsize=12)
            ax.grid(alpha=0.3)
            
            try:
                z = np.polyfit(X_valid[:, i], y_valid, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(X_valid[:, i].min(), X_valid[:, i].max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            except:
                pass
        
        plt.suptitle(f'Parameter Response: {metric_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'response_{metric_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze sensitivity results')
    parser.add_argument('--visualize', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    
    config = load_config()
    param_names = list(config['parameter_space'].keys())
    
    print("Loading experiment results...")
    X, Y = load_results()
    print(f"Loaded {len(X)} experiments")
    print(f"Metrics: {list(Y.keys())}")
    print()
    
    print("Computing Sobol indices...")
    sobol_results = compute_sobol_indices(X, Y, config)
    print()
    
    output_dir = Path("sensitivity_analysis/results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'sensitivity_indices.json', 'w') as f:
        json.dump(sobol_results, f, indent=2)
    
    print_report(sobol_results)
    
    report_file = output_dir / 'sensitivity_report.txt'
    with open(report_file, 'w') as f:
        sys.stdout = f
        print_report(sobol_results)
        sys.stdout = sys.__stdout__
    
    print(f"\nReport saved: {report_file}")
    
    if args.visualize:
        print("\nGenerating visualizations...")
        fig_dir = output_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)
        
        plot_sobol_indices(sobol_results, fig_dir)
        plot_parameter_response(X, Y, param_names, fig_dir)
        
        print(f"Figures saved: {fig_dir}")
    
    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
