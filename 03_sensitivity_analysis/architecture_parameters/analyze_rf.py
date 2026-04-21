import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    return df


def train_rf(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    return rf


def plot_feature_importance(importances, feature_names, metric_name, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    bars = ax.barh(range(len(sorted_features)), sorted_importances, color='steelblue')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Random Forest Feature Importance - {metric_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
        ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='03_sensitivity_analysis/architecture_parameters/results/architecture_search_150.csv')
    args = parser.parse_args()
    
    print("Random Forest Analysis - Architecture Parameters")
    print("=" * 60)
    
    df = load_results(args.csv)
    print(f"Loaded {len(df)} samples\n")
    
    metric_cols = {'mae', 'rankic'}
    feature_names = [c for c in df.columns if c not in metric_cols]
    X = df[feature_names].values
    
    output_dir = Path("03_sensitivity_analysis/architecture_parameters/results")
    
    for metric in ['mae', 'rankic']:
        print(f"\n{metric.upper()}:")
        print("-" * 60)
        
        y = df[metric].values
        
        rf = train_rf(X, y)
        importances = rf.feature_importances_
        
        for fname, imp in zip(feature_names, importances):
            print(f"  {fname:15s}: {imp:.4f}")
        
        plot_path = output_dir / f"rf_importance_{metric}.png"
        plot_feature_importance(importances, feature_names, metric.upper(), plot_path)
        print(f"\nPlot saved: {plot_path}")
    
    print("\n" + "=" * 60)
    print("Analysis complete")


if __name__ == "__main__":
    main()
