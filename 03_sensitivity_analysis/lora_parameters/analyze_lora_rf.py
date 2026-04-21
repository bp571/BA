import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path


FEATURE_NAMES = ['lora_r', 'lora_alpha', 'lora_dropout', 'learning_rate', 'use_ffn']
FEATURE_LABELS = {
    'lora_r':        'LoRA rank (r)',
    'lora_alpha':    'LoRA alpha',
    'lora_dropout':  'LoRA dropout',
    'learning_rate': 'Learning rate',
    'use_ffn':       'Target FFN layers',
}


def train_rf(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    return rf


def plot_importance(importances, feature_names, metric_name, output_path):
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    indices = np.argsort(importances)[::-1]
    sorted_labels = [labels[i] for i in indices]
    sorted_vals   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(sorted_labels)), sorted_vals, color='steelblue')
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'RF Feature Importance – LoRA Hyperparameters ({metric_name})', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for bar, val in zip(bars, sorted_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv',
        type=str,
        default='03_sensitivity_analysis/lora_parameters/results/lora_search_60.csv',
    )
    args = parser.parse_args()

    print("Random Forest Analysis – LoRA Hyperparameters")
    print("=" * 60)

    df = pd.read_csv(args.csv).dropna()
    print(f"Loaded {len(df)} samples\n")

    X = df[FEATURE_NAMES].values
    output_dir = Path(args.csv).parent

    for metric in ['mae', 'rankic']:
        print(f"\n{metric.upper()}:")
        print("-" * 60)

        rf = train_rf(X, df[metric].values)
        importances = rf.feature_importances_

        for fname, imp in zip(FEATURE_NAMES, importances):
            print(f"  {FEATURE_LABELS.get(fname, fname):25s}: {imp:.4f}")

        plot_path = output_dir / f"rf_importance_lora_{metric}.png"
        plot_importance(importances, FEATURE_NAMES, metric.upper(), plot_path)
        print(f"  Plot: {plot_path}")

    top_features = _top_features(df, X)
    print(f"\n{'='*60}")
    print(f"Recommended: set the top 1-2 LoRA params from the MAE ranking")
    print(f"Top features (MAE): {top_features}")


def _top_features(df, X):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, df['mae'].values)
    order = np.argsort(rf.feature_importances_)[::-1]
    return [FEATURE_NAMES[i] for i in order[:2]]


if __name__ == "__main__":
    main()
