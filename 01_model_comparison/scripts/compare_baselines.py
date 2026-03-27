"""
Baseline Vergleich zwischen zwei Modellen (z.B. Chronos vs. Kronos)

Führt statistische Vergleiche zwischen zwei Zero-Shot Modellen durch.
Unterstützt Multi-Seed Experimente.
"""

import subprocess
import sys
from pathlib import Path
import json
import os

SEEDS = [13, 42, 123, 456, 789]


def check_results_exist(results_dir: str, seed: int = None) -> bool:
    if seed is not None:
        path = Path(results_dir) / f"seed_{seed}" / "final_energy_study.json"
    else:
        path = Path(results_dir) / "final_energy_study.json"
    return path.exists()


def run_evaluation(script_name: str, model_name: str, seed: int = None):
    seed_suffix = f" (Seed {seed})" if seed is not None else ""
    print(f"\n{'='*80}")
    print(f"EVALUIERE: {model_name}{seed_suffix}")
    print(f"{'='*80}\n")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    cmd = [sys.executable, script_name]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
            cwd=str(project_root),
            env=env
        )
        print(f"\n✅ {model_name}{seed_suffix} Evaluation erfolgreich")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FEHLER bei {model_name}{seed_suffix}: {e}")
        return False


def run_comparison(baseline_dir, comparison_dir, baseline_name, comparison_name):
    print(f"\n{'='*80}")
    print(f"STATISTISCHER VERGLEICH: {baseline_name} vs. {comparison_name}")
    print(f"{'='*80}\n")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    
    try:
        result = subprocess.run(
            [sys.executable, "01_model_comparison/scripts/compare_models.py",
             "--baseline", baseline_dir,
             "--comparison", comparison_dir,
             "--baseline-name", baseline_name,
             "--comparison-name", comparison_name],
            check=True,
            capture_output=False,
            text=True,
            cwd=str(project_root)
        )
        print(f"\n✅ Vergleich erfolgreich")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FEHLER beim Vergleich: {e}")
        return False


def print_summary(baseline_dir, comparison_dir, baseline_name, comparison_name, seeds):
    print(f"\n{'='*80}")
    print(f"ZUSAMMENFASSUNG")
    print(f"{'='*80}\n")
    
    for model_name, results_dir in [(baseline_name, baseline_dir), 
                                     (comparison_name, comparison_dir)]:
        print(f"{model_name}:")
        found_seeds = 0
        for seed in seeds:
            path = Path(results_dir) / f"seed_{seed}" / "final_energy_study.json"
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                print(f"  ✅ Seed {seed}: {data.get('n_assets_processed', 0)}/{data.get('n_assets_total', 0)} Assets, {data.get('processing_time_seconds', 0):.1f}s")
                found_seeds += 1
            else:
                print(f"  ❌ Seed {seed}: Nicht gefunden")
        
        if found_seeds > 0:
            print(f"  📁 Ergebnisse: {results_dir}/seed_*/")
        print()
    
    print(f"📊 Visualisierung: model_comparison_rankic.png")


def main():
    print("\n" + "="*80)
    print("BASELINE VERGLEICH (Multi-Seed)")
    print("="*80)
    print(f"\nSeeds: {SEEDS} (Total: {len(SEEDS)})")
    
    baseline_dir = "01_model_comparison/results/chronos"
    comparison_dir = "01_model_comparison/results/kronos"
    baseline_name = "Chronos"
    comparison_name = "Kronos"
    
    print(f"\nBaseline:   {baseline_name} ({baseline_dir})")
    print(f"Comparison: {comparison_name} ({comparison_dir})")
    
    # Iteriere über Seeds
    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"SEED {seed} / {len(SEEDS)}")
        print(f"{'='*80}")
        
        # Baseline
        if check_results_exist(baseline_dir, seed):
            print(f"\n✅ {baseline_name} (Seed {seed}) vorhanden")
        else:
            print(f"\n⚠️  {baseline_name} (Seed {seed}) nicht gefunden - starte Evaluation")
            if not run_evaluation("01_model_comparison/zeroshot/main_chronos.py", baseline_name, seed):
                print(f"\n❌ Abbruch: {baseline_name} Seed {seed} fehlgeschlagen")
                continue
        
        # Comparison
        if check_results_exist(comparison_dir, seed):
            print(f"\n✅ {comparison_name} (Seed {seed}) vorhanden")
        else:
            print(f"\n⚠️  {comparison_name} (Seed {seed}) nicht gefunden - starte Evaluation")
            if not run_evaluation("01_model_comparison/zeroshot/main_kronos.py", comparison_name, seed):
                print(f"\n❌ Abbruch: {comparison_name} Seed {seed} fehlgeschlagen")
                continue
    
    # Statistischer Vergleich
    print(f"\n{'='*80}")
    print(f"AGGREGIERTE AUSWERTUNG")
    print(f"{'='*80}")
    print(f"\n📊 Starte statistischen Vergleich über alle Seeds...")
    
    if not run_comparison(baseline_dir, comparison_dir, baseline_name, comparison_name):
        print("\n❌ Vergleich fehlgeschlagen")
        return
    
    # Zusammenfassung
    print_summary(baseline_dir, comparison_dir, baseline_name, comparison_name, SEEDS)
    
    print("\n" + "="*80)
    print("✅ BASELINE VERGLEICH ABGESCHLOSSEN")
    print("="*80)
    print(f"\n✅ Alle {len(SEEDS)} Seeds verarbeitet")
    print(f"\nErgebnisse:")
    print(f"  • Visualisierung: model_comparison_rankic.png")
    print(f"  • Detaillierte Statistiken in Terminal-Output")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline Model Comparison')
    parser.add_argument('--baseline-dir', type=str, 
                       default='01_model_comparison/results/chronos',
                       help='Baseline results directory')
    parser.add_argument('--comparison-dir', type=str,
                       default='01_model_comparison/results/kronos',
                       help='Comparison results directory')
    parser.add_argument('--baseline-name', type=str, default='Chronos',
                       help='Baseline model name')
    parser.add_argument('--comparison-name', type=str, default='Kronos',
                       help='Comparison model name')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation, only run comparison')
    
    args = parser.parse_args()
    
    if args.skip_evaluation:
        # Nur Vergleich, keine Evaluation
        baseline_dir = args.baseline_dir
        comparison_dir = args.comparison_dir
        baseline_name = args.baseline_name
        comparison_name = args.comparison_name
        
        print(f"\n⚠️  Skip Evaluation Mode - nur Vergleich")
        run_comparison(baseline_dir, comparison_dir, baseline_name, comparison_name)
        print_summary(baseline_dir, comparison_dir, baseline_name, comparison_name, SEEDS)
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\n\n⚠️  Abbruch durch Benutzer")
        except Exception as e:
            print(f"\n\n❌ Unerwarteter Fehler: {e}")
            import traceback
            traceback.print_exc()
