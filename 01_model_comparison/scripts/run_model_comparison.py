"""
Automatisierter Workflow für wissenschaftlich korrekten Modellvergleich

Führt folgende Schritte aus:
1. Prüft, ob Zero-Shot Ergebnisse existieren (sonst evaluieren)
2. Evaluiert Fine-Tuned Modell
3. Führt statistischen Vergleich durch

Unterstützt Multi-Seed Experimente für Robustheit.
"""

import subprocess
import sys
from pathlib import Path
import json
import os

# Zentrale Seed-Konfiguration
SEEDS = [13, 42, 123, 456, 789]  # Anpassen für mehr/weniger Seeds


def check_results_exist(results_dir: str, seed: int = None) -> bool:
    """Prüft, ob Ergebnisse bereits existieren."""
    if seed is not None:
        path = Path(results_dir) / f"seed_{seed}" / "final_energy_study.json"
    else:
        path = Path(results_dir) / "final_energy_study.json"
    return path.exists()


def run_evaluation(script_name: str, model_name: str, seed: int = None, adapter_path: str = None):
    """Führt eine Evaluation aus."""
    seed_suffix = f" (Seed {seed})" if seed is not None else ""
    print(f"\n{'='*80}")
    print(f"EVALUIERE: {model_name}{seed_suffix}")
    print(f"{'='*80}\n")
    
    # Working directory: Project root
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    cmd = [sys.executable, script_name]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if adapter_path is not None:
        cmd.extend(["--adapter-path", adapter_path])
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
            cwd=str(project_root),
            env=env
        )
        print(f"\n✅ {model_name}{seed_suffix} Evaluation erfolgreich abgeschlossen")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FEHLER bei {model_name}{seed_suffix} Evaluation:")
        print(f"   {e}")
        return False


def run_comparison():
    """Führt den statistischen Vergleich durch."""
    print(f"\n{'='*80}")
    print(f"STATISTISCHER VERGLEICH")
    print(f"{'='*80}\n")
    
    # Working directory: Project root
    project_root = Path(__file__).resolve().parent.parent.parent
    
    try:
        result = subprocess.run(
            [sys.executable, "01_model_comparison/scripts/compare_models.py"],
            check=True,
            capture_output=False,
            text=True,
            cwd=str(project_root)
        )
        print(f"\n✅ Vergleich erfolgreich abgeschlossen")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FEHLER beim Vergleich:")
        print(f"   {e}")
        return False


def print_summary(seeds):
    """Gibt eine Zusammenfassung der Ergebnisse aus."""
    print(f"\n{'='*80}")
    print(f"ZUSAMMENFASSUNG")
    print(f"{'='*80}\n")
    
    results_dirs = {
        "Zero-Shot": "results_chronos",
        "Fine-Tuned": "results_chronos_finetuned"
    }
    
    for model_name, results_dir in results_dirs.items():
        print(f"{model_name}:")
        found_seeds = 0
        for seed in seeds:
            path = Path(results_dir) / f"seed_{seed}" / "final_energy_study.json"
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                print(f"  ✅ Seed {seed}: {data['n_assets_processed']}/{data['n_assets_total']} Assets, {data['processing_time_seconds']:.1f}s")
                found_seeds += 1
            else:
                print(f"  ❌ Seed {seed}: Nicht gefunden")
        
        if found_seeds > 0:
            print(f"  📁 Ergebnisse: {results_dir}/seed_*/")
        print()
    
    print(f"📊 Visualisierung: model_comparison_rankic.png")
    print(f"📖 Dokumentation: README_MODEL_COMPARISON.md")


def main():
    """Hauptfunktion für automatisierten Workflow."""
    print("\n" + "="*80)
    print("AUTOMATISIERTER MODELLVERGLEICH: Zero-Shot vs. Fine-Tuned (Multi-Seed)")
    print("="*80)
    print(f"\nSeeds: {SEEDS} (Total: {len(SEEDS)})")
    
    # 1. Prüfe LoRA-Adapter
    adapter_path = Path(__file__).resolve().parent.parent.parent / "02_finetuning/models/chronos-2-lora-finetuned/final"
    if not adapter_path.exists():
        print(f"\n❌ FEHLER: LoRA-Adapter nicht gefunden: {adapter_path}")
        print("\n💡 Tipp: Führen Sie zuerst das Fine-Tuning aus:")
        print("   python finetune/train_chronos_lora.py")
        return
    
    print(f"\n✅ LoRA-Adapter gefunden: {adapter_path}")
    
    # 2. Iteriere über alle Seeds
    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"SEED {seed} / {len(SEEDS)}")
        print(f"{'='*80}")
        
        # Zero-Shot Evaluation (nur wenn noch nicht vorhanden)
        if check_results_exist("results_chronos", seed):
            print(f"\n✅ Zero-Shot (Seed {seed}) Ergebnisse bereits vorhanden (überspringen)")
        else:
            print(f"\n⚠️  Zero-Shot (Seed {seed}) nicht gefunden - starte Evaluation")
            if not run_evaluation("01_model_comparison/zeroshot/main_chronos.py", "Zero-Shot Chronos", seed):
                print(f"\n❌ Abbruch: Zero-Shot Seed {seed} fehlgeschlagen")
                continue
        
        # Fine-Tuned Evaluation
        print(f"\n🔧 Starte Fine-Tuned Evaluation (Seed {seed})...")
        if not run_evaluation("02_finetuning/evaluation/main_chronos_finetuned.py", "Fine-Tuned Chronos", seed, str(adapter_path)):
            print(f"\n❌ Abbruch: Fine-Tuned Seed {seed} fehlgeschlagen")
            continue
    
    # 3. Statistischer Vergleich (aggregiert über alle Seeds)
    print(f"\n{'='*80}")
    print(f"AGGREGIERTE AUSWERTUNG")
    print(f"{'='*80}")
    print(f"\n📊 Starte statistischen Vergleich über alle Seeds...")
    if not run_comparison():
        print("\n❌ Vergleich fehlgeschlagen")
        return
    
    # 4. Zusammenfassung
    print_summary(SEEDS)
    
    print("\n" + "="*80)
    print("✅ WORKFLOW ERFOLGREICH ABGESCHLOSSEN")
    print("="*80)
    print(f"\n✅ Alle {len(SEEDS)} Seeds verarbeitet")
    print("\nNächste Schritte für Ihre Thesis:")
    print("1. Ergebnisse in README_MODEL_COMPARISON.md interpretieren")
    print("2. Visualisierung model_comparison_rankic.png analysieren")
    print("3. Aggregierte Statistiken über Seeds dokumentieren")
    print("4. Robustheit der Ergebnisse über Seeds analysieren")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Abbruch durch Benutzer")
    except Exception as e:
        print(f"\n\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
