"""
Automatisierter Workflow für wissenschaftlich korrekten Modellvergleich

Führt folgende Schritte aus:
1. Prüft, ob Zero-Shot Ergebnisse existieren (sonst evaluieren)
2. Evaluiert Fine-Tuned Modell
3. Führt statistischen Vergleich durch
"""

import subprocess
import sys
from pathlib import Path
import json


def check_results_exist(results_dir: str) -> bool:
    """Prüft, ob Ergebnisse bereits existieren."""
    path = Path(results_dir) / "final_energy_study.json"
    return path.exists()


def run_evaluation(script_name: str, model_name: str):
    """Führt eine Evaluation aus."""
    print(f"\n{'='*80}")
    print(f"EVALUIERE: {model_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {model_name} Evaluation erfolgreich abgeschlossen")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FEHLER bei {model_name} Evaluation:")
        print(f"   {e}")
        return False


def run_comparison():
    """Führt den statistischen Vergleich durch."""
    print(f"\n{'='*80}")
    print(f"STATISTISCHER VERGLEICH")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "compare_models.py"],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ Vergleich erfolgreich abgeschlossen")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FEHLER beim Vergleich:")
        print(f"   {e}")
        return False


def print_summary():
    """Gibt eine Zusammenfassung der Ergebnisse aus."""
    print(f"\n{'='*80}")
    print(f"ZUSAMMENFASSUNG")
    print(f"{'='*80}\n")
    
    results_dirs = {
        "Zero-Shot": "results_chronos",
        "Fine-Tuned": "results_chronos_finetuned"
    }
    
    for model_name, results_dir in results_dirs.items():
        path = Path(results_dir) / "final_energy_study.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"✅ {model_name}:")
            print(f"   - Assets verarbeitet: {data['n_assets_processed']}/{data['n_assets_total']}")
            print(f"   - Rechenzeit: {data['processing_time_seconds']:.1f}s")
            print(f"   - Ergebnisse: {results_dir}/")
        else:
            print(f"❌ {model_name}: Keine Ergebnisse gefunden")
    
    print(f"\n📊 Visualisierung: model_comparison_rankic.png")
    print(f"\n📖 Dokumentation: README_MODEL_COMPARISON.md")


def main():
    """Hauptfunktion für automatisierten Workflow."""
    print("\n" + "="*80)
    print("AUTOMATISIERTER MODELLVERGLEICH: Zero-Shot vs. Fine-Tuned")
    print("="*80)
    
    # 1. Prüfe LoRA-Adapter
    adapter_path = Path("models/chronos-2-lora-finetuned/final")
    if not adapter_path.exists():
        print(f"\n❌ FEHLER: LoRA-Adapter nicht gefunden: {adapter_path}")
        print("\n💡 Tipp: Führen Sie zuerst das Fine-Tuning aus:")
        print("   python finetune/train_chronos_lora.py")
        return
    
    print(f"\n✅ LoRA-Adapter gefunden: {adapter_path}")
    
    # 2. Zero-Shot Evaluation (nur wenn noch nicht vorhanden)
    if check_results_exist("results_chronos"):
        print(f"\n✅ Zero-Shot Ergebnisse bereits vorhanden (überspringen)")
    else:
        print(f"\n⚠️  Zero-Shot Ergebnisse nicht gefunden - starte Evaluation")
        if not run_evaluation("main_chronos.py", "Zero-Shot Chronos"):
            print("\n❌ Abbruch: Zero-Shot Evaluation fehlgeschlagen")
            return
    
    # 3. Fine-Tuned Evaluation
    print(f"\n🔧 Starte Fine-Tuned Evaluation...")
    if not run_evaluation("main_chronos_finetuned.py", "Fine-Tuned Chronos"):
        print("\n❌ Abbruch: Fine-Tuned Evaluation fehlgeschlagen")
        return
    
    # 4. Statistischer Vergleich
    print(f"\n📊 Starte statistischen Vergleich...")
    if not run_comparison():
        print("\n❌ Vergleich fehlgeschlagen")
        return
    
    # 5. Zusammenfassung
    print_summary()
    
    print("\n" + "="*80)
    print("✅ WORKFLOW ERFOLGREICH ABGESCHLOSSEN")
    print("="*80)
    print("\nNächste Schritte für Ihre Thesis:")
    print("1. Ergebnisse in README_MODEL_COMPARISON.md interpretieren")
    print("2. Visualisierung model_comparison_rankic.png analysieren")
    print("3. Statistische Signifikanz in Thesis dokumentieren")
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
