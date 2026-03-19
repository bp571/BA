"""
Führt alle Main-Skripte mit mehreren Seeds nacheinander aus.
Einfach und direkt ohne subprocess.
"""

import sys
from pathlib import Path

# Projekt-Root zum Path hinzufügen
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Jetzt können wir die Module importieren
from zeroshot.main_chronos import main as chronos_main
from zeroshot.main_kronos import main as kronos_main
from finetune.main_chronos_finetuned import main as finetuned_main


# Seeds konfigurieren
SEEDS = [13, 42, 123, 456, 789]


def main():
    print("\n" + "="*80)
    print("MULTI-SEED EXPERIMENT RUNNER")
    print("="*80)
    print(f"\nSeeds: {SEEDS}")
    print(f"Total: {len(SEEDS)} Seeds\n")
    
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*80}")
        print(f"SEED {seed} ({i}/{len(SEEDS)})")
        print(f"{'='*80}\n")
        
        try:
            # 1. Chronos Zero-Shot
            print(f"[1/3] Running Chronos Zero-Shot (Seed {seed})...")
            chronos_main(seed=seed)
            print(f"✅ Chronos Zero-Shot (Seed {seed}) completed")
            
        except Exception as e:
            print(f"❌ Chronos Zero-Shot (Seed {seed}) failed: {e}")
            continue
        
        try:
            # 2. Kronos
            print(f"\n[2/3] Running Kronos (Seed {seed})...")
            kronos_main(seed=seed)
            print(f"✅ Kronos (Seed {seed}) completed")
            
        except Exception as e:
            print(f"❌ Kronos (Seed {seed}) failed: {e}")
            continue
        
        try:
            # 3. Fine-Tuned
            print(f"\n[3/3] Running Chronos Fine-Tuned (Seed {seed})...")
            finetuned_main(seed=seed)
            print(f"✅ Chronos Fine-Tuned (Seed {seed}) completed")
            
        except Exception as e:
            print(f"❌ Chronos Fine-Tuned (Seed {seed}) failed: {e}")
            continue
    
    print("\n" + "="*80)
    print("✅ ALL SEEDS COMPLETED")
    print("="*80)
    print(f"\nProcessed {len(SEEDS)} seeds")
    print("\nResults saved in:")
    print("  - results_chronos/seed_*/")
    print("  - results_kronos/seed_*/")
    print("  - results_chronos_finetuned/seed_*/")
    print("\nNext steps:")
    print("  python scripts/evaluate_results.py --results-dir results_chronos")
    print("  python scripts/compare_models.py")


if __name__ == "__main__":
    main()
