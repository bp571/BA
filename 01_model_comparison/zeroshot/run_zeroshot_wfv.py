"""
Zero-Shot Walk-Forward-Evaluation der Basis-Modelle (Kronos, Chronos).

Spiegelt die Fine-Tuned-WFV-Eval (02_finetuning/run_phase3_wfv.py) exakt, damit
die Cross-Sectional-RankIC direkt vergleichbar ist. Kein Training/Refit: das
Basis-Modell ist über alle Folds identisch; die Per-Fold-Laeufe schraenken die
Evaluation nur auf das Testfenster [Y-01-01, Y-12-31] ein (Context aus Jahr-1
gebuffert) — identisch zur Fine-Tuned-Eval.

Multi-Seed (42, 1, 7) reproduziert das Seed-Ensemble der FT-Eval; Kronos sampelt
stochastisch, daher veraendern Seeds die Predictions.

Output:    01_model_comparison/results/<model>/wfv/<pool>/<year>/seed_<seed>/
Auswerten: python 02_finetuning/scripts/evaluate_wfv_cs_rankic.py \
               --wfv-dir 01_model_comparison/results/<model>/wfv
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_FOLDS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_SEEDS = [42, 1, 7]

# Beide ZS-Skripte teilen dasselbe CLI (--config/--seed/--context/--forecast/
# --test-start/--test-end/--results-subdir) und schreiben unter results/<model>/.
SCRIPTS = {
    "kronos":  "01_model_comparison/zeroshot/main_kronos.py",
    "chronos": "01_model_comparison/zeroshot/main_chronos.py",
}


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (exit {res.returncode})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["kronos", "chronos"],
                   choices=list(SCRIPTS), help="ZS-Modelle, die per WFV laufen")
    p.add_argument("--folds", type=int, nargs="+", default=DEFAULT_FOLDS,
                   help="Testjahre (ein Fold pro Jahr) — identisch zur FT-WFV")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Seed-Ensemble (Kronos-Sampling ist stochastisch)")
    p.add_argument("--config", type=str, default="config/energy_assets_holdout.yaml")
    p.add_argument("--pool", type=str, default="holdout")
    p.add_argument("--context", type=int, default=80)
    p.add_argument("--forecast", type=int, default=18)
    args = p.parse_args()

    for model in args.models:
        print("\n" + "#" * 80)
        print(f"  MODEL: {model}")
        print("#" * 80)
        for year in args.folds:
            test_start = f"{year}-01-01"
            test_end = f"{year}-12-31"
            for seed in args.seeds:
                run([
                    sys.executable, SCRIPTS[model],
                    "--config",         args.config,
                    "--seed",           str(seed),
                    "--context",        str(args.context),
                    "--forecast",       str(args.forecast),
                    "--test-start",     test_start,
                    "--test-end",       test_end,
                    "--results-subdir", f"wfv/{args.pool}/{year}/seed_{seed}",
                ])

    print("\n" + "=" * 80)
    print("  ZS-WFV COMPLETE")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Folds:  {args.folds}")
    print(f"Seeds:  {args.seeds}")
    print("\nCS-RankIC auswerten (je Modell):")
    for model in args.models:
        print(f"  python 02_finetuning/scripts/evaluate_wfv_cs_rankic.py "
              f"--wfv-dir 01_model_comparison/results/{model}/wfv")


if __name__ == "__main__":
    main()
