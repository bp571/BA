"""
Phase 3: Walk-Forward Validation with annual LoRA refit.

For each test year Y in 2021..2025:
  1. Prepare training data covering 2010 .. (Y-1).12.31 (expanding window)
  2. Fine-tune LoRA on that data (1000 steps)
  3. Evaluate fold's adapter on test window [Y.01.01, Y.12.31] for both
     asset pools: 70 train-assets (zeitliches OOS) and 34 holdout-assets
     (Asset-Generalisierbarkeit)

Design decisions (see HANDOFF.md "WFV / Refit"):
  - Expanding train window
  - Annual refit (5 folds total)
  - Single seed per fold (Multi-Seed = future work)
  - Hyperparameter from Phase 2.2 optimum: r=4, alpha=16, dropout=0.20,
    lr=5e-5, use_ffn=1; data params from Phase 2.1: c=80, f=18
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FOLDS = [2021, 2022, 2023, 2024, 2025]
POOLS = {
    "train":   "config/energy_assets_train.yaml",
    "holdout": "config/energy_assets_holdout.yaml",
}


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (exit {res.returncode})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=DEFAULT_FOLDS,
                        help="Test years to evaluate (one fold per year)")
    parser.add_argument("--train-config", type=str,
                        default="config/energy_assets_train.yaml",
                        help="Asset config for LoRA training (70 train assets)")
    parser.add_argument("--holdout-config", type=str,
                        default="config/energy_assets_holdout.yaml",
                        help="Asset config to mask out from training data")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    # Phase 2.2 LoRA optimum
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--use-ffn", type=int, default=1)
    # Phase 2.1 data optimum
    parser.add_argument("--context", type=int, default=80)
    parser.add_argument("--forecast", type=int, default=18)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip fine-tuning (reuse existing adapters)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation (only refit adapters)")
    args = parser.parse_args()

    for year in args.folds:
        print("\n" + "=" * 80)
        print(f"  FOLD {year}")
        print("=" * 80)

        train_end   = f"{year - 1}-12-31"
        test_start  = f"{year}-01-01"
        test_end    = f"{year}-12-31"
        max_data    = test_end  # never see future data during this fold

        train_arrow = f"data/processed/wfv/train_data_kronos_{year}.arrow"
        val_arrow   = f"data/processed/wfv/val_data_kronos_{year}.arrow"
        adapter_dir = f"models/wfv/fold_{year}"

        if not args.skip_train:
            run([
                sys.executable, "02_finetuning/training/prepare_data_kronos.py",
                "--config", args.train_config,
                "--holdout-config-path", args.holdout_config,
                "--train-end", train_end,
                "--val-start", f"{year - 1}-01-01",   # last year of train used as val sliver
                "--val-end",   train_end,
                "--max-data-date", max_data,
                "--train-output", train_arrow,
                "--val-output",   val_arrow,
            ])

            run([
                sys.executable, "02_finetuning/training/train_kronos_lora.py",
                "--data-path",     train_arrow,
                "--output-dir",    adapter_dir,
                "--lora-r",        str(args.lora_r),
                "--lora-alpha",    str(args.lora_alpha),
                "--lora-dropout",  str(args.lora_dropout),
                "--learning-rate", str(args.learning_rate),
                "--use-ffn",       str(args.use_ffn),
                "--max-steps",     str(args.max_steps),
                "--seed",          str(args.seed),
            ])

        if args.skip_eval:
            continue

        for pool_name, pool_config in POOLS.items():
            run([
                sys.executable, "02_finetuning/evaluation/main_kronos_finetuned.py",
                "--config",         pool_config,
                "--adapter-path",   f"{adapter_dir}/final",
                "--seed",           str(args.seed),
                "--context",        str(args.context),
                "--forecast",       str(args.forecast),
                "--test-start",     test_start,
                "--test-end",       test_end,
                "--results-subdir", f"wfv/{pool_name}/{year}",
            ])

    print("\n" + "=" * 80)
    print("  WFV COMPLETE")
    print("=" * 80)
    print(f"Folds: {args.folds}")
    print(f"Adapters: models/wfv/fold_<year>/final")
    print(f"Results:  02_finetuning/results/kronos_finetuned/wfv/<pool>/<year>")


if __name__ == "__main__":
    main()
