"""Erzeugt config/energy_assets_train.yaml aus filtered - holdout.

Stellt sicher, dass Train- und Holdout-Set immer disjunkt sind und nur eine
Single Source of Truth gepflegt werden muss (die Holdout-Liste).
"""

from pathlib import Path

import yaml

FILTERED = Path("config/energy_assets_filtered.yaml")
HOLDOUT = Path("config/energy_assets_holdout.yaml")
TRAIN_OUT = Path("config/energy_assets_train.yaml")


def main() -> None:
    filtered = yaml.safe_load(FILTERED.read_text(encoding="utf-8"))["energy_assets"]
    holdout = yaml.safe_load(HOLDOUT.read_text(encoding="utf-8"))["holdout_assets"]

    holdout_symbols = {a["symbol"] for a in holdout}
    train = [a for a in filtered if a["symbol"] not in holdout_symbols]

    TRAIN_OUT.write_text(
        yaml.safe_dump({"energy_assets": train}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Filtered: {len(filtered)}  Holdout: {len(holdout)}  Train: {len(train)}")
    print(f"Geschrieben: {TRAIN_OUT}")


if __name__ == "__main__":
    main()
