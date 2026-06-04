"""Multi-Modell Final-Vergleich auf gemeinsamer Eval-Basis.

Liest results aus mehreren Modell-Verzeichnissen (jedes mit seed_X/*).
Erstellt:
    - RankIC-Tabelle (Mean +/- CI95) ueber alle Modelle
    - DM-Test gegen Referenzmodell (default: kronos_finetuned)
    - Plot der mittleren Time-Series-RankIC mit CI

WICHTIG: Vergleich ist nur sinnvoll, wenn alle Modelle mit identischem
Setup (Assets, Test-Zeitraum, context/forecast/stride) gelaufen sind.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from experiments.metrics import calculate_ic_statistics  # noqa: E402


def find_seed_dirs(base: Path) -> List[Path]:
    if not base.exists():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("seed_"))


def load_model_results(model_dir: Path) -> Dict:
    """Aggregiert alle Seeds und Assets eines Modells.

    Returns dict mit:
        rankic_values: Liste aller per-Asset-Mean-RankIC-Werte (Asset x Seed flach)
        per_asset_predictions: dict ticker -> dict mit 'actual','predicted','anchors' (vom ersten Seed)
        mae: Liste aller per-Asset-MAE
        da: Liste aller per-Asset-Directional Accuracy
    """
    seed_dirs = find_seed_dirs(model_dir)
    if not seed_dirs:
        seed_dirs = [model_dir]  # single-run layout

    rankic_vals, mae_vals, da_vals = [], [], []
    per_asset_preds: Dict[str, Dict] = {}

    for sd in seed_dirs:
        summary_path = sd / "final_energy_study.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        for ticker, metrics in summary.get("summary", {}).items():
            ric = metrics.get("RankIC_TimeSeries_Mean")
            if ric is not None and np.isfinite(ric):
                rankic_vals.append(ric)
            mae = metrics.get("MAE_indicative")
            if mae is not None and np.isfinite(mae):
                mae_vals.append(mae)
            # DA ist optional; greifen wir aus result_<ticker>.json
            res_path = sd / f"result_{ticker}.json"
            if res_path.exists():
                with open(res_path) as f:
                    res = json.load(f)
                # nur erstes Seed je Asset fuer DM-Predictions
                if ticker not in per_asset_preds and "raw_values" in res:
                    rv = res["raw_values"]
                    per_asset_preds[ticker] = {
                        "actual": np.asarray(rv["actual"], dtype=float),
                        "predicted": np.asarray(rv["predicted"], dtype=float),
                        "dates": rv.get("dates", []),
                    }
    return {
        "rankic_values": rankic_vals,
        "mae_values": mae_vals,
        "per_asset_predictions": per_asset_preds,
        "n_seeds": len(seed_dirs),
    }


def dm_test(err_ref: np.ndarray, err_alt: np.ndarray) -> Dict:
    """Diebold-Mariano-Test (Newey-West, Lag-1) auf squared-error loss differential.

    H0: gleiche Forecast-Accuracy.
    Vorzeichen mean_d > 0 bedeutet: alt-Modell hat kleinere Loss als ref (besser).
    """
    if len(err_ref) != len(err_alt) or len(err_ref) < 3:
        return {"dm": np.nan, "p_value": np.nan}
    d = err_ref ** 2 - err_alt ** 2  # > 0: alt besser als ref
    n = len(d)
    mean_d = float(np.mean(d))
    gamma_0 = float(np.var(d, ddof=1))
    gamma_1 = float(np.mean((d[:-1] - mean_d) * (d[1:] - mean_d))) if n > 1 else 0.0
    var_d = (gamma_0 + 2 * gamma_1) / n
    if var_d <= 0:
        var_d = gamma_0 / n
    dm = mean_d / np.sqrt(var_d) if var_d > 0 else np.nan
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(dm))) if np.isfinite(dm) else np.nan
    return {"dm": dm, "p_value": p, "mean_loss_diff": mean_d}


def main(models: Dict[str, str], reference: str, out_dir: str) -> None:
    rows = []
    loaded: Dict[str, Dict] = {}
    for name, path in models.items():
        print(f"Lade {name} aus {path} ...")
        loaded[name] = load_model_results(Path(path))

    ref_data = loaded.get(reference)
    if ref_data is None:
        raise ValueError(f"Referenzmodell '{reference}' nicht in models.")

    # 1. RankIC + MAE Tabelle
    for name, data in loaded.items():
        stats = calculate_ic_statistics(data["rankic_values"], prefix="RankIC")
        mae_mean = float(np.mean(data["mae_values"])) if data["mae_values"] else np.nan
        mae_std = float(np.std(data["mae_values"], ddof=1)) if len(data["mae_values"]) > 1 else np.nan
        ci_lo, ci_hi = stats.get("RankIC_CI95", (np.nan, np.nan))
        rows.append({
            "Model": name,
            "N_Assets": len(data["rankic_values"]),
            "RankIC_Mean": stats.get("RankIC_Mean", np.nan),
            "RankIC_CI95_Lower": ci_lo,
            "RankIC_CI95_Upper": ci_hi,
            "MAE_Mean": mae_mean,
            "MAE_Std": mae_std,
            "N_Seeds": data["n_seeds"],
        })

    df = pd.DataFrame(rows).sort_values("RankIC_Mean", ascending=False).reset_index(drop=True)

    # 2. DM-Test gegen Referenzmodell (gepoolte Predictions ueber Assets)
    dm_rows = []
    ref_preds = ref_data["per_asset_predictions"]
    for name, data in loaded.items():
        if name == reference:
            continue
        alt_preds = data["per_asset_predictions"]
        common = sorted(set(ref_preds) & set(alt_preds))
        if not common:
            dm_rows.append({"Model": name, "vs": reference, "dm": np.nan, "p_value": np.nan, "n_obs": 0})
            continue
        err_ref_all, err_alt_all = [], []
        for ticker in common:
            r, a = ref_preds[ticker], alt_preds[ticker]
            n = min(len(r["actual"]), len(r["predicted"]), len(a["actual"]), len(a["predicted"]))
            if n < 3:
                continue
            actual = r["actual"][:n]
            err_ref_all.append(actual - r["predicted"][:n])
            err_alt_all.append(actual - a["predicted"][:n])
        if not err_ref_all:
            dm_rows.append({"Model": name, "vs": reference, "dm": np.nan, "p_value": np.nan, "n_obs": 0})
            continue
        e_ref = np.concatenate(err_ref_all)
        e_alt = np.concatenate(err_alt_all)
        res = dm_test(e_ref, e_alt)
        dm_rows.append({
            "Model": name, "vs": reference,
            "dm": res["dm"], "p_value": res["p_value"],
            "mean_loss_diff_sq": res.get("mean_loss_diff", np.nan),
            "n_obs": len(e_ref),
            "interpretation": ("alt besser" if res.get("mean_loss_diff", 0) > 0 else "ref besser"),
        })
    dm_df = pd.DataFrame(dm_rows)

    # 3. Output
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rankic_comparison.csv", index=False)
    dm_df.to_csv(out / "dm_test_vs_reference.csv", index=False)

    print("\n=== RankIC Vergleich (sortiert) ===")
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"\n=== DM-Test (vs. {reference}; H0: gleiche Accuracy) ===")
    print(dm_df.to_string(index=False, float_format="%.4f"))

    # 4. Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        order = df["Model"].tolist()
        means = df["RankIC_Mean"].values
        lo = df["RankIC_CI95_Lower"].values
        hi = df["RankIC_CI95_Upper"].values
        yerr = np.vstack([means - lo, hi - means])
        ax.bar(order, means, yerr=yerr, capsize=4, color="steelblue", edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylabel("Mean Time-Series RankIC (95% CI)")
        ax.set_title("Final Model Comparison — RankIC")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out / "rankic_comparison.png", dpi=150)
        print(f"\nPlot: {out / 'rankic_comparison.png'}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"\nCSVs gespeichert in: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="kronos_finetuned",
                   help="Modellname, gegen den DM-Tests laufen (default: kronos_finetuned)")
    p.add_argument("--out-dir", default="04_benchmarks/results/_comparison")
    # Modell -> Results-Pfad. Defaults gehen davon aus dass run_benchmark.py + fine-tuned-Skript gelaufen sind.
    p.add_argument("--kronos-finetuned-dir", default="02_finetuning/results/kronos_finetuned")
    p.add_argument("--kronos-zeroshot-dir", default="01_model_comparison/results/kronos")
    p.add_argument("--chronos-dir", default="01_model_comparison/results/chronos")
    p.add_argument("--naive-dir", default="04_benchmarks/results/naive")
    p.add_argument("--arima-dir", default="04_benchmarks/results/arima")
    p.add_argument("--xgboost-dir", default="04_benchmarks/results/xgboost")
    a = p.parse_args()

    candidates = {
        "kronos_finetuned": a.kronos_finetuned_dir,
        "kronos_zeroshot": a.kronos_zeroshot_dir,
        "chronos": a.chronos_dir,
        "naive": a.naive_dir,
        "arima": a.arima_dir,
        "xgboost": a.xgboost_dir,
    }
    # nur die nehmen, die existieren
    models = {k: v for k, v in candidates.items() if Path(v).exists()}
    if not models:
        raise SystemExit("Keines der Modell-Verzeichnisse existiert.")
    print(f"Gefundene Modelle: {list(models)}")
    main(models, a.reference, a.out_dir)
