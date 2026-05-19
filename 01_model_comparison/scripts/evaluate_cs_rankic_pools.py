"""
Cross-Sectional RankIC evaluation for zero-shot results — by pool, multi-seed.

Same methodology as evaluate_results.py (per-day Spearman across the
cross-section on log-returns vs. anchor, seed-ensemble, autocorr-corrected
CI95). Difference: reads a pool-structured layout

    <root>/<pool>/seed_<seed>/{final_energy_study.json, result_<TICKER>.json}

so zero-shot CS-RankIC can be compared directly to WFV-finetuned CS-RankIC
(02_finetuning/scripts/evaluate_wfv_cs_rankic.py).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from experiments.metrics import calculate_ic_statistics


def load_pool(pool_dir: Path):
    """Return wide DataFrames (actual, pred-ensemble, anchor) per ticker, index=date."""
    per_ticker_actual: dict[str, list[pd.Series]] = {}
    per_ticker_anchor: dict[str, list[pd.Series]] = {}
    per_ticker_preds:  dict[str, list[pd.Series]] = {}

    for seed_dir in sorted(pool_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        summary = seed_dir / "final_energy_study.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text())
        for ticker in data.get("summary", {}).keys():
            res_file = seed_dir / f"result_{ticker}.json"
            if not res_file.exists():
                continue
            rv = json.loads(res_file.read_text())["raw_values"]
            dates = pd.to_datetime(rv["dates"])
            act = pd.Series(rv["actual"],    index=dates, name=ticker)
            pre = pd.Series(rv["predicted"], index=dates, name=ticker)
            anc = pd.Series(rv["anchors"],   index=dates, name=ticker) if "anchors" in rv else None
            per_ticker_actual.setdefault(ticker, []).append(act)
            if anc is not None:
                per_ticker_anchor.setdefault(ticker, []).append(anc)
            per_ticker_preds.setdefault(ticker, []).append(pre)

    if not per_ticker_preds:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    actual_cols, pred_cols, anchor_cols = [], [], []
    for ticker, preds in per_ticker_preds.items():
        # Seed-Ensemble: Mittel über Seeds pro (ticker, date)
        pred_ens = pd.concat(preds, axis=1).mean(axis=1)
        pred_ens.name = ticker
        pred_cols.append(pred_ens)
        act_full = per_ticker_actual[ticker][0]  # seed-unabhängig
        actual_cols.append(act_full)
        if ticker in per_ticker_anchor:
            anchor_cols.append(per_ticker_anchor[ticker][0])

    df_act = pd.concat(actual_cols, axis=1).sort_index()
    df_pre = pd.concat(pred_cols,   axis=1).sort_index()
    df_anc = pd.concat(anchor_cols, axis=1).sort_index() if anchor_cols else pd.DataFrame()
    return df_act, df_pre, df_anc


def evaluate_pool(pool_dir: Path, min_cross_section: int = 10):
    df_act, df_pre, df_anc = load_pool(pool_dir)
    if df_act.empty:
        return None

    if not df_anc.empty:
        df_act_ret = np.log(df_act / df_anc)
        df_pre_ret = np.log(df_pre / df_anc)
    else:
        df_act_ret = np.log(df_act / df_act.shift(1))
        df_pre_ret = np.log(df_pre / df_pre.shift(1))

    df_act_ret = df_act_ret.dropna(how="all")
    df_pre_ret = df_pre_ret.dropna(how="all")
    common = df_act_ret.index.intersection(df_pre_ret.index)
    df_act_ret = df_act_ret.loc[common]
    df_pre_ret = df_pre_ret.loc[common]

    daily_ic, daily_year = [], []
    for t in df_act_ret.index:
        a_t = df_act_ret.loc[t]
        p_t = df_pre_ret.loc[t]
        mask = a_t.notna() & p_t.notna()
        if mask.sum() < min_cross_section:
            continue
        ric, _ = spearmanr(p_t[mask], a_t[mask])
        if np.isnan(ric):
            continue
        daily_ic.append(ric)
        daily_year.append(t.year)

    if not daily_ic:
        return None

    ic_stats = calculate_ic_statistics(daily_ic, prefix="RankIC")
    mean_ic = ic_stats["RankIC_Mean"]
    ci_lo, ci_hi = ic_stats["RankIC_CI95"]
    n_total = ic_stats["RankIC_Count"]
    n_eff = ic_stats["RankIC_Effective_N"]
    std_ic = ic_stats["RankIC_Std"]

    if n_eff > 1:
        t_stat = mean_ic / (std_ic / np.sqrt(n_eff))
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=max(1, int(n_eff - 1))))
    else:
        t_stat = float("nan"); p_value = float("nan")

    df_daily = pd.DataFrame({
        "date": df_act_ret.index[: len(daily_ic)],
        "year": daily_year,
        "CS_RankIC": daily_ic,
    })
    per_year = df_daily.groupby("year")["CS_RankIC"].agg(["mean", "std", "count"]).round(4)

    return {
        "mean": mean_ic, "ci95": (ci_lo, ci_hi), "std": std_ic,
        "n": n_total, "n_eff": n_eff, "t": t_stat, "p": p_value,
        "pos_frac": sum(1 for v in daily_ic if v > 0) / n_total,
        "per_year": per_year, "daily": df_daily,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Directory containing <pool>/seed_<seed>/ subtrees")
    parser.add_argument("--min-cross-section", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"❌ {root} not found")

    for pool_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pool = pool_dir.name
        print(f"\n{'=' * 80}\nPool: {pool}\n{'=' * 80}")
        res = evaluate_pool(pool_dir, min_cross_section=args.min_cross_section)
        if res is None:
            print("  (no data)")
            continue
        print(f"  Daily CS-RankIC values:   {res['n']}")
        print(f"  Effective N (autocorr):   {res['n_eff']:.1f}")
        print(f"  Mean CS-RankIC:           {res['mean']:.4f}")
        print(f"  95% CI:                   [{res['ci95'][0]:.4f}, {res['ci95'][1]:.4f}]")
        print(f"  Std:                      {res['std']:.4f}")
        print(f"  t-stat:                   {res['t']:.2f}")
        print(f"  p-value (H0: IC=0):       {res['p']:.4f}  ({'SIGNIFICANT' if res['p'] < 0.05 else 'n.s.'})")
        print(f"  Positive IC days:         {res['pos_frac'] * 100:.1f}%")
        print(f"\n  Per-year:\n{res['per_year']}")
        res["daily"].to_csv(pool_dir / "cs_rankic_daily.csv", index=False)
        res["per_year"].to_csv(pool_dir / "cs_rankic_per_year.csv")


if __name__ == "__main__":
    main()
