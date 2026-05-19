"""
Cross-Sectional RankIC evaluation for WFV results.

Replicates the exact methodology of 01_model_comparison/scripts/evaluate_results.py
(per-day Spearman rank correlation across assets on log-returns vs. anchor,
seed-ensemble of predictions, CI95 with autocorrelation correction) so the
fine-tuned WFV numbers are directly comparable to the zero-shot baseline.

Layout consumed:
  02_finetuning/results/kronos_finetuned/wfv/<pool>/<year>/seed_<seed>/
      ├── final_energy_study.json
      └── result_<TICKER>.json   (raw_values: actual, predicted, dates, anchors)

Per pool, all years are concatenated (no date overlap across folds) and the
CS-RankIC is computed daily.
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


def load_pool(pool_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (actual_prices, ensemble_pred_prices, anchors) wide DataFrames
    indexed by date, columns = tickers. Predictions are seed-averaged within
    each fold (year)."""
    # ticker -> list of (date -> price) series (one per seed, accumulated across years)
    per_ticker_actual: dict[str, list[pd.Series]] = {}
    per_ticker_anchor: dict[str, list[pd.Series]] = {}
    # ticker -> { (year): [pred_series_per_seed, ...] }  for fold-level ensembling
    per_ticker_pred_by_year: dict[str, dict[int, list[pd.Series]]] = {}

    for year_dir in sorted(pool_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        for seed_dir in sorted(year_dir.iterdir()):
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

                # Actuals + anchors are data-driven, identical across seeds → collect once per fold.
                per_ticker_actual.setdefault(ticker, []).append(act)
                if anc is not None:
                    per_ticker_anchor.setdefault(ticker, []).append(anc)
                per_ticker_pred_by_year.setdefault(ticker, {}).setdefault(year, []).append(pre)

    if not per_ticker_pred_by_year:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    actual_cols, pred_cols, anchor_cols = [], [], []
    for ticker, preds_by_year in per_ticker_pred_by_year.items():
        # Seed ensemble within each fold, then concat across non-overlapping folds.
        fold_means = [pd.concat(seed_list, axis=1).mean(axis=1) for seed_list in preds_by_year.values()]
        pred_full = pd.concat(fold_means).sort_index()
        pred_full = pred_full[~pred_full.index.duplicated(keep="first")]
        pred_full.name = ticker
        pred_cols.append(pred_full)

        act_full = pd.concat(per_ticker_actual[ticker]).sort_index()
        act_full = act_full[~act_full.index.duplicated(keep="first")]
        act_full.name = ticker
        actual_cols.append(act_full)

        if ticker in per_ticker_anchor:
            anc_full = pd.concat(per_ticker_anchor[ticker]).sort_index()
            anc_full = anc_full[~anc_full.index.duplicated(keep="first")]
            anc_full.name = ticker
            anchor_cols.append(anc_full)

    df_act = pd.concat(actual_cols, axis=1).sort_index()
    df_pre = pd.concat(pred_cols,   axis=1).sort_index()
    df_anc = pd.concat(anchor_cols, axis=1).sort_index() if anchor_cols else pd.DataFrame()
    return df_act, df_pre, df_anc


def evaluate_pool(pool_dir: Path, min_cross_section: int = 10) -> dict | None:
    df_act, df_pre, df_anc = load_pool(pool_dir)
    if df_act.empty:
        print(f"  ⚠️  No data in {pool_dir}")
        return None

    if not df_anc.empty:
        df_act_ret = np.log(df_act / df_anc)
        df_pre_ret = np.log(df_pre / df_anc)
    else:
        # Fallback: day-over-day log-returns (less accurate within multi-day horizons)
        df_act_ret = np.log(df_act / df_act.shift(1))
        df_pre_ret = np.log(df_pre / df_pre.shift(1))

    df_act_ret = df_act_ret.dropna(how="all")
    df_pre_ret = df_pre_ret.dropna(how="all")
    common_idx = df_act_ret.index.intersection(df_pre_ret.index)
    df_act_ret = df_act_ret.loc[common_idx]
    df_pre_ret = df_pre_ret.loc[common_idx]

    daily_ic = []
    daily_year = []
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
        t_stat = float("nan")
        p_value = float("nan")

    pos = sum(1 for v in daily_ic if v > 0)

    df_daily = pd.DataFrame({"date": df_act_ret.index[: len(daily_ic)], "year": daily_year, "CS_RankIC": daily_ic})
    per_year = df_daily.groupby("year")["CS_RankIC"].agg(["mean", "std", "count"]).round(4)

    return {
        "mean":   mean_ic,
        "ci95":   (ci_lo, ci_hi),
        "std":    std_ic,
        "n":      n_total,
        "n_eff":  n_eff,
        "t":      t_stat,
        "p":      p_value,
        "pos_frac": pos / n_total,
        "per_year": per_year,
        "daily":  df_daily,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wfv-dir", type=str, default="02_finetuning/results/kronos_finetuned/wfv")
    parser.add_argument("--min-cross-section", type=int, default=10)
    args = parser.parse_args()

    wfv_dir = Path(args.wfv_dir)
    if not wfv_dir.exists():
        raise SystemExit(f"❌ {wfv_dir} not found")

    for pool_dir in sorted(p for p in wfv_dir.iterdir() if p.is_dir()):
        pool = pool_dir.name
        print(f"\n{'=' * 80}\nPool: {pool}\n{'=' * 80}")
        res = evaluate_pool(pool_dir, min_cross_section=args.min_cross_section)
        if res is None:
            continue
        print(f"  Daily CS-RankIC values:   {res['n']}")
        print(f"  Effective N (autocorr):   {res['n_eff']:.1f}")
        print(f"  Mean CS-RankIC:           {res['mean']:.4f}")
        print(f"  95% CI:                   [{res['ci95'][0]:.4f}, {res['ci95'][1]:.4f}]")
        print(f"  Std:                      {res['std']:.4f}")
        print(f"  t-stat:                   {res['t']:.2f}")
        print(f"  p-value (H0: IC=0):       {res['p']:.4f}  ({'SIGNIFICANT' if res['p'] < 0.05 else 'n.s.'})")
        print(f"  Positive IC days:         {res['pos_frac'] * 100:.1f}%")
        print(f"\n  Per-year breakdown (diagnostic):")
        print(res["per_year"].to_string())

        out_daily = pool_dir / "cs_rankic_daily.csv"
        res["daily"].to_csv(out_daily, index=False)
        res["per_year"].to_csv(pool_dir / "cs_rankic_per_year.csv")
        print(f"\n  Saved: {out_daily}")


if __name__ == "__main__":
    main()
