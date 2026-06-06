"""Cross-Sectional Long-Short-Backtest zur ökonomischen Einordnung des CS-RankIC.

Pro Prognose-Origin werden die Assets nach vorhergesagter 18-Tage-Rendite
gerankt; Long Top-Tertil, Short Bottom-Tertil (gleichgewichtet, dollar-neutral).
Realisiert wird die tatsächliche 18-Tage-Rendite (Anchor -> Forecast-Ende) --
exakt dieselben anchor-basierten Log-Returns wie beim CS-RankIC. Fenster sind
nicht-überlappend (Stride = Forecast = 18) -> ca. 14 Perioden/Jahr.
Reines Post-Processing der gespeicherten Vorhersagen, kein Modell-Rerun.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "02_finetuning" / "scripts"))
from evaluate_wfv_cs_rankic import load_pool  # WFV: seed-ensemble pro Fold

PERIODS_PER_YEAR = 252 / 18  # Forecast-Horizont = 18 Handelstage

MODELS = {
    "kronos_finetuned": ("wfv",  ROOT / "02_finetuning/results/kronos_finetuned/wfv/holdout"),
    "kronos_zeroshot":  ("wfv",  ROOT / "01_model_comparison/results/kronos/wfv/holdout"),
    "chronos_zeroshot": ("wfv",  ROOT / "01_model_comparison/results/chronos/wfv/holdout"),
    "naive":            ("flat", ROOT / "04_benchmarks/results/naive/seed_13"),
    "arima":            ("flat", ROOT / "04_benchmarks/results/arima/seed_13"),
    "xgboost":          ("flat", ROOT / "04_benchmarks/results/xgboost/seed_13"),
}


def _window_end_returns(df_act: pd.DataFrame, df_pre: pd.DataFrame,
                        df_anc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduziert die täglichen kumulierten-seit-Anchor-Pfade auf je eine
    Beobachtung pro 18-Tage-Fenster (Fenster-Endpunkt = volle Horizont-Rendite)
    und indiziert sie nach (Fold-Jahr, Fenster-Ordinalzahl).

    Die dates sind tagesweise; alle 18 Tage eines Fensters teilen denselben
    Anchor. Pro Ticker wird der letzte Tag jedes Anchor-Runs als 18-Tage-Return
    behalten. Da die Fenster-Endtage über Assets versetzt liegen, erfolgt die
    Indizierung über die Fenster-Ordinalzahl innerhalb des Test-Jahres (das k-te
    18-Tage-Fenster eines Jahres ist über alle Assets näherungsweise
    zeitgleich) -> vollständige Cross-Section pro Rebalancing, nicht-überlappend."""
    ar = np.log(df_act / df_anc)
    pr = np.log(df_pre / df_anc)
    cols_a, cols_p = {}, {}
    for col in df_anc.columns:
        anc = df_anc[col].dropna()
        if anc.empty:
            continue
        run_id = (anc != anc.shift()).cumsum()            # neuer Run bei Anchor-Wechsel
        end_dates = anc.groupby(run_id).apply(lambda s: s.index[-1]).tolist()
        end_dates.sort()
        year_counter: dict[int, int] = {}
        idx, av, pv = [], [], []
        for d in end_dates:
            y = d.year
            o = year_counter.get(y, 0)
            year_counter[y] = o + 1
            idx.append((y, o))
            av.append(ar.at[d, col])
            pv.append(pr.at[d, col])
        mi = pd.MultiIndex.from_tuples(idx, names=["year", "win"])
        cols_a[col] = pd.Series(av, index=mi)
        cols_p[col] = pd.Series(pv, index=mi)
    A = pd.DataFrame(cols_a).sort_index()
    P = pd.DataFrame(cols_p).sort_index()
    return A.dropna(how="all"), P.dropna(how="all")


def returns_flat(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Window-End 18-Tage-Returns (wide: date x ticker) für Single-Split-Ergebnisse."""
    act, pre, anc = [], [], []
    for f in sorted(result_dir.glob("result_*.json")):
        rv = json.loads(f.read_text())["raw_values"]
        tic = f.stem.replace("result_", "")
        d = pd.to_datetime(rv["dates"])
        keep = ~d.duplicated()
        act.append(pd.Series(rv["actual"],    index=d, name=tic)[keep])
        pre.append(pd.Series(rv["predicted"], index=d, name=tic)[keep])
        anc.append(pd.Series(rv["anchors"],   index=d, name=tic)[keep])
    A = pd.concat(act, axis=1).sort_index()
    P = pd.concat(pre, axis=1).sort_index()
    C = pd.concat(anc, axis=1).sort_index()
    return _window_end_returns(A, P, C)


def returns_wfv(pool_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Window-End 18-Tage-Returns aus den WFV-Pools (Seed-Ensemble pro Fold)."""
    df_act, df_pre, df_anc = load_pool(pool_dir)
    return _window_end_returns(df_act, df_pre, df_anc)


def long_short(act_ret: pd.DataFrame, pre_ret: pd.DataFrame,
               q: float = 1 / 3, min_cs: int = 10) -> dict | None:
    """Tägliche dollar-neutrale Long-Short-Spread-Rendite (Top- minus Bottom-Tertil)."""
    act_ret = act_ret.dropna(how="all")
    pre_ret = pre_ret.dropna(how="all")
    idx = act_ret.index.intersection(pre_ret.index)
    spreads = []
    for t in idx:
        a, p = act_ret.loc[t], pre_ret.loc[t]
        m = a.notna() & p.notna()
        n = int(m.sum())
        if n < min_cs:
            continue
        k = max(1, int(np.floor(n * q)))
        ranked = p[m].sort_values()
        short_leg = ranked.index[:k]
        long_leg = ranked.index[-k:]
        spreads.append(float(a[long_leg].mean() - a[short_leg].mean()))
    s = pd.Series(spreads, dtype=float)
    if s.empty:
        return None
    mean, std = s.mean(), s.std(ddof=1)
    sharpe = (mean / std) * np.sqrt(PERIODS_PER_YEAR) if std > 0 else float("nan")
    return dict(
        mean_period=mean,
        ann_return=mean * PERIODS_PER_YEAR,
        sharpe=sharpe,
        pos_frac=float((s > 0).mean()),
        n_periods=int(len(s)),
    )


def main() -> None:
    rows = []
    for name, (kind, path) in MODELS.items():
        if not path.exists():
            print(f"  übersprungen (fehlt): {name}")
            continue
        ar, pr = returns_wfv(path) if kind == "wfv" else returns_flat(path)
        r = long_short(ar, pr)
        if r:
            rows.append({"model": name, "setup": kind, **r})

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    out = ROOT / "04_benchmarks/results/_comparison"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "long_short_backtest.csv", index=False)

    print("\n=== Cross-Sectional Long-Short (Top/Bottom-Tertil, 18d, gleichgewichtet, brutto) ===")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
