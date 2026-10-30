"""月別の平均気温と合計降雪量の関係を調べる（青森・弘前）。

- 日別データを月単位にまとめる（気温は日平均の月平均、降雪は日合計の月合計）。
- 欠損日が多い月は除く。降雪が出うる冬の月（11〜3月）に絞った結果も出す。
- 結果は 01-temp-snowfall/monthly.csv に保存する。実行: cd 01-temp-snowfall && ../.venv/bin/python analyze.py
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from preprocess import AOMORI_COLS, HIROSAKI_COLS, parse_daily

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / "data" / "raw"
WINTER = [11, 12, 1, 2, 3]


def monthly(name: str, cols: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(RAW / f"{name}_daily.csv", encoding="utf-8-sig", dtype=str)
    d = parse_daily(raw, cols)[["temp_mean", "snowfall"]]
    g = d.groupby([d.index.year.rename("year"), d.index.month.rename("month")])
    m = pd.DataFrame({
        "temp_mean": g["temp_mean"].mean(),
        "snowfall_total": g["snowfall"].sum(min_count=1),
        "n_days": g["temp_mean"].count(),
        "n_snow_days": g["snowfall"].apply(lambda s: (s > 0).sum()),
    }).reset_index()
    days = pd.to_datetime(dict(year=m.year, month=m.month, day=1)).dt.days_in_month
    m = m[m.n_days == days].copy()  # 全日そろった月だけ
    m["station"] = name
    return m


def report(m: pd.DataFrame, label: str):
    x, y = m.temp_mean, m.snowfall_total
    r, p = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    print(f"{label}: n={len(m)} pearson r={r:.2f} (p={p:.3g}) spearman={rho:.2f} (p={ps:.3g})")


if __name__ == "__main__":
    out = []
    for name, cols in [("aomori", AOMORI_COLS), ("hirosaki", HIROSAKI_COLS)]:
        m = monthly(name, cols)
        out.append(m)
        print(f"\n== {name} ({m.year.min()}-{m.month.iloc[0]} 〜 {m.year.max()}-{m.month.iloc[-1]}) ==")
        report(m, "全月")
        w = m[m.month.isin(WINTER)]
        report(w, "冬(11-3月)")
        snowy = w[w.snowfall_total > 0]
        report(snowy, "冬で降雪あり")
        print("月別(平均気温と降雪量の相関 / 月平均):")
        for mo in WINTER:
            s = w[w.month == mo]
            r = stats.pearsonr(s.temp_mean, s.snowfall_total)[0] if len(s) > 2 else float("nan")
            print(f"  {mo:>2}月: n={len(s)} r={r:+.2f} 平均気温={s.temp_mean.mean():.1f}℃ 降雪={s.snowfall_total.mean():.0f}cm")
        # 気温帯別の月降雪量
        bins = pd.cut(w.temp_mean, [-10, -2, 0, 2, 5, 8, 20])
        print("気温帯別の月降雪量(冬):")
        print(w.groupby(bins, observed=True).snowfall_total.agg(["count", "mean", "median"]).round(1).to_string())
    pd.concat(out).to_csv(HERE / "monthly.csv", index=False, encoding="utf-8-sig")
