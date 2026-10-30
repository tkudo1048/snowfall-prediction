"""夏の平均気温と、その直後の冬（11月〜翌3月）の合計降雪量の関係（青森・弘前）。

- 入力: ../data/monthly/{aomori,hirosaki}_monthly.csv（fetch_monthly.py で取得した気象庁の月別値）
- 夏の定義を3通り試す: 6〜8月、7〜8月、8月のみ。比較用に、その冬自身の平均気温も見る。
- 冬は5か月すべて値がそろったものだけ使う（資料不足 "]" や欠測は除く）。
- 1年ずつ抜いて相関を計算し直し（leave-one-out）、1年に引っ張られていないか確かめる。
- 出力: summer_vs_winter.csv, images/fig3-summer-vs-winter.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "monthly"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
C_SUMMER, C_WINTER = "#d9822b", "#2a6fbb"
plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SUMMERS = {"summer_jja": [6, 7, 8], "summer_ja": [7, 8], "summer_aug": [8]}
NAMES = {"summer_jja": "夏(6〜8月)の平均気温", "summer_ja": "7〜8月の平均気温", "summer_aug": "8月の平均気温",
         "winter_temp": "その冬(11〜3月)の平均気温"}


def table(station):
    m = pd.read_csv(RAW / f"{station}_monthly.csv").fillna({"temp_flag": "", "snow_flag": ""})
    ok_t = m.temp_flag.isin(["", "quasi"]) & m.temp_mean.notna()
    ok_s = m.snow_flag.isin(["", "quasi", "none"]) & m.snowfall_total.notna()
    m = m[ok_t & ok_s]
    rows = []
    for y in sorted(m.year.unique()):
        s = m[m.year == y]
        w = m[((m.year == y) & m.month.isin([11, 12])) | ((m.year == y + 1) & m.month.isin([1, 2, 3]))]
        if len(w) < 5:
            continue
        row = dict(station=station, summer_year=y, winter=f"{y}-{y + 1}",
                   winter_temp=w.temp_mean.mean(), winter_snow=w.snowfall_total.sum())
        for k, months in SUMMERS.items():
            ss = s[s.month.isin(months)]
            row[k] = ss.temp_mean.mean() if len(ss) == len(months) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def corr_report(d):
    out = []
    for c in list(SUMMERS) + ["winter_temp"]:
        dd = d.dropna(subset=[c])
        r, p = stats.pearsonr(dd[c], dd.winter_snow)
        rho, ps = stats.spearmanr(dd[c], dd.winter_snow)
        loo = [stats.pearsonr(dd[c].drop(i), dd.winter_snow.drop(i))[0] for i in dd.index]
        out.append(dict(station=d.station.iloc[0], x=c, n=len(dd), pearson=r, p=p, spearman=rho, p_spearman=ps,
                        loo_min=min(loo), loo_max=max(loo)))
    return pd.DataFrame(out)


def plot(t):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), dpi=200, sharey="row")
    for i, (st, title) in enumerate([("aomori", "青森"), ("hirosaki", "弘前")]):
        d = t[t.station == st]
        for j, (c, color) in enumerate([("summer_jja", C_SUMMER), ("winter_temp", C_WINTER)]):
            ax = axes[i, j]
            dd = d.dropna(subset=[c])
            ax.scatter(dd[c], dd.winter_snow, s=60, color=color, edgecolor="white", linewidth=1.2, zorder=3)
            for _, r in dd.iterrows():
                ax.annotate(f"{r.summer_year % 100:02d}-{(r.summer_year + 1) % 100:02d}", (r[c], r.winter_snow),
                            xytext=(5, 4), textcoords="offset points", fontsize=8, color=MUTED)
            k, b = np.polyfit(dd[c], dd.winter_snow, 1)
            xs = np.linspace(dd[c].min(), dd[c].max(), 20)
            ax.plot(xs, k * xs + b, color=color, ls="--", lw=1.4, zorder=2)
            r, p = stats.pearsonr(dd[c], dd.winter_snow)
            ax.text(0.03, 0.04, f"r = {r:.2f}（p = {p:.2f}）\n{len(dd)}シーズン", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=10, color=INK)
            ax.set_title(f"{title}：{NAMES[c]}", loc="left", fontsize=11.5, color=INK, fontweight="bold")
            ax.set_xlabel(NAMES[c] + "（℃）", color=MUTED)
            ax.grid(color=GRID, lw=0.8, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=MUTED, length=0)
        axes[i, 0].set_ylabel("冬(11〜3月)の合計降雪量（cm）", color=MUTED)
    fig.suptitle("前の夏の気温と冬の降雪量には関係が見えない。その冬自身の気温の方が関係が強い（点の数字はシーズン）",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "images" / "fig3-summer-vs-winter.png", facecolor="white")


if __name__ == "__main__":
    t = pd.concat([table("aomori"), table("hirosaki")], ignore_index=True)
    t.to_csv(HERE / "summer_vs_winter.csv", index=False, encoding="utf-8-sig")
    print(t.round(2).to_string(index=False))
    rep = pd.concat([corr_report(t[t.station == s].reset_index(drop=True)) for s in ("aomori", "hirosaki")])
    print(rep.round(2).to_string(index=False))
    plot(t)
