"""降水量と降雪量の関係（青森・弘前、2014〜2026年の月別値）。summer.py の降水量版。

気象庁の「降水量」は雨と雪（とけた水の量）を合わせた値で、雨だけの量はない。そのため
- 夏の降水量はほぼ雨の量として扱える
- 冬の降水量は雪を含むので、降雪量と関係が強くなるのは当然。そこで「降水量のうちどれだけが雪として積もったか」
  （降雪量cm ÷ 降水量mm）を気温と比べる

分析:
1. 夏（6〜8月、7〜8月、8月、9〜10月）の降水量 → その後の冬（11〜3月）の合計降雪量（12シーズン）
2. その冬自身の降水量 → その冬の合計降雪量（12シーズン）
3. 冬の各月: 月降水量と月降雪量、気温と「降雪量÷降水量」
出力: rain_vs_winter.csv, images/fig4-rain-vs-snow.png
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

PERIODS = {"precip_jja": [6, 7, 8], "precip_ja": [7, 8], "precip_aug": [8], "precip_so": [9, 10]}
NAMES = {"precip_jja": "夏(6〜8月)の降水量", "precip_ja": "7〜8月の降水量", "precip_aug": "8月の降水量",
         "precip_so": "9〜10月の降水量", "winter_precip": "その冬(11〜3月)の降水量"}
WINTER = [11, 12, 1, 2, 3]


def load(station):
    m = pd.read_csv(RAW / f"{station}_monthly.csv").fillna({"temp_flag": "", "snow_flag": "", "precip_flag": ""})
    ok = (m.temp_flag.isin(["", "quasi"]) & m.snow_flag.isin(["", "quasi", "none"]) & m.precip_flag.isin(["", "quasi"])
          & m[["temp_mean", "snowfall_total", "precip_total"]].notna().all(axis=1))
    return m[ok].copy()


def table(station):
    m = load(station)
    rows = []
    for y in sorted(m.year.unique()):
        s = m[m.year == y]
        w = m[((m.year == y) & m.month.isin([11, 12])) | ((m.year == y + 1) & m.month.isin([1, 2, 3]))]
        if len(w) < 5:
            continue
        row = dict(station=station, summer_year=y, winter=f"{y}-{y + 1}", winter_precip=w.precip_total.sum(),
                   winter_temp=w.temp_mean.mean(), winter_snow=w.snowfall_total.sum())
        for k, months in PERIODS.items():
            ss = s[s.month.isin(months)]
            row[k] = ss.precip_total.sum() if len(ss) == len(months) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def corr_report(d):
    out = []
    for c in list(PERIODS) + ["winter_precip"]:
        dd = d.dropna(subset=[c]).reset_index(drop=True)
        r, p = stats.pearsonr(dd[c], dd.winter_snow)
        rho, ps = stats.spearmanr(dd[c], dd.winter_snow)
        loo = [stats.pearsonr(dd[c].drop(i), dd.winter_snow.drop(i))[0] for i in dd.index]
        out.append(dict(station=d.station.iloc[0], x=c, n=len(dd), pearson=r, p=p, spearman=rho, p_spearman=ps,
                        loo_min=min(loo), loo_max=max(loo)))
    return pd.DataFrame(out)


def monthly_report(station):
    m = load(station)
    w = m[m.month.isin(WINTER) & (m.year * 100 + m.month >= 201411)].copy()
    w["snow_per_mm"] = w.snowfall_total / w.precip_total
    r1 = stats.spearmanr(w.precip_total, w.snowfall_total)
    r2 = stats.spearmanr(w.temp_mean, w.snow_per_mm)
    # 気温の影響を除いた後の降水量と降雪量（順位の回帰残差）
    rk = w[["temp_mean", "precip_total", "snowfall_total"]].rank()
    res = lambda c: rk[c] - np.polyval(np.polyfit(rk.temp_mean, rk[c], 1), rk.temp_mean)
    r3 = stats.pearsonr(res("precip_total"), res("snowfall_total"))
    print(f"{station} 冬の各月 n={len(w)}: 降水量と降雪量 rho={r1[0]:.2f}(p={r1[1]:.2g})  "
          f"気温と降雪量÷降水量 rho={r2[0]:.2f}(p={r2[1]:.2g})  気温を除いた降水量と降雪量 r={r3[0]:.2f}(p={r3[1]:.2g})")
    b = pd.cut(w.temp_mean, [-5, -1, 0, 1, 2, 4, 6, 12])
    print(w.groupby(b, observed=True).agg(months=("snow_per_mm", "size"), precip=("precip_total", "mean"),
                                          snow=("snowfall_total", "mean"), snow_per_mm=("snow_per_mm", "median")).round(2).to_string())
    return w


def style(ax):
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def plot(t, months):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), dpi=200)
    for i, (st, title) in enumerate([("aomori", "青森"), ("hirosaki", "弘前")]):
        d = t[t.station == st]
        for j, (c, color) in enumerate([("precip_jja", C_SUMMER), ("winter_precip", C_WINTER)]):
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
            ax.set_title(f"{title}：{NAMES[c]}\nr = {r:.2f}（p = {p:.2f}、{len(dd)}シーズン）", loc="left",
                         fontsize=11, color=INK, fontweight="bold")
            ax.set_xlabel(NAMES[c] + "（mm）", color=MUTED)
            style(ax)
        axes[i, 0].set_ylabel("冬(11〜3月)の合計降雪量（cm）", color=MUTED)
        ax = axes[i, 2]
        w = months[st]
        sc = ax.scatter(w.temp_mean, w.snow_per_mm, s=40, c=w.precip_total, cmap="Blues", vmin=0, vmax=w.precip_total.max(),
                        edgecolor="#555555", linewidth=0.5, zorder=3)
        rho = stats.spearmanr(w.temp_mean, w.snow_per_mm)[0]
        ax.text(0.97, 0.95, f"順位相関 = {rho:.2f}\n冬の{len(w)}か月", transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=INK)
        ax.set_title(f"{title}：気温と「降雪量÷降水量」（冬の各月）", loc="left", fontsize=11.5, color=INK, fontweight="bold")
        ax.set_xlabel("月の平均気温（℃）", color=MUTED)
        ax.set_ylabel("降雪量(cm) ÷ 降水量(mm)", color=MUTED)
        style(ax)
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label("月降水量（mm）", color=MUTED)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, length=0)
    fig.suptitle("前の夏の降水量と冬の降雪量には関係が見えない。冬の降雪量は、その冬の降水量と気温（雪になる割合）の影響が大きい",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "images" / "fig4-rain-vs-snow.png", facecolor="white")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t = pd.concat([table("aomori"), table("hirosaki")], ignore_index=True)
    t.to_csv(HERE / "rain_vs_winter.csv", index=False, encoding="utf-8-sig")
    print(t.round(1).to_string(index=False))
    print(pd.concat([corr_report(t[t.station == s]) for s in ("aomori", "hirosaki")]).round(2).to_string(index=False))
    months = {s: monthly_report(s) for s in ("aomori", "hirosaki")}
    # その冬の降水量と気温の両方を使った重回帰（シーズン単位）
    for s in ("aomori", "hirosaki"):
        d = t[t.station == s]
        X = np.column_stack([np.ones(len(d)), d.winter_precip, d.winter_temp])
        beta, *_ = np.linalg.lstsq(X, d.winter_snow, rcond=None)
        pred = X @ beta
        r2 = 1 - ((d.winter_snow - pred) ** 2).sum() / ((d.winter_snow - d.winter_snow.mean()) ** 2).sum()
        print(f"{s} シーズン降雪量 ≈ {beta[0]:.0f} + {beta[1]:.2f}×降水量 + {beta[2]:.0f}×気温  R²={r2:.2f}")
    plot(t, months)
