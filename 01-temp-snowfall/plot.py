"""月平均気温と月合計降雪量の散布図（青森・弘前）。01-temp-snowfall/monthly.csv を読んで画像にする。"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "images"
WINTER = [11, 12, 1, 2, 3]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
# 月は12個で循環するので、色相を青→緑→黄→橙→赤→紫→藍と1周させる（12月と1月が隣り合う）
MONTH_COLORS = {1: "#1b3a8a", 2: "#2f6fc4", 3: "#5fb0e0", 4: "#4fb39a", 5: "#7cc04a", 6: "#c5cf3a",
                7: "#f2b53a", 8: "#ec7a2c", 9: "#d1443a", 10: "#a8307a", 11: "#7a3fa8", 12: "#4a4aa8"}

plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def panel(ax, m, title):
    w = m.month.isin(WINTER)
    for mo in range(1, 13):
        d = m[m.month == mo]
        ax.scatter(d.temp_mean, d.snowfall_total, s=52, color=MONTH_COLORS[mo], edgecolor="white",
                   linewidth=1.2, label=f"{mo}月", zorder=3)
    mw = m[w]
    k, b = np.polyfit(mw.temp_mean, mw.snowfall_total, 1)
    xs = np.linspace(mw.temp_mean.min(), mw.temp_mean.max(), 50)
    ax.plot(xs, k * xs + b, color=MUTED, lw=1.5, ls="--", zorder=2)
    r_all = stats.pearsonr(m.temp_mean, m.snowfall_total)[0]
    r_win = stats.pearsonr(mw.temp_mean, mw.snowfall_total)[0]
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold")
    ax.text(0.97, 0.95, f"相関係数 r\n全月 {r_all:.2f}\n冬のみ {r_win:.2f}\n（破線は冬の近似直線）",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK, linespacing=1.5)
    ax.set_xlabel("月の平均気温（℃）", color=MUTED)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


SEASON_COLORS = ["#2a6fbb", "#d9822b", "#3a9a5b", "#8a4fb0", "#c8453d"]


def season_panel(ax, m, title, seasons):
    for color, sea in zip(SEASON_COLORS, seasons):
        d = m[m.season == sea].sort_values("order")
        ax.scatter(d.temp_mean, d.snowfall_total, s=56, color=color, edgecolor="white",
                   linewidth=1.2, label=f"{sea}シーズン", zorder=3)
        for _, r in d.iterrows():
            ax.annotate(f"{int(r.month)}", (r.temp_mean, r.snowfall_total), xytext=(5, 4),
                        textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold")
    ax.set_xlabel("月の平均気温（℃）", color=MUTED)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def season_figure(df):
    w = df[df.month.isin(WINTER)].copy()
    w["season_start"] = np.where(w.month >= 11, w.year, w.year - 1)
    w["season"] = w.season_start.astype(str) + "-" + (w.season_start + 1).astype(str)
    w["order"] = (w.month - 11) % 12  # 11月→0 ... 3月→4
    # 5か月そろったシーズンだけ使う
    ok = w.groupby(["station", "season"]).month.count()
    ok = ok[ok == 5].reset_index()[["station", "season"]]
    w = w.merge(ok)
    seasons = sorted(w.season.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, dpi=200)
    for ax, (name, title) in zip(axes, [("aomori", "青森"), ("hirosaki", "弘前")]):
        season_panel(ax, w[w.station == name], title, seasons)
    axes[0].set_ylabel("月の合計降雪量（cm）", color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(seasons), frameon=False, labelcolor=INK, fontsize=10)
    fig.suptitle("シーズン別に見ても、気温が低い月ほど降雪量は多い（点の横の数字は月）",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / "fig2-by-season.png", facecolor="white")
    print(w.groupby(["station", "season"]).apply(
        lambda d: pd.Series({"n": len(d), "r": stats.pearsonr(d.temp_mean, d.snowfall_total)[0],
                             "snow_total": d.snowfall_total.sum(), "temp_mean": d.temp_mean.mean()}),
        include_groups=False).round(2).to_string())


if __name__ == "__main__":
    df = pd.read_csv(HERE / "monthly.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, dpi=200)
    for ax, (name, title) in zip(axes, [("aomori", "青森"), ("hirosaki", "弘前")]):
        panel(ax, df[df.station == name], title)
    axes[0].set_ylabel("月の合計降雪量（cm）", color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=12, frameon=False, labelcolor=INK, fontsize=10,
               handletextpad=0.2, columnspacing=1.0)
    fig.suptitle("月の平均気温が低いほど、降雪量は多い（2021年8月〜2026年8月、各61か月）",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / "fig1-temp-vs-snowfall.png", facecolor="white")
    season_figure(df)
