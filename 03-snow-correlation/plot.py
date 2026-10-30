"""相関の結果を図にする。
fig1: 変数ごとの相関（全体・月内・気温の影響を除いた後）、fig2: 主な変数どうしの相関、fig3: 主な変数の値帯ごとの降雪。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import build, label

HERE = Path(__file__).resolve().parent
OUT = HERE / "images"
OUT.mkdir(exist_ok=True)
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
BLUE, ORANGE, GRAY = "#2a6fbb", "#d9822b", "#9a9a9a"

plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def fig1(r, top=22):
    c = r[r.kind == "原因候補"].head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 8.2), dpi=200)
    y = np.arange(len(c))
    ax.hlines(y, c.ci_lo, c.ci_hi, color=GRAY, lw=3, alpha=0.45, zorder=1)
    ax.scatter(c.spearman, y, s=70, color=INK, zorder=3, label="全体（冬の全日）")
    ax.scatter(c.spearman_within_month, y, s=55, marker="s", color=BLUE, zorder=3, label="月内（月ごとの平均を引いた後）")
    ax.scatter(c.partial_after_temp, y, s=60, marker="D", color=ORANGE, zorder=3, label="気温の影響を除いた後")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(c.label, color=INK, fontsize=10)
    ax.set_xlim(-0.9, 0.9)
    ax.set_xlabel("スピアマンの順位相関（青森の日降雪量との相関。灰色の太線は全体の95%信頼区間）", color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    clean(ax)
    ax.legend(loc="upper right", frameon=False, labelcolor=INK, fontsize=10)
    ax.set_title(f"青森の日降雪量と相関が強い項目（原因の候補、上位{top}、冬の{int(c.n.max())}日）",
                 loc="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1-correlation-ranking.png", facecolor="white")


def fig2(df):
    cols = ["a_snowfall", "a_temp_mean", "a_temp_range", "a_humidity_mean", "a_sunshine", "a_wind_mean",
            "a_pressure_sea", "a_wind_ew", "prev_a_snowfall", "prev_a_temp_mean", "h_temp_mean", "a_precip_total"]
    names = ["降雪量", "日平均気温", "気温の日較差", "平均湿度", "日照時間", "平均風速", "海面気圧",
             "風向(東寄り+)", "前日の降雪量", "前日の平均気温", "弘前の平均気温", "降水量"]
    m = df[cols].corr(method="spearman").to_numpy()
    fig, ax = plt.subplots(figsize=(8.6, 7.4), dpi=200)
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(names, rotation=45, ha="right", color=INK, fontsize=9.5)
    ax.set_yticklabels(names, color=INK, fontsize=9.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(m[i, j]) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    ax.set_title("主な項目どうしの相関（スピアマン、青森・冬）", loc="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2-heatmap.png", facecolor="white")


def fig3(df):
    items = [("a_temp_mean", "日平均気温（℃）"), ("a_humidity_mean", "平均湿度（%）"),
             ("a_sunshine", "日照時間（時間）"), ("a_wind_mean", "平均風速（m/s）")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.6), dpi=200)
    for ax, (c, lab) in zip(axes, items):
        d = df[[c, "a_snowfall"]].dropna()
        d["bin"] = pd.qcut(d[c], 8, duplicates="drop")
        g = d.groupby("bin", observed=True)
        x = g[c].median()
        ax.bar(range(len(x)), g.a_snowfall.mean(), color=BLUE, alpha=0.85, width=0.7, zorder=2)
        ax.plot(range(len(x)), g.a_snowfall.apply(lambda s: (s > 0).mean()) * 40, color=ORANGE, lw=2, marker="o", ms=5, zorder=3)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([f"{v:.1f}" if abs(v) < 20 else f"{v:.0f}" for v in x], color=MUTED, fontsize=8)
        ax.set_title(lab, loc="left", fontsize=11, color=INK)
        ax.set_ylim(0, 41)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        clean(ax)
        ax.set_xlabel("値の低い順に8等分した各帯の中央値", color=MUTED, fontsize=8)
    axes[0].set_ylabel("青色: 日降雪量の平均(cm)", color=BLUE)
    sec = axes[-1].secondary_yaxis("right", functions=(lambda v: v / 40 * 100, lambda v: v * 40 / 100))
    sec.set_ylabel("橙の線: 降雪あり日の割合(%)", color=ORANGE)
    sec.tick_params(colors=MUTED, length=0)
    sec.spines["right"].set_visible(False)
    fig.suptitle("値の帯ごとの降雪量と降雪日率（青森・冬の全日）", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig3-binned.png", facecolor="white")


if __name__ == "__main__":
    df = build()
    r = pd.read_csv(HERE / "corr_aomori.csv")
    fig1(r)
    fig2(df)
    fig3(df)
