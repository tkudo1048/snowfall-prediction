"""大雪日の特徴を図にする。fig1: 指標ごとの差の大きさ(AUC)、fig2: 主な指標の分布。"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze import FEATURES, HEAVY_CM, auc, build

HERE = Path(__file__).resolve().parent
OUT = HERE / "images"
OUT.mkdir(exist_ok=True)
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
C_HEAVY, C_OTHER = "#2a6fbb", "#9a9a9a"
HI, LO = "#d9822b", "#2a6fbb"  # 大雪日の方が大きい / 小さい

plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def fig1(d):
    hv, ot = d[d.group == "heavy"], d[d.group == "other"]
    rows = sorted(((label, auc(hv[c], ot[c])[0], auc(hv[c], ot[c])[1]) for c, label in FEATURES),
                  key=lambda r: abs(r[1] - 0.5))
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=200)
    for i, (label, a, p) in enumerate(rows):
        color = HI if a > 0.5 else LO
        ax.plot([0.5, a], [i, i], color=color, lw=2.5, alpha=0.5, zorder=2)
        ax.scatter([a], [i], s=70, color=color, edgecolor="white", linewidth=1.2, zorder=3,
                   marker="o" if p < 0.05 else "D")
        ax.text(a + (0.015 if a > 0.5 else -0.015), i, f"{a:.2f}", ha="left" if a > 0.5 else "right",
                va="center", fontsize=9, color=INK)
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=10)
    ax.set_xlim(0.1, 1.08)
    ax.set_xlabel("AUC（0.5で差なし。右は大雪日の方が大きい／左は小さい）", color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    clean(ax)
    ax.set_title(f"大雪日（降雪{HEAVY_CM}cm以上）とふつうの雪の日で、差が大きい指標（青森、冬の降雪日{len(d)}日）",
                 loc="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.01, "● 差が偶然とは考えにくい（p<0.05）　◆ 偶然の範囲（p≧0.05）　※降水量は降雪とほぼ同じ現象を測っているため、原因ではなく結果", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig1-feature-auc.png", facecolor="white")


def fig2(d):
    items = [("temp_mean", "日平均気温（℃）"), ("humidity_mean", "平均湿度（%）"),
             ("sunshine", "日照時間（時間）"), ("wind_max", "最大風速（m/s）")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.6), dpi=200)
    rng = np.random.default_rng(0)
    for ax, (c, label) in zip(axes, items):
        for x, (g, color, name) in enumerate([("other", C_OTHER, "ふつうの\n雪の日"), ("heavy", C_HEAVY, "大雪日")]):
            v = d[d.group == g][c].dropna()
            ax.scatter(x + rng.uniform(-0.25, 0.25, len(v)), v, s=14, color=color, alpha=0.55, linewidth=0, zorder=2)
            ax.hlines(v.median(), x - 0.34, x + 0.34, color=INK, lw=2, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ふつうの\n雪の日", "大雪日"], color=INK, fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(label, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        clean(ax)
    fig.suptitle("大雪日は気温が低く、湿度が高く、日が差さない。風の強さは差がない（横線は中央値）",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig2-distributions.png", facecolor="white")


if __name__ == "__main__":
    d = build()
    fig1(d)
    fig2(d)
