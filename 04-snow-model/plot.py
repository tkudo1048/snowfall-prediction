"""評価結果を図にする（evaluate.py の後に実行）。

fig1: 評価データ（2025年の冬）の実際の降雪量と予測
fig2: 手法・基準ごとの評価MAE（持続予測とのMAE差の95%信頼区間。日単位と7日ブロックの広い方）
fig3: 特徴量グループごとの permutation importance（評価データ）
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import HERE, RESULTS

OUT = HERE / "images"
OUT.mkdir(exist_ok=True)
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
BLUE, ORANGE, GRAY, GREEN = "#2a6fbb", "#d9822b", "#9a9a9a", "#3a9a5b"
plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def main():
    m = pd.read_csv(RESULTS / "metrics.csv")
    p = pd.read_csv(RESULTS / "predictions_eval.csv", parse_dates=["target_date"])
    ci = pd.read_csv(RESULTS / "bootstrap_mae_diff.csv")
    gp = pd.read_csv(RESULTS / "permutation_importance_groups.csv")
    names = dict(zip(m.key, m.name))
    models = m[~m.key.isin(["persistence", "monthly_mean", "zero"])]
    best = models.sort_values("cv_mae").key.iloc[0]

    # fig1
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(p.target_date, p.actual, color=GRID, width=1.0, label="実際の降雪量")
    ax.plot(p.target_date, p[best], color=BLUE, lw=1.6, label=f"予測: {names[best]}（CVで最良）")
    ax.plot(p.target_date, p["ridge"], color=ORANGE, lw=1.2, label="予測: Ridge回帰")
    ax.plot(p.target_date, p["persistence"], color=GRAY, lw=0.9, ls=":", label="基準: 持続予測")
    ax.axhline(20, color=MUTED, lw=0.6, ls="--")
    ax.text(p.target_date.iloc[0], 20.5, "大雪日の線（20cm）", color=MUTED, fontsize=8)
    ax.set_ylabel("翌日の日降雪量（cm）", color=INK)
    ax.set_title("評価データ（2025年11月〜2026年3月）の実際と予測（青森）", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1-eval-timeseries.png", dpi=160)
    plt.close(fig)

    # fig2: 評価MAE と、持続予測とのMAE差の区間（7日ブロック）
    order = m.sort_values("eval_mae")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(order))[::-1]
    colors = [GRAY if k in ("persistence", "monthly_mean", "zero") else BLUE for k in order.key]
    ax.barh(y, order.eval_mae, color=colors, height=0.6)
    for yi, (_, r) in zip(y, order.iterrows()):
        row = ci[(ci.A == r.key) & (ci.B == "persistence")]
        txt = f"{r.eval_mae:.2f}"
        if len(row):
            row = row.iloc[0]
            lo, hi = min(row.ci_lo_day, row.ci_lo_block7), max(row.ci_hi_day, row.ci_hi_block7)
            txt += f"（持続予測との差 {row.mae_diff:+.2f}, 95%区間 {lo:+.2f}〜{hi:+.2f}）"
        ax.text(r.eval_mae + 0.05, yi, txt, va="center", fontsize=8, color=INK)
    ax.set_yticks(y, order.name, fontsize=8)
    ax.set_xlabel("評価データのMAE（cm, 小さいほど良い）\n区間: 日単位と7日ブロックのブートストラップの広い方", fontsize=9, color=INK)
    ax.set_xlim(0, order.eval_mae.max() * 1.9)
    ax.set_title("翌日の降雪量の平均絶対誤差（2025年の冬, 151日）", color=INK, loc="left")
    clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2-eval-mae.png", dpi=160)
    plt.close(fig)

    # fig3
    piv = gp.pivot_table(index="group", columns="model", values="mae_increase")
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    yy = np.arange(len(piv))
    w = 0.2
    for i, (k, c) in enumerate(zip(["hgb", "rf", "svr", "ridge"], [BLUE, GREEN, ORANGE, GRAY])):
        if k in piv:
            ax.barh(yy + (i - 1.5) * w, piv[k], height=w, color=c, label=names[k])
    ax.set_yticks(yy, piv.index)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("その項目をばらばらに並べ替えたときのMAEの増加（cm）", color=INK)
    ax.set_title("何が翌日の降雪予測に効いているか（評価データ）", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3-group-importance.png", dpi=160)
    plt.close(fig)
    print("saved:", *sorted(x.name for x in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
