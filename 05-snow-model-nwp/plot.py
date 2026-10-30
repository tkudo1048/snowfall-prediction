"""評価結果を図にする（evaluate.py の後に実行）。

fig1: 特徴量セット × 手法ごとの評価MAE（2024・2025年の冬）と基準
fig2: 大雪日（翌日20cm以上）の分類の average precision（評価データ）と、でたらめの水準
fig3: SA（運用可能）の勾配ブースティング回帰・ロジスティック回帰の、グループ単位の permutation importance
fig4: 評価データの実際の降雪量と、S0・SA・SB の勾配ブースティング回帰の予測
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import HEAVY, IMAGES, RESULTS, TARGET_JA

OUT = IMAGES  # images/<対象地点>/（--target で選ぶ。既定は弘前）
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
COL = {"S0": "#9a9a9a", "SA": "#2a6fbb", "SB": "#d9822b"}
plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
SET_LABEL = {"S0": "S0 観測のみ", "SA": "SA ＋今日の数値予報（運用可能）", "SB": "SB ＋明日の数値予報（運用不可の上限）"}


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def bars(ax, df, value, methods, names):
    x = np.arange(len(methods))
    w = 0.26
    for i, s in enumerate(["S0", "SA", "SB"]):
        v = [df[(df.set == s) & (df.method == m)][value].iloc[0] for m in methods]
        b = ax.bar(x + (i - 1) * w, v, w, color=COL[s], label=SET_LABEL[s])
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [names[m] for m in methods])


def main():
    reg = pd.read_csv(RESULTS / "metrics_reg.csv")
    clf = pd.read_csv(RESULTS / "metrics_clf.csv")
    pe = pd.read_csv(RESULTS / "predictions_eval.csv", parse_dates=["target_date"])
    gp = pd.read_csv(RESULTS / "permutation_groups.csv")

    # fig1
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars(ax, reg, "eval_mae", ["ridge", "hgb", "rf"],
         {"ridge": "Ridge回帰", "hgb": "勾配ブースティング", "rf": "ランダムフォレスト"})
    z, pz = (reg[reg.key == k].eval_mae.iloc[0] for k in ("zero", "persistence"))
    hi_lo = {"zero": "bottom" if z >= pz else "top", "persistence": "top" if z >= pz else "bottom"}  # 近い2本の文字が重ならないよう上下に分ける
    for key, ls, va in (("persistence", "--", hi_lo["persistence"]), ("zero", ":", hi_lo["zero"]),
                        ("nwp_msm_scaled", "-.", "bottom")):
        r = reg[reg.key == key].iloc[0]
        ax.axhline(r.eval_mae, color=MUTED, ls=ls, lw=1)
        ax.text(2.48, r.eval_mae, f"{r['name'].split(': ')[-1].replace('t+1日の', '明日の')} {r.eval_mae:.2f}", fontsize=7.5, color=MUTED,
                va=va, ha="left")
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylabel("評価MAE（cm、小さいほど良い）", color=MUTED)
    ax.set_title(f"翌日の{TARGET_JA}の降雪量の平均誤差（評価: 2024・2025年の冬 {len(pe)}日）", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
    clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1-eval-mae.png", dpi=150)
    plt.close(fig)

    # fig2
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars(ax, clf[clf.set.isin(["S0", "SA", "SB"])], "eval_ap", ["logreg", "hgbc"],
         {"logreg": "ロジスティック回帰", "hgbc": "勾配ブースティング分類"})
    prev = clf.eval_prevalence.iloc[0]
    ax.axhline(prev, color=MUTED, ls=":", lw=1)
    ax.text(1.45, prev, f"でたらめの水準 {prev:.3f}", fontsize=8, color=MUTED, va="bottom", ha="right")
    r = clf[clf.key == "persistence"].iloc[0]
    ax.axhline(r.eval_ap, color=MUTED, ls="--", lw=1)
    ax.text(1.45, r.eval_ap, f"持続予測 {r.eval_ap:.3f}", fontsize=8, color=MUTED, va="bottom", ha="right")
    ax.set_ylabel("average precision（大きいほど良い）", color=MUTED)
    n_heavy = int((pe.actual >= HEAVY).sum())
    ax.set_title(f"{TARGET_JA}の大雪日（翌日{HEAVY:.0f}cm以上）の当てやすさ（評価: 大雪日{n_heavy}日）", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
    clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2-eval-heavy-ap.png", dpi=150)
    plt.close(fig)

    # fig3
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, key, xl in ((axes[0], "SA_hgb", "MAEの増加（cm）"), (axes[1], "clf_SA_logreg", "APの低下")):
        g = gp[gp.model == key].sort_values("increase")
        ax.barh(g.group, g.increase, xerr=g["std"], color=[COL["SA"] if "数値予報" in s else COL["S0"] for s in g.group])
        ax.axvline(0, color=GRID)
        ax.set_xlabel(xl, color=MUTED)
        ax.set_title({"SA_hgb": "SA 勾配ブースティング回帰", "clf_SA_logreg": "SA ロジスティック回帰（大雪の分類）"}[key],
                     color=INK, fontsize=10, loc="left")
        ax.tick_params(axis="y", labelsize=8)
        clean(ax)
    fig.suptitle("並べ替えたときにどれだけ悪くなるか（評価データ、50回の平均）。青 = 数値予報、灰 = 観測", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3-group-importance-SA.png", dpi=150)
    plt.close(fig)

    # fig4
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, s in zip(axes, sorted(pe.season.unique())):
        d = pe[pe.season == s]
        ax.bar(d.target_date, d.actual, color="#cfd8e3", width=1.0, label="実際の降雪量")
        for k in ("S0", "SA", "SB"):
            ax.plot(d.target_date, d[f"{k}_hgb"], color=COL[k], lw=1.2, label=SET_LABEL[k] + "（勾配ブースティング）")
        ax.axhline(20, color=MUTED, ls=":", lw=0.8)
        ax.set_ylabel("cm", color=MUTED)
        ax.set_title(f"{s}年の冬", color=INK, fontsize=10, loc="left")
        clean(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig4-eval-timeseries.png", dpi=150)
    plt.close(fig)
    print("saved:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
