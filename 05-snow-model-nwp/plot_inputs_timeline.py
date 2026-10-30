"""記事用: 3つの入力（S0・SA・SB）を時間軸で示す図。"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / "images" / "fig1-three-inputs.png"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
GRAY, BLUE, ORANGE = "#9aa1aa", "#2a6fbb", "#d9822b"
plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Sans GB", "Arial Unicode MS"]

# 横軸: 今日 0〜24時 = 0〜24、明日 0〜24時 = 24〜48
fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
rows = [  # (y, ラベル, 区間, 色, 斜線)
    (3, "S0  観測だけ", [(-8, 24)], GRAY, False),
    (2, "SA  S0 ＋ 今日の数値予報", [(-8, 24)], GRAY, False),
    (1, "SB  SA ＋ 明日の数値予報（運用不可）", [(-8, 24)], GRAY, False),
]
H = 0.55
# 観測（全セット共通）
for y, label, _, _, _ in rows:
    ax.add_patch(Rectangle((-8, y - H / 2), 32, H, color=GRAY, alpha=0.55, lw=0))
ax.text(-7.6, 3, "今日までの観測（青森・弘前）", va="center", fontsize=9.5, color=INK)
ax.text(-7.6, 2, "今日までの観測", va="center", fontsize=9.5, color=INK)
ax.text(-7.6, 1, "今日までの観測", va="center", fontsize=9.5, color=INK)
# 今日の数値予報（SA・SB）0〜21時
for y in (2, 1):
    ax.add_patch(Rectangle((0, y - H * 0.3), 21, H * 0.6, color=BLUE, lw=0))
ax.text(10.5, 2, "今日の数値予報（0〜21時）", va="center", ha="center", fontsize=8.5, color="white")
ax.text(10.5, 1, "今日の数値予報（0〜21時）", va="center", ha="center", fontsize=8.5, color="white")
# 明日の数値予報（SB）
ax.add_patch(Rectangle((24, 1 - H / 2), 24, H, facecolor="#fbeedf", edgecolor=ORANGE, lw=1.5, hatch="///"))
ax.text(36, 1, "明日の数値予報（0〜24時）", va="center", ha="center", fontsize=9.5, color="#a85a12", fontweight="bold")
# 予測する時点・目的変数
ax.plot([21, 21], [0.6, 4.1], color=INK, lw=1.4, ls="--")
ax.text(20.4, 4.45, "予測する時点（今日の21時）", ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold")
ax.add_patch(Rectangle((24, 0.6), 24, 3.5, facecolor="none", edgecolor=BLUE, lw=1.2, ls=":"))
ax.text(36, 4.45, "当てる日（明日）の弘前の降雪量（cm）", ha="center", va="center", fontsize=10.5, color=BLUE, fontweight="bold")
ax.axvline(24, color=GRID, lw=1)
for y, label, *_ in rows:
    ax.text(-8.6, y, label, ha="right", va="center", fontsize=11, color=INK, fontweight="bold")
ax.set_xlim(-8.2, 48.5)
ax.set_ylim(0.1, 4.8)
ax.set_xticks([0, 6, 12, 18, 21, 24, 30, 36, 42, 48])
ax.set_xticklabels(["0時", "6", "12", "18", "21", "24時\n0時", "6", "12", "18", "24時"], color=MUTED, fontsize=9)
ax.set_yticks([])
ax.text(10.5, 0.28, "今日", ha="center", fontsize=10, color=MUTED)
ax.text(36, 0.28, "明日", ha="center", fontsize=10, color=MUTED)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(length=0)
fig.subplots_adjust(left=0.30, right=0.98, top=0.97, bottom=0.12)
fig.savefig(OUT, facecolor="white")
print(OUT)
