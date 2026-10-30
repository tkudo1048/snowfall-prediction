"""日降雪量が大きく増える日（大雪日）の特徴を調べる（青森を主、弘前で確認）。

- 対象: 11〜3月で降雪のあった日。大雪日 = 日降雪量が20cm以上、ふつうの雪の日 = 1cm以上20cm未満。
- 同日の気象要素を2群で比べる（中央値、AUC、マン・ホイットニーのU検定）。
- AUC = 大雪日の値が、ふつうの雪の日の値より大きい確率（0.5で差なし、1に近いほど大雪日の方が大きい）。
- 結果は 02-heavy-snow/features.csv に保存する。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from preprocess import AOMORI_COLS, HIROSAKI_COLS, parse_daily

RAW = HERE.parent / "data" / "raw"
HEAVY_CM = 20
WESTERLY = {"西", "西南西", "西北西", "南西", "北西"}


def load(name, cols):
    return parse_daily(pd.read_csv(RAW / f"{name}_daily.csv", encoding="utf-8-sig", dtype=str), cols)


def build():
    a, h = load("aomori", AOMORI_COLS), load("hirosaki", HIROSAKI_COLS)
    d = a[["snowfall", "snow_depth", "temp_mean", "temp_max", "temp_min", "pressure_sea", "humidity_mean",
           "wind_mean", "wind_max", "gust_max", "sunshine", "precip_total", "wind_max_dir"]].copy()
    d["pressure_change"] = a.pressure_sea.diff()  # 前日からの海面気圧の変化(hPa)
    d["prev_temp_mean"] = a.temp_mean.shift(1)
    d["temp_change"] = a.temp_mean.diff()
    d["hiro_temp_mean"] = h.temp_mean
    d["west_wind"] = d.wind_max_dir.isin(WESTERLY).astype(float).where(d.wind_max_dir.notna())
    d["prev_snow"] = a.snowfall.shift(1)
    d["prev_snowy"] = (d.prev_snow > 0).astype(float).where(d.prev_snow.notna())
    d["snow_depth_prev"] = a.snow_depth.shift(1)
    d = d[d.index.month.isin([11, 12, 1, 2, 3]) & (d.snowfall > 0)].copy()
    d["group"] = np.where(d.snowfall >= HEAVY_CM, "heavy", "other")
    return d


def auc(x, y):
    """x(大雪日)の値がy(その他)より大きい確率。"""
    x, y = x.dropna(), y.dropna()
    u, p = stats.mannwhitneyu(x, y)
    return u / (len(x) * len(y)), p


FEATURES = [
    ("temp_mean", "日平均気温(℃)"), ("temp_max", "日最高気温(℃)"), ("temp_min", "日最低気温(℃)"),
    ("temp_change", "前日からの気温変化(℃)"),
    ("pressure_sea", "海面気圧(hPa)"), ("pressure_change", "前日からの気圧変化(hPa)"),
    ("humidity_mean", "平均湿度(%)"), ("wind_mean", "平均風速(m/s)"), ("wind_max", "最大風速(m/s)"),
    ("gust_max", "最大瞬間風速(m/s)"), ("sunshine", "日照時間(h)"), ("precip_total", "降水量(mm)"),
    ("west_wind", "最大風速の風向が西系の割合"), ("prev_snowy", "前日も降雪ありの割合"),
    ("snow_depth_prev", "前日の最深積雪(cm)"),
]

if __name__ == "__main__":
    d = build()
    d.to_csv(HERE / "features.csv", encoding="utf-8-sig")
    hv, ot = d[d.group == "heavy"], d[d.group == "other"]
    print(f"青森 冬の降雪日 {len(d)}日: 大雪日(>= {HEAVY_CM}cm) {len(hv)}日, その他 {len(ot)}日")
    print(f"{'指標':<28}{'大雪日':>9}{'その他':>9}{'AUC':>7}{'p':>10}")
    for c, label in FEATURES:
        A, p = auc(hv[c], ot[c])
        f = (lambda s: s.mean()) if c in ("west_wind", "prev_snowy") else (lambda s: s.median())
        print(f"{label:<{28-len(label)}}{f(hv[c]):>9.2f}{f(ot[c]):>9.2f}{A:>7.2f}{p:>10.3g}")
    print("\n風向の内訳(最大風速時、大雪日 / その他):")
    print(pd.crosstab(d.wind_max_dir, d.group).sort_values("heavy", ascending=False).head(8).to_string())
    print("\n降雪量と気温帯(日平均):")
    b = pd.cut(d.temp_mean, [-15, -4, -2, 0, 2, 5, 15])
    print(d.groupby(b, observed=True).snowfall.agg(["count", "median", "max", lambda s: (s >= HEAVY_CM).mean()]).rename(columns={"<lambda_0>": "大雪日率"}).round(2).to_string())
    print("\n前日気圧変化と大雪日率:")
    b = pd.cut(d.pressure_change, [-40, -6, -3, 0, 3, 6, 40])
    print(d.groupby(b, observed=True).snowfall.agg(["count", "median", lambda s: (s >= HEAVY_CM).mean()]).rename(columns={"<lambda_0>": "大雪日率"}).round(2).to_string())
    print("\n天気(昼)の内訳:")
    print(load("aomori", AOMORI_COLS).loc[hv.index].weather_day.str[:2].value_counts().head(6).to_string())
