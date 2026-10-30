"""日降雪量と、取得している全データ（青森・弘前の日別）との相関を詳しく調べる。

- 対象: 冬（11〜3月）の全日。目的変数は青森の日降雪量（確認として弘前の日降雪量でも同じ分析）。
- 説明変数: 青森・弘前の日別の全項目、前日の値、前日との差、風向の成分、天気の文字列から作った印。
  ※10分値(amedas_10min)は2026年9月下旬以降しかなく冬のデータがないため使えない。
- 指標:
  - ピアソン（直線の相関）、スピアマン（順位の相関、直線でなくてもよい）、相互情報量（曲線や山形も拾う）
  - スピアマンの信頼区間: 7日ずつのブロックでデータを復元抽出（日が連続していて独立ではないため）
  - 月内の相関: 月ごとの平均を引いてから計算（「冬の真ん中は寒くて雪が多い」という季節の影響を除く）
  - 降雪日のみの相関: 雪が降った日だけで計算（降る/降らないの差を除き、「降った日の量」の違いを見る）
  - 降る/降らない: 降雪あり(1cm以上)の日と降雪なしの日で値を比べたAUC（0.5で差なし）
  - 気温の影響を除いた相関: 青森の日平均気温で説明できる分（順位ベースの回帰残差）を引いてから計算。
    気温以外に、湿度・日照・風などが独立に効いているかを見る
  - 多重検定: 変数が多いので Benjamini-Hochberg 法で補正した q 値も出す
- 結果は 03-snow-correlation/corr_aomori.csv, corr_hirosaki.csv に保存する。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from preprocess import AOMORI_COLS, HIROSAKI_COLS, parse_daily

RAW = HERE.parent / "data" / "raw"
WINTER = [11, 12, 1, 2, 3]
DIRS = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
        "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
ANGLE = {d: i * 22.5 for i, d in enumerate(DIRS)}  # 風が吹いてくる方角（北=0, 東=90）

LABEL = {
    "pressure_local": "現地気圧", "pressure_sea": "海面気圧", "precip_total": "降水量", "precip_max_1h": "1時間降水量の最大",
    "precip_max_10m": "10分降水量の最大", "temp_mean": "日平均気温", "temp_max": "日最高気温", "temp_min": "日最低気温",
    "humidity_mean": "平均湿度", "humidity_min": "最小湿度", "wind_mean": "平均風速", "wind_max": "最大風速",
    "gust_max": "最大瞬間風速", "sunshine": "日照時間", "snowfall": "降雪量", "snow_depth": "最深積雪",
    "temp_range": "気温の日較差", "temp_change": "前日からの平均気温の変化", "humidity_change": "前日からの湿度の変化",
    "pressure_change": "前日からの気圧変化", "pressure_change_next": "翌日への気圧変化",
    "snow_depth_change": "最深積雪の前日差", "wind_ns": "最大風速の風向(北寄りほど+)", "wind_ew": "最大風速の風向(東寄りほど+)",
    "gust_ns": "瞬間風速の風向(北寄りほど+)", "gust_ew": "瞬間風速の風向(東寄りほど+)",
    "mode_ns": "最多風向(北寄りほど+)", "mode_ew": "最多風向(東寄りほど+)",
    "day_snow": "天気(昼)に雪を含む", "night_snow": "天気(夜)に雪を含む", "day_rain": "天気(昼)に雨を含む",
    "day_thunder": "天気(昼)に雷を含む", "day_clear": "天気(昼)が晴で始まる", "day_cloud": "天気(昼)が曇で始まる",
}
# 降雪の「結果」として決まる項目。原因の候補とは分けて扱う
OUTCOME = {"precip_total", "precip_max_1h", "precip_max_10m", "snow_depth", "snow_depth_change",
           "snowfall", "day_snow", "night_snow", "day_rain"}
STATION = {"a": "青森", "h": "弘前"}


def load(name, cols):
    return parse_daily(pd.read_csv(RAW / f"{name}_daily.csv", encoding="utf-8-sig", dtype=str), cols)


def vec(dirs, prefix):
    ang = np.deg2rad(dirs.map(ANGLE))
    return {f"{prefix}_ns": np.cos(ang), f"{prefix}_ew": np.sin(ang)}


def build() -> pd.DataFrame:
    a, h = load("aomori", AOMORI_COLS), load("hirosaki", HIROSAKI_COLS)
    out = {}
    for p, d in (("a", a), ("h", h)):
        for c in d.columns:
            if c in ("weather_day", "weather_night", "wind_max_dir", "gust_max_dir", "wind_mode_dir"):
                continue
            out[f"{p}_{c}"] = d[c]
        out[f"{p}_temp_range"] = d.temp_max - d.temp_min
        out[f"{p}_temp_change"] = d.temp_mean.diff()
        out[f"{p}_humidity_change"] = d.humidity_mean.diff()
        out[f"{p}_snow_depth_change"] = d.snow_depth.diff()
        for k, v in vec(d.wind_max_dir, "wind").items():
            out[f"{p}_{k}"] = v
        for k, v in vec(d.gust_max_dir, "gust").items():
            out[f"{p}_{k}"] = v
        if "wind_mode_dir" in d:
            for k, v in vec(d.wind_mode_dir, "mode").items():
                out[f"{p}_{k}"] = v
    out["a_pressure_change"] = a.pressure_sea.diff()
    out["a_pressure_change_next"] = a.pressure_sea.shift(-1) - a.pressure_sea
    for col, key in (("weather_day", "day"), ("weather_night", "night")):
        t = a[col].fillna("")
        valid = a[col].notna()
        out[f"a_{key}_snow"] = t.str.contains("雪|みぞれ").astype(float).where(valid)
        if key == "day":
            out["a_day_rain"] = t.str.contains("雨").astype(float).where(valid)
            out["a_day_thunder"] = t.str.contains("雷").astype(float).where(valid)
            out["a_day_clear"] = t.str.match("^(快晴|晴)").astype(float).where(valid)
            out["a_day_cloud"] = t.str.match("^(薄曇|曇)").astype(float).where(valid)
    df = pd.DataFrame(out)
    # 前日の値（その日の雪の「前触れ」になる項目があるかを見る）
    for c in ["a_temp_mean", "a_humidity_mean", "a_pressure_sea", "a_sunshine", "a_wind_mean", "a_snowfall", "a_precip_total",
              "h_temp_mean", "h_humidity_mean", "h_sunshine", "h_wind_mean", "h_snowfall"]:
        df[f"prev_{c}"] = df[c].shift(1)
    df["temp_gap_a_minus_h"] = df.a_temp_mean - df.h_temp_mean
    return df[df.index.month.isin(WINTER)]


def label(col: str) -> str:
    if col.startswith("prev_"):
        return "前日の" + label(col[5:])
    if col == "temp_gap_a_minus_h":
        return "気温差(青森−弘前)"
    p, base = col.split("_", 1)
    return f"{STATION[p]} {LABEL.get(base, base)}"


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg法のq値。"""
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def block_ci(x: pd.Series, y: pd.Series, n_boot=500, block=7, seed=0):
    ok = x.notna() & y.notna()
    xv, yv = x[ok].to_numpy(), y[ok].to_numpy()
    n = len(xv)
    if n < 30 or np.std(xv) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    rs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        if np.std(xv[idx]) == 0:
            continue
        rs.append(stats.spearmanr(xv[idx], yv[idx])[0])
    return tuple(np.percentile(rs, [2.5, 97.5]))


def analyze(df: pd.DataFrame, target: str) -> pd.DataFrame:
    y = df[target]
    ym = y - y.groupby(df.index.month).transform("mean")
    snowy = y > 0
    cols = [c for c in df.columns if c != target]
    rows = []
    for c in cols:
        x = df[c]
        ok = x.notna() & y.notna()
        if ok.sum() < 30 or x[ok].std() == 0:
            continue
        pr, pp = stats.pearsonr(x[ok], y[ok])
        sr, sp = stats.spearmanr(x[ok], y[ok])
        xm = x - x.groupby(df.index.month).transform("mean")
        ok2 = xm.notna() & ym.notna()
        sr_m = stats.spearmanr(xm[ok2], ym[ok2])[0]
        ok3 = ok & snowy
        sr_s = stats.spearmanr(x[ok3], y[ok3])[0] if ok3.sum() > 30 and x[ok3].std() > 0 else np.nan
        yr, xr = y[ok].rank(), x[ok].rank()
        tr = df.loc[ok, "a_temp_mean"].rank()
        if c in ("a_temp_mean",):
            part = np.nan
        else:
            ry = yr - np.polyval(np.polyfit(tr, yr, 1), tr)
            rx = xr - np.polyval(np.polyfit(tr, xr, 1), tr)
            part = stats.pearsonr(rx, ry)[0] if rx.std() > 0 else np.nan
        pos, neg = x[ok & snowy], x[ok & ~snowy]
        occ = stats.mannwhitneyu(pos, neg)[0] / (len(pos) * len(neg)) if len(pos) > 5 and len(neg) > 5 else np.nan
        mi = mutual_info_regression(x[ok].to_frame(), y[ok], random_state=0)[0]
        lo, hi = block_ci(x, y)
        base = c.split("_", 1)[1] if c[:2] in ("a_", "h_") else c
        rows.append(dict(variable=c, label=label(c), kind="結果" if base in OUTCOME and not c.startswith("prev_") else "原因候補",
                         n=int(ok.sum()), pearson=pr, spearman=sr, ci_lo=lo, ci_hi=hi, p=sp,
                         spearman_within_month=sr_m, spearman_snowdays_only=sr_s, occurrence_auc=occ,
                         partial_after_temp=part, mutual_info=mi))
    r = pd.DataFrame(rows)
    r["q"] = bh(r.p.to_numpy())
    return r.sort_values("spearman", key=abs, ascending=False)


if __name__ == "__main__":
    df = build()
    print(f"冬の日数 {len(df)}日（{df.index.min().date()}〜{df.index.max().date()}）、変数 {df.shape[1]}個")
    for target, name in (("a_snowfall", "aomori"), ("h_snowfall", "hirosaki")):
        r = analyze(df, target)
        r.to_csv(HERE / f"corr_{name}.csv", index=False, encoding="utf-8-sig")
        print(f"\n=== 目的変数: {label(target)} 日降雪量（n={r.n.max()}）  上位25（スピアマン絶対値順）===")
        pd.set_option("display.width", 250)
        show = r.head(25)[["label", "kind", "pearson", "spearman", "ci_lo", "ci_hi", "spearman_within_month",
                           "spearman_snowdays_only", "occurrence_auc", "partial_after_temp", "mutual_info", "q"]]
        print(show.round(2).to_string(index=False))
