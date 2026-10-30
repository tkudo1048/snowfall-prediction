"""翌日の青森の日降雪量(cm)を予測するための共通部品（特徴量・分割・モデル・探索空間）。

- 1行 = 「t日」。特徴量は t日以前の値だけ、目的変数は t+1日 の青森 snowfall。
- 学習・評価に使うのは、予測対象日（t+1）が11〜3月の行だけ。
- 冬（シーズン）は予測対象日で決める: 11〜12月はその年、1〜3月は前年の冬。
  例: 2025-11-01〜2026-03-31 の予測対象日 = 2025年の冬。
- 学習 = 2021〜2024年の冬（4冬）、評価 = 2025年の冬（2025-11〜2026-03）。

リーク対策で使わないもの:
- t+1日の値すべて（翌日への気圧変化など）
- t日の「天気概況（夜）」: 気象庁の日別値では夜 = 18時〜翌日6時で、予測対象日の0〜6時を含むため。
  代わりに t-1日の夜（= t日の6時まで）を使う。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]  # リポジトリ直下
sys.path.insert(0, str(ROOT / "lib"))
from preprocess import AOMORI_COLS, HIROSAKI_COLS, parse_daily  # noqa: E402

RAW = ROOT / "data" / "raw"
RESULTS = HERE / "results"
WINTER = [11, 12, 1, 2, 3]
TRAIN_SEASONS = [2021, 2022, 2023, 2024]
EVAL_SEASON = 2025
HEAVY = 20.0  # 大雪日 = 翌日の降雪20cm以上
COLD_KNOT = 2.0  # 雪と雨の境目の目安（日平均気温2℃）。線形モデルが「暖かいとほぼ0」を表せるようにする

DIRS = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
        "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
ANGLE = {d: i * 22.5 for i, d in enumerate(DIRS)}


def _load(name, cols):
    return parse_daily(pd.read_csv(RAW / f"{name}_daily.csv", encoding="utf-8-sig", dtype=str), cols)


def _vec(dirs: pd.Series, prefix: str) -> dict:
    ang = np.deg2rad(dirs.map(ANGLE))
    return {f"{prefix}_ns": np.cos(ang), f"{prefix}_ew": np.sin(ang)}  # 北寄りほど ns+、東寄りほど ew+


def build_dataset(target: str = "aomori") -> pd.DataFrame:
    """t日の特徴量と t+1日の目的変数を並べた表（予測対象日が11〜3月の行のみ）を返す。

    target: 目的変数の地点（"aomori" = 青森（既定。このフォルダの結果はすべて青森）、"hirosaki" = 弘前）。特徴量は同じ。
    """
    if target not in ("aomori", "hirosaki"):
        raise ValueError(target)
    a, h = _load("aomori", AOMORI_COLS), _load("hirosaki", HIROSAKI_COLS)
    idx = pd.date_range(a.index.min(), a.index.max(), freq="D")
    a, h = a.reindex(idx), h.reindex(idx)
    f = {}

    # --- 青森（t日） ---
    for c in ["pressure_sea", "precip_total", "precip_max_1h", "temp_mean", "temp_max", "temp_min",
              "humidity_mean", "humidity_min", "wind_mean", "wind_max", "gust_max", "sunshine",
              "snowfall", "snow_depth"]:
        f[f"a_{c}"] = a[c]
    f.update({f"a_{k}": v for k, v in _vec(a.wind_max_dir, "wind").items()})
    f.update({f"a_{k}": v for k, v in _vec(a.gust_max_dir, "gust").items()})
    f["a_temp_range"] = a.temp_max - a.temp_min
    f["a_temp_change"] = a.temp_mean.diff()
    f["a_humidity_change"] = a.humidity_mean.diff()
    f["a_pressure_change"] = a.pressure_sea.diff()  # t-1 → t の変化（翌日への変化は使わない）
    f["a_snow_depth_change"] = a.snow_depth.diff()
    t = a.weather_day.fillna("")
    ok = a.weather_day.notna()
    f["a_day_snow"] = t.str.contains("雪|みぞれ").astype(float).where(ok)
    f["a_day_rain"] = t.str.contains("雨").astype(float).where(ok)
    f["a_day_thunder"] = t.str.contains("雷").astype(float).where(ok)
    f["a_day_clear"] = t.str.match("^(快晴|晴)").astype(float).where(ok)
    # 夜の天気は t-1日 の値（t-1日18時〜t日6時）。t日の夜は予測対象日の0〜6時を含むので使わない
    tn = a.weather_night.shift(1)
    f["a_prevnight_snow"] = tn.fillna("").str.contains("雪|みぞれ").astype(float).where(tn.notna())

    # --- 弘前（t日） ---
    for c in ["precip_total", "temp_mean", "temp_max", "temp_min", "humidity_mean", "humidity_min",
              "wind_mean", "wind_max", "sunshine", "snowfall", "snow_depth"]:
        f[f"h_{c}"] = h[c]
    f.update({f"h_{k}": v for k, v in _vec(h.wind_mode_dir, "mode").items()})
    f["h_temp_change"] = h.temp_mean.diff()

    # --- 前日・数日間の値 ---
    f["prev_a_snowfall"] = a.snowfall.shift(1)
    f["prev_a_temp_mean"] = a.temp_mean.shift(1)
    f["prev_a_humidity_mean"] = a.humidity_mean.shift(1)
    f["prev_a_sunshine"] = a.sunshine.shift(1)
    f["a_snowfall_3d"] = a.snowfall.rolling(3, min_periods=3).sum()  # t-2〜t
    f["temp_gap_a_minus_h"] = a.temp_mean - h.temp_mean

    # --- 気温の折れ線（ヒンジ）と、寒さ×湿度 ---
    f["a_cold"] = np.maximum(0.0, COLD_KNOT - a.temp_mean)
    f["h_cold"] = np.maximum(0.0, COLD_KNOT - h.temp_mean)
    f["a_cold_x_humid"] = f["a_cold"] * a.humidity_mean / 100.0

    df = pd.DataFrame(f, index=idx)
    target_date = idx + pd.Timedelta(days=1)
    doy = target_date.dayofyear
    df["season_sin"] = np.sin(2 * np.pi * doy / 365.25)  # 予測対象日の暦（事前に分かる）
    df["season_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["target"] = (a if target == "aomori" else h).snowfall.shift(-1)  # t+1日の降雪量（既定は青森）
    df["target_date"] = target_date
    df["target_month"] = target_date.month
    df["season"] = np.where(target_date.month >= 11, target_date.year, target_date.year - 1)
    df.index.name = "date"
    df = df[df.target_month.isin(WINTER) & df.target.notna()]
    return df


HINGE = ["a_cold", "h_cold", "a_cold_x_humid"]
META = ["target", "target_date", "target_month", "season"]


def feature_cols(df: pd.DataFrame, hinge: bool = True) -> list[str]:
    cols = [c for c in df.columns if c not in META]
    return cols if hinge else [c for c in cols if c not in HINGE]


def split(df: pd.DataFrame):
    tr = df[df.season.isin(TRAIN_SEASONS)]
    ev = df[df.season == EVAL_SEASON]
    return tr, ev


def cv_folds(train: pd.DataFrame):
    """冬単位の前向きCV: [2021]→2022, [2021,2022]→2023, [2021..2023]→2024。"""
    s = train.season.to_numpy()
    out = []
    for k in range(1, len(TRAIN_SEASONS)):
        tr_idx = np.where(np.isin(s, TRAIN_SEASONS[:k]))[0]
        va_idx = np.where(s == TRAIN_SEASONS[k])[0]
        out.append((tr_idx, va_idx))
    return out


# ---------------------------------------------------------------- モデル
METHODS = {
    # name: (表示名, 特徴量にヒンジを含めるか)
    "ridge": ("Ridge回帰", True),
    "ridge_nohinge": ("Ridge回帰（気温の折れ線なし）", False),
    "enet": ("ElasticNet", True),
    "rf": ("ランダムフォレスト", True),
    "hgb": ("勾配ブースティング(HistGB)", True),
    "svr": ("SVR(RBFカーネル)", True),
}


def suggest(trial, method: str) -> dict:
    """Optunaの探索空間。全手法で「目的変数をlog1p変換するか」を同じ選択肢として探索する。"""
    p = {"target_log1p": trial.suggest_categorical("target_log1p", [False, True])}
    if method in ("ridge", "ridge_nohinge"):
        p["alpha"] = trial.suggest_float("alpha", 1e-3, 1e4, log=True)
    elif method == "enet":
        p["alpha"] = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        p["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 1.0)
    elif method == "rf":
        p["n_estimators"] = trial.suggest_int("n_estimators", 100, 600, step=50)
        p["max_depth"] = trial.suggest_int("max_depth", 2, 20)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 50, log=True)
        p["max_features"] = trial.suggest_float("max_features", 0.1, 1.0)
    elif method == "hgb":
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
        p["max_iter"] = trial.suggest_int("max_iter", 50, 800, step=25)
        p["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 4, 64, log=True)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 100, log=True)
        p["l2_regularization"] = trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True)
        p["max_features"] = trial.suggest_float("max_features", 0.2, 1.0)
    elif method == "svr":
        p["C"] = trial.suggest_float("C", 1e-2, 1e3, log=True)
        p["epsilon"] = trial.suggest_float("epsilon", 1e-3, 3.0, log=True)
        p["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    return p


def make_model(method: str, params: dict, seed: int = 0, n_jobs: int = 1):
    p = dict(params)
    log1p = p.pop("target_log1p")
    imp = ("imputer", SimpleImputer(strategy="median"))
    if method in ("ridge", "ridge_nohinge"):
        est = Pipeline([imp, ("scaler", StandardScaler()), ("reg", Ridge(**p))])
    elif method == "enet":
        est = Pipeline([imp, ("scaler", StandardScaler()), ("reg", ElasticNet(max_iter=50000, **p))])
    elif method == "rf":
        est = Pipeline([imp, ("reg", RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **p))])
    elif method == "hgb":
        # 欠損はそのまま扱える。early_stopping は使わず max_iter を探索
        est = HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **p)
    elif method == "svr":
        est = Pipeline([imp, ("scaler", StandardScaler()), ("reg", SVR(kernel="rbf", **p))])
    else:
        raise ValueError(method)
    if log1p:
        est = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    return est


def predict(model, X) -> np.ndarray:
    """降雪量は負にならないので、予測は0未満を0に切る（全手法・全基準で共通）。"""
    return np.clip(model.predict(X), 0.0, None)


# ---------------------------------------------------------------- 指標
def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
