"""数値予報モデル（GSM・MSM）の値を特徴量に加えて、翌日の日降雪量(cm)を予測するための共通部品。

目的変数の地点（対象地点）は切り替えられる: 弘前（既定）または青森。
- コマンドラインの `--target hirosaki|aomori`（どのスクリプトでも可）か、環境変数 SNOW_TARGET で選ぶ。
- 結果・図・ログは地点ごとに results/<地点>/、images/<地点>/、logs/<地点>/ に分ける。

- 1行 = 「t日」。目的変数は t+1日 の対象地点の snowfall（気象庁）。予測対象日（t+1）が11〜3月の行だけを使う。
- 冬（シーズン）は予測対象日で決める: 11〜12月はその年、1〜3月は前年の冬（例: 2025-11〜2026-03 = 2025年の冬）。
- 学習 = 2015年の冬（2016-01-02〜03-31のみ）〜2023年の冬（〜2024-03）、評価 = 2024年・2025年の冬。

特徴量セット:
- S0（基準）: 前回の snow-model と同じ観測の特徴量（青森・弘前の t日以前の日別値）。
  ただし弘前の湿度（h_humidity_mean, h_humidity_min）は 2021-03-11 からしか観測がないため除いた。
- SA（運用可能）: S0 ＋ t日の数値予報の値（t日 21時JST までの値だけ。GSMは補間されていない03・09・15・21時の値だけ）。
  test_no_leak.py で、t日21時より後の生データを壊しても SA の特徴量が変わらないことを確かめている。
- SB（上限の目安・運用不可）: SA ＋ t+1日の数値予報の値。「明日の予報が実況並みに当たった場合」の上限。

数値予報の値の注意: Open-Meteo Historical Forecast API の値は、各予報回の「最初の数時間」をつないだもの。
ほぼ実況（解析値に近い値）で、前日に出た予報ではない。SB の t+1日の値は、実際の「前日の予報」より当たりやすい。
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]  # リポジトリ直下
RAW_LONG = HERE.parent / "data" / "long"  # data/long
OM_DIR = HERE.parent / "data" / "openmeteo"  # data/openmeteo


def _pop_target_arg() -> str:
    """`--target 地点` をコマンドラインから取り出して消す（各スクリプトの引数の解析に影響させないため）。"""
    t = os.environ.get("SNOW_TARGET", "hirosaki")
    if "--target" in sys.argv:
        i = sys.argv.index("--target")
        t = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        os.environ["SNOW_TARGET"] = t  # 子プロセスにも同じ地点を渡す
    if t not in ("hirosaki", "aomori"):
        raise ValueError(f"--target は hirosaki か aomori: {t}")
    return t


TARGET = _pop_target_arg()
TARGET_JA = {"hirosaki": "弘前", "aomori": "青森"}[TARGET]
RESULTS = HERE / "results" / TARGET
IMAGES = HERE / "images" / TARGET
LOGS = HERE / "logs" / TARGET
# 持続予測（翌日も今日と同じ）に使う、対象地点の今日の降雪量の列
PERSIST_COL = {"hirosaki": "h_snowfall", "aomori": "a_snowfall"}[TARGET]

# 前回の 04-snow-model/common.py を読み込んで再利用する（特徴量づくり・回帰モデル・探索空間・指標）
_spec = importlib.util.spec_from_file_location("snow_base", HERE.parent / "04-snow-model" / "common.py")
base = importlib.util.module_from_spec(_spec)
_dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # 04-snow-model/ に __pycache__ を作らない
_spec.loader.exec_module(base)
sys.dont_write_bytecode = _dwb
HEAVY = base.HEAVY  # 大雪日 = 翌日の降雪20cm以上（主の閾値。地点によらず同じ）
# 参考の閾値（弘前のみ）: 弘前の学習データ（2015〜2023年の冬の予測対象日）の降雪量の95パーセンタイル = 15cm。
# 弘前は20cm以上の日が学習データに21日しかないため、参考として出す。主は20cm（評価結果を見る前に決めた）。
HEAVY_REF = {"hirosaki": 15.0, "aomori": None}[TARGET]
mae, rmse, r2, predict = base.mae, base.rmse, base.r2, base.predict

TRAIN_SEASONS = list(range(2015, 2024))  # 2015年の冬は2016-01〜03だけ（GSMの提供開始が2016-01-01のため）
EVAL_SEASONS = [2024, 2025]
CV_VAL_SEASONS = [2019, 2020, 2021, 2022, 2023]  # 前向きCV: 各冬を、それより前の冬すべてで学習して予測
DROP_OBS = ["h_humidity_mean", "h_humidity_min"]  # 2021-03-11より前は観測なし
T850_KNOT = -6.0  # 850hPaの気温 −6℃（平地の雪の目安）。線形モデルが「寒いほど雪」を折れ線で表せるようにする


# ---------------------------------------------------------------- 数値予報（毎時値の読み込み）
# GSM の毎時値のうち、モデル本来の値は 03・09・15・21時JST（00・06・12・18 UTC）の6時間ごとだけ。
# それ以外の時刻は Open-Meteo が前後の本来の値から補間（Catmull-Rom）したもので、たとえば t日16〜20時の値には
# t+1日03時の値が混ざる（check_gsm_native_hours.py で確認）。そこで GSM の瞬間値（気温・湿度・風・高度・気圧など）は
# 本来の4時刻だけを残し、ほかの時刻は欠損にしてから使う。
# GSM の降水量・降雪量は「03〜09時」「09〜15時」「15〜21時」「21〜翌03時」の6時間の合計を6等分した値なので、
# 6時間のかたまりの終わり（03・09・15・21時）までの範囲だけを足す。MSM は毎時が本来の値（補間なし）。
GSM_NATIVE_HOURS = (3, 9, 15, 21)
ACCUM = ("precipitation", "snowfall")


def read_raw(point: str, model: str) -> pd.DataFrame:
    """保存済みのAPI応答を結合した毎時値（時刻はJST、APIの表記のまま。降水量・降雪量は「その時刻までの1時間」）。"""
    frames = []
    for p in sorted(OM_DIR.glob(f"{point}_{model}_*.json")):
        d = json.loads(p.read_text())
        h = pd.DataFrame(d["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        frames.append(h.set_index("time"))
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.columns = [c.replace(f"_{model}", "") for c in df.columns]
    df = df.astype(float)
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))  # 冬と冬の間の空白も毎時の行にする


def prepare(raw: pd.DataFrame, model: str) -> pd.DataFrame:
    """特徴量づくり用に整える。

    - GSM: 瞬間値は 03・09・15・21時だけ残す（補間された時刻は欠損にする）。
    - 降水量・降雪量は1時間前にずらして「その時刻から1時間」の値にする（21:00 の値 = 20〜21時 → 20:00 の行へ）。
      これで「日付Dの h時の行」= D日 h〜h+1時 になる。
    - 850hPaの風を東西・南北の成分に分ける。
    """
    df = raw.copy()
    if model == "jma_gsm":
        inst = [c for c in df.columns if c not in ACCUM]
        df.loc[~df.index.hour.isin(GSM_NATIVE_HOURS), inst] = np.nan
    for c in ACCUM:
        if c in df:
            df[c] = df[c].shift(-1)
    if "wind_speed_850hPa" in df:
        ang = np.deg2rad(df["wind_direction_850hPa"])
        ws = df["wind_speed_850hPa"] / 3.6  # km/h → m/s
        df["u850"] = -ws * np.sin(ang)  # 西風ほど +（風が「吹いていく」向きの東西成分）
        df["v850"] = -ws * np.cos(ang)  # 南風ほど +（北風ほど −）
        df["ws850"] = ws
    return df


def load_hourly(point: str, model: str) -> pd.DataFrame:
    return prepare(read_raw(point, model), model)


def _day_agg(h: pd.DataFrame, col: str, how: str, hours) -> pd.Series:
    """その日の指定した時刻の値を集計する。指定した時刻が1つでも欠けていれば欠損にする。"""
    hours = list(hours)
    s = h[col][h.index.hour.isin(hours)]
    g = s.groupby(s.index.normalize())
    out = g.sum() if how == "sum" else getattr(g, how)()
    return out.where(g.count() == len(hours))


def _at(h: pd.DataFrame, col: str, hour: int) -> pd.Series:
    s = h[col][h.index.hour == hour]
    s.index = s.index.normalize()
    return s


def nwp_day_features(prefix: str, gsm: pd.DataFrame, msm: pd.DataFrame, upstream: dict, until21: bool) -> pd.DataFrame:
    """日付ごとの数値予報の特徴量。

    GSM（どちらの場合も同じ）: 瞬間値は その日の 03・09・15・21時 の4回の値（平均・最低）と21時の値。
      降水量・降雪量は その日の 0〜21時 の合計（= 6時間のかたまり「前日21〜03時」の後半＋03〜09＋09〜15＋15〜21時）。
    MSM: until21=True（SA用、t日）は 0〜21時の値、until21=False（SB用、t+1日）は 0〜24時の値。
    → SA（t日）は、t日21時より後の生データを1つも使わない。SB（t+1日）は、GSMはt+1日21時まで、MSMはt+1日24時まで。
    """
    g_inst = GSM_NATIVE_HOURS
    g_acc = range(0, 21)
    m_inst = range(0, 22) if until21 else range(0, 24)
    m_acc = range(0, 21) if until21 else range(0, 24)
    f = {}
    # --- 対象地点（弘前または青森）付近の上空（GSM） ---
    for c, short in [("temperature_850hPa", "t850"), ("relative_humidity_850hPa", "rh850"),
                     ("relative_humidity_700hPa", "rh700"), ("temperature_500hPa", "t500"),
                     ("geopotential_height_500hPa", "z500"), ("u850", "u850"), ("v850", "v850"), ("ws850", "ws850")]:
        f[f"{short}_mean"] = _day_agg(gsm, c, "mean", g_inst)
        if until21:
            f[f"{short}_21"] = _at(gsm, c, 21)
    f["t850_min"] = _day_agg(gsm, "temperature_850hPa", "min", g_inst)
    stab = gsm["temperature_850hPa"] - gsm["temperature_500hPa"]  # 850hPaと500hPaの気温差（大きいほど対流が起きやすい）
    gsm = gsm.assign(stab=stab)
    f["stab_mean"] = _day_agg(gsm, "stab", "mean", g_inst)
    if until21:
        f["stab_21"] = _at(gsm, "stab", 21)
        f["t850_chg"] = _at(gsm, "temperature_850hPa", 21) - _at(gsm, "temperature_850hPa", 9)  # t日中の変化
        f["z500_chg"] = _at(gsm, "geopotential_height_500hPa", 21) - _at(gsm, "geopotential_height_500hPa", 9)
        t850_ref = f["t850_21"]
        rh_ref = f["rh850_21"]
    else:
        t850_ref = f["t850_mean"]
        rh_ref = f["rh850_mean"]
    f["t850_cold"] = np.maximum(0.0, T850_KNOT - t850_ref)
    f["t850_cold_x_rh"] = f["t850_cold"] * rh_ref / 100.0
    # --- 対象地点付近の地上（MSM: 5km格子・毎時、GSM: 0.5度格子・6時間ごと） ---
    f["msm_t2m_mean"] = _day_agg(msm, "temperature_2m", "mean", m_inst)
    f["msm_precip_sum"] = _day_agg(msm, "precipitation", "sum", m_acc)
    f["msm_snow_sum"] = _day_agg(msm, "snowfall", "sum", m_acc)
    f["gsm_precip_sum"] = _day_agg(gsm, "precipitation", "sum", g_acc)
    f["gsm_snow_sum"] = _day_agg(gsm, "snowfall", "sum", g_acc)
    f["msm_pmsl_mean"] = _day_agg(msm, "pressure_msl", "mean", m_inst)
    if until21:
        f["msm_t2m_21"] = _at(msm, "temperature_2m", 21)
        f["msm_pmsl_21"] = _at(msm, "pressure_msl", 21)
        f["msm_pmsl_chg"] = f["msm_pmsl_21"] - _at(msm, "pressure_msl", 9)
        f["t2m_minus_t850_21"] = f["msm_t2m_21"] - f["t850_21"]
    # --- 上流の日本海（GSMの上空、GSM・MSMの降水量） ---
    for up, (ug, um) in upstream.items():
        for c, short in [("temperature_850hPa", "t850"), ("relative_humidity_850hPa", "rh850"),
                         ("relative_humidity_700hPa", "rh700"), ("u850", "u850"), ("v850", "v850")]:
            f[f"{up}_{short}"] = _at(ug, c, 21) if until21 else _day_agg(ug, c, "mean", g_inst)
        f[f"{up}_gsm_precip_sum"] = _day_agg(ug, "precipitation", "sum", g_acc)
        f[f"{up}_msm_precip_sum"] = _day_agg(um, "precipitation", "sum", m_acc)
    df = pd.DataFrame(f)
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df


SITE = TARGET  # 対象地点の数値予報（GSMは最寄りの0.5度格子点、MSMは地点の座標付近）
NWP_SOURCES = [(SITE, "jma_gsm"), (SITE, "jma_msm"), ("up_w", "jma_gsm"), ("up_w", "jma_msm"),
               ("up_nw", "jma_gsm"), ("up_nw", "jma_msm")]


def build_nwp(raws: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(t日の特徴量 nt_*, t+1日の値 n1_*) を、どちらも「日付 = t日」の index で返す。

    raws: {(地点, モデル): read_raw の結果}。テストで生データを書き換えて渡すため。省略時はファイルから読む。
    """
    raws = raws or {k: read_raw(*k) for k in NWP_SOURCES}
    h = {k: prepare(v, k[1]) for k, v in raws.items()}
    ups = {u: (h[(u, "jma_gsm")], h[(u, "jma_msm")]) for u in ("up_w", "up_nw")}
    nt = nwp_day_features("nt", h[(SITE, "jma_gsm")], h[(SITE, "jma_msm")], ups, until21=True)
    n1 = nwp_day_features("n1", h[(SITE, "jma_gsm")], h[(SITE, "jma_msm")], ups, until21=False)
    n1.index = n1.index - pd.Timedelta(days=1)  # t+1日の値を t日の行に置く
    return nt, n1


# ---------------------------------------------------------------- データセット
def hirosaki_persistence(daily: dict | None = None) -> pd.DataFrame:
    """弘前を対象にするときに足す「対象地点の持続」の特徴量（すべて t日以前の弘前の値）。

    04 の特徴量には、弘前の今日の降雪量・積雪（h_snowfall, h_snow_depth）はあるが、前日差・前日・3日合計は青森の分しかないため。
    daily: テストで日別データを書き換えて渡すため。省略時はファイルから読む。
    """
    h = daily["hirosaki"] if daily else base._load("hirosaki", base.HIROSAKI_COLS)
    idx = pd.date_range(h.index.min(), h.index.max(), freq="D")
    h = h.reindex(idx)
    f = pd.DataFrame(index=idx)
    f["h_snow_depth_change"] = h.snow_depth.diff()  # t-1 → t
    f["prev_h_snowfall"] = h.snowfall.shift(1)
    f["h_snowfall_3d"] = h.snowfall.rolling(3, min_periods=3).sum()  # t-2〜t
    f["prev_h_temp_mean"] = h.temp_mean.shift(1)
    return f


def build_dataset() -> pd.DataFrame:
    # 前回の特徴量づくり（04-snow-model/common.py の build_dataset）を、2016年からの日別データ（data/long）に向ける
    base.RAW = RAW_LONG
    df = base.build_dataset(target=TARGET).drop(columns=DROP_OBS)
    if TARGET == "hirosaki":
        df = df.join(hirosaki_persistence(), how="left")
    nt, n1 = build_nwp()
    df = df.join(nt, how="left").join(n1, how="left")
    df = df[df.season.isin(TRAIN_SEASONS + EVAL_SEASONS)]
    return df


META = ["target", "target_date", "target_month", "season"]
SETS = {
    "S0": "S0 観測のみ（基準）",
    "SA": "SA 観測＋t日の数値予報（運用可能）",
    "SB": "SB SA＋t+1日の数値予報（上限の目安・運用不可）",
}


def feature_cols(df: pd.DataFrame, fset: str) -> list[str]:
    cols = [c for c in df.columns if c not in META]
    obs = [c for c in cols if not c.startswith(("nt_", "n1_"))]
    if fset == "S0":
        return obs
    if fset == "SA":
        return obs + [c for c in cols if c.startswith("nt_")]
    if fset == "SB":
        return obs + [c for c in cols if c.startswith(("nt_", "n1_"))]
    raise ValueError(fset)


def split(df: pd.DataFrame):
    return df[df.season.isin(TRAIN_SEASONS)], df[df.season.isin(EVAL_SEASONS)]


def cv_folds(train: pd.DataFrame):
    s = train.season.to_numpy()
    return [(np.where(s < v)[0], np.where(s == v)[0]) for v in CV_VAL_SEASONS]


# ---------------------------------------------------------------- 回帰（前回と同じ探索空間・モデル）
REG_METHODS = {"ridge": "Ridge回帰", "hgb": "勾配ブースティング(HistGB)", "rf": "ランダムフォレスト"}
suggest_reg = base.suggest
make_reg = base.make_model

# ---------------------------------------------------------------- 大雪の分類
CLF_METHODS = {"logreg": "ロジスティック回帰", "hgbc": "勾配ブースティング分類(HistGB)"}


def suggest_clf(trial, method: str) -> dict:
    """クラス重み（大雪日1日を、ふつうの日の何日分として扱うか）も探索する。"""
    p = {"pos_weight": trial.suggest_float("pos_weight", 1.0, 50.0, log=True)}
    if method == "logreg":
        p["C"] = trial.suggest_float("C", 1e-4, 1e2, log=True)
    elif method == "hgbc":
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
        p["max_iter"] = trial.suggest_int("max_iter", 50, 800, step=25)
        p["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 4, 64, log=True)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 100, log=True)
        p["l2_regularization"] = trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True)
        p["max_features"] = trial.suggest_float("max_features", 0.2, 1.0)
    return p


def make_clf(method: str, params: dict, seed: int = 0):
    p = dict(params)
    cw = {0: 1.0, 1: p.pop("pos_weight")}
    if method == "logreg":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=p["C"], class_weight=cw, max_iter=5000))])
    if method == "hgbc":
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, class_weight=cw, **p)
    raise ValueError(method)


def proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def heavy_label(y, thr: float | None = None) -> np.ndarray:
    return (np.asarray(y) >= (HEAVY if thr is None else thr)).astype(int)


CLF_TASKS = {"clf": HEAVY}  # 分類のタスク名 → 大雪の閾値
if HEAVY_REF is not None:
    CLF_TASKS["clf_ref"] = HEAVY_REF


# ---------------------------------------------------------------- CV予測（チューニング・閾値決めで共通）
def cv_predict(task: str, method: str, params: dict, X: pd.DataFrame, y: pd.Series, folds, seed: int, n_jobs: int = 1):
    """前向きCVで、各検証冬の予測を返す（[(y_true, pred), ...] 冬ごと）。回帰は降雪量(cm)、分類は大雪の確率。"""
    out = []
    for tr_idx, va_idx in folds:
        if task == "reg":
            m = make_reg(method, params, seed=seed, n_jobs=n_jobs)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            out.append((y.iloc[va_idx].to_numpy(), predict(m, X.iloc[va_idx])))
        else:
            thr = CLF_TASKS[task]
            m = make_clf(method, params, seed=seed)
            m.fit(X.iloc[tr_idx], heavy_label(y.iloc[tr_idx], thr))
            out.append((heavy_label(y.iloc[va_idx], thr), proba(m, X.iloc[va_idx])))
    return out
