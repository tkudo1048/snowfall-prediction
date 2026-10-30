"""数値予報の特徴量に「未来の値」が混ざっていないことを確かめるテスト。

生データ（API応答を結合したもの、read_raw の結果）の一部をでたらめな値・欠損に書き換え、特徴量を作り直して比べる。

1. SA（t日の数値予報 nt_*）: t日21時より後の生データを全地点・全モデルで書き換えても、t日の行の nt_* は1セルも変わらない。
2. GSM の補間された時刻（03・09・15・21時以外）の瞬間値を、全期間で書き換えても、nt_* と n1_* は1セルも変わらない。
   （= t日16〜20時など、翌日03時の値が混ざった補間値を使っていない）
3. SB（t+1日の数値予報 n1_*）: GSM は t+1日21時より後、MSM は t+1日24時より後の生データを書き換えても、t日の行の n1_* は変わらない。
4. 確認の確認: t日21時の GSM の値や、t日20〜21時の MSM の降水量を書き換えると、nt_* は変わる（テストが変化を見逃さないこと）。
5. 修正前のやり方（GSMを0〜21時の毎時で平均・最低）に戻すと、2. のテストで10列が変わる（= 修正前は補間値を使っていた）。
6. 観測の特徴量（S0）: t日より後の日別データ（青森・弘前の全項目）と、t日の夜の天気概況（t日18時〜t+1日6時）を書き換えても、
   t日の行の S0 の特徴量は1セルも変わらない（対象地点の持続の特徴量、弘前を対象にしたときに足す列も含む）。

対象地点は `--target hirosaki|aomori` で選ぶ（既定は弘前）。数値予報の地点も対象地点に合わせて検査する。

実行: ../.venv/bin/python test_no_leak.py   （すべて通れば最後に「OK」と表示）
"""
import numpy as np
import pandas as pd

import common
from common import (ACCUM, DROP_OBS, GSM_NATIVE_HOURS, NWP_SOURCES, RAW_LONG, SITE, TARGET, base, build_dataset, build_nwp,
                    feature_cols, hirosaki_persistence, read_raw)


def obs_features(daily: dict) -> pd.DataFrame:
    """日別データ（書き換えたもの）から、観測の特徴量（S0）を作り直す。common.build_dataset と同じ手順。"""
    orig = base._load
    base._load = lambda name, cols: daily[name]
    try:
        base.RAW = RAW_LONG
        df = base.build_dataset(target=TARGET).drop(columns=DROP_OBS)
        if TARGET == "hirosaki":
            df = df.join(hirosaki_persistence(daily), how="left")
    finally:
        base._load = orig
    return df

N_DATES = 25


def corrupt(df: pd.DataFrame, mask, rng, mode: str) -> pd.DataFrame:
    out = df.copy()
    if mode == "nan":
        out.loc[mask, :] = np.nan
    else:  # 本来ありえない大きな値に置き換える（平均・合計・最低のどれに使われても値が変わる）
        out.loc[mask, :] = rng.uniform(-500, 5000, size=(int(np.sum(mask)), out.shape[1]))
    return out


def same(a: pd.Series, b: pd.Series) -> bool:
    return bool(((a == b) | (a.isna() & b.isna())).all())


def main():
    rng = np.random.default_rng(0)
    raws = {k: read_raw(*k) for k in NWP_SOURCES}
    nt0, n10 = build_nwp(raws)
    rows = build_dataset().index  # 学習・評価に使う t日
    dates = pd.DatetimeIndex(sorted(set(rng.choice(rows, N_DATES, replace=False)) | {rows.min(), rows.max()}))
    n_checked = {"1": 0, "2": 0, "3": 0, "4": 0}

    # --- 1. SA は t日21時より後を使わない ---
    for i, t in enumerate(dates):
        cut = t + pd.Timedelta(hours=21)
        mode = "random" if i % 2 == 0 else "nan"
        bad = {k: corrupt(v, v.index > cut, rng, mode) for k, v in raws.items()}
        nt, _ = build_nwp(bad)
        assert same(nt.loc[t], nt0.loc[t]), f"1: {t.date()} の nt_* が変わった: " + \
            str(nt.loc[t][~((nt.loc[t] == nt0.loc[t]) | nt0.loc[t].isna())].index.tolist())
        n_checked["1"] += nt0.shape[1]

    # --- 2. GSM の補間された時刻を使わない ---
    for mode in ("random", "nan"):
        bad = {}
        for k, v in raws.items():
            if k[1] == "jma_gsm":
                b = v.copy()
                inst = [c for c in v.columns if c not in ACCUM]
                m = ~v.index.hour.isin(GSM_NATIVE_HOURS)
                b.loc[m, inst] = rng.uniform(-500, 5000, size=(int(m.sum()), len(inst))) if mode == "random" else np.nan
                bad[k] = b
            else:
                bad[k] = v
        nt, n1 = build_nwp(bad)
        for name, a, b in (("nt", nt, nt0), ("n1", n1, n10)):
            eq = (a == b) | (a.isna() & b.isna())
            assert bool(eq.all().all()), f"2: {name} の {eq.all()[~eq.all()].index.tolist()} が変わった"
            n_checked["2"] += int(a.size)

    # --- 3. SB は GSM t+1日21時・MSM t+1日24時より後を使わない ---
    for i, t in enumerate(dates):
        mode = "random" if i % 2 == 0 else "nan"
        bad = {}
        for k, v in raws.items():
            cut = t + pd.Timedelta(days=1, hours=21) if k[1] == "jma_gsm" else t + pd.Timedelta(days=2)
            bad[k] = corrupt(v, v.index > cut, rng, mode)
        _, n1 = build_nwp(bad)
        assert same(n1.loc[t], n10.loc[t]), f"3: {t.date()} の n1_* が変わった"
        n_checked["3"] += n10.shape[1]

    # --- 4. 確認の確認（書き換えが特徴量に届けば、ちゃんと検出できる） ---
    t = dates[len(dates) // 2]
    for k, when in (((SITE, "jma_gsm"), t + pd.Timedelta(hours=21)), ((SITE, "jma_msm"), t + pd.Timedelta(hours=21))):
        bad = dict(raws)
        bad[k] = corrupt(raws[k], raws[k].index == when, rng, "random")
        nt, _ = build_nwp(bad)
        assert not same(nt.loc[t], nt0.loc[t]), f"4: {k} の {when} を書き換えても nt_* が変わらない（テストが鈍い）"
        n_checked["4"] += 1
    # 補間ありの古いやり方（0〜21時の毎時の平均）なら、t日16〜20時（翌日03時の値が混ざる）を書き換えると値が変わることも確認
    g = raws[(SITE, "jma_gsm")]
    old_mean = lambda r: r["temperature_850hPa"][(r.index.normalize() == t) & (r.index.hour <= 21)].mean()  # noqa: E731
    bad_g = corrupt(g, (g.index.normalize() == t) & g.index.hour.isin([16, 17, 18, 19, 20]), rng, "random")
    assert old_mean(bad_g) != old_mean(g)
    n_checked["4"] += 1

    # --- 5. 修正前のやり方なら 2. で引っかかる ---
    bad = dict(raws)
    for k in [k for k in raws if k[1] == "jma_gsm"]:
        b = raws[k].copy()
        inst = [c for c in b.columns if c not in ACCUM]
        m = ~b.index.hour.isin(GSM_NATIVE_HOURS)
        b.loc[m, inst] = rng.uniform(-500, 5000, size=(int(m.sum()), len(inst)))
        bad[k] = b
    saved = common.GSM_NATIVE_HOURS
    common.GSM_NATIVE_HOURS = tuple(range(0, 22))  # 修正前と同じ「0〜21時の毎時」
    try:
        old0, _ = build_nwp(raws)
        old1, _ = build_nwp(bad)
    finally:
        common.GSM_NATIVE_HOURS = saved
    changed = [c for c in old0.columns if not ((old0[c] == old1[c]) | (old0[c].isna() & old1[c].isna())).all()]
    assert len(changed) > 0
    # --- 6. 観測の特徴量（S0）は t日より後・t日の夜の天気を使わない ---
    base.RAW = RAW_LONG
    daily0 = {"aomori": base._load("aomori", base.AOMORI_COLS), "hirosaki": base._load("hirosaki", base.HIROSAKI_COLS)}
    obs0 = obs_features(daily0)
    s0_cols = feature_cols(build_dataset(), "S0")
    n_obs = 0
    for t in dates:
        bad = {}
        for name, d in daily0.items():
            b = d.copy()
            later = b.index > t
            for c in b.columns:
                if not pd.api.types.is_numeric_dtype(b[c]):
                    b.loc[later, c] = "大雪"
                else:
                    b.loc[later, c] = rng.uniform(-50, 500, size=int(later.sum()))
            if "weather_night" in b:
                b.loc[t, "weather_night"] = "大雪"  # t日の夜（t+1日の0〜6時を含む）
            bad[name] = b
        obs = obs_features(bad)
        assert t in obs.index, f"6: {t.date()} の行がない"
        a, b = obs.loc[t, s0_cols], obs0.loc[t, s0_cols]
        assert same(a, b), f"6: {t.date()} の S0 が変わった: {a.index[~((a == b) | (a.isna() & b.isna()))].tolist()}"
        n_obs += len(s0_cols)
    # 確認の確認: t日当日の対象地点の降雪量を書き換えると S0 は変わる
    t = dates[len(dates) // 2]
    site = "hirosaki" if TARGET == "hirosaki" else "aomori"
    bad = dict(daily0)
    bad[site] = daily0[site].copy()
    bad[site].loc[t, "snowfall"] = 999.0
    assert not same(obs_features(bad).loc[t, s0_cols], obs0.loc[t, s0_cols]), "6: t日の値を書き換えても変わらない"

    print(f"対象地点: {TARGET}（数値予報の地点: {sorted(set(k[0] for k in NWP_SOURCES))}）")
    print(f"検査した日付: {len(dates)}日（{dates.min().date()}〜{dates.max().date()} を含む）")
    print(f"1. SA: t日21時より後を書き換え → 変化なし（{n_checked['1']}セル）")
    print(f"2. GSMの補間された時刻を全期間で書き換え → nt_*・n1_* とも変化なし（{n_checked['2']}セル）")
    print(f"3. SB: GSM t+1日21時・MSM t+1日24時より後を書き換え → 変化なし（{n_checked['3']}セル）")
    print(f"4. 確認の確認: 21時の値を書き換えると変化する（{n_checked['4']}件）")
    print(f"5. 修正前のやり方に戻すと、補間された時刻の書き換えで変わる列: {len(changed)}列 {changed}")
    print(f"6. 観測（S0, {len(s0_cols)}列）: t日より後の日別データと t日の夜の天気を書き換え → 変化なし（{n_obs}セル）。t日当日の対象地点の降雪量を書き換えると変化する")
    print("OK")


if __name__ == "__main__":
    main()
