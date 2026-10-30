"""チューニング済みのモデルと基準を、評価データ（2024・2025年の冬、301日）で1回だけ評価する。

- 主結果: seed 42 の最良パラメータで、学習データ全体（2015〜2023年の冬）に学習し直したモデル。seed 7, 2024 はばらつきの確認。
- 回帰（翌日の降雪量 cm）: MAE / RMSE / R²。大雪日（翌日20cm以上）の検出も、予測値を点数として AUC・AP、
  予測値20cm以上を大雪とした場合と、CV予測でF1が最大になる閾値の場合の適合率・再現率・見逃し数。
- 大雪の分類（大雪日かどうか）: AUC、AP、CV予測でF1が最大になる確率の閾値での適合率・再現率・見逃し数。
- 基準: 持続予測（翌日も今日と同じ）、学習データの月別平均、常に0。
  参考（運用不可）: t+1日の数値予報の降雪量そのもの（MSM・GSM）と、MSMの降雪量に学習データで決めた倍率を掛けたもの。
- 区間: 評価期間の日単位・7日ブロック（冬ごとの循環ブロック）のブートストラップ（5000回）で、S0→SA、SA→SB の差の95%区間。
- 解釈: Ridge回帰・ロジスティック回帰の標準化係数、評価データ上の permutation importance（グループ単位・列単位）。

実行: ../.venv/bin/python -W ignore evaluate.py   （先に run_tune_all.sh を実行しておく）
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from functools import partial

from common import (CLF_METHODS, CLF_TASKS, EVAL_SEASONS, HEAVY, HEAVY_REF, PERSIST_COL, REG_METHODS, RESULTS, SETS,
                    TARGET, TARGET_JA, build_dataset, cv_folds,
                    cv_predict, feature_cols, heavy_label, mae, make_clf, make_reg, predict, proba, r2, rmse, split)

SEEDS = [42, 7, 2024]
MAIN_SEED = 42
N_BOOT = 5000
BLOCK = 7


def load_best(task, fset, method, seed):
    return json.loads((RESULTS / "best" / f"{task}_{fset}_{method}_seed{seed}.json").read_text())


# ---------------------------------------------------------------- 大雪の判定
def best_threshold(yb, score):
    """CV予測で大雪日のF1が最大になる閾値（同点なら高い方）。"""
    yb, score = np.asarray(yb), np.asarray(score)
    best, best_f1 = np.inf, -1.0
    for thr in np.unique(np.round(score, 4)):
        if thr <= 0:
            continue
        f1 = precision_recall_fscore_support(yb, (score >= thr).astype(int), average="binary", zero_division=0)[2]
        if f1 >= best_f1:
            best, best_f1 = float(thr), f1
    return best


def at_threshold(yb, score, thr, name):
    pb = (np.asarray(score) >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)
    return {f"thr_{name}": thr, f"pred_pos_{name}": int(pb.sum()), f"hit_{name}": int((pb & yb).sum()),
            f"miss_{name}": int(((1 - pb) & yb).sum()), f"false_alarm_{name}": int((pb & (1 - yb)).sum()),
            f"precision_{name}": pr, f"recall_{name}": rc, f"f1_{name}": f1}


def rank_metrics(yb, score, prefix=""):
    yb = np.asarray(yb)
    ok = 0 < yb.sum() < len(yb)
    return {f"{prefix}auc": roc_auc_score(yb, score) if ok else np.nan,
            f"{prefix}ap": average_precision_score(yb, score) if yb.sum() else np.nan}


# ---------------------------------------------------------------- ブートストラップ（冬ごとの循環ブロック）
def boot_indices(season, rng, block):
    """評価期間の各冬の中で日を選び直す。block=7 なら7日連続のかたまりで選ぶ（冬の最後と最初をつないだ輪の上で）。"""
    parts = []
    for s in np.unique(season):
        idx = np.where(season == s)[0]
        n = len(idx)
        if block == 1:
            parts.append(idx[rng.integers(0, n, (N_BOOT, n))])
        else:
            nb = int(np.ceil(n / block))
            starts = rng.integers(0, n, (N_BOOT, nb))
            parts.append(idx[((starts[:, :, None] + np.arange(block)) % n).reshape(N_BOOT, -1)[:, :n]])
    return np.concatenate(parts, axis=1)


def boot_diff(stat, y, pa, pb, season, seed=0, need_heavy=False, thr=None):
    """stat(B) − stat(A) の点推定と95%区間。stat は (y, p) → 値。
    need_heavy=True（AP・AUC）のときは、選び直した日に大雪日が1日もない回を除く。"""
    out = {"diff": stat(y, pb) - stat(y, pa)}
    for name, block in (("day", 1), ("block7", BLOCK)):
        idx = boot_indices(season, np.random.default_rng(seed), block)
        d = np.array([stat(y[i], pb[i]) - stat(y[i], pa[i]) if (not need_heavy or heavy_ok(y[i], thr)) else np.nan
                      for i in idx])
        out[f"ci_lo_{name}"], out[f"ci_hi_{name}"] = np.nanpercentile(d, [2.5, 97.5])
        out[f"p_B_gt_A_{name}"] = float(np.nanmean(d > 0))
        out[f"n_valid_{name}"] = int(np.isfinite(d).sum())
    return out


def heavy_ok(y, thr=None):
    yb = heavy_label(y, thr)
    return 0 < yb.sum() < len(yb)


def ap_stat(y, p, thr=None):
    """average precision（sklearn と同じ定義を numpy で。同点の点数は1つの閾値として扱う）。"""
    yb = heavy_label(y, thr)
    o = np.argsort(-p, kind="mergesort")
    ps, ys = p[o], yb[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]  # 同じ点数のかたまりの最後
    tp, fp = tp[last], fp[last]
    prec, rec = tp / (tp + fp), tp / yb.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def auc_stat(y, p, thr=None):
    """AUC（同点は0.5として数える。順位の平均を使う Mann-Whitney の式）。"""
    yb = heavy_label(y, thr)
    ranks = rankdata(p)
    n1 = yb.sum()
    n0 = len(yb) - n1
    return float((ranks[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mae_stat(y, p):
    return float(np.mean(np.abs(y - p)))


# ---------------------------------------------------------------- 特徴量のグループ
# 観測の特徴量のグループは、前回（04-snow-model/evaluate.py）と同じ分け方
OBS_GROUPS = {
    "気温": ["temp", "cold"],
    "湿度": ["humidity"],
    "日照": ["sunshine"],
    "降雪・積雪": ["snowfall", "snow_depth"],
    "降水・天気": ["precip", "day_", "prevnight"],
    "風": ["wind", "gust", "mode_"],
    "気圧": ["pressure"],
    "季節(暦)": ["season_"],
}


def obs_group_of(c):
    if c == "a_cold_x_humid":
        return "気温"
    for g, keys in OBS_GROUPS.items():
        if any(k in c for k in keys):
            return g
    return "その他"


def group_of(c):
    if not c.startswith(("nt_", "n1_")):
        return "観測: " + obs_group_of(c)
    day = "t日" if c.startswith("nt_") else "t+1日"
    r = c[3:]
    if r.startswith("up_"):
        g = "上流点（日本海）"
    elif r.startswith("t2m_minus") or any(k in r for k in ("t850", "t500", "stab")):
        g = "上空の気温（850/500hPa）"
    elif "rh" in r:
        g = "上空の湿度（850/700hPa）"
    elif any(k in r for k in ("u850", "v850", "ws850")):
        g = "上空の風（850hPa）"
    elif "z500" in r:
        g = "500hPa高度"
    elif "precip" in r or "snow" in r:
        g = "地上の降水・降雪"
    else:
        g = "地上の気温・気圧"
    return f"数値予報[{day}]: {g}"


def group_permutation(score_fn, X, n_repeats=50, seed=0):
    """同じグループの列をまとめて同じ順に並べ替え、点数の悪化を測る。score_fn(X) → 小さいほど良い値。"""
    base_score = score_fn(X)
    rng = np.random.default_rng(seed)
    groups = pd.Series({c: group_of(c) for c in X.columns})
    rows = []
    for g, cols in groups.groupby(groups).groups.items():
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[list(cols)] = X[list(cols)].to_numpy()[rng.permutation(len(X))]
            inc.append(score_fn(Xp) - base_score)
        rows.append(dict(group=g, n_features=len(cols), increase=np.mean(inc), std=np.std(inc)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 基準
def baselines(train, ev):
    """基準の予測（CVの各検証冬・評価）。CVの月別平均・倍率は、検証冬より前の冬だけで計算する。"""
    folds = cv_folds(train)

    def scale_k(d):  # MSMの降雪量に掛ける倍率（MAEが最小になる値を格子で探す）
        ks = np.arange(0.5, 6.01, 0.05)
        return float(ks[np.argmin([mae(d.target, k * d.n1_msm_snow_sum.fillna(0)) for k in ks])])

    defs = {
        "persistence": ("基準: 持続予測（翌日も今日と同じ）", lambda tr, d: d[PERSIST_COL].fillna(0).to_numpy()),
        "monthly_mean": ("基準: 学習データの月別平均",
                         lambda tr, d: d.target_month.map(tr.groupby("target_month").target.mean()).to_numpy()),
        "zero": ("基準: 常に0", lambda tr, d: np.zeros(len(d))),
        "nwp_msm_raw": ("参考(運用不可): t+1日のMSM降雪量そのまま", lambda tr, d: d.n1_msm_snow_sum.fillna(0).to_numpy()),
        "nwp_gsm_raw": ("参考(運用不可): t+1日のGSM降雪量そのまま", lambda tr, d: d.n1_gsm_snow_sum.fillna(0).to_numpy()),
        "nwp_msm_scaled": ("参考(運用不可): t+1日のMSM降雪量×倍率（学習データで決定）",
                           lambda tr, d: scale_k(tr) * d.n1_msm_snow_sum.fillna(0).to_numpy()),
    }
    out = {}
    for key, (name, fn) in defs.items():
        oof = [(train.iloc[va].target.to_numpy(), fn(train.iloc[tr], train.iloc[va])) for tr, va in folds]
        out[key] = dict(name=name, oof=oof, ev=fn(train, ev), tr=fn(train, train))
    out["nwp_msm_scaled"]["k"] = scale_k(train)
    return out


# ---------------------------------------------------------------- 本体
def reg_row(key, name, fset, method, oof, p_tr, p_ev, ytr, ev, cv_score=None):
    yev = ev.target.to_numpy()
    oy, op = np.concatenate([a for a, _ in oof]), np.concatenate([b for _, b in oof])
    yb_ev = heavy_label(yev)
    thr = best_threshold(heavy_label(oy), op)
    row = dict(key=key, name=name, set=fset, method=method,
               cv_mae=cv_score if cv_score is not None else float(np.mean([mae(a, b) for a, b in oof])),
               cv_fold_mae=[round(mae(a, b), 3) for a, b in oof], cv_rmse=rmse(oy, op), cv_r2=r2(oy, op),
               **rank_metrics(heavy_label(oy), op, "cv_heavy_"),
               train_mae=mae(ytr, p_tr), eval_mae=mae(yev, p_ev), eval_rmse=rmse(yev, p_ev), eval_r2=r2(yev, p_ev))
    for s in EVAL_SEASONS:
        m = (ev.season == s).to_numpy()
        row[f"eval_mae_{s}"] = mae(yev[m], p_ev[m])
    row.update(rank_metrics(yb_ev, p_ev, "eval_heavy_"))
    row.update(at_threshold(yb_ev, p_ev, HEAVY, "at20"))
    row.update(at_threshold(yb_ev, p_ev, thr, "atcv"))
    row["eval_mean_pred_on_heavy"] = float(p_ev[yb_ev == 1].mean())
    row["eval_max_pred"] = float(p_ev.max())
    return row


def clf_row(key, name, fset, method, oof, s_ev, ev, cv_score=None, s_tr=None, ytr=None, heavy_thr=None):
    yb_ev = heavy_label(ev.target, heavy_thr)
    oy, op = np.concatenate([a for a, _ in oof]), np.concatenate([b for _, b in oof])
    thr = best_threshold(oy, op)
    row = dict(key=key, name=name, set=fset, method=method, **rank_metrics(oy, op, "cv_"),
               cv_fold_ap=[round(average_precision_score(a, b), 3) if a.sum() else None for a, b in oof],
               cv_score=cv_score)
    if s_tr is not None:  # 学習データ全体で学習したモデルで、同じ学習データを予測した場合（過学習の確認用）
        row.update(rank_metrics(heavy_label(ytr, heavy_thr), s_tr, "train_"))
    row.update(at_threshold(oy, op, thr, "cv_oof"))  # CV予測そのものでの成績（閾値を決めたデータなので楽観的）
    row.update(rank_metrics(yb_ev, s_ev, "eval_"))
    for s in EVAL_SEASONS:
        m = (ev.season == s).to_numpy()
        row.update(rank_metrics(yb_ev[m], s_ev[m], f"eval_{s}_"))
    row.update(at_threshold(yb_ev, s_ev, thr, "atcv"))
    row["eval_prevalence"] = float(yb_ev.mean())
    row["heavy_threshold"] = HEAVY if heavy_thr is None else heavy_thr
    return row


def main():
    df = build_dataset()
    train, ev = split(df)
    yev, ytr, season = ev.target.to_numpy(), train.target, ev.season.to_numpy()
    folds = cv_folds(train)
    print(f"対象地点: {TARGET_JA}（{TARGET}）。大雪の閾値: 主 {HEAVY}cm" +
          (f"、参考 {HEAVY_REF}cm（学習データの予測対象日の降雪量の95パーセンタイル = {ytr.quantile(0.95)}cm）"
           if HEAVY_REF else ""))
    if HEAVY_REF:
        print(f"  参考の閾値での大雪日: 学習 {int((ytr >= HEAVY_REF).sum())}日、評価 {int((yev >= HEAVY_REF).sum())}日")
    print(f"学習: {train.target_date.min().date()}〜{train.target_date.max().date()} {len(train)}行 "
          f"(大雪日 {int((ytr >= HEAVY).sum())}日)")
    print(f"評価: {ev.target_date.min().date()}〜{ev.target_date.max().date()} {len(ev)}行 "
          f"(大雪日 {int((yev >= HEAVY).sum())}日: " +
          ", ".join(f"{s}年の冬 {int((ev[ev.season == s].target >= HEAVY).sum())}日" for s in EVAL_SEASONS) + ")")
    print("特徴量の数:", {s: len(feature_cols(df, s)) for s in SETS})
    print("CVの検証冬ごとの大雪日:", {s: int((train[train.season == s].target >= HEAVY).sum()) for s in
                                 sorted(set(train.season))})

    reg_rows, seed_rows = [], []
    clf_rows_by = {task: [] for task in CLF_TASKS}
    preds = {"target_date": ev.target_date.dt.date.to_numpy(), "season": season, "actual": yev}
    models = {}

    # --- 基準 ---
    bl = baselines(train, ev)
    print(f"MSM降雪量に掛ける倍率（学習データ全体で決定）: {bl['nwp_msm_scaled']['k']:.2f}")
    for key, b in bl.items():
        reg_rows.append(reg_row(key, b["name"], "-", key, b["oof"], b["tr"], b["ev"], ytr, ev))
        preds[key] = b["ev"]
        if key != "monthly_mean":  # 月別平均は大雪の点数にならない（月の中で同じ値）ので分類表には入れない
            for task, thr in CLF_TASKS.items():
                clf_rows_by[task].append(clf_row(key, b["name"], "-", key, [(heavy_label(a, thr), q) for a, q in b["oof"]],
                                                 b["ev"], ev, s_tr=b["tr"], ytr=ytr, heavy_thr=thr))

    # --- 回帰 ---
    for seed in SEEDS:
        for fset in SETS:
            cols = feature_cols(df, fset)
            for method, mname in REG_METHODS.items():
                e = load_best("reg", fset, method, seed)
                oof = cv_predict("reg", method, e["params"], train[cols], ytr, folds, seed, n_jobs=-1)
                m = make_reg(method, e["params"], seed=seed, n_jobs=-1).fit(train[cols], ytr)
                p_tr, p_ev = predict(m, train[cols]), predict(m, ev[cols])
                key = f"{fset}_{method}"
                row = reg_row(key, f"{fset} {mname}", fset, method, oof, p_tr, p_ev, ytr, ev, e["cv_score"])
                row["params"] = json.dumps(e["params"], ensure_ascii=False)
                seed_rows.append(dict(task="reg", seed=seed, **{k: row[k] for k in (
                    "key", "cv_mae", "eval_mae", "eval_rmse", "eval_r2", "eval_heavy_auc", "eval_heavy_ap",
                    "recall_at20", "recall_atcv", "precision_atcv")}))
                if seed == MAIN_SEED:
                    reg_rows.append(row)
                    preds[key] = p_ev
                    models[key] = (m, cols)
                print(f"[reg seed{seed}] {key}: CV MAE={e['cv_score']:.3f} eval MAE={row['eval_mae']:.3f}", flush=True)

    # --- 大雪の分類（主 = 20cm の "clf"。弘前は参考の15cm "clf_ref" も） ---
    pfx = {"clf": "p_", "clf_ref": "pref_"}
    for task, thr in CLF_TASKS.items():
        for seed in SEEDS:
            for fset in SETS:
                cols = feature_cols(df, fset)
                for method, mname in CLF_METHODS.items():
                    e = load_best(task, fset, method, seed)
                    oof = cv_predict(task, method, e["params"], train[cols], ytr, folds, seed)
                    m = make_clf(method, e["params"], seed=seed).fit(train[cols], heavy_label(ytr, thr))
                    s_ev = proba(m, ev[cols])
                    key = f"{fset}_{method}"
                    row = clf_row(key, f"{fset} {mname}", fset, method, oof, s_ev, ev, e["cv_score"],
                                  s_tr=proba(m, train[cols]), ytr=ytr, heavy_thr=thr)
                    row["params"] = json.dumps(e["params"], ensure_ascii=False)
                    seed_rows.append(dict(task=task, seed=seed, key=key, cv_ap=e["cv_score"], cv_auc=row["cv_auc"],
                                          eval_auc=row["eval_auc"], eval_ap=row["eval_ap"],
                                          recall_atcv=row["recall_atcv"], precision_atcv=row["precision_atcv"],
                                          miss_atcv=row["miss_atcv"]))
                    if seed == MAIN_SEED:
                        clf_rows_by[task].append(row)
                        preds[f"{pfx[task]}{key}"] = s_ev
                        if task == "clf":
                            models[f"clf_{key}"] = (m, cols)
                    print(f"[{task} seed{seed}] {key}: CV AP={e['cv_score']:.3f} eval AP={row['eval_ap']:.3f}", flush=True)

    reg = pd.DataFrame(reg_rows)
    z = reg.set_index("key").loc["zero"]
    for part in ("train", "cv", "eval"):  # 「常に0」に比べてMAEを何割減らせたか（データの雪の多さの違いをならして比べる）
        reg[f"skill_vs_zero_{part}"] = 1 - reg[f"{part}_mae"] / z[f"{part}_mae"]
    clf = pd.DataFrame(clf_rows_by["clf"])
    clf_ref = pd.DataFrame(clf_rows_by["clf_ref"]) if "clf_ref" in CLF_TASKS else None
    sr = pd.DataFrame(seed_rows)
    reg.to_csv(RESULTS / "metrics_reg.csv", index=False, encoding="utf-8-sig")
    clf.to_csv(RESULTS / "metrics_clf.csv", index=False, encoding="utf-8-sig")
    if clf_ref is not None:
        clf_ref.to_csv(RESULTS / "metrics_clf_ref.csv", index=False, encoding="utf-8-sig")
    sr.to_csv(RESULTS / "seed_robustness.csv", index=False, encoding="utf-8-sig")
    pe = pd.DataFrame(preds)
    pe.to_csv(RESULTS / "predictions_eval.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_colwidth", 60)
    print(f"\n=== 回帰: 翌日の{TARGET_JA}の降雪量。評価データ（2024・2025年の冬, {len(ev)}日）, seed 42 ===")
    print(reg[["name", "cv_mae", "cv_rmse", "cv_r2", "train_mae", "eval_mae", "eval_rmse", "eval_r2", "eval_mae_2024",
               "eval_mae_2025"]].round(3).to_string(index=False))
    print("\n=== 過学習の確認: 学習MAE（学習データ全体で学習→同じデータを予測）/ CV MAE / 評価MAE と、「常に0」に比べて減らせた割合 ===")
    print(reg[["name", "train_mae", "cv_mae", "eval_mae", "skill_vs_zero_train", "skill_vs_zero_cv", "skill_vs_zero_eval"]]
          .round(3).to_string(index=False))
    print("\n=== 回帰の予測値で大雪日（20cm以上）を判定した場合（評価データ） ===")
    print(reg[["name", "eval_heavy_auc", "eval_heavy_ap", "pred_pos_at20", "hit_at20", "miss_at20", "thr_atcv",
               "pred_pos_atcv", "hit_atcv", "miss_atcv", "precision_atcv", "recall_atcv",
               "eval_mean_pred_on_heavy", "eval_max_pred"]].round(3).to_string(index=False))
    print("\n=== 大雪の分類: CV（学習データ内）と評価データ, seed 42 ===")
    print(clf[["name", "train_auc", "train_ap", "cv_auc", "cv_ap", "cv_fold_ap", "thr_atcv", "eval_auc", "eval_ap", "eval_2024_ap", "eval_2025_ap",
               "pred_pos_atcv", "hit_atcv", "miss_atcv", "false_alarm_atcv", "precision_atcv", "recall_atcv"]]
          .round(3).to_string(index=False))
    print(f"（評価データの大雪日の割合 = APの「でたらめ」の水準: {clf.eval_prevalence.iloc[0]:.3f}）")
    print("CV予測そのものでの成績（閾値を決めたデータなので楽観的）:")
    print(clf[["name", "pred_pos_cv_oof", "hit_cv_oof", "miss_cv_oof", "precision_cv_oof", "recall_cv_oof"]]
          .round(3).to_string(index=False))
    if clf_ref is not None:
        print(f"\n=== 参考: 大雪の閾値を {HEAVY_REF}cm にした分類（seed 42） ===")
        print(clf_ref[["name", "train_auc", "train_ap", "cv_auc", "cv_ap", "thr_atcv", "eval_auc", "eval_ap", "eval_2024_ap",
                       "eval_2025_ap", "pred_pos_atcv", "hit_atcv", "miss_atcv", "false_alarm_atcv", "precision_atcv",
                       "recall_atcv"]].round(3).to_string(index=False))
        print(f"（評価データでの割合 = APの「でたらめ」の水準: {clf_ref.eval_prevalence.iloc[0]:.3f}）")

    print("\n=== seed違い（7 / 42 / 2024） ===")
    r = sr[sr.task == "reg"].pivot_table(index="key", columns="seed", values=["cv_mae", "eval_mae"])
    print(r.round(3).to_string())
    for task in CLF_TASKS:
        c = sr[sr.task == task].pivot_table(index="key", columns="seed", values=["cv_ap", "eval_ap", "eval_auc", "miss_atcv"])
        print(f"[{task}]")
        print(c.round(3).to_string())
    diffs = []
    combos = [("reg", "eval_mae", REG_METHODS)] + [(t, m, CLF_METHODS) for t in CLF_TASKS for m in ("eval_ap", "eval_auc")]
    for task, metric, methods in combos:
        t = sr[sr.task == task].set_index(["seed", "key"])[metric]
        for method in methods:
            for a_set, b_set in (("S0", "SA"), ("SA", "SB")):
                d = [t[(s, f"{b_set}_{method}")] - t[(s, f"{a_set}_{method}")] for s in SEEDS]
                diffs.append(dict(task=task, metric=metric, method=method, compare=f"{b_set}-{a_set}",
                                  **{f"seed{s}": v for s, v in zip(SEEDS, d)}))
    dd = pd.DataFrame(diffs)
    dd.to_csv(RESULTS / "seed_set_differences.csv", index=False, encoding="utf-8-sig")
    print("特徴量セットの差（後 − 前）の seed ごとの値（MAEは負なら改善、AP/AUCは正なら改善）:")
    print(dd.round(3).to_string(index=False))

    # --- ブートストラップ ---
    boot = []
    for method in REG_METHODS:
        for a_set, b_set in (("S0", "SA"), ("SA", "SB"), ("S0", "SB")):
            pa, pb = preds[f"{a_set}_{method}"], preds[f"{b_set}_{method}"]
            boot.append(dict(task="reg", metric="MAE", method=method, A=a_set, B=b_set,
                             **boot_diff(mae_stat, yev, pa, pb, season)))
            boot.append(dict(task="reg", metric="AP(予測値で大雪)", method=method, A=a_set, B=b_set,
                             **boot_diff(ap_stat, yev, pa, pb, season, need_heavy=True)))
        boot.append(dict(task="reg", metric="MAE", method=method, A="persistence", B="SA",
                         **boot_diff(mae_stat, yev, preds["persistence"], preds[f"SA_{method}"], season)))
    for task, thr in CLF_TASKS.items():
        stats = (("AP", partial(ap_stat, thr=thr)), ("AUC", partial(auc_stat, thr=thr)))
        for method in CLF_METHODS:
            for a_set, b_set in (("S0", "SA"), ("SA", "SB"), ("S0", "SB")):
                pa, pb = preds[f"{pfx[task]}{a_set}_{method}"], preds[f"{pfx[task]}{b_set}_{method}"]
                for metric, fn in stats:
                    boot.append(dict(task=task, metric=metric, method=method, A=a_set, B=b_set,
                                     **boot_diff(fn, yev, pa, pb, season, need_heavy=True, thr=thr)))
            for metric, fn in stats:
                boot.append(dict(task=task, metric=metric, method=method, A="persistence", B="SA",
                                 **boot_diff(fn, yev, preds["persistence"], preds[f"{pfx[task]}SA_{method}"], season,
                                             need_heavy=True, thr=thr)))
                boot.append(dict(task=task, metric=metric, method=method, A="nwp_msm_raw", B="SB",
                                 **boot_diff(fn, yev, preds["nwp_msm_raw"], preds[f"{pfx[task]}SB_{method}"], season,
                                             need_heavy=True, thr=thr)))
    bt = pd.DataFrame(boot)
    bt.to_csv(RESULTS / "bootstrap.csv", index=False, encoding="utf-8-sig")
    print("\n=== 差（B − A）の95%区間（評価データ、5000回。MAEは負ならBが良い、AP/AUCは正ならBが良い） ===")
    print(bt[["task", "metric", "method", "A", "B", "diff", "ci_lo_day", "ci_hi_day", "ci_lo_block7", "ci_hi_block7",
              "n_valid_block7"]].round(3).to_string(index=False))

    # --- 評価データの大雪日と、予測が大きく出た日 ---
    show = ["target_date", "actual", "persistence", "nwp_msm_raw", "S0_hgb", "SA_hgb", "SB_hgb", "S0_ridge", "SA_ridge",
            "SB_ridge", "p_S0_logreg", "p_SA_logreg", "p_SB_logreg", "p_S0_hgbc", "p_SA_hgbc", "p_SB_hgbc"]
    hd = pe[(pe.actual >= HEAVY) | (pe[["SA_hgb", "SA_ridge", "SA_rf"]].max(axis=1) >= 15)]
    hd[show].to_csv(RESULTS / "heavy_days_eval.csv", index=False, encoding="utf-8-sig")
    print("\n=== 評価データの大雪日（実測20cm以上）と、SAの回帰の予測が15cm以上だった日 ===")
    print(hd[show].round(2).to_string(index=False))

    # --- 解釈: 線形モデルの標準化係数 ---
    coef_rows = []
    for fset in SETS:
        m, cols = models[f"{fset}_ridge"]
        reg_ = m.regressor_ if hasattr(m, "regressor_") else m
        scale = "log1p(cm)" if hasattr(m, "regressor_") else "cm"
        for c_, v in zip(cols, reg_.named_steps["reg"].coef_):
            coef_rows.append(dict(model="ridge", set=fset, feature=c_, group=group_of(c_), std_coef=v, scale=scale))
        m, cols = models[f"clf_{fset}_logreg"]
        for c_, v in zip(cols, m.named_steps["clf"].coef_[0]):
            coef_rows.append(dict(model="logreg", set=fset, feature=c_, group=group_of(c_), std_coef=v,
                                  scale="対数オッズ"))
    coef = pd.DataFrame(coef_rows)
    coef["abs"] = coef.std_coef.abs()
    coef = coef.sort_values(["model", "set", "abs"], ascending=[True, True, False]).drop(columns="abs")
    coef.to_csv(RESULTS / "linear_coefficients.csv", index=False, encoding="utf-8-sig")
    print("\n=== 線形モデルの標準化係数（特徴量を平均0・標準偏差1にしたときの係数）上位10 ===")
    for model in ("ridge", "logreg"):
        for fset in ("SA", "SB"):
            cc = coef[(coef.model == model) & (coef.set == fset)]
            print(f"[{model} {fset}] 尺度: {cc.scale.iloc[0]}")
            print(cc.head(10)[["feature", "group", "std_coef"]].round(3).to_string(index=False))

    # --- 解釈: 評価データ上の permutation importance ---
    gp_rows, pi_rows = [], []
    for fset in ("S0", "SA", "SB"):
        for key, kind in ([(f"{fset}_{m}", "reg") for m in REG_METHODS] + [(f"clf_{fset}_{m}", "clf") for m in CLF_METHODS]):
            m, cols = models[key]
            Xev = ev[cols]
            if kind == "reg":
                fn = lambda X, m=m: mae(yev, predict(m, X))  # noqa: E731  MAEの増加
                sc = lambda est, X, y: -mae(y, predict(est, X))  # noqa: E731
                yy = yev
            else:
                fn = lambda X, m=m: -average_precision_score(heavy_label(yev), proba(m, X))  # noqa: E731  APの低下
                sc = lambda est, X, y: average_precision_score(y, proba(est, X))  # noqa: E731
                yy = heavy_label(yev)
            g = group_permutation(fn, Xev)
            g.insert(0, "model", key)
            g.insert(1, "kind", "MAEの増加(cm)" if kind == "reg" else "APの低下")
            gp_rows.append(g)
            if fset != "S0" and not key.endswith("_rf"):  # 列単位はRF以外（RFは時間がかかるため）
                pi = permutation_importance(m, Xev, yy, scoring=sc, n_repeats=30, random_state=0, n_jobs=-1)
                for c_, v, s_ in zip(cols, pi.importances_mean, pi.importances_std):
                    pi_rows.append(dict(model=key, feature=c_, group=group_of(c_), increase=v, std=s_))
    gp = pd.concat(gp_rows).sort_values(["model", "increase"], ascending=[True, False])
    gp.to_csv(RESULTS / "permutation_groups.csv", index=False, encoding="utf-8-sig")
    pi = pd.DataFrame(pi_rows).sort_values(["model", "increase"], ascending=[True, False])
    pi.to_csv(RESULTS / "permutation_features.csv", index=False, encoding="utf-8-sig")
    print("\n=== グループ単位の permutation importance（評価データ、50回の平均。回帰=MAEの増加cm、分類=APの低下） ===")
    for fset in ("S0", "SA", "SB"):
        keys = [f"{fset}_{m}" for m in REG_METHODS] + [f"clf_{fset}_{m}" for m in CLF_METHODS]
        t = gp[gp.model.isin(keys)].pivot_table(index="group", columns="model", values="increase")[keys]
        print(f"[{fset}]")
        print(t.round(3).sort_values(keys[1], ascending=False).to_string())
    print("\n=== 列単位の permutation importance 上位8（評価データ、30回） ===")
    for key in ("SA_ridge", "SA_hgb", "clf_SA_logreg", "clf_SA_hgbc", "SB_hgb", "clf_SB_hgbc"):
        print(f"[{key}]")
        print(pi[pi.model == key].head(8)[["feature", "increase", "std"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
