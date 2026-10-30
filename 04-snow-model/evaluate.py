"""チューニング済みの各手法と基準を、評価データ（2025年の冬）で1回だけ評価する。

- 主結果: seed 42 の最良パラメータで学習データ全体（4冬）に学習し直したモデル。
- 比べる基準: 持続予測（翌日も今日と同じ降雪量）、学習データの月別平均、常に0。
- 大雪日（翌日の降雪20cm以上）の検出: 予測値の順位でのAUC・平均適合率(AP)、
  予測値20cm以上を「大雪」とした場合の適合率・再現率、
  学習データのCV予測（各冬を前の冬だけで学習して予測した値）でF1が最大になる閾値を決め、それを評価データに当てた場合。
- 過学習: 学習データへの当てはまり（学習データ全体で学習→同じデータを予測）、CVスコア、評価データの差。
- 信頼区間: 評価期間の日単位・7日ブロック（循環ブロック）のブートストラップで MAE差 の95%区間（5000回）。
- seed違い: seed 7, 2024 でも同じ手順を回し、評価MAEのばらつきを見る。
- 解釈: Ridge/ElasticNet の標準化係数、評価データ上の permutation importance（単独の特徴量・特徴量のグループ）。

実行: ../.venv/bin/python evaluate.py   （先に tune.py を seed 42, 7, 2024 で実行しておく）
"""
import json

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from common import (HEAVY, METHODS, RESULTS, build_dataset, cv_folds, feature_cols, mae, make_model, predict, r2,
                    rmse, split)

SEEDS = [42, 7, 2024]
MAIN_SEED = 42
N_BOOT = 5000
BLOCK = 7


# ---------------------------------------------------------------- 大雪日の検出
def heavy_metrics(y, p, thr_cv):
    yb = (np.asarray(y) >= HEAVY).astype(int)
    out = {"heavy_days": int(yb.sum())}
    out["auc_heavy"] = roc_auc_score(yb, p) if 0 < yb.sum() < len(yb) else np.nan
    out["ap_heavy"] = average_precision_score(yb, p) if yb.sum() else np.nan
    for name, thr in (("at20", HEAVY), ("atcv", thr_cv)):
        pb = (np.asarray(p) >= thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)
        out[f"pred_heavy_{name}"] = int(pb.sum())
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = pr, rc, f1
    out["threshold_cv"] = thr_cv
    return out


def best_threshold(y, p):
    """CV予測で大雪日のF1が最大になる閾値（予測値の候補から選ぶ）。同点なら高い方。"""
    yb = (np.asarray(y) >= HEAVY).astype(int)
    best, best_f1 = HEAVY, -1.0
    for thr in np.unique(np.round(p, 2)):
        if thr <= 0:
            continue
        pb = (p >= thr).astype(int)
        f1 = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)[2]
        if f1 >= best_f1:
            best, best_f1 = float(thr), f1
    return best


# ---------------------------------------------------------------- ブートストラップ
def boot_indices(n, rng, block):
    if block == 1:
        return rng.integers(0, n, (N_BOOT, n))
    # 循環ブロック: 期間の最後と最初をつなげて輪にし、7日の連続ブロックを選ぶ（端の日が選ばれにくくなる偏りを避ける）
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, (N_BOOT, nb))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(N_BOOT, -1)[:, :n]


def mae_diff_ci(y, pa, pb, seed=0):
    """MAE(A) - MAE(B)。負なら A の方が誤差が小さい。"""
    d = np.abs(y - pa) - np.abs(y - pb)
    out = {"mae_diff": float(d.mean())}
    for name, block in (("day", 1), ("block7", BLOCK)):
        idx = boot_indices(len(d), np.random.default_rng(seed), block)
        m = d[idx].mean(axis=1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        out[f"ci_lo_{name}"], out[f"ci_hi_{name}"] = float(lo), float(hi)
        out[f"p_A_better_{name}"] = float((m < 0).mean())
    return out


# ---------------------------------------------------------------- 特徴量のグループ
GROUPS = {
    "気温": ["temp", "cold"],
    "湿度": ["humidity"],
    "日照": ["sunshine"],
    "降雪・積雪": ["snowfall", "snow_depth"],
    "降水・天気": ["precip", "day_", "prevnight"],
    "風": ["wind", "gust", "mode_"],
    "気圧": ["pressure"],
    "季節(暦)": ["season_"],
}


def group_of(col):
    if col == "a_cold_x_humid":
        return "気温"  # 寒さ×湿度。気温グループに入れる
    for g, keys in GROUPS.items():
        if any(k in col for k in keys):
            return g
    return "その他"


def group_permutation(model, X, y, n_repeats=50, seed=0):
    base = mae(y, predict(model, X))
    rng = np.random.default_rng(seed)
    rows = []
    groups = pd.Series({c: group_of(c) for c in X.columns})
    for g, cols in groups.groupby(groups).groups.items():
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[list(cols)] = X[list(cols)].to_numpy()[perm]  # グループ内の列は同じ並べ替え（列間の関係は保つ）
            inc.append(mae(y, predict(model, Xp)) - base)
        rows.append(dict(group=g, n_features=len(cols), mae_increase=np.mean(inc), std=np.std(inc)))
    return pd.DataFrame(rows)


def neg_mae(est, X, y):
    """評価と同じく、0未満を0に切った予測でMAEを測る。"""
    return -mae(y, predict(est, X))


# ---------------------------------------------------------------- 本体
def fit_eval(method, entry, train, ev, df, seed):
    cols = feature_cols(df, hinge=METHODS[method][1])
    Xtr, ytr, Xev, yev = train[cols], train.target, ev[cols], ev.target
    # CV予測（大雪の閾値決め・CVのRMSE/R²用）
    oof_y, oof_p = [], []
    for tr_idx, va_idx in cv_folds(train):
        m = make_model(method, entry["params"], seed=seed, n_jobs=-1)
        m.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
        oof_y.append(ytr.iloc[va_idx].to_numpy())
        oof_p.append(predict(m, Xtr.iloc[va_idx]))
    oof_y, oof_p = np.concatenate(oof_y), np.concatenate(oof_p)
    model = make_model(method, entry["params"], seed=seed, n_jobs=-1)
    model.fit(Xtr, ytr)
    p_tr, p_ev = predict(model, Xtr), predict(model, Xev)
    return model, cols, p_tr, p_ev, oof_y, oof_p


def baselines(train, ev):
    """基準の予測（学習・CV・評価）。CVは各冬を前の冬だけから計算（月別平均）。"""
    out = {}
    month_mean = train.groupby("target_month").target.mean()
    out["persistence"] = dict(name="基準: 持続予測（翌日も今日と同じ）",
                              tr=train.a_snowfall.fillna(0).to_numpy(), ev=ev.a_snowfall.fillna(0).to_numpy())
    out["monthly_mean"] = dict(name="基準: 学習データの月別平均",
                               tr=train.target_month.map(month_mean).to_numpy(), ev=ev.target_month.map(month_mean).to_numpy())
    out["zero"] = dict(name="基準: 常に0", tr=np.zeros(len(train)), ev=np.zeros(len(ev)))
    folds = cv_folds(train)
    for key in out:
        ys, ps = [], []
        for tr_idx, va_idx in folds:
            va = train.iloc[va_idx]
            if key == "persistence":
                p = va.a_snowfall.fillna(0).to_numpy()
            elif key == "monthly_mean":
                p = va.target_month.map(train.iloc[tr_idx].groupby("target_month").target.mean()).to_numpy()
            else:
                p = np.zeros(len(va))
            ys.append(va.target.to_numpy())
            ps.append(p)
        out[key]["oof_y"], out[key]["oof_p"] = np.concatenate(ys), np.concatenate(ps)
        out[key]["fold_mae"] = [round(mae(a, b), 4) for a, b in zip(ys, ps)]
    return out


def main():
    df = build_dataset()
    train, ev = split(df)
    yev = ev.target.to_numpy()
    print(f"学習: {train.target_date.min().date()}〜{train.target_date.max().date()} {len(train)}行 "
          f"(大雪日 {(train.target >= HEAVY).sum()}日) / 評価: {ev.target_date.min().date()}〜"
          f"{ev.target_date.max().date()} {len(ev)}行 (大雪日 {(ev.target >= HEAVY).sum()}日)")

    best = {s: json.loads((RESULTS / f"best_params_seed{s}.json").read_text()) for s in SEEDS}
    rows, preds, models, seed_rows = [], {"target_date": ev.target_date.dt.date.to_numpy(), "actual": yev}, {}, []

    # --- 基準 ---
    for key, b in baselines(train, ev).items():
        thr = best_threshold(b["oof_y"], b["oof_p"]) if key == "persistence" else HEAVY
        rows.append(dict(key=key, name=b["name"], cv_mae=np.mean(b["fold_mae"]), cv_fold_mae=b["fold_mae"],
                         cv_rmse=rmse(b["oof_y"], b["oof_p"]), cv_r2=r2(b["oof_y"], b["oof_p"]),
                         train_mae=mae(train.target, b["tr"]), eval_mae=mae(yev, b["ev"]), eval_rmse=rmse(yev, b["ev"]),
                         eval_r2=r2(yev, b["ev"]), **heavy_metrics(yev, b["ev"], thr)))
        preds[key] = b["ev"]

    # --- 手法（seed 42 が主結果、他seedはばらつき確認） ---
    for seed in SEEDS:
        for method, (name, _) in METHODS.items():
            entry = best[seed][method]
            model, cols, p_tr, p_ev, oof_y, oof_p = fit_eval(method, entry, train, ev, df, seed)
            seed_rows.append(dict(seed=seed, key=method, name=name, cv_mae=entry["cv_mae"], eval_mae=mae(yev, p_ev),
                                  eval_rmse=rmse(yev, p_ev), eval_r2=r2(yev, p_ev),
                                  params=json.dumps(entry["params"], ensure_ascii=False)))
            if seed != MAIN_SEED:
                continue
            thr = best_threshold(oof_y, oof_p)
            rows.append(dict(key=method, name=name, cv_mae=entry["cv_mae"], cv_fold_mae=entry["fold_mae"],
                             cv_rmse=rmse(oof_y, oof_p), cv_r2=r2(oof_y, oof_p),
                             train_mae=mae(train.target, p_tr), eval_mae=mae(yev, p_ev), eval_rmse=rmse(yev, p_ev),
                             eval_r2=r2(yev, p_ev), **heavy_metrics(yev, p_ev, thr),
                             params=json.dumps(entry["params"], ensure_ascii=False)))
            preds[method] = p_ev
            models[method] = (model, cols)

    res = pd.DataFrame(rows)
    zero = res.set_index("key")
    res["skill_vs_zero_cv"] = 1 - res.cv_mae / zero.loc["zero", "cv_mae"]  # 「常に0」よりMAEを何割減らせたか
    res["skill_vs_zero_eval"] = 1 - res.eval_mae / zero.loc["zero", "eval_mae"]
    res["eval_mean_pred"] = [float(np.mean(preds[k])) for k in res.key]
    res["eval_max_pred"] = [float(np.max(preds[k])) for k in res.key]
    res["eval_mean_actual"] = float(yev.mean())
    res["eval_max_actual"] = float(yev.max())
    res["gap_eval_minus_cv"] = res.eval_mae - res.cv_mae
    res["gap_eval_minus_train"] = res.eval_mae - res.train_mae
    res.to_csv(RESULTS / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(preds).to_csv(RESULTS / "predictions_eval.csv", index=False, encoding="utf-8-sig")
    sr = pd.DataFrame(seed_rows)
    sr.to_csv(RESULTS / "seed_robustness.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["name", "cv_mae", "train_mae", "eval_mae", "eval_rmse", "eval_r2", "auc_heavy", "ap_heavy",
            "pred_heavy_at20", "precision_at20", "recall_at20", "threshold_cv", "pred_heavy_atcv", "precision_atcv",
            "recall_atcv"]
    print("\n=== 評価データ（2025年の冬）の結果（seed 42） ===")
    print(res[show].round(3).to_string(index=False))
    print("\n過学習・偏りの確認:")
    print(res[["name", "train_mae", "cv_mae", "eval_mae", "gap_eval_minus_cv", "skill_vs_zero_cv", "skill_vs_zero_eval",
               "eval_mean_pred", "eval_max_pred", "eval_mean_actual", "eval_max_actual"]].round(3).to_string(index=False))
    print("\nCVの冬ごとのMAE [2022, 2023, 2024]:")
    for _, r in res.iterrows():
        print(f"  {r['name']}: {r.cv_fold_mae}  CV RMSE={r.cv_rmse:.3f} CV R2={r.cv_r2:.3f}")
    print("\n=== seed違い（Optunaのsampler・モデルのrandom_state） ===")
    print(sr.pivot_table(index="name", columns="seed", values=["cv_mae", "eval_mae"]).round(3).to_string())

    # --- 大雪日と、予測が大きく出た日の一覧（予測が1日遅れていないかの確認） ---
    pe = pd.DataFrame(preds)
    model_keys = list(METHODS)
    hd = pe[(pe.actual >= HEAVY) | (pe[model_keys].max(axis=1) >= 15)]
    hd.to_csv(RESULTS / "heavy_days_eval.csv", index=False, encoding="utf-8-sig")
    print("\n=== 評価データの大雪日（実測20cm以上）と、どれかの手法の予測が15cm以上だった日 ===")
    print(hd[["target_date", "actual", "persistence"] + model_keys].round(1).to_string(index=False))
    print("大雪日の予測の平均:", pe[pe.actual >= HEAVY][model_keys].mean().round(2).to_dict())

    # --- CVで最良の手法を選び、他と比べる（評価データでは選ばない） ---
    best_key = res[res.key.isin(model_keys)].sort_values("cv_mae").key.iloc[0]
    print(f"\nCV MAE が最小の手法: {best_key}")
    ci_rows = []
    pairs = [(m, b) for m in model_keys for b in ("persistence", "monthly_mean", "zero")]
    pairs += [(best_key, m) for m in model_keys if m != best_key]
    pairs += [("ridge", m) for m in model_keys if m not in ("ridge", best_key)]
    for a_key, b_key in dict.fromkeys(pairs):
        ci_rows.append(dict(A=a_key, B=b_key, **mae_diff_ci(yev, preds[a_key], preds[b_key])))
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(RESULTS / "bootstrap_mae_diff.csv", index=False, encoding="utf-8-sig")
    print("\n=== MAE差（A − B, 負ならAが良い）の95%信頼区間（評価データ、5000回） ===")
    print(ci.round(3).to_string(index=False))

    # --- 解釈: 線形モデルの標準化係数 ---
    coef_rows = []
    for key in ("ridge", "ridge_nohinge", "enet"):
        model, cols = models[key]
        reg = model.regressor_ if hasattr(model, "regressor_") else model
        coef = reg.named_steps["reg"].coef_
        scale = "log1p(cm)" if hasattr(model, "regressor_") else "cm"
        for c, v in zip(cols, coef):
            coef_rows.append(dict(model=key, feature=c, group=group_of(c), std_coef=v, target_scale=scale))
    coef = pd.DataFrame(coef_rows)
    coef["abs"] = coef.std_coef.abs()
    coef.sort_values(["model", "abs"], ascending=[True, False]).drop(columns="abs").to_csv(
        RESULTS / "linear_coefficients.csv", index=False, encoding="utf-8-sig")
    print("\n=== 線形モデルの標準化係数（特徴量を平均0・標準偏差1にしたときの係数）上位12 ===")
    for key in ("ridge", "enet"):
        c = coef[coef.model == key].sort_values("abs", ascending=False)
        print(f"[{key}] 目的変数の尺度: {c.target_scale.iloc[0]}  非ゼロ係数 {int((c.std_coef != 0).sum())}/{len(c)}")
        print(c.head(12)[["feature", "std_coef"]].round(3).to_string(index=False))

    # --- 解釈: 評価データ上の permutation importance ---
    pi_rows, gp_rows = [], []
    for key in ("ridge", "rf", "hgb", "svr"):
        model, cols = models[key]
        Xev = ev[cols]
        pi = permutation_importance(model, Xev, yev, scoring=neg_mae, n_repeats=50, random_state=0, n_jobs=-1)
        for c, m, s in zip(cols, pi.importances_mean, pi.importances_std):
            pi_rows.append(dict(model=key, feature=c, group=group_of(c), mae_increase=m, std=s))
        g = group_permutation(model, Xev, yev)
        g.insert(0, "model", key)
        gp_rows.append(g)
    pi = pd.DataFrame(pi_rows).sort_values(["model", "mae_increase"], ascending=[True, False])
    pi.to_csv(RESULTS / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    gp = pd.concat(gp_rows).sort_values(["model", "mae_increase"], ascending=[True, False])
    gp.to_csv(RESULTS / "permutation_importance_groups.csv", index=False, encoding="utf-8-sig")
    print("\n=== permutation importance（評価データで1列ずつ並べ替えたときのMAEの増加, cm）上位10 ===")
    for key in ("rf", "hgb", "svr", "ridge"):
        print(f"[{key}]")
        print(pi[pi.model == key].head(10)[["feature", "mae_increase", "std"]].round(3).to_string(index=False))
    print("\n=== グループ単位の permutation importance（同じグループの列をまとめて並べ替え） ===")
    print(gp.pivot_table(index="group", columns="model", values="mae_increase").round(3)
          .sort_values("hgb", ascending=False).to_string())


if __name__ == "__main__":
    main()
