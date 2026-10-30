"""特徴量セット × 手法ごとに Optuna（TPESampler, seed固定）で同じ試行数だけチューニングする。

- 学習データ（2015〜2023年の冬）内の、冬単位の前向きCV（検証 = 2019〜2023年の冬、5分割）だけを使う。
  評価データ（2024・2025年の冬）はここでは一切使わない。
- 回帰（--task reg）: 5つの検証冬の MAE の平均を最小化。
- 大雪の分類（--task clf）: 5つの検証冬のCV予測をまとめた average precision（AP）を最大化。
  大雪日が1日しかない冬もあるため、冬ごとのAPの平均ではなく、まとめたCV予測で1つのAPを計算する。
- 大雪の分類の参考（--task clf_ref、弘前のみ）: 閾値を15cm（学習データの95パーセンタイル）にした分類。主は20cm（--task clf）。
- 対象地点は --target hirosaki|aomori（既定は弘前）。
- 出力: results/<地点>/trials/<task>_<set>_<method>_seed<seed>.csv（全試行）, results/<地点>/best/<同じ名前>.json

実行例: ../.venv/bin/python tune.py --target hirosaki --task reg --set SA --method hgb --seed 42 --trials 100
"""
import argparse
import json
import time

import numpy as np
import optuna
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (CLF_METHODS, REG_METHODS, RESULTS, SETS, build_dataset, cv_folds, cv_predict, feature_cols, mae,
                    split, suggest_clf, suggest_reg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reg", "clf", "clf_ref"], required=True)
    ap.add_argument("--set", choices=list(SETS), required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=100)
    a = ap.parse_args()
    assert a.method in (REG_METHODS if a.task == "reg" else CLF_METHODS)

    df = build_dataset()
    train, _ = split(df)
    folds = cv_folds(train)
    X, y = train[feature_cols(df, a.set)], train.target
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tag = f"{a.task}_{a.set}_{a.method}_seed{a.seed}"

    def objective(trial):
        if a.task == "reg":
            p = suggest_reg(trial, a.method)
            res = cv_predict("reg", a.method, p, X, y, folds, a.seed)
            s = [mae(t, q) for t, q in res]
            trial.set_user_attr("fold", [round(v, 4) for v in s])
            return float(np.mean(s))
        p = suggest_clf(trial, a.method)
        res = cv_predict(a.task, a.method, p, X, y, folds, a.seed)
        yt, pp = np.concatenate([t for t, _ in res]), np.concatenate([q for _, q in res])
        trial.set_user_attr("fold", [round(average_precision_score(t, q), 4) if t.sum() else None for t, q in res])
        trial.set_user_attr("auc", round(roc_auc_score(yt, pp), 4))
        return float(average_precision_score(yt, pp))

    study = optuna.create_study(direction="minimize" if a.task == "reg" else "maximize",
                                sampler=optuna.samplers.TPESampler(seed=a.seed))
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials)
    b = study.best_trial
    out = dict(task=a.task, set=a.set, method=a.method, seed=a.seed, params=b.params, cv_score=b.value,
               cv_metric="mean fold MAE" if a.task == "reg" else "pooled AP", fold=b.user_attrs["fold"],
               cv_auc=b.user_attrs.get("auc"), trial=b.number, n_trials=a.trials, seconds=round(time.time() - t0, 1),
               n_features=X.shape[1])
    (RESULTS / "trials").mkdir(parents=True, exist_ok=True)
    (RESULTS / "best").mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(RESULTS / "trials" / f"{tag}.csv", index=False)
    (RESULTS / "best" / f"{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{tag}: CV={b.value:.4f} fold={b.user_attrs['fold']} #{b.number} {b.params} ({time.time() - t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
