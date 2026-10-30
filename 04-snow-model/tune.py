"""各手法を Optuna（TPESampler）で同じ試行数だけチューニングする。

- 目的: 学習データ（2021〜2024年の冬）内の、冬単位の前向きCV（3分割）の平均MAEを最小化。
- 評価データ（2025年の冬）はここでは一切使わない。
- 出力: results/trials/<method>_seed<seed>.csv（全試行）、results/best_params_seed<seed>.json

実行: ../.venv/bin/python tune.py --seed 42 --trials 100
"""
import argparse
import json
import time

import numpy as np
import optuna

from common import METHODS, RESULTS, build_dataset, cv_folds, feature_cols, mae, make_model, predict, split, suggest


def cv_score(method, params, X, y, folds, seed, n_jobs):
    scores = []
    for tr_idx, va_idx in folds:
        m = make_model(method, params, seed=seed, n_jobs=n_jobs)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        scores.append(mae(y.iloc[va_idx], predict(m, X.iloc[va_idx])))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    args = ap.parse_args()

    df = build_dataset()
    train, _ = split(df)
    folds = cv_folds(train)
    (RESULTS / "trials").mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best = {}
    for method in args.methods:
        X, y = train[feature_cols(df, hinge=METHODS[method][1])], train.target

        def objective(trial):
            p = suggest(trial, method)
            s = cv_score(method, p, X, y, folds, args.seed, args.n_jobs)
            trial.set_user_attr("fold_mae", [round(v, 4) for v in s])
            return float(np.mean(s))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
        t0 = time.time()
        study.optimize(objective, n_trials=args.trials)
        b = study.best_trial
        best[method] = dict(params=b.params, cv_mae=b.value, fold_mae=b.user_attrs["fold_mae"], trial=b.number,
                            seconds=round(time.time() - t0, 1))
        study.trials_dataframe().to_csv(RESULTS / "trials" / f"{method}_seed{args.seed}.csv", index=False)
        print(f"[seed {args.seed}] {method}: CV MAE={b.value:.3f} folds={b.user_attrs['fold_mae']} "
              f"#{b.number} {b.params} ({time.time() - t0:.0f}s)", flush=True)

    path = RESULTS / f"best_params_seed{args.seed}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(best)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
