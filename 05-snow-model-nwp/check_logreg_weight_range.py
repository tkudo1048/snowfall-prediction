"""確認用（CVのみ。評価データは使わない）: ロジスティック回帰のクラス重み pos_weight の探索上限を50→500、
C の下限を1e-4→1e-5 に広げて、seed 42・100試行でCVのAPがどう変わるかを見る。
本番の tune.py では S0・SA の最良 pos_weight が上限50付近だったため。

実行: ../.venv/bin/python -W ignore check_logreg_weight_range.py
"""
import sys, numpy as np, optuna
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import *
from sklearn.metrics import average_precision_score, roc_auc_score
df = build_dataset(); train, _ = split(df); folds = cv_folds(train)
optuna.logging.set_verbosity(optuna.logging.WARNING)
for fset in ("S0", "SA", "SB"):
    X, y = train[feature_cols(df, fset)], train.target
    def obj(t):
        p = {"pos_weight": t.suggest_float("pos_weight", 1.0, 500.0, log=True), "C": t.suggest_float("C", 1e-5, 1e2, log=True)}
        r = cv_predict("clf", "logreg", p, X, y, folds, 42)
        return average_precision_score(np.concatenate([a for a, _ in r]), np.concatenate([b for _, b in r]))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)); st.optimize(obj, n_trials=100)
    print(fset, round(st.best_value, 4), st.best_params, flush=True)
