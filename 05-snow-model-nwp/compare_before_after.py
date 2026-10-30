"""修正前（GSMの補間値を使っていた版、results_v1_before_fix/）と修正後（results/aomori/）の主な数字を並べる。青森のみ。

出力: results/aomori/compare_before_after.csv
実行: ../.venv/bin/python compare_before_after.py   （evaluate.py の後）
"""
import os

import pandas as pd

os.environ["SNOW_TARGET"] = "aomori"  # 修正前の結果は青森を対象にしたものだけ
from common import HERE, RESULTS  # noqa: E402

OLD = HERE / "results_v1_before_fix"


def main():
    rows = []
    for kind, fn, cols in (
        ("回帰", "metrics_reg.csv", ["cv_mae", "eval_mae", "eval_rmse", "eval_r2", "eval_heavy_auc", "eval_heavy_ap",
                                    "hit_at20", "pred_pos_atcv", "hit_atcv", "miss_atcv"]),
        ("分類", "metrics_clf.csv", ["cv_auc", "cv_ap", "eval_auc", "eval_ap", "thr_atcv", "pred_pos_atcv", "hit_atcv",
                                    "miss_atcv"]),
    ):
        o = pd.read_csv(OLD / fn).set_index("key")
        n = pd.read_csv(RESULTS / fn).set_index("key")
        for key in n.index:
            for c in cols:
                rows.append(dict(kind=kind, key=key, metric=c, before=o.loc[key, c] if key in o.index else None,
                                 after=n.loc[key, c]))
    ob = pd.read_csv(OLD / "bootstrap.csv")
    nb = pd.read_csv(RESULTS / "bootstrap.csv")
    on = ["task", "metric", "method", "A", "B"]
    m = ob.merge(nb, on=on, suffixes=("_before", "_after"))
    for _, r in m.iterrows():
        for c in ("diff", "ci_lo_block7", "ci_hi_block7"):
            rows.append(dict(kind="区間", key=f"{r.task} {r.metric} {r.method} {r.B}-{r.A}", metric=c,
                             before=r[f"{c}_before"], after=r[f"{c}_after"]))
    out = pd.DataFrame(rows)
    out["change"] = pd.to_numeric(out.after, errors="coerce") - pd.to_numeric(out.before, errors="coerce")
    out.to_csv(RESULTS / "compare_before_after.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    for kind in ("回帰", "分類"):
        t = out[out.kind == kind]
        print(f"=== {kind}（修正前 → 修正後） ===")
        w = t.pivot_table(index="key", columns="metric", values=["before", "after"], aggfunc="first")
        for key in t.key.unique():
            parts = [f"{c} {w.loc[key, ('before', c)]:.3f}→{w.loc[key, ('after', c)]:.3f}"
                     for c in t[t.key == key].metric]
            print(f"  {key}: " + ", ".join(parts))
    print("=== 区間（7日ブロック、修正前 → 修正後） ===")
    t = out[out.kind == "区間"].pivot_table(index="key", columns="metric", values=["before", "after"], aggfunc="first")
    for key in t.index:
        b = t.loc[key]
        print(f"  {key}: {b[('before', 'diff')]:+.3f} [{b[('before', 'ci_lo_block7')]:+.3f}, {b[('before', 'ci_hi_block7')]:+.3f}]"
              f" → {b[('after', 'diff')]:+.3f} [{b[('after', 'ci_lo_block7')]:+.3f}, {b[('after', 'ci_hi_block7')]:+.3f}]")


if __name__ == "__main__":
    main()
