"""Open-Meteo の GSM・MSM の毎時値のうち、どの時刻が「モデル本来の値」で、どの時刻が補間値かを調べる。

1. 候補の本来の時刻（6時間ごと・3時間ごと、ずらし方を全通り）から Catmull-Rom 補間で他の時刻を作り、実際の値との差を見る。
   差が丸め誤差程度になる候補が、本来の時刻。
2. GSM の t日16〜20時の850hPa気温を、前後の本来の値（t日09・15・21時、t+1日03時）で回帰し、
   t+1日03時の係数が Catmull-Rom の理論値と一致するか（= 翌日03時の値が混ざっているか）を見る。
3. GSM の降水量・降雪量が、どの時刻で値が切り替わる「かたまり」になっているか。降雪量がどの時刻の気温で決まるか。

実行: ../.venv/bin/python check_gsm_native_hours.py
"""
import numpy as np
import pandas as pd

from common import read_raw


def cr_err(s, step, off):
    v, hrs = s.to_numpy(), s.index.hour.to_numpy()
    k = np.where((hrs - off) % step == 0)[0]
    errs = []
    for j in range(1, len(k) - 2):
        i0, i1, i2, i3 = k[j - 1], k[j], k[j + 1], k[j + 2]
        if i2 - i1 != step:
            continue
        P0, P1, P2, P3 = v[i0], v[i1], v[i2], v[i3]
        for m in range(1, step):
            t = m / step
            pred = 0.5 * (2 * P1 + (-P0 + P2) * t + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t ** 2
                          + (-P0 + 3 * P1 - 3 * P2 + P3) * t ** 3)
            errs.append(abs(pred - v[i1 + m]))
    e = np.array(errs)
    return float(np.nanmean(e))


def main():
    print("1. 本来の時刻の候補ごとの補間誤差の平均（小さい順に3つ。「6h+3」= 03・09・15・21時）")
    for point, model in (("aomori", "jma_gsm"), ("aomori", "jma_msm"), ("hirosaki", "jma_gsm"), ("hirosaki", "jma_msm"),
                         ("up_nw", "jma_gsm")):
        h = read_raw(point, model)
        for c in h.columns:
            if c in ("precipitation", "snowfall", "wind_direction_850hPa"):
                continue
            s = h[c].iloc[: 24 * 400]
            res = {f"{st}h+{o}": cr_err(s, st, o) for st, offs in ((6, range(6)), (3, range(3))) for o in offs}
            best = sorted(res.items(), key=lambda x: x[1])[:3]
            print(f"  {point} {model} {c}: " + ", ".join(f"{k} {v:.3f}" for k, v in best))

    print("\n2. GSM 850hPa気温（青森）の t日h時 = a·(09時) + b·(15時) + c·(21時) + d·(t+1日03時) の回帰")
    g = read_raw("aomori", "jma_gsm")["temperature_850hPa"]
    for hr in (16, 17, 18, 19, 20):
        t = hr / 6 - 15 / 6  # 15時と21時の間の位置（0〜1）
        theory = 0.5 * (-t ** 2 + t ** 3)
        y = g[g.index.hour == hr]
        day = y.index.normalize()
        X = np.column_stack([g.reindex(day + pd.Timedelta(hours=x)).to_numpy() for x in (9, 15, 21, 27)])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y.to_numpy())
        coef, *_ = np.linalg.lstsq(X[ok], y.to_numpy()[ok], rcond=None)
        print(f"  {hr}時: 係数 {np.round(coef, 3).tolist()}  t+1日03時の理論値 {theory:.3f}  (n={ok.sum()})")

    print("\n3. GSM の降水量・降雪量: 前の時刻から値が変わる割合（時刻はAPIの表記 = その時刻までの1時間）")
    h = read_raw("aomori", "jma_gsm")
    for c in ("precipitation", "snowfall"):
        s = h[c]
        ch = ((s.diff() != 0) & s.notna() & s.shift().notna()).groupby(s.index.hour).mean()
        print(f"  {c}: " + " ".join(f"{k}:{v:.2f}" for k, v in ch.items()))
    b = pd.DataFrame({"p": h.precipitation, "s": h.snowfall, "t": h.temperature_2m})
    e = b[b.index.hour == 21]
    e = e.assign(t15=b.t.reindex(e.index - pd.Timedelta(hours=6)).to_numpy(),
                 t03=b.t.reindex(e.index + pd.Timedelta(hours=6)).to_numpy())
    e = e[e.p > 0]
    snowy = (e.s > 0).to_numpy()
    for name, col in (("21時（かたまりの終わり）", e.t), ("15時", e.t15), ("翌03時", e.t03)):
        acc = max(((col.to_numpy() < th) == snowy).mean() for th in np.arange(-3, 3, 0.1))
        print(f"  15〜21時のかたまりの降雪あり/なしを、{name}の地上気温の閾値1つで当てた割合: {acc:.4f}")


if __name__ == "__main__":
    main()
