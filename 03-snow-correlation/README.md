# 日降雪量と、取得している全データとの相関

青森・弘前の日別データ（2021年11月〜2026年3月の冬、11〜3月の756日）について、日降雪量と全項目（69変数）の相関を調べた。
10分値（amedas_10min）は2026年9月下旬以降しかなく冬のデータがないため使っていない。

## ファイル
- analyze.py: 相関の計算（結果は corr_aomori.csv, corr_hirosaki.csv）
- plot.py: 図の作成
- corr_aomori.csv: 青森の日降雪量との相関（弘前の日降雪量が目的変数のものは corr_hirosaki.csv）
- images/fig1-correlation-ranking.png: 相関が強い項目（全体・月内・気温の影響を除いた後）
- images/fig2-heatmap.png: 主な項目どうしの相関
- images/fig3-binned.png: 値の帯ごとの降雪量と降雪日率

## 実行
```
cd 03-snow-correlation
../.venv/bin/python analyze.py
../.venv/bin/python plot.py
```

## 使った指標（corr_*.csv の列）
- pearson: 直線の相関
- spearman: 順位の相関（直線でなくてもよい）。ci_lo/ci_hi は7日ブロックのブートストラップによる95%信頼区間
- spearman_within_month: 月ごとの平均を引いてから計算（季節の影響を除く）
- spearman_snowdays_only: 降雪のあった日だけで計算（降った日の量の違い）
- occurrence_auc: 降雪あり日と降雪なし日の比較（0.5で差なし、1に近いほど降雪日の方が値が大きい）
- partial_after_temp: 青森の日平均気温で説明できる分を引いた後の相関
- mutual_info: 相互情報量（曲線や山形の関係も拾う）
- q: 多重検定（Benjamini-Hochberg）の補正後p値。68変数中64個が q<0.05
- kind: 降水量・積雪・天気の雪など、降雪の「結果」として決まる項目は「結果」、それ以外は「原因候補」

## 限界
- 日別の値で、低気圧の通過や寒気の流入など時間単位の動きは見ていない
- 冬は5シーズン分。日は連続していて独立ではない（信頼区間はブロック法で補った）
- 相関は因果ではなく、変数どうしも強く連動している（青森と弘前の気温は0.99）
