# 月平均気温と月合計降雪量の関係調査

気象庁の日別データ（青森・弘前、2021年8月〜2026年8月）から、月の平均気温と月の合計降雪量の関係を調べた。

## ファイル
- analyze.py: 月単位の集計と相関（ピアソン・スピアマン）、気温帯別の平均
- plot.py: 散布図の作成
- monthly.csv: 月別の平均気温・合計降雪量（全日そろった61か月×2地点）
- images/fig1-temp-vs-snowfall.png: 散布図（月別の色分け）
- fetch_monthly.py: 気象庁の月別値（月平均気温・月降雪量）を2014〜2026年分取得 → ../data/monthly/（data/monthly）
- summer.py: 夏の平均気温と、その後の冬の合計降雪量の関係（12シーズン）→ summer_vs_winter.csv
- rain.py: 降水量と降雪量の関係（前の夏・その冬の降水量、気温と「降雪量÷降水量」）→ rain_vs_winter.csv
- images/fig4-rain-vs-snow.png: 降水量と降雪量の散布図
- images/fig3-summer-vs-winter.png: 夏の気温・冬の気温と冬の降雪量の散布図
- images/fig2-by-season.png: シーズン別の散布図（シーズン＝11月〜翌3月。5か月そろった5シーズン）

## 実行
```
cd 01-temp-snowfall
../.venv/bin/python analyze.py
../.venv/bin/python plot.py
```
入力はdata/raw/{aomori,hirosaki}_daily.csv と data/monthly/。日別の数値化は lib/preprocess.py を使う。

## 結果の要約
- 相関係数（ピアソン）: 全月 青森 −0.72・弘前 −0.72、冬(11〜3月) 青森 −0.81・弘前 −0.83
- 気温が約2℃を境に降雪量が大きく変わる。5℃を超えるとほぼ0cm
- 冬の各月は5年分（1か月5点）なので、月別の相関は参考程度
- 相関は因果ではない（冬という季節が両方に効いている）
