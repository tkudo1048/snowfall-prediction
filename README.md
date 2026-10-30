# snowfall-prediction(降雪量予測・青森・弘前)

気象庁の観測データ（青森・弘前）をもとに、降雪量について調べ、翌日の降雪量を予測するモデルを作った実験です。
翌日の天気（晴・曇・雨・雪）を当てる別プロジェクトとは、リポジトリを分けています。

## 使い方
Python 3.13 で動作を確認しています。パッケージのバージョンは `requirements.txt` に固定しています。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd 05-snow-model-nwp
../.venv/bin/python fetch_openmeteo.py   # 数値予報を取得（保存済みは再取得しない）
```
各フォルダの手順は、それぞれの `README.md` にあります。

## フォルダ（番号は調べた順）
- `01-temp-snowfall/`: 月の平均気温と月の降雪量の関係。夏の気温・降水量と、その後の冬の降雪量の関係（12シーズン）
- `02-heavy-snow/`: 1日に20cm以上降る日（大雪日）は、ふつうの雪の日と何が違うか
- `03-snow-correlation/`: 日降雪量と、取得している全項目（69変数）との相関
- `04-snow-model/`: 翌日の降雪量を予測するモデル（観測データのみ、5冬）。Ridge・ElasticNet・RF・HGB・SVR を Optuna で調整
- `05-snow-model-nwp/`: 数値予報の値（Open-Meteo の気象庁 GSM/MSM）を加えたモデル（10冬）。S0/SA/SB の3段階で比較
- `data/`: 降雪の調査だけで使うデータ
  - `monthly/`: 気象庁の月別値（2014〜2026年）。`01-temp-snowfall/fetch_monthly.py` で取得
  - `long/`: 気象庁の日別値（2016〜2026年）。`05-snow-model-nwp/fetch_jma_long.py` で取得
  - `openmeteo/`: Open-Meteo の数値予報（CC BY 4.0、非商用）。`05-snow-model-nwp/fetch_openmeteo.py` で取得
- `slides/`: 予測の仕方をまとめたスライド（`snow-forecast-method.pdf`、元のHTMLは `src/`）

## 共通で使うもの
- `lib/`: 気象庁の日別データの取得（`download_jma_data.py`）と数値化（`preprocess.py`）
- `data/raw/`: 2021-08〜2026-08 の日別データ（`aomori_daily.csv`, `hirosaki_daily.csv`）
- Python: リポジトリ直下の `.venv`（各フォルダで `../.venv/bin/python <スクリプト>` のように実行する）

## 結果の要点
- 月単位では「寒い月ほど雪が多い」（相関 −0.8 前後）。前の夏の気温・降水量とは関係が見えない
- 大雪日は気温が低く、湿度が高く、日が差さない日。風の強さは差がない
- 翌日の降雪量の予測は、単純な予測より外れが小さくなるが、今夜わかる情報だけでは大雪（20cm以上）をほとんど当てられない
- 大雪を当てる力は、ほぼ「明日の数値予報」から来ていた（詳しくは `05-snow-model-nwp/README.md`）

## 出典
- 気象庁「過去の気象データ」（青森：気象官署、弘前：アメダス）を加工して作成
- 数値予報：Open-Meteo（気象庁 GSM・MSM の予報値を加工して作成）。データは CC BY 4.0、無料版は非商用での利用が条件です
- 数値予報は、各予報の発表直後の部分をつなぎ合わせたもので、「前日に出ていた予報」そのものではありません（詳しくは `05-snow-model-nwp/README.md`）
