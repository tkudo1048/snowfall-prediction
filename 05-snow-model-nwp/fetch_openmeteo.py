"""Open-Meteo Historical Forecast API から、気象庁の数値予報モデル（GSM・MSM）の毎時値を取得する。

出典: Open-Meteo（https://open-meteo.com/）, 元データは気象庁 GSM（全球モデル）・MSM（メソモデル）。
ライセンス: CC BY 4.0（非商用利用）。

- このAPIの値は、各予報回の「最初の数時間」をつないだ時系列。ほぼ実況（解析値に近い値）で、
  「前日に出た明日の予報」ではない。
- 上空の値（850hPa・700hPa・500hPa）は GSM のみ（MSMはAPI上で上空がnull）。地上の値は GSM と MSM の両方を取る。
- 呼び出しは「冬ごと（10/25〜翌4/1）× 地点 × モデル」の単位にまとめる（青森・上流2点で66回、弘前の追加で22回、計88回）。
  呼び出しの間に WAIT_SEC 秒待つ。保存済みのファイルは再取得しない。
- 保存先: data/openmeteo/<地点>_<モデル>_<開始日>_<終了日>.json（APIの応答そのまま）
- 呼び出し回数は data/openmeteo/api_calls.log に追記する。

実行: ../.venv/bin/python fetch_openmeteo.py
"""
import json
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]  # リポジトリ直下
OUT = ROOT / "data" / "openmeteo"
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WAIT_SEC = 6.0

POINTS = {
    "aomori": (40.82, 140.77),   # 青森（気象台付近）
    "hirosaki": (40.61, 140.46),  # 弘前（アメダス 0166 付近）。GSMは最寄りの0.5度格子点（APIが選ぶ）
    "up_w": (40.50, 139.50),     # 上流（西）: 日本海、男鹿半島の北
    "up_nw": (41.30, 139.90),    # 上流（北西）: 日本海、津軽半島の西
}
GSM_UPPER = ["temperature_850hPa", "relative_humidity_850hPa", "wind_speed_850hPa", "wind_direction_850hPa",
             "relative_humidity_700hPa", "temperature_500hPa", "geopotential_height_500hPa"]
SURFACE = ["temperature_2m", "precipitation", "snowfall", "pressure_msl"]
VARS = {
    ("aomori", "jma_gsm"): GSM_UPPER + SURFACE,
    ("aomori", "jma_msm"): SURFACE,
    ("hirosaki", "jma_gsm"): GSM_UPPER + SURFACE,
    ("hirosaki", "jma_msm"): SURFACE,
    ("up_w", "jma_gsm"): GSM_UPPER[:5] + ["precipitation"],
    ("up_w", "jma_msm"): ["precipitation"],
    ("up_nw", "jma_gsm"): GSM_UPPER[:5] + ["precipitation"],
    ("up_nw", "jma_msm"): ["precipitation"],
}


def windows():
    """取得期間: 2016-01-01〜2016-04-01（GSMの提供開始から）と、各冬の 10/25〜翌4/1（2016〜2025年の冬）。"""
    out = [("2016-01-01", "2016-04-01")]
    for y in range(2016, 2026):
        out.append((f"{y}-10-25", f"{y + 1}-04-01"))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_call = n_cache = 0
    for start, end in windows():
        for (point, model), hourly in VARS.items():
            path = OUT / f"{point}_{model}_{start}_{end}.json"
            if path.exists():
                n_cache += 1
                continue
            lat, lon = POINTS[point]
            params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=",".join(hourly),
                          models=model, timezone="Asia/Tokyo")
            resp = requests.get(URL, params=params, timeout=60)
            n_call += 1
            with open(OUT / "api_calls.log", "a") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{resp.status_code}\t{point}\t{model}\t"
                        f"{start}\t{end}\t{len(hourly)}vars\n")
            if resp.status_code != 200:
                print(f"失敗 {resp.status_code}: {point} {model} {start}〜{end}: {resp.text[:200]}")
                if resp.status_code == 429:
                    print("回数制限に達したため中断します。")
                    return
            else:
                path.write_text(resp.text)
                print(f"取得: {point} {model} {start}〜{end}")
            time.sleep(WAIT_SEC)
    print(f"API呼び出し {n_call}回 / 保存済みを利用 {n_cache}件")


if __name__ == "__main__":
    main()
