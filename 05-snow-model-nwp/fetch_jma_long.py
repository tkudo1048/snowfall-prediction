"""気象庁の日別データ（青森・弘前）を 2016-01〜2026-08 の範囲でそろえる。

- 保存先: data/long/（data/raw/ の既存ファイルは書き換えない）
- data/raw/ にある 2021-08 以降の月別CSVは、data/long/ にコピーして再利用する（通信しない）。
- 足りない月だけを lib/download_jma_data.py の download_station（内部で fetch_month）で取得する。
  待機時間は既存スクリプトと同じ SLEEP_SEC（1.5秒）。すでに long/ にある月は再取得しない。
- 出力: data/long/{aomori,hirosaki}_YYYYMM.csv と、結合した {aomori,hirosaki}_daily.csv

実行: ../.venv/bin/python fetch_jma_long.py
"""
import os
import shutil
import sys
from datetime import date
from pathlib import Path

SNOW = Path(__file__).resolve().parents[1]  # リポジトリ直下
ROOT = SNOW
sys.path.insert(0, str(ROOT / "lib"))
from download_jma_data import STATIONS, download_station, month_range  # noqa: E402

RAW = ROOT / "data" / "raw"
LONG = SNOW / "data" / "long"
START, END = date(2016, 1, 1), date(2026, 8, 1)


def main():
    os.chdir(SNOW)  # ログにローカルの絶対パスを出さないよう、リポジトリ直下からの相対パスで扱う
    LONG.mkdir(parents=True, exist_ok=True)
    n_copy = 0
    for name in STATIONS:
        for d in month_range(START, END):
            fn = f"{name}_{d.year}{d.month:02d}.csv"
            if (RAW / fn).exists() and not (LONG / fn).exists():
                shutil.copy2(RAW / fn, LONG / fn)
                n_copy += 1
    print(f"data/raw からコピーした月別CSV: {n_copy}")
    for name, info in STATIONS.items():
        # refetch_months=0: long/ にある月は取り直さない（確定値の取り直しはしない）
        download_station(name, info["page"], info["prec_no"], info["block_no"], START, END,
                         out_dir=LONG.relative_to(SNOW), refetch_months=0, refresh=False)


if __name__ == "__main__":
    main()
