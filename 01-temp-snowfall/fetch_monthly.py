"""気象庁「過去の気象データ検索」の月別値ページから、月平均気温・月降水量・月降雪量合計を取得する（青森・弘前）。

- 1年分（12か月）を1ページで取れるので、1地点1年1リクエスト。待機時間を入れる。
- 値の後ろの記号: ")" 準正常値は採用、"]" 資料不足値・"///" "×" "--"(現象なし) などは flag 列に残す。
  "--" は降雪0として扱う。
- 出力: ../data/monthly/{aomori,hirosaki}_monthly.csv
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "monthly"  # data/monthly
URL = "https://www.data.jma.go.jp/stats/etrn/view/{page}.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (research use; JMA monthly data)"}
# page, block_no, 気温(日平均の月平均)の列位置, 降雪合計の列位置, 降水量合計の列位置
STATIONS = {"aomori": ("monthly_s1", "47575", 7, 21, 3), "hirosaki": ("monthly_a1", "0166", 5, 18, 1)}


def parse(v: str):
    v = v.strip()
    if v == "--":
        return 0.0, "none"
    m = re.match(r"^(-?[\d.]+)\s*([)\]]?)$", v)
    if not m:
        return None, v or "blank"
    return float(m.group(1)), {"": "", ")": "quasi", "]": "insufficient"}[m.group(2)]


def fetch(name, year):
    page, block, ti, si, pi = STATIONS[name]
    r = requests.get(URL.format(page=page), headers=HEADERS, timeout=30,
                     params=dict(prec_no=31, block_no=block, year=year, month="", day="", view=""))
    r.raise_for_status()
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.S):
        c = [re.sub("<[^>]+>", "", x).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if c and c[0].isdigit() and len(c) > si:
            t, tf = parse(c[ti])
            s, sf = parse(c[si])
            p, pf = parse(c[pi])
            out.append(dict(year=year, month=int(c[0]), temp_mean=t, temp_flag=tf, snowfall_total=s, snow_flag=sf,
                            precip_total=p, precip_flag=pf))
    return out


if __name__ == "__main__":
    y0, y1 = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (2014, 2026)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in STATIONS:
        rows = []
        for y in range(y0, y1 + 1):
            rows += fetch(name, y)
            time.sleep(1.5)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{name}_monthly.csv", index=False, encoding="utf-8-sig")
        print(name, len(df), "months;", df.temp_flag.value_counts().to_dict(), df.snow_flag.value_counts().to_dict(), df.precip_flag.value_counts().to_dict())
