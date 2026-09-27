"""Tải và dựng dữ liệu XAUUSD cho nghiên cứu (chỉ dùng nguồn public trên GitHub).

- data/old_m15.pkl : XAUUSD M15 2012-05 -> 2022-03, giờ server MT5 (ejtraderLabs/historical-data)
- data/nav_m1.pkl  : XAUUSD M1 2025-10 -> hiện tại, UTC (TradingView FX:XAUUSD), ghép từ lịch sử
                     commit hằng ngày của navtemmt/xauusd-autodata (mỗi commit giữ ~3,5 ngày M1)

Chạy: python fetch_data.py
"""
import glob
import io
import os
import subprocess

import pandas as pd

D = "data"


def sh(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True).stdout


def clone(url, dest, full=False):
    if os.path.isdir(dest):
        return
    args = ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url, dest]
    if not full:
        args[3:3] = ["--depth", "1"]
    sh(*args)


def build_old():
    repo = os.path.join(D, "historical-data")
    clone("https://github.com/ejtraderLabs/historical-data.git", repo)
    raw = sh("git", "show", "HEAD:XAUUSD/XAUUSDm15.csv", cwd=repo)
    o = pd.read_csv(io.BytesIO(raw), parse_dates=["Date"]).set_index("Date")
    o[["open", "high", "low", "close"]] /= 100  # file lưu giá x100
    o.to_pickle(os.path.join(D, "old_m15.pkl"))
    print("old_m15", o.index.min(), o.index.max(), len(o))


def build_new():
    repo = os.path.join(D, "xauusd-autodata")
    clone("https://github.com/navtemmt/xauusd-autodata.git", repo, full=True)
    # cũ -> mới, để khi trùng thời điểm thì giữ giá của snapshot mới nhất (nến cuối snapshot cũ có thể chưa đóng)
    commits = sh("git", "log", "--reverse", "--format=%h", "--", "data/XAUUSD_data.csv", cwd=repo).decode().split()
    frames = []
    for c in commits:
        try:
            raw = sh("git", "show", f"{c}:data/XAUUSD_data.csv", cwd=repo)
        except subprocess.CalledProcessError:
            continue
        if not raw.strip():
            continue
        d = pd.read_csv(io.BytesIO(raw))
        if "OANDA" in str(d["symbol"].iloc[0]):
            continue
        frames.append(d)
    nav = pd.concat(frames)
    nav["datetime"] = pd.to_datetime(nav["datetime"])
    nav = (nav.drop_duplicates("datetime", keep="last").set_index("datetime").sort_index()
           [["open", "high", "low", "close", "volume"]])
    nav.to_pickle(os.path.join(D, "nav_m1.pkl"))
    print("nav_m1", nav.index.min(), nav.index.max(), len(nav))


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    os.makedirs("out", exist_ok=True)  # các script run_*.py ghi kết quả vào out/
    build_old()
    build_new()
