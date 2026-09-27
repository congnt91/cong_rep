"""Xuất dữ liệu từng lệnh ra CSV để phân tích ngoài (Excel, Python...). Kết quả: out/export/*.csv

- lenh_limit.csv       : lệnh Limit tại mốc như Nola-X (RR >= 1, TP mốc kế tiếp, SL 0,75 ATR), kèm đặc trưng lúc khớp
- lenh_xac_nhan.csv    : lệnh xác nhận nến M15 + chốt 50% tại 1R (cấu hình đề xuất Phần 2; cột atr_rel >= 1 là bộ lọc ATR)
- cham_moc.csv         : mọi lần chạm mốc (Phần 5), mốc thật và mốc giả, Limit và xác nhận M15, kèm độ bật/độ ngược

Giờ: thoi_gian_server = giờ server MT5 (New York + 7); gio_vn = giờ Việt Nam.
R = khoảng SL của lệnh. ATR = ATR(14) H1 tại lúc đặt lệnh. USD tính cho 0,01 lot (1 oz).
"""
import os

import numpy as np
import pandas as pd

import bounce as bo
import fxlevels as fx
import improve as im
from run_bounce import R_ATR
from run_combo import build
from run_hours import FAMILIES, WINDOWS, window_of
from run_improve import period_of

OUT = "out/export"
FAM_OF = {t: f for f, ts in FAMILIES.items() for t in ts}
WIN_LABEL = {k: f"{a:02d}-{b:02d}h {lab}" for k, lab, a, b, _ in WINDOWS}


def vn_time(server_times):
    """Giờ server (NY + 7) -> giờ Việt Nam."""
    ny = pd.DatetimeIndex(server_times) - pd.Timedelta("7h")
    ny = ny.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    return ny.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)


def common(df, tcol="fill_t"):
    T = pd.DatetimeIndex(df[tcol].values)
    out = pd.DataFrame({
        "giai_doan": period_of(T),
        "thoi_gian_server": T,
        "gio_vn": vn_time(T),
        "khung_gio": [WIN_LABEL.get(w, "00-01h") for w in window_of(T.hour.values)],
        "loai_moc": df["type"].values,
        "nhom_moc": [FAM_OF.get(t, "Mốc giả") for t in df["type"].values],
        "huong": np.where(df["dir"].values == 1, "Buy", "Sell"),
        "gia_moc": df["level"].values.round(2),
        "atr_h1": df["atr"].values.round(3),
    })
    return out


def trades(base, ctx, **kw):
    s = build(base, ctx, **kw)
    out = common(s)
    out["gia_vao"] = s["entry"].values.round(2)
    out["sl_usd"] = s["risk"].values.round(2)
    out["tp_usd"] = s["tpd_struct"].values.round(2)
    out["rr"] = s["rr_struct"].values.round(2)
    out["so_moc_gop"] = s["confluence"].values
    for c in fx.REGIMES:
        out[c] = s[c].values
    out["atr_rel"] = s["atr_rel"].values.round(3)
    out["rsi_h1_theo_huong"] = s["rsi_dir"].values.round(1)
    out["cach_ema20_atr"] = s["stretch"].values.round(3)
    out["tiep_can_1h_atr"] = s["approach"].values.round(3)
    out["bien_do_ngay_da_dung"] = s["day_used"].values.round(3)
    out["da_cham_truoc"] = s["touched_before"].values
    out["gio_cho_khop"] = s["hrs_to_fill"].values.round(2)
    out["thoi_gian_dong"] = pd.DatetimeIndex(s["exit_t"].values)
    out["ket_qua_R"] = s["R"].values.round(4)
    out["ket_qua_usd"] = (s["R"].values * s["risk"].values).round(2)
    return out


def touches(base, ctx, o, po):
    rows = []
    for name, oo, entry in [("real", o, "limit"), ("pseudo", po, "limit"), ("confirm", o, "confirm"), ("confirm_pseudo", po, "confirm")]:
        tr = bo.touches(base, oo, entry).reset_index(drop=True)
        P = bo.paths(base, tr, at_level=entry == "limit")
        A = tr.atr.values
        d = common(tr)
        d.insert(0, "bo", name)
        d["gia_vao"] = tr["entry"].values.round(2)
        for y in (0.5, 0.75, 1.0):
            d[f"bat_truoc_sl{int(y * 100):03d}_atr"] = bo.mfe_before(P, y).round(4)
        _, code, _, _ = bo.evaluate(P, np.zeros(len(tr)) + np.inf, R_ATR * A, spread=0.0)
        d["cham_sl075_24h"] = code == -1
        for x in (0.05, 0.1, 0.2):
            mae, ok = bo.mae_before(P, np.full(len(tr), x))
            d[f"nguoc_truoc_tp{int(x * 100):03d}_atr"] = mae.round(4)
            d[f"bat_du_tp{int(x * 100):03d}_24h"] = ok
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = {"lenh_limit": [], "lenh_xac_nhan": [], "cham_moc": []}
    for which in ["old", "new"]:
        base = fx.load_new() if which == "new" else fx.load_old()
        ctx = im.build_context(base)
        parts["lenh_limit"].append(trades(base, ctx, entry="limit"))
        parts["lenh_xac_nhan"].append(trades(base, ctx, entry="confirm", rule="partial"))
        o = bo.real_orders(base, ctx)
        parts["cham_moc"].append(touches(base, ctx, o, bo.pseudo_orders(base, ctx, o)))
        print(which, {k: sum(len(x) for x in v) for k, v in parts.items()}, flush=True)
    for k, v in parts.items():
        df = pd.concat(v, ignore_index=True).sort_values("thoi_gian_server" if k != "cham_moc" else ["bo", "thoi_gian_server"])
        df.to_csv(f"{OUT}/{k}.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M")
        print(k, len(df), "rows")


if __name__ == "__main__":
    main()
