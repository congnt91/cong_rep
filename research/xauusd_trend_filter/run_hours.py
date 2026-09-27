"""Tối ưu theo khung giờ: giờ nào mốc hay bị xuyên, giờ nào nên cấm, kiểm tra ngoài mẫu.
Giờ server MT5 = giờ New York + 7 (00:00 server = 17:00 NY). Giờ VN = server + 4 (mùa hè Mỹ) / + 5 (mùa đông).
Kết quả: out/hours.json"""
import json
import time

import numpy as np
import pandas as pd

import fxlevels as fx
import improve as im
from run_combo import build
from run_improve import period_of, PERIODS

# Khung giờ định nghĩa trước theo cấu trúc thị trường (giờ server), không chọn từ dữ liệu
WINDOWS = [
    ("W1", "Mở cửa lại sau giờ nghỉ", 1, 2, "Thị trường mở lại lúc 01:00 server sau 1 giờ nghỉ; spread thực tế rộng"),
    ("W2", "Á sớm (Tokyo, Thượng Hải mở)", 2, 5, "Tokyo mở 02:00–03:00 server, SGE mở 04:00"),
    ("W3", "Á muộn", 5, 8, "Giữa và cuối phiên Á, thanh khoản mỏng"),
    ("W4", "Trước giờ Âu", 8, 10, "Đầu phiên 8 giờ thứ hai của Nola-X, Frankfurt mở 09:00"),
    ("W5", "London mở cửa", 10, 12, "London mở 10:00 server"),
    ("W6", "Giữa phiên London", 12, 15, "London giữa phiên, chờ tin Mỹ"),
    ("W7", "Tin Mỹ 8:30 NY và NY mở cửa", 15, 17, "Tin Mỹ 15:30 server, COMEX mở 15:20, chứng khoán NY 16:30"),
    ("W8", "Tin Mỹ 10:00 NY và London fix", 17, 19, "Tin 10:00 NY = 17:00 server, London PM fix 17:00, London đóng 18:00–19:00"),
    ("W9", "Chiều NY", 19, 24, "Phiên chiều Mỹ, FOMC 21:00 server"),
]
FAMILIES = {
    "PDH/PDL": ["PDH", "PDL"],
    "Pivot ngày, Fibo": ["D_P", "D_R1", "D_S1", "FIB50"],
    "Pivot phiên 8h": ["S8_P", "S8_R1", "S8_S1"],
    "Pivot 12h": ["H12_R1", "H12_S1"],
}


def window_of(hours):
    out = np.full(len(hours), "W0", dtype=object)
    for key, _, a, b, _ in WINDOWS:
        out[(hours >= a) & (hours < b)] = key
    return out


def add_break(base, tr, horizon="4h"):
    """Độ xuyên mốc: giá đi ngược bao xa (ATR H1) qua mốc trong `horizon` sau lúc chạm."""
    t = base.index.values
    hi, lo = base.high.values, base.low.values
    step = pd.Timedelta(np.diff(t[:1000]).min())
    k = int(pd.Timedelta(horizon) / step)
    touch = np.searchsorted(t, tr.fill_t.values, "left") if "touch_i" not in tr else tr.touch_i.values
    L, D, A = tr.level.values, tr.dir.values, tr.atr.values
    pen = np.empty(len(tr))
    for j in range(len(tr)):
        a = touch[j]
        b = min(a + k, len(t))
        pen[j] = ((L[j] - lo[a:b].min()) if D[j] == 1 else (hi[a:b].max() - L[j])) / A[j]
    tr = tr.copy()
    tr["pen"] = pen
    return tr


def stat(R):
    R = np.asarray(R, float)
    return {"n": int(len(R)), "E": float(R.mean()) if len(R) else None, "win": float((R > 0).mean()) if len(R) else None}


def by_period(df, mask=None, col="R"):
    d = df if mask is None else df[mask]
    return {p: stat(d.loc[d.period == p, col].values) for p in PERIODS}


def select_bans(df, periods, delta, min_n):
    """Chọn khung cấm trên các giai đoạn `periods`: E khung thấp hơn E chung ít nhất `delta`, đủ lệnh."""
    m = df.period.isin(periods)
    base = df.loc[m, "R"].mean()
    bans = []
    for key, *_ in WINDOWS:
        w = m & (df.win == key)
        if w.sum() >= min_n and df.loc[w, "R"].mean() <= base - delta:
            bans.append(key)
    return bans


def select_bans_hour(df, periods, delta, min_n):
    m = df.period.isin(periods)
    base = df.loc[m, "R"].mean()
    return [int(h) for h in range(24) if ((m & (df.h == h)).sum() >= min_n) and df.loc[m & (df.h == h), "R"].mean() <= base - delta]


def main():
    t0 = time.time()
    data = {}
    for which in ["old", "new"]:
        base = fx.load_new() if which == "new" else fx.load_old()
        data[which] = (base, im.build_context(base))

    sets = {}
    for key, kw, flt in [
        ("limit", dict(entry="limit"), None),
        ("confirm", dict(entry="confirm"), None),
        ("rec", dict(entry="confirm", rule="partial"), "atr"),
    ]:
        parts = []
        for w in ["old", "new"]:
            base, ctx = data[w]
            s = build(base, ctx, **kw)
            if key != "limit":
                # thời điểm chạm mốc = nến M15 xác nhận (fill_t là lúc nến đóng); độ xuyên đo từ lúc đóng nến
                pass
            s = add_break(base, s)
            parts.append(s)
        df = pd.concat(parts, ignore_index=True)
        if flt == "atr":
            df = df[df.atr_rel >= 1.0].copy()
        df["period"] = period_of(df.fill_t.values)
        T = pd.DatetimeIndex(df.fill_t.values)
        # với lệnh có xác nhận, dùng giờ bắt đầu của nến xác nhận (fill_t là giờ đóng nến)
        if key != "limit":
            T = T - pd.Timedelta("15min")
        df["h"] = T.hour
        df["win"] = window_of(df.h.values)
        df["fam"] = df.type.map({t: f for f, ts in FAMILIES.items() for t in ts})
        sets[key] = df
        print(key, len(df), round(time.time() - t0), "s", flush=True)

    OUT = {"windows": [{"key": k, "label": l, "a": a, "b": b, "note": n} for k, l, a, b, n in WINDOWS], "sets": {}}
    for key, df in sets.items():
        S = OUT["sets"][key] = {}
        S["overall"] = by_period(df)
        S["by_window"] = {w: by_period(df, df.win == w) for w, *_ in WINDOWS}
        S["by_hour"] = {int(h): by_period(df, df.h == h) for h in range(24)}
        # độ xuyên mốc theo khung: tỷ lệ giá đi quá 1,5 ATR qua mốc trong 4h (mốc "vỡ")
        S["break"] = {w: {p: float((df.pen[(df.win == w) & (df.period == p)] >= 1.5).mean()) if ((df.win == w) & (df.period == p)).any() else None for p in PERIODS} for w, *_ in WINDOWS}
        S["break_all"] = {p: float((df.pen[df.period == p] >= 1.5).mean()) for p in PERIODS}
        S["mech"] = {w: {p: ({"win": float((x.R > 0).mean()), "avg_win": float(x.R[x.R > 0].mean()), "rr": float(x.rr_struct.median()),
                              "brk": float((x.pen >= 1.5).mean())} if len(x) else None)
                         for p in PERIODS for x in [df[(df.win == w) & (df.period == p)]]} for w, *_ in WINDOWS}
        S["fam_window"] = {f: {w: by_period(df, (df.fam == f) & (df.win == w)) for w, *_ in WINDOWS} for f in FAMILIES}
        S["dir_window"] = {str(d): {w: by_period(df, (df.dir == d) & (df.win == w)) for w, *_ in WINDOWS} for d in (1, -1)}
        # chọn khung cấm ngoài mẫu
        sel = {}
        for scheme, train, delta, min_n in [("A1", ["A1"], 0.04, 150 if key == "limit" else 60), ("A", ["A1", "A2"], 0.04, 300 if key == "limit" else 120)]:
            bans = select_bans(df, train, delta, min_n)
            keep = ~df.win.isin(bans)
            sel[scheme] = {"bans": bans, "kept": by_period(df, keep), "banned": by_period(df, ~keep)}
            hb = select_bans_hour(df, train, delta, min_n // 4)
            keeph = ~df.h.isin(hb)
            sel[scheme + "_hour"] = {"bans": hb, "kept": by_period(df, keeph), "banned": by_period(df, ~keeph)}
        S["select"] = sel
        print(f"== {key}")
        print("   overall", {p: round(S['overall'][p]['E'], 3) for p in PERIODS})
        for w, lab, a, b, _ in WINDOWS:
            r = S["by_window"][w]
            print(f"   {w} {a:02d}-{b:02d} {lab:34s}", " ".join(f"{p}: n={r[p]['n']:5d} E={r[p]['E'] if r[p]['E'] is None else round(r[p]['E'],3):>7}" for p in PERIODS), "| vỡ", {p: None if S['break'][w][p] is None else round(S['break'][w][p], 2) for p in PERIODS})
        for sc, v in sel.items():
            print(f"   select[{sc}] bans={v['bans']}", "kept", {p: (v['kept'][p]['n'], None if v['kept'][p]['E'] is None else round(v['kept'][p]['E'], 3)) for p in PERIODS}, "banned", {p: (v['banned'][p]['n'], None if v['banned'][p]['E'] is None else round(v['banned'][p]['E'], 3)) for p in PERIODS})
    # cấu hình cuối cùng (khung cấm chọn trên A1 hoặc A, xem OUT["sets"][*]["select"])
    L, C, R = sets["limit"], sets["confirm"], sets["rec"]
    asia = ["W2"]
    FINAL = [
        ("limit", "Hiện tại: Limit tại mốc", L, np.ones(len(L), bool), None),
        ("limit_ban_A1", "Limit, cấm 01–02h và 10–12h server (chọn trên A1)", L, ~L.win.isin(["W1", "W5"]), "A1"),
        ("limit_ban_A", "Limit, cấm 01–02h, 10–15h server (chọn trên A1+A2)", L, ~L.win.isin(["W1", "W5", "W6"]), "A"),
        ("conf", "Xác nhận M15", C, np.ones(len(C), bool), None),
        ("conf_ban_A", "Xác nhận M15, cấm 10–15h server (chọn trên A1+A2)", C, ~C.win.isin(["W5", "W6"]), "A"),
        ("rec", "Đề xuất trước: xác nhận + lọc ATR + chốt 50%", R, np.ones(len(R), bool), None),
        ("rec_ban", "Đề xuất trước + cấm 01–02h, 10–15h server", R, ~R.win.isin(["W1", "W5", "W6"]), "A"),
        ("conf_asia", "Xác nhận M15, chỉ phiên Á sớm 02–05h server", C, C.win.isin(asia), "A1"),
        ("conf_asia_pd", "Xác nhận M15, chỉ PDH/PDL, chỉ 02–05h server", C, C.win.isin(asia) & (C.fam == "PDH/PDL"), "A1"),
    ]
    OUT["final"] = []
    for key, label, df, m, chosen in FINAL:
        d = df[m]
        row = {"key": key, "label": label, "chosen_on": chosen, "stats": {}}
        for p in PERIODS:
            x = d[d.period == p]
            if len(x) == 0:
                row["stats"][p] = {"n": 0}
                continue
            lo, hi = im.boot_mean_ci(x.R.values, x.exit_t.values)
            months = max(1, len(pd.DatetimeIndex(x.fill_t.values).to_period("M").unique()))
            row["stats"][p] = {"n": int(len(x)), "E": float(x.R.mean()), "lo": lo, "hi": hi, "win": float((x.R > 0).mean()),
                               "per_month": round(len(x) / months, 1), "totR": float(x.R.sum())}
        OUT["final"].append(row)
        print(f"   FINAL {label:58s}", " ".join(f"{p}: n={row['stats'][p].get('n',0):5d} E={row['stats'][p].get('E', float('nan')):+.3f} [{row['stats'][p].get('lo', float('nan')):+.3f},{row['stats'][p].get('hi', float('nan')):+.3f}]" for p in PERIODS), flush=True)
    # breakout (Stop) theo khung: lệnh Stop thuận Trend H4, SL 1,5 ATR, TP 2R — có hiệu quả ở giờ mốc hay vỡ?
    stops = []
    for w in ["old", "new"]:
        base, ctx = data[w]
        o = fx.make_orders(base, ctx, "static", sl_atr=1.5, rr_min=0.0, otype="stop")
        tr = fx.simulate(base, o).dropna(subset=["fill_t"])
        ex = fx.excursions(base, tr)
        ex["R"] = np.where(ex.mfe >= 2.0, 2.0, np.where(ex.slhit, ex.slR, ex.rend)) - ex.cost
        stops.append(ex)
    st = pd.concat(stops, ignore_index=True)
    st["period"] = period_of(st.fill_t.values)
    st["h"] = pd.DatetimeIndex(st.fill_t.values).hour
    st["win"] = window_of(st.h.values)
    al = st.dir == st.trend_h4
    OUT["stop"] = {"all": by_period(st, al), "by_window": {w: by_period(st, al & (st.win == w)) for w, *_ in WINDOWS}}
    print("== stop thuận Trend H4, SL 1.5 ATR, TP 2R", {p: round(OUT['stop']['all'][p]['E'], 3) for p in PERIODS})
    for w, lab, a, b, _ in WINDOWS:
        r = OUT["stop"]["by_window"][w]
        print(f"   {w} {lab:34s}", " ".join(f"{p}: n={r[p]['n']:5d} E={r[p]['E'] if r[p]['E'] is None else round(r[p]['E'],3):>7}" for p in PERIODS))
    json.dump(OUT, open("out/hours.json", "w"), ensure_ascii=False, default=float)
    print("saved", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
