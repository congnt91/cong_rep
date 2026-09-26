"""Chạy toàn bộ thử nghiệm cải thiện Auto mốc. Kết quả: out/improve.json
Chia thời gian: A1 = 05/2012-04/2017 (chọn quy tắc), A2 = 05/2017-03/2022 (kiểm tra), B = 10/2025-09/2026 (kiểm tra)."""
import json
import time

import numpy as np
import pandas as pd

import fxlevels as fx
import improve as im

SPLIT = pd.Timestamp("2017-05-01")
PERIODS = ["A1", "A2", "B"]
FEATURES = {
    "session": ("Phiên khớp lệnh", None),
    "dow": ("Thứ trong tuần", None),
    "type": ("Loại mốc", None),
    "dir": ("Hướng lệnh", None),
    "confluence": ("Số mốc gộp", [0.5, 1.5, 9]),
    "dist": ("Khoảng cách mốc lúc đặt (ATR)", [0, 0.5, 1.0, 2.0, 9]),
    "rr_struct": ("RR cấu trúc lúc đặt", [0.99, 1.5, 2.0, 2.99, 3.01]),
    "stretch": ("Mốc cách EMA20 H1 theo hướng lệnh (ATR)", [-9, 0, 0.5, 1.0, 2.0, 99]),
    "rsi_dir": ("RSI14 H1 theo hướng lệnh", [0, 30, 40, 50, 60, 100]),
    "approach": ("Giá cách mốc 1 giờ trước khi khớp (ATR)", [-0.01, 0.3, 0.6, 1.0, 2.0, 99]),
    "atr_rel": ("ATR H1 so với trung vị 20 ngày", [0, 0.7, 1.0, 1.5, 99]),
    "day_used": ("Biên độ ngày đã dùng (ATR ngày)", [0, 0.3, 0.6, 1.0, 99]),
    "touched_before": ("Mốc đã bị chạm trước đó trong ngày", None),
    "hrs_to_fill": ("Giờ từ lúc đặt tới lúc khớp", [-0.01, 0.5, 1.5, 4, 8]),
}


def period_of(times):
    T = pd.DatetimeIndex(times)
    return np.where(T.year >= 2025, "B", np.where(T < SPLIT, "A1", "A2"))


def stat_by_period(df, Rcol="R"):
    out = {}
    for p in PERIODS:
        m = df.period == p
        if m.sum() == 0:
            out[p] = {"n": 0}
            continue
        s = im.summary(df.loc[m, Rcol].values, df.loc[m, "exit_t"].values, ci=True)
        out[p] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
    return out


def run_period_set(base, ctx, which, cfg_fn):
    """cfg_fn(base, ctx) -> DataFrame có cột R, exit_t. Gắn period."""
    df = cfg_fn(base, ctx)
    df["period"] = period_of(df.fill_t.values)
    return df


def main():
    t0 = time.time()
    OUT = {"experiments": {}, "filters": {}, "combo": {}}
    data = {}
    for which in ["old", "new"]:
        base = fx.load_new() if which == "new" else fx.load_old()
        ctx = im.build_context(base)
        data[which] = (base, ctx)

    def both(cfg_fn):
        parts = [run_period_set(*data[w], w, cfg_fn) for w in ["old", "new"]]
        return pd.concat(parts, ignore_index=True)

    def limit_cfg(sl_atr=0.75, offset=0.0, rr_min=1.0, rule="fixed", tp_mode="struct", time_stop=None, min_tp=None):
        def f(base, ctx):
            o = im.candidate_orders(base, ctx, sl_atr=sl_atr, offset=offset)
            tr = im.fill_limit(base, o)
            tr = tr[tr.rr_struct >= rr_min].copy()
            if min_tp is not None:
                tr["tpd_struct"] = np.maximum(tr.tpd_struct, min_tp * tr.risk)
            return im.simulate_managed(base, tr, rule, tp_mode, time_stop=time_stop)
        return f

    def confirm_cfg(tf="15min", strong=False, sl_from="level", rr_min=1.0, sl_atr=0.75, new_only=False):
        def f(base, ctx):
            if new_only and base.index[0].year < 2025:
                return pd.DataFrame(columns=["R", "exit_t", "fill_t"])
            o = im.candidate_orders(base, ctx, sl_atr=sl_atr)
            tr = im.fill_confirm(base, o, tf=tf, strong=strong, sl_from=sl_from)
            tr = tr[tr.rr_struct >= rr_min].copy()
            return im.simulate_managed(base, tr, "fixed", "struct")
        return f

    EXP = OUT["experiments"]

    def add(group, key, label, cfg_fn, keep=False):
        df = both(cfg_fn)
        EXP.setdefault(group, []).append({"key": key, "label": label, "stats": stat_by_period(df)})
        print(f"[{group}] {label:52s}", {p: (EXP[group][-1]['stats'][p].get('n'), EXP[group][-1]['stats'][p].get('avgR')) for p in PERIODS}, round(time.time() - t0), "s", flush=True)
        return df if keep else None

    # ---------------- baseline
    base_df = add("entry", "base", "Limit tại mốc (hiện tại)", limit_cfg(), keep=True)

    # ---------------- G1: cách vào lệnh
    for off in [-0.25, -0.1, 0.1, 0.25]:
        add("entry", f"off{off}", f"Limit lệch {off:+.2f} ATR {'(sâu hơn mốc)' if off < 0 else '(trước mốc)'}", limit_cfg(offset=off))
    add("entry", "conf15_lvl", "Xác nhận nến M15 đóng cửa quay lại, SL sau mốc", confirm_cfg())
    add("entry", "conf15_ent", "Xác nhận nến M15, SL 0,75 ATR từ giá vào", confirm_cfg(sl_from="entry"))
    add("entry", "conf15_strong", "Xác nhận nến M15 mạnh (đóng ở 40% trên/dưới), SL sau mốc", confirm_cfg(strong=True))
    add("entry", "conf5_lvl", "Xác nhận nến M5 (chỉ B), SL sau mốc", confirm_cfg(tf="5min", new_only=True))
    add("entry", "conf15_lvl_rr0", "Xác nhận M15, SL sau mốc, không lọc RR", confirm_cfg(rr_min=0))

    # ---------------- G2: SL / TP / RR
    for sl in [0.5, 1.0, 1.25, 1.5]:
        add("sl", f"sl{sl}", f"SL {sl} ATR (TP cấu trúc, RR ≥ 1)", limit_cfg(sl_atr=sl))
    for rr in [1.5, 2.0]:
        add("sl", f"rr{rr}", f"Chỉ nhận lệnh RR ≥ {rr}", limit_cfg(rr_min=rr))
    for tp in [1.5, 2.0, 3.0]:
        add("tp", f"tp{tp}", f"TP cố định {tp}R", limit_cfg(tp_mode=tp))
    add("tp", "tpmin1.5", "TP = max(mốc kế tiếp, 1,5R)", limit_cfg(min_tp=1.5))
    add("tp", "tpmin2", "TP = max(mốc kế tiếp, 2R)", limit_cfg(min_tp=2.0))

    # ---------------- G3: quản lý lệnh
    add("manage", "be1", "Dời SL hoà vốn khi lời ≥ 1R", limit_cfg(rule="be1"))
    add("manage", "trail1", "Trailing 1R sau khi lời ≥ 1R", limit_cfg(rule="trail1"))
    add("manage", "partial", "Chốt 50% tại 1R, dời SL hoà vốn", limit_cfg(rule="partial"))
    for h in [4, 8, 12]:
        add("manage", f"time{h}", f"Đóng lệnh sau {h} giờ nếu chưa chạm SL/TP", limit_cfg(time_stop=h))

    # ---------------- G4: bộ lọc theo đặc trưng (trên lệnh baseline)
    feats = []
    for w in ["old", "new"]:
        base, ctx = data[w]
        o = im.candidate_orders(base, ctx)
        tr = im.fill_limit(base, o)
        tr = tr[tr.rr_struct >= 1].copy()
        s = im.simulate_managed(base, tr, "fixed", "struct")
        s = im.add_features(base, ctx, s)
        s["dist"] = np.abs(s.level0 - s.price) / s.atr
        feats.append(s)
    F = pd.concat(feats, ignore_index=True)
    F["period"] = period_of(F.fill_t.values)
    F.to_pickle("out/features.pkl")
    FL = OUT["filters"]
    for feat, (label, bins) in FEATURES.items():
        if bins is None:
            F["_bin"] = F[feat].astype(str)
        else:
            F["_bin"] = pd.cut(F[feat], bins).astype(str)
        rows = []
        for b in sorted(F["_bin"].unique(), key=lambda x: (F.loc[F._bin == x, feat].min() if bins else x)):
            r = {"bin": str(b)}
            for p in PERIODS:
                m = (F._bin == b) & (F.period == p)
                r[p] = {"n": int(m.sum()), "E": float(F.loc[m, "R"].mean()) if m.sum() else None}
            rows.append(r)
        FL[feat] = {"label": label, "rows": rows}
        print(f"[filter] {label}")
        for r in rows:
            print("   ", f"{str(r['bin']):22s}", " ".join(f"{p}: n={r[p]['n']:5d} E={r[p]['E'] if r[p]['E'] is None else round(r[p]['E'],3)}" for p in PERIODS))
    OUT["baseline_by_period"] = stat_by_period(F)

    # ---------------- G5: chọn quy tắc trên A1, kiểm tra A2 và B
    # quy tắc ứng viên: mỗi bin của mỗi đặc trưng; giữ bin nếu trên A1 có n >= 400 và E cao hơn baseline A1 >= 0.04R
    baseE = {p: F.loc[F.period == p, "R"].mean() for p in PERIODS}
    cand = []
    for feat, (label, bins) in FEATURES.items():
        col = F[feat].astype(str) if bins is None else pd.cut(F[feat], bins).astype(str)
        for b in col.unique():
            m1 = (col == b) & (F.period == "A1")
            if m1.sum() < 400:
                continue
            e1 = F.loc[m1, "R"].mean()
            if e1 - baseE["A1"] >= 0.04:
                m2 = (col == b) & (F.period == "A2"); mb = (col == b) & (F.period == "B")
                cand.append({"feat": feat, "label": label, "bin": str(b), "n1": int(m1.sum()), "E1": float(e1),
                             "E2": float(F.loc[m2, "R"].mean()), "n2": int(m2.sum()), "EB": float(F.loc[mb, "R"].mean()), "nB": int(mb.sum()),
                             "gain1": float(e1 - baseE["A1"]), "gain2": float(F.loc[m2, "R"].mean() - baseE["A2"]), "gainB": float(F.loc[mb, "R"].mean() - baseE["B"])})
    cand.sort(key=lambda r: -r["gain1"])
    OUT["combo"]["candidates"] = cand
    print("[combo] baseline", {p: round(v, 4) for p, v in baseE.items()})
    for r in cand:
        print("   ", f"{str(r['label'])[:34]:34s} {str(r['bin']):18s} A1 n={r['n1']:5d} +{r['gain1']:.3f} | A2 n={r['n2']:5d} {r['gain2']:+.3f} | B n={r['nB']:4d} {r['gainB']:+.3f}")
    OUT["runtime_s"] = round(time.time() - t0)
    json.dump(OUT, open("out/improve.json", "w"), ensure_ascii=False, default=float)
    print("saved", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
