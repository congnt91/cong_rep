"""Cấu hình kết hợp: so sánh trên A1 (2012-2017), A2 (2017-2022) và B (2025-26).
Kết quả: out/combo.json"""
import json
import time

import numpy as np
import pandas as pd

import fxlevels as fx
import improve as im
from run_improve import period_of, stat_by_period, PERIODS


def build(base, ctx, entry="limit", offset=0.0, sl_atr=0.75, rr_min=1.0, tp_mode="struct", rule="fixed", tf="15min"):
    o = im.candidate_orders(base, ctx, sl_atr=sl_atr, offset=offset)
    if entry == "limit":
        tr = im.fill_limit(base, o)
    else:
        tr = im.fill_confirm(base, o, tf=tf, sl_from="level")
    tr = tr[tr.rr_struct >= rr_min].copy()
    s = im.simulate_managed(base, tr, rule, tp_mode)
    s = im.add_features(base, ctx, s)
    return s


F_ATR = ("ATR H1 ≥ trung vị 20 ngày", lambda d: d.atr_rel >= 1.0)
F_NONE = ("", lambda d: np.ones(len(d), bool))
F_RR15 = ("RR ≥ 1,5", lambda d: d.rr_struct >= 1.5)
F_TOUCH = ("Mốc đã bị chạm trước đó trong ngày", lambda d: d.touched_before)

# (key, nhãn, tham số dựng lệnh, bộ lọc)
CONFIGS = [
    ("base", "Hiện tại: Limit tại mốc, SL 0,75 ATR, TP mốc kế tiếp", dict(entry="limit"), F_NONE),
    ("base_atr", "Hiện tại + lọc ATR ≥ trung vị", dict(entry="limit"), F_ATR),
    ("base_partial", "Hiện tại + chốt 50% tại 1R", dict(entry="limit", rule="partial"), F_NONE),
    ("conf", "Xác nhận nến M15, SL 0,75 ATR sau mốc", dict(entry="confirm"), F_NONE),
    ("conf_atr", "Xác nhận M15 + lọc ATR ≥ trung vị", dict(entry="confirm"), F_ATR),
    ("conf_partial", "Xác nhận M15 + chốt 50% tại 1R", dict(entry="confirm", rule="partial"), F_NONE),
    ("conf_tp2", "Xác nhận M15 + TP cố định 2R", dict(entry="confirm", tp_mode=2.0), F_NONE),
    ("conf_atr_partial", "Xác nhận M15 + lọc ATR + chốt 50% tại 1R", dict(entry="confirm", rule="partial"), F_ATR),
    ("conf_atr_tp2", "Xác nhận M15 + lọc ATR + TP 2R", dict(entry="confirm", tp_mode=2.0), F_ATR),
    ("conf_h1", "Xác nhận nến H1, SL 0,75 ATR sau mốc", dict(entry="confirm", tf="1h"), F_NONE),
    ("conf_sl1", "Xác nhận M15, SL 1,0 ATR sau mốc", dict(entry="confirm", sl_atr=1.0), F_NONE),
    ("conf_rr15", "Xác nhận M15 + RR ≥ 1,5 (chọn máy móc trên A1)", dict(entry="confirm"), F_RR15),
    ("conf_touch", "Xác nhận M15 + mốc đã chạm trong ngày", dict(entry="confirm"), F_TOUCH),
]
RECOMMENDED = "conf_atr_partial"


def main():
    t0 = time.time()
    data = {}
    for which in ["old", "new"]:
        base = fx.load_new() if which == "new" else fx.load_old()
        data[which] = (base, im.build_context(base))
    cache = {}
    rows, frames = [], {}
    for key, label, kw, (flabel, fn) in CONFIGS:
        ck = json.dumps(kw, sort_keys=True)
        if ck not in cache:
            df = pd.concat([build(*data[w], **kw) for w in ["old", "new"]], ignore_index=True)
            df["period"] = period_of(df.fill_t.values)
            cache[ck] = df
        sub = cache[ck][fn(cache[ck])].copy()
        frames[key] = sub
        st = stat_by_period(sub)
        # tần suất lệnh / tháng
        months = {p: max(1, len(pd.DatetimeIndex(sub.fill_t[sub.period == p]).to_period("M").unique())) for p in PERIODS}
        for p in PERIODS:
            st[p]["per_month"] = round(st[p].get("n", 0) / months[p], 1)
        rows.append({"key": key, "label": label, "filter": flabel, "stats": st, "recommended": key == RECOMMENDED})
        print(f"{label:52s}", " ".join(f"{p}: n={st[p].get('n',0):5d} E={st[p].get('avgR', float('nan')):+.3f} [{st[p].get('lo', float('nan')):+.3f},{st[p].get('hi', float('nan')):+.3f}] win={st[p].get('win', float('nan')):.2f}" for p in PERIODS), round(time.time() - t0), "s", flush=True)
    # theo năm và đường vốn cho base, conf, recommended
    yearly, eq = {}, {}
    for key in ["base", "conf", RECOMMENDED]:
        sub = frames[key].sort_values("exit_t")
        yrs = pd.DatetimeIndex(sub.fill_t.values).year
        yearly[key] = [{"year": int(y), "n": int((yrs == y).sum()), "E": float(sub.R[yrs == y].mean())} for y in sorted(set(yrs))]
        for per, m in [("A", sub.period != "B"), ("B", sub.period == "B")]:
            s = sub[m]
            e = s.R.cumsum()
            e.index = pd.DatetimeIndex(s.exit_t.values)
            e = e.groupby(e.index.normalize()).last()
            if per == "A":
                e = e.resample("W").last().dropna()
            eq[f"{key}_{per}"] = {"t": [str(x)[:10] for x in e.index], "v": e.round(2).tolist()}
    # cơ chế: lệnh Limit có / không có nến xác nhận sau đó
    L, C = frames["base"], frames["conf"]
    keycols = ["t", "type", "level0", "dir"]
    m = L.merge(C[keycols + ["R"]].rename(columns={"R": "Rc"}), on=keycols, how="left")
    mech = {}
    for name, mask in [("with_conf", m.Rc.notna()), ("no_conf", m.Rc.isna())]:
        mech[name] = {p: {"n": int(((m.period == p) & mask).sum()), "E": float(m.R[(m.period == p) & mask].mean())} for p in PERIODS}
    # hình học: risk/ATR, RR trung vị, thắng, lãi TB, lỗ TB
    geo = {}
    for key in ["base", "conf", RECOMMENDED]:
        d = frames[key]
        geo[key] = {p: {"risk_atr": float((d.risk / d.atr)[d.period == p].median()), "rr": float(d.rr_struct[d.period == p].median()),
                        "avg_win": float(d.R[(d.period == p) & (d.R > 0)].mean()), "avg_loss": float(d.R[(d.period == p) & (d.R <= 0)].mean())} for p in PERIODS}
    out = {"rows": rows, "recommended": RECOMMENDED, "yearly": yearly, "eq": eq, "mechanism": mech, "geometry": geo,
           "median_risk_B": {k: float(frames[k].loc[frames[k].period == "B", "risk"].median()) for k in ["base", "conf", RECOMMENDED]}}
    json.dump(out, open("out/combo.json", "w"), ensure_ascii=False, default=float)
    print("saved", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
