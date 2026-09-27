"""Phần 5: chạm mốc bao nhiêu % bật lại, và chiến lược chỉ ăn đoạn bật (TP mỏng). Kết quả: out/bounce.json

Đơn vị: R = 0,75 ATR H1 (SL hiện tại, ≈ 14 USD / 0,01 lot năm 2026); USD tính cho 0,01 lot (1 oz).
Giai đoạn A (2012–2022) chỉ có nến M15 nên đo thiếu các cú bật nhỏ trong nến; B (2025–26) dùng nến M1.
Chi phí: B 0,30 USD/lệnh; A quy theo tỉ lệ chi phí/ATR hiện nay (0,30 USD / ATR H1 trung vị của B), thuận lợi
hơn thực tế lúc đó.

Sai số do độ phân giải (không thấy cú bật trong nến khớp) tác động như nhau lên mốc thật và mốc giả, nên
chênh lệch thật − giả là thước đo chính. Kỳ vọng ước tính = −chi phí + chênh lệch (mốc giả coi như giá ngẫu
nhiên, kỳ vọng trước phí bằng 0).
"""
import json
import time

import numpy as np
import pandas as pd

import bounce as bo
import fxlevels as fx
import improve as im
from run_hours import FAMILIES, WINDOWS, window_of
from run_improve import PERIODS, period_of

SPREAD = 0.30
R_ATR = 0.75
DAY = 24 * 60
X_ATR = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5]
Y_ATR = [0.5, 0.75, 1.0]
X_USD = [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15]
GRID_TP_USD = [0.5, 1, 2, 3, 5, 8]
GRID_SL_USD = [("5", 5.0), ("10", 10.0), ("15", 15.0), ("0.75atr", "0.75atr"), ("20", 20.0), ("30", 30.0), ("none", np.inf)]
GRID_TP_ATR = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
GRID_SL_ATR = [0.5, 0.75, 1.0, 1.5, 2.0, np.inf]
TAIL_BINS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]
FAM_OF = {t: f for f, ts in FAMILIES.items() for t in ts}


def sub(P, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == (P["n"],) else v) for k, v in P.items()} | {"n": int(m.sum())}


def months(times):
    T = pd.DatetimeIndex(times)
    return max((T.max() - T.min()) / pd.Timedelta("30.44D"), 1e-9)


def rnd(v, k=4):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), k)


def cell(pnl, code, atr, y, tp, times=None, ci=False):
    """Tóm tắt một cấu hình: tỉ lệ thắng, tỉ lệ thắng cần để hoà vốn, kỳ vọng USD và R."""
    n = len(pnl)
    if n == 0:
        return {"n": 0}
    R = pnl / (R_ATR * atr)
    loss = -pnl[pnl < 0]
    win = pnl[pnl > 0]
    be = loss.mean() / (loss.mean() + win.mean()) if len(loss) and len(win) else np.nan
    out = {"n": n, "win": rnd((pnl > 0).mean()), "tp_rate": rnd((code == 1).mean()), "be_win": rnd(be),
           "E_usd": rnd(pnl.mean(), 3), "E_R": rnd(R.mean()), "avg_win_usd": rnd(win.mean() if len(win) else np.nan, 3),
           "avg_loss_usd": rnd(-loss.mean() if len(loss) else np.nan, 3)}
    if ci and times is not None:
        lo, hi = im.boot_mean_ci(R, times)
        out["lo"], out["hi"] = rnd(lo), rnd(hi)
    return out


def equity_stats(pnl_R, pnl_usd, times, per_month):
    o = np.argsort(times)
    r, u = pnl_R[o], pnl_usd[o]
    cum = np.cumsum(u)
    dd = np.max(np.maximum.accumulate(np.concatenate([[0], cum]))[1:] - cum)
    cumR = np.cumsum(r)
    ddR = np.max(np.maximum.accumulate(np.concatenate([[0], cumR]))[1:] - cumR)
    streak = best = 0
    for v in r:
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    return {"per_month": rnd(per_month, 1), "max_dd_R": rnd(ddR, 2), "max_dd_usd": rnd(dd, 1), "lose_streak": int(best),
            "worst_R": rnd(r.min(), 2), "worst_usd": rnd(u.min(), 1), "total_R": rnd(r.sum(), 1), "total_usd": rnd(u.sum(), 1)}


def boot_diff_ci(Ra, ta, Rb, tb, reps=2000, seed=7):
    """KTC 95% của mean(Ra) − mean(Rb), bootstrap theo khối tuần chung."""
    rng = np.random.default_rng(seed)
    wa = pd.DatetimeIndex(ta).to_period("W").astype(str).values
    wb = pd.DatetimeIndex(tb).to_period("W").astype(str).values
    weeks = np.union1d(wa, wb)
    ia = {w: np.nonzero(wa == w)[0] for w in weeks}
    ib = {w: np.nonzero(wb == w)[0] for w in weeks}
    ms = []
    for _ in range(reps):
        pick = rng.choice(weeks, len(weeks))
        sa = np.concatenate([ia[w] for w in pick])
        sb = np.concatenate([ib[w] for w in pick])
        if len(sa) and len(sb):
            ms.append(Ra[sa].mean() - Rb[sb].mean())
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def y_of(spec, atr):
    return R_ATR * atr if spec == "0.75atr" else np.full(len(atr), spec, float)


def main():
    t0 = time.time()
    OUT = {"meta": {"spread": SPREAD, "R_atr": R_ATR, "x_atr": X_ATR, "y_atr": Y_ATR, "x_usd": X_USD,
                    "grid_tp_usd": GRID_TP_USD, "grid_sl_usd": [k for k, _ in GRID_SL_USD],
                    "grid_tp_atr": GRID_TP_ATR, "grid_sl_atr": [("none" if not np.isfinite(v) else v) for v in GRID_SL_ATR],
                    "windows": {k: [lab, a, b] for k, lab, a, b, _ in WINDOWS}, "families": list(FAMILIES)},
           "counts": {}, "curve": {}, "curve_usd": {}, "resolution": {}, "grid_usd": {}, "grid_atr": {},
           "tail": {}, "breakdown": {}, "configs": [], "equity": {}, "spread_sens": {}}

    SETS = {}  # (set, period) -> (tr, P)
    data = {}
    for which in ["new", "old"]:
        base = fx.load_new() if which == "new" else fx.load_old()
        ctx = im.build_context(base)
        data[which] = (base, ctx, bo.real_orders(base, ctx))
    atr_b = float(np.median(im.fill_limit(data["new"][0], data["new"][2]).atr))
    rel = SPREAD / atr_b
    OUT["meta"]["atr_B_median"] = rnd(atr_b, 2)
    OUT["meta"]["spread_rel_atr"] = rnd(rel, 5)

    def spread_of(p, atr):
        return np.full(len(atr), SPREAD) if p == "B" else rel * atr

    for which in ["old", "new"]:
        base, ctx, o = data[which]
        po = bo.pseudo_orders(base, ctx, o)
        built = {
            "real": bo.touches(base, o, "limit"),
            "pseudo": bo.touches(base, po, "limit"),
            "confirm": bo.touches(base, o, "confirm"),
            "confirm_pseudo": bo.touches(base, po, "confirm"),
        }
        for name, tr in built.items():
            tr = tr.reset_index(drop=True)
            tr["period"] = period_of(tr.fill_t.values)
            P = bo.paths(base, tr, at_level=not name.startswith("confirm"))
            for p in np.unique(tr.period):
                m = (tr.period == p).values
                SETS[(name, p)] = (tr[m].reset_index(drop=True), sub(P, m))
        if which == "new":
            # độ phân giải: cùng các lần chạm, đo bằng nến M1 / M5 / M15
            for tf in ["1min", "5min", "15min"]:
                b = base if tf == "1min" else fx.resample(base, tf)[["open", "high", "low", "close"]]
                res = {}
                for name in ["real", "pseudo"]:
                    tr = built[name].copy()
                    tr["fill_i"] = np.searchsorted(b.index.values, tr.fill_t.values, "right") - 1
                    P = bo.paths(b, tr)
                    res[name] = [rnd(bo.bounce_prob(P, x, R_ATR)) for x in X_ATR]
                OUT["resolution"][tf] = res
        print(which, {k: len(v) for k, v in built.items()}, round(time.time() - t0), "s", flush=True)

    for (name, p), (tr, P) in SETS.items():
        OUT["counts"].setdefault(p, {})[name] = {"n": len(tr), "per_month": rnd(len(tr) / months(tr.fill_t.values), 1)}
        if name == "real":
            OUT["meta"].setdefault("cost_R", {})[p] = rnd(np.mean(spread_of(p, tr.atr.values) / (R_ATR * tr.atr.values)))

    # ---------------- 1. tỉ lệ bật theo x (ATR) với SL y
    for p in PERIODS:
        OUT["curve"][p] = {}
        for y in Y_ATR:
            d = {"theory": [rnd(y / (x + y)) for x in X_ATR]}
            for name in ["real", "pseudo", "confirm", "confirm_pseudo"]:
                tr, P = SETS[(name, p)]
                d[name] = [rnd(bo.bounce_prob(P, x, y)) for x in X_ATR]
            OUT["curve"][p][str(y)] = d
    # B theo USD, SL 0,75 ATR
    for name in ["real", "pseudo", "confirm", "confirm_pseudo"]:
        tr, P = SETS[(name, "B")]
        OUT["curve_usd"][name] = []
        for x in X_USD:
            _, code, _, _ = bo.evaluate(P, x, R_ATR * tr.atr.values, DAY, 0.0)
            OUT["curve_usd"][name].append(rnd((code == 1).mean()))
    trB = SETS[("real", "B")][0]
    OUT["curve_usd"]["theory"] = [rnd(np.mean(R_ATR * trB.atr.values / (x + R_ATR * trB.atr.values))) for x in X_USD]
    print("curves", round(time.time() - t0), "s", flush=True)

    # ---------------- 2. lưới TP mỏng (B, USD)
    for name in ["real", "confirm", "pseudo", "confirm_pseudo"]:
        tr, P = SETS[(name, "B")]
        g = []
        for x in GRID_TP_USD:
            row = []
            for key, spec in GRID_SL_USD:
                y = y_of(spec, tr.atr.values)
                pnl, code, _, _ = bo.evaluate(P, x, y, DAY, SPREAD)
                row.append(cell(pnl, code, tr.atr.values, y, x))
            g.append(row)
        OUT["grid_usd"][name] = g

    # ---------------- 3. lưới theo ATR, mọi giai đoạn; kèm chênh lệch với mốc giả
    for p in PERIODS:
        OUT["grid_atr"][p] = {}
        for name in ["real", "confirm", "pseudo", "confirm_pseudo"]:
            tr, P = SETS[(name, p)]
            A = tr.atr.values
            g = []
            for x in GRID_TP_ATR:
                row = []
                for y in GRID_SL_ATR:
                    pnl, code, _, _ = bo.evaluate(P, x * A, y * A, DAY, spread_of(p, A))
                    row.append(cell(pnl, code, A, y, x))
                g.append(row)
            OUT["grid_atr"][p][name] = g
    print("grids", round(time.time() - t0), "s", flush=True)

    # ---------------- 4. đuôi lỗ khi không đặt SL
    for p in PERIODS:
        OUT["tail"][p] = {}
        for name in ["real", "confirm"]:
            tr, P = SETS[(name, p)]
            A = tr.atr.values
            d = {}
            for x in [0.05, 0.1, 0.2]:
                mae, ok = bo.mae_before(P, x * np.ones(len(tr)), DAY)
                hist = []
                for a, b in zip(TAIL_BINS[:-1], TAIL_BINS[1:]):
                    hist.append(rnd(np.mean(ok & (mae >= a) & (mae < b))))
                pnl, code, _, _ = bo.evaluate(P, x * A, np.inf, DAY, spread_of(p, A))
                lost = code == 0
                d[str(x)] = {"hist": hist, "never": rnd(np.mean(~ok)),
                             "never_avg_R": rnd((pnl[lost] / (R_ATR * A[lost])).mean() if lost.any() else np.nan, 2),
                             "never_avg_usd": rnd(pnl[lost].mean() if lost.any() else np.nan, 1),
                             "never_worst_usd": rnd(pnl[lost].min() if lost.any() else np.nan, 1),
                             "never_worst_R": rnd((pnl[lost] / (R_ATR * A[lost])).min() if lost.any() else np.nan, 2),
                             "win_avg_usd": rnd(pnl[~lost].mean(), 2),
                             "wins_per_loss": rnd(-pnl[lost].mean() / pnl[~lost].mean() if lost.any() and pnl[~lost].mean() > 0 else np.nan, 0),
                             "cell": cell(pnl, code, A, None, x)}
            OUT["tail"][p][name] = d

    # ---------------- 5. theo loại mốc và khung giờ (TP 0,2 ATR, SL 0,75 ATR, không spread)
    for p in PERIODS:
        tr, P = SETS[("real", p)]
        trp, Pp = SETS[("pseudo", p)]
        out = {}
        for x in [0.1, 0.2, 0.5]:
            _, code, _, _ = bo.evaluate(P, x * tr.atr.values, R_ATR * tr.atr.values, DAY, 0.0)
            _, codep, _, _ = bo.evaluate(Pp, x * trp.atr.values, R_ATR * trp.atr.values, DAY, 0.0)
            w = code == 1
            wp = codep == 1
            fam = np.array([FAM_OF.get(t, "?") for t in tr.type])
            win_r = window_of(pd.DatetimeIndex(tr.fill_t.values).hour.values)
            win_p = window_of(pd.DatetimeIndex(trp.fill_t.values).hour.values)
            d = {"all": {"real": rnd(w.mean()), "pseudo": rnd(wp.mean()), "n": len(w), "n_p": len(wp)}, "family": {}, "window": {}}
            for f in FAMILIES:
                m = fam == f
                d["family"][f] = {"real": rnd(w[m].mean()) if m.any() else None, "pseudo": rnd(wp.mean()), "n": int(m.sum())}
            for k, *_ in WINDOWS:
                m, mp = win_r == k, win_p == k
                d["window"][k] = {"real": rnd(w[m].mean()) if m.any() else None,
                                  "pseudo": rnd(wp[mp].mean()) if mp.any() else None, "n": int(m.sum()), "n_p": int(mp.sum())}
            out[str(x)] = d
        OUT["breakdown"][p] = out
    print("tail/breakdown", round(time.time() - t0), "s", flush=True)

    # ---------------- 6. cấu hình tiêu biểu
    CONFIGS = [
        ("lim_tp005", "Limit tại mốc, TP 0,05 ATR (≈ 1 USD), SL 0,75 ATR", "real", 0.05, R_ATR),
        ("lim_tp01", "Limit tại mốc, TP 0,1 ATR (≈ 2 USD), SL 0,75 ATR", "real", 0.1, R_ATR),
        ("lim_tp02", "Limit tại mốc, TP 0,2 ATR (≈ 3,7 USD), SL 0,75 ATR", "real", 0.2, R_ATR),
        ("lim_tp01_nosl", "Limit tại mốc, TP 0,1 ATR, không SL (đóng sau 24h)", "real", 0.1, np.inf),
        ("lim_tp01_sl2", "Limit tại mốc, TP 0,1 ATR, SL 2 ATR", "real", 0.1, 2.0),
        ("conf_tp01", "Xác nhận M15, TP 0,1 ATR, SL 0,75 ATR", "confirm", 0.1, R_ATR),
        ("conf_tp02", "Xác nhận M15, TP 0,2 ATR, SL 0,75 ATR", "confirm", 0.2, R_ATR),
        ("pseudo_tp01", "Đối chứng: Limit tại mốc giả, TP 0,1 ATR, SL 0,75 ATR", "pseudo", 0.1, R_ATR),
    ]
    PSEUDO_OF = {"real": "pseudo", "confirm": "confirm_pseudo", "pseudo": "pseudo"}
    for key, label, name, x, y in CONFIGS:
        row = {"key": key, "label": label, "stats": {}}
        for p in PERIODS:
            tr, P = SETS[(name, p)]
            A = tr.atr.values
            sp = spread_of(p, A)
            pnl, code, _, mins = bo.evaluate(P, x * A, y * A, DAY, sp)
            c = cell(pnl, code, A, y, x, tr.fill_t.values, ci=True)
            c.update(equity_stats(pnl / (R_ATR * A), pnl, tr.fill_t.values, len(tr) / months(tr.fill_t.values)))
            c["med_min_tp"] = rnd(np.median(mins[code == 1]) if (code == 1).any() else np.nan, 0)
            # chênh lệch với mốc giả cùng cách vào lệnh, cùng TP/SL
            trq, Pq = SETS[(PSEUDO_OF[name], p)]
            Aq = trq.atr.values
            pq, cq, _, _ = bo.evaluate(Pq, x * Aq, y * Aq, DAY, spread_of(p, Aq))
            Ra, Rb = pnl / (R_ATR * A), pq / (R_ATR * Aq)
            c["pseudo_E_R"] = rnd(Rb.mean())
            c["pseudo_win"] = rnd((pq > 0).mean())
            c["d_R"] = rnd(Ra.mean() - Rb.mean())
            c["d_win"] = rnd((pnl > 0).mean() - (pq > 0).mean())
            if name != "pseudo":
                c["d_lo"], c["d_hi"] = (rnd(v) for v in boot_diff_ci(Ra, tr.fill_t.values, Rb, trq.fill_t.values))
            cost_R = np.mean(sp / (R_ATR * A))
            c["est_R"] = rnd(c["d_R"] - cost_R)
            if p == "B":
                c["d_usd"] = rnd(pnl.mean() - pq.mean(), 3)
                c["est_usd"] = rnd(pnl.mean() - pq.mean() - SPREAD, 3)
            row["stats"][p] = c
            if p == "B" and key in ("lim_tp01", "lim_tp01_nosl", "conf_tp01"):
                o = np.argsort(tr.fill_t.values)
                cum = np.cumsum(pnl[o])
                ts = pd.DatetimeIndex(tr.fill_t.values[o])
                OUT["equity"][key] = {"t": [str(t.date()) for t in ts], "cum": [round(float(v), 1) for v in cum]}
            if p == "B" and key in ("lim_tp005", "lim_tp01", "lim_tp02", "conf_tp01", "conf_tp02"):
                OUT["spread_sens"][key] = {str(s_): rnd(bo.evaluate(P, x * A, y * A, DAY, s_)[0].mean(), 3) for s_ in [0.0, 0.15, 0.30, 0.50]}
        OUT["configs"].append(row)
        print(f"{label:58s}", {p: (row['stats'][p]['n'], row['stats'][p]['win'], row['stats'][p]['be_win'], row['stats'][p]['E_R'], row['stats'][p]['d_R'], row['stats'][p]['est_R']) for p in PERIODS}, flush=True)

    OUT["runtime_s"] = round(time.time() - t0)
    with open("out/bounce.json", "w") as f:
        json.dump(OUT, f, ensure_ascii=False)
    print("done", OUT["runtime_s"], "s")


if __name__ == "__main__":
    main()
