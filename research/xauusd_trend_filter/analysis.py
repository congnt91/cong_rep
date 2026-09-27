"""Tổng hợp kết quả: thuận/ngược theo từng bộ lọc, bootstrap theo tuần, kịch bản lot/TP, lưới SL.
Chạy sau run_sim.py và run_grid.py. Kết quả: out/results.json"""
import json, pandas as pd, numpy as np, fxlevels as fx
rng = np.random.default_rng(7)
OUT = {}
def tag(ex):
    ex = ex.copy()
    ex["strong"] = np.where((ex.trend_h4==ex.trend_d1)&(ex.trend_d1==ex.swing_h4), ex.trend_h4, 0)
    ex["tr_sw"] = np.where(ex.trend_h4==ex.swing_h1, ex.trend_h4, 0)
    return ex
def R_at(ex, x):
    """R theo TP x (R) hoặc 'struct'."""
    if x == "struct":
        return ex["R_struct"].values
    return (np.where(ex.mfe >= x, x, np.where(ex.slhit, ex.slR, ex.rend)) - ex.cost).values
def exit_time(ex, x):
    return ex["exit_t_struct"].values if x=="struct" else ex["exit_t_1R"].values  # xấp xỉ thứ tự thời gian
def boot_diff(ex, rg, x, reps=2000):
    a = ex.dir.values == ex[rg].values; c = ex.dir.values == -ex[rg].values
    R = R_at(ex, x); wk = pd.DatetimeIndex(ex.t.values).to_period("W").astype(str).values
    weeks = np.unique(wk); idx = {w: np.nonzero(wk == w)[0] for w in weeks}
    d = []
    for _ in range(reps):
        s = np.concatenate([idx[w] for w in rng.choice(weeks, len(weeks))])
        ra, rc = R[s][a[s]], R[s][c[s]]
        d.append(ra.mean() - rc.mean())
    d = np.array(d)
    return float(R[a].mean() - R[c].mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())

def portfolio(ex, rg, al_lot, al_tp, ct_lot, ct_tp, neutral_as="counter"):
    al = ex.dir.values == ex[rg].values
    R = np.where(al, R_at(ex, al_tp), R_at(ex, ct_tp))
    w = np.where(al, al_lot, ct_lot).astype(float)
    t = np.where(al, exit_time(ex, al_tp), exit_time(ex, ct_tp))
    usd = R * ex.sld.values * w  # USD / 0.01 lot (XAUUSD: 1 giá = 1 USD / 0.01 lot)
    s = fx.stats(R, w, t)
    m = w > 0
    s["usd_001"] = float(usd[m].sum())
    s["exposure"] = float(w.sum())
    order = np.argsort(t[m]); eq = np.cumsum((R*w)[m][order])
    return s, pd.Series(eq, index=pd.DatetimeIndex(t[m][order]))

VARIANTS = [
    ("V0", "Tất cả lệnh (hiện tại)",            dict(al_lot=1, al_tp="struct", ct_lot=1,   ct_tp="struct")),
    ("V1", "Chỉ đánh thuận, bỏ lệnh ngược",     dict(al_lot=1, al_tp="struct", ct_lot=0,   ct_tp="struct")),
    ("V2", "Ngược: lot ½ + TP ngắn 0,5R",       dict(al_lot=1, al_tp="struct", ct_lot=0.5, ct_tp=0.5)),
    ("V3", "Ngược: chỉ giảm lot ½",             dict(al_lot=1, al_tp="struct", ct_lot=0.5, ct_tp="struct")),
    ("V4", "Ngược: chỉ TP ngắn 0,5R",           dict(al_lot=1, al_tp="struct", ct_lot=1,   ct_tp=0.5)),
]
REG_LABEL = {"trend_h4":"Trend H4 (EMA50/200)", "trend_d1":"Trend D1 (giá vs EMA50)", "swing_h4":"Swing H4 (BOS)", "swing_h1":"Swing H1 (BOS)", "strong":"Trend mạnh (H4+D1+Swing H4 đồng thuận)", "tr_sw":"Trend H4 + Swing H1 đồng thuận", "ema_h1":"EMA20/50 H1"}
for which in ["old", "new"]:
    D = OUT[which] = {}
    for fam in ["static", "ma"]:
        ex = tag(pd.read_pickle(f"out/ex_{which}_{fam}.pkl"))
        F = D[fam] = {"n": len(ex), "sl_med": float(ex.sld.median()), "cost_med": float(ex.cost.median()),
                      "period": [str(ex.t.min())[:10], str(ex.t.max())[:10]]}
        # thuận vs ngược theo từng định nghĩa
        rows = []
        for rg in ["trend_h4","trend_d1","swing_h4","swing_h1","ema_h1","strong","tr_sw"]:
            a = ex.dir == ex[rg]; c = ex.dir == -ex[rg]
            r = {"reg": rg, "label": REG_LABEL[rg], "n_al": int(a.sum()), "n_ct": int(c.sum())}
            for x in ["struct", 0.5, 1, 1.5, 2]:
                R = R_at(ex, x)
                r[f"al_{x}"] = float(R[a].mean()); r[f"ct_{x}"] = float(R[c].mean())
                r[f"alw_{x}"] = float((R[a] > 0).mean()); r[f"ctw_{x}"] = float((R[c] > 0).mean())
            if fam == "static":
                r["diff_struct"], r["lo_struct"], r["hi_struct"], r["p_struct"] = boot_diff(ex, rg, "struct")
                r["diff_1"], r["lo_1"], r["hi_1"], r["p_1"] = boot_diff(ex, rg, 1)
            rows.append(r)
        F["align"] = rows
        # đường kỳ vọng theo TP
        xs = np.round(np.arange(0.25, 3.01, 0.25), 2)
        F["curve_x"] = xs.tolist()
        F["curve"] = {}
        for rg in ["trend_h4", "strong", "swing_h1"]:
            a = ex[ex.dir == ex[rg]]; c = ex[ex.dir == -ex[rg]]
            F["curve"][rg] = {"al": fx.exp_curve(a, xs).round(4).tolist(), "ct": fx.exp_curve(c, xs).round(4).tolist()}
        F["curve"]["all"] = fx.exp_curve(ex, xs).round(4).tolist()
        # MFE phân phối
        F["mfe"] = {}
        for rg in ["trend_h4", "strong"]:
            a = ex[ex.dir == ex[rg]]; c = ex[ex.dir == -ex[rg]]
            F["mfe"][rg] = {"al": [float((a.mfe >= x).mean()) for x in xs], "ct": [float((c.mfe >= x).mean()) for x in xs]}
        if fam != "static":
            continue
        # kịch bản danh mục
        F["variants"] = {}
        F["eq"] = {}
        for rg in ["trend_h4", "strong", "swing_h1"]:
            F["variants"][rg] = []
            for code, name, kw in VARIANTS:
                s, eq = portfolio(ex, rg, **kw)
                s.update(code=code, name=name)
                F["variants"][rg].append(s)
                if rg in ("trend_h4", "strong") and code in ("V0", "V1", "V2"):
                    # đường vốn theo ngày
                    e = eq.groupby(eq.index.normalize()).last()
                    if which == "old":
                        e = e.resample("W").last().dropna()
                    F["eq"][f"{rg}_{code}"] = {"t": [str(x)[:10] for x in e.index], "v": e.round(2).tolist()}
        # theo năm (old) / theo tháng (new): chênh lệch thuận - ngược
        per = pd.DatetimeIndex(ex.t.values).year if which == "old" else pd.DatetimeIndex(ex.t.values).to_period("Q").astype(str)
        yr = []
        for p in sorted(pd.unique(per)):
            m = per == p; e2 = ex[m]
            row = {"p": str(p), "n": int(m.sum()), "all": float(R_at(e2, "struct").mean())}
            for rg in ["trend_h4", "strong", "swing_h1"]:
                a = e2.dir == e2[rg]; c = e2.dir == -e2[rg]
                row[rg] = [float(R_at(e2, "struct")[a.values].mean()) if a.any() else None, float(R_at(e2, "struct")[c.values].mean()) if c.any() else None]
            yr.append(row)
        F["by_period"] = yr
        # theo loại mốc
        lt = []
        for typ in fx.STATIC_TYPES:
            e2 = ex[ex.type == typ]
            if len(e2) < 20: continue
            a = e2.dir == e2.trend_h4; c = e2.dir == -e2.trend_h4
            R = R_at(e2, "struct")
            lt.append({"type": typ, "n": len(e2), "all": float(R.mean()), "al": float(R[a.values].mean()), "ct": float(R[c.values].mean()), "n_al": int(a.sum()), "n_ct": int(c.sum())})
        F["by_type"] = lt
        # Long / Short
        F["by_dir"] = {str(d): {"n": int((ex.dir==d).sum()), "E": float(R_at(ex[ex.dir==d], "struct").mean())} for d in (1, -1)}
        if which == "new":
            F["autoclose"] = {"struct": fx.stats(ex.R_struct), "auto": fx.stats(ex.R_auto)}
# lưới SL x loại lệnh
G = OUT["grid"] = []
for which in ["old", "new"]:
    for otype, fam in [("limit","static"),("stop","static"),("limit","ma")]:
        for sl in [0.5, 0.75, 1.0, 1.5]:
            ex = tag(pd.read_pickle(f"out/grid_{which}_{otype}_{fam}_{sl}.pkl"))
            r = {"which": which, "otype": otype, "fam": fam, "sl": sl, "n": len(ex), "cost": float(ex.cost.median())}
            for x in [0.5, 1, 1.5, 2]:
                r[f"all_{x}"] = float(fx.exp_curve(ex, [x])[0])
                for rg in ["trend_h4", "strong"]:
                    r[f"{rg}_al_{x}"] = float(fx.exp_curve(ex[ex.dir == ex[rg]], [x])[0])
                    r[f"{rg}_ct_{x}"] = float(fx.exp_curve(ex[ex.dir == -ex[rg]], [x])[0])
            G.append(r)
json.dump(OUT, open("out/results.json", "w"), ensure_ascii=False, default=float)
print("saved")
