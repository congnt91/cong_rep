"""Backtest lệnh chờ tại các mốc phản ứng (pivot / PDH-PDL / Fibo / EMA) cho XAUUSD,
so sánh hiệu quả khi lọc theo Trend (khung lớn) hoặc Swing (cấu trúc H1/H4).

Mọi chỉ báo dùng nến ĐÃ ĐÓNG trước thời điểm đặt lệnh (không nhìn trước).
Giá dữ liệu là mid/bid; chi phí spread trừ trực tiếp vào mỗi lệnh.
"""
import numpy as np
import pandas as pd

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


# ----------------------------------------------------------------------------- data
def load_new():
    """M1 XAUUSD 2025-10 -> 2026-09 (UTC) -> đổi sang giờ server MT5 (NY + 7h)."""
    d = pd.read_pickle("data/nav_m1.pkl")[["open", "high", "low", "close"]].copy()
    idx = d.index.tz_localize("UTC").tz_convert("America/New_York") + pd.Timedelta(hours=7)
    d.index = idx.tz_localize(None)
    return d.sort_index()


def load_old():
    """M15 XAUUSD 2012-05 -> 2022-03, đã ở giờ server MT5."""
    return pd.read_pickle("data/old_m15.pkl")[["open", "high", "low", "close"]].copy()


def resample(df, rule):
    r = df.resample(rule, label="left", closed="left").agg(AGG).dropna()
    r["t_close"] = r.index + pd.Timedelta(rule)  # thời điểm nến đóng (dữ liệu có sẵn)
    return r


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def swing_state(df, k):
    """Cấu trúc swing kiểu BOS: fractal k nến mỗi bên (xác nhận trễ k nến).
    +1 khi đóng cửa vượt đỉnh swing gần nhất, -1 khi thủng đáy swing gần nhất."""
    h, l, c = df.high.values, df.low.values, df.close.values
    n = len(df)
    out = np.zeros(n, dtype=np.int8)
    sh = sl = np.nan
    st = 0
    for i in range(n):
        j = i - k
        if j >= k:
            if h[j] == h[j - k:j + k + 1].max():
                sh = h[j]
            if l[j] == l[j - k:j + k + 1].min():
                sl = l[j]
        if c[i] > sh:
            st = 1
        elif c[i] < sl:
            st = -1
        out[i] = st
    return out


def asof(series_df, col, times):
    """Giá trị `col` của nến gần nhất đã đóng tại thời điểm `times`."""
    s = series_df[["t_close", col]].sort_values("t_close")
    left = pd.DataFrame({"t": times})
    m = pd.merge_asof(left.sort_values("t"), s, left_on="t", right_on="t_close", direction="backward")
    return m.set_index(left.sort_values("t").index).sort_index()[col].values


# ----------------------------------------------------------------------------- context
def build_context(base):
    h1 = resample(base, "1h")
    h4 = resample(base, "4h")
    d1 = resample(base, "1D")
    h8 = resample(base, "8h")
    h1["atr"] = atr(h1)
    for n in (20, 50, 200):
        h1[f"ema{n}"] = ema(h1.close, n)
    h4["ema50"], h4["ema200"] = ema(h4.close, 50), ema(h4.close, 200)
    d1["ema50"] = ema(d1.close, 50)
    # regime (dấu +1/-1)
    h4["trend_h4"] = np.where(h4.ema50 > h4.ema200, 1, -1)
    d1["trend_d1"] = np.where(d1.close > d1.ema50, 1, -1)
    h1["ema_h1"] = np.where(h1.ema20 > h1.ema50, 1, -1)
    h1["swing_h1"] = swing_state(h1, 3)
    h4["swing_h4"] = swing_state(h4, 2)
    return dict(h1=h1, h4=h4, d1=d1, h8=h8)


REGIMES = ["trend_h4", "trend_d1", "swing_h4", "swing_h1", "ema_h1"]
STATIC_TYPES = ["PDH", "PDL", "D_P", "D_R1", "D_S1", "FIB50", "S8_P", "S8_R1", "S8_S1", "H12_R1", "H12_S1"]
MA_TYPES = ["EMA20", "EMA50", "EMA200"]


def _regime_at(ctx, times):
    out = {}
    out["trend_h4"] = asof(ctx["h4"], "trend_h4", times)
    out["swing_h4"] = asof(ctx["h4"], "swing_h4", times)
    out["trend_d1"] = asof(ctx["d1"], "trend_d1", times)
    out["swing_h1"] = asof(ctx["h1"], "swing_h1", times)
    out["ema_h1"] = asof(ctx["h1"], "ema_h1", times)
    return out


def static_levels(base, ctx):
    """Mốc tĩnh, làm mới đầu mỗi phiên 8h (00/08/16 giờ server)."""
    h8 = ctx["h8"]
    times = h8.index[1:]  # thời điểm bắt đầu phiên = lúc đặt lệnh
    T = pd.DatetimeIndex(times)
    d1 = ctx["d1"]
    pdh = asof(d1, "high", T)
    pdl = asof(d1, "low", T)
    pdc = asof(d1, "close", T)
    P = (pdh + pdl + pdc) / 3
    prev8 = h8.shift(1).loc[times]  # phiên 8h liền trước
    p8 = (prev8.high + prev8.low + prev8.close).values / 3
    # 12h trượt
    h1 = ctx["h1"]
    r12h = h1.high.rolling(12).max()
    r12l = h1.low.rolling(12).min()
    tmp = pd.DataFrame({"t_close": h1.t_close, "hh": r12h, "ll": r12l, "cc": h1.close}).dropna()
    hh, ll, cc = asof(tmp, "hh", T), asof(tmp, "ll", T), asof(tmp, "cc", T)
    p12 = (hh + ll + cc) / 3
    lv = pd.DataFrame(index=T)
    lv["PDH"], lv["PDL"] = pdh, pdl
    lv["D_P"], lv["D_R1"], lv["D_S1"] = P, 2 * P - pdl, 2 * P - pdh
    lv["FIB50"] = (pdh + pdl) / 2
    lv["S8_P"] = p8
    lv["S8_R1"] = 2 * p8 - prev8.low.values
    lv["S8_S1"] = 2 * p8 - prev8.high.values
    lv["H12_R1"] = 2 * p12 - ll
    lv["H12_S1"] = 2 * p12 - hh
    lv["atr"] = asof(h1, "atr", T)
    lv["expire"] = T + pd.Timedelta("8h")
    lv["block"] = np.arange(len(T))
    return lv.dropna()


def ma_levels(base, ctx):
    """EMA H1, làm mới mỗi giờ (lệnh chờ bám theo đường MA)."""
    h1 = ctx["h1"]
    T = pd.DatetimeIndex(h1.t_close.values[:-1])
    lv = pd.DataFrame(index=T)
    for n in (20, 50, 200):
        lv[f"EMA{n}"] = asof(h1, f"ema{n}", T)
    lv["atr"] = asof(h1, "atr", T)
    lv["expire"] = T + pd.Timedelta("1h")
    # block 8h dùng để giới hạn 1 lần khớp / MA / phiên
    lv["block"] = ((T - T[0].normalize()) // pd.Timedelta("8h")).astype(int)
    return lv.dropna()


# ----------------------------------------------------------------------------- orders
def make_orders(base, ctx, family, min_d=0.15, max_d=3.0, merge=0.25, sl_atr=0.75, rr_min=1.0, tp_cap=3.0, otype="limit"):
    """Sinh lệnh chờ: mốc dưới giá -> Buy Limit, mốc trên giá -> Sell Limit.
    TP cấu trúc = mốc kế tiếp theo hướng lệnh; bỏ lệnh nếu RR < rr_min (giống Nola-X)."""
    stat = static_levels(base, ctx)
    lv = stat if family == "static" else ma_levels(base, ctx)
    types = STATIC_TYPES if family == "static" else MA_TYPES
    closes = base.close
    tbase = base.index.values
    # giá hiện tại = close M1/M15 cuối cùng trước T
    pos = np.searchsorted(tbase, lv.index.values, side="left") - 1
    ok = pos >= 0
    lv, pos = lv[ok], pos[ok]
    price = closes.values[pos]
    # mốc tĩnh gần nhất để tính TP cấu trúc (cho cả family MA)
    st_al = stat.reindex(lv.index, method="ffill")[STATIC_TYPES].values
    rows = []
    A = lv["atr"].values
    for typ in types:
        L = lv[typ].values
        d = L - price
        dist = np.abs(d) / A
        good = (dist >= min_d) & (dist <= max_d)
        for i in np.nonzero(good)[0]:
            direc = 1 if d[i] < 0 else -1
            if otype == "stop":
                direc = -direc  # mốc trên giá -> Buy Stop, dưới giá -> Sell Stop
            rows.append((i, typ, L[i], direc))
    o = pd.DataFrame(rows, columns=["i", "type", "level", "dir"])
    o["t"] = lv.index.values[o.i]
    o["expire"] = lv["expire"].values[o.i]
    o["block"] = lv["block"].values[o.i]
    o["atr"] = A[o.i]
    o["price"] = price[o.i]
    o["trig"] = np.where(o.level > o.price, 1, -1)  # +1: khớp khi high>=level, -1: khi low<=level
    o["sld"] = sl_atr * o.atr
    # gộp mốc trùng (cùng phiên, cùng hướng, cách nhau < merge*ATR): giữ mốc gần giá nhất
    if family == "static":
        o = o.sort_values(["i", "dir", "level"])
        keep = np.ones(len(o), bool)
        conf = np.ones(len(o), int)
        vals = o[["i", "dir", "level", "atr", "price"]].values
        order = np.argsort(np.abs(vals[:, 2] - vals[:, 4]))  # gần giá trước
        kept = {}
        for r in order:
            key = (vals[r, 0], vals[r, 1])
            lst = kept.setdefault(key, [])
            dup = [q for q in lst if abs(vals[q, 2] - vals[r, 2]) < merge * vals[r, 3]]
            if dup:
                keep[r] = False
                conf[dup[0]] += 1
            else:
                lst.append(r)
        o["confluence"] = conf
        o = o[keep]
    else:
        o["confluence"] = 1
    # TP cấu trúc
    tps = []
    for r in o.itertuples():
        cand = st_al[r.i]
        cand = cand[~np.isnan(cand)]
        if r.dir == 1:
            c = cand[cand > r.level + 0.1 * r.atr]
            tp = c.min() - r.level if len(c) else np.nan
        else:
            c = cand[cand < r.level - 0.1 * r.atr]
            tp = r.level - c.max() if len(c) else np.nan
        tps.append(tp)
    o["tpd_struct"] = np.minimum(np.array(tps), tp_cap * o.sld.values)
    o["rr_struct"] = o.tpd_struct / o.sld
    o = o[o.rr_struct >= rr_min].copy()
    reg = _regime_at(ctx, pd.DatetimeIndex(o.t.values))
    for kk, v in reg.items():
        o[kk] = v
    o["family"] = family
    o["otype"] = otype
    return o.reset_index(drop=True)


# ----------------------------------------------------------------------------- simulate
TP_MODES = {"struct": None, "0.5R": 0.5, "0.75R": 0.75, "1R": 1.0, "1.5R": 1.5, "2R": 2.0}


def simulate(base, o, spread=0.30, max_hold="24h", autoclose=None):
    """Mô phỏng từng lệnh độc lập. Cùng nến chạm cả SL và TP -> tính SL (bảo thủ).
    Trả về cột R (theo đơn vị SL) và pnl giá (USD / 0.01 lot) cho từng chế độ TP."""
    t = base.index.values
    op, hi, lo, cl = (base[c].values for c in ("open", "high", "low", "close"))
    n = len(t)
    mh = int(pd.Timedelta(max_hold) / pd.Timedelta(np.diff(t[:1000]).min()))
    i0s = np.searchsorted(t, o.t.values, "left")
    i1s = np.searchsorted(t, o.expire.values, "left")
    res = {f"R_{m}": np.full(len(o), np.nan) for m in TP_MODES}
    res.update({f"exit_{m}": np.full(len(o), np.nan) for m in TP_MODES})
    fill_t = np.full(len(o), np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_t = {m: np.full(len(o), np.datetime64("NaT"), dtype="datetime64[ns]") for m in TP_MODES}
    L, D, SLd, TPs = o.level.values, o.dir.values, o.sld.values, o.tpd_struct.values
    TRG = o.trig.values if "trig" in o else -D
    ac_R = np.full(len(o), np.nan)
    ac_t = np.full(len(o), np.datetime64("NaT"), dtype="datetime64[ns]")
    for j in range(len(o)):
        a, b = i0s[j], i1s[j]
        if b <= a:
            continue
        lv, dr, sld = L[j], D[j], SLd[j]
        hit = (lo[a:b] <= lv) if TRG[j] == -1 else (hi[a:b] >= lv)
        if not hit.any():
            continue
        f = a + int(hit.argmax())
        fill_t[j] = t[f]
        e = min(f + mh, n)
        sl = lv - dr * sld
        slhit = (lo[f:e] <= sl) if dr == 1 else (hi[f:e] >= sl)
        if TRG[j] == dr and len(slhit):  # lệnh stop: ở nến khớp chỉ tính SL nếu đóng cửa đã qua SL
            slhit[0] = (cl[f] <= sl) if dr == 1 else (cl[f] >= sl)
        ks = int(slhit.argmax()) if slhit.any() else None
        if ks is not None and ks > 0:
            # gap qua SL -> khớp ở giá mở cửa
            sl_px = min(sl, op[f + ks]) if dr == 1 else max(sl, op[f + ks])
        else:
            sl_px = sl
        for m, mult in TP_MODES.items():
            tpd = TPs[j] if mult is None else mult * sld
            tp = lv + dr * tpd
            tphit = (hi[f + 1:e] >= tp) if dr == 1 else (lo[f + 1:e] <= tp)
            kt = int(tphit.argmax()) + 1 if tphit.any() else None
            if ks is not None and (kt is None or ks <= kt):
                px, k = sl_px, ks
            elif kt is not None:
                px, k = tp, kt
            else:
                px, k = cl[e - 1], e - 1 - f
            pnl = dr * (px - lv) - spread
            res[f"R_{m}"][j] = pnl / sld
            res[f"exit_{m}"][j] = pnl
            exit_t[m][j] = t[f + k]
        if autoclose is not None:
            # Tự chốt: giữ >= hold phút và lời >= thr giá -> đóng (kiểm tra mỗi 15 phút), TP cấu trúc vẫn giữ
            hold_min, thr = autoclose
            tpd = TPs[j]
            tp = lv + dr * tpd
            tphit = (hi[f + 1:e] >= tp) if dr == 1 else (lo[f + 1:e] <= tp)
            kt = int(tphit.argmax()) + 1 if tphit.any() else None
            el = (t[f:e] - t[f]) / np.timedelta64(1, "m")
            mins = pd.DatetimeIndex(t[f:e]).minute.values
            chk = (el >= hold_min) & (mins % 15 == 0) & (dr * (cl[f:e] - lv) - spread >= thr)
            kc = int(chk.argmax()) if chk.any() else None
            cands = [(ks, 0, sl_px), (kt, 1, tp), (kc, 2, None)]
            cands = [c for c in cands if c[0] is not None]
            if cands:
                k, _, px = min(cands, key=lambda c: (c[0], c[1]))
                if px is None:
                    px = cl[f + k]
            else:
                k, px = e - 1 - f, cl[e - 1]
            ac_R[j] = (dr * (px - lv) - spread) / sld
            ac_t[j] = t[f + k]
    out = o.copy()
    out["fill_t"] = fill_t
    for m in TP_MODES:
        out[f"R_{m}"] = res[f"R_{m}"]
        out[f"usd_{m}"] = res[f"exit_{m}"]
        out[f"exit_t_{m}"] = exit_t[m]
    if autoclose is not None:
        out["R_auto"] = ac_R
        out["exit_t_auto"] = ac_t
    return out


def one_fill_per_block(tr):
    """MA: tối đa 1 lần khớp / loại MA / phiên 8h."""
    tr = tr.dropna(subset=["fill_t"]).sort_values("fill_t")
    return tr.drop_duplicates(["block", "type"], keep="first")


# ----------------------------------------------------------------------------- metrics
def stats(R, w=None, times=None):
    R = np.asarray(R, float)
    w = np.ones_like(R) if w is None else np.asarray(w, float)
    m = ~np.isnan(R) & (w > 0)
    R, w = R[m], w[m]
    if len(R) == 0:
        return dict(n=0)
    x = R * w
    if times is not None:
        order = np.argsort(np.asarray(times)[m])
        x_ord = x[order]
    else:
        x_ord = x
    eq = np.cumsum(x_ord)
    dd = (np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq).max()
    gw, gl = x[x > 0].sum(), -x[x < 0].sum()
    return dict(
        n=int(len(R)),
        win=float((R > 0).mean()),
        avgR=float(x.sum() / w.sum()),
        totR=float(x.sum()),
        pf=float(gw / gl) if gl > 0 else np.inf,
        maxdd=float(dd),
        ret_dd=float(x.sum() / dd) if dd > 0 else np.inf,
    )


def excursions(base, tr, spread=0.30, max_hold="24h"):
    """MFE (tính bằng R) trước khi chạm SL, cờ chạm SL, R cuối kỳ nếu không chạm SL.
    Dùng để tính kỳ vọng theo mọi mức TP: E(x) = x nếu MFE>=x, ngược lại -1 (SL) hoặc R lúc hết giờ."""
    t = base.index.values
    op, hi, lo, cl = (base[c].values for c in ("open", "high", "low", "close"))
    n = len(t)
    mh = int(pd.Timedelta(max_hold) / pd.Timedelta(np.diff(t[:1000]).min()))
    fs = np.searchsorted(t, tr.fill_t.values, "left")
    L, D, S = tr.level.values, tr.dir.values, tr.sld.values
    TRG = tr.trig.values if "trig" in tr else -D
    mfe = np.zeros(len(tr)); mae = np.zeros(len(tr)); slh = np.zeros(len(tr), bool); rend = np.zeros(len(tr)); slR = np.full(len(tr), -1.0)
    for j in range(len(tr)):
        f = fs[j]; e = min(f + mh, n); lv, dr, sd = L[j], D[j], S[j]
        sl = lv - dr * sd
        if dr == 1:
            fav = (hi[f + 1:e] - lv) / sd; adv = (lv - lo[f:e]) / sd
            hit = lo[f:e] <= sl
        else:
            fav = (lv - lo[f + 1:e]) / sd; adv = (hi[f:e] - lv) / sd
            hit = hi[f:e] >= sl
        if TRG[j] == dr and len(hit):
            hit[0] = (cl[f] <= sl) if dr == 1 else (cl[f] >= sl)
        if hit.any():
            ks = int(hit.argmax()); slh[j] = True
            mfe[j] = fav[:max(ks - 1, 0)].max() if ks >= 2 else 0.0  # fav bắt đầu từ f+1
            px = min(sl, op[f + ks]) if (dr == 1 and ks > 0) else (max(sl, op[f + ks]) if ks > 0 else sl)
            slR[j] = dr * (px - lv) / sd
            mae[j] = 1.0
        else:
            mfe[j] = fav.max() if len(fav) else 0.0
            mae[j] = adv.max()
            rend[j] = dr * (cl[e - 1] - lv) / sd
    out = tr.copy()
    out["mfe"], out["mae"], out["slhit"], out["rend"], out["slR"] = mfe, mae, slh, rend, slR
    out["cost"] = spread / out.sld
    return out


def exp_curve(ex, xs):
    """Kỳ vọng (R) theo mức TP x (đơn vị R), đã trừ spread."""
    res = []
    for x in xs:
        R = np.where(ex.mfe >= x, x, np.where(ex.slhit, ex.slR, ex.rend)) - ex.cost
        res.append(R.mean())
    return np.array(res)
