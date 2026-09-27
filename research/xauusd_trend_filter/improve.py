"""Thử nghiệm các phương án cải thiện module "Auto mốc phản ứng".

Mở rộng fxlevels: vào lệnh có xác nhận, quản lý lệnh (BE / trailing / chốt một phần / time stop),
đặc trưng chất lượng mốc tại thời điểm khớp để lọc, và lệnh Limit lệch khỏi mốc.
"""
import numpy as np
import pandas as pd

import fxlevels as fx


# ----------------------------------------------------------------------------- context bổ sung
def build_context(base):
    ctx = fx.build_context(base)
    h1, d1 = ctx["h1"], ctx["d1"]
    # RSI14 H1
    d = h1.close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    h1["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    h1["atr_rel"] = h1.atr / h1.atr.rolling(480, min_periods=100).median()  # so với 20 ngày
    d1["atr_d"] = fx.atr(d1)
    return ctx


# ----------------------------------------------------------------------------- lệnh ứng viên
def candidate_orders(base, ctx, sl_atr=0.75, offset=0.0, **kw):
    """Mọi lệnh Limit tại mốc tĩnh (không lọc RR). `offset` (ATR) dịch giá đặt lệnh:
    âm = sâu hơn mốc (giá tốt hơn, ít khớp), dương = trước mốc."""
    o = fx.make_orders(base, ctx, "static", sl_atr=sl_atr, rr_min=0.0, **kw)
    o["level0"] = o.level
    o["level"] = o.level0 - o.dir * offset * o.atr
    # TP giữ nguyên giá mục tiêu; SL đo từ giá vào mới
    o["tp_px"] = o.level0 + o.dir * o.tpd_struct
    o["tpd_struct"] = (o.tp_px - o.level) * o.dir
    o["rr_struct"] = o.tpd_struct / o.sld
    return o


def fill_limit(base, o):
    """Thời điểm/chỉ số khớp của lệnh Limit (chạm giá)."""
    t = base.index.values
    hi, lo = base.high.values, base.low.values
    i0 = np.searchsorted(t, o.t.values, "left")
    i1 = np.searchsorted(t, o.expire.values, "left")
    L, TRG = o.level.values, o.trig.values
    fi = np.full(len(o), -1)
    for j in range(len(o)):
        a, b = i0[j], i1[j]
        if b <= a:
            continue
        hit = (lo[a:b] <= L[j]) if TRG[j] == -1 else (hi[a:b] >= L[j])
        if hit.any():
            fi[j] = a + int(hit.argmax())
    tr = o.copy()
    tr["fill_i"] = fi
    tr = tr[fi >= 0].copy()
    tr["entry"] = tr.level
    tr["fill_t"] = t[tr.fill_i.values]
    tr["risk"] = tr.sld
    tr["sl_px"] = tr.entry - tr.dir * tr.risk
    return tr


def fill_confirm(base, o, tf="15min", strong=False, sl_from="level"):
    """Vào lệnh sau khi có nến `tf` chạm mốc và đóng cửa quay lại phía "đúng":
    Buy: low <= mốc và close > mốc (strong: close nằm ở 40% trên của nến). Vào ở giá đóng nến đó.
    SL: 0,75 ATR sau mốc (sl_from='level') hoặc từ giá vào (sl_from='entry'). TP giữ giá mục tiêu."""
    bars = fx.resample(base, tf)
    tb = bars.index.values
    tclose = bars.t_close.values
    bh, bl, bc = bars.high.values, bars.low.values, bars.close.values
    t = base.index.values
    i0 = np.searchsorted(tb, o.t.values, "left")
    i1 = np.searchsorted(tclose, o.expire.values, "right")  # nến phải đóng trước khi lệnh hết hạn
    L, D = o.level.values, o.dir.values
    ci = np.full(len(o), -1)
    for j in range(len(o)):
        a, b = i0[j], i1[j]
        if b <= a:
            continue
        h, l, c = bh[a:b], bl[a:b], bc[a:b]
        rng = np.where(h - l > 0, h - l, np.nan)
        if D[j] == 1:
            ok = (l <= L[j]) & (c > L[j])
            if strong:
                ok &= (c - l) / rng >= 0.6
        else:
            ok = (h >= L[j]) & (c < L[j])
            if strong:
                ok &= (h - c) / rng >= 0.6
        if ok.any():
            ci[j] = a + int(ok.argmax())
    tr = o.copy()
    tr["conf_i"] = ci
    tr = tr[ci >= 0].copy()
    tr["entry"] = bc[tr.conf_i.values]
    tr["fill_t"] = tclose[tr.conf_i.values]
    tr["fill_i"] = np.searchsorted(t, tr.fill_t.values, "left")
    tr = tr[tr.fill_i < len(t)].copy()
    if sl_from == "level":
        tr["sl_px"] = tr.level - tr.dir * tr.sld
    else:
        tr["sl_px"] = tr.entry - tr.dir * tr.sld
    tr["risk"] = (tr.entry - tr.sl_px) * tr.dir
    tr["tpd_struct"] = (tr.tp_px - tr.entry) * tr.dir
    tr["rr_struct"] = tr.tpd_struct / tr.risk
    return tr


# ----------------------------------------------------------------------------- mô phỏng có quản lý lệnh
def simulate_managed(base, tr, rule="fixed", tp_mode="struct", spread=0.30, max_hold="24h", partial_at=1.0, time_stop=None):
    """Đi từng nến sau khi khớp. Cùng nến chạm SL và TP -> tính SL. SL bị gap -> khớp ở giá mở.
    rule: fixed | be1 (dời SL về hoà vốn khi lời >= 1R) | trail1 (trailing 1R sau khi lời >= 1R)
          | partial (chốt 50% tại partial_at R, dời SL hoà vốn, phần còn lại tới TP)
    tp_mode: 'struct' hoặc số (bội R). time_stop: giờ, đóng lệnh nếu vẫn mở."""
    t = base.index.values
    op, hi, lo, cl = (base[c].values for c in ("open", "high", "low", "close"))
    n = len(t)
    step = pd.Timedelta(np.diff(t[:1000]).min())
    mh = int(pd.Timedelta(max_hold) / step)
    ts = int(pd.Timedelta(hours=time_stop) / step) if time_stop else None
    E, D, RK, SL0 = tr.entry.values, tr.dir.values, tr.risk.values, tr.sl_px.values
    FI = tr.fill_i.values
    TPD = tr.tpd_struct.values if tp_mode == "struct" else tp_mode * RK
    R = np.full(len(tr), np.nan)
    ET = np.full(len(tr), np.datetime64("NaT"), dtype="datetime64[ns]")
    for j in range(len(tr)):
        f = FI[j]
        e = min(f + mh, n)
        if ts is not None:
            e = min(e, f + ts + 1)
        en, dr, rk = E[j], D[j], RK[j]
        sl = SL0[j]
        tp = en + dr * TPD[j]
        be = en + dr * spread
        got = 0.0        # R đã thực hiện (chốt một phần)
        size = 1.0
        mfe_px = en
        k_exit, px_exit = None, None
        for k in range(f, e):
            # nến khớp: Limit khớp tại giá chạm, phần còn lại của nến có thể chạm SL; TP không tính ở nến khớp
            lo_k, hi_k = lo[k], hi[k]
            sl_hit = lo_k <= sl if dr == 1 else hi_k >= sl
            if sl_hit:
                px = sl
                if k > f:  # gap qua SL
                    px = min(sl, op[k]) if dr == 1 else max(sl, op[k])
                k_exit, px_exit = k, px
                break
            if k > f:
                tp_hit = hi_k >= tp if dr == 1 else lo_k <= tp
                if tp_hit:
                    k_exit, px_exit = k, tp
                    break
                if rule == "partial" and size == 1.0:
                    p_px = en + dr * partial_at * rk
                    p_hit = hi_k >= p_px if dr == 1 else lo_k <= p_px
                    if p_hit:
                        got += 0.5 * (partial_at - spread / rk)
                        size = 0.5
                        sl = be
                # cập nhật sau khi nến đóng
                mfe_px = max(mfe_px, hi_k) if dr == 1 else min(mfe_px, lo_k)
                mfe_r = dr * (mfe_px - en) / rk
                if rule in ("be1", "trail1") and mfe_r >= 1.0:
                    if rule == "be1":
                        sl = max(sl, be) if dr == 1 else min(sl, be)
                    else:
                        tr_px = mfe_px - dr * rk
                        sl = max(sl, tr_px) if dr == 1 else min(sl, tr_px)
        if k_exit is None:
            k_exit, px_exit = e - 1, cl[e - 1]
        R[j] = got + size * (dr * (px_exit - en) - spread) / rk
        ET[j] = t[k_exit]
    out = tr.copy()
    out["R"] = R
    out["exit_t"] = ET
    return out


# ----------------------------------------------------------------------------- đặc trưng tại thời điểm khớp
def add_features(base, ctx, tr):
    t = base.index.values
    hi, lo, cl = base.high.values, base.low.values, base.close.values
    step = pd.Timedelta(np.diff(t[:1000]).min())
    per_h = int(pd.Timedelta("1h") / step)
    h1, d1 = ctx["h1"], ctx["d1"]
    T = pd.DatetimeIndex(tr.fill_t.values)
    tr = tr.copy()
    tr["hour"] = T.hour
    tr["dow"] = T.dayofweek
    tr["session"] = pd.cut(T.hour, [-1, 7, 15, 23], labels=["Asia", "London", "NY"]).astype(str)
    tr["ema20"] = fx.asof(h1, "ema20", T)
    tr["rsi"] = fx.asof(h1, "rsi", T)
    tr["atr_rel"] = fx.asof(h1, "atr_rel", T)
    tr["atr_d"] = fx.asof(d1, "atr_d", T)
    # stretch: mốc nằm sâu bao nhiêu ATR so với EMA20 H1 theo hướng lệnh (buy: EMA20 - mốc)
    tr["stretch"] = tr.dir * (tr.ema20 - tr.level) / tr.atr
    tr["rsi_dir"] = np.where(tr.dir == 1, tr.rsi, 100 - tr.rsi)  # thấp = quá bán theo hướng lệnh
    fi = tr.fill_i.values
    # tốc độ tiếp cận: giá 1 giờ trước lúc khớp cách mốc bao nhiêu ATR
    prev = np.clip(fi - per_h, 0, len(t) - 1)
    tr["approach"] = np.abs(cl[prev] - tr.level.values) / tr.atr.values
    # biên độ ngày đã dùng (tính từ 00:00 server tới lúc khớp) / ATR ngày
    day0 = np.searchsorted(t, T.normalize().values, "left")
    used = np.array([hi[a:b + 1].max() - lo[a:b + 1].min() if b >= a else np.nan for a, b in zip(day0, fi)])
    tr["day_used"] = used / tr.atr_d.values
    # mốc đã bị chạm trước đó trong ngày (trước lúc đặt lệnh)?
    p0 = np.searchsorted(t, tr.t.values, "left")
    lv = tr.level.values
    touched = np.array([(lo[a:b].min() <= L <= hi[a:b].max()) if b > a else False for a, b, L in zip(day0, p0, lv)])
    tr["touched_before"] = touched
    tr["hrs_to_fill"] = (tr.fill_t - tr.t) / pd.Timedelta("1h")
    return tr


# ----------------------------------------------------------------------------- tiện ích thống kê
def boot_mean_ci(R, times, reps=2000, seed=7):
    """Khoảng tin cậy 95% của kỳ vọng, bootstrap theo khối tuần."""
    rng = np.random.default_rng(seed)
    R = np.asarray(R, float)
    wk = pd.DatetimeIndex(times).to_period("W").astype(str).values
    weeks = np.unique(wk)
    idx = {w: np.nonzero(wk == w)[0] for w in weeks}
    ms = []
    for _ in range(reps):
        s = np.concatenate([idx[w] for w in rng.choice(weeks, len(weeks))])
        ms.append(R[s].mean())
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def summary(R, times=None, ci=False):
    s = fx.stats(R, None, times)
    if ci and s["n"] > 0 and times is not None:
        s["lo"], s["hi"] = boot_mean_ci(R, times)
    return s
