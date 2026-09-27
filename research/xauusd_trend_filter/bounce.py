"""Tỉ lệ bật tại mốc phản ứng và chiến lược "ăn đoạn bật" (TP mỏng).

Với mỗi lần giá chạm mốc (lệnh Limit khớp), dựng đường giá sau đó theo từng nến rồi hỏi:
giá chạy thuận được x trước khi chạy ngược y hay không. So với "mốc giả" (giá ngẫu nhiên cùng
khoảng cách, cách xa mọi mốc thật) và với lý thuyết bước ngẫu nhiên P = y / (x + y).

Thứ tự trong nến:
- Nến khớp: đã biết chạm mốc -> đáy/đỉnh ngược -> đóng cửa. Chỉ tính phần thuận bằng giá đóng cửa,
  phần ngược tính trước (không dùng đỉnh/đáy thuận của nến khớp vì có thể xảy ra trước khi chạm).
- Nến sau: nếu cùng nến chạm cả TP và SL -> 'cons' tính SL; 'neutral' đoán theo màu nến
  (nến tăng O-L-H-C, nến giảm O-H-L-C), 'opt' tính TP. Gap qua SL khớp ở giá mở.
"""
import numpy as np
import pandas as pd

import fxlevels as fx
import improve as im


# ----------------------------------------------------------------------------- lệnh
def real_orders(base, ctx):
    """Mọi lệnh Limit tại mốc tĩnh (không lọc RR)."""
    return im.candidate_orders(base, ctx)


def pseudo_orders(base, ctx, o_real, seed=11, min_gap=0.25):
    """Mốc giả: mỗi lệnh thật sinh một mức giá cùng phiên, cùng hướng, khoảng cách lấy ngẫu nhiên
    theo phân bố khoảng cách của mốc thật, cách mọi mốc thật (11 loại, chưa gộp) >= min_gap ATR."""
    rng = np.random.default_rng(seed)
    lv = fx.static_levels(base, ctx)
    raw = lv[fx.STATIC_TYPES].values
    blk = {b: i for i, b in enumerate(lv.block.values)}
    dist_pool = (np.abs(o_real.level0 - o_real.price) / o_real.atr).values
    rows = []
    for r in o_real.itertuples():
        R = raw[blk[r.block]]
        for _ in range(40):
            u = rng.choice(dist_pool) * rng.uniform(0.85, 1.15)
            L = r.price - r.dir * u * r.atr
            if np.min(np.abs(R - L)) >= min_gap * r.atr:
                rows.append((r.Index, L))
                break
    idx, L = zip(*rows)
    p = o_real.loc[list(idx)].copy()
    p["level"] = np.array(L)
    p["level0"] = p.level
    p["trig"] = -p.dir
    p["type"] = "PSEUDO"
    return p.reset_index(drop=True)


def touches(base, o, entry="limit"):
    """entry='limit': khớp khi chạm mốc; 'confirm': vào ở giá đóng nến M15 chạm mốc rồi đóng cửa quay lại."""
    if entry == "limit":
        return im.fill_limit(base, o)
    return im.fill_confirm(base, o, tf="15min", sl_from="level")


# ----------------------------------------------------------------------------- đường giá
def paths(base, tr, hours=25, at_level=True, fill_mode="neutral"):
    """Ma trận (lệnh x nến) cho phần thuận/ngược tính từ giá vào, tối đa `hours` giờ.
    at_level=True: nến đầu là nến chạm mốc (xử lý như mô tả ở đầu file).
    at_level=False: vào lệnh ở đầu nến đầu (lệnh xác nhận).
    fill_mode (nến khớp, phần ngược luôn tính trước phần thuận, trừ 'opt'):
      'cons'    phần thuận chỉ tính tới giá đóng cửa (cận dưới);
      'neutral' nến đóng cùng chiều lệnh (O-L-H-C với Buy) thì tính cả đỉnh/đáy thuận, ngược lại chỉ giá đóng cửa;
      'opt'     coi đỉnh/đáy thuận xảy ra sau lúc chạm và trước phần ngược (cận trên, lỏng)."""
    t = base.index.values
    op, hi, lo, cl = (base[c].values for c in ("open", "high", "low", "close"))
    n = len(t)
    step = pd.Timedelta(np.diff(t[:1000]).min())
    H = int(pd.Timedelta(f"{hours}h") / step)
    f = tr.fill_i.values.astype(int)
    idx = f[:, None] + np.arange(H)[None, :]
    valid = idx < n
    idx = np.minimum(idx, n - 1)
    D = tr.dir.values[:, None]
    E = tr.entry.values[:, None]
    fav = np.where(D == 1, hi[idx] - E, E - lo[idx])
    adv = np.where(D == 1, E - lo[idx], hi[idx] - E)
    oadv = np.where(D == 1, E - op[idx], op[idx] - E)
    clr = D * (cl[idx] - E)
    adv_first = np.where(D == 1, cl[idx] > op[idx], cl[idx] < op[idx]) | (cl[idx] == op[idx])
    if at_level:
        with_dir = np.where(D[:, 0] == 1, cl[idx[:, 0]] > op[idx[:, 0]], cl[idx[:, 0]] < op[idx[:, 0]])
        close_only = np.maximum(clr[:, 0], 0.0)
        if fill_mode == "cons":
            fav[:, 0] = close_only
        elif fill_mode == "neutral":
            fav[:, 0] = np.where(with_dir, np.maximum(fav[:, 0], 0.0), close_only)
        adv[:, 0] = np.maximum(adv[:, 0], 0.0)
        oadv[:, 0] = 0.0
        adv_first[:, 0] = fill_mode != "opt"
    el = (t[idx] - t[f][:, None]) / np.timedelta64(1, "m")
    el = np.where(valid, el, np.inf)
    f32 = lambda a: a.astype(np.float32)
    return dict(fav=f32(fav), adv=f32(adv), oadv=f32(oadv), clr=f32(clr), adv_first=adv_first, el=f32(el),
                step_min=step / pd.Timedelta("1min"), atr=tr.atr.values, n=len(tr))


def _first(mask):
    k = mask.argmax(1)
    return np.where(mask.any(1), k, 10**9)


def evaluate(P, x, y, T_min=24 * 60, spread=0.30, tie="neutral"):
    """Kết quả từng lệnh với TP cách x, SL cách y (giá, mảng theo lệnh; y = inf: không SL),
    đóng lệnh sau T_min phút nếu chưa chạm. Trả về pnl (USD / 0.01 lot, đã trừ spread) và mã kết quả."""
    n = P["n"]
    x = np.broadcast_to(np.asarray(x, float), (n,))
    y = np.broadcast_to(np.asarray(y, float), (n,))
    within = P["el"] < T_min - 1e-9
    # nến cuối còn trong thời gian: nến bắt đầu trước T và (với M15) đóng muộn nhất lúc T
    kT = within.sum(1)
    kt = _first((P["fav"] >= x[:, None]) & within)
    ks = _first((P["adv"] >= y[:, None]) & within)
    rows = np.arange(n)
    both = (kt == ks) & (kt < 10**9)
    if tie == "cons":
        sl_first_tie = np.ones(n, bool)
    elif tie == "opt":
        sl_first_tie = np.zeros(n, bool)
    else:
        k = np.minimum(kt, P["fav"].shape[1] - 1)
        gap_sl = P["oadv"][rows, k] >= y
        gap_tp = -P["oadv"][rows, k] >= x
        sl_first_tie = np.where(gap_sl, True, np.where(gap_tp, False, P["adv_first"][rows, k]))
    is_sl = (ks < kt) | (both & sl_first_tie)
    is_tp = (kt < ks) | (both & ~sl_first_tie)
    ksc = np.minimum(ks, P["fav"].shape[1] - 1)
    sl_px = np.where(ksc > 0, np.maximum(y, P["oadv"][rows, ksc]), y)
    last = np.maximum(kT - 1, 0)
    pnl = np.where(is_tp, x, np.where(is_sl, -sl_px, P["clr"][rows, last])) - spread
    code = np.where(is_tp, 1, np.where(is_sl, -1, 0))
    k_exit = np.where(is_tp, kt, np.where(is_sl, ks, last))
    mins = P["el"][rows, np.minimum(k_exit, P["el"].shape[1] - 1)]
    return pnl, code, both, mins


def bounce_prob(P, x_atr, y_atr, T_min=24 * 60, tie="neutral"):
    """P(chạy thuận x ATR trước khi ngược y ATR), không tính spread."""
    _, code, _, _ = evaluate(P, x_atr * P["atr"], y_atr * P["atr"], T_min, spread=0.0, tie=tie)
    return float((code == 1).mean())


def mfe_before(P, y_atr, T_min=24 * 60):
    """Biên độ bật lớn nhất (ATR) trước khi giá đi ngược y ATR (hoặc hết giờ)."""
    within = P["el"] < T_min - 1e-9
    ks = _first((P["adv"] >= (y_atr * P["atr"])[:, None]) & within)
    H = P["fav"].shape[1]
    mask = (np.arange(H)[None, :] < np.minimum(ks, H)[:, None]) & within
    m = np.where(mask, P["fav"], 0.0).max(1)
    return m / P["atr"]


def mae_before(P, x_atr, T_min=24 * 60):
    """Giá đi ngược sâu nhất (ATR) trước khi bật được x ATR (hoặc hết giờ)."""
    within = P["el"] < T_min - 1e-9
    kt = _first((P["fav"] >= (x_atr * P["atr"])[:, None]) & within)
    H = P["fav"].shape[1]
    # tính cả nến chạm TP (phần ngược của nến đó có thể xảy ra trước) -> bảo thủ
    mask = (np.arange(H)[None, :] <= np.minimum(kt, H - 1)[:, None]) & within
    m = np.where(mask, P["adv"], 0.0).max(1)
    return m / P["atr"], kt < 10**9
