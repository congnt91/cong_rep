"""Bộ chỉ báo đo tại thời điểm vào lệnh, dùng để kiểm tra chỉ báo nào giúp mốc phản ứng.

Mọi giá trị lấy từ nến ĐÃ ĐÓNG tại hoặc trước thời điểm vào lệnh (không nhìn trước).
Chỉ báo có hướng được nhân theo hướng lệnh (dir = +1 Buy, -1 Sell):
  - đo "độ dốc / động lượng" x dir: dương = đang đi cùng hướng lệnh;
  - dao động 0-100 đổi thành "_dir": thấp = quá bán theo hướng lệnh (giá đã giảm nhiều trước khi Buy).
"""
import numpy as np
import pandas as pd

import fxlevels as fx

AGGV = {"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"}


# ----------------------------------------------------------------------------- dữ liệu có volume
def load_with_volume(which):
    if which == "old":
        d = pd.read_pickle("data/old_m15.pkl").rename(columns={"tick_volume": "vol"})
    else:
        d = pd.read_pickle("data/nav_m1.pkl").rename(columns={"volume": "vol"})
        idx = d.index.tz_localize("UTC").tz_convert("America/New_York") + pd.Timedelta(hours=7)
        d.index = idx.tz_localize(None)
    return d[["open", "high", "low", "close", "vol"]].sort_index()


def rs(df, rule):
    r = df.resample(rule, label="left", closed="left").agg(AGGV).dropna(subset=["close"])
    r["t_close"] = r.index + pd.Timedelta(rule)
    return r


# ----------------------------------------------------------------------------- chỉ báo cơ bản
def wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx_di(df, n=14):
    h, l, c = df.high, df.low, df.close
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = wilder(tr, n)
    pdi = 100 * wilder(pd.Series(pdm, index=df.index), n) / atr
    mdi = 100 * wilder(pd.Series(mdm, index=df.index), n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n), pdi - mdi


def supertrend(df, n=10, mult=3.0):
    atr = fx.atr(df, n).values
    h, l, c = df.high.values, df.low.values, df.close.values
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    fub, flb = ub.copy(), lb.copy()
    d = np.ones(len(c), dtype=np.int8)
    for i in range(1, len(c)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < flb[i] else 1
        else:
            d[i] = 1 if c[i] > fub[i] else -1
    return d


def psar(df, step=0.02, mx=0.2):
    h, l = df.high.values, df.low.values
    n = len(h)
    out = np.ones(n, dtype=np.int8)
    up, af, ep, sar = True, step, h[0], l[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, mx)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, mx)
        out[i] = 1 if up else -1
    return out


def linreg_slope(s, n=20):
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    return s.rolling(n).apply(lambda y: ((x - xm) * (y - y.mean())).sum() / denom, raw=True)


def aroon_osc(df, n=25):
    up = df.high.rolling(n + 1).apply(lambda a: a.argmax(), raw=True) / n * 100
    dn = df.low.rolling(n + 1).apply(lambda a: a.argmin(), raw=True) / n * 100
    return up - dn


def swings(df, k=3):
    """Đỉnh/đáy fractal và thời điểm được xác nhận (sau k nến)."""
    h, l = df.high.values, df.low.values
    n = len(h)
    sh, sl = [], []
    tc = df.t_close.values
    for j in range(k, n - k):
        if h[j] == h[j - k:j + k + 1].max():
            sh.append((tc[j + k], h[j]))
        if l[j] == l[j - k:j + k + 1].min():
            sl.append((tc[j + k], l[j]))
    return sh, sl


# ----------------------------------------------------------------------------- dựng bảng chỉ báo theo khung
def build_frames(bv):
    m15, h1, h4, d1 = rs(bv, "15min"), rs(bv, "1h"), rs(bv, "4h"), rs(bv, "1D")
    for df in (m15, h1, h4, d1):
        df["atr"] = fx.atr(df)
    # M15
    m15["rsi"] = rsi(m15.close)
    m15["vol_rel"] = m15.vol / m15.vol.rolling(20).mean()
    day = m15.index.normalize()
    tp = (m15.high + m15.low + m15.close) / 3
    pv = (tp * m15.vol).groupby(day).cumsum()
    vv = m15.vol.groupby(day).cumsum()
    m15["vwap"] = pv / vv
    m15["body_dir"] = np.sign(m15.close - m15.open)
    # H1
    c = h1.close
    for n in (20, 50, 200):
        h1[f"ema{n}"] = fx.ema(c, n)
    h1["ema50_slope"] = (h1.ema50 - h1.ema50.shift(5)) / h1.atr
    h1["adx"], h1["di"] = adx_di(h1)
    h1["st"] = supertrend(h1)
    h1["psar"] = psar(h1)
    macd = fx.ema(c, 12) - fx.ema(c, 26)
    h1["macd_h"] = (macd - fx.ema(macd, 9)) / h1.atr
    tenkan = (h1.high.rolling(9).max() + h1.low.rolling(9).min()) / 2
    kijun = (h1.high.rolling(26).max() + h1.low.rolling(26).min()) / 2
    spa = ((tenkan + kijun) / 2).shift(26)
    spb = ((h1.high.rolling(52).max() + h1.low.rolling(52).min()) / 2).shift(26)
    h1["cloud_top"], h1["cloud_bot"] = np.maximum(spa, spb), np.minimum(spa, spb)
    h1["lr_slope"] = linreg_slope(c, 20) / h1.atr
    h1["aroon"] = aroon_osc(h1)
    h1["rsi"] = rsi(c)
    ll, hh = h1.low.rolling(14).min(), h1.high.rolling(14).max()
    h1["stoch"] = (100 * (c - ll) / (hh - ll)).rolling(3).mean()
    tp1 = (h1.high + h1.low + c) / 3
    mad = tp1.rolling(20).apply(lambda a: np.abs(a - a.mean()).mean(), raw=True)
    h1["cci"] = (tp1 - tp1.rolling(20).mean()) / (0.015 * mad)
    h1["willr"] = 100 * (c - ll) / (hh - ll)  # 0..100 (thấp = sát đáy 14 nến)
    h1["roc6"] = (c - c.shift(6)) / h1.atr
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    h1["bb_w"] = 4 * s20 / m20
    h1["bb_wrel"] = h1.bb_w / h1.bb_w.rolling(100).median()
    h1["m20"], h1["s20"] = m20, s20
    h1["dc_hi"], h1["dc_lo"] = h1.high.rolling(20).max(), h1.low.rolling(20).min()
    lr = np.log(c).diff()
    h1["rv_ratio"] = lr.rolling(24).std() / lr.rolling(480, min_periods=100).std()
    h1["atr_rel"] = h1.atr / h1.atr.rolling(480, min_periods=100).median()
    h1["vol_rel"] = h1.vol / h1.vol.rolling(20).mean()
    obv = (np.sign(c.diff()) * h1.vol).cumsum()
    h1["obv_slope"] = (obv - obv.shift(10)) / h1.vol.rolling(10).sum()
    h1["rsi_lo3"], h1["rsi_lo24"] = h1.rsi.rolling(3).min(), h1.rsi.shift(3).rolling(21).min()
    h1["rsi_hi3"], h1["rsi_hi24"] = h1.rsi.rolling(3).max(), h1.rsi.shift(3).rolling(21).max()
    h1["low3"], h1["low24"] = h1.low.rolling(3).min(), h1.low.shift(3).rolling(21).min()
    h1["high3"], h1["high24"] = h1.high.rolling(3).max(), h1.high.shift(3).rolling(21).max()
    dn = (c.diff() < 0).astype(int)
    upc = (c.diff() > 0).astype(int)
    h1["run_dn"] = dn.groupby((dn != dn.shift()).cumsum()).cumsum() * dn
    h1["run_up"] = upc.groupby((upc != upc.shift()).cumsum()).cumsum() * upc
    # H4
    h4["ema50"] = fx.ema(h4.close, 50)
    h4["ema50_slope"] = (h4.ema50 - h4.ema50.shift(3)) / h4.atr
    h4["adx"], h4["di"] = adx_di(h4)
    h4["st"] = supertrend(h4)
    macd4 = fx.ema(h4.close, 12) - fx.ema(h4.close, 26)
    h4["macd_h"] = (macd4 - fx.ema(macd4, 9)) / h4.atr
    h4["rsi"] = rsi(h4.close)
    # D1
    d1["ema20"] = fx.ema(d1.close, 20)
    d1["ema20_slope"] = (d1.ema20 - d1.ema20.shift(3)) / d1.atr
    rng = d1.high - d1.low
    d1["nr7"] = (rng == rng.rolling(7).min()).astype(int)
    d1["prev_rng"] = rng / d1.atr
    d1["dc_hi"], d1["dc_lo"] = d1.high.rolling(20).max(), d1.low.rolling(20).min()
    d1["rsi"] = rsi(d1.close)
    return dict(m15=m15, h1=h1, h4=h4, d1=d1, swings=swings(h1, 3))


def asof(df, col, T):
    return fx.asof(df, col, T)


def features(bv, fr, tr, when="fill_t"):
    """Tính chỉ báo cho từng lệnh tại thời điểm `when` (lúc khớp / lúc nến xác nhận đóng)."""
    T = pd.DatetimeIndex(tr[when].values)
    m15, h1, h4, d1 = fr["m15"], fr["h1"], fr["h4"], fr["d1"]
    D = tr.dir.values.astype(float)
    L = tr.level.values
    A = tr.atr.values
    X = pd.DataFrame(index=tr.index)

    def g(df, col):
        return asof(df, col, T)

    def osc_dir(v):
        return np.where(D == 1, v, 100 - v)

    # --- xu hướng (x dir: dương = cùng hướng lệnh)
    X["ema_slope_h1"] = g(h1, "ema50_slope") * D
    X["ema_slope_h4"] = g(h4, "ema50_slope") * D
    X["ema_slope_d1"] = g(d1, "ema20_slope") * D
    X["adx_h1"] = g(h1, "adx")
    X["adx_h4"] = g(h4, "adx")
    X["di_h1"] = g(h1, "di") * D
    X["di_h4"] = g(h4, "di") * D
    X["supertrend_h1"] = g(h1, "st") * D
    X["supertrend_h4"] = g(h4, "st") * D
    X["psar_h1"] = g(h1, "psar") * D
    X["macd_h1"] = g(h1, "macd_h") * D
    X["macd_h4"] = g(h4, "macd_h") * D
    ct, cb = g(h1, "cloud_top"), g(h1, "cloud_bot")
    X["ichimoku_h1"] = np.where(L > ct, 1, np.where(L < cb, -1, 0)) * D
    X["linreg_h1"] = g(h1, "lr_slope") * D
    X["aroon_h1"] = g(h1, "aroon") * D
    X["dist_ema200_h1"] = (L - g(h1, "ema200")) / A * D
    X["dist_ema50_h1"] = (L - g(h1, "ema50")) / A * D
    # --- dao động (thấp = quá bán theo hướng lệnh)
    X["rsi_m15"] = osc_dir(g(m15, "rsi"))
    X["rsi_h1"] = osc_dir(g(h1, "rsi"))
    X["rsi_h4"] = osc_dir(g(h4, "rsi"))
    X["rsi_d1"] = osc_dir(g(d1, "rsi"))
    X["stoch_h1"] = osc_dir(g(h1, "stoch"))
    X["willr_h1"] = osc_dir(g(h1, "willr"))
    X["cci_h1"] = g(h1, "cci") * D
    X["roc6_h1"] = g(h1, "roc6") * D
    div_bull = (g(h1, "low3") < g(h1, "low24")) & (g(h1, "rsi_lo3") > g(h1, "rsi_lo24"))
    div_bear = (g(h1, "high3") > g(h1, "high24")) & (g(h1, "rsi_hi3") < g(h1, "rsi_hi24"))
    X["rsi_div_h1"] = np.where(D == 1, div_bull, div_bear).astype(int)
    # --- biến động
    X["atr_rel_h1"] = g(h1, "atr_rel")
    X["atr_h1_d1"] = g(h1, "atr") / g(d1, "atr")
    X["bb_width_rel_h1"] = g(h1, "bb_wrel")
    m20, s20 = g(h1, "m20"), g(h1, "s20")
    X["bb_pctb_h1"] = osc_dir(100 * (L - (m20 - 2 * s20)) / (4 * s20))
    X["keltner_h1"] = (L - g(h1, "ema20")) / (2 * A) * D
    X["rv_ratio_h1"] = g(h1, "rv_ratio")
    X["nr7_d1"] = g(d1, "nr7")
    X["prev_range_d1"] = g(d1, "prev_rng")
    # --- vị trí / kéo giãn
    X["vwap_dist"] = (L - g(m15, "vwap")) / A * D
    X["zscore_h1"] = (L - m20) / s20 * D
    dh, dl = g(h1, "dc_hi"), g(h1, "dc_lo")
    X["donchian_h1"] = osc_dir(100 * (L - dl) / (dh - dl))
    dh, dl = g(d1, "dc_hi"), g(d1, "dc_lo")
    X["donchian_d1"] = osc_dir(100 * (L - dl) / (dh - dl))
    # --- volume
    X["vol_rel_m15"] = g(m15, "vol_rel")
    X["vol_rel_h1"] = g(h1, "vol_rel")
    X["obv_slope_h1"] = g(h1, "obv_slope") * D
    X["last_m15_body"] = g(m15, "body_dir") * D
    # --- hành vi giá trước khi chạm
    X["run_into_h1"] = np.where(D == 1, g(h1, "run_dn"), g(h1, "run_up"))
    # --- mốc trùng đỉnh/đáy swing H1 gần đây, số tròn
    sh, sl = fr["swings"]
    sh_t = np.array([a for a, _ in sh]); sh_p = np.array([b for _, b in sh])
    sl_t = np.array([a for a, _ in sl]); sl_p = np.array([b for _, b in sl])
    near = np.full(len(tr), np.nan)
    Tv = T.values
    for j in range(len(tr)):
        if D[j] == 1:
            k = np.searchsorted(sl_t, Tv[j], "right")
            pts = sl_p[max(0, k - 5):k]
        else:
            k = np.searchsorted(sh_t, Tv[j], "right")
            pts = sh_p[max(0, k - 5):k]
        if len(pts):
            near[j] = np.abs(pts - L[j]).min() / A[j]
    X["swing_near"] = near
    X["round10"] = np.abs(L - np.round(L / 10) * 10) / A
    X["round50"] = np.abs(L - np.round(L / 50) * 50) / A
    return X


FEATURE_INFO = {
    # key: (nhóm, mô tả, kiểu) ; kiểu 'num' chia 5 nhóm, 'cat' theo giá trị
    "ema_slope_h1": ("Xu hướng", "Độ dốc EMA50 H1 theo hướng lệnh", "num"),
    "ema_slope_h4": ("Xu hướng", "Độ dốc EMA50 H4 theo hướng lệnh", "num"),
    "ema_slope_d1": ("Xu hướng", "Độ dốc EMA20 D1 theo hướng lệnh", "num"),
    "adx_h1": ("Xu hướng", "ADX14 H1 (sức mạnh xu hướng)", "num"),
    "adx_h4": ("Xu hướng", "ADX14 H4", "num"),
    "di_h1": ("Xu hướng", "DI+ − DI− H1 theo hướng lệnh", "num"),
    "di_h4": ("Xu hướng", "DI+ − DI− H4 theo hướng lệnh", "num"),
    "supertrend_h1": ("Xu hướng", "Supertrend(10,3) H1 cùng/ngược hướng lệnh", "cat"),
    "supertrend_h4": ("Xu hướng", "Supertrend(10,3) H4 cùng/ngược hướng lệnh", "cat"),
    "psar_h1": ("Xu hướng", "Parabolic SAR H1 cùng/ngược hướng lệnh", "cat"),
    "macd_h1": ("Xu hướng", "MACD histogram H1 theo hướng lệnh", "num"),
    "macd_h4": ("Xu hướng", "MACD histogram H4 theo hướng lệnh", "num"),
    "ichimoku_h1": ("Xu hướng", "Mốc trên/trong/dưới mây Ichimoku H1 (theo hướng lệnh)", "cat"),
    "linreg_h1": ("Xu hướng", "Độ dốc hồi quy 20 nến H1 theo hướng lệnh", "num"),
    "aroon_h1": ("Xu hướng", "Aroon oscillator H1 theo hướng lệnh", "num"),
    "dist_ema200_h1": ("Vị trí", "Mốc cách EMA200 H1 (ATR, theo hướng lệnh)", "num"),
    "dist_ema50_h1": ("Vị trí", "Mốc cách EMA50 H1 (ATR, theo hướng lệnh)", "num"),
    "rsi_m15": ("Dao động", "RSI14 M15 theo hướng lệnh (thấp = quá bán)", "num"),
    "rsi_h1": ("Dao động", "RSI14 H1 theo hướng lệnh", "num"),
    "rsi_h4": ("Dao động", "RSI14 H4 theo hướng lệnh", "num"),
    "rsi_d1": ("Dao động", "RSI14 D1 theo hướng lệnh", "num"),
    "stoch_h1": ("Dao động", "Stochastic(14,3) H1 theo hướng lệnh", "num"),
    "willr_h1": ("Dao động", "Williams %R 14 H1 theo hướng lệnh", "num"),
    "cci_h1": ("Dao động", "CCI20 H1 theo hướng lệnh", "num"),
    "roc6_h1": ("Dao động", "Biến động giá 6 nến H1 theo hướng lệnh (ATR)", "num"),
    "rsi_div_h1": ("Dao động", "Phân kỳ RSI H1 ủng hộ hướng lệnh", "cat"),
    "atr_rel_h1": ("Biến động", "ATR H1 so với trung vị 20 ngày", "num"),
    "atr_h1_d1": ("Biến động", "ATR H1 / ATR D1", "num"),
    "bb_width_rel_h1": ("Biến động", "Độ rộng Bollinger H1 so với trung vị 100 nến", "num"),
    "bb_pctb_h1": ("Biến động", "Vị trí mốc trong Bollinger H1 (thấp = sát dải ngoài)", "num"),
    "keltner_h1": ("Biến động", "Vị trí mốc trong kênh Keltner H1", "num"),
    "rv_ratio_h1": ("Biến động", "Biến động 24h / 20 ngày", "num"),
    "nr7_d1": ("Biến động", "Ngày trước là NR7 (biên độ hẹp nhất 7 ngày)", "cat"),
    "prev_range_d1": ("Biến động", "Biên độ ngày trước / ATR ngày", "num"),
    "vwap_dist": ("Vị trí", "Mốc cách VWAP ngày (ATR, theo hướng lệnh)", "num"),
    "zscore_h1": ("Vị trí", "Z-score mốc so với SMA20 H1", "num"),
    "donchian_h1": ("Vị trí", "Vị trí mốc trong kênh 20 nến H1 (thấp = sát biên)", "num"),
    "donchian_d1": ("Vị trí", "Vị trí mốc trong kênh 20 ngày", "num"),
    "vol_rel_m15": ("Volume", "Tick volume M15 trước khi chạm / TB 20 nến", "num"),
    "vol_rel_h1": ("Volume", "Tick volume H1 / TB 20 nến", "num"),
    "obv_slope_h1": ("Volume", "Độ dốc OBV 10 nến H1 theo hướng lệnh", "num"),
    "last_m15_body": ("Hành vi giá", "Nến M15 cuối trước khi chạm cùng/ngược hướng lệnh", "cat"),
    "run_into_h1": ("Hành vi giá", "Số nến H1 liên tiếp chạy vào mốc", "num"),
    "swing_near": ("Cấu trúc", "Khoảng cách mốc tới đỉnh/đáy swing H1 gần nhất (ATR)", "num"),
    "round10": ("Cấu trúc", "Khoảng cách mốc tới số tròn 10 USD (ATR)", "num"),
    "round50": ("Cấu trúc", "Khoảng cách mốc tới số tròn 50 USD (ATR)", "num"),
}
