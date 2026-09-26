"""Mốc trùng với đường chỉ báo (EMA, VWAP, Bollinger, số tròn, đỉnh/đáy swing) có phản ứng tốt hơn không?
Kết quả: out/confluence.json"""
import json

import numpy as np
import pandas as pd

import improve as im
import indicators as ind
from run_combo import build
from run_improve import period_of, PERIODS

FLAGS = {
    "ema50": "Mốc trùng EMA50 H1 (< 0,25 ATR)",
    "ema200": "Mốc trùng EMA200 H1 (< 0,25 ATR)",
    "vwap": "Mốc trùng VWAP ngày (< 0,25 ATR)",
    "bb": "Mốc trùng dải Bollinger ngoài H1 (< 0,25 ATR)",
    "round50": "Mốc trùng số tròn 50 USD (< 0,25 ATR)",
    "swing": "Mốc trùng đỉnh/đáy swing H1 gần đây (< 0,1 ATR)",
    "any_ind": "Trùng ít nhất một đường EMA/VWAP/Bollinger",
}


def main():
    OUT = {"flags": FLAGS, "sets": {}}
    sets = {"limit": [], "confirm": []}
    for which in ["old", "new"]:
        bv = ind.load_with_volume(which)
        base = bv[["open", "high", "low", "close"]]
        ctx = im.build_context(base)
        fr = ind.build_frames(bv)
        for key, kw in [("limit", dict(entry="limit")), ("confirm", dict(entry="confirm"))]:
            tr = build(base, ctx, **kw).reset_index(drop=True)
            X = ind.features(bv, fr, tr)
            T = pd.DatetimeIndex(tr.fill_t.values)
            L, A, D = tr.level.values, tr.atr.values, tr.dir.values
            h1, m15 = fr["h1"], fr["m15"]
            f = pd.DataFrame(index=tr.index)
            f["ema50"] = np.abs(L - ind.asof(h1, "ema50", T)) / A < 0.25
            f["ema200"] = np.abs(L - ind.asof(h1, "ema200", T)) / A < 0.25
            f["vwap"] = np.abs(L - ind.asof(m15, "vwap", T)) / A < 0.25
            m20, s20 = ind.asof(h1, "m20", T), ind.asof(h1, "s20", T)
            band = np.where(D == 1, m20 - 2 * s20, m20 + 2 * s20)
            f["bb"] = np.abs(L - band) / A < 0.25
            f["round50"] = X.round50.values < 0.25
            f["swing"] = X.swing_near.values < 0.1
            f["any_ind"] = f.ema50 | f.ema200 | f.vwap | f.bb
            keep_cols = ["fill_t", "R", "exit_t", "atr_rel"]
            sets[key].append(pd.concat([tr[keep_cols], f, X[["vwap_dist"]]], axis=1))
        # cấu hình đề xuất (xác nhận + chốt 50%) để ghép bộ lọc
        trp = build(base, ctx, entry="confirm", rule="partial").reset_index(drop=True)
        Tp = pd.DatetimeIndex(trp.fill_t.values)
        Xp = ind.features(bv, fr, trp)
        m20, s20 = ind.asof(fr["h1"], "m20", Tp), ind.asof(fr["h1"], "s20", Tp)
        band = np.where(trp.dir.values == 1, m20 - 2 * s20, m20 + 2 * s20)
        fp = pd.DataFrame({"bb": np.abs(trp.level.values - band) / trp.atr.values < 0.25, "swing": Xp.swing_near.values < 0.1})
        sets.setdefault("rec", []).append(pd.concat([trp[["fill_t", "R", "exit_t", "atr_rel"]], fp], axis=1))
    from run_hours import window_of
    for key, parts in list(sets.items()):
        df = pd.concat(parts, ignore_index=True)
        T = pd.DatetimeIndex(df.fill_t.values) - (pd.Timedelta("15min") if key != "limit" else pd.Timedelta(0))
        df["win"] = window_of(T.hour.values)
        if key == "rec":
            df["period"] = period_of(df.fill_t.values)
            rec_df = df
            continue
        df["period"] = period_of(df.fill_t.values)
        S = OUT["sets"][key] = {"overall": {p: float(df.R[df.period == p].mean()) for p in PERIODS}, "flags": {}}
        print("==", key, {p: round(v, 3) for p, v in S["overall"].items()})
        for fl in FLAGS:
            r = {}
            for p in PERIODS:
                m = df.period == p
                on, off = m & df[fl], m & ~df[fl]
                r[p] = {"n_on": int(on.sum()), "E_on": float(df.R[on].mean()) if on.any() else None, "E_off": float(df.R[off].mean()) if off.any() else None,
                        "share": float(on.sum() / m.sum())}
            S["flags"][fl] = r
            print(f"   {FLAGS[fl]:48s}", " | ".join(f"{p}: {r[p]['share']:.0%} n={r[p]['n_on']:4d} on {r[p]['E_on'] if r[p]['E_on'] is None else round(r[p]['E_on'],3):>6} off {round(r[p]['E_off'],3):>6}" for p in PERIODS))
    # cấu hình cuối: ghép bộ lọc "tránh" với các cấu hình trước
    L = pd.concat(sets["limit"], ignore_index=True)
    L["period"] = period_of(L.fill_t.values)
    L["win"] = window_of(pd.DatetimeIndex(L.fill_t.values).hour.values)
    R = rec_df
    tban = lambda d: ~d.win.isin(["W1", "W5"])
    CFG = [
        ("limit", "Hiện tại: Limit tại mốc", L, np.ones(len(L), bool)),
        ("limit_avoid", "Limit, tránh mốc trùng swing H1 và dải Bollinger ngoài", L, ~L.swing & ~L.bb),
        ("limit_avoid_vwap", "Limit, tránh swing + Bollinger, mốc không sâu dưới VWAP", L, ~L.swing & ~L.bb & (L.vwap_dist > -0.61)),
        ("limit_avoid_time", "Limit, tránh swing + Bollinger, cấm 01–02h và 10–12h", L, ~L.swing & ~L.bb & tban(L)),
        ("rec", "Xác nhận M15 + ATR ≥ trung vị + chốt 50%", R, (R.atr_rel >= 1.0).values),
        ("rec_bb", "Như trên + tránh mốc trùng dải Bollinger ngoài", R, ((R.atr_rel >= 1.0) & ~R.bb).values),
        ("rec_bb_time", "Như trên + cấm 01–02h và 10–12h", R, ((R.atr_rel >= 1.0) & ~R.bb & tban(R)).values),
    ]
    OUT["final"] = []
    for key, label, d, m in CFG:
        x = d[m]
        row = {"key": key, "label": label, "stats": {}}
        for p in PERIODS:
            y = x[x.period == p]
            lo, hi = im.boot_mean_ci(y.R.values, y.exit_t.values)
            months = max(1, len(pd.DatetimeIndex(y.fill_t.values).to_period("M").unique()))
            row["stats"][p] = {"n": int(len(y)), "E": float(y.R.mean()), "lo": lo, "hi": hi, "win": float((y.R > 0).mean()), "per_month": round(len(y) / months, 1)}
        OUT["final"].append(row)
        print(f"   FINAL {label:62s}", " ".join(f"{p}: n={row['stats'][p]['n']:5d} E={row['stats'][p]['E']:+.3f} [{row['stats'][p]['lo']:+.3f},{row['stats'][p]['hi']:+.3f}]" for p in PERIODS))
    json.dump(OUT, open("out/confluence.json", "w"), ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
