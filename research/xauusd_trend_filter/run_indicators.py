"""Mốc phản ứng + chỉ báo: chỉ báo nào (hoặc tổ hợp nào) làm tăng hiệu quả?
Quy trình: chọn trên A1 (2012-2017), kiểm tra trên A2 (2017-2022) và B (2025-26).
Kết quả: out/indicators.json"""
import json
import time
import warnings

import numpy as np
import pandas as pd

import fxlevels as fx
import improve as im
import indicators as ind
from run_combo import build
from run_improve import period_of, PERIODS
from run_hours import window_of

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(11)
KEEP_FRAC = 0.4          # bộ lọc 1 chỉ báo: giữ 40% lệnh (một đuôi) — chọn đuôi trên A1
MODEL_KEEP = 0.3         # mô hình: giữ 30% lệnh có điểm cao nhất (ngưỡng tính trên dữ liệu huấn luyện)


def week_index(times):
    wk = pd.DatetimeIndex(times).to_period("W").astype(str).values
    weeks = np.unique(wk)
    return weeks, {w: np.nonzero(wk == w)[0] for w in weeks}


def boot_gain(R, keep, times, reps=1000):
    """KTC 95% cho (E lệnh giữ − E tất cả), bootstrap theo tuần."""
    weeks, idx = week_index(times)
    g = []
    for _ in range(reps):
        s = np.concatenate([idx[w] for w in RNG.choice(weeks, len(weeks))])
        k = keep[s]
        if k.sum() == 0:
            continue
        g.append(R[s][k].mean() - R[s].mean())
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def boot_mean(R, times, reps=1000):
    weeks, idx = week_index(times)
    m = [R[np.concatenate([idx[w] for w in RNG.choice(weeks, len(weeks))])].mean() for _ in range(reps)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def scan_feature(df, col, kind):
    """Chia nhóm theo A1, chọn đuôi/nhóm tốt nhất trên A1, đo mức cải thiện ở từng giai đoạn."""
    x = df[col].values.astype(float)
    R = df.R.values
    per = df.period.values
    a1 = (per == "A1") & ~np.isnan(x)
    out = {"col": col, "kind": kind}
    if kind == "num":
        edges = np.unique(np.nanquantile(x[a1], [0.2, 0.4, 0.6, 0.8]))
        b = np.digitize(x, edges)  # 0..4
        b = np.where(np.isnan(x), -1, b)
        nb = len(edges) + 1
        labels = []
        lo = [None] + list(edges)
        hi = list(edges) + [None]
        for i in range(nb):
            labels.append(("≤ " + f"{hi[i]:.3g}") if lo[i] is None else ("> " + f"{lo[i]:.3g}") if hi[i] is None else f"{lo[i]:.3g} – {hi[i]:.3g}")
        k = int(round(KEEP_FRAC * nb)) or 1
        low = np.isin(b, range(0, k))
        high = np.isin(b, range(nb - k, nb))
        eL, eH = R[a1 & low].mean(), R[a1 & high].mean()
        keep = low if eL >= eH else high
        out["rule"] = ("thấp" if eL >= eH else "cao") + f" ({'≤ ' + format(edges[k-1], '.3g') if eL >= eH else '> ' + format(edges[nb-k-1], '.3g')})"
        out["bins"] = [{"label": labels[i], **{p: {"n": int(((b == i) & (per == p)).sum()), "E": float(R[(b == i) & (per == p)].mean()) if ((b == i) & (per == p)).any() else None} for p in PERIODS}} for i in range(nb)]
    else:
        cats = sorted(pd.unique(x[~np.isnan(x)]))
        best, bestE = None, -9
        for c in cats:
            m = a1 & (x == c)
            if m.sum() >= 0.15 * a1.sum() and R[m].mean() > bestE:
                best, bestE = c, R[m].mean()
        keep = x == best
        out["rule"] = f"= {best:g}"
        out["bins"] = [{"label": f"{c:g}", **{p: {"n": int(((x == c) & (per == p)).sum()), "E": float(R[(x == c) & (per == p)].mean()) if ((x == c) & (per == p)).any() else None} for p in PERIODS}} for c in cats]
    for p in PERIODS:
        m = per == p
        Rp, kp = R[m], keep[m]
        g = float(Rp[kp].mean() - Rp.mean()) if kp.any() else None
        rec = {"n_keep": int(kp.sum()), "E_keep": float(Rp[kp].mean()) if kp.any() else None, "gain": g}
        if p != "A1" and kp.sum() >= 20:
            rec["lo"], rec["hi"] = boot_gain(Rp, kp, df.exit_t.values[m])
        # tương quan hạng
        xm = x[m]
        ok = ~np.isnan(xm)
        rec["rho"] = float(pd.Series(xm[ok]).rank().corr(pd.Series(Rp[ok]).rank())) if ok.sum() > 30 else None
        out[p] = rec
    out["robust"] = bool(out["A2"].get("lo") is not None and out["A2"]["lo"] > 0 and (out["B"]["gain"] or -1) > 0)
    out["same_sign"] = bool((out["A2"]["gain"] or -1) > 0 and (out["B"]["gain"] or -1) > 0)
    return out, keep


def model_eval(df, cols, train_p, test_ps, kind):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    X = df[cols].astype(float)
    tr = df.period.isin(train_p).values
    med = X[tr].median()
    X = X.fillna(med)
    mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
    Z = ((X - mu) / sd).values
    R = df.R.values
    if kind == "logit":
        m = LogisticRegression(C=0.05, max_iter=2000).fit(Z[tr], (R[tr] > 0).astype(int))
        score = m.predict_proba(Z)[:, 1]
        coefs = dict(zip(cols, m.coef_[0]))
    elif kind == "ridge":
        m = Ridge(alpha=50.0).fit(Z[tr], np.clip(R[tr], -1.2, 3))
        score = m.predict(Z)
        coefs = dict(zip(cols, m.coef_))
    else:
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=150, l2_regularization=1.0, random_state=0)
        m.fit(Z[tr], np.clip(R[tr], -1.2, 3))
        score = m.predict(Z)
        coefs = None
    thr = np.quantile(score[tr], 1 - MODEL_KEEP)
    keep = score >= thr
    res = {}
    for p in PERIODS:
        mp = df.period.values == p
        Rp, kp = R[mp], keep[mp]
        r = {"n_all": int(mp.sum()), "E_all": float(Rp.mean()), "n_keep": int(kp.sum()), "E_keep": float(Rp[kp].mean()) if kp.any() else None, "in_sample": p in train_p}
        if kp.sum() >= 20:
            r["lo"], r["hi"] = boot_mean(Rp[kp], df.exit_t.values[mp][kp])
            r["glo"], r["ghi"] = boot_gain(Rp, kp, df.exit_t.values[mp])
        res[p] = r
    # 5 nhóm điểm (theo ngưỡng huấn luyện) để xem điểm có xếp hạng được lệnh không
    qs = np.quantile(score[tr], [0.2, 0.4, 0.6, 0.8])
    qb = np.digitize(score, qs)
    res["quintiles"] = {p: [float(R[(df.period.values == p) & (qb == i)].mean()) if ((df.period.values == p) & (qb == i)).any() else None for i in range(5)] for p in PERIODS}
    top = None
    if coefs:
        top = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:12]
        top = [(k, float(v)) for k, v in top]
    return res, top


def main():
    t0 = time.time()
    sets = {"limit": [], "confirm": []}
    for which in ["old", "new"]:
        bv = ind.load_with_volume(which)
        base = bv[["open", "high", "low", "close"]]
        ctx = im.build_context(base)
        fr = ind.build_frames(bv)
        for key, kw in [("limit", dict(entry="limit")), ("confirm", dict(entry="confirm"))]:
            tr = build(base, ctx, **kw)
            X = ind.features(bv, fr, tr)
            d = pd.concat([tr.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
            sets[key].append(d)
        print(which, "features", round(time.time() - t0), "s", flush=True)
    OUT = {"features": {k: {"family": v[0], "desc": v[1], "kind": v[2]} for k, v in ind.FEATURE_INFO.items()}, "sets": {}}
    feat_cols = list(ind.FEATURE_INFO.keys())
    for key in sets:
        df = pd.concat(sets[key], ignore_index=True)
        df["period"] = period_of(df.fill_t.values)
        T = pd.DatetimeIndex(df.fill_t.values) - (pd.Timedelta("15min") if key == "confirm" else pd.Timedelta(0))
        df["win"] = window_of(T.hour.values)
        S = OUT["sets"][key] = {"overall": {p: {"n": int((df.period == p).sum()), "E": float(df.R[df.period == p].mean())} for p in PERIODS}}
        scans, keeps = [], {}
        for col in feat_cols:
            r, k = scan_feature(df, col, ind.FEATURE_INFO[col][2])
            scans.append(r)
            keeps[col] = k
        scans.sort(key=lambda r: -(r["A2"]["gain"] or -9))
        S["scan"] = scans
        n_same = sum(r["same_sign"] for r in scans)
        n_rob = sum(r["robust"] for r in scans)
        S["n_features"], S["n_same_sign"], S["n_robust"] = len(scans), n_same, n_rob
        print(f"== {key}: {len(scans)} chỉ báo, cùng dấu A2&B: {n_same}, vững (KTC A2 > 0 và B > 0): {n_rob}", flush=True)
        for r in scans[:15]:
            print(f"   {r['col']:16s} {r['rule']:18s} A1 {r['A1']['gain']:+.3f} | A2 {r['A2']['gain']:+.3f} [{r['A2'].get('lo', float('nan')):+.3f},{r['A2'].get('hi', float('nan')):+.3f}] | B {r['B']['gain']:+.3f} [{r['B'].get('lo', float('nan')):+.3f},{r['B'].get('hi', float('nan')):+.3f}]")
        # mô hình kết hợp tất cả chỉ báo (+ bối cảnh mốc)
        ctx_cols = ["rr_struct", "dist", "confluence", "touched_before", "hrs_to_fill", "day_used", "approach"] if "dist" in df else ["rr_struct", "confluence", "touched_before", "hrs_to_fill", "day_used", "approach"]
        df["touched_before"] = df.touched_before.astype(float)
        for w in ["W1", "W2", "W5", "W6", "W7", "W8"]:
            df[f"win_{w}"] = (df.win == w).astype(float)
        for fam in ["PDH", "PDL", "S8_P", "S8_R1", "S8_S1", "D_P"]:
            df[f"type_{fam}"] = (df.type == fam).astype(float)
        all_cols = feat_cols + ctx_cols + [c for c in df.columns if c.startswith("win_") or c.startswith("type_")]
        S["models"] = {}
        for mk in ["logit", "ridge", "gbm"]:
            for train in (["A1"], ["A1", "A2"]):
                res, top = model_eval(df, all_cols, train, None, mk)
                tag = f"{mk}_{'A1' if train == ['A1'] else 'A'}"
                S["models"][tag] = {"res": res, "top": top}
                print(f"   model {tag:10s}", " ".join(f"{p}{'*' if res[p]['in_sample'] else ''}: keep {res[p]['n_keep']:4d} E {res[p]['E_keep']:+.3f} vs {res[p]['E_all']:+.3f}" for p in PERIODS), "| quintiles B", [None if v is None else round(v, 3) for v in res["quintiles"]["B"]], flush=True)
        # tổ hợp các chỉ báo "cùng dấu" tốt nhất (chọn bằng A1 + A2) — chỉ B là ngoài mẫu
        good = [r["col"] for r in scans if r["A2"].get("lo") is not None and r["A2"]["lo"] > 0][:4]
        combo = {}
        for kreq in (1, 2):
            if len(good) >= kreq:
                votes = np.sum([keeps[c] for c in good], axis=0)
                kp = votes >= kreq
                combo[f"{kreq}_of_{len(good)}"] = {"cols": good, **{p: {"n": int((kp & (df.period.values == p)).sum()), "E": float(df.R.values[kp & (df.period.values == p)].mean()) if (kp & (df.period.values == p)).any() else None} for p in PERIODS}}
        S["combo_good"] = combo
        print("   combo", combo, flush=True)
    OUT["runtime_s"] = round(time.time() - t0)
    json.dump(OUT, open("out/indicators.json", "w"), ensure_ascii=False, default=float)
    print("saved", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
