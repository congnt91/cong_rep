"""Lưới SL (0,5-1,5 ATR) x loại lệnh (Limit/Stop), không lọc RR. Chạy: python run_grid.py old|new"""
import sys, time, fxlevels as fx
which = sys.argv[1]
base = fx.load_new() if which == "new" else fx.load_old()
ctx = fx.build_context(base)
t0=time.time()
for otype in ["limit", "stop"]:
    for sl in [0.5, 0.75, 1.0, 1.5]:
        for fam in (["static", "ma"] if otype == "limit" else ["static"]):
            o = fx.make_orders(base, ctx, fam, sl_atr=sl, rr_min=0.0, otype=otype)
            tr = fx.simulate(base, o)
            if fam == "ma": tr = fx.one_fill_per_block(tr)
            tr = tr.dropna(subset=["fill_t"])
            ex = fx.excursions(base, tr)
            ex.to_pickle(f"out/grid_{which}_{otype}_{fam}_{sl}.pkl")
            print(which, otype, fam, sl, len(o), len(ex), round(time.time()-t0,1), flush=True)
