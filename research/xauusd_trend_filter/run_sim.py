"""Mô phỏng hệ thống "Auto mốc" (SL 0,75 ATR H1, TP mốc kế tiếp, RR >= 1).
Chạy: python run_sim.py old   |   python run_sim.py new"""
import os
import sys
import time

import fxlevels as fx

which = sys.argv[1]
os.makedirs("out", exist_ok=True)
base = fx.load_new() if which == "new" else fx.load_old()
t0 = time.time()
ctx = fx.build_context(base)
print(which, base.index.min(), base.index.max(), len(base))
for fam in ["static", "ma"]:
    o = fx.make_orders(base, ctx, fam)
    tr = fx.simulate(base, o, spread=0.30, autoclose=(60, 0.5) if which == "new" else None)
    if fam == "ma":
        tr = fx.one_fill_per_block(tr)
    tr = tr.dropna(subset=["fill_t"])
    ex = fx.excursions(base, tr)
    ex.to_pickle(f"out/ex_{which}_{fam}.pkl")
    print(fam, "orders", len(o), "fills", len(ex), round(time.time() - t0, 1), "s")
