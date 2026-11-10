import math
import random

import numpy as np
import pandas as pd

from .model import solve
from .verify import chosen, verify


def make_instance(size, seed=10217):
    rng = random.Random(seed)
    cats = []
    for i in range(6):
        cats.append({"category_id": f"C{i + 1}", "name": f"Категория {i + 1}",
                     "min_skus": 2})
    base = size // 6
    counts = [base + (1 if i < size % 6 else 0) for i in range(6)]
    skus = []
    k = 1
    for ci, cnt in enumerate(counts):
        for _ in range(cnt):
            d = rng.randint(6, 28)
            skus.append({"sku_id": f"SKU{k:02d}", "name": f"Товар {k:02d}",
                         "category_id": f"C{ci + 1}", "demand": d,
                         "min_stock": rng.randint(3, 8),
                         "unit_space": round(rng.uniform(0.05, 0.15), 3),
                         "max_stock": math.ceil(1.1 * d)})
            k += 1
    cat_rows = []
    need_space = 0.0
    for ci, c in enumerate(cats):
        members = [s for s in skus if s["category_id"] == c["category_id"]]
        total_u = sum(s["max_stock"] for s in members)
        total_d = sum(s["demand"] for s in members)
        D = min(int(0.75 * total_d), total_u)
        sp = 0.0
        need = D
        for s in sorted(members, key=lambda v: v["unit_space"]):
            take = min(s["max_stock"], need)
            sp += take * s["unit_space"]
            need -= take
        D -= max(0, need)
        need_space += sp
        cat_rows.append({"category_id": c["category_id"], "name": c["name"],
                         "demand": D,
                         "min_skus": min(2, len(members))})
    space = round(1.3 * need_space + 0.5, 1)
    return pd.DataFrame(skus), pd.DataFrame(cat_rows), space


def gap(z_milp, z_lp):
    if not z_milp or not z_lp:
        return None
    return (z_milp - z_lp) / abs(z_milp) * 100


def run_all(skus, cats, space):
    out = {}

    base = solve(skus, cats, space)
    base_lp = solve(skus, cats, space, mip=False)
    out["base"] = {"ctx": base, "verify": verify(base),
                   "lp": base_lp, "chosen": chosen(base),
                   "chosen_lp": chosen(base_lp)}

    deficit = solve(skus, cats, space, space_scale=0.8)
    out["deficit"] = {"ctx": deficit, "verify": verify(deficit),
                      "chosen": chosen(deficit)}

    stress = solve(skus, cats, space, demand_scale=1.2)
    out["stress"] = {"ctx": stress, "verify": verify(stress),
                     "chosen": chosen(stress)}

    bigm = solve(skus, cats, space, bigm=1e6)
    out["bigm"] = {"ctx": bigm, "verify": verify(bigm)}

    curve = []
    for size in [10, 20, 40]:
        s2, c2, sp2 = make_instance(size)
        g_mip = solve(s2, c2, sp2)
        g_lp = solve(s2, c2, sp2, mip=False)
        row = {"size": size,
               "z_milp": g_mip["result"].fun if g_mip["result"].success else None,
               "z_lp": g_lp["result"].fun if g_lp["result"].success else None,
               "time_milp": g_mip["elapsed"], "time_lp": g_lp["elapsed"]}
        row["gap_pct"] = (gap(row["z_milp"], row["z_lp"])
                          if row["z_milp"] is not None else None)
        curve.append(row)
    out["curve"] = curve
    return out
