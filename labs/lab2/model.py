import io
import time
from contextlib import redirect_stdout

import numpy as np
from scipy.optimize import linprog


def build(warehouses, shops, routes):
    wids = list(warehouses["warehouse_id"])
    sids = list(shops["shop_id"])
    m, n = len(wids), len(sids)
    arc_ij = [(a, b) for a in wids for b in sids]
    k = len(arc_ij)
    cost = routes["cost"].to_numpy(float)
    cap = routes["capacity"].to_numpy(float)
    supply = warehouses["supply"].to_numpy(float)
    demand = shops["demand"].to_numpy(float)
    penalty = shops["penalty"].to_numpy(float)
    c = np.concatenate([cost, penalty])

    A_ub, b_ub = [], []

    B = np.zeros((m, k + n))
    for i in range(m):
        for t, (a, _) in enumerate(arc_ij):
            if a == wids[i]:
                B[i, t] = 1.0
    A_ub.append(B)
    b_ub.append(supply)

    B = np.zeros((n, k + n))
    for j in range(n):
        for t, (_, b) in enumerate(arc_ij):
            if b == sids[j]:
                B[j, t] = -1.0
        B[j, k + j] = -1.0
    A_ub.append(B)
    b_ub.append(-demand)

    B = np.zeros((k, k + n))
    for t in range(k):
        B[t, t] = 1.0
    A_ub.append(B)
    b_ub.append(cap)

    return {
        "wids": wids, "sids": sids, "arc_ij": arc_ij, "k": k, "m": m, "n": n,
        "c": c, "cost": cost, "cap": cap, "supply": supply, "demand": demand,
        "penalty": penalty,
        "A_ub": np.vstack(A_ub),
        "b_ub": np.concatenate(b_ub),
    }


def solve(warehouses, shops, routes, supply=None, demand=None, closed=None):
    g = build(warehouses, shops, routes)
    s = g["supply"].copy() if supply is None else np.asarray(supply, float)
    d = g["demand"].copy() if demand is None else np.asarray(demand, float)
    b = g["b_ub"].copy()
    b[:g["m"]] = s
    b[g["m"]:g["m"] + g["n"]] = -d
    cap = g["cap"].copy()
    if closed:
        for t, (a, bb) in enumerate(g["arc_ij"]):
            if (a, bb) in closed:
                cap[t] = 0.0
    b[g["m"] + g["n"]:] = cap
    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = linprog(c=g["c"], A_ub=g["A_ub"], b_ub=b, bounds=(0, None),
                      method="highs")
    elapsed = time.perf_counter() - t0
    g.update({"result": res, "supply_used": s, "demand_used": d,
              "cap_used": cap, "elapsed": elapsed, "log": buf.getvalue()})
    return g


def split(g):
    res = g["result"]
    k = g["k"]
    x = res.x[:k]
    u = res.x[k:]
    return x, u


def solver_log(name, g):
    res, elapsed = g["result"], g["elapsed"]
    return "\n".join([
        f"=== solver log: {name} ===",
        "solver: HiGHS via scipy.optimize.linprog (method='highs')",
        f"status: {res.status} ({res.message})",
        f"iterations (simplex): {getattr(res, 'nit', 'n/a')}",
        f"elapsed: {elapsed:.4f} s",
        f"objective (solver): {res.fun if res.success else 'n/a'}",
        "",
        "--- HiGHS console output ---",
        g["log"] or "(HiGHS не печатает протокол итераций через scipy-интерфейс)",
        "",
    ])
