import io
import time
from contextlib import redirect_stdout

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def build(skus, cats, space):
    n = len(skus)
    m = len(cats)
    d = skus["demand"].to_numpy(float)
    ms = skus["min_stock"].to_numpy(float)
    a = skus["unit_space"].to_numpy(float)
    U = skus["max_stock"].to_numpy(float)
    D = cats["demand"].to_numpy(float)
    nmin = cats["min_skus"].to_numpy(float)
    cat_of = {cid: i for i, cid in enumerate(cats["category_id"])}
    mem = np.array([cat_of[c] for c in skus["category_id"]])
    rows, lb, ub = [], [], []

    B = np.zeros((n, 2 * n))
    B[np.arange(n), np.arange(n)] = 1.0
    B[np.arange(n), n + np.arange(n)] = -U
    rows.append(B)
    lb.append(np.full(n, -np.inf))
    ub.append(np.zeros(n))

    B = np.zeros((n, 2 * n))
    B[np.arange(n), np.arange(n)] = -1.0
    B[np.arange(n), n + np.arange(n)] = ms
    rows.append(B)
    lb.append(np.full(n, -np.inf))
    ub.append(np.zeros(n))

    B = np.zeros((m, 2 * n))
    B[mem, np.arange(n)] = -1.0
    rows.append(B)
    lb.append(np.full(m, -np.inf))
    ub.append(-D)

    B = np.zeros((m, 2 * n))
    B[mem, n + np.arange(n)] = -1.0
    rows.append(B)
    lb.append(np.full(m, -np.inf))
    ub.append(-nmin)

    rows.append(np.concatenate([a, np.zeros(n)])[None, :])
    lb.append([-np.inf])
    ub.append([float(space)])

    A = np.vstack(rows)
    c = np.concatenate([np.zeros(n), np.ones(n)])
    return {
        "n": n, "m": m, "d": d, "min_stock": ms, "unit_space": a,
        "max_stock": U, "D": D, "min_skus": nmin, "mem": mem,
        "space": float(space), "A": A, "lb": np.concatenate(lb),
        "ub": np.concatenate(ub), "c": c,
        "sku_ids": list(skus["sku_id"]), "cat_ids": list(cats["category_id"]),
    }


def solve(skus, cats, space, demand_scale=None, space_scale=None,
          mip=True, bigm=None):
    g = build(skus, cats, space)
    ub = g["ub"].copy()
    if demand_scale is not None:
        ub[2 * g["n"]:2 * g["n"] + g["m"]] = -np.ceil(g["D"] * demand_scale)
    if space_scale is not None:
        ub[-1] = g["space"] * space_scale
    if bigm is not None:
        rows = g["A"][:g["n"]].copy()
        rows[np.arange(g["n"]), g["n"] + np.arange(g["n"])] = -bigm
        A = np.vstack([rows, g["A"][g["n"]:]])
    else:
        A = g["A"]
    integrality = np.concatenate([np.zeros(g["n"]),
                                  np.ones(g["n"]) if mip else np.zeros(g["n"])])
    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = milp(c=g["c"],
                   constraints=LinearConstraint(A, g["lb"], ub),
                   bounds=Bounds(np.zeros(2 * g["n"]),
                                 np.concatenate([np.full(g["n"], np.inf),
                                                 np.ones(g["n"])])),
                   integrality=integrality,
                   options={"time_limit": 120.0, "mip_rel_gap": 1e-9})
    elapsed = time.perf_counter() - t0
    g.update({"result": res, "elapsed": elapsed, "log": buf.getvalue(),
              "mip": mip})
    return g


def solver_log(name, g):
    res, elapsed = g["result"], g["elapsed"]
    kind = "MILP (HiGHS branch-and-bound)" if g["mip"] else "LP-relaxation"
    return "\n".join([
        f"=== solver log: {name} ===",
        f"solver: HiGHS via scipy.optimize.milp [{kind}]",
        f"status: {res.status} ({res.message})",
        f"elapsed: {elapsed:.4f} s",
        f"objective (solver): {res.fun if res.success else 'n/a'}",
        "",
        "--- HiGHS console output ---",
        g["log"] or "(HiGHS не печатает протокол итераций через scipy-интерфейс)",
        "",
    ])
