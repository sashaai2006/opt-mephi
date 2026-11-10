import io
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

OUT_DIR = Path(__file__).parent / "output"


def build_matrices(sheets, items, patterns):
    kcols = [f"k_{iid}" for iid in items["item_id"]]
    A = patterns[kcols].to_numpy(dtype=float).T
    cost = patterns.merge(sheets[["sheet_id", "cost"]], on="sheet_id")["cost"].to_numpy(float)
    demand = items["demand"].to_numpy(float)
    supply = sheets["supply"].to_numpy(float)
    sheet_ids = list(sheets["sheet_id"])
    S = np.zeros((len(sheet_ids), len(patterns)))
    for t, sid in enumerate(sheet_ids):
        S[t, (patterns["sheet_id"] == sid).to_numpy()] = 1.0
    return A, cost, demand, supply, S, sheet_ids


def solve(sheets, items, patterns, demand=None, supply=None):
    A, cost, d, s, S, sheet_ids = build_matrices(sheets, items, patterns)
    if demand is not None:
        d = np.asarray(demand, dtype=float)
    if supply is not None:
        s = np.asarray(supply, dtype=float)
    A_ub = np.vstack([-A, S])
    b_ub = np.concatenate([-d, s])
    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = linprog(c=cost, A_ub=A_ub, b_ub=b_ub, bounds=(0, None),
                      method="highs")
    elapsed = time.perf_counter() - t0
    return {
        "result": res, "A": A, "cost": cost, "demand": d, "supply": s,
        "S": S, "sheet_ids": sheet_ids, "elapsed": elapsed,
        "log": buf.getvalue(),
    }


def solver_log(name, ctx):
    res, elapsed = ctx["result"], ctx["elapsed"]
    return "\n".join([
        f"=== solver log: {name} ===",
        "solver: HiGHS via scipy.optimize.linprog (method='highs')",
        f"status: {res.status} ({res.message})",
        f"iterations (simplex): {getattr(res, 'nit', 'n/a')}",
        f"elapsed: {elapsed:.4f} s",
        f"objective (solver): {res.fun if res.success else 'n/a'}",
        "",
        "--- HiGHS console output ---",
        res and (ctx["log"] or "(HiGHS не печатает протокол итераций через scipy-интерфейс)"),
        "",
    ])
