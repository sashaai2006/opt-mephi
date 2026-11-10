import numpy as np

TOL = 1e-7


def verify(ctx):
    if not ctx["result"].success:
        return {"feasible": False, "checks": [], "ok": False}
    x = np.asarray(ctx["result"].x, dtype=float)
    A, cost, d = ctx["A"], ctx["cost"], ctx["demand"]
    S, s = ctx["S"], ctx["supply"]
    produced = A @ x
    used_sheets = S @ x
    checks = [
        ("x >= 0", bool(np.all(x >= -TOL))),
        ("спрос: A x >= d (допуск 1e-7)", bool(np.all(produced >= d - TOL))),
        ("запас по типам: S x <= supply (допуск 1e-7)",
         bool(np.all(used_sheets <= s + TOL))),
    ]
    obj = float(cost @ x)
    obj_ok = abs(obj - ctx["result"].fun) <= 1e-6 * max(1.0, abs(obj))
    return {
        "feasible": True,
        "checks": checks,
        "produced": produced,
        "overproduction": produced - d,
        "used_sheets": used_sheets,
        "slack_supply": s - used_sheets,
        "objective_recomputed": obj,
        "objective_matches_solver": bool(obj_ok),
        "ok": all(ok for _, ok in checks) and obj_ok,
    }
