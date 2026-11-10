import numpy as np

TOL = 1e-7


def flow_on(g, x, wid):
    return sum(x[t] for t, (a, _) in enumerate(g["arc_ij"]) if a == wid)


def flow_to(g, x, sid):
    return sum(x[t] for t, (_, b) in enumerate(g["arc_ij"]) if b == sid)


def verify(g):
    res = g["result"]
    if not res.success:
        return {"ok": False, "checks": ["решение не найдено"]}
    x, u = res.x[:g["k"]], res.x[g["k"]:]
    checks = []

    ok = bool(np.all(x >= -TOL) and np.all(u >= -TOL))
    checks.append(f"неотрицательность x, u: {'OK' if ok else 'FAIL'}")

    ok = bool(np.all(x <= g["cap_used"] + TOL))
    checks.append(f"границы дуг x <= capacity: {'OK' if ok else 'FAIL'}")

    ok = bool(np.all(u <= g["demand_used"] + TOL))
    checks.append(f"u <= demand: {'OK' if ok else 'FAIL'}")

    for i, wid in enumerate(g["wids"]):
        out = flow_on(g, x, wid)
        if out > g["supply_used"][i] + TOL:
            checks.append(f"баланс склада {wid}: FAIL ({out} > {g['supply_used'][i]})")
            break
    else:
        checks.append("баланс складов (отгрузка <= запас): OK")

    for j, sid in enumerate(g["sids"]):
        inn = flow_to(g, x, sid)
        if inn + u[j] < g["demand_used"][j] - TOL:
            checks.append(f"баланс магазина {sid}: FAIL")
            break
    else:
        checks.append("баланс магазинов (поставка + дефицит >= спрос): OK")

    obj = float(g["c"] @ res.x)
    ok = abs(obj - res.fun) <= 1e-6 * max(1.0, abs(res.fun))
    checks.append(f"пересчёт цели {obj:.2f} == solver {res.fun:.2f}: "
                  f"{'OK' if ok else 'FAIL'}")

    return {"ok": all("OK" in c for c in checks), "checks": checks,
            "objective": obj}
