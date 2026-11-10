import numpy as np

TOL = 1e-6


def split(g):
    res = g["result"]
    return res.x[:g["n"]], res.x[g["n"]:]


def verify(g):
    res = g["result"]
    if not res.success:
        return {"ok": False, "checks": ["решение не найдено"]}
    x, y = split(g)
    checks = []

    ok = bool(np.all(x >= -TOL))
    checks.append(f"x >= 0: {'OK' if ok else 'FAIL'}")

    ok = bool(np.all(y >= -TOL) and np.all(y <= 1 + TOL))
    checks.append(f"0 <= y <= 1: {'OK' if ok else 'FAIL'}")

    if g["mip"]:
        ok = bool(np.all(np.abs(y - np.round(y)) <= 1e-4))
        checks.append(f"целочисленность y (бинарность): {'OK' if ok else 'FAIL'}")

    sel = y > 0.5
    ok = bool(np.all(x[sel] <= g["max_stock"][sel] + 1e-4)
              and np.all(x[~sel] <= 1e-4))
    checks.append(f"связки big-M x <= U*y: {'OK' if ok else 'FAIL'}")

    ok = bool(np.all(x + 1e-4 >= g["min_stock"] * y))
    checks.append(f"мин. выкладка x >= m*y: {'OK' if ok else 'FAIL'}")

    for c in range(g["m"]):
        cov = x[g["mem"] == c].sum()
        if cov < g["D"][c] - 1e-4:
            checks.append(f"покрытие категории {g['cat_ids'][c]}: FAIL")
            break
    else:
        checks.append("покрытие спроса всех категорий: OK")

    for c in range(g["m"]):
        cnt = y[g["mem"] == c].sum()
        if cnt < g["min_skus"][c] - 1e-4:
            checks.append(f"мин. число SKU в {g['cat_ids'][c]}: FAIL")
            break
    else:
        checks.append("мин. число SKU по категориям: OK")

    used = float(g["unit_space"] @ x)
    ok = used <= g["space"] + 1e-4
    checks.append(f"бюджет полки {used:.2f} <= {g['space']:.2f} м: "
                  f"{'OK' if ok else 'FAIL'}")

    obj = float(g["c"] @ res.x)
    ok = abs(obj - res.fun) <= 1e-6 * max(1.0, abs(res.fun))
    checks.append(f"пересчёт цели {obj:.2f} == solver {res.fun:.2f}: "
                  f"{'OK' if ok else 'FAIL'}")

    return {"ok": all("OK" in c for c in checks), "checks": checks,
            "objective": obj, "space_used": used}


def chosen(g):
    res = g["result"]
    if not res.success:
        return set()
    x, y = split(g)
    return {g["sku_ids"][s] for s in range(g["n"]) if y[s] > 0.5}
