import numpy as np


def greedy(warehouses, shops, routes):
    s = warehouses["supply"].to_numpy(float).copy()
    d = shops["demand"].to_numpy(float).copy()
    cap = routes["capacity"].to_numpy(float).copy()
    cost = routes["cost"].to_numpy(float)
    pen = shops["penalty"].to_numpy(float)
    wids = list(warehouses["warehouse_id"])
    sids = list(shops["shop_id"])
    arc_ij = [(a, b) for a in wids for b in sids]

    order = np.argsort(cost)
    x = np.zeros(len(arc_ij))
    for t in order:
        a, b = arc_ij[t]
        i, j = wids.index(a), sids.index(b)
        g = min(s[i], cap[t], d[j])
        x[t] = g
        s[i] -= g
        d[j] -= g
    u = np.maximum(d, 0.0)
    obj = float(cost @ x + pen @ u)
    return {"x": x, "unmet": u, "objective": obj}
