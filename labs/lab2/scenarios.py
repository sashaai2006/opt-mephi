import numpy as np

from .model import solve, split
from .verify import verify


def marginals(g):
    res = g["result"]
    if not res.success:
        return None
    mu = res.ineqlin.marginals
    m, n, k = g["m"], g["n"], g["k"]
    return {"supply": mu[:m], "demand": mu[m:m + n], "capacity": mu[m + n:]}


def bottlenecks(g, tol=1e-6):
    res = g["result"]
    if not res.success:
        return []
    x, _ = split(g)
    mg = marginals(g)
    rows = []
    for t, (a, b) in enumerate(g["arc_ij"]):
        sat = g["cap_used"][t] > 0 and x[t] >= g["cap_used"][t] - 1e-6
        rows.append({"route": f"{a}→{b}", "flow": round(float(x[t]), 2),
                     "capacity": g["cap_used"][t],
                     "saturated": bool(sat),
                     "shadow_price": round(float(-mg["capacity"][t]), 2)})
    return rows


def run_all(warehouses, shops, routes):
    out = {}

    base = solve(warehouses, shops, routes)
    out["base"] = {"ctx": base, "verify": verify(base), "marginals": marginals(base)}

    x, _ = split(base)
    t_max = int(np.argmax(x))
    closed_arc = base["arc_ij"][t_max]
    failure = solve(warehouses, shops, routes, closed={closed_arc})
    out["failure"] = {"ctx": failure, "verify": verify(failure),
                      "marginals": marginals(failure),
                      "closed_arc": closed_arc}

    i_max = int(np.argmax(base["supply_used"]))
    s_def = base["supply_used"].copy()
    s_def[i_max] = np.floor(0.8 * s_def[i_max])
    deficit = solve(warehouses, shops, routes, supply=s_def)
    out["deficit"] = {"ctx": deficit, "verify": verify(deficit),
                      "marginals": marginals(deficit),
                      "changed_supply": (base["wids"][i_max],
                                         float(base["supply_used"][i_max]),
                                         float(s_def[i_max]))}

    d_up = np.ceil(1.2 * base["demand_used"])
    stress = solve(warehouses, shops, routes, demand=d_up)
    out["stress"] = {"ctx": stress, "verify": verify(stress),
                     "marginals": marginals(stress)}

    return out
