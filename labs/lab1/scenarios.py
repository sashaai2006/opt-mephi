import numpy as np

from .model import solve
from .verify import verify


def marginals(ctx):
    res = ctx["result"]
    if not res.success:
        return None
    m = ctx["A"].shape[0]
    mu = res.ineqlin.marginals
    return {"demand": -mu[:m], "supply": mu[m:]}


def run_all(sheets, items, patterns):
    out = {}

    base = solve(sheets, items, patterns)
    out["base"] = {"ctx": base, "verify": verify(base), "marginals": marginals(base)}

    vb = out["base"]["verify"]
    x = np.asarray(base["result"].x)
    used = base["S"] @ x
    t_def = int(np.argmax(used))
    s_def = base["supply"].copy()
    s_def[t_def] = np.floor(0.8 * s_def[t_def])
    deficit = solve(sheets, items, patterns, supply=s_def)
    out["deficit"] = {"ctx": deficit, "verify": verify(deficit),
                      "marginals": marginals(deficit),
                      "changed_supply": (base["sheet_ids"][t_def],
                                         float(base["supply"][t_def]),
                                         float(s_def[t_def]))}

    d_stress = base["demand"].copy()
    top2 = np.argsort(d_stress)[-2:]
    d_stress[top2] = np.ceil(1.2 * d_stress[top2])
    stress = solve(sheets, items, patterns, demand=d_stress, supply=s_def)
    out["stress"] = {"ctx": stress, "verify": verify(stress),
                     "marginals": marginals(stress),
                     "changed_supply": out["deficit"]["changed_supply"],
                     "changed_demand": [(items["item_id"].iloc[i],
                                         float(base["demand"][i]),
                                         float(d_stress[i])) for i in top2]}
    return out


def shadow_price_check(results):
    base = results["base"]
    mb = base["marginals"]
    rows = []
    if results["deficit"]["ctx"]["result"].success and mb is not None:
        sid, s0, s1 = results["deficit"]["changed_supply"]
        t = base["ctx"]["sheet_ids"].index(sid)
        pred = mb["supply"][t] * (s1 - s0)
        actual = results["deficit"]["ctx"]["result"].fun - base["ctx"]["result"].fun
        rows.append(("запас " + sid, s1 - s0, pred, actual))
    return rows
