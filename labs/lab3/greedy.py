import time

import numpy as np

from .model import solve


def greedy(skus, cats, space):
    n = len(skus)
    d = skus["demand"].to_numpy(float)
    ms = skus["min_stock"].to_numpy(float)
    a = skus["unit_space"].to_numpy(float)
    U = skus["max_stock"].to_numpy(float)
    D = cats["demand"].to_numpy(float)
    nmin = cats["min_skus"].to_numpy(float)
    cat_of = {cid: i for i, cid in enumerate(cats["category_id"])}
    mem = np.array([cat_of[c] for c in skus["category_id"]])

    t0 = time.perf_counter()
    y = np.zeros(n)
    x = np.zeros(n)
    for c in range(len(cats)):
        idx = np.where(mem == c)[0]
        order = idx[np.argsort(ms[idx])]
        covered = 0.0
        taken = []
        for s in order:
            if len(taken) >= int(nmin[c]) and covered >= D[c]:
                break
            taken.append(s)
            covered += U[s]
        need = D[c]
        for s in taken:
            give = min(U[s], max(ms[s], need))
            x[s] = give
            y[s] = 1.0
            need -= give
        if need > 0:
            for s in order:
                if y[s] == 0 and need > 0:
                    give = min(U[s], need)
                    x[s] = max(ms[s], give)
                    y[s] = 1.0
                    need -= give
    elapsed = time.perf_counter() - t0
    obj = float(y.sum())
    feasible = bool(a @ x <= space + 1e-6)
    return {"objective": obj, "x": x, "y": y, "elapsed": elapsed,
            "feasible_space": feasible}
