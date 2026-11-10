import argparse

import pandas as pd

from .data import load
from .datagen import DATA_DIR, generate
from .model import solver_log, split
from .report import OUT, plot_network, plot_scenarios, report
from .scenarios import run_all


def ensure_data():
    if not (DATA_DIR / "routes.csv").exists():
        generate()


def cmd_generate(_):
    generate()
    print("data regenerated")


def cmd_all(_):
    ensure_data()
    warehouses, shops, routes = load()
    results = run_all(warehouses, shops, routes)

    OUT.mkdir(exist_ok=True)
    logs = [solver_log(n, r["ctx"]) for n, r in results.items()]
    (OUT / "solver_log.txt").write_text("\n".join(logs), encoding="utf-8")

    g = results["base"]["ctx"]
    res = g["result"]
    if res.success:
        x, u = split(g)
        rows = []
        for t, (a, b) in enumerate(g["arc_ij"]):
            if x[t] > 1e-9:
                rows.append({"route": f"{a}->{b}", "from": a, "to": b,
                             "flow": round(float(x[t]), 4),
                             "capacity": g["cap_used"][t],
                             "cost_per_unit": g["cost"][t],
                             "cost_total": round(float(x[t] * g["cost"][t]), 2)})
        for j, sid in enumerate(g["sids"]):
            if u[j] > 1e-9:
                rows.append({"route": f"deficit:{sid}", "from": "-",
                             "to": sid, "flow": round(float(u[j]), 4),
                             "capacity": g["demand_used"][j],
                             "cost_per_unit": g["penalty"][j],
                             "cost_total": round(float(u[j] * g["penalty"][j]), 2)})
        pd.DataFrame(rows).to_csv(OUT / "solution_base.csv", index=False)

    srows = []
    for n, r in results.items():
        c = r["ctx"]
        srows.append({"scenario": n,
                      "status": "optimal" if c["result"].success else "infeasible",
                      "objective": c["result"].fun if c["result"].success else None,
                      "nit": getattr(c["result"], "nit", None),
                      "verify_ok": r["verify"]["ok"],
                      "elapsed_s": round(c["elapsed"], 4)})
    pd.DataFrame(srows).to_csv(OUT / "scenarios.csv", index=False)

    plot_network(warehouses, shops, routes, g, OUT / "network_flow.png")
    plot_scenarios(results, OUT / "cost_by_scenario.png")
    report(warehouses, shops, routes, results)

    print(f"base objective: {res.fun:.2f}")
    for row in srows:
        print(row)
    print(f"report: {OUT / 'report.md'}, plots and CSV in {OUT}")


def main():
    ap = argparse.ArgumentParser(prog="lab2")
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["generate", "all"])
    args = ap.parse_args()
    {"generate": cmd_generate, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
