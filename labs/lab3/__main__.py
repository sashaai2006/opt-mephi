import argparse
from pathlib import Path

import pandas as pd

from .data import load
from .datagen import DATA_DIR, generate
from .model import solver_log
from .report import OUT, plot_lp_vs_milp, plot_scale_curve, report
from .scenarios import run_all
from .verify import split

OUT = Path(__file__).parent / "output"


def ensure_data():
    if not (DATA_DIR / "skus.csv").exists():
        generate()


def cmd_generate(_):
    generate()
    print("data regenerated")


def cmd_all(_):
    ensure_data()
    skus, cats, space = load()
    results = run_all(skus, cats, space)

    OUT.mkdir(exist_ok=True)
    logs = [solver_log("base (MILP)", results["base"]["ctx"]),
            solver_log("base (LP-relaxation)", results["base"]["lp"]),
            solver_log("deficit", results["deficit"]["ctx"]),
            solver_log("stress", results["stress"]["ctx"]),
            solver_log("big-M = 1e6 (MILP)", results["bigm"]["ctx"])]
    (OUT / "solver_log.txt").write_text("\n".join(logs), encoding="utf-8")

    g = results["base"]["ctx"]
    res = g["result"]
    if res.success:
        x, y = split(g)
        rows = []
        for s in range(g["n"]):
            if y[s] > 0.5:
                rows.append({"sku_id": g["sku_ids"][s],
                             "category_id": g["cat_ids"][g["mem"][s]],
                             "stock_x": round(float(x[s]), 4),
                             "space_m": round(float(g["unit_space"][s] * x[s]), 4)})
        pd.DataFrame(rows).to_csv(OUT / "solution_base.csv", index=False)

    srows = []
    for n in ["base", "deficit", "stress"]:
        r = results[n]
        c = r["ctx"]
        srows.append({"scenario": n,
                      "status": "optimal" if c["result"].success else "infeasible",
                      "objective": c["result"].fun if c["result"].success else None,
                      "n_chosen": len(r["chosen"]),
                      "verify_ok": r["verify"]["ok"],
                      "elapsed_s": round(c["elapsed"], 4)})
    srows.append({"scenario": "base_lp_relaxation",
                  "status": "optimal" if results["base"]["lp"]["result"].success else "infeasible",
                  "objective": results["base"]["lp"]["result"].fun,
                  "n_chosen": len(results["base"]["chosen_lp"]),
                  "verify_ok": None,
                  "elapsed_s": round(results["base"]["lp"]["elapsed"], 4)})
    pd.DataFrame(srows).to_csv(OUT / "scenarios.csv", index=False)
    pd.DataFrame(results["curve"]).to_csv(OUT / "scale_curve.csv", index=False)

    plot_lp_vs_milp(results["base"], results["base"]["lp"],
                    OUT / "lp_vs_milp.png")
    plot_scale_curve(results["curve"], OUT / "scale_curve.png")
    report(skus, cats, space, results)

    print(f"base objective (MILP): {res.fun:.2f}")
    for row in srows:
        print(row)
    print(f"report: {OUT / 'report.md'}, plots and CSV in {OUT}")


def main():
    ap = argparse.ArgumentParser(prog="lab3")
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["generate", "all"])
    args = ap.parse_args()
    {"generate": cmd_generate, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
