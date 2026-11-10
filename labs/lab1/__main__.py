import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load
from .datagen import DATA_DIR, generate
from .model import solver_log
from .report import plots, report
from .scenarios import run_all

OUT = Path(__file__).parent / "output"


def ensure_data():
    if not (DATA_DIR / "sheets.csv").exists():
        generate()


def cmd_generate(_):
    generate()
    print("data regenerated")


def cmd_all(_):
    ensure_data()
    sheets, items, patterns = load()
    results = run_all(sheets, items, patterns)

    OUT.mkdir(exist_ok=True)
    logs = [solver_log(n, r["ctx"]) for n, r in results.items()]
    (OUT / "solver_log.txt").write_text("\n".join(logs))

    res = results["base"]["ctx"]["result"]
    if res.success:
        sol = pd.DataFrame({
            "pattern_id": patterns["pattern_id"],
            "sheet_id": patterns["sheet_id"],
            "sheets_used": res.x,
        })
        sol[sol["sheets_used"] > 1e-9].to_csv(OUT / "solution_base.csv",
                                              index=False)

    rows = []
    for n, r in results.items():
        ctx = r["ctx"]
        rows.append({
            "scenario": n,
            "status": "optimal" if ctx["result"].success else "infeasible",
            "objective": ctx["result"].fun if ctx["result"].success else None,
            "nit": getattr(ctx["result"], "nit", None),
            "verify_ok": r["verify"]["ok"],
            "elapsed_s": round(ctx["elapsed"], 4),
        })
    pd.DataFrame(rows).to_csv(OUT / "scenarios.csv", index=False)

    plots(results)
    sol_df = report(sheets, items, patterns, results)
    print(f"base objective: {res.fun:.2f}")
    for row in rows:
        print(row)
    print(f"report: {OUT / 'report.md'}, plots and CSV in {OUT}")


def main():
    ap = argparse.ArgumentParser(prog="lab1")
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["generate", "all"])
    args = ap.parse_args()
    {"generate": cmd_generate, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
