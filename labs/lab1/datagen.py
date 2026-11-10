from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 10217
DATA_DIR = Path(__file__).parent / "data"


def generate():
    rng = np.random.default_rng(SEED)

    sheets = pd.DataFrame([
        {"sheet_id": "S1", "name": "Лист 2000x1000", "L": 2000, "W": 1000,
         "cost_base": 300.0, "supply_base": 60},
        {"sheet_id": "S2", "name": "Лист 2500x1250", "L": 2500, "W": 1250,
         "cost_base": 375.0, "supply_base": 50},
        {"sheet_id": "S3", "name": "Лист 3000x1500", "L": 3000, "W": 1500,
         "cost_base": 520.0, "supply_base": 40},
        {"sheet_id": "S4", "name": "Лист 1500x1500", "L": 1500, "W": 1500,
         "cost_base": 290.0, "supply_base": 70},
    ])
    sheets["cost"] = np.round(
        sheets["cost_base"] * (1 + rng.uniform(-0.03, 0.03, len(sheets))), 1)
    sheets["supply"] = sheets["supply_base"] + rng.integers(-5, 6, len(sheets))
    sheets = sheets.drop(columns=["cost_base", "supply_base"])

    items = pd.DataFrame([
        {"item_id": "I1", "name": "Панель 900x500", "l": 900, "w": 500,
         "demand_base": 40},
        {"item_id": "I2", "name": "Панель 700x600", "l": 700, "w": 600,
         "demand_base": 55},
        {"item_id": "I3", "name": "Панель 500x500", "l": 500, "w": 500,
         "demand_base": 70},
        {"item_id": "I4", "name": "Панель 1200x400", "l": 1200, "w": 400,
         "demand_base": 25},
        {"item_id": "I5", "name": "Панель 600x300", "l": 600, "w": 300,
         "demand_base": 90},
    ])
    items["demand"] = items["demand_base"] + rng.integers(-3, 4, len(items))
    items = items.drop(columns=["demand_base"])

    patterns = []
    pid = 0
    k_max = 8
    for _, sh in sheets.iterrows():
        fits = [(i, it) for i, it in items.iterrows() if it["w"] <= sh["W"]]
        ranges = [range(0, min(int(sh["L"] // it["l"]), k_max) + 1)
                  for _, it in fits]
        for combo in product(*ranges):
            if sum(combo) == 0 or sum(combo) > k_max:
                continue
            if sum(k * it["l"] for k, (_, it) in zip(combo, fits)) > sh["L"]:
                continue
            if sum(1 for k in combo if k > 0) > 3:
                continue
            pid += 1
            row = {"pattern_id": f"P{pid}", "sheet_id": sh["sheet_id"]}
            counts = dict(zip((it["item_id"] for _, it in fits), combo))
            for iid in items["item_id"]:
                row[f"k_{iid}"] = counts.get(iid, 0)
            patterns.append(row)
    patterns = pd.DataFrame(patterns)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sheets.to_csv(DATA_DIR / "sheets.csv", index=False)
    items.to_csv(DATA_DIR / "items.csv", index=False)
    patterns.to_csv(DATA_DIR / "patterns.csv", index=False)
    return sheets, items, patterns


if __name__ == "__main__":
    s, i, p = generate()
    print(f"sheets: {len(s)}, items: {len(i)}, patterns: {len(p)}")
    print(p.groupby("sheet_id").size())
