import csv
import random
from pathlib import Path

SEED = 10217
DATA_DIR = Path(__file__).parent / "data"


def generate():
    rng = random.Random(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    warehouses = []
    supplies = rng.sample(range(28, 46), 4)
    for i, s in enumerate(supplies, 1):
        warehouses.append({"warehouse_id": f"W{i}",
                           "name": f"Склад {i}",
                           "supply": s})

    shops = []
    demands = rng.sample(range(18, 34), 6)
    for j, d in enumerate(demands, 1):
        shops.append({"shop_id": f"M{j}",
                      "name": f"Магазин {j}",
                      "demand": d,
                      "penalty": rng.randint(34, 58)})

    routes = []
    k = 1
    for w in warehouses:
        for s in shops:
            routes.append({"route_id": f"R{k}",
                           "from": w["warehouse_id"],
                           "to": s["shop_id"],
                           "cost": round(rng.uniform(7.0, 24.0), 2),
                           "capacity": rng.randint(6, 22)})
            k += 1

    def dump(name, rows):
        with open(DATA_DIR / name, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    dump("warehouses.csv", warehouses)
    dump("shops.csv", shops)
    dump("routes.csv", routes)
