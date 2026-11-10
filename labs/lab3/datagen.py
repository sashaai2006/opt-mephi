import csv
import math
import random
from pathlib import Path

SEED = 10217
DATA_DIR = Path(__file__).parent / "data"


def generate():
    rng = random.Random(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    cats = []
    cat_names = ["Молочные продукты", "Хлеб и выпечка", "Овощи и фрукты",
                 "Бакалея", "Напитки", "Консервы"]
    for i, name in enumerate(cat_names, 1):
        cats.append({"category_id": f"C{i}", "name": name})

    skus = []
    counts = [4, 3, 4, 3, 3, 3]
    k = 1
    for ci, cnt in enumerate(counts):
        for _ in range(cnt):
            d = rng.randint(6, 28)
            m = rng.randint(3, 8)
            a = round(rng.uniform(0.05, 0.15), 3)
            skus.append({"sku_id": f"SKU{k:02d}",
                         "name": f"Товар {k:02d}",
                         "category_id": f"C{ci + 1}",
                         "demand": d,
                         "min_stock": m,
                         "unit_space": a,
                         "max_stock": math.ceil(1.1 * d)})
            k += 1

    cat_rows = []
    need_space = 0.0
    for ci, c in enumerate(cats):
        members = [s for s in skus if s["category_id"] == c["category_id"]]
        total_u = sum(s["max_stock"] for s in members)
        total_d = sum(s["demand"] for s in members)
        D = min(int(0.75 * total_d), total_u)
        sp = 0.0
        need = D
        for s in sorted(members, key=lambda v: v["unit_space"]):
            take = min(s["max_stock"], need)
            sp += take * s["unit_space"]
            need -= take
        D -= max(0, need)
        need_space += sp
        cat_rows.append({"category_id": c["category_id"], "name": c["name"],
                         "demand": D, "min_skus": 2})

    total_space = round(1.3 * need_space + 0.5, 1)

    def dump(name, rows):
        with open(DATA_DIR / name, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    dump("categories.csv", cat_rows)
    dump("skus.csv", skus)
    with open(DATA_DIR / "shelf.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["total_space"])
        wr.writeheader()
        wr.writerow({"total_space": total_space})
