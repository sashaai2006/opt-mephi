import pandas as pd

from .datagen import DATA_DIR


def load():
    frames = {
        "warehouses": pd.read_csv(DATA_DIR / "warehouses.csv"),
        "shops": pd.read_csv(DATA_DIR / "shops.csv"),
        "routes": pd.read_csv(DATA_DIR / "routes.csv"),
    }
    return frames["warehouses"], frames["shops"], frames["routes"]
