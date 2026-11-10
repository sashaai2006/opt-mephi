import pandas as pd

from .datagen import DATA_DIR


def load():
    skus = pd.read_csv(DATA_DIR / "skus.csv")
    cats = pd.read_csv(DATA_DIR / "categories.csv")
    shelf = pd.read_csv(DATA_DIR / "shelf.csv")
    return skus, cats, shelf["total_space"].iloc[0]
