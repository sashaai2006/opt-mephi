from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load():
    sheets = pd.read_csv(DATA_DIR / "sheets.csv")
    items = pd.read_csv(DATA_DIR / "items.csv")
    patterns = pd.read_csv(DATA_DIR / "patterns.csv")
    return sheets, items, patterns
