"""
Script for embedding the train and test sets
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("baselines/dataset")
assert DATA_DIR.exists, FileNotFoundError("The path doesn't exist")


def main():
    model = SentenceTransformer("distilbert-base-uncased")
    file_names = ["train.csv", "test.csv"]
    for file in file_names:
        filepath = DATA_DIR / file
        df = pd.read_csv(filepath)
        embs = model.encode(df["input"].tolist())
        np.save(DATA_DIR / f"{filepath.stem}.npy", embs)


if __name__ == "__main__":
    main()
