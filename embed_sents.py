"""
Script for embedding the train and test sets into separate sentences
"""
from typing import Iterable, List
import pandas as pd
import numpy as np
from pathlib import Path
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("baselines/dataset")
assert DATA_DIR.exists, FileNotFoundError("The path doesn't exist")


def flatten_list(lst: Iterable) -> List[List]:
    return [x for sublist in lst for x in sublist]


def main():
    model = SentenceTransformer("distilbert-base-uncased")
    file_names = ["train.csv", "test.csv"]
    for file in file_names:
        filepath = DATA_DIR / file
        df = pd.read_csv(filepath)
        df = df.assign(sents=df["input"].apply(sent_tokenize))
        df = df.reset_index().explode("sents")
        indeces = df["index"].values
        np.save(DATA_DIR / f"{filepath.stem}_sent_idx.npy", indeces)

        flat_sents = df["sents"].tolist()
        sent_embs = model.encode(flat_sents)
        np.save(DATA_DIR / f"{filepath.stem}_sents.npy", sent_embs)


if __name__ == "__main__":
    main()
