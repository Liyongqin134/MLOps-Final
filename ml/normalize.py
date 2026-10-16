"""Normalize digits pixels to [0,1] (dataset v2).

Global min-max normalization so that constant (all-zero) columns stay 0
instead of producing NaN values (column-wise min-max would divide by 0).
"""
import pandas as pd

DATA_PATH = "data/digits.csv"


def normalize(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    lo, hi = X.min().min(), X.max().max()
    Xs = ((X - lo) / (hi - lo)).fillna(0.0)
    pd.concat([Xs, y], axis=1).to_csv(path, index=False)
    print(f"normalized: global min={lo}, max={hi}, NaN={Xs.isna().sum().sum()}")


if __name__ == "__main__":
    normalize()
