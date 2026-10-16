"""Export sklearn digits dataset to CSV for DVC tracking."""
import pandas as pd
from sklearn.datasets import load_digits

def build_digits_csv(path="data/digits.csv"):
    digits = load_digits()
    df = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(64)])
    df["target"] = digits.target
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols to {path}")

if __name__ == "__main__":
    build_digits_csv()
