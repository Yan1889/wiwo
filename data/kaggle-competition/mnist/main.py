import pandas as pd

"""
Train
"""
df = pd.read_csv("original/train.csv")
df[[f"pixel{i}" for i in range(784)]] /= 255.0
df.to_csv("processed/train.csv", index=False)

"""
Test
"""
df = pd.read_csv("original/test.csv")
df[[f"pixel{i}" for i in range(784)]] /= 255.0
df.insert(0, "ImageId", pd.Series(range(1, len(df) + 1)))
df.to_csv("processed/test.csv", index=False)
