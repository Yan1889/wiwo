import pandas as pd

df = pd.read_csv("original.csv")

# remove unnecessary columns
del df["Id"]

# normalize features in [0; 1]
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
df[features] /= df[features].max()

# make classes integers
df["Species"] = df["Species"].map(
    {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
)

# put label column in front
df.insert(0, "Species", df.pop("Species"))

df.to_csv("processed.csv", index=False)
