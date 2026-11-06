import pandas as pd


# binary encoding
def map_sex(x: str) -> float:
    match x:
        case "male":
            return 0
        case "female":
            return 1
        case _:
            raise NotImplementedError


def map_age(x: str, default: float) -> float:
    if not x:
        return default
    return float(x)


"""
TRAIN
"""

df = pd.read_csv("original/train.csv")

# we use: Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked

# Feature #1: Pclass
df["Pclass_1"] = df["Pclass"].map(lambda x: 1 if x == 1 else 0)
df["Pclass_2"] = df["Pclass"].map(lambda x: 1 if x == 2 else 0)
df["Pclass_3"] = df["Pclass"].map(lambda x: 1 if x == 3 else 0)

# Feature #2: Sex {0, 1}
df["Sex"] = df["Sex"].map(map_sex)

# Feature #3: Age [0; 1]
df["Age"] = df["Age"] / df["Age"].max()
df["Age"] = df["Age"].fillna(0.5)

# Feature #4: SibSp [0; 1]
df["SibSp"] = df["SibSp"] / df["SibSp"].max()

# Feature #5 Parch [0; 1]
df["Parch"] = df["Parch"] / df["Parch"].max()

# Feature #6: Fare [0; 1]
df["Fare"] = df["Fare"] / df["Fare"].max()
df["Fare"] = df["Fare"].fillna(0.5)

# Feature #7: Embarked
df["Embarked_C"] = df["Embarked"].map(lambda x: 1 if x == "C" else 0)
df["Embarked_Q"] = df["Embarked"].map(lambda x: 1 if x == "Q" else 0)
df["Embarked_S"] = df["Embarked"].map(lambda x: 1 if x == "S" else 0)

# drop some of the original columns
df = df.drop(
    [
        "PassengerId",
        "Pclass",
        "Name",
        "Ticket",
        "Cabin",
        "Embarked",
    ],
    axis=1,
)

df.to_csv("processed/train_processed.csv", index=False)


"""
TEST
"""

df = pd.read_csv("original/test.csv")

# we use: Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked

# Feature #1: Pclass
df["Pclass_1"] = df["Pclass"].map(lambda x: 1 if x == 1 else 0)
df["Pclass_2"] = df["Pclass"].map(lambda x: 1 if x == 2 else 0)
df["Pclass_3"] = df["Pclass"].map(lambda x: 1 if x == 3 else 0)

# Feature #2: Sex {0, 1}
df["Sex"] = df["Sex"].map(map_sex)

# Feature #3: Age [0; 1]
df["Age"] = df["Age"] / df["Age"].max()
df["Age"] = df["Age"].fillna(0.5)

# Feature #4: SibSp [0; 1]
df["SibSp"] = df["SibSp"] / df["SibSp"].max()

# Feature #5 Parch [0; 1]
df["Parch"] = df["Parch"] / df["Parch"].max()

# Feature #6: Fare [0; 1]
df["Fare"] = df["Fare"] / df["Fare"].max()
df["Fare"] = df["Fare"].fillna(0.5)

# Feature #7: Embarked
df["Embarked_C"] = df["Embarked"].map(lambda x: 1 if x == "C" else 0)
df["Embarked_Q"] = df["Embarked"].map(lambda x: 1 if x == "Q" else 0)
df["Embarked_S"] = df["Embarked"].map(lambda x: 1 if x == "S" else 0)

# drop some of the original columns
df = df.drop(
    [
        "Pclass",
        "Name",
        "Ticket",
        "Cabin",
        "Embarked",
    ],
    axis=1,
)

df.to_csv("processed/test_processed.csv", index=False)
