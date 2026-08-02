import pandas as pd

df = pd.read_csv("./data/iris_feast.csv")

df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
df["petal_area"] = df["petal_length"] * df["petal_width"]
df["sepal_ratio"] = df["sepal_length"] / df["sepal_width"]
df["petal_ratio"] = df["petal_length"] / df["petal_width"]
df["flower_area"] = df["sepal_area"] + df["petal_area"]
df["petal_to_sepal_ratio"] = df["petal_area"] / df["sepal_area"]

df["event_timestamp"] = pd.to_datetime(
    df["event_timestamp"],
    utc=True
)

df["created_timestamp"] = pd.to_datetime(
    df["created_timestamp"],
    utc=True
)

df.to_parquet("./iris_feature_store/feature_repo/data/iris.parquet", index=False)