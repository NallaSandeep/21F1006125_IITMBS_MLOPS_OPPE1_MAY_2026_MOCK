import os
import joblib
import feast
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# -------------------------------------------------------
# Connect to Feast Feature Store
# -------------------------------------------------------

fs = feast.FeatureStore(repo_path="./iris_feature_store/feature_repo")

# -------------------------------------------------------
# Create Entity DataFrame
# -------------------------------------------------------

entity_df = pd.read_csv("./data/iris_feast.csv")

entity_df = entity_df[["iris_id", "event_timestamp"]]
entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True)

# -------------------------------------------------------
# Retrieve historical features from Feast
# -------------------------------------------------------

training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
        "iris_features:sepal_area",
        "iris_features:petal_area",
        "iris_features:sepal_ratio",
        "iris_features:petal_ratio",
        "iris_features:flower_area",
        "iris_features:petal_to_sepal_ratio",
    ],
).to_df()

# -------------------------------------------------------
# Add target labels
# -------------------------------------------------------

labels = pd.read_csv("./data/iris_feast.csv")[["iris_id", "species"]]

training_df = training_df.merge(labels, on="iris_id")

# -------------------------------------------------------
# Prepare train/test data
# -------------------------------------------------------

X = training_df.drop(
    columns=[
        "iris_id",
        "event_timestamp",
        "species",
    ]
)

y = training_df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    stratify=y,
    random_state=42,
)

# -------------------------------------------------------
# Train model
# -------------------------------------------------------

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=1,
)

model.fit(X_train, y_train)

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------

prediction = model.predict(X_test)

print(
    "Accuracy:",
    metrics.accuracy_score(y_test, prediction),
)

# -------------------------------------------------------
# Save predictions
# -------------------------------------------------------

results = X_test.copy()

results["Actual"] = y_test.values
results["Predicted"] = prediction

artifact_dir = "./artifacts"

os.makedirs(artifact_dir, exist_ok=True)

results.to_csv(
    f"{artifact_dir}/predictions.csv",
    index=False,
)