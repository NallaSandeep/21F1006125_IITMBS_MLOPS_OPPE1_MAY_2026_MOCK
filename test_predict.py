import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://34.66.27.54:8100")

model = mlflow.sklearn.load_model(
    model_uri="models:/IrisDecisionTree/latest"
)

test = pd.read_csv("./data/iris_test.csv")

X_test = test.drop("species", axis=1)
y_test = test["species"]

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
