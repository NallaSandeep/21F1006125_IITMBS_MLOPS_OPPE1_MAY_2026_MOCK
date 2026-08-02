import pandas as pd


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def impute_last_10(values):
    """
    Impute each missing value using the mean of the last
    10 available (non-missing) values.
    """
    history = []
    result = []

    for value in values:
        if pd.isna(value):
            previous_values = history[-10:]

            if previous_values:
                value = sum(previous_values) / len(previous_values)

        result.append(value)

        # Add only available values to history
        if not pd.isna(value):
            history.append(value)

    return pd.Series(result, index=values.index)


def impute_missing_values(data):
    """
    Impute missing feature values using the mean of the last
    10 available samples belonging to the same species.
    """
    data = data.copy()

    for feature in FEATURE_COLUMNS:
        data[feature] = (
            data.groupby("species", group_keys=False)[feature]
            .apply(impute_last_10)
        )

    return data


def main():
    input_path = "./data/iris_missing.csv"
    output_path = "./data/iris_imputed.csv"

    data = pd.read_csv(input_path)

    print("Missing values before imputation:")
    print(data[FEATURE_COLUMNS].isnull().sum())

    data = impute_missing_values(data)

    print("\nMissing values after imputation:")
    print(data[FEATURE_COLUMNS].isnull().sum())

    data.to_csv(output_path, index=False)

    print(f"\nImputed data saved to {output_path}")


if __name__ == "__main__":
    main()