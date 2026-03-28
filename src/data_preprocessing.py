# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    # Cleaning
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop("customerID", axis=1)

    # Target encoding
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Split features
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    num_cols = X_train.select_dtypes(exclude=["object"]).columns

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, scaler