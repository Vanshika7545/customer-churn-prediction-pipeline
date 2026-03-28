# src/model.py

from sklearn.linear_model import LogisticRegression
import pickle

def train_model(X_train, y_train):
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    return model


def save_model(model, path="models/model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path="models/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
    
import pickle

def save_scaler(scaler, path="models/scaler.pkl"):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path="models/scaler.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)    