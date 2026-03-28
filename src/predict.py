# src/predict.py

import pandas as pd
from src.model import load_model

def predict(data, model):
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]

    return prediction, probability