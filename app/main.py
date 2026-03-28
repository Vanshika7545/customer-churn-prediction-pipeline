from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

from src.model import load_model, load_scaler

app = FastAPI()

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Load columns
with open("models/columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running 🚀"}


class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict(data: CustomerData):

    input_dict = data.dict()

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in model_columns:
        if col not in input_df:
            input_df[col] = 0

    # Ensure correct order
    input_df = input_df[model_columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn": int(prediction),
        "probability": float(probability)
    }