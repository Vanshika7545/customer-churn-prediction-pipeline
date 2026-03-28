from src.data_preprocessing import load_data, preprocess_data, split_data, scale_data
from src.model import train_model, save_model, save_scaler
import pickle

# Step 1: Load data
df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 2: Preprocess
X, y = preprocess_data(df)

# Step 3: Split
X_train, X_test, y_train, y_test = split_data(X, y)

# Step 4: Scale
X_train, X_test, scaler = scale_data(X_train, X_test)

# Step 5: Train model
model = train_model(X_train, y_train)

# Step 6: Save model + scaler
save_model(model)
save_scaler(scaler)

print("Model trained and saved successfully ")

# Save feature columns
with open("models/columns.pkl", "wb") as f:
    pickle.dump(list(X_train.columns), f)