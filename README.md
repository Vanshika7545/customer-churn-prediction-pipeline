# 📊 Customer Churn Prediction Pipeline

---

## 🚀 Overview

This project builds an **end-to-end machine learning system** to predict customer churn and provide explainable insights into customer behavior.

Unlike basic ML projects, this solution focuses on:

* **Business understanding (Cohort Analysis)**
* **Recall optimization (minimizing missed churners)**
* **Explainability (SHAP)**
* **Full-stack deployment (FastAPI + Streamlit)**

---

## 🎯 Objectives

* Identify customers likely to churn
* Optimize model for **Recall** to reduce missed churners
* Perform **Cohort Analysis** for business insights
* Build an **interactive web application** for real-time predictions

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **Explainability:** SHAP
* **Backend:** FastAPI
* **Frontend:** Streamlit

---

## 📂 Project Structure

```
customer-churn-prediction-pipeline/
├── data/              # Raw dataset
├── notebooks/         # EDA & analysis notebooks
├── src/               # ML pipeline (modular code)
├── app/               # FastAPI backend
├── models/            # Saved model, scaler, columns
├── streamlit_app.py   # Frontend UI
├── train.py           # Training pipeline script
├── README.md
```

---

## 📊 Data Understanding

* **Dataset:** Telco Customer Churn Dataset
* **Problem Type:** Binary Classification (Churn / No Churn)

### 🔍 Key Observations

* Significant **class imbalance** observed
* Customers with **low tenure** show higher churn rates
* **Contract type** strongly influences churn behavior
* Pricing-related features impact retention

---

## 🔥 Cohort Analysis (Business Understanding)

Performed cohort-based segmentation on:

* Tenure groups
* Contract types
* Monthly charges

### 📌 Insights

* Month-to-month users churn the most
* New customers (low tenure) are high-risk
* High charges increase churn probability
* Long-term contracts improve retention

---

## 📈 Model Performance

| Model               | Recall | Precision | F1 Score |
| ------------------- | ------ | --------- | -------- |
| Logistic Regression | 0.71   | 0.51      | 0.59     |
| Decision Tree       | 0.49   | 0.49      | 0.49     |
| Random Forest       | 0.48   | 0.62      | 0.55     |
| XGBoost             | 0.54   | 0.58      | 0.56     |

👉 **Final Model: Logistic Regression**
Chosen due to **high recall**, aligning with business goal of minimizing missed churners.

---

## 🎯 Threshold Optimization

Instead of default 0.5 threshold:

* Lowered threshold → increased recall
* Accepted slight drop in precision

👉 Trade-off aligned with business:

> Missing a churner is more costly than false positives

---

## 🔍 Explainability (SHAP)

Used SHAP (SHapley Additive exPlanations) to interpret model predictions.

### 🌍 Global Insights

* **Tenure** is the most influential feature
* High **monthly charges** increase churn risk
* **Fiber optic users** show higher churn tendency
* Long-term contracts reduce churn

### 👤 Individual Prediction

SHAP explains:

* Why a specific customer will churn
* Which features contributed positively/negatively

---

## 🧠 Business Insights

* Improve onboarding for new customers
* Promote long-term contracts
* Optimize pricing strategies
* Target high-risk users with retention offers

---

## ⚙️ System Architecture

```
User → Streamlit UI → FastAPI → ML Model → Prediction
```

* Frontend: Streamlit
* Backend: FastAPI
* Model served via API
* Real-time predictions enabled

---

## 🖥️ Demo

### 🔹 Streamlit UI

(Add screenshot here)

### 🔹 Prediction Output

(Add screenshot here)

### 🔹 API Docs

(Add screenshot here)

---

## ▶️ How to Run

### 1. Clone repository

```
git clone <your-repo-link>
cd customer-churn-prediction-pipeline
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run backend

```
uvicorn app.main:app --reload
```

### 4. Run frontend

```
streamlit run streamlit_app.py
```

---

## 🚀 Features

* End-to-end ML pipeline
* Cohort-based business analysis
* Recall-focused model optimization
* SHAP-based explainability
* FastAPI backend
* Streamlit interactive UI

---

## 💡 Future Improvements

* Deploy application online (Streamlit Cloud / Render)
* Add SHAP explanations in UI
* Improve UI/UX with advanced visualizations
* Integrate real-time data pipeline

---

## 📌 Key Takeaway

This project demonstrates the ability to:

* Solve real-world business problems using ML
* Build scalable and modular pipelines
* Deploy machine learning models as applications
* Bridge the gap between data science and production systems

---

