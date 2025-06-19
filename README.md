# 🧠 Customer Churn & LTV Intelligence Platform

An enterprise-grade analytics platform that unifies **SQL-driven feature engineering**, **machine learning modeling**, and a **GenAI-powered conversational assistant** into one seamless solution.

This end-to-end system predicts **customer churn risk**, estimates **lifetime value (LTV)**, uncovers **behavioral segments**, and empowers business users to ask natural language questions through **Muffin**, an intelligent AI assistant trained on churn dynamics.

Built for **data scientists, analysts, and growth teams**, this solution delivers explainable predictions and retention insights at scale—powered by modular SQL pipelines, interpretable ML models, and interactive Streamlit dashboards.

---

## 🚀 Live Demo

- **📊 Main Dashboard:** [churn-intel-ai.streamlit.app](https://churn-intel-ai.streamlit.app/)
- **💬 Explainability Chatbot:** [churn-genai-predictor.streamlit.app](https://churn-genai-predictor.streamlit.app/)

---

## 🧱 Architecture

```mermaid
graph TD
    A[SQL Feature Scripts] --> B[ETL Automation (etl_runner.py)]
    B --> C[model_features.csv]
    C --> D[ML/DL Models]
    D --> E[Streamlit app.py / explain_app.py]
```

---

## 🛠 Tech Stack

- **Backend**: Python, Pandas, Scikit-learn, SQLAlchemy
- **Modeling**: Random Forest, LSTM (PyTorch), SHAP, LIME
- **Visualization**: Plotly, Seaborn
- **NLP**: SentenceTransformers (for question matching)
- **UI**: Streamlit (multi-app interface)

---

## 📂 Project Structure & Key Modules

### 🔹 `sql/` – Feature Engineering

- `create_tables.sql`, `load_data.sql`
- `churn_flags.sql`, `rfm_features.sql`, `payment_behavior.sql`, etc.

### 🔹 `etl/` – ETL Automation

- `etl_runner.py`: Automates SQL ingestion → CSV export

### 🔹 `models/` – ML & DL Models

- `model.pkl`: Random Forest churn predictor
- `ltv_regressor.pkl`: LTV estimator
- `ltv_scaler.pkl`: Feature scaler
- `train_ml_model.py`: Builds churn model + SHAP explainer
- `train_lstm_multitask.py`: PyTorch LSTM for churn + LTV multitask

### 🔹 `segment_customers.py` + `segment_app.py` – Segment Explorer

- Customer clustering using KMeans + PCA
- Interactive Streamlit visualization of personas

### 🔹 `explain_app.py` – Muffin GenAI Chatbot

- Chat interface to ask natural questions about churn
- `nlp_matcher.py` uses SentenceTransformers for intent detection

### 🔹 `app.py` – Main Streamlit UI

- Churn + LTV predictor
- Segment viewer
- Link to Muffin chatbot

---

## 🖼 Screenshots

> Add demo screenshots here (`/screenshots/` folder)

---

## 📦 Usage

```bash
# Clone and set up
git clone https://github.com/Prudhvirajrekula/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt

# Launch full app
streamlit run app.py

# Or launch chatbot
streamlit run explain_app.py
```

---
