

# 🧁 Muffin – Customer Churn & LTV Intelligence Platform

A full-stack analytics platform integrating **SQL-based feature pipelines**, **hybrid ML/LSTM modeling**, and a **Gemini-powered conversational assistant** — purpose-built for data science, product, and growth teams.

Muffin predicts **customer churn**, estimates **lifetime value (LTV)**, segments user behavior, and delivers **real-time, explainable insights** through a warm, intelligent chatbot experience. It empowers stakeholders to ask natural language questions and receive context-aware answers, powered by **Gemini 2.0 Flash (Google AI)**.

---

## 🚀 Live Demos

- **📊 Churn + LTV Dashboard**  
  [`churn-intel-ai.streamlit.app`](https://churn-intel-ai.streamlit.app/)

- **💬 Muffin Chat Assistant**  
  [`via streamlit`](https://churn-gemini.streamlit.app/)  
  [`via hugging space`](https://huggingface.co/spaces/prudhvirekula/muffin-chatbot)

---

## 🧱 Architecture Overview

```
SQL Features  →  ETL Pipeline  →  ML/LSTM Models  →  Streamlit Dashboard & Muffin Chatbot
```

### 🔹 1. SQL Feature Engineering
- RFM metrics, churn flags, payment behavior indicators

### 🔹 2. ETL Automation
- `etl_runner.py` compiles and exports model-ready features (`model_features.csv`)

### 🔹 3. ML & Deep Learning
- `train_ml_model.py`: Random Forest with SHAP explanations
- `train_lstm_multitask.py`: Multitask LSTM for churn + LTV forecasting

### 🔹 4. Streamlit Frontends
- `app.py`: Predict churn, LTV, and visualize customer segments
- `explain_app.py`: Conversational chatbot for explainability

### 🔹 5. Muffin – GenAI Assistant
- Chatbot interface using **Gemini 2.0 Flash (Google AI)** via Gemini API
- Responds to natural queries with emotional context and business logic
- Supports fallback chaining and semantic similarity (via SentenceTransformers)

---

## 🛠 Tech Stack

- **Backend**: Python, Pandas, Scikit-learn, SQLAlchemy
- **Modeling**: Random Forest, LSTM
- **Visualization**: Plotly, Seaborn
- **NLP**: SentenceTransformers, Gemini API
- **UI**: Streamlit (modular multi-app interface)

---

## 📂 Project Structure

### `sql/` – Feature Engineering
- `create_tables.sql`, `load_data.sql`, `churn_flags.sql`, `rfm_features.sql`, etc.

### `etl/` – ETL Automation
- `etl_runner.py`: Automates SQL ingestion → CSV export

### `models/` – ML & DL Models
- `model.pkl`: Churn classifier  
- `ltv_regressor.pkl`: LTV estimator  
- `ltv_scaler.pkl`: Scaler object  
- `train_ml_model.py`, `train_lstm_multitask.py`: Training scripts

### `segment_customers.py` + `segment_app.py`
- Customer segmentation using KMeans + PCA  
- Streamlit-based cluster exploration

### `explain_app.py` – Muffin Chatbot
- Gemini-powered conversational interface  
- `nlp_matcher.py`: Intent matching logic

### `app.py` – Main Dashboard
- Unified view for churn risk, LTV forecast, and segment drilldown

---

## 📦 Usage

```bash
# Clone repository
git clone https://github.com/Prudhvirajrekula/customer-churn-prediction
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Launch main analytics dashboard
streamlit run app.py

# Launch Muffin GenAI chatbot (Gemini)
streamlit run explain_app.py
```


## Screenshots
<img width="1916" height="1085" alt="gemini" src="https://github.com/user-attachments/assets/0de465e9-12f9-48d0-9db2-83c325654479" />
<img width="1917" height="1090" alt="gemini2" src="https://github.com/user-attachments/assets/823844e9-d5b6-4ce9-aa55-fe36528a8d37" />
<img width="1915" height="1078" alt="churn" src="https://github.com/user-attachments/assets/7ab3028b-3dbe-42d2-9db4-91ff42fdf343" />
<img width="1913" height="1077" alt="churn1" src="https://github.com/user-attachments/assets/372b3979-4bfa-4c1f-ac1e-3f72258b26f6" />
<img width="1907" height="1062" alt="churn2" src="https://github.com/user-attachments/assets/1507d679-3d0f-4f5b-8b5f-1a8260c103e2" />
<img width="1911" height="1077" alt="churn3" src="https://github.com/user-attachments/assets/37814b18-86b9-43be-980c-336a37181e8f" />
<img width="1912" height="1081" alt="churn4" src="https://github.com/user-attachments/assets/ab6aebe5-5a81-4834-bac6-bed4f73b2b03" />

---

## ❤️ About Muffin

Muffin is more than a bot — she’s a companion built with care. Always learning.
