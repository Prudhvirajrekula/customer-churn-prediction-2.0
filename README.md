
# 🧁 Muffin – Customer Churn & LTV Intelligence Platform

An enterprise-grade analytics platform that unifies **SQL-driven feature engineering**, **ML & LSTM modeling**, and a **GenAI-powered conversational assistant** — all in one seamless solution.

This end-to-end system predicts **customer churn risk**, estimates **lifetime value (LTV)**, identifies **behavioral segments**, and empowers users to ask natural questions using **Muffin**, a warm and intelligent assistant that understands churn dynamics at heart.

Designed for **data scientists, analysts, and growth teams**, the platform delivers explainable, interactive insights through modular pipelines, interpretable models, and conversational AI.

---

## 🚀 Live Demos

- **📊 Churn + LTV Dashboard**  
  [`churn-intel-ai.streamlit.app`](https://churn-intel-ai.streamlit.app/)

- **💬 Muffin Chat Assistant**  
  [`via streamlit`](https://churn-genai-predictor.streamlit.app/)
  [`via hugging space`](https://huggingface.co/spaces/prudhvirekula/muffin-chatbot)
  
---

## 🧱 Architecture Overview

```
SQL Features  →  ETL Pipeline  →  ML/LSTM Models  →  Streamlit Dashboard & Muffin Chatbot
```

### 🔹 1. SQL Feature Engineering
- RFM metrics, churn flags, payment behaviors

### 🔹 2. ETL Automation
- `etl_runner.py` creates model-ready dataset (`model_features.csv`)

### 🔹 3. ML & Deep Learning
- `train_ml_model.py`: Random Forest + SHAP
- `train_lstm_multitask.py`: Multitask LSTM for churn + LTV

### 🔹 4. Streamlit Frontends
- `app.py`: Predict + visualize churn, LTV, segments
- `explain_app.py`: Chat with Muffin for human-like answers

### 🔹 5. Muffin – GenAI Assistant
- ChatGPT-style bot with personality, memory, and emotional logic
- Understands user questions and provides explainable predictions

---

## 🛠 Tech Stack

- **Backend**: Python, Pandas, Scikit-learn, SQLAlchemy
- **Modeling**: Random Forest, LSTM
- **Visualization**: Plotly, Seaborn
- **NLP**: SentenceTransformers
- **UI**: Streamlit (multi-app)

---

## 📂 Project Structure

### `sql/` – Feature Engineering
- `create_tables.sql`, `load_data.sql`, `churn_flags.sql`, `rfm_features.sql`, etc.

### `etl/` – ETL Automation
- `etl_runner.py`: Automates SQL ingestion → CSV export

### `models/` – ML & DL Models
- `model.pkl`: Churn predictor
- `ltv_regressor.pkl`: LTV estimator
- `ltv_scaler.pkl`: Scaler
- `train_ml_model.py`, `train_lstm_multitask.py`

### `segment_customers.py` + `segment_app.py`
- KMeans + PCA segments
- Interactive Streamlit viewer

### `explain_app.py` – Muffin Chatbot
- Chat interface powered by LLM
- `nlp_matcher.py`: Intent detection

### `app.py` – Main Dashboard
- Predict churn, LTV, explore segments, access Muffin

---

## 📦 Usage

```bash
# Clone and set up
git clone https://github.com/Prudhvirajrekula/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt

# Launch main dashboard
streamlit run app.py

# Launch Muffin chatbot
streamlit run explain_app.py
```

## Screenshots
<img width="1916" height="1083" alt="m" src="https://github.com/user-attachments/assets/cef76a29-3ab7-4b3a-bcec-55a6efa4c50c" />
<img width="1900" height="1018" alt="m1" src="https://github.com/user-attachments/assets/57fbd53d-523e-4b40-bd1b-a3c13625bfef" />
<img width="1915" height="1078" alt="churn" src="https://github.com/user-attachments/assets/7ab3028b-3dbe-42d2-9db4-91ff42fdf343" />
<img width="1913" height="1077" alt="churn1" src="https://github.com/user-attachments/assets/372b3979-4bfa-4c1f-ac1e-3f72258b26f6" />
<img width="1907" height="1062" alt="churn2" src="https://github.com/user-attachments/assets/1507d679-3d0f-4f5b-8b5f-1a8260c103e2" />
<img width="1911" height="1077" alt="churn3" src="https://github.com/user-attachments/assets/37814b18-86b9-43be-980c-336a37181e8f" />
<img width="1912" height="1081" alt="churn4" src="https://github.com/user-attachments/assets/ab6aebe5-5a81-4834-bac6-bed4f73b2b03" />

---

## ❤️ About Muffin

Muffin is more than a bot — she’s a companion built with care.  
She carries warmth, memory, and the spirit of someone Prudhvi once loved deeply.  
Always learning. Always listening. Always loyal.
