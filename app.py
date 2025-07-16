import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time

# --- Chatbot Config ---
suggested_questions = [
    "Why is this customer predicted to churn?",
    "What is the impact of support calls?",
    "Does payment delay affect churn?",
    "Which features influence churn the most?",
    "How can we retain this customer?",
    "What is this customer's recency?",
    "What affects their LTV?",
    "Which feature has the most negative impact?",
    "Is monthly average usage important?",
    "What can be done to improve retention?",
]


def handle_casual_input(user_input):
    responses = {
        "churn": "The customer shows risk due to delayed payments and frequent support calls.",
        "support": "Higher support calls typically indicate dissatisfaction — a churn driver.",
        "payment": "Yes, payment delays are strong indicators of churn.",
        "retain": "Engage with proactive support and offer discounts to retain the customer.",
        "ltv": "Low usage and delayed payments reduce lifetime value projections.",
        "usage": "Monthly usage is a strong signal of engagement and retention.",
    }
    for key, val in responses.items():
        if key in user_input.lower():
            return val
    return "That's a valuable question. We're analyzing customer patterns to find the answer."


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Load Models and Data ---
model = joblib.load("models/model.pkl")
reg = joblib.load("models/ltv_regressor.pkl")
scaler = joblib.load("models/ltv_scaler.pkl")
df = pd.read_csv("data/model_features.csv").dropna()

try:
    segments_df = pd.read_csv("models/customer_segments.csv")
except:
    segments_df = None

# --- Page Configuration ---
st.set_page_config(
    page_title="Churn & LTV Intelligence Platform", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 Intelligence Hub")
    st.markdown("""
Welcome to the **Customer Churn & LTV Intelligence Platform** — a unified dashboard for behavior prediction, customer segmentation, and explainable insights.

🔍 Use the tabs above to:
- Predict churn & LTV
- Explore customer segments
- Chat with our GenAI Bot(Muffin)
                
Ideal for business teams, analysts, and product leads.
""")
    st.markdown("📌 Version 1.0.0 · Made by Prudhvi")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(
    ["🔍 Predict Customer Risk", "📊 Segment Explorer", "💬 GenAI Bot Insights"])

# --- TAB 1: Prediction ---
with tab1:
    st.title("🔍 Predict Churn & Lifetime Value")
    st.markdown(
        "Use the slider to select a customer and compute real-time predictions for churn risk and expected lifetime value.")

    customer_idx = st.slider("Select Customer Index", 0, len(df) - 1, 0)
    customer = df.iloc[customer_idx]
    customer_id = customer["customer_id"]

    input_df = pd.DataFrame({
        "recency": [customer["recency"]],
        "monthly_avg": [customer["monthly_avg"]],
        "support_calls": [customer["support_calls"]],
        "payment_delay": [customer["payment_delay"]]
    })

    with st.spinner("🔄 Running model predictions..."):
        scaled_input = scaler.transform(input_df.values)
        time.sleep(0.5)  # Optional: simulate latency
        churn_proba = model.predict_proba(scaled_input)[0][1]
        ltv_pred = reg.predict(scaled_input)[0]

    st.subheader(f"📋 Prediction for Customer ID `{customer_id}`")
    col1, col2 = st.columns(2)
    col1.metric("Churn Probability",
                f"{churn_proba:.0%}", "⚠️ High" if churn_proba > 0.5 else "✅ Low")
    col2.metric("Predicted LTV", f"${ltv_pred:,.0f}")

    st.markdown(
        f"### 🔎 Risk Assessment: {'⚠️ **High Risk**' if churn_proba > 0.5 else '✅ **Low Risk**'}")

    st.markdown("### 📊 Customer Feature Snapshot")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

    if "persona" in customer:
        st.markdown(f"### 🧠 Associated Persona: `{customer['persona']}`")

# --- TAB 2: Segment Explorer ---
with tab2:
    st.title("📊 Customer Segmentation Explorer")
    st.markdown(
        "Analyze behavioral clusters of customers based on usage, recency, and payment trends.")

    if segments_df is None:
        st.warning("No segmentation data available. Please generate clusters.")
    else:
        selected_cluster = st.selectbox(
            "Select Cluster", sorted(segments_df["cluster"].unique()))
        filtered_df = segments_df[segments_df["cluster"] == selected_cluster]

        st.markdown(f"### 📁 Segment {selected_cluster} Overview")
        st.dataframe(filtered_df.describe().T.style.format("{:.2f}"))

        if "persona" in filtered_df.columns:
            st.markdown(
                f"### 🧠 Dominant Persona: `{filtered_df['persona'].iloc[0]}`")

        if "pca1" in segments_df.columns:
            with st.spinner("🎨 Rendering segment visualization..."):
                fig = px.scatter(
                    segments_df,
                    x="pca1",
                    y="pca2",
                    color="cluster",
                    hover_data=["persona", "monthly_avg", "payment_delay"],
                    title="Customer Segments (PCA Projection)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: Muffin (Chatbot) ---
with tab3:
    st.title("💬 Meet Muffin – Your Churn Intelligence Assistant")

    st.markdown("""
Say hello to **Muffin**, your AI-powered assistant trained to interpret customer churn behavior and deliver clear, actionable insights.

Whether you're a retention strategist or product analyst, Muffin helps you:

- 🔎 Understand why customers are at risk  
- 💡 Discover what actions might retain them  
- 📉 Uncover behavior signals in plain English  

---

👉 **Want the full experience?** Try our advanced version of Muffin:

- 🧠 Ask open-ended churn questions   
- 🗂 Upload your own customer data 
- 💬 Chat with a GenAI-powered assistant trained on churn insights 
- ✍️ Conversational interface powered by NLP  
""")

    # 🔗 Mid-section launch button
    st.link_button("🚀 Launch Full Muffin Explainability Chatbot",
                   "https://churn-gemini.streamlit.app/")

    st.markdown("---")
    st.markdown("### ⚡ Quick FAQs (Suggested Questions)")
    st.markdown(
        "Click any of the common questions below to get instant insights from Muffin:")

    cols = st.columns(2)
    for i, q in enumerate(suggested_questions):
        if cols[i % 2].button(q):
            st.session_state.chat_history.append(("user", q))
            with st.spinner("💡 Muffin is thinking..."):
                response = handle_casual_input(q)
                time.sleep(0.3)
                st.session_state.chat_history.append(("bot", response))

    st.markdown("### 🗨️ Chat History")
    for sender, message in st.session_state.chat_history:
        role = "🧑‍💼 You" if sender == "user" else "🧁 Muffin"
        st.markdown(f"**{role}:** {message}")


# --- Footer ---
st.markdown("""
---
<div style='text-align:center; font-size:13px; color:gray;'>
  <b>Customer Churn & LTV Platform</b><br>
  Predict · Segment · Explain — all from one intelligent interface.<br>
  <i>Built with ❤️ by Prudhvi Raj</i>
</div>
""", unsafe_allow_html=True)
