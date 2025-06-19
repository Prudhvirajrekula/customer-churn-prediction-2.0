import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import datetime
import markdown
from html import escape
from nlp_matcher import match_question_nlp

import streamlit.components.v1 as components


# Load data and models
df = pd.read_csv("data/model_features.csv").dropna().head(100)
model = joblib.load("models/model.pkl")
reg = joblib.load("models/ltv_regressor.pkl")
scaler = joblib.load("models/ltv_scaler.pkl")

# Fast casual handler
def handle_casual_input(user_input):
    casual = user_input.lower().strip()

    greetings = ["hello", "hi", "hey"]
    goodbyes = ["bye", "goodbye", "see you"]
    thanks = ["thanks", "thank you", "ok", "okay", "cool"]

    if casual in greetings:
        return random.choice(["Hey there! 👋", "Hi! How can I assist you with churn or LTV?", "Hello! 😊"])
    elif casual in goodbyes:
        return random.choice(["Goodbye! 👋", "Take care!", "Hope to see you again!"])
    elif casual in thanks:
        return random.choice(["You're welcome!", "Glad to help!", "No problem at all!"])
    
    return None


def generate_summary(customer, churn_label, churn_proba, ltv):
    summary = f"""
This customer is predicted to be **{churn_label.upper()}** with a confidence of `{churn_proba:.2f}`.

📊 **Key Highlights**:
- Recency: {customer['recency']} days since last activity
- Monthly Usage: {customer['monthly_avg']}
- Support Calls: {customer['support_calls']}
- Payment Delay: {customer['payment_delay']} days
- Predicted LTV: ${ltv:,.2f}

💡 **Summary**: The customer shows{' signs of churn' if churn_label == 'churned' else ' stable engagement'}, with {"high" if customer['support_calls'] > 5 else "moderate"} support usage and {"significant" if customer['payment_delay'] > 10 else "minor"} payment delays.
"""
    return summary

# ML response generator
def generate_response(user_input, customer_idx):
    customer = df.iloc[customer_idx]
    raw = customer[["recency", "monthly_avg", "support_calls", "payment_delay"]].values.reshape(1, -1)
    scaled = scaler.transform(raw)
    churn_proba = model.predict_proba(scaled)[0][1]
    churn_label = "churned" if churn_proba > 0.5 else "not churned"
    ltv = reg.predict(scaled)[0]

    q = user_input.lower().strip()

    # Smalltalk & Identity
    if "your name" in q or "who are you" in q:
        return """🧁 **I'm Muffin!**  
    I  help explain why customers might churn and estimate their LTV using machine learning insights."""

    if "who created" in q or "your creator" in q:
        return """👨‍💻 I was created by **Prudhvi Raj**, using Python, Streamlit, and machine learning to interpret customer churn and lifetime value."""
    
    if "what do you do" in q or "your purpose" in q:
        return """📊 I analyze customer behavior and help explain:
- Why they might churn
- What affects their LTV
- How to improve retention"""

    # Manually handle known intents
    if "why" in q and "churn" in q:
        percentage = int(churn_proba * 100)
        filled = int(churn_proba * 20)
        bar = "█" * filled + "░" * (20 - filled)
        return f"""**🔍 Prediction Insight**

This customer is predicted to be **{churn_label.upper()}**  
📊 **Confidence Score:** `{churn_proba:.2f}`

**Possible contributing factors:**
- 📞 High number of support calls
- 💸 Payment delays

**🔦 Churn Risk Confidence:**  
<div style='font-family:monospace; color:#00e676; background:#111; padding:6px 10px; border-radius:6px; display:inline-block; margin-top:6px;'>
{bar} {churn_proba:.2f}
</div>
"""

    if "ltv" in q:
        return f"""**📈 Lifetime Value Estimate**

The projected **Lifetime Value (LTV)** for this customer is **${ltv:,.2f}**.

**Why?**
- Based on monthly usage: **{customer['monthly_avg']}**
- Adjusted for payment behavior and recency
"""

    if "support" in q:
        return f"""**📞 Support Calls**

This customer made **{int(customer['support_calls'])} support calls**.

Frequent support interactions often indicate friction or dissatisfaction, which can increase churn risk.
"""

    if "payment delay" in q:
        return f"""**💸 Payment Delay**

This customer has delayed payments by **{int(customer['payment_delay'])} days**.

**Insight:** Delayed payments are a strong churn signal — customers struggling to pay often disengage.
"""

    if "retain" in q or "retention" in q:
        return """**💡 Retention Strategy**

To retain this customer, consider:
- Offering proactive support or check-ins
- Providing loyalty rewards for prompt payments
- Tailoring messages based on usage behavior
"""

    if "usage" in q or "monthly_avg" in q:
        return f"""**📊 Monthly Usage**

This customer’s average monthly usage is **{customer['monthly_avg']}**.

**Insight:** Lower usage often correlates with disengagement — keep them engaged with value-driven features.
"""

    if "recency" in q:
        return f"""**⏱️ Recency**

The customer last interacted **{customer['recency']} days ago**.

**Insight:** Customers who haven't engaged recently are at higher churn risk. Consider re-engagement campaigns.
"""
    
    if "risky" in q or "risk" in q:
        percentage = int(churn_proba * 100)
        filled = int(churn_proba * 20)
        bar = "█" * filled + "░" * (20 - filled)

        return f"""
**⚠️ Customer Risk Score**

This customer has a **churn risk of `{churn_proba:.2f}`**, meaning there's a {percentage}% chance they may churn soon.

**Key drivers:**
- 📞 Support Calls: **{customer['support_calls']}**
- 💸 Payment Delays: **{customer['payment_delay']} days**

**Visual Risk Bar:**

`{bar} {churn_proba:.2f}`
"""


    if "most" in q or "important" in q or "influence" in q:
        return """**🔥 Top Churn Predictors**

- **Support Calls**: High contact volumes indicate dissatisfaction  
- **Payment Delay**: Late payments suggest disengagement  
- **Recency**: Less recent activity increases churn probability
"""

    if "negative" in q:
        return """**⚠️ Most Harmful Indicators**

Features with strongest negative impact:
- **Excessive support calls**
- **Long payment delays**
- **Low usage frequency**
"""

    # ✅ NLP fallback when nothing matches
    matched = match_question_nlp(q, suggested_questions)
    if matched:
        matched_text = matched[0]
        return generate_response(matched_text, customer_idx)


    # Default fallback
    return """I'm here to help explain churn or LTV insights.  
You can ask about:
- 🔍 Prediction details
- 📈 Lifetime Value
- ⚙️ Feature impact
- 💡 Retention strategies
"""

# Suggested questions
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


# Set layout
st.set_page_config(page_title="Muffin Chatbot", layout="wide")
st.markdown("<h1 style='text-align:center;'>🧠 Churn & LTV Chatbot</h1>", unsafe_allow_html=True)

# Layout columns
sidebar, main = st.columns([1, 2], gap="large")



# LEFT SIDEBAR = Customer Panel
with sidebar:
    st.header("📋 Customer Info")
    idx = st.slider(
        "Select Customer Index",
        0,
        len(df) - 1,
        0,
        help="Pick a customer row from the dataset to explore their churn prediction."
    )
    customer = df.iloc[idx]

    raw = customer[["recency", "monthly_avg", "support_calls", "payment_delay"]].values.reshape(1, -1)
    scaled = scaler.transform(raw)
    churn_proba = model.predict_proba(scaled)[0][1]
    churn_label = "churned" if churn_proba > 0.5 else "not churned"
    ltv = reg.predict(scaled)[0]

    st.markdown(f"🧠 **Prediction:** `{churn_label.upper()}`  \n📊 **Confidence:** `{churn_proba:.2f}`", help="Churn prediction is based on key behavioral patterns.")
    st.markdown(f"💰 **Predicted LTV:** `${ltv:,.2f}`", help="Estimated customer lifetime value based on usage and payment trends.")

    st.markdown("### 🔍 Customer Features")
    st.markdown("""
    <ul style="color: #aaa; font-size: 13px; line-height: 1.6;">
        <li><b>Customer ID</b>: Unique identifier</li>
        <li><b>Recency</b>: Days since last interaction</li>
        <li><b>Monthly Avg</b>: Avg usage per month</li>
        <li><b>Support Calls</b>: Number of times customer contacted support</li>
        <li><b>Payment Delay</b>: Days past due on recent payments</li>
    </ul>
    """, unsafe_allow_html=True)

    st.dataframe(customer.to_frame(), use_container_width=True)


# RIGHT MAIN CHAT
with main:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "chat_input_text" not in st.session_state:
        st.session_state.chat_input_text = ""
    if "input_submitted" not in st.session_state:
        st.session_state.input_submitted = False

    st.markdown("### 💬 Suggested Questions")
    rows = [suggested_questions[i:i+5] for i in range(0, len(suggested_questions), 5)]
    for row in rows:
        cols = st.columns(len(row))
        for i, q in enumerate(row):
            with cols[i]:
                if st.button(q, key=f"suggest_{q}"):
                    st.session_state.chat.append({
                        "role": "user",
                        "text": q,
                        "time": datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
                    })
                    reply = generate_response(q, idx)
                    st.session_state.chat.append({
                        "role": "bot",
                        "text": reply,
                        "time": datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
                    })

    # Styling for layout
    st.markdown("""
    <style>
    .chat-bubble-user, .chat-bubble-bot {
        max-width: 80%;
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 10px;
        font-size: 15px;
        display: flex;
        flex-direction: column;
    }
    .chat-bubble-user {
        background-color: #2c2f33;
        color: #e0e0e0;
        border-left: 6px solid #607d8b;
        margin-left: auto;
        text-align: right;
    }
    .chat-bubble-bot {
        background-color: #1e2227;
        color: #e0e0e0;
        border-left: 6px solid #009688;
        margin-right: auto;
        text-align: left;
    }
    .chat-time {
        font-size: 10px;
        color: #aaa;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Input handling
    if not st.session_state.input_submitted:
        user_input = st.text_input("💬 Ask your question", key="chat_input_text", placeholder="Type your question and hit Enter...")
        if user_input:
            st.session_state.input_submitted = True
            st.rerun()
    else:
        user_input = st.session_state.chat_input_text
        st.session_state.chat.append({
            "role": "user",
            "text": user_input,
            "time": datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
        })
        # Simulate typing animation
        placeholder = st.empty()
        placeholder.markdown("🧁 *Muffin is typing...*", unsafe_allow_html=True)
        import time; time.sleep(1.2)  # delay to simulate typing

        # Generate and display final response
        reply = handle_casual_input(user_input) or generate_response(user_input, idx)
        placeholder.empty()  # remove "typing..."

        st.session_state.chat.append({
            "role": "bot",
            "text": reply,
            "time": datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
        })

        st.markdown("""
        <script>
            document.getElementById("muffin-footer").scrollIntoView({ behavior: "smooth" });
        </script>
        """, unsafe_allow_html=True)


        st.session_state.chat_input_text = ""
        st.session_state.input_submitted = False
        st.rerun()

    # Clear history
    st.markdown("### 🗨️ Chat Panel")

    # Side-by-side buttons with tighter spacing
    col1, col2 = st.columns([1, 1], gap="small")

    with col1:
        st.button("🧹 Clear Chat History", key="clear_chat")

    with col2:
        if st.button("⬇️ Scroll to Latest", key="scroll_latest"):
            components.html("""
                <script>
                    const footer = window.parent.document.getElementById("muffin-footer");
                    if (footer) {
                        footer.scrollIntoView({ behavior: "smooth" });
                    }
                </script>
            """, height=0)




    # Chat Display (latest at top)
    st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)

    # Iterate through chat in steps of 2 (user + bot)
    for i in range(0, len(st.session_state.chat), 2):
        user_msg = st.session_state.chat[i]
        bot_msg = st.session_state.chat[i + 1] if i + 1 < len(st.session_state.chat) else None

        # User bubble
        st.markdown(
            f"""
            <div class='chat-bubble-user'>
                <div><b>🧍 You:</b></div>
                <div>{user_msg['text']}</div>
                <div class='chat-time'>{user_msg['time']}</div>
            </div>
            """, unsafe_allow_html=True
        )

        # Bot bubble with markdown support
        if bot_msg:
            # Convert markdown message to raw HTML
            html_body = markdown.markdown(bot_msg['text'])

            # Render everything in one HTML block
            st.markdown(
                f"""
                <div class='chat-bubble-bot'>
                <div><b>🧁 Muffin:</b></div>
                    <div>{html_body}</div>
                    <div class='chat-time'>{escape(bot_msg['time'])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
st.markdown("</div>", unsafe_allow_html=True)



# 👇 Footer + Auto-scroll anchor element
st.markdown("""
<div id="muffin-footer"></div>
<hr style="margin-top:2rem; margin-bottom:1rem;">
<div style='text-align:center; font-size:13px; color:#888;'>
    🧁 <b>Muffin</b> is an AI-powered assistant designed to help you interpret churn and LTV predictions.<br>
    While it uses predictive models and rule-based logic, it's still learning and may not always reflect the full business context.<br><br>
    <i>Responses are based on available data as of now and do not include real-time updates.</i><br>
    Created with ❤️ by <b>Prudhvi Raj</b>.
</div>
""", unsafe_allow_html=True)

