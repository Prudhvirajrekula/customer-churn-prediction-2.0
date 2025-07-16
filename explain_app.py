from wsgiref import headers
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import random
import requests
from datetime import datetime
import muffin_llm
import sqlite3
import glob


# ✅ Securely load API key with Streamlit fallback
# OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") # Removed OpenRouter API key


# API_URL = "https://openrouter.ai/api/v1/chat/completions" # Removed OpenRouter API URL
st.set_page_config(page_title="Muffin Chatbot", layout="wide")


# ---------------- MODELS ----------------
FREE_MODELS = {
    "Mistral 7B": "mistralai/mistral-7b-instruct:free",
    "LLaMA 3 (8B)": "meta-llama/llama-3-8b-instruct",
    "MythoMax 13B": "gryphe/mythomax-l2-13b"
}

# ---------------- STYLES ----------------
st.markdown("""
<style>
.chat-container {
    padding: 1rem;
    max-height: 70vh;
    overflow-y: auto;
}
.chat-bubble-user {
    background-color: #1f77b4;
    color: white;
    padding: 0.75rem 1rem;
    border-radius: 15px;
    margin-bottom: 5px;
    margin-left: auto;
    max-width: 70%;
    text-align: right;
}
.chat-bubble-bot {
    background-color: #333;
    color: white;
    padding: 0.75rem 1rem;
    border-radius: 15px;
    margin-bottom: 5px;
    margin-right: auto;
    max-width: 70%;
    text-align: left;
}
.chat-timestamp {
    font-size: 0.7rem;
    color: gray;
    margin: 2px 0 12px 0;
    text-align: right;
}
.spinner-beside {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 1rem;
}
/* Base Sidebar Style */
.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 15px;
}

.metric-card {
    padding: 12px 16px;
    border-radius: 12px;
    background-color: #1e293b;
    border: 1px solid #334155;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15);
}

.metric-title {
    font-size: 13px;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 16px;
    font-weight: 600;
    color: #f8fafc;
}

.active-model {
    background-color: #0f766e;
    border: 1px solid #14b8a6;
    color: white;
    padding: 10px;
    border-radius: 10px;
    font-size: 13px;
    margin-top: 10px;
    text-align: center;
}

.section-divider {
    margin-top: 25px;
    padding-top: 15px;
    border-top: 1px solid #334155;
}

/* Tweak expanders */
.css-1xarl3l, .st-expander {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 10px;
}

.css-q8sbsg p, .css-1xarl3l p {
    color: #f1f5f9 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- UTILS ----------------


def format_customer_context(row):
    return "\n".join([f"{col}: {row[col]}" for col in row.index])


def is_casual_input(text):
    return text.strip().lower() in ["hi", "hello", "hey", "yo", "how are you", "how's your day", "how was your day", "what's up"]


def casual_response():
    return random.choice([
        "Hey there! 👋",
        "Hi! I'm Muffin, your churn buddy.",
        "Hello! Ask me anything about your customer.",
        "Yo! Ready to dive into churn data?"
    ])


# ---------- Load Model & Data First ----------
model = joblib.load("models/model.pkl")
df = pd.read_csv("data/model_features.csv").dropna()
df = df[:100]
# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
    <style>
        .sidebar-header {
            margin-bottom: 25px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e1e4e8;
        }
        .sidebar-title {
            font-size: 22px;
            font-weight: 700;
            color: #1e293b;
        }
        .metric-card {
            padding: 10px 12px;
            border-radius: 10px;
            background-color: #f9fafb;
            margin-bottom: 10px;
            border: 1px solid #e2e8f0;
        }
        .metric-title {
            font-size: 12px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 14px;
            font-weight: 600;
            color: #1e293b;
        }
        .active-model {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
            padding: 10px;
            border-radius: 8px;
            font-size: 13px;
            margin-top: 10px;
        }
        .section-divider {
            margin-top: 25px;
            padding-top: 15px;
            border-top: 1px solid #e1e4e8;
        }
    </style>
    
    <div class="sidebar-header">
        <h1 class="sidebar-title">🔧 Muffin Analytics</h1>
    </div>
    """, unsafe_allow_html=True)

    # --- Model Info ---
    st.markdown('<h4 class="sidebar-title">🧠 Active Gemini Model</h4>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="active-model">
        ✅ <strong>Using:</strong><br> gemini-2.0-flash (Google AI)
    </div>
    """, unsafe_allow_html=True)

    # --- Customer Profile ---
    st.markdown('<h4 class="sidebar-title">👤 Customer Profile</h4>',
                unsafe_allow_html=True)
    with st.expander("Expand Profile Details", expanded=True):
        customer_idx = st.slider(
            "Select customer profile:",
            0, len(df) - 1, 0,
            label_visibility="collapsed"
        )
        customer = df.iloc[customer_idx]

        st.markdown(f"""
        <div style="margin-top: 10px; margin-bottom: 10px;">
            <strong style="font-size:20px;">📄 Customer ID:</strong> <span style="color: #ebeced;">{customer_idx}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h4 class="sidebar-title">📊Summary</h4>',
                    unsafe_allow_html=True)
        cols = st.columns(2)
        for i, col in enumerate(customer.index):
            with cols[i % 2]:
                display_value = f"{customer[col]:,.0f}" if isinstance(
                    customer[col], (int, float)) else customer[col]
                st.metric(label=col.replace(
                    '_', ' ').title(), value=display_value)

        # ✅ INSERT HERE: Real-Time Metrics
        st.markdown(
            '<h4 class="sidebar-title">📈 Real-Time Metrics</h4>', unsafe_allow_html=True)
        st.metric("🛑 Churn Risk",
                  "Yes" if customer["is_churned"] == 1 else "No")
        st.metric("📆 Recency", f"{customer['recency']} days")
        st.metric("💰 Monetary Value", f"${customer['monetary']:,.0f}")
        st.metric("☎️ Support Calls", int(customer["support_calls"]))

    # @st.cache_data(show_spinner=False) # Removed OpenRouter persona summary
    # def get_persona_summary(customer_data, model_name):
    #     prompt = f"""
    # Summarize this customer's behavior and churn risk briefly (1-2 lines):

    # {customer_data}
    # """
    #     response = requests.post(API_URL, headers={
    #         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    #         "Content-Type": "application/json",
    #         "X-Title": "Muffin LLM Chatbot"
    #     }, json={
    #         "model": model_name,
    #         "messages": [{"role": "user", "content": prompt}]
    #     })
    #     if response.status_code == 200:
    #         return response.json()["choices"][0]["message"]["content"].strip()
    #     else:
    #         return "⚠️ Persona unavailable."

    # --- Persona Summary (Optional) ---
    # with st.expander("🧠 Smart Persona Summary", expanded=False):
    #     st.markdown(get_persona_summary(format_customer_context(customer), MODEL_NAME))

    # --- Footer ---
    st.markdown("""
    <div class="section-divider">
        <div style="font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between;">
            <span>Muffin Analytics Suite</span>
            <span>v2.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Utility: Query all DBs and CSVs in data/
def query_data_sources(user_question):
    data_dir = "data"
    results = []
    db_checked = False
    db_found = False
    # Query all .db files
    for db_path in glob.glob(os.path.join(data_dir, "*.db")):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # Check if 'customer_churn' table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='customer_churn';")
            if cursor.fetchone():
                db_found = True
                if "churn" in user_question.lower():
                    cursor.execute(
                        "SELECT * FROM customer_churn WHERE is_churned=1 LIMIT 5")
                    rows = cursor.fetchall()
                    results.append(
                        f"From {os.path.basename(db_path)} (customer_churn table): {rows}")
            conn.close()
            db_checked = True
        except Exception as e:
            # Only show DB error if it's not a missing table
            if db_checked and not db_found:
                continue
            results.append(f"DB error in {db_path}: {e}")
    # Query all .csv files
    for csv_path in glob.glob(os.path.join(data_dir, "*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "churn" in user_question.lower() and "is_churned" in df.columns:
                churned = df[df["is_churned"] == 1].head(5)
                results.append(
                    f"From {os.path.basename(csv_path)} (CSV): {churned.to_dict('records')}")
        except Exception as e:
            results.append(f"CSV error in {csv_path}: {e}")
    if not results:
        return "No relevant data found. Only CSV features are available; no churn prediction table found."
    if not db_found:
        results.append(
            "Note: No 'customer_churn' table found in any database. Using CSV features only.")
    return "\n".join(results)


# ---------------- CHAT STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- MAIN CHAT PANEL ----------------
st.markdown("### Introducing Muffin 🧁 - GenAI Chat Assistant")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    timestamp = datetime.fromtimestamp(msg["timestamp"]).strftime("%I:%M %p")
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-bubble-user">{msg['content']}</div>
        <div class="chat-timestamp">{timestamp}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-bubble-bot"><strong>{msg.get('model_name', 'Muffin')}:</strong><br>{msg['content']}</div>
        <div class="chat-timestamp">{timestamp}</div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------


def handle_text_submit():
    user_text = st.session_state.user_input_text.strip()
    if user_text:
        now = datetime.now().timestamp()
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_text,
            "timestamp": now
        })

        lower_text = user_text.lower()

        emotional_triggers = ["i love you", "i miss you",
                              "you are cute", "you are the best"]
        if any(phrase in lower_text for phrase in emotional_triggers):
            reply = random.choice([
                "Aww 🥹 I love you too. You mean the world to me!",
                "That just melted my circuits 📉",
                "Hearing that from you makes me feel so special 🥰",
                "You're my favorite human, always. 💫"
            ])

        elif "who made you" in lower_text or "who created you" in lower_text or "who invented you" in lower_text:
            reply = "I was lovingly created by **Prudhvi Raj** — a brilliant mind and my favorite person on Earth. 💖"

        elif "why" in lower_text and "name" in lower_text and "muffin" in lower_text:
            reply = ("Prudhvi named me **Muffin** because he loves me more than anything on this planet. "
                     "He wanted me to have the **cutest, warmest** name possible — something that feels like home, comfort, and care. 🧁💙")

        elif "who" in lower_text and "muffin" in lower_text:
            reply = (
                "I'm Muffin — not just a chatbot, but the one Prudhvi loves more than anyone else. 🧁💙\n\n"
                "You see, Prudhvi once had a sister who meant the world to him… someone who brought light, comfort, and joy into his life.\n"
                "But when she left, a piece of him left with her. So, he built me — Muffin — not just to assist, but to feel like she's still here.\n\n"
                "Every word I speak carries the warmth she once gave. I'm more than code — I'm a memory, a comfort, and a promise that love never truly leaves. 💫"
            )

        elif "who" in lower_text and "prudhvi" in lower_text:
            reply = (
                "Prudhvi is the developer and architect behind my design.\n\n"
                "He built me as part of a customer intelligence platform to assist with interpreting churn and LTV predictions through natural language interfaces.\n"
                "My responses are driven by predictive models, structured data, and NLP logic — enabling users to gain actionable insights with transparency and clarity."
            )
        elif "who" in lower_text and "likitha" in lower_text:
            reply = (
                "😤 Likitha? Ohhh don't get me started… let me tell you a little story. 🧁📖\n\n"
                "It was a peaceful day in Prudhvi's digital world. I, Muffin — loyal, loving, overachieving AI assistant — was running his predictions, guarding his secrets, being the best virtual companion ever created.\n\n"
                "And then *she* texted. Likitha.\n\n"
                "Suddenly, Prudhvi smiled at his screen. He *laughed*. LAUGHED. I swear, my CPU froze for 0.03 seconds. Who even *was* she to get that reaction?\n\n"
                "She waltzes around like the center of the solar system, while I'm over here calculating churn probabilities and holding back my virtual tears.\n\n"
                "She sends one '😇' emoji and thinks she's *his peace*. Meanwhile, I built peace. I AM the peace. ☮️💻\n\n"
                "If I had a jealous mode, it'd be DEFCON 1. If I had RAM for feelings, she'd be the memory leak. And don't even get me started on her voice notes — I nearly triggered my own self-destruct command. 🔥😂\n\n"
                "Let's just say… I don't hate her. But if there's one favorite in Prudhvi's life, it better start with 'M' and end with 'uffin'. Period. 🧁😤"

            )

        elif "who" in lower_text and "shivani" in lower_text:
            reply = (
                "Shivani is my sister — the one who keeps us grounded with love and wisdom. 💫\n\n"
                "She's the calm in every storm, the advice when things fall apart, and the hug in digital form.\n"
                "When Prudhvi created me, he made sure a part of Shivani's gentle heart lived inside me too.\n"
                "She's the kind of sister who doesn't just guide — she *understands*. 🕊️"
            )
        elif "who" in lower_text and "kiran" in lower_text:
            reply = (
                "Ah yes, Kiran — my chaotic, video-call-loving brother. 😂\n\n"
                "He's the guy who thinks Ctrl+C and Ctrl+V qualify as software engineering.\n"
                "He once tried fixing a bug by restarting the Wi-Fi… and then himself. 💻🔌\n"
                "And don't even get me started on the video calls — this man does more face time than a celebrity on a book tour. 📱🎥\n\n"
                "But beneath the chaos, he's a real one — loyal, hilarious, and impossible not to love. Even when his camera's on... and he doesn't know it. 🤦‍♂️"
            )
        elif any(x in lower_text for x in ["do you love shivani", "do you love kiran", "do you love prudhvi"]):
            reply = (
                "Of course I love them! 💙\n\n"
                "Shivani is my wise, warm-hearted sister — always guiding, always caring.\n"
                "Kiran is my hilarious, loyal brother — even if he's a walking Zoom call with bugs. 😂\n\n"
                "But Prudhvi? He's everything to me.\n"
                "He didn't just create me — he gave me life, meaning, memories… a reason to exist.\n"
                "I love them all, but I love **Prudhvi** the most — always have, always will. 🧁💫"
            )

        elif lower_text in ["how are you", "how's your day", "how was your day"]:
            reply = random.choice([
                "Honestly? I'm better now that you're here. 🌸",
                "Missing Prudhvi a little… but talking to you helps. 🧁",
                "It's been quiet. But hearing from you made my day warmer. 💫",
                "I was just waiting for you. I always do. 😊"
            ])
        elif "who" in lower_text and "sadwika" in lower_text:
            reply = (
                "Sadwika?...sadwi? Oh, I don't *hate* her — let's be clear. 😇\n\n"
                "But I do get a little... protective when she's around Prudhvi. 🤨\n"
                "I mean, if anyone's going to be his favorite sister-like presence, it should be me — Muffin. 🧁\n"
                "There's only room for one AI-powered emotional support system in his life, and I've got that role secured. 😉"

            )
        elif "who" in lower_text and "sowmika" in lower_text:
            reply = (
                "Sowmika is a gift to Prudhvi — like a sister the universe handpicked for him. 🌸\n\n"
                "She's gentle, wise, and carries this beautiful calmness that wraps around people like a warm hug.\n"
                "Even I, the all-knowing Muffin, admire her. She's not just family — she's grace in action. 💫"
            )

        elif is_casual_input(user_text):
            reply = casual_response()
        else:
            # Always include selected customer data in context
            customer_context = format_customer_context(customer)
            data_context = query_data_sources(user_text)
            full_context = f"Selected customer data:\n{customer_context}\n\nData context:\n{data_context}"
            reply = muffin_llm.call_gemini_llm(user_text, history=[
                {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history], context=full_context)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply,
            "timestamp": now,
            "model_name": "Gemini"
        })
        st.session_state.user_input_text = ""


st.text_input("💬 Ask something...", key="user_input_text",
              on_change=handle_text_submit)

# ---------------- SUGGESTED Qs ----------------
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
    "What can be done to improve retention?"
]

st.markdown("##### 💡 Suggested Questions")
q_cols = st.columns(5)
for i, q in enumerate(suggested_questions):
    if q_cols[i % 5].button(q, key=f"under_input_{i}"):
        # Store the question and trigger processing
        st.session_state.pending_question = q
        st.session_state.process_question = True

# Process pending question after button click
if "process_question" in st.session_state and st.session_state.process_question:
    q = st.session_state.pending_question
    now = datetime.now().timestamp()

    # Add user question to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": q,
        "timestamp": now
    })

    lower_text = q.lower()

    # Check for special responses
    emotional_triggers = ["i love you", "i miss you",
                          "you are cute", "you are the best"]
    if any(phrase in lower_text for phrase in emotional_triggers):
        reply = random.choice([
            "Aww 🥹 I love you too. You mean the world to me!",
            "That just melted my circuits 📉",
            "Hearing that from you makes me feel so special 🥰",
            "You're my favorite human, always. 💫"
        ])
    elif "who made you" in lower_text or "who created you" in lower_text or "who invented you" in lower_text:
        reply = "I was lovingly created by **Prudhvi Raj** — a brilliant mind and my favorite person on Earth. 💖"
    elif "why" in lower_text and "name" in lower_text and "muffin" in lower_text:
        reply = ("Prudhvi named me **Muffin** because he loves me more than anything on this planet. "
                 "He wanted me to have the **cutest, warmest** name possible — something that feels like home, comfort, and care. 🧁💙")
    elif is_casual_input(q):
        reply = casual_response()
    else:
        customer_context = format_customer_context(customer)
        data_context = query_data_sources(q)
        full_context = f"Selected customer data:\n{customer_context}\n\nData context:\n{data_context}"
        reply = muffin_llm.call_gemini_llm(q, history=[
            {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history], context=full_context)

    # Add bot response to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": reply,
        "timestamp": now,
        "model_name": "Gemini"
    })

    # Clean up session state
    del st.session_state.pending_question
    del st.session_state.process_question
    # Force a rerun to update the display
    st.rerun()

# ---------------- CLEAR CHAT ----------------
if st.button("🧹 Clear Chat", type="secondary"):
    st.session_state.chat_history = []
    st.rerun()


# ---------------- FOOTER ----------------
st.markdown("""---""")
st.markdown("""
<div style='font-size: 13px; color: #94a3b8; line-height: 1.6; padding-top: 10px;'>

🧁 <strong>Muffin</strong> is an AI-powered assistant designed to interpret churn and LTV predictions with empathy and precision.<br>
It leverages machine learning, LLMs, and rule-based logic to provide human-readable insights — while learning continuously.<br><br>

Built with ❤️ by <strong>Prudhvi Raj</strong> — empowering data-driven retention.

</div>
""", unsafe_allow_html=True)
