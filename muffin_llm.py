# muffin_llm.py
import requests
import os
import streamlit as st

# ✅ Load OpenRouter API key (Streamlit Cloud > Local fallback)
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")


API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "X-Title": "Muffin LLM Chatbot"
}


MODEL_NAME = "mistralai/mistral-7b-instruct:free"

def call_muffin_llm(user_message, history=None):
    """
    Call Mistral LLM via OpenRouter.
    Args:
        user_message (str): Input from user
        history (list): Optional chat history as list of {role:..., content:...}
    Returns:
        str: Response from LLM
    """
    messages = history or []
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"


# Example usage
if __name__ == "__main__":
    query = "What features most impact churn risk?"
    reply = call_muffin_llm(query)
    print("LLM Response:\n", reply)