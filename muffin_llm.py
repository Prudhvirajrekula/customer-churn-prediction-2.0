# muffin_llm.py
# If you get an import error, run: pip install google-generativeai
import requests
import streamlit as st
import os

# Load Gemini API key
GEMINI_API_KEY = st.secrets.get(
    "GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

MUFFIN_PERSONA_ANSWERS = [
    (lambda q: any(x in q for x in ["your name", "who are you", "what is your name", "what's your name", "assistant name", "are you muffin", "who is muffin", "muffin"]),
     "I'm Muffin — your AI-powered churn and retention assistant, built with love and a sprinkle of personality! 🧁 If you need insights, predictions, or just a friendly chat, Muffin's here for you."),
]


def call_gemini_llm(user_message, history=None, context=None):
    """
    Call Gemini 2.0 Flash REST API with optional chat history and context.
    If the user asks about Muffin or the assistant's name, always respond as Muffin.
    """
    q = user_message.lower()
    for matcher, answer in MUFFIN_PERSONA_ANSWERS:
        if matcher(q):
            return answer
    prompt = ""
    if context:
        prompt += f"Context:\n{context}\n\n"
    if history:
        for msg in history:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += f"User: {user_message}\nAssistant:"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(
            GEMINI_ENDPOINT + f"?key={GEMINI_API_KEY}",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"⚠️ Gemini LLM Error: {str(e)}"


# Example usage
if __name__ == "__main__":
    query = "What features most impact churn risk?"
    reply = call_gemini_llm(query)
    print("LLM Response:\n", reply)
