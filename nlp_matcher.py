# nlp_matcher.py

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Define your suggested questions
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

# Load the transformer model and embed the questions
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
suggested_embeddings = embed_model.encode(suggested_questions)

# Function to match user query to closest suggestion
def match_question_nlp(user_input, suggested_questions):
    user_vec = embed_model.encode([user_input])
    similarities = cosine_similarity(user_vec, suggested_embeddings)[0]
    best_idx = np.argmax(similarities)
    return suggested_questions[best_idx], similarities[best_idx]

# Expose both questions and function
__all__ = ["suggested_questions", "match_question_nlp"]
