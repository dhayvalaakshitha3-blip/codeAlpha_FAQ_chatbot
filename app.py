from flask import Flask, render_template, request
import json
import re

from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# Load FAQ data
with open("faq_data.json", "r", encoding="utf-8") as file:
    faq_data = json.load(file)


# Text preprocessing
def preprocess_text(text):
    text = text.lower()

    tokens = wordpunct_tokenize(text)

    cleaned_tokens = []

    for token in tokens:
        token = re.sub(r"[^a-z0-9]", "", token)

        if token:
            cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


# Prepare FAQ questions
faq_questions = [
    preprocess_text(item["question"])
    for item in faq_data
]


# Convert FAQ questions into TF-IDF vectors
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(faq_questions)


# Find the best answer
def get_best_answer(user_question):

    processed_question = preprocess_text(user_question)

    if not processed_question:
        return "Please enter a question."

    user_vector = vectorizer.transform([processed_question])

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    best_index = similarity_scores.argmax()

    best_score = similarity_scores[best_index]

    if best_score < 0.15:
        return "Sorry, I could not find a suitable answer for your question."

    return faq_data[best_index]["answer"]


# Home page
@app.route("/", methods=["GET", "POST"])
def home():

    answer = ""
    question = ""

    if request.method == "POST":

        question = request.form.get("question", "").strip()

        answer = get_best_answer(question)

    return render_template(
        "index.html",
        answer=answer,
        question=question
    )


# Start the application
if __name__ == "__main__":
    app.run(debug=True)
