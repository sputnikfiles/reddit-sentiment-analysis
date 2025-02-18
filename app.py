import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

def cleaning(message):
    if not isinstance(message, str):
        return []
    
    stop_words = set(stopwords.words('english'))
    
    cleaned_words = [word.lower() for word in message if word not in string.punctuation]
    cleaned_message = "".join(cleaned_words).split()
    
    filtered_words = [word for word in cleaned_message if word not in stop_words]
    
    return filtered_words

with open("naive_bayes_model.pkl", "rb") as model_file:
    nb_class = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    text_vectorized = vectorizer.transform([user_text])

    sentiment_label = nb_class.predict(text_vectorized)[0]

    sentiment_mapping = {1: "Positive", 0: "Negative"}
    sentiment = sentiment_mapping.get(sentiment_label, "Unknown")

    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
