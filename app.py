# SPAM EMAIL CLASSIFIER - Flask Web Application
# Description: Serves a web UI where users type a message and get a Spam / Ham prediction with confidence score.

import os
import re
import pickle
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# NLTK data (download silently if missing)
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)

# Initialise Flask app 
app = Flask(__name__)

# Load saved model & vectorizer
# These were created by train_model.py.  We load them ONCE at startup so
# every prediction request is fast (no re-loading on each request).
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

try:
    with open(os.path.join(MODEL_DIR, 'spam_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    print("✅ Model and vectorizer loaded successfully.")
except FileNotFoundError:
    raise RuntimeError(
        "❌ Model files not found.\n"
        "   Please run  python train_model.py  first."
    )

# Text preprocessing (must match train_model.py exactly)
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """Lowercase → strip non-alpha → remove stopwords → stem."""
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

# ROUTES

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive a JSON body { "message": "..." }, classify it, and return JSON:
    {
        "prediction"  : "Spam" | "Ham",
        "confidence"  : <float 0–100>,
        "spam_prob"   : <float 0–1>,
        "ham_prob"    : <float 0–1>,
        "clean_text"  : <preprocessed string>
    }
    """
    data    = request.get_json(silent=True)
    message = (data or {}).get('message', '').strip()

    # Input validation 
    if not message:
        return jsonify({'error': 'Please enter a message.'}), 400

    if len(message) < 3:
        return jsonify({'error': 'Message too short to classify.'}), 400

    # Preprocess
    clean = preprocess_text(message)

    if not clean:
        return jsonify({'error': 'Message contained no meaningful words.'}), 400

    # Vectorise 
    features = vectorizer.transform([clean])

    # Predict
    pred_label = model.predict(features)[0]          # 0 = ham, 1 = spam
    proba      = model.predict_proba(features)[0]    # [P(ham), P(spam)]

    ham_prob  = float(proba[0])
    spam_prob = float(proba[1])

    prediction = 'Spam' if pred_label == 1 else 'Ham'
    confidence = spam_prob if pred_label == 1 else ham_prob

    return jsonify({
        'prediction' : prediction,
        'confidence' : round(confidence * 100, 2),
        'spam_prob'  : round(spam_prob * 100, 2),
        'ham_prob'   : round(ham_prob  * 100, 2),
        'clean_text' : clean,
    })

@app.route('/health')
def health():
    """Simple health-check endpoint (useful for deployment platforms)."""
    return jsonify({'status': 'ok', 'model': 'MultinomialNB', 'vectorizer': 'TF-IDF'})


# ENTRY POINT

if __name__ == '__main__':
    print("\n🚀 Starting Spam Classifier Flask App...")
    print("   Open  http://127.0.0.1:5000  in your browser.\n")
    app.run(debug=True, host='0.0.0.0', port=5000)