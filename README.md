# 🛡️ Spam Email / SMS Classifier

> An end-to-end Machine Learning web application that classifies messages as **Spam** or **Ham (Not Spam)** using TF-IDF vectorisation and Multinomial Naive Bayes, deployed via a Flask web interface with real-time confidence scores.

<br>

## 📌 Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [How It Works](#how-it-works)
5. [Setup & Installation](#setup--installation)
6. [Running the Project](#running-the-project)
7. [Model Performance](#model-performance)
8. [API Endpoints](#api-endpoints)
9. [Visualisations](#visualisations)
10. [Future Improvements](#future-improvements)

<br>

---

## Overview

This project builds a complete **Spam Detection pipeline** from scratch:

- Loads and cleans the **UCI SMS Spam Collection** dataset (5,572 messages)
- Preprocesses text through a 5-step NLP pipeline (lowercasing, cleaning, stopword removal, short-token filtering, stemming)
- Converts text to numerical features using **TF-IDF** (5,000 features + bigrams)
- Trains a **Multinomial Naive Bayes** classifier achieving ~97.58% accuracy
- Saves model artefacts with Python's built-in **Pickle** module for fast inference
- Serves real-time predictions through a **Flask REST API** with a dark-mode web UI

<br>

---

## Tech Stack

| Layer | Library / Tool | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core development |
| Data handling | Pandas, NumPy | Loading and processing dataset |
| NLP | NLTK | Stopword removal, stemming |
| ML | Scikit-learn | TF-IDF vectorisation, Naive Bayes |
| Visualisation | Matplotlib, Seaborn | EDA and evaluation plots |
| Serialisation | Pickle | Saving and loading model artefacts |
| Web framework | Flask | REST API and web interface |

<br>

---

## Project Structure

```
Spam-Email-Classifier/
│
├── train_model.py              ← Model training & evaluation pipeline (run this first)
├── app.py                      ← Flask web application & REST API
│
├── templates/
│   └── index.html              ← Dark-mode web interface
│
├── static/
│   ├── style.css               ← Web interface stylesheet
│   ├── eda_plots.png           ← (Generated after running train_model.py)
│   ├── evaluation_plots.png    ← (Generated after running train_model.py)
│   └── top_spam_words.png      ← (Generated after running train_model.py)
│
├── models/                     ← (Created automatically after running train_model.py)
│   ├── spam_model.pkl          ← Trained Naive Bayes classifier (Pickle)
│   └── tfidf_vectorizer.pkl    ← Fitted TF-IDF Vectorizer (Pickle)
│
├── spam.csv                    ← Dataset (download separately — see instructions below)
├── requirements.txt            ← Project dependencies
├── .gitignore                  ← Git ignore rules
└── README.md                   ← Project documentation
```

<br>

---

## How It Works

Every message passes through this preprocessing pipeline before prediction:

```
Raw message  →  Lowercase  →  Remove non-alpha  →  Remove stopwords & short words (≤ 2)  →  Stem  →  TF-IDF  →  Predict
```

| Step | What happens | Example |
|---|---|---|
| **1. Lowercase** | Normalise all text to lowercase | `"FREE PRIZE"` → `"free prize"` |
| **2. Remove non-alpha** | Strip numbers, punctuation, and symbols | `"win £900!"` → `"win  "` |
| **3. Tokenise & Stopwords** | Split into words and drop common stopwords | `"you have won"` → `["won"]` |
| **4. Filter short words** | Remove tokens with length $\le 2$ | `["to", "go", "win"]` → `["win"]` |
| **5. Porter Stemming** | Reduce words to their root form | `"winning"` → `"win"` |

After preprocessing, **TF-IDF** converts the cleaned text into a numerical matrix where rare, distinctive words score higher than common words. **Multinomial Naive Bayes** computes the probability of Spam vs Ham and returns the class with the higher confidence score.

<br>

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

---

### Step 1 — Clone the repository
```bash
git clone https://github.com/chathurya03-tech/Spam-Email-Classifier.git
cd Spam-Email-Classifier
```

---

### Step 2 — Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

### Step 4 — Download the dataset

This project uses the **UCI SMS Spam Collection Dataset** (5,572 messages).
You can download `spam.csv` from:

| Source | Link |
|---|---|
| Kaggle | https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset |
| UCI ML Repository | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |

> **Note:** The file must be named `spam.csv` and placed in the project root folder (same level as `train_model.py`).
> The dataset has two columns: a label column (`ham`/`spam`) and a message column.

<br>

---

## Running the Project

### Step 1 — Train the model
```bash
python train_model.py
```

This will:
- Load and parse `spam.csv`
- Perform Exploratory Data Analysis and save 3 plots into `static/`
- Preprocess messages through the NLP pipeline
- Train and evaluate the Multinomial Naive Bayes model on a stratified 80/20 split
- Save model artefacts to `models/spam_model.pkl` and `models/tfidf_vectorizer.pkl` using Pickle

---

### Step 2 — Start the Flask server
```bash
python app.py
```

---

### Step 3 — Open the app
Visit **http://127.0.0.1:5000** in your web browser.

Type or paste any SMS or email message and click **Classify Message** to get an instant Spam or Ham verdict with confidence score.

<br>

---

## Model Performance

Evaluated on a held-out test set (20% stratified test split, `random_state=42`):

| Metric | Score |
|---|---|
| ✅ **Accuracy** | **97.58%** |
| 🎯 **Precision** | **96.21%** |
| 🔍 **Recall** | **85.23%** |
| ⚖️ **F1-Score** | **90.39%** |

<br>

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/predict` | Classifies a message, returns JSON |
| `GET` | `/health` | Health check for deployment platforms |

### POST /predict

**Request Body:**
```json
{
  "message": "WINNER!! As a valued customer you have been selected to receive a £900 prize reward! Call 09061701461. Claim code KL341. Valid 12h only."
}
```

**Response:**
```json
{
  "prediction": "Spam",
  "confidence": 99.97,
  "spam_prob": 99.97,
  "ham_prob": 0.03,
  "clean_text": "winner valu custom select receiv prize reward call claim code valid onli"
}
```

<br>

---

## Visualisations

Running `python train_model.py` automatically generates three high-resolution plots in the `static/` directory:

| File | Contents |
|---|---|
| `static/eda_plots.png` | Class distribution pie chart, message length histogram, and average length by class |
| `static/evaluation_plots.png` | Confusion matrix heatmap and performance metrics bar chart |
| `static/top_spam_words.png` | Top 20 spam-indicating words by log-probability |

<br>

---

## Future Improvements

| # | Improvement | Why |
|---|---|---|
| 1 | **BERT / DistilBERT fine-tuning** | Contextual word understanding for higher accuracy |
| 2 | **Gmail API integration** | Automatically scan and label real inbox messages |
| 3 | **LIME / SHAP explainability** | Show users which words triggered the spam verdict |
| 4 | **Active learning feedback loop** | Let users correct wrong predictions and retrain periodically |
| 5 | **Feature engineering** | Add URL count, ALL-CAPS ratio, special character density as extra features |

<br>

---

<div align="center">

Built with Python · Scikit-learn · Flask &nbsp;|&nbsp; UCI SMS Spam Collection Dataset

</div>
