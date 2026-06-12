# SPAM EMAIL CLASSIFIER - Model Training Script

# Description: End-to-end ML pipeline to classify SMS/Email messages as Spam or Ham (Not Spam) using TF-IDF + Naive Bayes

# Imports 
import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# Download required NLTK data (only runs once) 
print(" Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# STEP 1 – LOAD DATASET

# This loader is designed to handle ANY variant of the SMS Spam Collection:

#   Format A (tab-separated, no header):
#       ham\tHey how are you
#       spam\tWINNER!! Call 09061...

#   Format B (comma-separated, with header, extra columns):
#       v1,v2,v3,v4,v5
#       ham,"Hey, how are you",,, 
#       spam,"WINNER!! Call 09061",,, 

# The strategy: try the most common formats in order and use the first
# one that produces a clean two-column (label / message) DataFrame.

print("\n Loading dataset...")

df = None

# Attempt 1: tab-separated (original UCI format) 
try:
    _df = pd.read_csv(
        'spam.csv',
        sep='\t',
        header=None,
        names=['label', 'message'],
        encoding='latin-1',
        on_bad_lines='skip',   # silently skip any malformed rows
    )
    # A valid load produces exactly 2 columns and labels are 'ham' or 'spam'
    if set(_df['label'].str.lower().unique()).issubset({'ham', 'spam'}):
        df = _df
        print(" Detected format : tab-separated (UCI original)")
except Exception:
    pass

# Attempt 2: comma-separated, possibly with extra columns 
if df is None:
    try:
        _df = pd.read_csv(
            'spam.csv',
            encoding='latin-1',
            on_bad_lines='skip',   # skip rows with unexpected extra commas
        )
        # Keep only the first two columns and rename them
        _df = _df.iloc[:, :2].copy()
        _df.columns = ['label', 'message']
        if set(_df['label'].str.lower().unique()).issubset({'ham', 'spam'}):
            df = _df
            print("   Detected format : comma-separated (Kaggle variant)")
    except Exception:
        pass

# Attempt 3: comma-separated WITHOUT a header row 
if df is None:
    try:
        _df = pd.read_csv(
            'spam.csv',
            encoding='latin-1',
            header=None,
            on_bad_lines='skip',
        )
        _df = _df.iloc[:, :2].copy()
        _df.columns = ['label', 'message']
        if set(_df['label'].str.lower().unique()).issubset({'ham', 'spam'}):
            df = _df
            print("   Detected format : comma-separated, no header")
    except Exception:
        pass

# Give up if nothing worked 
if df is None:
    raise ValueError(
        "\n❌ Could not parse spam.csv.\n"
        "   Make sure the file is the SMS Spam Collection dataset with\n"
        "   'ham' / 'spam' labels in the first column and messages in\n"
        "   the second column, separated by a tab OR comma."
    )

# Normalise labels to lowercase (handles 'Ham', 'HAM', etc.) 
df['label'] = df['label'].str.lower().str.strip()

# Drop rows where label or message is missing 
df.dropna(subset=['label', 'message'], inplace=True)

# Ensure message column is always a string (never float/NaN after dropna)
df['message'] = df['message'].astype(str).str.strip()

# Drop blank messages and any label that is NOT ham/spam
df = df[df['message'] != '']
df = df[df['label'].isin(['ham', 'spam'])].reset_index(drop=True)

print(f" Dataset shape : {df.shape}")
print(f" Columns       : {df.columns.tolist()}")
print(f"\n Label distribution:\n{df['label'].value_counts()}\n")


# STEP 2 – EXPLORATORY DATA ANALYSIS (EDA)

# Before building a model, understand what the data looks like.

print("📊 Performing EDA...")

# Add a column for message length (useful feature to visualise)
df['msg_length'] = df['message'].apply(len)

# Plot 1: Class Distribution (Spam vs Ham) 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Spam Classifier – Exploratory Data Analysis', fontsize=16, fontweight='bold')

label_counts = df['label'].value_counts()
colors = ['#2ecc71', '#e74c3c']   # green = ham, red = spam
axes[0].pie(
    label_counts,
    labels=label_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[0].set_title('Class Distribution', fontweight='bold')

# Plot 2: Message Length Distribution 
for label, color in zip(['ham', 'spam'], colors):
    subset = df[df['label'] == label]['msg_length']
    axes[1].hist(subset, bins=40, alpha=0.7, label=label, color=color, edgecolor='white')
axes[1].set_xlabel('Message Length (characters)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Message Length Distribution', fontweight='bold')
axes[1].legend()

# Plot 3: Average Message Length per Class 
avg_len = df.groupby('label')['msg_length'].mean()
axes[2].bar(avg_len.index, avg_len.values, color=colors, edgecolor='white', linewidth=1.5)
axes[2].set_xlabel('Label')
axes[2].set_ylabel('Average Length (characters)')
axes[2].set_title('Average Message Length by Class', fontweight='bold')
for i, v in enumerate(avg_len.values):
    axes[2].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
os.makedirs('static', exist_ok=True)
plt.savefig('static/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   EDA plots saved → static/eda_plots.png")


# STEP 3 – TEXT PREPROCESSING

# Raw text is messy. We clean it so the model focuses on meaningful words.

# Pipeline for each message:
#   1. Lowercase  → "FREE" and "free" become the same word
#   2. Remove non-alpha characters  → strip numbers, punctuation, symbols
#   3. Tokenise   → split sentence into individual words (tokens)
#   4. Remove stopwords → drop common words like "the", "is", "a"
#   5. Stem       → reduce words to their root ("running" → "run")

print("\n🔧 Preprocessing text...")

stemmer   = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Clean and normalise a raw text message.

    Parameters
    ----------
    text : str  – raw message string

    Returns
    -------
    str – cleaned, stemmed, stopword-free string
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove everything that is NOT a letter or space
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenise (split into words)
    tokens = text.split()

    # 4. Remove stopwords AND very short words (≤ 2 chars)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    # 5. Stem each remaining token
    tokens = [stemmer.stem(w) for w in tokens]

    return ' '.join(tokens)

# Apply preprocessing to every message
df['clean_message'] = df['message'].apply(preprocess_text)

print("   Sample transformations:")
for i in [0, 1, 5]:
    print(f"\n   Original : {df['message'].iloc[i][:80]}")
    print(f"   Cleaned  : {df['clean_message'].iloc[i][:80]}")


# STEP 4 – LABEL ENCODING

# ML models need numbers, not strings.
#   ham  → 0
#   spam → 1

df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})


# STEP 5 – TRAIN / TEST SPLIT

# We hold out 20 % of the data as a test set.
# stratify=y ensures both sets have the same spam/ham ratio.
# random_state=42 makes the split reproducible (same result every run).

print("\n✂️  Splitting dataset (80 % train / 20 % test)...")

X = df['clean_message']
y = df['label_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")


# STEP 6 – TF-IDF VECTORISATION

# "TF-IDF" stands for Term Frequency – Inverse Document Frequency.

# • TF  : how often a word appears in THIS message
# • IDF : how rare the word is across ALL messages

# Words that appear everywhere ("the", "is") get a LOW score.
# Rare, distinctive words ("winner", "prize") get a HIGH score.

# max_features=5000  → keep only the 5000 most informative words
# ngram_range=(1,2)  → also consider two-word phrases ("win prize")

print("\n🔢 Vectorising text with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2), # unigrams + bigrams
    sublinear_tf=True   # apply log(1 + tf) to dampen very high frequencies
)

X_train_tfidf = vectorizer.fit_transform(X_train) # learn vocab + transform
X_test_tfidf  = vectorizer.transform(X_test)  # transform only (no leakage)

print(f" Vocabulary size  : {len(vectorizer.vocabulary_)}")
print(f" Feature matrix   : {X_train_tfidf.shape}")


# STEP 7 – TRAIN MULTINOMIAL NAIVE BAYES

# Why Naive Bayes for text?
#   • Fast and memory-efficient
#   • Works very well even with small datasets
#   • The "naive" independence assumption between words is surprisingly OK for spam

# alpha=0.1  → Laplace smoothing to avoid zero-probability words

print("\n🤖 Training Multinomial Naive Bayes model...")

model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)
print("   Training complete ✓")


# STEP 8 – EVALUATION

# • Accuracy   : overall % correct   (misleading when classes are imbalanced)
# • Precision  : of predicted spam,  how many were actually spam?
# • Recall     : of actual spam,      how many did we catch?
# • F1-Score   : harmonic mean of Precision & Recall (best single metric)

print("\n📈 Evaluating model...")

y_pred = model.predict(X_test_tfidf)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"\n   ┌────────────────────┬────────┐")
print(f"   │ Metric             │ Score  │")
print(f"   ├────────────────────┼────────┤")
print(f"   │ Accuracy           │ {accuracy:.4f} │")
print(f"   │ Precision          │ {precision:.4f} │")
print(f"   │ Recall             │ {recall:.4f} │")
print(f"   │ F1-Score           │ {f1:.4f} │")
print(f"   └────────────────────┴────────┘")

print(f"\n   Full Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion Matrix Plot – shows true positives, false positives, true negatives, false negatives
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Evaluation', fontsize=16, fontweight='bold')

# Heatmap
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Ham', 'Spam'],
    yticklabels=['Ham', 'Spam'],
    linewidths=1, linecolor='white',
    ax=axes[0]
)
axes[0].set_xlabel('Predicted Label', fontweight='bold')
axes[0].set_ylabel('Actual Label',    fontweight='bold')
axes[0].set_title('Confusion Matrix', fontweight='bold')

# Metrics bar chart
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
bar_colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
bars = axes[1].bar(metrics.keys(), metrics.values(), color=bar_colors, edgecolor='white', linewidth=1.5)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel('Score')
axes[1].set_title('Performance Metrics', fontweight='bold')
for bar, val in zip(bars, metrics.values()):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f'{val:.4f}', ha='center', fontweight='bold', fontsize=10
    )

plt.tight_layout()
plt.savefig('static/evaluation_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Evaluation plots saved → static/evaluation_plots.png")


# STEP 9 – TOP SPAM-INDICATING WORDS

print("\n🔍 Top spam-indicating words...")

feature_names = vectorizer.get_feature_names_out()
spam_class_idx = 1   # index for 'spam' in the model

# log-probability of each feature for the spam class
log_probs   = model.feature_log_prob_[spam_class_idx]
top_indices = np.argsort(log_probs)[-20:][::-1]
top_words   = [(feature_names[i], log_probs[i]) for i in top_indices]

print("   Top 20 spam keywords:")
for word, prob in top_words[:10]:
    print(f"     {word:<20} {prob:.4f}")

# Visualise top words
words, scores = zip(*top_words)
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(words))
bars = ax.barh(y_pos, scores, color='#e74c3c', edgecolor='white', linewidth=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(words, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Log Probability (spam class)', fontweight='bold')
ax.set_title('Top 20 Spam-Indicating Words / Phrases', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/top_spam_words.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Top-words plot saved → static/top_spam_words.png")


# STEP 10 – SAVE MODEL & VECTORIZER WITH PICKLE

# Pickle serialises Python objects to disk so we can reload them in Flask without retraining every time.

print("\n💾 Saving model and vectorizer...")
os.makedirs('models', exist_ok=True)

with open('models/spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(" models/spam_model.pkl ✓")
print(" models/tfidf_vectorizer.pkl ✓")


# DONE

print("\n✅ Training pipeline complete!")
print("   Run  python app.py  to start the Flask web application.\n")