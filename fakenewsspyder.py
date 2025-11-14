# AI-Powered Fake News Detection System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset — try remote CSV, fall back to small built-in sample if download fails
print("Loading dataset...")
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_news_classification.csv'
try:
    df = pd.read_csv(url)
    print(f"Dataset loaded: {len(df)} records (downloaded)")
except Exception as e:
    print(f"Warning: failed to download dataset ({e}). Using built-in sample.")
    sample_texts = [
        "The economy is improving, experts say.",
        "Celebrity endorses miracle cure for weight loss.",
        "New study shows chocolate linked to longer life.",
        "Government confirms new infrastructure plan next year.",
        "Politician caught in fabricated scandal, sources say.",
        "Scientists discover method to reverse aging (unverified).",
        "Local sports team wins championship after dramatic comeback.",
        "Unconfirmed report: alien life found on Mars.",
        "Company announces layoffs after weak quarter.",
        "Hoax: drinking bleach cures all diseases."
    ]
    labels = [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
    texts = (sample_texts * 20)[:200]
    labs = (labels * 20)[:200]
    df = pd.DataFrame({'text': texts, 'label': labs})
    print(f"Dataset loaded: {len(df)} records (built-in sample)")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['text'] = df['text'].astype(str).apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
# plt.show()  # Commented out to prevent blocking

# Save predictions to SQLite database
conn = sqlite3.connect('fake_news_predictions.db')
cursor = conn.cursor()

# Create table
cursor.execute('''CREATE TABLE IF NOT EXISTS predictions
                  (id INTEGER PRIMARY KEY, actual INTEGER, predicted INTEGER, probability_real REAL, probability_fake REAL)''')

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)

# Insert predictions
for idx, (actual, pred, proba) in enumerate(zip(y_test, y_pred, y_pred_proba)):
    cursor.execute('''INSERT INTO predictions (actual, predicted, probability_real, probability_fake)
                      VALUES (?, ?, ?, ?)''', (int(actual), int(pred), float(proba[0]), float(proba[1])))

conn.commit()
conn.close()

print("\n✓ Predictions saved to 'fake_news_predictions.db'")
print("You can query the database using SQL commands.")