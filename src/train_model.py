# src/train_model.py
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Paths
DATA_PATH = "../data/spam.csv"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Step 1: Load dataset
df = pd.read_csv(DATA_PATH, encoding='latin-1')

# Step 2: Rename columns
df = df.rename(columns={'v1':'label', 'v2':'message'})

# Step 3: Encode labels
df['label_num'] = df['label'].map({'ham':0,'spam':1})

# Step 4: Preprocess text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['clean_message'] = df['message'].apply(clean_text)

# Step 5: Split dataset
X = df['clean_message']
y = df['label_num']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Step 6: TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 8: Evaluate
y_val_pred = model.predict(X_val_vec)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

y_test_pred = model.predict(X_test_vec)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# Step 9: Save model and vectorizer
joblib.dump(model, os.path.join(MODEL_DIR, "spam_model.joblib"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

print("Model and vectorizer saved to 'models/' folder.")