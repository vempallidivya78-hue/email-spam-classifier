# COMPLETE SPAM CLASSIFIER - All 7 Steps

import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- Step 2: Load ---
df = pd.read_csv('spam.tsv', sep='\t', header=None, names=['label', 'message'])

# --- Step 3: Clean ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# --- Step 4: Vectorize ---
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label_num'],
    test_size=0.2, random_state=42
)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Step 5: Train ---
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --- Step 6: Test accuracy ---
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# --- Step 7: Save the model and vectorizer ---
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved to spam_model.pkl")
print("Vectorizer saved to vectorizer.pkl")

# --- Step 7b: Load and USE the model on NEW emails! ---
print("\n--- Testing on brand new emails! ---")

with open('spam_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    loaded_vec = pickle.load(f)

def predict_email(email_text):
    cleaned = clean_text(email_text)
    vectorized = loaded_vec.transform([cleaned])
    result = loaded_model.predict(vectorized)[0]
    return "🚨 SPAM" if result == 1 else "✅ HAM (normal)"

# Try these test emails!
test_emails = [
    "Congratulations! You won a FREE iPhone. Click now to claim your prize!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your bank account has been suspended. Verify now to win cash!",
    "Don't forget mom's birthday is this Sunday!"
]

for email in test_emails:
    print(f"\nEmail: {email[:50]}...")
    print(f"Result: {predict_email(email)}")