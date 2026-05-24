import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

@st.cache_resource
def load_model():
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    return model, vec

model, vectorizer = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

st.title("📧 Email Spam Classifier")
st.markdown("Built with Naive Bayes — 98% accuracy")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "98%")
col2.metric("Training emails", "4,457")
col3.metric("Unique words", "7,455")

st.divider()

st.subheader("Check an email")
email_input = st.text_area("Paste any email message below:", height=120,
    placeholder="e.g. Congratulations! You won a FREE iPhone...")

st.markdown("**Try an example:**")
ex1, ex2, ex3 = st.columns(3)
if ex1.button("Spam example"):
    email_input = "Congratulations! You won a FREE iPhone. Claim your prize now!"
if ex2.button("Normal example"):
    email_input = "Hey, are we still meeting for lunch tomorrow?"
if ex3.button("Urgent spam"):
    email_input = "URGENT: Your account suspended. Verify now to win cash!"

if st.button("Classify email", type="primary") and email_input:
    cleaned = clean_text(email_input)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]

    if result == 1:
        st.error(f"🚨 SPAM DETECTED  —  {round(proba[1]*100)}% confidence")
    else:
        st.success(f"✅ Normal email (Ham)  —  {round(proba[0]*100)}% confidence")

    with st.expander("See details"):
        st.write("Cleaned text:", cleaned)
        st.write(f"Spam probability: {round(proba[1]*100, 1)}%")
        st.write(f"Ham probability:  {round(proba[0]*100, 1)}%")