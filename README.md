# 📧 Email Spam Classifier

A machine learning project that classifies emails as spam or ham using Naive Bayes.

## 🎯 Result
98% accuracy on 5,572 real emails

## 🛠️ Tech Stack
- Python
- Scikit-learn
- NLTK
- Pandas
- Streamlit

## 📁 Files
- `spam_classifier.py` — trains and saves the ML model
- `dashboard.py` — Streamlit web dashboard
- `spam.tsv` — dataset of 5,572 labelled emails

## 🚀 How to Run

### Train the model
python spam_classifier.py

### Launch the dashboard
streamlit run dashboard.py

## 📊 Results
- Training emails: 4,457
- Testing emails: 1,115
- Unique words: 7,455
- Final accuracy: 98%
