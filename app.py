import streamlit as st
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 👉 Move this function OUTSIDE the cached function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@st.cache_data
def load_model():
    df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv', sep='\t', header=None)
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['cleaned'] = df['message'].apply(preprocess)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

# Load model
model, vectorizer = load_model()

# UI
st.title("📧 Spam Email Classifier")
st.write("Paste your email content below and click **Classify**.")

user_input = st.text_area("Email Text", height=200, placeholder="Paste your email here...")

if st.button("Classify"):
    if user_input.strip():
        cleaned = preprocess(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        if prediction == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Good Email (Not Spam)")
    else:
        st.warning("Please enter some text.")
