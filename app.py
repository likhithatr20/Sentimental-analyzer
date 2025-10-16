import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# --- Load stopwords ---
stop_words = set(stopwords.words('english'))

# --- Load the saved model and vectorizer ---
with open('best_sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('count_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# --- Preprocessing function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Tweet Sentiment Analyzer ",
    page_icon="",
    layout="centered",
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f0f4f8, #dfe9f3);
        }
        .stApp {
            background: linear-gradient(135deg, #f8fbff 0%, #eef3f8 100%);
        }
        .main-title {
            text-align: center;
            font-size: 2.2em;
            color: #1e3d59;
            font-weight: 800;
        }
        .subtext {
            text-align: center;
            font-size: 1.1em;
            color: #364f6b;
        }
        .result-box {
            padding: 1em;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            margin-top: 15px;
        }
        .positive {
            background-color: #d4edda;
            color: #155724;
        }
        .negative {
            background-color: #f8d7da;
            color: #721c24;
        }
        .neutral {
            background-color: #fff3cd;
            color: #856404;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<p class="main-title"> COVID-19 Vaccination Tweet Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Analyze what people feel about COVID-19 vaccines — Positive, Negative, or Neutral </p>', unsafe_allow_html=True)
st.write("")

# --- Input Area ---
user_input = st.text_area("Enter a tweet below:", placeholder="Type or paste a tweet...")

# --- Button Action ---
if st.button(" Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning(" Please enter some text to analyze.")
    else:
        cleaned_text = clean_text(user_input)
        text_vector = vectorizer.transform([cleaned_text])

        # --- ML Model Prediction ---
        prediction = model.predict(text_vector)[0]

        # --- TextBlob Polarity ---
        polarity = TextBlob(user_input).sentiment.polarity

        # --- Hybrid Correction Logic ---
        if polarity < -0.1:
            prediction = "Negative"
        elif polarity > 0.1:
            prediction = "Positive"
        else:
            prediction = "Neutral"

        # --- Display Results ---
        if prediction == "Positive":
            st.markdown(f'<div class="result-box positive">😊 <b>Positive Sentiment</b><br>Polarity: {polarity:.2f}</div>', unsafe_allow_html=True)
        elif prediction == "Negative":
            st.markdown(f'<div class="result-box negative">😞 <b>Negative Sentiment</b><br>Polarity: {polarity:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box neutral">😐 <b>Neutral Sentiment</b><br>Polarity: {polarity:.2f}</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>By Likhitha</p>", unsafe_allow_html=True)
