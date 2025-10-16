# Sentimental-analyzer
**Vaccine Tweet Sentiment Analysis
Overview**

This project analyzes public opinions on COVID-19 vaccination by classifying tweets into Positive, Neutral, and Negative sentiments. The goal is to understand public sentiment trends and identify areas for awareness campaigns or misinformation monitoring.

**Features**

Data Cleaning: Removed URLs, special characters, stopwords, and applied stemming.

Feature Extraction: TF-IDF vectorization with unigrams and bigrams.

Modeling: Logistic Regression and LinearSVC with hyperparameter tuning for better accuracy.

Visualization: Sentiment distribution plots and word clouds for each sentiment category.

Interactive App: Streamlit interface for predicting sentiment of custom tweets.

**Tech Stack**

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, NLTK, TextBlob, Seaborn, WordCloud, Streamlit

**Usage**

Enter any tweet in the Streamlit input box to predict its sentiment (Positive, Neutral, or Negative).

View visualizations and trends directly in the app.

**Purpose**

To provide insights into public sentiment regarding COVID-19 vaccination, helping researchers, policymakers, and health organizations make informed decisions.

**Note**

The model may not always be 100% accurate due to the complexity of natural language and sarcasm in tweets.

Best results are achieved with clear, concise sentences.
