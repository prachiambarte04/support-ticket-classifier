# 🎫 Support Ticket Categorizer

An NLP-based machine learning application that automatically categorizes incoming support tickets into four departments:

- Billing
- Technical
- HR
- General

## 🚀 Project Overview

The application uses Natural Language Processing and Machine Learning to automatically classify support tickets based on their subject and description.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Streamlit
- Joblib

## 🔄 Machine Learning Pipeline

Support Ticket
→ Text Cleaning
→ Subject + Body Combination
→ TF-IDF Vectorization
→ Multinomial Naive Bayes
→ Category Prediction

## 📂 Project Structure

support-ticket-classifier/
│
├── data/
│   └── tickets.txt
│
├── app.py
├── nlp_model.py
├── ticket_classifier.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md

## ▶️ Run Locally

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
