import streamlit as st
import joblib
import re


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Support Ticket Categorizer",
    page_icon="🎫",
    layout="centered"
)


# ==========================================
# Load Model and TF-IDF Vectorizer
# ==========================================

model = joblib.load("ticket_classifier.pkl")

tfidf = joblib.load("tfidf_vectorizer.pkl")


# ==========================================
# Text Cleaning Function
# ==========================================

def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove special characters and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# Application UI
# ==========================================

st.title("🎫 Support Ticket Categorizer")

st.write(
    "Automatically classify incoming support tickets "
    "into Billing, Technical, HR, or General categories."
)


# ==========================================
# Input Fields
# ==========================================

subject = st.text_input(
    "Ticket Subject",
    placeholder="Example: Payment failed"
)


body = st.text_area(
    "Ticket Description",
    placeholder="Example: My credit card payment was declined."
)


# ==========================================
# Predict Button
# ==========================================

if st.button("🔍 Categorize Ticket"):

    # Check if input is empty

    if subject.strip() == "" or body.strip() == "":

        st.warning(
            "Please enter both subject and ticket description."
        )

    else:

        # Combine subject and body

        text = subject + " " + body


        # Clean text

        cleaned_text = clean_text(text)


        # Convert text into TF-IDF

        text_tfidf = tfidf.transform(
            [cleaned_text]
        )


        # Predict category

        prediction = model.predict(
            text_tfidf
        )[0]


        # Get prediction probability

        probabilities = model.predict_proba(
            text_tfidf
        )[0]


        # Get confidence

        confidence = max(
            probabilities
        ) * 100


        # ==========================================
        # Display Result
        # ==========================================

        st.success(
            f"Predicted Category: {prediction}"
        )


        st.info(
            f"Confidence: {confidence:.2f}%"
        )


        # ==========================================
        # Display Routing Department
        # ==========================================

        st.subheader(
            "📌 Ticket Routing"
        )


        if prediction == "Billing":

            st.write(
                "➡️ Route this ticket to the **Billing Department**."
            )


        elif prediction == "Technical":

            st.write(
                "➡️ Route this ticket to the **Technical Support Department**."
            )


        elif prediction == "HR":

            st.write(
                "➡️ Route this ticket to the **HR Department**."
            )


        else:

            st.write(
                "➡️ Route this ticket to the **General Support Department**."
            )