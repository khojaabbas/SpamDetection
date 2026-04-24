# src/spam_app.py
import streamlit as st
import joblib
import string
import pandas as pd
import os
import altair as alt

# -----------------------------
# Step 1: Load Model & Vectorizer
# -----------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
model = joblib.load(os.path.join(MODEL_DIR, "spam_model.joblib"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

# -----------------------------
# Step 2: Text Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# -----------------------------
# Step 3: App Layout
# -----------------------------
st.markdown("<h1 style='color:#4B8BBE;'>Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:gray;'>Enter one or multiple messages (one per line) to check if they are Spam or Ham.</p>", unsafe_allow_html=True)
st.write("---")

# Optional: Example Messages
if st.button("Use Example Messages"):
    st.session_state.user_input = "Win $1000 now!\nMeeting at 10am tomorrow."

# Input Box
user_input = st.text_area("Enter your messages here:", key="user_input", height=150)

# -----------------------------
# Step 4: Predict Button
# -----------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter at least one message.")
    else:
        # Process multiple messages
        messages = [msg.strip() for msg in user_input.split("\n") if msg.strip() != ""]
        cleaned_messages = [clean_text(msg) for msg in messages]

        # Vectorize
        msg_vec = vectorizer.transform(cleaned_messages)

        # Predictions
        predictions = model.predict(msg_vec)
        probabilities = model.predict_proba(msg_vec)[:, 1]  # probability of spam

        # Create DataFrame for display
        df_results = pd.DataFrame({
            "Message": messages,
            "Prediction": ["Spam" if p==1 else "Not Spam" for p in predictions],
            "Probability of Spam (%)": [f"{prob*100:.2f}" for prob in probabilities]
        })

        # -----------------------------
        # Step 5: Color-coded Table
        # -----------------------------
        def highlight_row(row):
            return ['background-color: #ffcccc' if row['Prediction']=='Spam' else 'background-color: #ccffcc']*len(row)

        st.subheader("Prediction Results")
        st.dataframe(df_results.style.apply(highlight_row, axis=1))

        # -----------------------------
        # Step 6: Probability Bar Chart
        # -----------------------------
        chart = alt.Chart(df_results).mark_bar().encode(
            x=alt.X('Probability of Spam (%)', title='Spam Probability (%)'),
            y=alt.Y('Message', sort=None, title='Message'),
            color=alt.Color('Prediction', scale=alt.Scale(domain=['Spam','Not Spam'], range=['red','green']))
        )

        st.subheader("Spam Probability Chart")
        st.altair_chart(chart, use_container_width=True)