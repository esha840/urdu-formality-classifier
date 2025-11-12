import streamlit as st
import pickle
import pandas as pd

# -----------------------------------
# Load Model and Vectorizer
# -----------------------------------
import streamlit as st
import pickle
import pandas as pd
import os

# -----------------------------------
# Load Model and Vectorizer (Safe Method)
# -----------------------------------
# Find the correct absolute path of current app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "urdu_formality_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "urdu_vectorizer.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

except FileNotFoundError:
    st.error("❌ Model or vectorizer file not found. Please make sure both .pkl files are uploaded in your GitHub repo.")
    st.stop()
except Exception as e:
    st.error(f"⚠ Error loading model/vectorizer: {e}")
    st.stop()

# -----------------------------------
# Streamlit Web App UI
# -----------------------------------
st.set_page_config(page_title="Urdu Formality Classifier", page_icon="🇵🇰", layout="centered")

st.title("🇵🇰 Urdu Formality Classifier")
st.write("یہ ایپ چیک کرتی ہے کہ دیا گیا جملہ *ادب والا (Formal)* ہے یا *عام بول چال والا (Informal)*۔")

# -----------------------------------
# 1. Single Sentence Input
# -----------------------------------
sentence = st.text_area("🔹 Enter your Urdu sentence here:", height=120)

if st.button("Check Formality for Single Sentence"):
    if sentence.strip() == "":
        st.warning("Please enter a sentence first.")
    else:
        text_vec = vectorizer.transform([sentence])
        prediction = model.predict(text_vec)[0]

        if prediction == "formal":
            st.success("🌷 This sentence is *Formal (ادب والی زبان)*")
        else:
            st.info("😊 This sentence is *Informal (عام بول چال والی زبان)*")

st.markdown("---")

# -----------------------------------
# 2. File Upload Section
# -----------------------------------
st.subheader("📁 Upload a .txt or .csv file with Urdu sentences")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if "sentence" not in df.columns:
                st.error("CSV must have a column named 'sentence'.")
            else:
                sentences = df['sentence'].astype(str).tolist()
        else:  # .txt file
            sentences = [line.strip() for line in uploaded_file.readlines() if line.strip() != ""]

        # Transform and predict
        text_vec = vectorizer.transform(sentences)
        predictions = model.predict(text_vec)

        # Show results
        results_df = pd.DataFrame({"sentence": sentences, "formality": predictions})
        st.write(results_df)

        # Optional: Download results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='urdu_formality_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")