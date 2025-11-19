import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


vectorizer = joblib.load('vectorizer.joblib')
LR = joblib.load('logistic_regression_model.joblib')
DT = joblib.load('decision_tree_model.joblib')
GBC = joblib.load('gradient_boosting_model.joblib')
RFC = joblib.load('random_forest_model.joblib')
tokenizer = joblib.load("tokenizer.joblib")
LSTM_model = load_model("lstm_model.h5")

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App 🐱")
st.markdown("### Detect whether a piece of news is **Real** or **Fake** using multiple ML models.")

st.write("Enter the news content below 👇")

newsInput = st.text_area("🗞️ Paste your news article here:")

def output_label(output):
    return "✅ Real News" if output == 1 else "❌ Fake News"

def lstm_predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=300)   # make sure maxlen matches training
    pred = LSTM_model.predict(padded)[0][0]
    return 1 if pred >= 0.5 else 0


if st.button("🔍 Predict"):
    if newsInput.strip():
        transformed_text = vectorizer.transform([newsInput])

        LR_pred = LR.predict(transformed_text)[0]
        DT_pred = DT.predict(transformed_text)[0]
        GBC_pred = GBC.predict(transformed_text)[0]
        RFC_pred = RFC.predict(transformed_text)[0]
        LSTM_pred = lstm_predict(newsInput)

        results = {
            "Logistic Regression": output_label(LR_pred),
            "Decision Tree": output_label(DT_pred),
            "Gradient Boosting": output_label(GBC_pred),
            "Random Forest": output_label(RFC_pred),
            "LSTM Model": output_label(LSTM_pred)
        }

        st.subheader("📊 Model Predictions:")
        for model, result in results.items():
            st.write(f"**{model}** → {result}")

        # Majority vote (ensemble)
        preds = [LR_pred, DT_pred, GBC_pred, RFC_pred, LSTM_pred]
        final = 1 if sum(preds) >= 3 else 0
        st.markdown("---")
        st.markdown(f"### 🧭 Final Verdict: **{output_label(final)}**")

    else:
        st.warning("⚠️ Please enter a news article to analyze!")

# ==============================
# Sidebar
# # ==============================
# st.sidebar.header("ℹ️ About the Project")
# st.sidebar.write("""
# This Fake News Detection system uses **Machine Learning models** trained on real-world datasets.

# **Models used:**
# - Logistic Regression  
# - Decision Tree  
# - Gradient Boosting  
# - Random Forest  

# It also uses **TF-IDF Vectorization** to convert text into numerical features for model input.

# Developed by: **Aditya Singh** 🚀
# """)
# st.sidebar.markdown("---")
# st.sidebar.write("Made with ❤️ using Streamlit")

