import streamlit as st
import joblib
import numpy as np
# Source - https://stackoverflow.com/a/71515681
# Posted by brsjak, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-20, License - CC BY-SA 4.0

# from tensorflow.python.keras.layers import Dense
# import tensorflow as tf
# import tensorflow.keras
# Import load_model with fallbacks for different TensorFlow/Keras setups
TENSORFLOW_AVAILABLE = False
try:
    from tensorflow.keras.models import load_model  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        from keras.models import load_model  # type: ignore
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        try:
            import tensorflow as tf  # type: ignore
            load_model = tf.keras.models.load_model
            TENSORFLOW_AVAILABLE = True
        except ImportError:
            # TensorFlow not available - set load_model to None and handle gracefully
            load_model = None
            TENSORFLOW_AVAILABLE = False

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0):
    """
    Pads sequences to the same length.
    Compatible replacement for tensorflow.keras.preprocessing.sequence.pad_sequences
    """
    if not sequences:
        return np.array([], dtype=dtype)
    
    sequences = [np.array(seq, dtype=dtype) for seq in sequences]
    
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    num_samples = len(sequences)
    output = np.full((num_samples, maxlen), value, dtype=dtype)
    
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        
        if truncating == 'pre':
            trunc = seq[-maxlen:]
        else:  # truncating == 'post'
            trunc = seq[:maxlen]
        
        if padding == 'post':
            output[i, :len(trunc)] = trunc
        else:  # padding == 'pre'
            output[i, -len(trunc):] = trunc
    
    return output


vectorizer = joblib.load('vectorizer.joblib')
LR = joblib.load('logistic_regression_model.joblib')
DT = joblib.load('decision_tree_model.joblib')
GBC = joblib.load('gradient_boosting_model.joblib')
RFC = joblib.load('random_forest_model.joblib')
tokenizer = joblib.load("tokenizer.joblib")

# Load LSTM model only if TensorFlow is available
if TENSORFLOW_AVAILABLE and load_model:
    try:
        LSTM_model = load_model("lstm_model.h5")
        LSTM_AVAILABLE = True
    except Exception:
        LSTM_AVAILABLE = False
        LSTM_model = None
else:
    LSTM_AVAILABLE = False
    LSTM_model = None

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App 🐱")
st.markdown("### Detect whether a piece of news is **Real** or **Fake** using multiple ML models.")

st.write("Enter the news content below 👇")

newsInput = st.text_area("🗞️ Paste your news article here:")

def output_label(output):
    return "✅ Real News" if output == 1 else "❌ Fake News"

def lstm_predict(text):
    if not LSTM_AVAILABLE or LSTM_model is None:
        return None  # LSTM not available
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=300)   # make sure maxlen matches training
    pred = LSTM_model.predict(padded, verbose=0)[0][0]
    return 1 if pred >= 0.5 else 0


if st.button("🔍 Predict"):
    if newsInput.strip():
        # Show warning if LSTM is not available
        if not LSTM_AVAILABLE:
            st.warning("⚠️ LSTM model is not available (TensorFlow not installed). Using other models only.")
        
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
        }
        
        # Only add LSTM if available
        if LSTM_AVAILABLE and LSTM_pred is not None:
            results["LSTM Model"] = output_label(LSTM_pred)

        st.subheader("📊 Model Predictions:")
        for model, result in results.items():
            st.write(f"**{model}** → {result}")

        # Majority vote (ensemble) - exclude None values
        preds = [p for p in [LR_pred, DT_pred, GBC_pred, RFC_pred, LSTM_pred] if p is not None]
        final = 1 if sum(preds) >= len(preds) / 2 else 0
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

