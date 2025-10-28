import streamlit as st
import joblib
import numpy as np
from models.optimized_custom_logistic_regression import OptimizedCustomLogisticRegression
# from models.custom_gradient_boosting import GradientBoostingStumpClassifier

# Load models
vectorizer = joblib.load('vectorizer.joblib')
DT = joblib.load('decision_tree_model.joblib')
GBC = joblib.load('gradient_boosting_model.joblib')
RFC = joblib.load('random_forest_model.joblib')

# Load optimized custom logistic regression
try:
	optimized_custom_LR = OptimizedCustomLogisticRegression()
	optimized_custom_LR.load_model('optimized_custom_logistic_regression_model.joblib')
	print("✅ Custom LR model loaded successfully")
except FileNotFoundError:
	print("⚠️ Custom LR model not found, will use sklearn LR instead")
	optimized_custom_LR = None

# Load custom Gradient Boosting (stumps)
try:
	GBS = joblib.load('custom_gb_stumps_model.joblib')
	print("✅ Custom GB (stumps) model loaded successfully")
except FileNotFoundError:
	GBS = None
	print("⚠️ Custom GB model not found. Train it in the notebook to enable.")

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App 🐱")
st.markdown("### Detect whether a piece of news is **Real** or **Fake** using multiple ML models.")

st.write("Enter the news content below 👇")

newsInput = st.text_area("🗞️ Paste your news article here:")

def output_label(output):
	return "✅ Real News" if output == 1 else "❌ Fake News"

if st.button("🔍 Predict"):
	if newsInput.strip():
		transformed_text = vectorizer.transform([newsInput])

		# Initialize results dictionary
		results = {}
		preds = []
		
		# Optimized Custom Logistic Regression prediction (if available)
		if optimized_custom_LR is not None:
			try:
				optimized_custom_LR_pred = optimized_custom_LR.predict(transformed_text)[0]
				results["Optimized Custom Logistic Regression"] = output_label(optimized_custom_LR_pred)
				preds.append(optimized_custom_LR_pred)
			except Exception as e:
				st.warning(f"⚠️ Custom LR prediction failed: {str(e)}")
				st.info("Using sklearn Logistic Regression instead...")
				# Fallback to sklearn LR
				from sklearn.linear_model import LogisticRegression
				sklearn_LR = LogisticRegression(random_state=42, max_iter=1000)
				# Note: This would need to be trained, but for demo purposes we'll skip
				results["Logistic Regression (Sklearn)"] = "Model not available"
		else:
			# Use sklearn LR if custom model not available
			try:
				sklearn_LR = joblib.load('logistic_regression_model.joblib')
				sklearn_LR_pred = sklearn_LR.predict(transformed_text)[0]
				results["Logistic Regression (Sklearn)"] = output_label(sklearn_LR_pred)
				preds.append(sklearn_LR_pred)
			except:
				results["Logistic Regression (Sklearn)"] = "Model not available"
		
		# Custom Gradient Boosting (stumps) prediction if available
		if GBS is not None:
			try:
				GBS_pred = GBS.predict(transformed_text)[0]
				results["Custom Gradient Boosting (Stumps)"] = output_label(GBS_pred)
				preds.append(GBS_pred)
			except Exception as e:
				st.warning(f"⚠️ Custom GB prediction failed: {str(e)}")
		
		# Other models
		DT_pred = DT.predict(transformed_text)[0]
		GBC_pred = GBC.predict(transformed_text)[0]
		RFC_pred = RFC.predict(transformed_text)[0]
		
		results.update({
			"Decision Tree": output_label(DT_pred),
			"Gradient Boosting": output_label(GBC_pred),
			"Random Forest": output_label(RFC_pred)
		})
		
		preds.extend([DT_pred, GBC_pred, RFC_pred])

		st.subheader("📊 Model Predictions:")
		for model, result in results.items():
			st.write(f"**{model}** → {result}")

		# Majority vote (ensemble) - only if we have enough predictions
		if len(preds) >= 3:
			final = 1 if sum(preds) >= len(preds) // 2 + 1 else 0
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

