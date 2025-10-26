# Fake News Detection App 📰

This is a Machine Learning-based web application built with Streamlit that detects whether a piece of news is **Real** or **Fake**. The app uses multiple ML models for prediction and provides visual insights like confusion matrices.

## Features

- **Predicts news as Real or Fake** using multiple models:
  - Logistic Regression
  - Decision Tree
  - Gradient Boosting
  - Random Forest
- Displays **accuracy, confusion matrix, and classification report** for each model
- **Decision Tree visualization** for interpretability
- User-friendly interface built with **Streamlit**

## Installation

1. **Clone this repository:**

```bash
git clone <repository_url>
cd <repository_folder>
```

2. **Create a virtual environment** (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the Streamlit app:**

```bash
streamlit run app.py
```

2. Open the local URL shown in the terminal (usually `http://localhost:8501`)

3. Paste your news text in the text area and click **Predict**

4. The app will display predictions from all models and a majority-vote final verdict

## Files

- `app.py` — Main Streamlit application
- `vectorizer.joblib` — Saved TF-IDF vectorizer
- `logistic_regression_model.joblib` — Trained Logistic Regression model
- `decision_tree_model.joblib` — Trained Decision Tree model
- `gradient_boosting_model.joblib` — Trained Gradient Boosting model
- `random_forest_model.joblib` — Trained Random Forest model

## Dataset

The models are trained on a labeled dataset of news articles. The dataset contains text samples classified as either real or fake news.

## Technologies Used

- **Python** — Core programming language
- **Streamlit** — Web app framework
- **scikit-learn** — Machine learning models and preprocessing
- **joblib** — Model serialization
- **pandas** — Data manipulation
- **matplotlib/seaborn** — Visualization


---

**Note:** Make sure you have the trained model files (`.joblib`) in your project directory before running the app.