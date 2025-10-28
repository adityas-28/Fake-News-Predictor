import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from custom_gradient_boosting import GradientBoostingStumpClassifier


def main() -> None:
	# Load data
	true_df = pd.read_csv('True.csv')
	fake_df = pd.read_csv('Fake.csv')

	# Determine text column
	candidates = [
		c for c in true_df.columns
		if c.lower() in ['text', 'content', 'article', 'news', 'body'] or c.lower().startswith('text')
	]
	text_col = candidates[0] if candidates else true_df.columns[0]

	true_texts = true_df[text_col].astype(str).tolist()
	fake_texts = fake_df[text_col].astype(str).tolist()

	X_text = true_texts + fake_texts
	y = np.array([1] * len(true_texts) + [0] * len(fake_texts))

	# Vectorizer (load if exists, else fit new)
	try:
		vectorizer = joblib.load('vectorizer.joblib')
		print('Loaded vectorizer.joblib')
	except Exception:
		from sklearn.feature_extraction.text import TfidfVectorizer
		print('Fitting new vectorizer...')
		vectorizer = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2))
		vectorizer.fit(X_text)
		joblib.dump(vectorizer, 'vectorizer.joblib')
		print('Saved vectorizer.joblib')

	X = vectorizer.transform(X_text)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	# Train custom Gradient Boosting (stumps)
	model = GradientBoostingStumpClassifier(
		n_estimators=300,
		learning_rate=0.1,
		row_subsample=0.8,
		feature_subsample=0.2,
		random_state=42,
		verbose=True,
	)
	model.fit(X_train, y_train)

	# Evaluate
	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f'Custom GB (stumps) accuracy: {acc:.4f}')

	# Save
	joblib.dump(model, 'custom_gb_stumps_model.joblib')
	print('Saved custom_gb_stumps_model.joblib')


if __name__ == '__main__':
	main()


