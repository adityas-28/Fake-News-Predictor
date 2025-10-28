import numpy as np
from typing import Optional, Tuple, List, Union
from scipy.sparse import csr_matrix, issparse

class GradientBoostingStumpClassifier:
	"""
	From-scratch Gradient Boosting for binary classification with logistic loss.
	
	- Weak learner: decision stump on feature presence (X[:, j] > 0)
	- Handles sparse CSR matrices efficiently (TF-IDF)
	- Stage-wise additive model in the logit space F(x)
	- Subsampling (rows) and feature bagging supported for scalability
	
	API:
	- fit(X, y)
	- predict_proba(X) -> probability of class 1
	- predict(X) -> {0,1}
	"""
	def __init__(
		self,
		n_estimators: int = 200,
		learning_rate: float = 0.1,
		row_subsample: float = 0.8,
		feature_subsample: float = 0.2,
		random_state: Optional[int] = 42,
		max_newton_steps: int = 10,
		newton_tol: float = 1e-6,
		verbose: bool = False,
	):
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.row_subsample = row_subsample
		self.feature_subsample = feature_subsample
		self.random_state = random_state
		self.max_newton_steps = max_newton_steps
		self.newton_tol = newton_tol
		self.verbose = verbose

		self._stumps: List[Tuple[int, float]] = []  # (feature_index, gamma)
		self._classes_ = np.array([0, 1])
		self._F0: float = 0.0
		self._n_features: int = 0

		if random_state is not None:
			np.random.seed(random_state)

	@staticmethod
	def _sigmoid(z: np.ndarray) -> np.ndarray:
		z = np.clip(z, -500, 500)
		return 1.0 / (1.0 + np.exp(-z))

	def _init_F0(self, y: np.ndarray) -> float:
		# Prior log-odds
		p = np.clip(np.mean(y), 1e-12, 1 - 1e-12)
		return np.log(p / (1 - p))

	def _choose_best_feature(self, X: csr_matrix, y: np.ndarray, F: np.ndarray,
							 row_idx: np.ndarray, feat_idx: np.ndarray) -> Tuple[int, float]:
		"""
		Choose feature j that maximizes |corr(h_j, residuals)| where
		residuals r = y - p, p = sigmoid(F), and h_j(x)=+1 if feature j present else -1.
		Returns (j_best, corr_score).
		"""
		p = self._sigmoid(F[row_idx])
		r = y[row_idx] - p  # pseudo-residuals

		best_j = -1
		best_score = -np.inf

		# For each candidate feature, compute correlation efficiently with sparse presence
		for j in feat_idx:
			col = X[row_idx, j]  # shape (len(row_idx), 1) sparse
			if issparse(col):
				col = col.toarray().ravel()
			else:
				col = np.asarray(col).ravel()

			# Presence mask
			present = col > 0
			# h = +1 for present, -1 for absent
			h = np.where(present, 1.0, -1.0)
			# Correlation (sum r_i * h_i). Scale by 1/len to normalize
			score = np.abs(np.sum(r * h))
			if score > best_score:
				best_score = score
				best_j = j

		return best_j, best_score

	def _fit_gamma_newton(self, h: np.ndarray, y: np.ndarray, F: np.ndarray) -> float:
		"""
		Find optimal gamma for direction h by 1D Newton updates on logistic loss.
		Solves: sum_i h_i * (y_i - sigmoid(F_i + gamma*h_i)) = 0
		"""
		gamma = 0.0
		for _ in range(self.max_newton_steps):
			z = F + gamma * h
			p = self._sigmoid(z)
			grad = np.sum(h * (y - p))
			hess = np.sum((h ** 2) * (p * (1 - p))) + 1e-12
			step = grad / hess
			gamma += step
			if np.abs(step) < self.newton_tol:
				break
		return gamma

	def fit(self, X: Union[csr_matrix, np.ndarray], y: np.ndarray) -> "GradientBoostingStumpClassifier":
		# Ensure formats
		if not issparse(X):
			X = csr_matrix(X)
		y = np.asarray(y).astype(int)
		if set(np.unique(y)) - {0, 1}:
			raise ValueError("y must be binary {0,1}")

		self._n_features = X.shape[1]
		self._stumps = []
		self._F0 = self._init_F0(y)

		F = np.full(X.shape[0], self._F0, dtype=float)

		for m in range(self.n_estimators):
			# Row subsample
			rows = np.arange(X.shape[0])
			if 0 < self.row_subsample < 1.0:
				k = max(1, int(self.row_subsample * X.shape[0]))
				rows = np.random.choice(rows, size=k, replace=False)

			# Feature subsample
			features = np.arange(self._n_features)
			if 0 < self.feature_subsample < 1.0:
				f = max(1, int(self.feature_subsample * self._n_features))
				features = np.random.choice(features, size=f, replace=False)

			# Choose best feature
			j_best, score = self._choose_best_feature(X, y, F, rows, features)
			if j_best < 0:
				# No improvement
				if self.verbose:
					print(f"Iter {m}: no feature improves loss.")
				break

			# Build stump h: +1 if feature present else -1 (on all rows)
			col_full = X[:, j_best]
			present_full = (col_full.indptr[col_full.indices] if False else None)  # placeholder to satisfy linter
			# Efficient presence check on CSR: values > 0
			if issparse(col_full):
				present_mask = np.asarray(col_full.toarray().ravel() > 0, dtype=float)
			else:
				present_mask = (np.asarray(col_full).ravel() > 0).astype(float)
			h = np.where(present_mask > 0, 1.0, -1.0)

			# Compute optimal gamma on subsampled rows for speed
			gamma = self._fit_gamma_newton(h[rows], y[rows], F[rows])

			# Update model on all rows (stage-wise)
			F += self.learning_rate * gamma * h
			self._stumps.append((j_best, float(gamma)))

			if self.verbose and (m % 10 == 0 or m == self.n_estimators - 1):
				p_all = self._sigmoid(F)
				loss = -np.mean(y * np.log(np.clip(p_all, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p_all, 1e-12, 1)))
				print(f"Iter {m}: feature={j_best}, gamma={gamma:.4f}, loss={loss:.5f}")

		return self

	def _decision_values(self, X: Union[csr_matrix, np.ndarray]) -> np.ndarray:
		if not issparse(X):
			X = csr_matrix(X)
		F = np.full(X.shape[0], self._F0, dtype=float)
		for j, gamma in self._stumps:
			col = X[:, j]
			if issparse(col):
				present = np.asarray(col.toarray().ravel() > 0, dtype=float)
			else:
				present = (np.asarray(col).ravel() > 0).astype(float)
			h = np.where(present > 0, 1.0, -1.0)
			F += self.learning_rate * gamma * h
		return F

	def predict_proba(self, X: Union[csr_matrix, np.ndarray]) -> np.ndarray:
		F = self._decision_values(X)
		p = self._sigmoid(F)
		return np.vstack([1 - p, p]).T

	def predict(self, X: Union[csr_matrix, np.ndarray]) -> np.ndarray:
		proba = self.predict_proba(X)[:, 1]
		return (proba >= 0.5).astype(int)
