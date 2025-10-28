import numpy as np
import joblib
from custom_decision_tree import DecisionTree

class CustomRandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples=2,
                 feature_subsample_size=None, bootstrap=True, random_state=None):
        """
        Custom Random Forest using your custom DecisionTree.

        Parameters:
        ----------
        n_estimators : int
            Number of trees in the forest.
        max_depth : int
            Maximum depth of each tree.
        min_samples : int
            Minimum samples per split in each tree.
        feature_subsample_size : int or None
            Number of features to sample per tree. If None, use sqrt(n_features).
        bootstrap : bool
            Whether to use bootstrap sampling for training trees.
        random_state : int or None
            Random seed.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_subsample_size = feature_subsample_size
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_indices_list = []

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        """
        Train the random forest using bootstrapped samples and feature subsets.
        """
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        if self.feature_subsample_size is None:
            self.feature_subsample_size = int(np.sqrt(n_features))

        for i in range(self.n_estimators):
            # --- Bootstrap sample ---
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample, y_sample = X, y

            # --- Random feature selection ---
            feature_indices = np.random.choice(n_features, self.feature_subsample_size, replace=False)
            self.feature_indices_list.append(feature_indices)

            # Subset of features
            X_subset = X_sample[:, feature_indices]

            # --- Train custom decision tree ---
            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            dataset = np.concatenate((X_subset, y_sample), axis=1)
            tree.root = tree.build_tree(dataset)

            self.trees.append(tree)

            if (i + 1) % 5 == 0 or i == self.n_estimators - 1:
                print(f"Trained {i+1}/{self.n_estimators} trees.")

    def predict(self, X):
        """
        Predict by majority voting among all trees.
        """
        all_preds = []

        for tree, feature_indices in zip(self.trees, self.feature_indices_list):
            X_subset = X[:, feature_indices]
            preds = np.array(tree.predict(X_subset))
            all_preds.append(preds)

        # Convert to numpy array of shape (n_trees, n_samples)
        all_preds = np.array(all_preds)

        # Majority voting along axis 0
        y_pred = np.round(np.mean(all_preds, axis=0)).astype(int)
        return y_pred

    def save_model(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load_model(path):
        return joblib.load(path)
