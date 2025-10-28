import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import warnings
from scipy.sparse import csr_matrix, issparse
from sklearn.utils import shuffle

class OptimizedCustomLogisticRegression:
    """
    Optimized Custom Logistic Regression implementation for large datasets.
    
    This class implements logistic regression using mini-batch gradient descent
    with sparse matrix support to handle large datasets efficiently.
    
    Features:
    - Sparse matrix support for memory efficiency
    - Mini-batch gradient descent
    - L1 and L2 regularization
    - Learning rate scheduling
    - Early stopping
    - Progress tracking
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 max_iterations: int = 1000,
                 regularization: str = 'l2',
                 lambda_reg: float = 0.01,
                 batch_size: int = 1000,
                 early_stopping: bool = True,
                 tolerance: float = 1e-4,
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize the Optimized Custom Logistic Regression model.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        max_iterations : int, default=1000
            Maximum number of iterations
        regularization : str, default='l2'
            Type of regularization ('l1', 'l2', or None)
        lambda_reg : float, default=0.01
            Regularization strength
        batch_size : int, default=1000
            Batch size for mini-batch gradient descent
        early_stopping : bool, default=True
            Whether to use early stopping
        tolerance : float, default=1e-4
            Tolerance for early stopping
        random_state : int, optional
            Random state for reproducibility
        verbose : bool, default=True
            Whether to print progress
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.tolerance = tolerance
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize parameters
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.accuracy_history = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function with numerical stability.
        
        Parameters:
        -----------
        z : np.ndarray
            Input values
            
        Returns:
        --------
        np.ndarray
            Sigmoid output
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost_sparse(self, X: Union[csr_matrix, np.ndarray], y: np.ndarray) -> float:
        """
        Compute the logistic regression cost function for sparse or dense matrices.
        
        Parameters:
        -----------
        X : Union[csr_matrix, np.ndarray]
            Sparse or dense feature matrix
        y : np.ndarray
            Target labels
            
        Returns:
        --------
        float
            Cost value
        """
        m = X.shape[0]
        z = X.dot(self.weights) + self.bias
        h = self._sigmoid(z)
        
        # Avoid log(0) by clipping
        h = np.clip(h, 1e-15, 1 - 1e-15)
        
        # Cross-entropy cost
        cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        
        # Add regularization
        if self.regularization == 'l2':
            cost += (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            cost += (self.lambda_reg / m) * np.sum(np.abs(self.weights))
        
        return cost
    
    def _compute_gradients_sparse(self, X: Union[csr_matrix, np.ndarray], y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias using sparse matrix operations.
        
        Parameters:
        -----------
        X : Union[csr_matrix, np.ndarray]
            Sparse or dense feature matrix
        y : np.ndarray
            Target labels
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            Gradient for weights and bias
        """
        m = X.shape[0]
        z = X.dot(self.weights) + self.bias
        h = self._sigmoid(z)
        
        # Compute gradients using sparse matrix operations
        error = h - y
        
        if issparse(X):
            gradient_result = X.T.dot(error)
            if issparse(gradient_result):
                dw = (1/m) * gradient_result.toarray().flatten()
            else:
                dw = (1/m) * gradient_result.flatten()
        else:
            dw = (1/m) * X.T.dot(error)
        
        db = (1/m) * np.sum(error)
        
        # Add regularization gradients
        if self.regularization == 'l2':
            dw += (self.lambda_reg / m) * self.weights
        elif self.regularization == 'l1':
            dw += (self.lambda_reg / m) * np.sign(self.weights)
        
        return dw, db
    
    def _initialize_parameters(self, n_features: int):
        """
        Initialize weights and bias.
        
        Parameters:
        -----------
        n_features : int
            Number of features
        """
        # Xavier initialization
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
    
    def _get_mini_batch(self, X: csr_matrix, y: np.ndarray, batch_indices: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """
        Get a mini-batch from the dataset.
        
        Parameters:
        -----------
        X : csr_matrix
            Sparse feature matrix
        y : np.ndarray
            Target labels
        batch_indices : np.ndarray
            Indices for the mini-batch
            
        Returns:
        --------
        Tuple[csr_matrix, np.ndarray]
            Mini-batch features and labels
        """
        return X[batch_indices], y[batch_indices]
    
    def fit(self, X: Union[csr_matrix, np.ndarray], y: np.ndarray, 
            X_val: Optional[Union[csr_matrix, np.ndarray]] = None, 
            y_val: Optional[np.ndarray] = None) -> 'OptimizedCustomLogisticRegression':
        """
        Fit the logistic regression model.
        
        Parameters:
        -----------
        X : Union[csr_matrix, np.ndarray]
            Training features (sparse or dense)
        y : np.ndarray
            Training labels
        X_val : Union[csr_matrix, np.ndarray], optional
            Validation features for early stopping
        y_val : np.ndarray, optional
            Validation labels for early stopping
            
        Returns:
        --------
        self
        """
        # Convert to sparse matrix if needed
        if not issparse(X):
            X = csr_matrix(X)
        
        # Ensure y is binary
        if len(np.unique(y)) != 2:
            raise ValueError("Target variable must be binary (0 and 1)")
        
        # Initialize parameters
        self._initialize_parameters(X.shape[1])
        
        # Reset history
        self.cost_history = []
        self.accuracy_history = []
        
        # Training loop
        best_cost = float('inf')
        patience_counter = 0
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iterations):
            # Shuffle data for each epoch
            X_shuffled, y_shuffled = shuffle(X, y, random_state=self.random_state)
            # Convert y to numpy array to avoid pandas indexing issues
            y_shuffled = np.array(y_shuffled)
            
            # Mini-batch gradient descent
            total_cost = 0
            total_accuracy = 0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                # Get mini-batch
                end_idx = min(i + self.batch_size, n_samples)
                batch_indices = np.arange(i, end_idx)
                X_batch, y_batch = self._get_mini_batch(X_shuffled, y_shuffled, batch_indices)
                
                # Compute gradients
                dw, db = self._compute_gradients_sparse(X_batch, y_batch)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Compute cost for this batch
                batch_cost = self._compute_cost_sparse(X_batch, y_batch)
                total_cost += batch_cost
                
                # Compute accuracy for this batch
                batch_predictions = self.predict(X_batch)
                batch_accuracy = np.mean(batch_predictions == y_batch)
                total_accuracy += batch_accuracy
                
                n_batches += 1
            
            # Average cost and accuracy for this epoch
            avg_cost = total_cost / n_batches
            avg_accuracy = total_accuracy / n_batches
            
            self.cost_history.append(avg_cost)
            self.accuracy_history.append(avg_accuracy)
            
            # Early stopping
            if self.early_stopping:
                if avg_cost < best_cost - self.tolerance:
                    best_cost = avg_cost
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:  # Stop if no improvement for 10 iterations
                    if self.verbose:
                        print(f"Early stopping at iteration {iteration}")
                    break
            
            # Print progress every 50 iterations
            if self.verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: Cost = {avg_cost:.6f}, Accuracy = {avg_accuracy:.4f}")
        
        return self
    
    def predict_proba(self, X: Union[csr_matrix, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : Union[csr_matrix, np.ndarray]
            Feature matrix (sparse or dense)
            
        Returns:
        --------
        np.ndarray
            Probability of positive class
        """
        if not issparse(X):
            X = csr_matrix(X)
        
        z = X.dot(self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X: Union[csr_matrix, np.ndarray]) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : Union[csr_matrix, np.ndarray]
            Feature matrix (sparse or dense)
            
        Returns:
        --------
        np.ndarray
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def score(self, X: Union[csr_matrix, np.ndarray], y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Parameters:
        -----------
        X : Union[csr_matrix, np.ndarray]
            Feature matrix (sparse or dense)
        y : np.ndarray
            True labels
            
        Returns:
        --------
        float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)):
        """
        Plot training history (cost and accuracy).
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot cost
        ax1.plot(self.cost_history)
        ax1.set_title('Training Cost')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.accuracy_history)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_parameters(self) -> dict:
        """
        Get model parameters.
        
        Returns:
        --------
        dict
            Model parameters
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'lambda_reg': self.lambda_reg
        }
    
    def save_model(self, filepath: str):
        """
        Save model parameters to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        import joblib
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'lambda_reg': self.lambda_reg,
            'cost_history': self.cost_history,
            'accuracy_history': self.accuracy_history
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """
        Load model parameters from file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
        """
        import joblib
        model_data = joblib.load(filepath)
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.learning_rate = model_data['learning_rate']
        self.regularization = model_data['regularization']
        self.lambda_reg = model_data['lambda_reg']
        self.cost_history = model_data['cost_history']
        self.accuracy_history = model_data['accuracy_history']
