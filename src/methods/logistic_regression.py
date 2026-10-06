import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, append_bias_term


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr=1e-1, max_iters=500, reg=1e-3):
        """
        Initialize the new object and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
            reg (float): regularization strength (L2 penalty)
        """
        self.lr = lr
        self.max_iters = max_iters
        self.reg = reg

    def compute_class_probabilities(self, scores):
        """
        Compute class probabilities with the "softmax" function.
        
        Arguments:
            scores (array): scores of shape (N, n_classes)
        Returns:
            probs (array): probabilities of shape (N, n_classes)
        """
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): class labels of shape (N,)
        Returns:
            pred_labels (array): predicted class labels of shape (N,)
        """
        # Get number of classes and convert labels to one-hot encoding
        n_classes = get_n_classes(training_labels)
        onehot_labels = label_to_onehot(training_labels, n_classes)
        
        # Input data already includes the bias column.
        N, D = training_data.shape
        
        # Initialize weights (including bias)
        self.W = np.zeros((D, n_classes))
        
        # Initialize scores
        scores = training_data @ self.W
        
        # Gradient descent
        for _ in range(self.max_iters):
            # Forward pass
            scores = training_data @ self.W
            probs = self.compute_class_probabilities(scores)
            
            # Backward pass
            dscores = (probs - onehot_labels)/N
            
            # Compute gradients
            dW = training_data.T @ dscores + self.reg * self.W
            
            # Update parameters
            self.W -= self.lr * dW
        
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # Evaluation data uses the same bias column as training data.
    
        
        # Compute scores using learned weights
        scores = test_data @ self.W
        
        # Convert scores to labels using onehot_to_label
        pred_labels = onehot_to_label(scores)
        
        return pred_labels
