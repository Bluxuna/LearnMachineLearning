import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias to zero

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent optimization
        for _ in range(self.number_of_iterations):


            y_prediction = np.dot(X, self.weights) + self.bias

            # Calculate gradients

            dw = (1 / n_samples) * np.dot(X.T, (y_prediction - y))
            db = (1 / n_samples) * np.sum(y_prediction - y)

            # Update parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        y_prediction = np.dot(X, self.weights) + self.bias
        return y_prediction