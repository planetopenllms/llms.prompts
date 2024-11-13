import numpy as np

def perceptron(X, y, learning_rate=0.1, epochs=100):
  """
  Trains a perceptron model.

  Args:
    X: The training data, a 2D numpy array of shape (n_samples, n_features).
    y: The target labels, a 1D numpy array of shape (n_samples,).
    learning_rate: The learning rate for the gradient descent.
    epochs: The number of training epochs.

  Returns:
    weights: The learned weights of the perceptron.
    bias: The learned bias of the perceptron.
  """

  n_samples, n_features = X.shape
  weights = np.zeros(n_features)
  bias = 0

  for _ in range(epochs):
    for i in range(n_samples):
      # Calculate the prediction
      prediction = np.dot(X[i], weights) + bias
      # Apply activation function (step function)
      output = 1 if prediction >= 0 else 0
      # Update weights and bias if prediction is wrong
      if output != y[i]:
        weights += learning_rate * (y[i] - output) * X[i]
        bias += learning_rate * (y[i] - output)

  return weights, bias

# Define the logical OR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Train the perceptron
weights, bias = perceptron(X, y)

# Test the trained perceptron
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = []
for i in range(len(test_data)):
  prediction = np.dot(test_data[i], weights) + bias
  output = 1 if prediction >= 0 else 0
  predictions.append(output)

print("Predictions:", predictions)
print("Actual:", y)

# Output should be:
# Predictions: [0, 1, 1, 1]
# Actual: [0, 1, 1, 1]
