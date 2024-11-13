import numpy as np

# Define the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Define the MLP architecture
n_hidden = 2  # Number of neurons in the hidden layer
n_inputs = X.shape[1]  # Number of input features
n_outputs = 1  # Number of output neurons

# Initialize weights and biases randomly
W1 = np.random.randn(n_inputs, n_hidden)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_outputs)
b2 = np.zeros((1, n_outputs))

# Define activation functions
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return sigmoid(x) * (1 - sigmoid(x))

# Define the training function
def train(X, y, epochs=10000, learning_rate=0.1):
  for epoch in range(epochs):
    # Forward propagation
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # Calculate the error
    error = y - a2

    # Backpropagation
    delta2 = error * sigmoid_derivative(a2)
    delta1 = delta2 @ W2.T * sigmoid_derivative(a1)

    # Update weights and biases
    W2 += learning_rate * a1.T @ delta2
    b2 += learning_rate * np.sum(delta2, axis=0, keepdims=True)
    W1 += learning_rate * X.T @ delta1
    b1 += learning_rate * np.sum(delta1, axis=0, keepdims=True)

    # Print the error every 100 epochs
    if epoch % 100 == 0:
      print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Train the MLP
train(X, y)

# Test the trained MLP
def predict(X):
  z1 = X @ W1 + b1
  a1 = sigmoid(z1)
  z2 = a1 @ W2 + b2
  a2 = sigmoid(z2)
  return np.round(a2)

# Predict the output for the XOR dataset
predictions = predict(X)
print("Predictions:", predictions)
print("Actual Values:", y)

