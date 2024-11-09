import numpy as np

# Define the Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        # Initialize weights and bias
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def step_function(self, x):
        # Step function (threshold function)
        return 1 if x >= 0 else 0

    def predict(self, X):
        # Calculate the weighted sum and apply the step function
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.step_function(i) for i in linear_output])

    def train(self, X, y):
        # Training the Perceptron
        for epoch in range(self.epochs):
            for inputs, target in zip(X, y):
                # Compute the model's prediction
                prediction = self.predict(inputs.reshape(1, -1))

                # Update rule: weights and bias
                error = target - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
            if epoch % 100 == 0:
                # Optionally print the loss every 100 epochs
                print(f"Epoch {epoch}/{self.epochs}")

# Training data (XOR Problem)
X_train = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

# Target labels (XOR output)
y_train = np.array([0, 1, 1, 0])

# Create the Perceptron model
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=1000)

# Train the model
perceptron.train(X_train, y_train)

# Test the Perceptron model
print("\nPredictions after training:")
predictions = perceptron.predict(X_train)
print(predictions)
