import numpy as np

# Define the Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_size)  # Random initialization of weights
        self.bias = np.random.randn()  # Random initialization of bias

    def step_function(self, x):
        """ Step activation function """
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        """ Make a prediction for a single input """
        total_input = np.dot(inputs, self.weights) + self.bias
        return self.step_function(total_input)

    def train(self, training_inputs, labels, epochs=100):
        """ Train the perceptron using the perceptron learning rule """
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                # Update weights and bias based on the error
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
            # Optional: print the error during training
            if (epoch+1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs} - Weights: {self.weights}, Bias: {self.bias}')

# Define the training data for logical OR
training_inputs = np.array([
    [0, 0],  # Input: [0, 0]
    [0, 1],  # Input: [0, 1]
    [1, 0],  # Input: [1, 0]
    [1, 1]   # Input: [1, 1]
])

labels = np.array([0, 1, 1, 1])  # OR outputs: [0, 1, 1, 1]

# Initialize the perceptron
perceptron = Perceptron(input_size=2)

# Train the perceptron on the logical OR data
perceptron.train(training_inputs, labels, epochs=1000)

# Test the trained perceptron
print("\nTesting the trained perceptron:")
for inputs in training_inputs:
    prediction = perceptron.predict(inputs)
    print(f"Input: {inputs}, Prediction: {prediction}")
