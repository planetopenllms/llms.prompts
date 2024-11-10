import random

# Define the Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights and bias with small random values
        self.weights = [random.random() for _ in range(input_size)]  # Random weights for each input
        self.bias = random.random()  # Random bias term
        self.learning_rate = learning_rate  # Learning rate

    def step_function(self, x):
        """ Step activation function: Outputs 1 if x >= 0 else 0 """
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        """ Predict the output for given inputs """
        total_input = sum(weight * input_value for weight, input_value in zip(self.weights, inputs)) + self.bias
        return self.step_function(total_input)

    def train(self, training_inputs, labels, epochs=1000):
        """ Train the perceptron using the perceptron learning rule """
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                # Update weights and bias based on the error
                self.weights = [weight + self.learning_rate * error * input_value for weight, input_value in zip(self.weights, inputs)]
                self.bias += self.learning_rate * error
            # Optionally, print the error after every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs} - Weights: {self.weights}, Bias: {self.bias}')

# Define the training data for logical OR
training_inputs = [
    [0, 0],  # Input: [0, 0]
    [0, 1],  # Input: [0, 1]
    [1, 0],  # Input: [1, 0]
    [1, 1]   # Input: [1, 1]
]

labels = [0, 1, 1, 1]  # Output labels for OR operation: [0, 1, 1, 1]

# Initialize the perceptron
perceptron = Perceptron(input_size=2)  # Two inputs for logical OR

# Train the perceptron on the logical OR data
perceptron.train(training_inputs, labels, epochs=1000)

# Test the trained perceptron
print("\nTesting the trained perceptron:")
for inputs in training_inputs:
    prediction = perceptron.predict(inputs)
    print(f"Input: {inputs}, Prediction: {prediction}")
