import random
import math

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the Multi-Layer Perceptron class
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases with small random values
        self.weights_input_hidden = [[random.random() for _ in range(hidden_size)] for _ in range(input_size)]  # weights between input and hidden layer
        self.weights_hidden_output = [random.random() for _ in range(hidden_size)]  # weights between hidden and output layer

        self.bias_hidden = [random.random() for _ in range(hidden_size)]  # biases for hidden layer
        self.bias_output = random.random()  # bias for output layer

        self.learning_rate = learning_rate

    def forward(self, X):
        """ Forward pass through the network """
        # Input to hidden layer
        self.hidden_input = [sum(X[i] * self.weights_input_hidden[i][j] for i in range(len(X))) + self.bias_hidden[j] for j in range(len(self.bias_hidden))]
        self.hidden_output = [sigmoid(x) for x in self.hidden_input]

        # Hidden to output layer
        self.output_input = sum(self.hidden_output[j] * self.weights_hidden_output[j] for j in range(len(self.hidden_output))) + self.bias_output
        self.output = sigmoid(self.output_input)

        return self.output

    def backward(self, X, y):
        """ Backward pass (backpropagation) for updating weights and biases """
        # Convert y to a scalar since it's a list with a single value
        y = y[0]  # Extract the scalar value from the list

        # Calculate output layer error
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Calculate hidden layer error
        hidden_error = [output_delta * self.weights_hidden_output[j] for j in range(len(self.weights_hidden_output))]
        hidden_delta = [hidden_error[j] * sigmoid_derivative(self.hidden_output[j]) for j in range(len(self.hidden_output))]

        # Update weights and biases
        # Update weights between hidden and output layer
        for j in range(len(self.weights_hidden_output)):
            self.weights_hidden_output[j] += self.hidden_output[j] * output_delta * self.learning_rate

        # Update bias for output layer
        self.bias_output += output_delta * self.learning_rate

        # Update weights between input and hidden layer
        for i in range(len(X)):
            for j in range(len(self.weights_input_hidden[i])):
                self.weights_input_hidden[i][j] += X[i] * hidden_delta[j] * self.learning_rate

        # Update bias for hidden layer
        for j in range(len(self.bias_hidden)):
            self.bias_hidden[j] += hidden_delta[j] * self.learning_rate

    def train(self, X, y, epochs=10000):
        """ Train the network """
        for epoch in range(epochs):
            for i in range(len(X)):
                self.forward(X[i])  # Forward pass
                self.backward(X[i], y[i])  # Backward pass (weight updates)

            # Print the error every 1000 epochs
            if epoch % 1000 == 0:
                error = sum(abs(y[i][0] - self.forward(X[i])) for i in range(len(X))) / len(X)
                print(f"Epoch {epoch}/{epochs} - Error: {error}")

    def predict(self, X):
        """ Make predictions with the trained network """
        return [self.forward(x) for x in X]

# Define the training data for XOR
X = [
    [0, 0],  # Input: [0, 0]
    [0, 1],  # Input: [0, 1]
    [1, 0],  # Input: [1, 0]
    [1, 1]   # Input: [1, 1]
]

y = [
    [0],  # Output: XOR([0, 0]) = 0
    [1],  # Output: XOR([0, 1]) = 1
    [1],  # Output: XOR([1, 0]) = 1
    [0]   # Output: XOR([1, 1]) = 0
]

# Initialize and train the MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
mlp.train(X, y, epochs=10000)

# Test the trained MLP
print("\nTesting the trained MLP:")
for input_data in X:
    prediction = mlp.predict([input_data])
    print(f"Input: {input_data}, Prediction: {prediction[0]}")
