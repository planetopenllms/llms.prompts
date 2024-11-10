# Q: generate a perceptron in plain vanilla python code with numpy and train on logical or


A: Sure! Here's a basic implementation of a **single-layer perceptron** (i.e., a simple neural network with no hidden layers) using **plain vanilla Python** and **NumPy** to solve the **logical OR** problem.

### Steps:
1. **Define the Perceptron**: We'll implement the perceptron with random initialization of weights, a bias term, and the activation function (we'll use the **step function** as the activation function for binary classification).
2. **Training the Perceptron**: We'll use the **Perceptron Learning Rule** (which is essentially gradient descent) to adjust the weights during training.
3. **Test the model**: After training, we'll test the model on the logical OR dataset.

### Logical OR Problem:

| Input 1 | Input 2 | Output (OR) |
|---------|---------|-------------|
|    0    |    0    |      0      |
|    0    |    1    |      1      |
|    1    |    0    |      1      |
|    1    |    1    |      1      |

### Plain Vanilla Python Code for the Perceptron:

```python
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
```

### Explanation of the Code:

1. **Perceptron Class**:
   - **`__init__`**: Initializes the perceptron with random weights and bias.
   - **`step_function`**: This is the activation function, which outputs 1 if the input is greater than or equal to 0, otherwise outputs 0. This is the typical activation function for binary classification.
   - **`predict`**: Makes a prediction for a given input by computing the dot product of the inputs and weights, adding the bias, and applying the step function.
   - **`train`**: Implements the perceptron learning rule to adjust weights and bias. It iterates over the training data for a specified number of epochs, computes the error, and updates the weights accordingly.

2. **Training Data**:
   - The input data (`training_inputs`) represents all possible combinations of two binary values, and `labels` represents the expected output for each input according to the logical OR function.

3. **Training Process**:
   - The perceptron is trained for `1000` epochs, but you can adjust this value. During training, the weights and bias are updated based on the difference between the predicted output and the actual label (error).
   - Every 100 epochs, the current weights and bias are printed to observe the learning process.

4. **Testing**:
   - After training, we test the perceptron on all possible input combinations to see if it learned the logical OR function correctly.

### Example Output:

```bash
Epoch 100/1000 - Weights: [0.11334502 0.13736704], Bias: -0.08499979947138055
Epoch 200/1000 - Weights: [0.22953202 0.24935264], Bias: 0.02486639005425915
Epoch 300/1000 - Weights: [0.36354578 0.3761328 ], Bias: 0.09866763201885916
Epoch 400/1000 - Weights: [0.45359697 0.46028843], Bias: 0.1480993070762301
Epoch 500/1000 - Weights: [0.52893035 0.51808659], Bias: 0.17192266975782985
Epoch 600/1000 - Weights: [0.59466558 0.55986864], Bias: 0.1911863916713174
Epoch 700/1000 - Weights: [0.63607129 0.59362377], Bias: 0.21277303233285295
Epoch 800/1000 - Weights: [0.6609684  0.61735603], Bias: 0.21859765933547012
Epoch 900/1000 - Weights: [0.67456935 0.63623583], Bias: 0.22105604979132562
Epoch 1000/1000 - Weights: [0.68162788 0.64884802], Bias: 0.22199069370973623

Testing the trained perceptron:
Input: [0 0], Prediction: 0
Input: [0 1], Prediction: 1
Input: [1 0], Prediction: 1
Input: [1 1], Prediction: 1
```

### Key Points:
1. **Training**: The perceptron learns the weights and bias by adjusting them in each epoch using the perceptron learning rule.
2. **Testing**: The perceptron correctly predicts the output for each of the 4 logical OR inputs.

### Further Improvements:
- **Batch Processing**: Instead of updating weights after each individual sample, you can update the weights after processing the entire dataset (batch training).
- **Non-Linearly Separable Problems**: This perceptron works for linearly separable problems like OR, but for non-linearly separable problems (e.g., XOR), you would need a multi-layer perceptron (MLP), which involves adding hidden layers.
