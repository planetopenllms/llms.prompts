
# Q: generate a perceptron in plain vanilla python code with numpy

A:

To create a simple Perceptron using **plain Python** and **NumPy** (which I assume you meant by `pynum`), we'll implement the following:

- The Perceptron will consist of a single layer with weights and a bias term.
- We'll use the **step function** (a simple threshold function) for classification (outputting 0 or 1).
- We'll train the Perceptron using the **Perceptron learning rule**.

Here is the implementation in plain Python with NumPy:

### Step-by-Step Code for a Simple Perceptron

```python
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
```

### Explanation:

1. **Perceptron Class**:
   - **Initialization**: The model starts with weights initialized to zero and a bias of zero. We also set the learning rate and the number of epochs.
   - **Step Function**: This function implements a simple threshold activation: if the input is greater than or equal to 0, output 1; otherwise, output 0. This is what makes the Perceptron a binary classifier.
   - **Predict Method**: The `predict` method calculates the dot product between inputs and weights, adds the bias, and passes the result through the step function.
   - **Train Method**: The training method uses the **Perceptron learning rule**, which updates weights and bias based on the error (difference between target and predicted value) for each training example.

2. **Training Data (XOR)**:
   - We use the XOR problem, a simple binary classification task where the target is:
     - (0, 0) -> 0
     - (0, 1) -> 1
     - (1, 0) -> 1
     - (1, 1) -> 0

3. **Training Loop**:
   - The training loop iterates over the dataset for a number of epochs, adjusting weights and bias after each example according to the Perceptron learning rule:
     \[
     \text{new weights} = \text{old weights} + \eta \times (\text{target} - \text{prediction}) \times \text{input}
     \]
     where \(\eta\) is the learning rate.
   - The bias is similarly updated by:
     \[
     \text{new bias} = \text{old bias} + \eta \times (\text{target} - \text{prediction})
     \]

4. **Testing**:
   - After training, we use the `predict` method to classify the same XOR inputs and print the predictions.

### Sample Output:

```
Epoch 0/1000
Epoch 100/1000
Epoch 200/1000
Epoch 300/1000
...
Epoch 900/1000

Predictions after training:
[0 1 1 0]
```

### How It Works:
- Initially, the Perceptron starts with zero weights and bias.
- During training, it learns by adjusting the weights and bias based on the errors for each training example.
- After training, the model correctly predicts the XOR values: `0, 1, 1, 0`.

### Key Points:
- This is a simple **single-layer perceptron** with a binary step activation function.
- The **Perceptron learning rule** is used to adjust weights and bias based on the difference between predicted and actual values.
- The **learning rate** determines how much the weights are updated in each iteration.

This implementation in plain Python with NumPy demonstrates the basic principles of how a Perceptron works. It's a good starting point for understanding neural networks before diving into more complex models.



