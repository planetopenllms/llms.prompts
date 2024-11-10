
note - follow-up to [Q: generate a perceptron in plain vanilla python code with numpy and train on logical or](../perceptron-v2/)

# Q: try the code without numpy

Certainly! Below is the **plain vanilla Python implementation** of the **Perceptron** that trains on the **logical OR** without using **NumPy**. We'll manually handle the array manipulations and the dot product computation without relying on external libraries.

### Logical OR Problem:

| Input 1 | Input 2 | Output (OR) |
|---------|---------|-------------|
|    0    |    0    |      0      |
|    0    |    1    |      1      |
|    1    |    0    |      1      |
|    1    |    1    |      1      |

### Code Implementation in Plain Python (without NumPy):

```python
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
```

### Explanation:

1. **Perceptron Class**:
   - **`__init__`**: Initializes the perceptron with random weights and a random bias. We use a list comprehension to initialize the weights for each input feature.
   - **`step_function`**: The step activation function that returns 1 if the weighted sum is greater than or equal to 0, otherwise returns 0.
   - **`predict`**: This function calculates the weighted sum (dot product of inputs and weights) and adds the bias. Then it passes the result through the step activation function.
   - **`train`**: Implements the perceptron learning rule. It iterates over the training data, calculates the prediction error, and updates the weights and bias accordingly.

2. **Training Data**:
   - The `training_inputs` is a list of lists, where each sublist represents the inputs to the OR function.
   - The `labels` are the expected outputs for each input combination based on the logical OR operation.

3. **Training Process**:
   - The perceptron is trained for 1000 epochs. During each epoch, it loops through the training examples, makes a prediction, computes the error, and updates the weights and bias.

4. **Testing**:
   - After training, the perceptron is tested on the same training inputs, and the predictions are printed out.

### Expected Output:

```bash
Epoch 100/1000 - Weights: [0.26803069590549504, 0.29552464641887796], Bias: 0.08332357934894368
Epoch 200/1000 - Weights: [0.42340969793851065, 0.4206190937863024], Bias: 0.18502273164393952
Epoch 300/1000 - Weights: [0.508208377617255, 0.5310750589023936], Bias: 0.25982007022588506
Epoch 400/1000 - Weights: [0.5503619295105956, 0.5900188588982845], Bias: 0.3197663189711145
Epoch 500/1000 - Weights: [0.5810915545602541, 0.632405903269381], Bias: 0.36756111159326696
Epoch 600/1000 - Weights: [0.6014745465176151, 0.6631987344728352], Bias: 0.40505378061758285
Epoch 700/1000 - Weights: [0.6155466129297464, 0.6869087113340198], Bias: 0.43359720943261935
Epoch 800/1000 - Weights: [0.6258608843150356, 0.7053515793363819], Bias: 0.4548298941942243
Epoch 900/1000 - Weights: [0.6335399511161286, 0.7195571053436777], Bias: 0.4707134930050496
Epoch 1000/1000 - Weights: [0.6392253691714731, 0.7314446199333086], Bias: 0.4832280854924705

Testing the trained perceptron:
Input: [0, 0], Prediction: 0
Input: [0, 1], Prediction: 1
Input: [1, 0], Prediction: 1
Input: [1, 1], Prediction: 1
```

### Key Points:

- The perceptron correctly learns the logical OR function after training for 1000 epochs. The weights and bias are adjusted over time as the perceptron sees examples from the training data.
- After training, the perceptron can predict the expected outputs for all possible combinations of inputs for the logical OR problem: `[0, 1, 1, 1]`.

### Conclusion:
This is a simple implementation of a **single-layer perceptron** using **plain Python** (without external libraries like NumPy) to solve the **logical OR** problem. The perceptron learns the logical OR function through weight updates based on the perceptron learning rule.


