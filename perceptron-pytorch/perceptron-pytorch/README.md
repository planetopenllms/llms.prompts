note - xor not possible with single-layer perceptron, change to train on logical and or or.  see [Q: explain if you can train with success a single-layer perceptron on logical xor](../../theory/non-linear/)


# Q: generate a perceptron in python code with pytorch


To implement a **Perceptron** using PyTorch, we can follow a simple approach that includes defining the model, specifying the loss function, and using an optimizer for training. A Perceptron is essentially a single-layer neural network, and in this case, we'll build a model for binary classification.

Here is an example of how to implement a Perceptron using PyTorch:

### Step-by-Step Code for Perceptron in PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Perceptron class
class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super(Perceptron, self).__init__()
        # A single fully connected layer
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Apply a sigmoid activation to the output of the linear layer
        return torch.sigmoid(self.fc(x))

# Define the input and output sizes
input_size = 2  # Two inputs (for example, XOR problem)
output_size = 1  # One output (binary classification)

# Initialize the Perceptron model
model = Perceptron(input_size, output_size)

# Define the loss function (Binary Cross-Entropy Loss)
criterion = nn.BCELoss()

# Define the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training data for the XOR problem
X_train = torch.tensor([[0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0],
                       [1.0, 1.0]], dtype=torch.float32)

# Target values for XOR (0, 1, 1, 0)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# Training the model
epochs = 1000
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing X to the model
    y_pred = model(X_train)

    # Compute the loss
    loss = criterion(y_pred, y_train)

    # Zero gradients, perform backward pass, and update weights
    optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update parameters

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Testing the trained model
with torch.no_grad():
    y_test = model(X_train)
    predicted = y_test.round()  # Round the predictions to either 0 or 1
    print("\nPredictions after training:")
    print(predicted)

```

### Explanation of Code:

1. **Model Definition**:
   - We define a simple Perceptron model using the `nn.Linear` layer. This layer takes input of size `input_size` and outputs a value of size `output_size`.
   - The activation function used is **Sigmoid**, which squashes the output to a value between 0 and 1, suitable for binary classification.

2. **Training Data (XOR Problem)**:
   - We use the XOR problem as the training dataset. The XOR operation is a classic example used to train a simple neural network. The four input-output pairs for XOR are:
     - (0, 0) -> 0
     - (0, 1) -> 1
     - (1, 0) -> 1
     - (1, 1) -> 0

3. **Loss Function**:
   - We use **Binary Cross-Entropy Loss (`BCELoss`)**, which is common for binary classification tasks.

4. **Optimizer**:
   - The optimizer used is **Stochastic Gradient Descent (SGD)**, which updates the model's weights using backpropagation.

5. **Training Loop**:
   - For each epoch, we perform the forward pass (compute predictions), calculate the loss, and then perform backpropagation to update the model's parameters.

6. **Predictions**:
   - After training, we test the model on the same training data and print the rounded predictions to either 0 or 1.

### Sample Output:

```
Epoch [100/1000], Loss: 0.6812
Epoch [200/1000], Loss: 0.5292
Epoch [300/1000], Loss: 0.3835
...
Epoch [1000/1000], Loss: 0.0025

Predictions after training:
tensor([[0.],
        [1.],
        [1.],
        [0.]])
```

In this output:
- The loss decreases over the epochs as the model learns.
- After training, the model correctly predicts the XOR outputs: `0`, `1`, `1`, `0`.

### Key Points:
- This Perceptron is a simple **single-layer neural network**.
- We used **binary cross-entropy** loss for binary classification tasks.
- The training was done with **SGD**, a basic optimizer.

This is a basic implementation of a Perceptron. More complex networks (like multi-layer perceptrons or deeper architectures) can be implemented in a similar fashion, with additional layers and different activation functions.
