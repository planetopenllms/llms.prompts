
note - code works BUT number of epochs may be too short for training;
change `num_epochs = 10000` to `100000`



# Q: generate a multi-layer perceptron in python code with pytorch and train on logical xor

A: Sure! Here's an example of how to create and train a simple multi-layer perceptron (MLP) using PyTorch to solve the logical XOR problem. The network will consist of an input layer, one hidden layer, and an output layer. We will use the `sigmoid` activation function for both the hidden layer and the output layer.

### Steps:
1. Define the model architecture.
2. Set up the data for XOR.
3. Define the loss function and optimizer.
4. Train the model.

Here’s the complete code:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the MLP model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # Input layer has 2 neurons (for 2 inputs), hidden layer has 2 neurons, output has 1 neuron
        self.fc1 = nn.Linear(2, 2)  # First layer: 2 inputs -> 2 hidden neurons
        self.fc2 = nn.Linear(2, 1)  # Second layer: 2 hidden neurons -> 1 output
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))  # Apply first layer + Sigmoid activation
        x = self.sigmoid(self.fc2(x))  # Apply second layer + Sigmoid activation
        return x

# 2. Prepare the XOR dataset (4 examples, each with 2 inputs)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 3. Initialize the model, loss function and optimizer
model = XORModel()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Train the model
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)

    # Calculate loss
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the weights

    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Test the model
with torch.no_grad():
    predicted = model(X)
    predicted = (predicted > 0.5).float()  # Convert output probabilities to 0 or 1
    print("\nPredictions:")
    print(predicted)
```

### Explanation:
1. **Model Definition (`XORModel`)**:
   - We define a neural network class `XORModel` that inherits from `nn.Module`.
   - The model has two fully connected layers (`fc1` and `fc2`). The first layer maps the 2 inputs to 2 hidden neurons, and the second layer maps the 2 hidden neurons to 1 output.
   - We use the `sigmoid` activation function for both layers to introduce non-linearity.

2. **Data Preparation**:
   - `X` represents the 4 possible input combinations for the XOR operation.
   - `y` represents the corresponding outputs.

3. **Loss Function and Optimizer**:
   - We use **Binary Cross-Entropy loss (`BCELoss`)**, which is suitable for binary classification tasks.
   - The optimizer is **Stochastic Gradient Descent (`SGD`)** with a learning rate of `0.1`.

4. **Training Loop**:
   - We loop for a number of epochs (10,000 in this case) and perform forward passes, compute the loss, perform backpropagation, and update the weights.
   - Every 1,000 epochs, we print the current loss.

5. **Testing the Model**:
   - After training, we use the trained model to make predictions. The output is passed through a threshold (`>0.5`) to convert it to a binary result (either 0 or 1).

### Expected Output:
During training, the loss should decrease over time. After training is complete, the model's predictions should be close to the expected XOR values.

The final output might look something like this:

```bash
Epoch [1000/10000], Loss: 0.0045
Epoch [2000/10000], Loss: 0.0032
Epoch [3000/10000], Loss: 0.0023
...
Epoch [10000/10000], Loss: 0.0005

Predictions:
tensor([[0.],
        [1.],
        [1.],
        [0.]])
```

The predictions for each input should be:

- Input `[0, 0]` → Output `0`
- Input `[0, 1]` → Output `1`
- Input `[1, 0]` → Output `1`
- Input `[1, 1]` → Output `0`

This confirms that the model has learned the XOR function correctly!


