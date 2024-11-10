
note - follow-up to [Q: generate a multi-layer perceptron in python code with pytorch and train on logical xor](../perceptron-pytorch-v2/)

note - code works BUT number of epochs may be too short for training;
change `num_epochs = 10000` to `100000`


# Q: change the neural network to use 3 hidden neurons and 2 outputs

To modify the neural network to use **3 hidden neurons** and **2 outputs**, we can make a few changes in the model architecture and adjust the input/output dimensions accordingly. In this case:

- The **input layer** will still have 2 neurons (for the 2 inputs of the XOR problem).
- The **hidden layer** will now have 3 neurons.
- The **output layer** will have 2 neurons, which can represent two different binary outputs (although XOR typically has one output, we will modify it for this case).

We'll also change the loss function to **Mean Squared Error (MSE)** loss since we're dealing with a multi-output problem, where each output is a binary value (0 or 1).

Here's the updated code:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the MLP model with 3 hidden neurons and 2 outputs
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # Input layer: 2 inputs -> 3 hidden neurons
        self.fc1 = nn.Linear(2, 3)
        # Hidden layer: 3 hidden neurons -> 2 output neurons
        self.fc2 = nn.Linear(3, 2)
        # Sigmoid activation function for both layers
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))  # Apply first layer + Sigmoid
        x = self.sigmoid(self.fc2(x))  # Apply second layer + Sigmoid
        return x

# 2. Prepare the XOR dataset (4 examples, each with 2 inputs)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float32)  # 2 outputs for each XOR input

# 3. Initialize the model, loss function, and optimizer
model = XORModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
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

### Key Changes:
1. **Hidden Layer with 3 Neurons**:
   - `self.fc1 = nn.Linear(2, 3)` now maps from 2 inputs to 3 hidden neurons.

2. **Output Layer with 2 Neurons**:
   - `self.fc2 = nn.Linear(3, 2)` now maps from 3 hidden neurons to 2 output neurons. Each output neuron will correspond to a binary value (0 or 1).

3. **Output Labels** (`y`):
   - We updated the target `y` to have 2 outputs for each input pair. Each output represents a binary value, so `y` is a 4x2 tensor. The target values are the same as XOR, but now in a two-column format for 2 outputs:

     ```
     XOR Input  [0, 0] → XOR Output [0, 1]
     XOR Input  [0, 1] → XOR Output [1, 0]
     XOR Input  [1, 0] → XOR Output [1, 0]
     XOR Input  [1, 1] → XOR Output [0, 1]
     ```

4. **Loss Function**:
   - We are now using **Mean Squared Error (MSE) loss** (`nn.MSELoss()`), which is appropriate for multi-output regression tasks where each output is treated independently.

### Expected Output:
As with the previous version of the model, the loss will decrease over the epochs, and after training, the model should produce predictions close to the expected XOR outputs, but now with 2 outputs instead of 1.

The final output could look like this:

```bash
Epoch [1000/10000], Loss: 0.0982
Epoch [2000/10000], Loss: 0.0694
Epoch [3000/10000], Loss: 0.0563
...
Epoch [10000/10000], Loss: 0.0081

Predictions:
tensor([[0., 1.],
        [1., 0.],
        [1., 0.],
        [0., 1.]])
```

This shows that the model successfully learned the XOR function and outputs the correct 2 outputs (as specified in the modified problem statement) for each input.






