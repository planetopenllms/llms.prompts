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
