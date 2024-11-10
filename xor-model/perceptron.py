import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model with 3 hidden neurons
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # Input layer: 2 inputs -> 3 hidden neurons
        self.fc1 = nn.Linear(2, 3)
        # Output layer: 3 hidden neurons -> 1 output neuron
        self.fc2 = nn.Linear(3, 1)
        # Sigmoid activation function for both layers
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))  # Apply first layer + Sigmoid
        x = self.sigmoid(self.fc2(x))  # Apply second layer + Sigmoid
        return x

# Prepare the XOR dataset (4 examples, each with 2 inputs)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # XOR outputs

# Initialize the model, loss function, and optimizer
model = XORModel()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Use SGD optimizer

# Train the model
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

# Test the model
with torch.no_grad():
    predicted = model(X)
    predicted = (predicted > 0.5).float()  # Convert output probabilities to 0 or 1
    print("\nPredictions:")
    print(predicted)
