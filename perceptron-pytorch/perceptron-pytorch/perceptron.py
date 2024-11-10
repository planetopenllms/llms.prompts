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

