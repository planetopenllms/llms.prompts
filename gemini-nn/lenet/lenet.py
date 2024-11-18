import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, activation=nn.ReLU, num_classes=10):
        super(LeNet, self).__init__()
        self.activation = activation() # Allows easy swapping of activation functions

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            self.activation,
            nn.Linear(120, 84),
            self.activation,
            nn.Linear(84, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Example usage:
# LeNet with ReLU activation and 10 output classes (for MNIST)
model_relu = LeNet(activation=nn.ReLU)
print("LeNet with ReLU:")
print(model_relu)

# LeNet with Sigmoid activation and 10 output classes
model_sigmoid = LeNet(activation=nn.Sigmoid)
print("\nLeNet with Sigmoid:")
print(model_sigmoid)


#LeNet with Tanh activation and 10 output classes
model_tanh = LeNet(activation=nn.Tanh)
print("\nLeNet with Tanh:")
print(model_tanh)


# Example input to check the output shape
input_tensor = torch.randn(1, 1, 32, 32) # Batch size 1, 1 input channel, 32x32 image
output_relu = model_relu(input_tensor)
print("\nOutput shape with ReLU:", output_relu.shape)

output_sigmoid = model_sigmoid(input_tensor)
print("\nOutput shape with Sigmoid:", output_sigmoid.shape)

output_tanh = model_tanh(input_tensor)
print("\nOutput shape with Tanh:", output_tanh.shape)

