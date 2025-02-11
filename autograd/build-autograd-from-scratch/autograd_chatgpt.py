import numpy as np

# Basic Custom Tensor Class
class MyTensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)  # Store the data as a NumPy array
        self.requires_grad = requires_grad  # If this tensor should track gradients
        self.grad = None  # Gradient is initially None
        self.shape = self.data.shape  # Shape of the tensor
        
    def __repr__(self):
        return f"MyTensor(data={self.data}, requires_grad={self.requires_grad})"
    
    def zero_grad(self):
        """Reset gradients to None"""
        self.grad = None
    
    def backward(self, grad=None):
        """Backward pass to compute gradients"""
        if grad is None:
            # If no grad is provided, assume it's 1 (useful for scalar output)
            grad = np.ones_like(self.data)
        
        if self.requires_grad:
            # Assign the gradient for this tensor
            self.grad = grad

# Define some operations to make it more interesting (e.g., addition, multiplication)

def add(tensor1, tensor2):
    """Addition operation"""
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape for addition")
    
    result = MyTensor(tensor1.data + tensor2.data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    # If the result requires gradients, we define the backward pass
    if result.requires_grad:
        def backward():
            if tensor1.requires_grad:
                tensor1.backward(grad=result.grad)
            if tensor2.requires_grad:
                tensor2.backward(grad=result.grad)
        result.backward = backward
    return result

def multiply(tensor1, tensor2):
    """Multiplication operation"""
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape for multiplication")
    
    result = MyTensor(tensor1.data * tensor2.data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    # If the result requires gradients, we define the backward pass
    if result.requires_grad:
        def backward():
            if tensor1.requires_grad:
                tensor1.backward(grad=result.grad * tensor2.data)
            if tensor2.requires_grad:
                tensor2.backward(grad=result.grad * tensor1.data)
        result.backward = backward
    return result

# Example usage

# Create Tensors
a = MyTensor([2.0, 3.0], requires_grad=True)
b = MyTensor([4.0, 5.0], requires_grad=True)

# Perform some operations
c = add(a, b)  # c = a + b
d = multiply(a, b)  # d = a * b

# Perform backward
c.grad = np.ones_like(c.data)  # Let's assume an upstream gradient of 1 for simplicity
c.backward()

d.grad = np.ones_like(d.data)  # Let's assume an upstream gradient of 1 for simplicity
d.backward()

# Print out the results
print("Tensor a:", a)
print("Tensor b:", b)
print("Result c (a + b):", c)
print("Result d (a * b):", d)

print("Gradient of a:", a.grad)
print("Gradient of b:", b.grad)


