import numpy as np

class MyTensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)  # Store the data as a NumPy array
        self.requires_grad = requires_grad  # Whether this tensor should track gradients
        self.grad = None  # Gradient is initially None
        self.shape = self.data.shape  # Shape of the tensor
        self.children = []  # List to track the children of this tensor
        self.parents = []  # List to track the parent tensors that depend on this tensor
        
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
        
        # We need to traverse the graph in topological order
        # We will do a depth-first search (DFS) to get the correct order
        visited = set()
        def dfs(tensor):
            if tensor in visited:
                return
            visited.add(tensor)
            for parent in tensor.parents:
                dfs(parent)
            tensor.compute_gradients()

        # Start DFS from the current tensor (and its parents)
        dfs(self)
        
    def add_child(self, child_tensor):
        """Add a child tensor that depends on this tensor"""
        self.children.append(child_tensor)
        child_tensor.parents.append(self)

    def compute_gradients(self):
        """Compute gradients for this tensor based on its children"""
        if self.grad is None:
            return
        for child in self.children:
            # Call the backward method on the child tensor
            child.backward(grad=self.grad)

# Define some operations to make it more interesting (e.g., addition, multiplication)

def add(tensor1, tensor2):
    """Addition operation"""
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape for addition")
    
    result = MyTensor(tensor1.data + tensor2.data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    # If the result requires gradients, we define the backward pass
    if result.requires_grad:
        tensor1.add_child(result)
        tensor2.add_child(result)
        
    return result

def multiply(tensor1, tensor2):
    """Multiplication operation"""
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape for multiplication")
    
    result = MyTensor(tensor1.data * tensor2.data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    # If the result requires gradients, we define the backward pass
    if result.requires_grad:
        tensor1.add_child(result)
        tensor2.add_child(result)
        
    return result

# Example usage

# Create Tensors
a = MyTensor([2.0, 3.0], requires_grad=True)
b = MyTensor([4.0, 5.0], requires_grad=True)

# Perform some operations
c = add(a, b)  # c = a + b
d = multiply(a, b)  # d = a * b

# Perform backward propagation
c.grad = np.ones_like(c.data)  # Let's assume an upstream gradient of 1 for simplicity
c.backward()  # This will propagate the gradients through the graph

# Print out the results
print("Tensor a:", a)
print("Tensor b:", b)
print("Result c (a + b):", c)
print("Result d (a * b):", d)

print("Gradient of a:", a.grad)
print("Gradient of b:", b.grad)
