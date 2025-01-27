import numpy as np

# Define the Hopfield network class
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))  # Weight matrix
        self.state = np.ones(size)  # Initial state of the neurons

    # Training the network with a set of patterns
    def train(self, patterns):
        # Reset weights to zero
        self.weights = np.zeros((self.size, self.size))
        
        for pattern in patterns:
            # Convert pattern to numpy array of +1, -1
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)  # Hebbian learning rule

        # Set the diagonal to 0 to avoid self-feedback
        np.fill_diagonal(self.weights, 0)

    # Update the state of the network
    def update(self):
        for i in range(self.size):
            # Compute the sum of inputs to the neuron
            sum_inputs = np.dot(self.weights[i], self.state)
            # Update the state of the neuron using the sign function
            self.state[i] = 1 if sum_inputs >= 0 else -1

    # Recall a stored pattern by updating the network state
    def recall(self, noisy_pattern, max_iterations=100):
        self.state = np.array(noisy_pattern)
        for _ in range(max_iterations):
            self.update()
        return self.state

# Define the patterns you want to store in the network (binary patterns of +1, -1)
pattern_1 = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1]
pattern_2 = [-1, 1, 1, -1, 1, 1, -1, -1, -1, 1]
patterns = [pattern_1, pattern_2]

# Create a Hopfield network with 10 neurons
hopfield_net = HopfieldNetwork(size=10)

# Train the network with the patterns
hopfield_net.train(patterns)

# Test the network with a noisy pattern
noisy_pattern = [1, 1, -1, -1, 1, 1, -1, -1, -1, 1]  # This is a noisy version of pattern_1

# Recall the pattern from the noisy input
recalled_pattern = hopfield_net.recall(noisy_pattern)

# Display the results
print("Noisy Pattern:")
print(noisy_pattern)
print("\nRecalled Pattern:")
print(recalled_pattern)
