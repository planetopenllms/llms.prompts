import numpy as np

# Configuration
vocab_size = 10  # Vocabulary size
embedding_dim = 5  # Dimension of word embeddings
hidden_dim = 10  # Number of hidden units
output_dim = 2  # Number of output classes
batch_size = 4  # Number of sentences in a batch
learning_rate = 0.01  # Learning rate

# Sample data (word indices in a batch of 4 sentences)
input_data = np.array([[1, 2, 3], 
                       [4, 5, 6], 
                       [7, 8, 9], 
                       [1, 5, 9]])  # 4 sentences, each with 3 words
labels = np.array([0, 1, 0, 1])  # Labels for binary classification

# Initialize weights (embedding matrix, hidden layer weights, output layer weights)
embedding = np.random.randn(vocab_size, embedding_dim)  # Embedding matrix (vocab_size, embedding_dim)
W1 = np.random.randn(embedding_dim, hidden_dim)  # Hidden layer weights
b1 = np.zeros((1, hidden_dim))  # Hidden layer biases
W2 = np.random.randn(hidden_dim, output_dim)  # Output layer weights
b2 = np.zeros((1, output_dim))  # Output layer biases

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick for overflow
    return e_x / e_x.sum(axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    p = softmax(y_pred)
    log_likelihood = -np.log(p[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss

# Forward pass
def forward(input_data):
    # Get the embeddings for the input words
    embedded = embedding[input_data]  # Shape: (batch_size, seq_len, embedding_dim)
    embedded = embedded.mean(axis=1)  # Average embeddings over words (for simplicity)

    # Pass through the hidden layer
    z1 = np.dot(embedded, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU activation

    # Output layer
    z2 = np.dot(a1, W2) + b2
    return z2, a1  # Return logits and activations

# Backpropagation
def backward(input_data, a1, output, labels):
    m = labels.shape[0]

    # Compute the gradient of the loss w.r.t. the output layer
    y_true = np.zeros_like(output)
    y_true[range(m), labels] = 1  # One-hot encoded labels
    dL_dz2 = (softmax(output) - y_true) / m  # Gradient of loss w.r.t. z2 (output logits)
    print( "==> dL_dz2\n", 
           dL_dz2, "shape=", dL_dz2.shape )

    # Backpropagate to the hidden layer
    dL_dW2 = np.dot(a1.T, dL_dz2)  # Gradient of loss w.r.t. W2
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # Gradient of loss w.r.t. b2

    # Backpropagate further to the hidden layer (ReLU derivative)
    dL_da1 = np.dot(dL_dz2, W2.T)  # Gradient of loss w.r.t. a1
    dL_dz1 = dL_da1 * (a1 > 0)  # ReLU derivative (1 where a1 > 0, else 0)
    print( "==> dL_dz1\n", 
           dL_dz1, "shape=", dL_dz1.shape )

    # Update hidden layer weights
    dL_dW1 = np.dot(embedding[input_data].mean(axis=1).T, dL_dz1)  # Gradient w.r.t. W1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)  # Gradient w.r.t. b1

    # Update the embedding weights 
    # (this is where the backpropagation through embeddings happens)
    dL_dembedding = np.dot(dL_dz1, W1.T)  # Gradient w.r.t. embedding layer

    return dL_dembedding, dL_dW1, dL_db1, dL_dW2, dL_db2

# Training loop
for epoch in range(10):  # Number of epochs
    # Forward pass
    output, a1 = forward(input_data)
    
    # Compute the loss
    loss = cross_entropy_loss(output, labels)
    print(f"Epoch {epoch+1}, Loss: {loss}")

    # Backward pass
    dL_dembedding, dL_dW1, dL_db1, dL_dW2, dL_db2 = backward(input_data, a1, output, labels)
    print( "==> dL_dembedding\n", dL_dembedding, "shape=", dL_dembedding.shape )
    print( "==> embedding\n", embedding, "shape=", embedding.shape )


    # Update weights using gradient descent
    embedding -= learning_rate * dL_dembedding  # Update embedding layer
  #=> error reported here is:
  ##  operands could not be broadcast together with shapes (10,5) (4,5) (10,5)

    W1 -= learning_rate * dL_dW1  # Update hidden layer weights
    b1 -= learning_rate * dL_db1  # Update hidden layer biases
    W2 -= learning_rate * dL_dW2  # Update output layer weights
    b2 -= learning_rate * dL_db2  # Update output layer biases


# Final learned embeddings
print("Learned Embeddings:")
print(embedding)
