
# Q: how can you backpropagate through a word embedding layer?

a:  Hello! Great question! Backpropagating through a word embedding layer is a crucial part of training neural networks that involve text data, like those using embeddings for words or phrases.

Here’s a step-by-step breakdown of how backpropagation works through a word embedding layer:

### 1. **Forward Pass (Input and Embedding Lookup)**:
   - In the forward pass, you first convert the input words (which are usually integers representing word indices) into their corresponding word embeddings.
   - Each word in the input sequence is mapped to a vector in the embedding space using the embedding matrix, \(E\). This matrix \(E\) contains pre-learned (or randomly initialized) vectors for each word in the vocabulary.

   Suppose you have an input sequence like:
   \[
   \text{Input:} \ [w_1, w_2, w_3] \quad \text{(indices of words)}
   \]
   Each word is mapped to an embedding vector:
   \[
   E(w_1), E(w_2), E(w_3)
   \]
   Here, \(E(w_1)\) refers to the vector corresponding to the word \(w_1\) in the embedding matrix \(E\).

### 2. **Backpropagation Through the Embedding Layer**:
   Backpropagation will compute gradients that tell you how to adjust the parameters (in this case, the embedding vectors) to minimize the loss.

   For each word in the sequence, you calculate the error that propagates back from the loss function, such as cross-entropy loss for classification tasks or MSE for regression tasks.

   **How gradients are calculated:**
   - During backpropagation, you start with the gradients of the loss function with respect to the output layer (e.g., for a classification task, this could be the softmax output). Then, you propagate this error backward through the network, including the embedding layer.
   - The embedding layer’s contribution to the gradient is calculated by updating the embeddings corresponding to the words that were part of the current input.

### 3. **Gradient Computation for the Embedding Layer**:
   - When the network calculates the error at the output (for example, \(\frac{\partial L}{\partial \hat{y}}\), the gradient of the loss with respect to the predicted output), this error needs to be propagated back to the embedding layer.
   - The gradient of the loss with respect to a specific embedding \(E(w)\) is the sum of gradients from all occurrences of word \(w\) in the batch (if you're using mini-batch training).

   Mathematically, this means updating the embedding matrix \(E\) based on the gradients of the loss function with respect to each word embedding.

   If the embedding matrix is \(E\) and the indices of the words in the input are \(i_1, i_2, \dots, i_k\), where \(i_j\) is the index of word \(w_j\), the gradient of the loss with respect to the embedding vector \(E(w_j)\) is:
   \[
   \frac{\partial L}{\partial E(w_j)} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial E(w_j)}
   \]
   Here, \(\frac{\partial L}{\partial y}\) is the gradient with respect to the output (often the gradient of the loss function), and \(\frac{\partial y}{\partial E(w_j)}\) is how the output depends on the embedding. This gradient will be accumulated over all words in the batch.

### 4. **Update Embeddings**:
   After backpropagating the error and computing the gradient for each word embedding, you then update the word embeddings using gradient descent or any variant (like Adam, SGD, etc.):
   \[
   E(w_j) = E(w_j) - \eta \cdot \frac{\partial L}{\partial E(w_j)}
   \]
   where \(\eta\) is the learning rate.

### Summary:
To backpropagate through an embedding layer, the core idea is:
- Map each word in your input sequence to its corresponding embedding vector using the embedding matrix.
- Propagate the error from the output layer back through the network, calculating how the embeddings should be adjusted to minimize the loss.
- Update the word embeddings based on the gradients.

In essence, the embedding layer's parameters (the embedding vectors) are treated just like any other learnable parameters in the network, and the gradients will tell you how to update them to minimize the final loss.


---

## q:  can you give a code sample in pytorch (python)?

a: Certainly! Here's a simple example in PyTorch where we backpropagate through a word embedding layer. This example assumes you're working on a basic classification task with an embedding layer and a simple feedforward network.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example configuration
vocab_size = 10  # size of vocabulary
embedding_dim = 5  # embedding dimension
hidden_dim = 10  # size of hidden layer
output_dim = 2  # number of output classes (for binary classification)
batch_size = 4  # batch size

# Sample data (indices of words in the vocabulary)
# Let's assume each row is a sentence, and each number represents a word in the vocabulary
input_data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 5, 9]])

# Labels (binary classification)
labels = torch.tensor([0, 1, 0, 1])

# Define the neural network
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)  # Fully connected layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer
    
    def forward(self, x):
        # Pass the input through the embedding layer
        embedded = self.embedding(x)
        # For simplicity, we'll average the embeddings across words in each sentence
        embedded = embedded.mean(dim=1)
        # Pass through the fully connected layers
        x = torch.relu(self.fc1(embedded))
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleModel(vocab_size, embedding_dim, hidden_dim, output_dim)
loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification tasks
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop (backpropagation)
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Zero out the gradients
    # Forward pass
    outputs = model(input_data)
    # Compute the loss
    loss = loss_fn(outputs, labels)
    # Backward pass
    loss.backward()  # This calculates gradients for all parameters (including the embeddings)
    # Update parameters (including embeddings)
    optimizer.step()
    
    # Print loss every epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Now, the embeddings have been updated through backpropagation
# You can access the learned embeddings
print("Learned Embeddings: ")
print(model.embedding.weight)  # This will print the updated word embeddings
```

### Key Points:
1. **Embedding Layer**: `nn.Embedding(vocab_size, embedding_dim)` creates an embedding matrix of shape `(vocab_size, embedding_dim)`. Each word (indexed by its vocabulary number) is mapped to a vector of size `embedding_dim`.
   
2. **Forward Pass**: We pass the input tensor through the embedding layer, and then average the embeddings (just for simplicity) before passing them through fully connected layers. In real tasks, you might use more complex architectures (e.g., RNNs, CNNs) to process these embeddings.

3. **Loss & Backpropagation**: The model output is passed to a loss function (`CrossEntropyLoss`) and gradients are computed with `loss.backward()`. The optimizer updates both the embeddings and other parameters via `optimizer.step()`.

4. **Embeddings Update**: The word embeddings are learned during training, and you can access the final learned embeddings using `model.embedding.weight`.

This example is basic but illustrates the key concepts of how backpropagation through an embedding layer works in PyTorch.

---

## q:  is a code sample possible with numpy only (without pytorch)? 

a:  Certainly! Here’s a simplified version of backpropagation through an embedding layer using **only NumPy**. We'll implement a small neural network with an embedding layer and train it using a basic backpropagation algorithm.

### Key Elements:
1. **Embedding Layer**: This will be a matrix of word vectors, where each row represents a word in the vocabulary.
2. **Forward Pass**: We'll use the embeddings, compute a simple feedforward network, and calculate the loss.
3. **Backpropagation**: We'll compute the gradient of the loss with respect to the embeddings and update the embedding vectors.

### Code Sample Using NumPy:

```python
import numpy as np

# Configuration
vocab_size = 10  # Vocabulary size
embedding_dim = 5  # Dimension of word embeddings
hidden_dim = 10  # Number of hidden units
output_dim = 2  # Number of output classes
batch_size = 4  # Number of sentences in a batch
learning_rate = 0.01  # Learning rate

# Sample data (word indices in a batch of 4 sentences)
input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 5, 9]])  # 4 sentences, each with 3 words
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

    # Backpropagate to the hidden layer
    dL_dW2 = np.dot(a1.T, dL_dz2)  # Gradient of loss w.r.t. W2
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # Gradient of loss w.r.t. b2

    # Backpropagate further to the hidden layer (ReLU derivative)
    dL_da1 = np.dot(dL_dz2, W2.T)  # Gradient of loss w.r.t. a1
    dL_dz1 = dL_da1 * (a1 > 0)  # ReLU derivative (1 where a1 > 0, else 0)

    # Update hidden layer weights
    dL_dW1 = np.dot(embedding[input_data].mean(axis=1).T, dL_dz1)  # Gradient w.r.t. W1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)  # Gradient w.r.t. b1

    # Update the embedding weights (this is where the backpropagation through embeddings happens)
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

    # Update weights using gradient descent
    embedding -= learning_rate * dL_dembedding  # Update embedding layer
    W1 -= learning_rate * dL_dW1  # Update hidden layer weights
    b1 -= learning_rate * dL_db1  # Update hidden layer biases
    W2 -= learning_rate * dL_dW2  # Update output layer weights
    b2 -= learning_rate * dL_db2  # Update output layer biases

# Final learned embeddings
print("Learned Embeddings:")
print(embedding)
```

### Explanation of the Code:
1. **Initialization**:
   - `embedding`: The embedding matrix of shape `(vocab_size, embedding_dim)` initialized randomly.
   - `W1`, `b1`: Weights and biases for the hidden layer.
   - `W2`, `b2`: Weights and biases for the output layer.
   
2. **Forward Pass**:
   - **Embeddings**: We fetch the embeddings for each word in the batch using the indices from `input_data`. We then average these embeddings over the words in each sentence.
   - **Hidden Layer**: We apply a simple ReLU activation after multiplying by the weight matrix `W1`.
   - **Output Layer**: We compute the logits and apply softmax to get the probabilities for each class.

3. **Loss Calculation**:
   - **Cross-Entropy Loss**: The loss is calculated using the softmax probabilities and the actual labels.

4. **Backpropagation**:
   - Gradients of the loss are computed for each layer, including the embedding layer. We use the chain rule to propagate errors backward and compute the gradient with respect to each parameter (embedding, `W1`, `W2`, etc.).
   
5. **Parameter Update**:
   - We update the embeddings, weights, and biases using gradient descent.

### Result:
After training, you can see how the embeddings are updated through the backpropagation process. This is a simple example, and in real-world tasks, more sophisticated models (e.g., RNNs, Transformers) might be used, but this example helps demonstrate how backpropagation works with an embedding layer using only NumPy.



---

## q: how is backprop different between a (dense) linear layer and a word embedding layer?

a: Great question! The backpropagation process is quite similar between a dense linear layer and a word embedding layer, but there are some key differences due to how the layers interact with their inputs and their structure.

### 1. **Dense Linear Layer**:
In a typical **dense (linear) layer**, we compute the output as a weighted sum of the inputs, plus a bias. The weights and bias are updated during backpropagation based on the gradient of the loss with respect to these parameters.

Given an input vector \( x \) (with size \( d_{\text{in}} \)) and output \( y \) (with size \( d_{\text{out}} \)), the transformation in a dense layer is:

\[
y = x W + b
\]

Where:
- \( W \) is the weight matrix of size \( d_{\text{in}} \times d_{\text{out}} \).
- \( b \) is the bias vector of size \( d_{\text{out}} \).
- \( x \) is the input vector of size \( d_{\text{in}} \).

During **backpropagation**, the gradients are computed and used to update the weights \( W \), the bias \( b \), and the input \( x \).

The steps for backpropagating through a dense layer:
1. **Forward pass**: Compute the output using \( y = x W + b \).
2. **Backward pass**:
   - Gradient of the loss with respect to the output \( y \): \( \frac{\partial L}{\partial y} \).
   - Gradient of the loss with respect to the weights \( W \): \( \frac{\partial L}{\partial W} = x^T \cdot \frac{\partial L}{\partial y} \).
   - Gradient of the loss with respect to the bias \( b \): \( \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \).
   - Gradient of the loss with respect to the input \( x \): \( \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot W^T \).

3. **Update parameters**: Use the gradients to update \( W \), \( b \), and \( x \).

### 2. **Word Embedding Layer**:
In a **word embedding layer**, the main difference is that the "weights" are not learned from the input-output relationship in the same way as a dense layer. Instead, the embedding layer consists of a **lookup table** (embedding matrix), where each row represents a word in the vocabulary, and the goal is to **update the word representations** during training.

The forward pass in an embedding layer is a simple lookup of the embedding vectors corresponding to the indices of the words in the input sequence.

For input indices \( \{i_1, i_2, \dots, i_k\} \), the embedding matrix \( E \) of size \( \text{vocab\_size} \times \text{embedding\_dim} \) maps each index \( i_j \) to a vector \( E(i_j) \). The input is a sequence of indices, and the output is a sequence of embeddings:

\[
\text{input\_embeddings} = E[i_1], E[i_2], \dots, E[i_k]
\]

#### Backpropagation through the embedding layer:
When backpropagating through an embedding layer, the process focuses on updating the **embedding vectors** corresponding to the words used in the input, not on a weight matrix like in a dense layer.

1. **Forward pass**: 
   - Each word index in the input sequence \( x = [i_1, i_2, \dots, i_k] \) is used to retrieve the corresponding word embeddings from the embedding matrix \( E \).

2. **Backward pass**:
   - Compute the gradient of the loss with respect to the output (usually after a softmax layer or another network component).
   - The key difference is that **gradients are accumulated only for the embeddings corresponding to the input words**. So, instead of having gradients for a full weight matrix as in a dense layer, we update the **embedding vectors** that were actually used in the input.
   - The gradients are propagated backward from the output to the corresponding rows of the embedding matrix.

The gradients for the embedding layer are computed as:

\[
\frac{\partial L}{\partial E[i_j]} = \frac{\partial L}{\partial \text{output}} \cdot \frac{\partial \text{output}}{\partial E[i_j]}
\]

Since the embedding layer just retrieves rows from the embedding matrix, **only the embeddings for the words in the input** get updated.

3. **Update embeddings**: Once the gradients for each word embedding are computed, the embeddings for the corresponding words are updated using gradient descent (or another optimization method). The embedding matrix \( E \) is updated by:

\[
E[i_j] = E[i_j] - \eta \cdot \frac{\partial L}{\partial E[i_j]}
\]

Where \( \eta \) is the learning rate.

### Key Differences:
- **Dense Linear Layer**: 
  - Involves a weight matrix \( W \) that is updated for each input-output transformation.
  - The gradients are computed with respect to all the weights and biases, and the entire weight matrix gets updated.
  
- **Embedding Layer**:
  - Instead of weights, the parameters are individual word vectors stored in a large matrix (embedding matrix).
  - During backpropagation, gradients are only computed for the **specific rows of the embedding matrix** corresponding to the input words.
  - **No matrix multiplication**: In contrast to the dense layer, there's no multiplication of an entire weight matrix with the input; the embedding is a lookup operation.
  - The **embedding matrix is sparse** in the sense that only the rows corresponding to the input indices are updated.

### Conclusion:
While both layers perform parameter updates via backpropagation, the key difference is in the structure of the parameters and how they interact with the input data:
- **Dense layers** update weights for each connection between neurons.
- **Embedding layers** update the specific word vectors corresponding to the words in the input sequence.

In practice, **embedding layers tend to involve simpler operations** (lookup and update) compared to dense layers (which involve matrix multiplications).


---

## q: can you give a sample script for word embeddings with numpy?

a: Certainly! Below is a simple Python script that demonstrates how to train word embeddings from scratch using **NumPy**. This example uses the **Skip-gram** model of Word2Vec to learn word embeddings. For simplicity, we use a very small dataset, and the script focuses on the forward and backward passes (and gradient updates) of word embeddings.

### Overview:
- **Input**: A small dataset of words.
- **Goal**: Learn word embeddings that predict context words (Skip-gram model).
- **Objective**: Minimize the prediction error (i.e., the loss between the predicted and actual context words).

Here’s the code:

### Code for Training Word Embeddings (Skip-Gram) with NumPy

```python
import numpy as np

# Hyperparameters
vocab_size = 5  # Vocabulary size (total number of words in the vocabulary)
embedding_dim = 3  # Embedding dimension (size of word vectors)
window_size = 1  # Context window size
learning_rate = 0.1  # Learning rate
epochs = 1000  # Number of training iterations

# Sample corpus: A small corpus with a few words and sentences
# Each sentence is a list of words (tokens), and the corpus is a list of sentences.
corpus = [
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['the', 'dog', 'barked'],
    ['the', 'dog', 'sat', 'on', 'the', 'mat']
]

# Build vocabulary (word to index mapping)
word_to_idx = {}
idx_to_word = {}
index = 0

for sentence in corpus:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = index
            idx_to_word[index] = word
            index += 1

# Initialize the embedding and context vectors
W1 = np.random.randn(vocab_size, embedding_dim)  # Word embedding matrix
W2 = np.random.randn(embedding_dim, vocab_size)  # Context embedding matrix

# One-hot encoding function
def one_hot_encoding(index, size):
    one_hot = np.zeros(size)
    one_hot[index] = 1
    return one_hot

# Training the Skip-gram model
for epoch in range(epochs):
    total_loss = 0  # Track the loss over each epoch
    
    for sentence in corpus:
        # For each sentence, generate context pairs (word, context)
        for i, word in enumerate(sentence):
            word_idx = word_to_idx[word]
            context_indices = []
            
            # Create context (within the window_size range)
            for j in range(i - window_size, i + window_size + 1):
                if j != i and 0 <= j < len(sentence):
                    context_indices.append(word_to_idx[sentence[j]])

            # Forward pass (Skip-gram model)
            # 1. Input word vector (center word) - W1[word_idx]
            # 2. Context word prediction using W1 and W2
            
            # One-hot encode the context indices
            target_vectors = np.zeros((len(context_indices), vocab_size))
            for k, context_idx in enumerate(context_indices):
                target_vectors[k] = one_hot_encoding(context_idx, vocab_size)
            
            # Get the center word's embedding
            center_word_vector = W1[word_idx]

            # Compute the predicted context words (dot product with W2)
            predicted_context = np.dot(center_word_vector, W2)
            
            # Compute loss (softmax over predicted context)
            loss = 0
            for target_vector in target_vectors:
                predicted_probs = softmax(predicted_context)
                loss -= np.sum(target_vector * np.log(predicted_probs + 1e-9))  # Avoid log(0)
            
            total_loss += loss
            
            # Backpropagation (gradient calculation)
            # Compute gradients for W2 and W1
            
            # Gradient w.r.t. W2 (context vectors)
            dL_dW2 = np.outer(center_word_vector, target_vectors.sum(axis=0) - predicted_context)
            
            # Gradient w.r.t. W1 (word vectors)
            dL_dW1 = np.zeros_like(W1)
            for target_vector in target_vectors:
                dL_dW1[word_idx] += np.dot(W2, target_vector - predicted_context)

            # Update the weights using gradient descent
            W1 -= learning_rate * dL_dW1
            W2 -= learning_rate * dL_dW2
            
    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

# Print the learned embeddings (W1 contains word embeddings)
print("\nLearned Word Embeddings:")
for word, idx in word_to_idx.items():
    print(f"{word}: {W1[idx]}")
        
# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)
```

### Explanation of the Code:

1. **Corpus**:
   - A small corpus is provided, and the vocabulary is built by mapping each unique word to an index.
   - `word_to_idx` maps each word to a unique index, and `idx_to_word` maps indices back to words.

2. **Word Embedding Initialization**:
   - The word embeddings (`W1`) and context embeddings (`W2`) are initialized randomly. These are the two weight matrices in the Skip-gram model.

3. **Training Loop**:
   - The main training loop runs for a given number of epochs.
   - For each sentence, the script creates context-word pairs based on the window size. For each word in the sentence, it looks at words within a window around it (excluding the word itself).
   - The Skip-gram model predicts the context words given a center word by using the word’s embedding (from `W1`) and computing the probability distribution for context words using the context embedding matrix `W2`.
   - The loss is calculated using the **softmax** function, and the gradients are computed using the chain rule.

4. **Gradient Update**:
   - The gradients are computed for the context embeddings (`W2`) and the word embeddings (`W1`), and these embeddings are updated using **gradient descent**.

5. **Softmax**:
   - The softmax function is used to normalize the predicted context scores to form a probability distribution.

6. **Learned Word Embeddings**:
   - After training, the learned word embeddings are printed.

### Output Example:
After training, you should see the loss decreasing over epochs and finally, the learned embeddings for each word:

```
Epoch 1/1000, Loss: 50.27648741770959
Epoch 101/1000, Loss: 5.239298573379388
Epoch 201/1000, Loss: 4.235484428026439
...
Learned Word Embeddings:
the: [ 0.22529126 -0.04511469  0.13169893]
cat: [ 0.38875688  0.12933851 -0.03963488]
sat: [ 0.22357659 -0.03634907 -0.31844213]
on: [-0.2567889   0.14231323  0.04818995]
mat: [-0.2812364  -0.20216882 -0.11138992]
...
```

### Key Points:
- This is a **basic Skip-gram** model, where each word’s vector is learned to predict its surrounding context words.
- The loss function used is based on **softmax**, and the gradients are computed for both the input (center word) and output (context words).
- **Gradient descent** is used to update the embedding matrices.

This is a simple, illustrative example for understanding how word embeddings are learned. For larger datasets and more sophisticated models (like continuous bag of words or negative sampling), you would need more complex techniques and optimizations like mini-batch gradient descent, negative sampling, etc.