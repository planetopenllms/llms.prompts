import numpy as np

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


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
        
