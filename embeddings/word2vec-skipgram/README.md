
# Q: can you show a word2vec skipgram model for word embeddings in python with numpy?

a: Certainly! Below is a simple implementation of a **Word2Vec Skip-Gram model** using `numpy` in Python. This example demonstrates the fundamental approach to training word embeddings using a Skip-Gram model with negative sampling.

### Word2Vec Skip-Gram with Negative Sampling
In a Skip-Gram model, the goal is to predict the context words (surrounding words) given a target word. For example, for the sentence "The cat sat on the mat", the target word could be "sat", and the context words could be "The", "cat", "on", "the", "mat".

In negative sampling, the model learns to distinguish between actual context words (positive samples) and random words (negative samples). This reduces the computation required during training.

### Step-by-Step Implementation

#### Step 1: Preprocess the Text
We will first preprocess the text to generate a vocabulary, and then create training data for the Skip-Gram model.

```python
import numpy as np
from collections import Counter
import random

# Example sentence
text = "The cat sat on the mat"
words = text.lower().split()  # Tokenize the text and convert to lowercase

# Create a vocabulary (word-to-index mapping)
vocab = Counter(words)
vocab_size = len(vocab)

# Create word-to-index and index-to-word mappings
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Convert words to indices
word_indices = [word2idx[word] for word in words]

print("Vocabulary:", vocab)
print("Word to index mapping:", word2idx)
print("Index to word mapping:", idx2word)
```

#### Step 2: Generate Training Data (Target and Context Pairs)

For each word in the text, we will create training examples consisting of a target word and its context words. We'll use a window size to define the range of context words around each target word.

```python
window_size = 2  # For example, look at 2 words before and after the target word

# Generate target-context pairs
target_context_pairs = []
for idx, word_idx in enumerate(word_indices):
    # Look at words before and after the current word within the window size
    for i in range(1, window_size + 1):
        if idx - i >= 0:  # Look left
            target_context_pairs.append((word_idx, word_indices[idx - i]))
        if idx + i < len(word_indices):  # Look right
            target_context_pairs.append((word_idx, word_indices[idx + i]))

print("Target-Context Pairs:", target_context_pairs)
```

#### Step 3: Initialize the Model (Embeddings)

Now, we will initialize the embeddings for the words. We'll have two matrices: one for the **input embeddings** (for the target word) and one for the **output embeddings** (for the context word).

```python
embedding_dim = 3  # Dimension of the embeddings

# Randomly initialize the word vectors for the input and output layers
W_in = np.random.randn(vocab_size, embedding_dim)  # Input word embeddings
W_out = np.random.randn(vocab_size, embedding_dim)  # Output word embeddings

print("Initial input embeddings:\n", W_in)
print("Initial output embeddings:\n", W_out)
```

#### Step 4: Define the Training Process (Skip-Gram with Negative Sampling)

We will use the **negative sampling loss function** to update the word vectors. The loss function is defined as:

\[
L = -\log(\sigma(v_c \cdot v_t)) - \sum_{i=1}^k \log(\sigma(-v_{neg_i} \cdot v_t))
\]

Where:
- \( v_c \) is the context vector (positive sample)
- \( v_t \) is the target vector
- \( v_{neg_i} \) are the negative samples
- \( k \) is the number of negative samples

Here’s how to train the model:

```python
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training parameters
learning_rate = 0.1
epochs = 100
negative_samples = 2  # Number of negative samples

# Training loop
for epoch in range(epochs):
    total_loss = 0

    for target, context in target_context_pairs:
        # Positive sample
        target_vector = W_in[target]
        context_vector = W_out[context]

        # Compute the positive score and loss
        score = np.dot(target_vector, context_vector)
        loss_positive = -np.log(sigmoid(score))

        # Negative sampling
        negative_loss = 0
        for _ in range(negative_samples):
            # Sample a random negative word (different from context word)
            negative_idx = random.choice([i for i in range(vocab_size) if i != context])
            negative_vector = W_out[negative_idx]
            negative_score = np.dot(target_vector, negative_vector)
            negative_loss += -np.log(sigmoid(-negative_score))

        # Total loss
        loss = loss_positive + negative_loss
        total_loss += loss

        # Compute gradients and update the embeddings
        grad_target = sigmoid(score) - 1
        grad_context = sigmoid(score) * target_vector

        # Update input and output word embeddings
        W_in[target] -= learning_rate * grad_target * context_vector
        W_out[context] -= learning_rate * grad_target * target_vector

        for _ in range(negative_samples):
            # Sample a random negative word
            negative_idx = random.choice([i for i in range(vocab_size) if i != context])
            grad_negative = sigmoid(-np.dot(target_vector, W_out[negative_idx])) - 1
            W_out[negative_idx] -= learning_rate * grad_negative * target_vector

    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# Final word embeddings
print("Final input embeddings:\n", W_in)
```

#### Step 5: Use the Trained Embeddings

After training, you can use the embeddings for various tasks, such as finding similar words. Here’s an example of finding the most similar words to a given word:

```python
def most_similar(word_idx, top_n=3):
    # Compute the cosine similarity between the word and all other words
    word_vector = W_in[word_idx]
    similarities = []
    
    for idx in range(vocab_size):
        other_word_vector = W_in[idx]
        cosine_similarity = np.dot(word_vector, other_word_vector) / (np.linalg.norm(word_vector) * np.linalg.norm(other_word_vector))
        similarities.append((idx, cosine_similarity))
    
    # Sort by similarity and return the top_n most similar words
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [(idx2word[idx], similarity) for idx, similarity in similarities[:top_n]]

# Find the most similar words to "cat" (index 1)
similar_words = most_similar(word2idx['cat'])
print("Most similar words to 'cat':", similar_words)
```

### Conclusion

This implementation trains a **Word2Vec Skip-Gram model** using **negative sampling**. It's a very basic version of Word2Vec, but it illustrates how the skip-gram approach works at a fundamental level with `numpy`.

If you are working with real-world data or want to implement more advanced features, consider using libraries like **Gensim** or **TensorFlow**/**PyTorch** for more scalable and efficient training.

Let me know if you need any further details or improvements!


