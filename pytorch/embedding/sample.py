import torch
import torch.nn as nn

# Define the parameters
num_embeddings = 10   # Vocabulary size (10 distinct tokens)
embedding_dim = 5     # Size of the embedding vector

# Create an embedding layer
embedding = nn.Embedding(num_embeddings, embedding_dim)
print(embedding)

# Example input: a batch of indices (e.g., 3 tokens)
input_indices = torch.tensor([1, 2, 4])

# Get the embedding for the input indices
output_embeddings = embedding(input_indices)

print(output_embeddings)
"""
tensor([[-0.3319,  1.3582,  0.6428,  0.8681, -1.8925],
        [ 1.0084, -0.0929,  1.5257,  0.2028, -1.0639],
        [ 0.6911,  1.2004, -0.3970,  0.7203, -0.0182]],
       grad_fn=<EmbeddingBackward0>)
"""


## print weights matrix
print( embedding.weight )

## try lookup by hand
for id in torch.tensor([1, 2, 4]):
  print( f"{id} -> {embedding.weight[id]}" )


print("bye")