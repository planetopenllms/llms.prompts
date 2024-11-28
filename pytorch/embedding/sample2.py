import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # Get the embeddings for input indices
        output = self.fc(embedded.mean(dim=1))  # Simple example using the average embedding
        return output

# Example use case
model = SimpleNN(vocab_size=100, embedding_dim=50)
print( model )

input_indices = torch.tensor([[1, 2, 3],
                              [4, 5, 6]])  # Batch of sentences
output = model(input_indices)

print(output)


print( "bye" )

"""
SimpleNN(
  (embedding): Embedding(100, 50)
  (fc): Linear(in_features=50, out_features=1, bias=True)
)

tensor([[-0.2049],
        [-0.0288]], grad_fn=<AddmmBackward0>)
"""

