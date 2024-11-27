import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size

        # Linear layers to generate Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N = x.shape[0]  # Batch size
        seq_len = x.shape[1]  # Sequence length

        # Step 1: Linear projections to get Q, K, V
        Q = self.query(x)  # (N, seq_len, embed_size)
        K = self.key(x)    # (N, seq_len, embed_size)
        V = self.value(x)  # (N, seq_len, embed_size)

        # Step 2: Scaled dot-product attention
        # Compute attention scores: Q * K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # (N, seq_len, seq_len)
        attention_scores = attention_scores / (self.embed_size ** 0.5)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Step 3: Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (N, seq_len, seq_len)

        # Apply dropout for regularization
        attention_weights = self.dropout(attention_weights)

        # Step 4: Weighted sum of the values
        output = torch.matmul(attention_weights, V)  # (N, seq_len, embed_size)

        return output

# Example usage:
embed_size = 256
seq_len = 10
batch_size = 32

# Random input data (batch_size, seq_len, embed_size)
x = torch.rand(batch_size, seq_len, embed_size)

# Create the self-attention layer
self_attention_layer = SelfAttention(embed_size=embed_size)

# Perform a forward pass
output = self_attention_layer(x)

# Output shape will be (batch_size, seq_len, embed_size)
print(output)
print(output.shape)  # Expected output: (32, 10, 256)
