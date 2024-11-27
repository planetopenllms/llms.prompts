import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by num_heads"

        # Linear layers for Q, K, and V
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

        # Output linear layer
        self.out_linear = nn.Linear(embed_size, embed_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """
        Split the input tensor into multiple heads.
        x: (batch_size, seq_len, embed_size)
        """
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Step 1: Apply linear layers to Q, K, and V
        query = self.query_linear(query)  # (batch_size, seq_len, embed_size)
        key = self.key_linear(key)        # (batch_size, seq_len, embed_size)
        value = self.value_linear(value)  # (batch_size, seq_len, embed_size)

        # Step 2: Split into multiple heads
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_heads(key, batch_size)      # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len, head_dim)

        # Step 3: Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5  # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)

        # Step 4: Attention output
        attention_output = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, head_dim)

        # Step 5: Concatenate heads and pass through output linear layer
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        attention_output = attention_output.view(batch_size, -1, self.num_heads * self.head_dim)  # (batch_size, seq_len, embed_size)

        output = self.out_linear(attention_output)  # (batch_size, seq_len, embed_size)

        return output




### Example Usage:

# Create a sample input tensor (batch_size=2, seq_len=5, embed_size=8)
query = torch.rand(2, 5, 8)
key   = torch.rand(2, 5, 8)
value = torch.rand(2, 5, 8)

# Create the multi-head attention layer
attention_layer = MultiHeadAttention(embed_size=8, num_heads=2)

# Forward pass
output = attention_layer(query, key, value)
print(output)
print(output.shape)  # Expected output: torch.Size([2, 5, 8])


print('bye')
