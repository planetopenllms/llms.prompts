##
# see PYTORCH.md
#
# q: generate a (vanilla) self-attention layer in pytorch for a neural network.
#    plus wrap the single-head self-attention into a multi-head class.


#
# => RuntimeError: batch1 must be a 3D tensor


import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute attention scores
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention_scores = attention_scores / (self.query.weight.shape[1] ** 0.5) # scaling

        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        output = torch.bmm(attention_weights, value)  # (batch_size, seq_len, embed_dim)

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # each head gets this many dimensions

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.attention = SelfAttention(self.head_dim, dropout)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)

        batch_size, seq_len, embed_dim = x.size()

        # Linear projections for each head
        query = self.linear(x) # (batch_size, seq_len, embed_dim)
        key = self.linear(x) # (batch_size, seq_len, embed_dim)
        value = self.linear(x) # (batch_size, seq_len, embed_dim)

        # Split into multiple heads
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_len, head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_len, head_dim)


        # Apply single-head attention to each head
        x = self.attention(value) # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim) # (batch_size, seq_len, embed_dim)

        # Final linear layer
        x = self.linear(x)
        x = self.dropout(x)

        return x


# Example usage
embed_dim = 512
num_heads = 8
batch_size = 32
seq_len = 100

x = torch.randn(batch_size, seq_len, embed_dim)
multi_head_attention = MultiHeadSelfAttention(embed_dim, num_heads)

print(multi_head_attention)

output = multi_head_attention(x)

print(output)
print(output.shape)  # Output: torch.Size([32, 100, 512])


print("bye")
