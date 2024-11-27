import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super(SingleHeadSelfAttention, self).__init__()

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

        print( "x", x.shape, x.ndim )

        # Step 1: Linear projections to get Q, K, V
        Q = self.query(x)  # (N, seq_len, embed_size)
        K = self.key(x)    # (N, seq_len, embed_size)
        V = self.value(x)  # (N, seq_len, embed_size)

        # Step 2: Scaled dot-product attention: Q * K^T / sqrt(d_k)
        energy = torch.matmul(Q, K.transpose(-1, -2))  # (N, seq_len, seq_len)
        energy = energy / (self.embed_size ** 0.5)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        # Step 3: Softmax to get attention weights
        attention_weights = F.softmax(energy, dim=-1)  # (N, seq_len, seq_len)

        # Apply dropout for regularization
        attention_weights = self.dropout(attention_weights)

        # Step 4: Weighted sum of the values
        out = torch.matmul(attention_weights, V)  # (N, seq_len, embed_size)

        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by the number of heads"

        # Create a list of single-head self-attention modules for each head
        self.heads = nn.ModuleList([SingleHeadSelfAttention(self.head_dim, dropout) for _ in range(num_heads)])

        # Final linear layer to combine the outputs from all heads
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        N = x.shape[0]  # Batch size
        seq_len = x.shape[1]  # Sequence length

        # List to store outputs from all heads
        head_outputs = []

        print( "x", x.shape, x.ndim )
        #=> torch.Size([32, 10, 256]) 3

        for head in self.heads:
            # Apply the single-head self-attention for each head
            head_output = head(x, mask)  # (N, seq_len, head_dim)
            print( "head_output", head_output.shape, head_output.ndim )
            head_outputs.append(head_output)


        # Step 2: Concatenate all heads (output shape: (N, seq_len, embed_size))
        out = torch.cat(head_outputs, dim=-1)  # (N, seq_len, num_heads * head_dim)
        print( "out", out.shape, out.ndim )

        # Step 3: Pass through the final linear layer
        out = self.fc_out(out)  # (N, seq_len, embed_size)

        return out


# Example usage:
embed_size = 256
num_heads = 8
seq_len = 10
batch_size = 32

# Random input data (batch_size, seq_len, embed_size)
x = torch.rand(batch_size, seq_len, embed_size)

# Create the multi-head self-attention layer
multi_head_attention = MultiHeadSelfAttention(embed_size=embed_size, num_heads=num_heads)
print(multi_head_attention)

# Perform a forward pass
output = multi_head_attention(x)

# Output shape will be (batch_size, seq_len, embed_size)
print(output.shape)  # Expected output: (32, 10, 256)
