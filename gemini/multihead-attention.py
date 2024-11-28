import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k=None, d_v=None, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Determine key, query, and value dimensions
        self.d_k = d_k or d_model // num_heads
        self.d_v = d_v or d_model // num_heads

        # Linear projections for key, query, and value
        self.W_q = nn.Linear(d_model, self.d_k * num_heads)
        self.W_k = nn.Linear(d_model, self.d_k * num_heads)
        self.W_v = nn.Linear(d_model, self.d_v * num_heads)

        # Linear projection for output
        self.W_o = nn.Linear(self.d_v * num_heads, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)


    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # Scaled dot-product attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # Apply masking
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, value)
        return context

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Multi-head attention
        context = self.scaled_dot_product_attention(query, key, value, mask)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v * self.num_heads)
        output = self.W_o(context)

        #Layer Normalization
        output = self.layer_norm(output + query.transpose(1,2).contiguous().view(batch_size, -1, self.d_model))


        return output


# Example usage:
d_model = 512
num_heads = 8
batch_size = 32
seq_len = 100

query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)

attention = MultiHeadAttention(d_model, num_heads)
print(attention)

output = attention(query, key, value)
print(output.shape)  # Output shape should be (batch_size, seq_len, d_model)


#Example with masking
mask = torch.triu(torch.ones(seq_len,seq_len),diagonal=1).bool() #Upper triangular mask for causal attention
output_masked = attention(query,key,value,mask=mask)
print(output_masked.shape)

print("bye")


"""
MultiHeadAttention(
  (W_q): Linear(in_features=512, out_features=512, bias=True)
  (W_k): Linear(in_features=512, out_features=512, bias=True)
  (W_v): Linear(in_features=512, out_features=512, bias=True)
  (W_o): Linear(in_features=512, out_features=512, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
torch.Size([32, 100, 512])
torch.Size([32, 100, 512])
"""