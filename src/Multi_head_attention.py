import torch.nn as nn
import torch
import math
from Scaled_dot_attention import scaled_dot_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        B, S, D = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        B, H, S, D = x.shape
        x = x.transpose(1, 2).contigious()
        return x.view(B, S, H*D)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.Wq(q))
        k = self.split_heads(self.Wk(k))
        v = self.split_heads(self.Wv(v))
        output, attention = scaled_dot_attention(q, k, v, mask)
        output = self.combine_heads(output)
        output = self.out(output)
        return output, attention
