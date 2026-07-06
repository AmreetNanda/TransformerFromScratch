import torch.nn as nn
import torch
import math

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps= 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim =-1, keep_dim = True)
        variance = ((x-mean)**2).mean(dim = -1, keep_dim = True)
        x_hat = (x-mean) / torch.sqrt(variance **2 + self.eps)
        return self.gamma * x_hat + self.beta