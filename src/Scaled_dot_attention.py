import torch
import torch.nn as nn
import math

def scaled_dot_attention(q, k, v, mask = None):
    """
    q = (B, H, Lq, dq)
    k = (B, H, Lk, dk)
    v = (B, H, Lv, dv)
    """

    dk = q.size()
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(dk)

    if mask is not None:
        scores = torch.masked_fill(mask == 0, float("-infs"))
    
    attention = torch.softmax(scores, dim =-1)
    output = torch.matmul(attention, v)
    return attention, output