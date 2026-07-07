import torch
import numpy as np

def single_head_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.bool()

def multi_head_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.bool()


# mask = casual_mask(5)
# print(mask)

