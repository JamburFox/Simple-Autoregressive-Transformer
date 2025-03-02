import torch
import torch.nn as nn
import os

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def generate_square_subsequent_mask(sz) -> torch.Tensor:
    mask = torch.tril(torch.ones(sz, sz)).type(torch.bool)
    return mask