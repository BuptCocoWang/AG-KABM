import torch

def masked_mse_loss(input, target, mask):
    loss = torch.square(input - target) * mask
    return loss.sum() / mask.sum()