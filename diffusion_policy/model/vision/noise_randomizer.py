import torch
from torch import nn

class Noiser(nn.Module):
    def __init__(self, noise_level=0.3):
        super().__init__()
        self.noise_level = noise_level

    def forward(self, x):
        noise = torch.randn_like(x) * torch.rand(1).item() * self.noise_level
        return x + noise