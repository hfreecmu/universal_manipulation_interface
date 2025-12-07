import torch
from torch import nn
from torchvision.transforms import ColorJitter, RandomPosterize, RandomAdjustSharpness, RandomAutocontrast, RandomEqualize, RandomErasing
from torchvision import transforms
import random
from PIL import Image
import matplotlib.pyplot as plt

class OtherRandomizer(nn.Module):
    def __init__(self, prob=0.1, do_jitter=False):
        super().__init__()

        if do_jitter:
            self.transforms = [
                ColorJitter(brightness=0.5, hue=0.5, contrast =0.5, saturation=0.5),
            ]
        else:
            self.transforms = None

        self.prob = prob

    def forward(self, x):
        if self.transforms is not None and random.random() < self.prob:
            transform = random.choice(self.transforms)
            x = transform(x)

        eraser = RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.7, 1.3), value='random')
        x = eraser(x)
        
        return x
