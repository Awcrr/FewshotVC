import torch
import sys
import torchvision
import torch.nn as nn

# VGGPool3 Model
class VGGPool3(nn.Module):
    def __init__(self, layers):
        super(VGGPool3, self).__init__()
        self.pool3 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool3(x)
        return x
