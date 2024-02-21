"""
changelogs:
    head:
        change output to ReLU
"""

import torch
from torch import nn


def calc_padding(kernel_size, padding):
    if padding == 'same':
        return kernel_size//2
    elif padding == 'valid':
        return 0

def conv1d(Ci, Co, kernel_size, stride, padding):
    module = nn.Conv1d(Ci, Co,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=calc_padding(kernel_size, padding))
    nn.init.kaiming_normal_(module.weight) # He normal
    return module

def conv1d_relu(Ci, Co, kernel_size, stride, padding='same'):
    module = nn.Sequential(
        conv1d(Ci, Co, kernel_size, stride, padding),
        nn.ReLU()
    )
    return module

def CNN_Block(Ci, Co):
    module = nn.Sequential(
        conv1d_relu(Ci, Co, 3, 1),
        conv1d_relu(Co, Co, 3, 1),
        conv1d_relu(Co, Co, 2, 2)
    )
    return module

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            conv1d_relu(1, 16, 11, 6, 'valid'),
            conv1d_relu(16, 16, 3, 2, 'valid'),
            conv1d_relu(16, 16, 3, 2, 'valid')
        )
        self.body = nn.Sequential(
            CNN_Block(16, 32),
            CNN_Block(32, 64),
            CNN_Block(64, 64)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        y = self.stem(x)
        y = self.body(y)
        y = self.head(y)
        y = torch.clamp(y, max=1.)
        return y
