"""
changelogs:
    add SENet in ResNet block
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

class SENetBlock(nn.Module):
    def __init__(self, Ci):
        super().__init__()
        Cm = Ci//4
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(Ci, Cm),
            nn.ReLU(),
            nn.Linear(Cm, Ci),
            nn.Sigmoid(),
            nn.Unflatten(1, (Ci, 1)) # reshape Ci to Ci x 1
        )
        
    def forward(self, x):
        y = self.layers(x)
        y = torch.mul(x, y)
        return y  

class ResNetBlock(nn.Module):
    def __init__(self, Ci):
        super().__init__()
        self.layers = nn.Sequential(
            conv1d_relu(Ci, Ci, 3, 1),
            conv1d_relu(Ci, Ci, 3, 1),
            SENetBlock(Ci)
        )
        
    def forward(self, x):
        y = self.layers(x)
        y = torch.add(x, y)
        return y

class CSPNetBlock(nn.Module):
    def __init__(self, Ci, Co):
        super().__init__()
        Cn = Ci//2
        self.layers1 = conv1d_relu(Ci, Cn, 1, 1)
        self.layers2 = nn.Sequential(
            conv1d_relu(Ci, Cn, 1, 1),
            ResNetBlock(Cn)
        )
        self.layers3 = conv1d_relu(Ci, Co, 2, 2)
        
    def forward(self, x):
        y1 = self.layers1(x)
        y2 = self.layers2(x)
        y = torch.cat((y1, y2), 1)
        y = self.layers3(y)
        return y

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            conv1d_relu(1, 16, 11, 6, 'valid'),
            conv1d_relu(16, 16, 3, 2, 'valid'),
            conv1d_relu(16, 16, 3, 2, 'valid')
        )
        self.body = nn.Sequential(
            CSPNetBlock(16, 32),
            CSPNetBlock(32, 64),
            CSPNetBlock(64, 64)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        y = self.stem(x)
        y = self.body(y)
        y = self.head(y)
        return y
