# _0_NN.py

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
           # nn.Linear(28*28, 512),
            nn.Linear(32*32*3, 512),  # Adjust input size based on image size and channels
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


device = "cuda" if torch.cuda.is_available() else "cpu"